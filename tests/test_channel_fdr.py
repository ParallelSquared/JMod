import numpy as np
import pandas as pd
import polars as pl
import pytest

from src.channel_fdr import (FEATURE_FILE, FEATURE_FOLDER, build_channel_features, channel_specific_mask,
                             first_of_each, tag_sites, write_channel_features, _row_features)
from src.mass_tags import available_tags
from src.models.spec_lib.library_store import SpectrumLibraryStore
from src.utils.frag_encoding import encode_frag_names
from tests.models.spec_lib.test_decoy_differential import _entry


def _codes(*names):
    return np.asarray(encode_frag_names(list(names)), dtype=np.int32)


def _cos(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


class TestTagSites:
    """Read off the searched sequence's annotations: (sites, tags, residues)."""

    @pytest.mark.parametrize("seq, expected", [
        ("(PSMtag_5plex-0)PEPTIDEK(PSMtag_5plex-0)", ([0, 7], 2, 8)),   # N-terminus and C-terminal K
        ("(PSMtag_5plex-0)PEPTIDER", ([0], 1, 8)),                      # N-terminus only
        ("(PSMtag_5plex-0)K(PSMtag_5plex-0)PEPTIDE", ([0], 2, 8)),      # N-terminal K: one site, two tags
        ("(PSMtag_5plex-0)PEPC(UniMod:4)TIDER", ([0], 1, 9)),           # another mod is not a tag
        ("PEPTIDER", ([], 0, 8)),                                       # untagged
    ])
    def test_sites_tags_and_residues(self, seq, expected):
        assert tag_sites(seq, "PSMtag_5plex") == expected

    def test_the_sequence_not_the_rules(self):
        """nK would put tags on the N-terminus and the K; this library tagged only
        the K, so that is where the search's fragment masses moved."""
        assert tag_sites("PEPTIDEK(PSMtag_5plex-0)", "PSMtag_5plex") == ([7], 1, 8)


class TestChannelSpecificMask:
    """A fragment is channel-specific iff the residues it spans include a tag site."""

    def test_internal_site(self):
        # PEPKTIDE, K at index 3: b3 = PEP and y4 = TIDE miss it; b4 = PEPK and y5 = KTIDE span it
        codes = _codes("b3_1", "b4_1", "y4_1", "y5_1")
        assert channel_specific_mask(codes, 8, 3, 3).tolist() == [False, True, False, True]

    def test_n_terminal_tag_only(self):
        # PEPTIDER with nK: every b ion has the N-terminus; a y ion only when it is the whole peptide
        codes = _codes("b1_1", "b7_1", "y1_1", "y7_1", "y8_1")
        assert channel_specific_mask(codes, 8, 0, 0).tolist() == [True, True, False, False, True]

    def test_c_terminal_k_makes_every_fragment_channel_specific(self):
        # PEPTIDEK with nK: b ions carry the N-terminal tag, y ions the K's
        codes = _codes("b2_1", "b6_1", "y1_1", "y6_1")
        assert channel_specific_mask(codes, 8, 0, 7).all()

    def test_isotope_and_loss_do_not_change_the_answer(self):
        codes = _codes("y4_1", "y4_1_iso2", "y4-H2O_2", "y5_1", "y5_1_iso1")
        assert channel_specific_mask(codes, 8, 3, 3).tolist() == [False, False, False, True, True]


class TestRowFeatures:
    """PEPTIDER with nK (N-terminal site only): b2 and b3 are channel-specific,
    y3 is shared.  The library has all three plus y3's first isotope; the apex
    scan matched b2, y3 and the isotope."""

    lib_codes = _codes("b2_1", "b3_1", "y3_1", "y3_1_iso1")
    lib_int = np.array([0.5, 0.2, 1.0, 0.3])
    matched_codes = _codes("y3_1_iso1", "b2_1", "y3_1")     # not in library order
    matched_obs = np.array([6.0, 10.0, 20.0])

    @pytest.fixture
    def row(self):
        return _row_features(self.lib_codes, self.lib_int, self.matched_codes, self.matched_obs, 8, 0, 0)

    def test_counts_are_monoisotopic(self, row):
        assert (row["n_chan_frags"], row["n_shared_frags"]) == (1, 1)           # b2; y3 (not its isotope)
        assert (row["n_lib_chan_frags"], row["n_lib_shared_frags"]) == (2, 1)   # b2, b3; y3

    def test_fraction_matched(self, row):
        assert row["frac_chan_frags_matched"] == pytest.approx(1 / 2)
        assert row["frac_shared_frags_matched"] == pytest.approx(1 / 1)
        assert row["frac_frags_matched"] == pytest.approx(2 / 3)

    def test_intensity_sums_with_and_without_isotopes(self, row):
        assert row["chan_frag_int_sum"] == pytest.approx(10.0)
        assert row["shared_frag_int_sum"] == pytest.approx(26.0)        # y3 + its isotope
        assert row["chan_frag_int_mono_sum"] == pytest.approx(10.0)
        assert row["shared_frag_int_mono_sum"] == pytest.approx(20.0)

    def test_channel_specific_fractions(self, row):
        assert row["chan_frag_fraction"] == pytest.approx(1 / 2)               # b2 of b2, y3
        assert row["chan_intensity_fraction"] == pytest.approx(10 / 36)        # 10 of 10 + 20 + 6
        assert row["chan_mono_intensity_fraction"] == pytest.approx(10 / 30)   # 10 of 10 + 20

    def test_cosines_count_unmatched_library_fragments_as_zero(self, row):
        # monoisotopic library order b2, b3, y3; b3 was not matched
        assert row["cosine_chan"] == pytest.approx(_cos([10, 0], [0.5, 0.2]))
        assert row["cosine_all"] == pytest.approx(_cos([10, 0, 20], [0.5, 0.2, 1.0]))
        assert row["cosine_chan_minus_all"] == pytest.approx(row["cosine_chan"] - row["cosine_all"])

    def test_nothing_to_match_is_null_not_zero(self):
        # PEPTIDEK with nK: no fragment is shared, so the shared fraction and the
        # shared ratio's denominator are undefined
        codes = _codes("b2_1", "y2_1")
        row = _row_features(codes, np.array([1.0, 1.0]), codes[:1], np.array([5.0]), 8, 0, 7)
        assert row["n_lib_shared_frags"] == 0
        assert np.isnan(row["frac_shared_frags_matched"])
        assert row["chan_frag_fraction"] == 1.0       # finite where the ratio would be infinite

    def test_no_matched_fragments(self):
        row = _row_features(self.lib_codes, self.lib_int, _codes(), np.array([]), 8, 0, 0)
        assert (row["n_chan_frags"], row["n_shared_frags"]) == (0, 0)
        assert row["frac_frags_matched"] == 0
        assert np.isnan(row["chan_frag_fraction"])     # 0 / 0
        # The library predicted fragments and none were seen: a real 0, not unknown
        assert (row["cosine_chan"], row["cosine_all"]) == (0.0, 0.0)

    def test_cosine_is_undefined_only_when_the_library_has_no_fragments_of_the_class(self):
        # PEPKTIDE, K at 3: none of b2, b3, y2, y3 reaches it, so nothing is channel-specific
        codes = _codes("b2_1", "b3_1", "y2_1", "y3_1")
        row = _row_features(codes, np.ones(4), codes[:2], np.array([5.0, 5.0]), 8, 3, 3)
        assert row["n_lib_chan_frags"] == 0
        assert np.isnan(row["cosine_chan"])
        assert row["cosine_all"] == pytest.approx(_cos([5, 5, 0, 0], [1, 1, 1, 1]))


def _store():
    """Library entries for PEPTIDER in channels 0 and 20 (the decoy channel)."""
    frags = {"b2_1": (227.1, 0.5), "b3_1": (324.1, 0.2), "y3_1": (400.2, 1.0)}
    seqs = ["(PSMtag_5plex-0)PEPTIDER", "(PSMtag_5plex-20)PEPTIDER"]
    return SpectrumLibraryStore.from_dict({(s, 2.0): _entry(s, "PEPTIDER", 500.0, 2, frags) for s in seqs})


def _ids(rows):
    """*rows*: (seq, channel, is_decoy, is_decoy_channel, BestChannel_Qvalue)."""
    return pd.DataFrame({
        "file_name": "f", "seq": [r[0] for r in rows], "z": 2,
        "untag_seq": "PEPTIDER", "untag_prec": "PEPTIDER_2",
        "channel": [r[1] for r in rows], "is_decoy": [r[2] for r in rows],
        "is_decoy_channel": [r[3] for r in rows], "Qvalue": 0.001,
        "BestChannel_Qvalue": [r[4] for r in rows], "PredVal": 0.9,
        "__fdc_idx": list(range(len(rows))),
    })


def _list_cols(matched):
    """*matched*: per __fdc_idx, (fragment names, observed intensities), written in
    reverse so a join that loses the order would put them on the wrong rows."""
    rows = list(enumerate(matched))[::-1]
    return pl.DataFrame({"__fdc_idx": [i for i, _ in rows],
                         "frag_names": [encode_frag_names(names).tolist() for _, (names, _) in rows],
                         "obs_int": [list(obs) for _, (_, obs) in rows]},
                        schema={"__fdc_idx": pl.UInt32, "frag_names": pl.List(pl.Int32),
                                "obs_int": pl.List(pl.Float64)})


class TestBuildChannelFeatures:
    tag = available_tags["PSMtag_5plex"].with_decoy_channel()     # decoy channel 20

    def test_which_rows_are_scored(self):
        ids = _ids([("(PSMtag_5plex-0)PEPTIDER", 0, False, False, 0.001),
                    ("(PSMtag_5plex-20)PEPTIDER", 20, False, True, 0.001),       # the null: kept
                    ("Decoy_(PSMtag_5plex-0)PEPTIDER", 0, True, False, 0.001),   # library decoy: not
                    ("(PSMtag_5plex-0)PEPTIDER", 0, False, False, 0.5)])         # failed the FDR: not
        matched = [(["b2_1"], [1.0])] * 4
        features = build_channel_features(ids, _list_cols(matched), _store(), self.tag, 0.01)
        assert features.select("channel", "is_decoy_channel").rows() == [(0, False), (20, True)]

    def test_each_row_keeps_its_own_fragments(self):
        ids = _ids([("(PSMtag_5plex-0)PEPTIDER", 0, False, False, 0.001),
                    ("(PSMtag_5plex-20)PEPTIDER", 20, False, True, 0.001)])
        matched = [(["b2_1", "b3_1", "y3_1"], [1.0, 1.0, 1.0]),    # every fragment
                   (["y3_1"], [1.0])]                               # the shared one only
        features = build_channel_features(ids, _list_cols(matched), _store(), self.tag, 0.01)
        assert features["n_chan_frags"].to_list() == [2, 0]
        assert features["n_shared_frags"].to_list() == [1, 1]

    def test_a_precursor_missing_from_the_library_is_null_throughout(self):
        ids = _ids([("(PSMtag_5plex-0)PEPTIDER", 0, False, False, 0.001),
                    ("(PSMtag_5plex-0)NOTINLIB", 0, False, False, 0.001)])
        features = build_channel_features(ids, _list_cols([(["b2_1"], [1.0])] * 2),
                                          _store(), self.tag, 0.01)
        assert features["n_lib_chan_frags"].to_list() == [2, None]
        assert features["cosine_all"].is_null()[1] or np.isnan(features["cosine_all"][1])

    # (n_tags, pep_len, first_tag_from_n, last_tag_from_c): the two distances are
    # the number of tag-free b and y ions
    @pytest.mark.parametrize("seq, context", [
        ("(PSMtag_5plex-0)PEPTIDER", (1, 8, 0, 7)),                        # no tag on any y ion
        ("(PSMtag_5plex-0)PEPTIDEK(PSMtag_5plex-0)", (2, 8, 0, 0)),        # every y ion tagged
        ("(PSMtag_5plex-0)PEPK(PSMtag_5plex-0)TIDER", (2, 9, 0, 5)),       # y1..y5 tag-free
        ("(PSMtag_5plex-0)K(PSMtag_5plex-0)PEPTIDE", (2, 8, 0, 7)),        # two tags, one site
    ])
    def test_peptide_context(self, seq, context):
        ids = _ids([(seq, 0, False, False, 0.001)])
        features = build_channel_features(ids, _list_cols([(["b2_1"], [1.0])]), _store(), self.tag, 0.01)
        assert features.select("n_tags", "pep_len", "first_tag_from_n", "last_tag_from_c").row(0) == context

    def test_untagged_precursors_are_not_scored(self):
        """Under a K-only tag a peptide without K is searched once, untagged: no
        channels to tell apart."""
        ids = _ids([("PEPK(K_6C13-0)TIDE", 0, False, False, 0.001),
                    ("PEPTIDER", float("nan"), False, False, 0.001)])
        features = build_channel_features(ids, _list_cols([(["b2_1"], [1.0])] * 2), _store(),
                                          available_tags["K_6C13"].with_decoy_channel(), 0.01)
        assert features["seq"].to_list() == ["PEPK(K_6C13-0)TIDE"]


class TestTaggedOtherThanByTheRules:
    """A pre-tagged library whose tagging breaks the tag's rules: PSMtag_5plex is
    nK, but this entry carries a tag on its K only, not on its N-terminus.  The
    search computed the fragment masses from that tagging, so the split has to
    follow it: the b ions never reach the K, so they are shared, where nK would
    make every fragment channel-specific."""

    seq = "PEPTIDEK(PSMtag_5plex-0)"

    @pytest.fixture
    def features(self):
        frags = {"b2_1": (227.1, 0.5), "b3_1": (324.1, 0.2), "y2_1": (300.2, 1.0), "y3_1": (401.2, 0.8)}
        store = SpectrumLibraryStore.from_dict({(self.seq, 2.0): _entry(self.seq, "PEPTIDEK", 500.0, 2, frags)})
        ids = _ids([(self.seq, 0, False, False, 0.001)])
        return build_channel_features(ids, _list_cols([(["b2_1", "y2_1"], [4.0, 6.0])]), store,
                                      available_tags["PSMtag_5plex"].with_decoy_channel(), 0.01).row(0, named=True)

    def test_the_split_follows_the_sequence(self, features):
        # by the sequence: b2, b3 shared; y2, y3 channel-specific (by nK all four would be)
        assert (features["n_lib_chan_frags"], features["n_lib_shared_frags"]) == (2, 2)
        assert (features["n_chan_frags"], features["n_shared_frags"]) == (1, 1)     # y2; b2
        assert features["chan_frag_fraction"] == pytest.approx(0.5)
        assert features["chan_intensity_fraction"] == pytest.approx(6 / 10)

    def test_the_tag_context_follows_the_sequence(self, features):
        # by nK: two tags, the first at the N-terminus
        assert features["n_tags"] == 1
        assert features["first_tag_from_n"] == 7      # b1..b7 tag-free
        assert features["last_tag_from_c"] == 0


class TestRepeatedFragmentIdentities:
    """A library can list one identity twice at different intensities.  The
    first is kept, so one fragment is never counted as two."""

    def test_first_of_each(self):
        codes = _codes("b2_1", "y3_1", "b2_1")
        kept, values, repeated = first_of_each(codes, np.array([0.5, 1.0, 0.9]))
        assert (kept.tolist(), values.tolist(), repeated) == (codes[:2].tolist(), [0.5, 1.0], True)
        assert first_of_each(codes[:2], np.array([0.5, 1.0]))[2] is False

    def test_a_repeated_library_identity_is_counted_once_and_warned_about(self, app_log):
        store = _store()

        class Repeating:                      # the store, with b2 listed a second time
            key_to_idx = store.key_to_idx

            def get_ordered_frag_codes(self, idx):
                return np.append(store.get_ordered_frag_codes(idx), _codes("b2_1"))

            def get_frag_intensities(self, idx):
                return np.append(store.get_frag_intensities(idx), 0.9)

        ids = _ids([("(PSMtag_5plex-0)PEPTIDER", 0, False, False, 0.001)])
        features = build_channel_features(ids, _list_cols([(["b2_1"], [1.0])]), Repeating(),
                                          available_tags["PSMtag_5plex"].with_decoy_channel(), 0.01)
        assert features["n_lib_chan_frags"].to_list() == [2]          # b2 and b3, not b2 twice
        assert any("more than once" in r.getMessage() for r in app_log)


class TestWriteChannelFeatures:
    def test_written_to_the_channel_fdr_folder(self, tmp_path):
        tag = available_tags["PSMtag_5plex"].with_decoy_channel()
        ids = _ids([("(PSMtag_5plex-0)PEPTIDER", 0, False, False, 0.001)])
        write_channel_features(ids, _list_cols([(["b2_1"], [1.0])]), _store(), tag, str(tmp_path), 0.01)
        written = pl.read_parquet(tmp_path / FEATURE_FOLDER / FEATURE_FILE)
        assert written["n_chan_frags"].to_list() == [1]

    def test_nothing_without_a_decoy_channel(self, tmp_path):
        ids = _ids([("(PSMtag_5plex-0)PEPTIDER", 0, False, False, 0.001)])
        assert write_channel_features(ids, _list_cols([(["b2_1"], [1.0])]), _store(),
                                      available_tags["PSMtag_5plex"], str(tmp_path), 0.01) is None
        assert not (tmp_path / FEATURE_FOLDER).exists()
