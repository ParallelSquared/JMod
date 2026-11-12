import os
import re

import numpy as np
import pandas as pd
import polars as pl
import pytest

from src.channel_fdr import (FEATURE_FILE, FEATURE_FOLDER, FEATURE_IMPORTANCE_FILE, MIN_DECOYS,
                             build_channel_features, channel_qvalues, channel_specific_mask, fit_channel_fdr,
                             model_columns, model_matrix, per_channel_pep, per_channel_qvalues,
                             protein_channel_qvalues,
                             first_of_each, frag_error_features, longest_consecutive_scans, ms1_features, ms2_cycles, null_placeholders,
                             read_feature_importance, run_channel_fdr, tag_sites, write_feature_importance, _CARRIED_FEATURES,
                             _row_features)
from src.channel_fdr_plots import BEST_CHANNEL_PLOT
from src.fdr_analysis import estimate_pep, target_decoy_qvalues
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

    def test_no_intensities_only_fractions_of_them(self, row):
        assert not [k for k in row if k.endswith("_sum")]

    def test_median_intensity_fraction_is_per_fragment(self):
        # three channel-specific b ions at 10 each, one shared y at 30: the sums
        # split 30 / 30, the typical fragment 10 / 30; isotopes left out
        codes = _codes("b2_1", "b3_1", "b4_1", "y3_1", "y3_1_iso1")
        obs = np.array([10.0, 10.0, 10.0, 30.0, 500.0])
        row = _row_features(codes, np.ones(5), codes, obs, 8, 0, 0)
        assert row["chan_mono_intensity_fraction"] == pytest.approx(30 / 60)
        assert row["chan_median_intensity_fraction"] == pytest.approx(10 / 40)

    def test_median_intensity_fraction_needs_both_classes(self, row):
        assert row["chan_median_intensity_fraction"] == pytest.approx(10 / 30)     # b2; y3
        codes = _codes("b2_1", "y3_1")
        nothing_shared = _row_features(codes, np.ones(2), codes[:1], np.array([5.0]), 8, 0, 0)
        assert np.isnan(nothing_shared["chan_median_intensity_fraction"])

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


class TestFragErrorFeatures:
    """PEPTIDER with nK: b ions are channel-specific, y ions shared.  Errors
    measured from a run calibration of +2."""

    def test_median_distance_from_the_calibration_each_side(self):
        codes = _codes("b2_1", "b3_1", "b4_1", "y2_1", "y3_1")
        errors = np.array([-6.0, 10.0, 2.0, 3.0, 1.0])
        f = frag_error_features(codes, errors, 8, 0, 0, centre=2.0)
        assert f["chan_frag_error"] == pytest.approx(8.0)       # |−8|, |8|, |0|
        assert f["shared_frag_error"] == pytest.approx(1.0)     # |1|, |−1|
        assert f["chan_minus_shared_frag_error"] == pytest.approx(7.0)

    def test_monoisotopic_only(self):
        codes = _codes("b2_1", "b2_1_iso1", "y2_1")
        f = frag_error_features(codes, np.array([2.0, 50.0, 2.0]), 8, 0, 0, centre=2.0)
        assert f["chan_frag_error"] == 0.0

    def test_a_side_with_nothing_matched_is_null(self):
        f = frag_error_features(_codes("y2_1", "y3_1"), np.array([1.0, 3.0]), 8, 0, 0, centre=2.0)
        assert np.isnan(f["chan_frag_error"]) and np.isnan(f["chan_minus_shared_frag_error"])
        assert f["shared_frag_error"] == pytest.approx(1.0)

    def test_in_the_table_centred_on_the_whole_runs_errors(self):
        # the calibration is the median over every row of the list columns: 4.
        # A third row, not scored, still counts towards it
        ids = _ids([("(PSMtag_5plex-0)PEPTIDER", 0, False, False, 0.001),
                    ("(PSMtag_5plex-20)PEPTIDER", 20, False, True, 0.001)])
        list_cols = _list_cols([(["b2_1", "y3_1"], [1.0, 1.0], [4.0, 5.0]),
                                (["b2_1", "y3_1"], [1.0, 1.0], [12.0, 4.0]),
                                (["y2_1"], [1.0], [3.0])])
        features = _features(ids, list_cols, _store(), available_tags["PSMtag_5plex"].with_decoy_channel(), 0.01)
        assert features["chan_frag_error"].to_list() == [0.0, 8.0]
        assert features["shared_frag_error"].to_list() == [1.0, 0.0]
        assert features["chan_minus_shared_frag_error"].to_list() == [-1.0, 8.0]


def _store():
    """Library entries for PEPTIDER in channels 0 and 20 (the decoy channel)."""
    frags = {"b2_1": (227.1, 0.5), "b3_1": (324.1, 0.2), "y3_1": (400.2, 1.0)}
    seqs = ["(PSMtag_5plex-0)PEPTIDER", "(PSMtag_5plex-20)PEPTIDER"]
    return SpectrumLibraryStore.from_dict({(s, 2.0): _entry(s, "PEPTIDER", 500.0, 2, frags) for s in seqs})


def _last_aa(seq):
    """JMod's last_aa: the last letter of the stripped sequence."""
    return re.sub(r"\(.*?\)", "", seq)[-1]


def _ids(rows):
    """*rows*: (seq, channel, is_decoy, is_decoy_channel, BestChannel_Qvalue).
    Every carried feature is 1 (a real value for each of them) unless a test
    sets it; the plex set has two channels and the precursor three scans."""
    return pd.DataFrame({
        **{c: 1.0 for c in _CARRIED_FEATURES},
        "n_scans": 3, "n_corr_frags": 5, "channels_matched": 2, "plexfitMS1": 4.0,
        "last_aa": [_last_aa(r[0]) for r in rows], "file_name": "f", "seq": [r[0] for r in rows], "z": 2,
        "untag_seq": "PEPTIDER", "untag_prec": "PEPTIDER_2",
        "channel": [r[1] for r in rows], "is_decoy": [r[2] for r in rows],
        "is_decoy_channel": [r[3] for r in rows], "Qvalue": 0.001,
        "BestChannel_Qvalue": [r[4] for r in rows], "PredVal": 0.9, "rt": 61.5,
        # ms1_quant's outputs: one MS1 fit window of five scans, a peak in the middle
        "plexfittrace_all": "0;1;4;1;0", "plexfittrace_ps_all": "0.9;0.9;0.9;0.9;0.9",
        "plexfittrace_spec_all": "10;31;52;73;94", "ms1_apex_scan": 52, "Ms1_spec_id": 52,
        "gauss_r2_joint": 0.8,
        "__fdc_idx": list(range(len(rows))),
    })


def _list_cols(matched):
    """*matched*: per __fdc_idx, (fragment names, observed intensities[, mass
    errors, 0 if not given]), written in reverse so a join that loses the order
    would put them on the wrong rows."""
    rows = list(enumerate(matched))[::-1]
    return pl.DataFrame({"__fdc_idx": [i for i, _ in rows],
                         "frag_names": [encode_frag_names(m[0]).tolist() for _, m in rows],
                         "obs_int": [list(m[1]) for _, m in rows],
                         "frag_errors": [list(m[2]) if len(m) > 2 else [0.0] * len(m[0]) for _, m in rows]},
                        schema={"__fdc_idx": pl.UInt32, "frag_names": pl.List(pl.Int32),
                                "obs_int": pl.List(pl.Float64), "frag_errors": pl.List(pl.Float64)})


# The run's MS2 scans: two isolation windows, A and B, each visited once a
# cycle for five cycles (MS1 scans in between, which do not matter).  PEPTIDER
# in channel 0 was fitted in A in cycles 0, 1, 2 and 4, and in B in cycle 1 (its
# m/z is in both); in channel 20, in A in cycle 3 only
WINDOW_A, WINDOW_B = (500.0, 520.0), (518.0, 538.0)
MS2_SCANS = [(101 + 22 * k + w, window) for k in range(5) for w, window in enumerate((WINDOW_A, WINDOW_B))]


def _scan_cycles():
    return ms2_cycles([scan for scan, _ in MS2_SCANS], [window for _, window in MS2_SCANS])


def _dc():
    rows = [("(PSMtag_5plex-0)PEPTIDER", 101), ("(PSMtag_5plex-0)PEPTIDER", 123),
            ("(PSMtag_5plex-0)PEPTIDER", 124), ("(PSMtag_5plex-0)PEPTIDER", 145),
            ("(PSMtag_5plex-0)PEPTIDER", 189), ("(PSMtag_5plex-20)PEPTIDER", 167)]
    return pl.DataFrame({"seq": [s for s, _ in rows], "z": [2.0] * len(rows),
                         "coeff": [1.0] * len(rows), "spec_id": [m for _, m in rows]},
                        schema_overrides={"z": pl.Float32})


class _Spectra:
    ms2scans = [type("Spectrum", (), {"scan_num": scan, "ms1window": window})()
                for scan, window in MS2_SCANS]


def _features(ids, list_cols, library, mass_tag, fdr_threshold):
    return build_channel_features(ids, list_cols, _dc(), _scan_cycles(), library, mass_tag, fdr_threshold)


def _write(ids, list_cols, library, mass_tag, results_folder, fdr_threshold, plots="off"):
    return run_channel_fdr(ids, list_cols, _dc(), _Spectra(), library, mass_tag, results_folder,
                                  fdr_threshold, plots=plots)


class TestBuildChannelFeatures:
    tag = available_tags["PSMtag_5plex"].with_decoy_channel()     # decoy channel 20

    def test_which_rows_are_scored(self):
        ids = _ids([("(PSMtag_5plex-0)PEPTIDER", 0, False, False, 0.001),
                    ("(PSMtag_5plex-20)PEPTIDER", 20, False, True, 0.001),       # the null: kept
                    ("Decoy_(PSMtag_5plex-0)PEPTIDER", 0, True, False, 0.001),   # library decoy: not
                    ("(PSMtag_5plex-0)PEPTIDER", 0, False, False, 0.5)])         # failed the FDR: not
        matched = [(["b2_1"], [1.0])] * 4
        features = _features(ids, _list_cols(matched), _store(), self.tag, 0.01)
        assert features.select("channel", "is_decoy_channel").rows() == [(0, False), (20, True)]

    def test_each_row_keeps_its_own_fragments(self):
        ids = _ids([("(PSMtag_5plex-0)PEPTIDER", 0, False, False, 0.001),
                    ("(PSMtag_5plex-20)PEPTIDER", 20, False, True, 0.001)])
        matched = [(["b2_1", "b3_1", "y3_1"], [1.0, 1.0, 1.0]),    # every fragment
                   (["y3_1"], [1.0])]                               # the shared one only
        features = _features(ids, _list_cols(matched), _store(), self.tag, 0.01)
        assert features["n_chan_frags"].to_list() == [2, 0]
        assert features["n_shared_frags"].to_list() == [1, 1]

    def test_has_ms1_peak_is_a_feature_and_mz_error_only_the_error(self):
        ids = _ids([("(PSMtag_5plex-0)PEPTIDER", 0, False, False, 0.001),
                    ("(PSMtag_5plex-20)PEPTIDER", 20, False, True, 0.001)]).assign(mz_error=[2e-6, -1.0])
        features = _features(ids, _list_cols([(["b2_1"], [1.0])] * 2), _store(), self.tag, 0.01)
        assert features["has_ms1_peak"].to_list() == [True, False]
        assert features["mz_error"].to_list() == [pytest.approx(2e-6), None]
        assert "has_ms1_peak" in model_columns(features)

    def test_a_precursor_missing_from_the_library_is_null_throughout(self):
        ids = _ids([("(PSMtag_5plex-0)PEPTIDER", 0, False, False, 0.001),
                    ("(PSMtag_5plex-0)NOTINLIB", 0, False, False, 0.001)])
        features = _features(ids, _list_cols([(["b2_1"], [1.0])] * 2),
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
        features = _features(ids, _list_cols([(["b2_1"], [1.0])]), _store(), self.tag, 0.01)
        assert features.select("n_tags", "pep_len", "first_tag_from_n", "last_tag_from_c").row(0) == context

    def test_untagged_precursors_are_not_scored(self):
        """Under a K-only tag a peptide without K is searched once, untagged: no
        channels to tell apart."""
        ids = _ids([("PEPK(K_6C13-0)TIDE", 0, False, False, 0.001),
                    ("PEPTIDER", float("nan"), False, False, 0.001)])
        features = _features(ids, _list_cols([(["b2_1"], [1.0])] * 2), _store(),
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
        return _features(ids, _list_cols([(["b2_1", "y2_1"], [4.0, 6.0])]), store,
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


class TestMS1Features:
    """A channel's MS1 plex-fit coefficients across its set's fit window.  The
    symmetric trace 0, 1, 4, 1, 0 has its weighted mean at cycle 2 and weighted
    variance (1 + 0 + 1) / 6 = 1/3."""

    peak, specs = "0;1;4;1;0", "10;31;52;73;94"

    def test_gaussian_moments_and_spread(self):
        f = ms1_features(self.peak, "0.9;0.9;0.9;0.9;0.9", self.specs, 52, 31)
        assert f["ms1_trace_nonzero"] == 1.0
        assert f["plexfit_sigma"] == pytest.approx(np.sqrt(1 / 3))
        assert f["plexfit_trace_cv"] == pytest.approx(np.sqrt(2.16) / 1.2)   # population std / mean
        assert f["ms1_ms2_apex_delta"] == 52 - 31

    def test_a_peak_fits_a_gaussian_better_than_noise(self):
        peak = ms1_features("0.1;1;5;9;5;1;0.1", "1;1;1;1;1;1;1", "1;2;3;4;5;6;7", 4, 4)
        noise = ms1_features("5;0;6;0;5;0;6", "1;1;1;1;1;1;1", "1;2;3;4;5;6;7", 4, 4)
        assert peak["plexfit_gauss_r2"] > 0.95
        assert noise["plexfit_gauss_r2"] < 0.2

    def test_fit_quality_at_the_apex_and_weighted_by_the_coefficients(self):
        # quality per scan 0.1 0.2 0.9 0.3 nan: the apex is scan 2; the weights are
        # the coefficients, the unknown fifth scan left out
        f = ms1_features(self.peak, "0.1;0.2;0.9;0.3;nan", self.specs, 52, 52)
        assert f["plexfit_ps_at_apex"] == pytest.approx(0.9)
        assert f["plexfit_ps_weighted"] == pytest.approx((1 * 0.2 + 4 * 0.9 + 1 * 0.3) / 6)

    def test_no_fit_window_is_null_throughout(self):
        """ms1_quant merges its outputs with fillna(0): a 0 is no window, not a
        flat trace, and 0 - Ms1_spec_id is not an apex delta."""
        f = ms1_features(0, 0, 0, 0, 52)
        assert all(np.isnan(v) for v in f.values())

    def test_too_few_points_or_a_flat_trace(self):
        short = ms1_features("0;4;0", "1;1;1", "1;2;3", 2, 2)
        flat = ms1_features("0;0;0;0;0", "1;1;1;1;1", self.specs, 52, 52)
        assert short["ms1_trace_nonzero"] == 1.0 and np.isnan(short["plexfit_sigma"])
        assert flat["ms1_trace_nonzero"] == 0.0 and np.isnan(flat["plexfit_gauss_r2"])
        assert short["ms1_ms2_apex_delta"] == 0 and flat["ms1_ms2_apex_delta"] == 0

    def test_fit_quality_must_cover_the_trace(self):
        with pytest.raises(ValueError, match="points for a trace of"):
            ms1_features(self.peak, "0.9;0.9", self.specs, 52, 52)

    def test_in_the_table_without_the_traces(self):
        ids = _ids([("(PSMtag_5plex-0)PEPTIDER", 0, False, False, 0.001)])
        features = _features(ids, _list_cols([(["b2_1"], [1.0])]), _store(),
                                          available_tags["PSMtag_5plex"].with_decoy_channel(), 0.01)
        assert features["plexfit_sigma"].to_list() == [pytest.approx(np.sqrt(1 / 3))]
        assert not {"plexfittrace_all", "plexfittrace_ps_all", "plexfittrace_spec_all"} & set(features.columns)

    def test_the_joint_ms1_ms2_fit_is_carried_from_all_ids(self):
        ids = _ids([("(PSMtag_5plex-0)PEPTIDER", 0, False, False, 0.001)])
        features = _features(ids, _list_cols([(["b2_1"], [1.0])]), _store(),
                                          available_tags["PSMtag_5plex"].with_decoy_channel(), 0.01)
        assert features["gauss_r2_joint"].to_list() == [0.8]
        assert "MS1_quality" not in features.columns      # missing for the decoy channel: a label leak

    def test_a_missing_input_fails_loudly(self):
        ids = _ids([("(PSMtag_5plex-0)PEPTIDER", 0, False, False, 0.001)]).drop(columns="gauss_r2_joint")
        with pytest.raises(KeyError, match="gauss_r2_joint"):
            _features(ids, _list_cols([(["b2_1"], [1.0])]), _store(),
                                   available_tags["PSMtag_5plex"].with_decoy_channel(), 0.01)


class TestLongestConsecutiveScans:
    """Consecutive DIA cycles with an MS2 coefficient; JMod writes no zero
    coefficients, so a missing cycle is the gap."""

    def test_a_cycle_is_a_visit_to_the_scans_own_window(self):
        # each window's k-th scan is cycle k, however the two are interleaved
        cycles = dict(_scan_cycles().iter_rows())
        assert [cycles[101 + 22 * k] for k in range(5)] == [0, 1, 2, 3, 4]     # window A
        assert [cycles[102 + 22 * k] for k in range(5)] == [0, 1, 2, 3, 4]     # window B
        # two MS1 scans a cycle: the window comes round every second MS1 scan, and
        # it is still one cycle per visit
        assert dict(ms2_cycles([30, 52, 74], [WINDOW_A] * 3).iter_rows()) == {30: 0, 52: 1, 74: 2}

    def test_runs_of_cycles(self):
        longest = longest_consecutive_scans(_dc(), _scan_cycles(), ["seq", "z"])
        # cycles 0, 1, 2 then 4; cycle 1 seen in both windows counts once
        assert longest[("(PSMtag_5plex-0)PEPTIDER", 2.0)] == 3
        assert longest[("(PSMtag_5plex-20)PEPTIDER", 2.0)] == 1     # one scan

    def test_a_cycle_where_nothing_was_fitted_is_still_a_gap(self):
        # without channel 20's row no coefficient is in cycle 3, but the cycles
        # come from the run's scans, not from the coefficients
        only_zero = _dc().filter(pl.col("seq") == "(PSMtag_5plex-0)PEPTIDER")
        assert longest_consecutive_scans(only_zero, _scan_cycles(), ["seq", "z"]) == {
            ("(PSMtag_5plex-0)PEPTIDER", 2.0): 3}

    def test_in_the_table(self):
        ids = _ids([("(PSMtag_5plex-0)PEPTIDER", 0, False, False, 0.001),
                    ("(PSMtag_5plex-20)PEPTIDER", 20, False, True, 0.001),
                    ("(PSMtag_5plex-0)NOTSCANNED", 0, False, False, 0.001)])
        features = _features(ids, _list_cols([(["b2_1"], [1.0])] * 3), _store(),
                             available_tags["PSMtag_5plex"].with_decoy_channel(), 0.01)
        assert features["longest_consecutive_scans"].to_list() == [3, 1, None]


def _carried(**values):
    """One row of every carried feature at a real value, with *values* set."""
    row = {c: 1.0 for c in _CARRIED_FEATURES}
    row.update(n_scans=3, n_corr_frags=5, channels_matched=2, plexfitMS1=4.0)
    row.update(values)
    return null_placeholders(pl.DataFrame([row])).row(0, named=True)


class TestPlaceholders:
    """JMod's stand-ins for undefined values become null; real values that look
    like them stay.  ms1_quant merges with fillna(0) over every column, so most
    are 0s."""

    def test_pairwise_correlation_summaries_need_two_fragments(self):
        assert _carried(n_corr_frags=1, mean_frag_corr=0.0)["mean_frag_corr"] is None
        # with two or more, a 0 is real: the spread of a single pair
        assert _carried(n_corr_frags=2, std_frag_corr=0.0)["std_frag_corr"] == 0.0

    def test_ranked_slots_beyond_the_fragments_there_are(self):
        row = _carried(top10_pair_corr=-1.0, top9_pair_corr=0.4)
        assert row["top10_pair_corr"] is None and row["top9_pair_corr"] == 0.4

    @pytest.mark.parametrize("column", ["mean_prec_frag_corr", "max_prec_frag_corr", "ms1_cor", "iso_cor",
                                        "traceproduct"])
    def test_a_correlation_of_exactly_zero_is_fillna(self, column):
        assert _carried(**{column: 0.0})[column] is None

    def test_a_perfect_traceproduct_is_not_zero(self):
        # log10(1 * 1 * 1 + 1e-6): every correlation perfect
        assert _carried(traceproduct=4.3e-7)["traceproduct"] == 4.3e-7

    def test_single_scan_precursors(self):
        single = _carried(n_scans=1, smoothness=0.0, apex_pc1=0.0)
        assert single["smoothness"] is None and single["apex_pc1"] is None
        assert _carried(n_scans=2, smoothness=0.0)["smoothness"] == 0.0

    def test_single_channel_sets(self):
        alone = _carried(channels_matched=1, median_gof_stats=0.0, diff_gof_stats_from_median=0.0,
                         frac_shared_intensity=-1.0)
        assert alone["median_gof_stats"] is None and alone["diff_gof_stats_from_median"] is None
        assert alone["frac_shared_intensity"] is None
        # in a set of three, the median channel's distance from the median is a real 0
        assert _carried(channels_matched=3, diff_gof_stats_from_median=0.0)["diff_gof_stats_from_median"] == 0.0
        assert _carried(frac_shared_intensity=0.0)["frac_shared_intensity"] == 0.0     # nothing shared

    @pytest.mark.parametrize("column", ["im_error", "med_frag_error"])
    def test_fillna_zeros(self, column):
        # no ion mobility; no fragment errors
        assert _carried(**{column: 0.0})[column] is None

    def test_no_ms1_peak_is_the_searchs_minus_one(self):
        # the search's -1 for no MS1 peak in tolerance; any absolute error is real
        assert _carried(mz_error=-1.0)["mz_error"] is None
        assert _carried(mz_error=2e-6)["mz_error"] == pytest.approx(2e-6)

    def test_plex_fit_quality_needs_a_plex_fit(self):
        # no MS1 coefficient for the channel: argmax lands on the window's first
        # scan, whatever its quality
        assert _carried(plexfitMS1=0.0, plexfitMS1_p=0.7)["plexfitMS1_p"] is None
        assert _carried(plexfitMS1_p=0.0)["plexfitMS1_p"] is None
        # a Pearson r: negative is real
        assert _carried(plexfitMS1_p=-0.3)["plexfitMS1_p"] == -0.3

    def test_in_the_table_and_the_rule_columns_are_not(self):
        ids = _ids([("(PSMtag_5plex-0)PEPTIDER", 0, False, False, 0.001)]).assign(n_scans=1, smoothness=0.0)
        features = _features(ids, _list_cols([(["b2_1"], [1.0])]), _store(),
                             available_tags["PSMtag_5plex"].with_decoy_channel(), 0.01)
        assert features["smoothness"].to_list() == [None]
        assert features["n_scans"].to_list() == [1]
        assert not {"channels_matched", "plexfitMS1"} & set(features.columns)


class TestRunScoreAndSequence:
    def test_predval_and_rt_are_carried_and_qvalue_is_not(self):
        ids = _ids([("(PSMtag_5plex-0)PEPTIDER", 0, False, False, 0.001)])
        features = _features(ids, _list_cols([(["b2_1"], [1.0])]), _store(),
                             available_tags["PSMtag_5plex"].with_decoy_channel(), 0.01)
        assert features["PredVal"].to_list() == [0.9] and features["rt"].to_list() == [61.5]
        assert "Qvalue" not in features.columns and "BestChannel_Qvalue" in features.columns
        # JMod's cosine is the scan's fit, so it is carried under that name
        assert features["scan_fit_cosine"].to_list() == [1.0] and "cosine" not in features.columns
        assert features["z"].to_list() == [2]

    def test_last_aa_code(self):
        ids = _ids([("(PSMtag_5plex-0)PEPTIDER", 0, False, False, 0.001),
                    ("(PSMtag_5plex-0)PEPTIDEK(PSMtag_5plex-0)", 0, False, False, 0.001),
                    ("(PSMtag_5plex-0)PEPTIDEX", 0, False, False, 0.001)])
        features = _features(ids, _list_cols([(["b2_1"], [1.0])] * 3), _store(),
                             available_tags["PSMtag_5plex"].with_decoy_channel(), 0.01)
        # R and K by their place in the alphabet; a non-standard residue is -1
        assert features["last_aa_code"].to_list() == [14, 8, -1]
        assert "last_aa" not in features.columns


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
        features = _features(ids, _list_cols([(["b2_1"], [1.0])]), Repeating(),
                                          available_tags["PSMtag_5plex"].with_decoy_channel(), 0.01)
        assert features["n_lib_chan_frags"].to_list() == [2]          # b2 and b3, not b2 twice
        assert any("more than once" in r.getMessage() for r in app_log)


_IMPORTANCE = pd.DataFrame({"share_of_gain": [0.6, 0.4], "share_of_gain_sd": [0.05, 0.02]},
                           index=pd.Index(["ms1_cor", "z"], name="feature"))


class TestWriteChannelFeatures:
    def test_written_to_the_channel_fdr_folder(self, tmp_path):
        tag = available_tags["PSMtag_5plex"].with_decoy_channel()
        ids = _ids([("(PSMtag_5plex-0)PEPTIDER", 0, False, False, 0.001)])
        _write(ids, _list_cols([(["b2_1"], [1.0])]), _store(), tag, str(tmp_path), 0.01)
        written = pl.read_parquet(tmp_path / FEATURE_FOLDER / FEATURE_FILE)
        assert written["n_chan_frags"].to_list() == [1]

    def test_nothing_without_a_decoy_channel(self, tmp_path):
        ids = _ids([("(PSMtag_5plex-0)PEPTIDER", 0, False, False, 0.001)])
        assert _write(ids, _list_cols([(["b2_1"], [1.0])]), _store(),
                                      available_tags["PSMtag_5plex"], str(tmp_path), 0.01) is None
        assert not (tmp_path / FEATURE_FOLDER).exists()

    def test_too_few_decoys_for_a_fit_leaves_the_scores_null(self, tmp_path):
        tag = available_tags["PSMtag_5plex"].with_decoy_channel()
        ids = _ids([("(PSMtag_5plex-0)PEPTIDER", 0, False, False, 0.001),
                    ("(PSMtag_5plex-20)PEPTIDER", 20, False, True, 0.001)])
        _write(ids, _list_cols([(["b2_1"], [1.0])] * 2), _store(), tag, str(tmp_path), 0.01)
        written = pl.read_parquet(tmp_path / FEATURE_FOLDER / FEATURE_FILE)
        assert written["ChannelScore"].to_list() == [None, None]
        assert written["Channel_Qvalue"].to_list() == [None, None]
        assert not (tmp_path / FEATURE_FOLDER / FEATURE_IMPORTANCE_FILE).exists()

    def test_the_model_is_fitted_on_the_features_only(self):
        columns = ["seq", "z", "channel", "is_decoy_channel", "BestChannel_Qvalue", "untag_seq", "protein",
                   "PredVal", "cosine_chan", "ChannelScore", "Channel_Qvalue"]
        assert model_columns(pl.DataFrame({c: [1.0] for c in columns})) == ["z", "PredVal", "cosine_chan"]

    def test_no_ecdfs_unless_asked(self, tmp_path):
        tag = available_tags["PSMtag_5plex"].with_decoy_channel()
        ids = _ids([("(PSMtag_5plex-0)PEPTIDER", 0, False, False, 0.001)])
        _write(ids, _list_cols([(["b2_1"], [1.0])]), _store(), tag, str(tmp_path), 0.01)
        # the other plots always (here no fit, so no importance or IDs vs q-value)
        assert sorted(p.name for p in (tmp_path / FEATURE_FOLDER).iterdir()) == [BEST_CHANNEL_PLOT,
                                                                                  FEATURE_FILE]

    def test_written_beside_the_table_when_asked(self, tmp_path, monkeypatch):
        import src.channel_fdr_plots
        calls = []
        monkeypatch.setattr(src.channel_fdr_plots, "plot_channel_fdr",
                            lambda features, folder, importance, split, fdr_threshold, ecdfs, workers:
                            calls.append((features.height, folder, importance, split, fdr_threshold,
                                          ecdfs)))
        tag = available_tags["PSMtag_5plex"].with_decoy_channel()
        ids = _ids([("(PSMtag_5plex-0)PEPTIDER", 0, False, False, 0.001)])
        _write(ids, _list_cols([(["b2_1"], [1.0])]), _store(), tag, str(tmp_path), 0.01, plots="KR")
        # too few decoys to fit: no importance
        assert calls == [(1, os.path.join(str(tmp_path), FEATURE_FOLDER), None, "KR", 0.01, True)]
        _write(ids, _list_cols([(["b2_1"], [1.0])]), _store(), tag, str(tmp_path), 0.01)
        assert calls[1][3:] == ("all", 0.01, False)                    # off: no ECDFs, one panel

    def test_feature_importance_round_trips_through_its_file(self, tmp_path):
        write_feature_importance(_IMPORTANCE, str(tmp_path))
        back = read_feature_importance(str(tmp_path))
        pd.testing.assert_frame_equal(back, _IMPORTANCE)
        assert read_feature_importance(str(tmp_path / "elsewhere")) is None


class TestPerChannelQvalues:
    def test_each_channel_against_the_whole_decoy_channel(self):
        channel = np.array([0.0, 0.0, 4.0, 4.0, 20.0, 20.0])
        is_decoy = channel == 20
        score = np.array([9.0, 1.0, 8.0, 7.0, 5.0, 2.0])
        q = per_channel_qvalues(channel, score, is_decoy)
        assert np.isnan(q[is_decoy]).all()
        # channel 0 alone with the decoys: 9 T, 5 D, 2 D, 1 T
        assert q[:2] == pytest.approx(target_decoy_qvalues(np.array([9.0, 1.0, 5.0, 2.0]),
                                                           np.array([False, False, True, True]))[:2])
        # channel 4 is not ranked against channel 0: 8 T, 7 T, 5 D, 2 D
        assert q[2:4] == pytest.approx([0.5, 0.5])


class TestPerChannelPep:
    def test_each_channel_against_the_whole_decoy_channel(self):
        rng = np.random.default_rng(1)
        channel = np.repeat([0.0, 4.0, 20.0], 50)
        is_decoy = channel == 20
        score = rng.normal(size=channel.size) + np.where(channel == 0, 1.0, 0.0)
        pep = per_channel_pep(channel, score, is_decoy)
        assert np.isnan(pep[is_decoy]).all()
        alone = (channel == 0) | is_decoy
        assert pep[channel == 0] == pytest.approx(estimate_pep(score[alone], is_decoy[alone])[:50])
        # channel 4 does not change channel 0's
        moved = score.copy()
        moved[channel == 4] += 10
        assert per_channel_pep(channel, moved, is_decoy)[channel == 0] == pytest.approx(pep[channel == 0])

    def test_not_pooled(self):
        # two real channels, both pure null: each is as decoy-like as the decoy
        # channel itself, which pooling them would halve
        rng = np.random.default_rng(2)
        channel = np.repeat([0.0, 4.0, 20.0], 400)
        is_decoy = channel == 20
        score = rng.normal(size=channel.size)
        assert (np.median(per_channel_pep(channel, score, is_decoy)[~is_decoy])
                > 1.5 * np.median(estimate_pep(score, is_decoy)[~is_decoy]))


class TestModelMatrix:
    def test_columns_with_no_information_are_left_out(self):
        features = pl.DataFrame({"a": [1.0, 2.0, None], "constant": [3.0, 3.0, 3.0],
                                 "empty": [None, None, None], "n": [1, 2, 3]},
                                schema_overrides={"empty": pl.Float64})
        X, dead = model_matrix(features, ["a", "constant", "empty", "n"])
        assert list(X.columns) == ["a", "n"] and dead == ["constant", "empty"]
        assert np.isnan(X["a"].iloc[2])                       # a missing value stays missing

    def test_infinite_values_are_an_error(self):
        with pytest.raises(ValueError, match="infinite"):
            model_matrix(pl.DataFrame({"a": [1.0, np.inf, 2.0]}), ["a"])


def _fit_run(n_peptides=150, seed=0):
    """A synthetic run: each peptide in real channels 0 and 4 and the decoy
    channel 20.  A real channel is there for half the peptides, and then its
    "signal" is high; an empty one (and the decoy) draws from the null."""
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_peptides):
        for channel in (0.0, 4.0, 20.0):
            present = channel != 20 and p % 2 == 0
            rows.append({"untag_seq": f"PEP{p}", "channel": channel, "is_decoy_channel": channel == 20,
                         "present": present, "signal": rng.normal(4.0 if present else 0.0, 1.0),
                         "noise": rng.normal(), "constant": 1.0})
    return pl.DataFrame(rows)


@pytest.fixture(scope="module")
def fitted():
    features = _fit_run()
    return features, fit_channel_fdr(features, ["signal", "noise", "constant"])


class TestFit:
    def test_channels_that_are_there_pass_and_empty_ones_do_not(self, fitted):
        features, (score, q, _) = fitted
        present = features["present"].to_numpy()
        real = ~features["is_decoy_channel"].to_numpy()
        assert np.median(q[present]) < 0.05
        assert np.median(q[real & ~present]) > 0.3
        assert score.shape == q.shape == (features.height,)

    def test_the_decoy_channel_has_no_qvalue(self, fitted):
        features, (_, q, _) = fitted
        assert np.isnan(q[features["is_decoy_channel"].to_numpy()]).all()
        assert np.isfinite(q[~features["is_decoy_channel"].to_numpy()]).all()

    def test_importance_is_the_evidence(self, fitted):
        _, (_, _, importance) = fitted
        assert importance.index[0] == "signal"
        assert "constant" not in importance.index                 # left out, no information
        assert importance["share_of_gain"].sum() == pytest.approx(1.0)
        assert (importance["share_of_gain_sd"] >= 0).all()           # over the folds

    def test_too_few_decoys_is_no_fit(self):
        features = _fit_run(n_peptides=MIN_DECOYS - 1)
        assert fit_channel_fdr(features, ["signal", "noise"]) is None


class TestProteinChannelQvalues:
    """Whether a protein is in a channel: its best precursor in the channel,
    each real channel ranked against the decoy channel."""

    @staticmethod
    def _features():
        # P1 has two precursors in channel 0: its best, 0.9, counts
        return pl.DataFrame({
            "protein": ["P1", "P1", "P1", "P1", "P2", "P2", "P2"],
            "channel": [0.0, 0.0, 4.0, 20.0, 0.0, 4.0, 20.0],
            "is_decoy_channel": [False, False, False, True, False, False, True],
            "ChannelScore": [0.2, 0.9, 0.1, 0.5, 0.8, 0.7, 0.3]})

    def _ids(self):
        return pd.DataFrame({"protein": ["P1", "P1", "P2", "P1", "P1", "P3"],
                             "channel": [0, 4, 4, 20, 0, 0],
                             "is_decoy": [False, False, False, False, True, False]},
                            index=[10, 11, 12, 13, 14, 15])

    def test_each_channel_against_the_decoy_channel(self):
        q = protein_channel_qvalues(self._ids(), self._features())
        assert list(q.index) == [10, 11, 12, 13, 14, 15]
        # channel 0: P1 0.9 T, P2 0.8 T, then the decoy channel's 0.5, 0.3: 1/2 for both
        # channel 4: P2 0.7 T, 0.5 D, 0.3 D, P1 0.1 T: 1/1, 2/1, 3/1, 3/2, the lowest from below
        assert q.iloc[:3].tolist() == pytest.approx([0.5, 1.5, 1.0])
        # the decoy channel, a library decoy (its target's protein name), a protein not scored
        assert np.isnan(q.iloc[3]) and np.isnan(q.iloc[4]) and np.isnan(q.iloc[5])

    def test_nothing_fitted(self):
        assert protein_channel_qvalues(self._ids(), None).isna().all()
        assert protein_channel_qvalues(self._ids(), self._features().drop("ChannelScore")).isna().all()


class TestChannelQvalues:
    """Channel_Qvalue back onto the run's IDs, by tagged sequence and charge."""

    def _ids(self):
        return pd.DataFrame({"seq": ["(PSMtag_5plex-0)PEPTIDER", "Decoy_(PSMtag_5plex-0)REDITPEP",
                                     "(PSMtag_5plex-4)PEPTIDER", "(PSMtag_5plex-0)PEPTIDER"],
                             "z": [2, 2, 2, 3]}, index=[10, 11, 12, 13])

    def test_onto_its_rows_and_null_elsewhere(self):
        features = pl.DataFrame({"seq": ["(PSMtag_5plex-4)PEPTIDER", "(PSMtag_5plex-0)PEPTIDER"],
                                 "z": [2, 2], "Channel_Qvalue": [0.2, 0.001]})
        q = channel_qvalues(self._ids(), features)
        assert list(q.index) == [10, 11, 12, 13]
        assert q.iloc[0] == 0.001 and q.iloc[2] == 0.2
        assert np.isnan(q.iloc[1]) and np.isnan(q.iloc[3])        # a decoy; another charge

    def test_nothing_fitted(self):
        assert channel_qvalues(self._ids(), pl.DataFrame()).isna().all()

    def test_a_repeated_precursor_channel_is_an_error(self):
        features = pl.DataFrame({"seq": ["(PSMtag_5plex-0)PEPTIDER"] * 2, "z": [2, 2],
                                 "Channel_Qvalue": [0.1, 0.2]})
        with pytest.raises(ValueError, match="repeat"):
            channel_qvalues(self._ids(), features)
