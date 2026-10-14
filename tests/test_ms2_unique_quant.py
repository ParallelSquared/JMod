"""MS2 quant from fragments that can be assigned to one plexDIA channel."""
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ms2_unique_quant import (  # noqa: E402
    QUANT_COL, add_channel_unique_coeff, channel_unique_masks, jmod_coefficient)
from src.ms2_isotope_xtalk import (  # noqa: E402
    fragment_isotope_pattern, deconvolve_fragment_group, base_frag_key,
    cascading_channel_fit, amplitude_from_coeff, isolated_envelope_fraction)
from src.mass_tags import read_json_to_massTag  # noqa: E402

MASS_TAGS_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src", "MassTags")


class TestJModCoefficient(unittest.TestCase):
    """Closed form of the single-candidate NNLS that produces ``coeff``."""

    def test_recovers_a_known_scale_when_everything_is_matched(self):
        # frac_lib_int = 1 -> no penalty row, normalisation cancels
        lib = np.array([0.5, 0.3, 0.2])
        self.assertAlmostEqual(
            jmod_coefficient(lib, lib * 1000.0, 1.0), 1000.0, places=6)

    def test_penalty_row_pulls_the_estimate_down(self):
        lib = np.array([0.5, 0.3, 0.2])
        obs = lib * 1000.0
        full = jmod_coefficient(lib, obs, 1.0)
        half = jmod_coefficient(lib, obs, 0.5)   # half the library unmatched
        self.assertLess(half, full)

    def test_mask_drops_rows(self):
        lib = np.array([0.5, 0.3, 0.2])
        obs = np.array([500.0, 300.0, 9999.0])   # third row is contaminated
        clean = jmod_coefficient(lib, obs, 1.0, mask=[True, True, False])
        dirty = jmod_coefficient(lib, obs, 1.0)
        self.assertAlmostEqual(clean, 1000.0, places=6)
        self.assertGreater(dirty, clean)

    def test_degenerate_input(self):
        self.assertTrue(np.isnan(jmod_coefficient([], [], 1.0)))
        self.assertTrue(np.isnan(jmod_coefficient([0.0, 0.0], [1.0, 1.0], 1.0)))
        self.assertTrue(np.isnan(
            jmod_coefficient([1.0, 1.0], [1.0, 1.0], 1.0, mask=[False, False])))


class TestChannelUniqueMasks(unittest.TestCase):
    def test_shared_mz_is_flagged_across_channels(self):
        # b-ions shift with the tag; y-ions of an R-terminating peptide do not.
        names = [np.array(["b2", "b3", "y2", "y3"])] * 3
        mzs = [np.array([200.0, 300.0, 500.0, 600.0]),
               np.array([204.0, 304.0, 500.0, 600.0]),
               np.array([208.0, 308.0, 500.0, 600.0])]
        masks = channel_unique_masks(names, mzs)
        for m in masks:
            np.testing.assert_array_equal(m, [True, True, False, False])

    def test_all_unique_when_every_fragment_shifts(self):
        names = [np.array(["b2", "y2"])] * 2
        mzs = [np.array([200.0, 500.0]), np.array([204.0, 504.0])]
        for m in channel_unique_masks(names, mzs):
            np.testing.assert_array_equal(m, [True, True])

    def test_single_channel_precursor_has_nothing_to_compare(self):
        m = channel_unique_masks([np.array(["b2", "y2"])],
                                 [np.array([200.0, 500.0])])
        np.testing.assert_array_equal(m[0], [True, True])

    def test_tolerance_is_respected(self):
        names = [np.array(["y2"])] * 2
        mzs = [np.array([500.000]), np.array([500.005])]
        self.assertFalse(channel_unique_masks(names, mzs, mz_tol=0.01)[0][0])
        self.assertTrue(channel_unique_masks(names, mzs, mz_tol=0.001)[0][0])


class TestGroupScopedMasks(unittest.TestCase):
    """Uniqueness is a property of the whole plex group.

    Regression guard for a real bug: masks were first computed only over the
    rows surviving the q-value gate, so a superposed fragment was called unique
    whenever its partner channels missed the cutoff.  On JD0435 that inflated
    the library intensity retained by R-terminating precursors from 44% to 91%.
    """

    def frame(self):
        rows = []
        for lab, shift, q in (("0", 0.0, 0.001), ("4", 4.0, 0.5), ("8", 8.0, 0.5)):
            rows.append({
                "untag_prec": "PEPTIDER_2", "time_channel": 0.0,
                "channel": f"0_{lab}", "window_mz": 500.0, "rt": 20.0,
                "BestChannel_Qvalue": q, "is_decoy": False,
                "frag_names": ["b2", "y2"],
                "frag_mz": [200.0 + shift, 500.0],
                "frag_int": [1.0, 1.0],
            })
        return pd.DataFrame(rows)

    def test_y_ion_stays_superposed_even_when_partners_fail_fdr(self):
        names = [np.array(["b2", "y2"])] * 3
        mzs = [np.array([200.0, 500.0]), np.array([204.0, 500.0]),
               np.array([208.0, 500.0])]
        for m in channel_unique_masks(names, mzs):
            np.testing.assert_array_equal(m, [True, False])

    def test_q_gate_limits_the_fit_not_the_mask(self):
        d = self.frame()
        out = add_channel_unique_coeff(d, spectra=None, plexDIA=True)
        # No spectra -> nothing refitted, but the call must not raise.
        self.assertNotIn(QUANT_COL, out.columns)


class TestSkipConditions(unittest.TestCase):
    def base(self):
        return pd.DataFrame({
            "untag_prec": ["P_2"], "time_channel": [0.0], "window_mz": [500.0],
            "rt": [20.0], "frag_names": [["b2"]], "frag_mz": [[200.0]],
            "frag_int": [[1.0]],
        })

    def test_label_free_is_skipped(self):
        out = add_channel_unique_coeff(self.base(), spectra=object(),
                                       plexDIA=False)
        self.assertNotIn(QUANT_COL, out.columns)

    def test_no_spectra_is_skipped(self):
        out = add_channel_unique_coeff(self.base(), spectra=None, plexDIA=True)
        self.assertNotIn(QUANT_COL, out.columns)

    def test_missing_columns_is_a_no_op(self):
        out = add_channel_unique_coeff(pd.DataFrame({"untag_prec": ["x"]}),
                                       spectra=object(), plexDIA=True)
        self.assertNotIn(QUANT_COL, out.columns)

    def test_only_the_apex_value_is_emitted_and_normalized(self):
        """The summed variants recover fold change worse than the apex value.

        On JD0435: 0.79 / 0.75 / 0.90 against 0.93, at ~40% worse spread. The
        per-scan estimate rests on 3-4 fragments for an R-terminating precursor,
        so summing scans adds their errors instead of averaging them.

        The two `*_Area` variants were later reinstated as REPORTED quantities,
        so they do appear in MS2_QUANT_COLS and are RT-normalized like any other
        reported intensity. What the measurement above rules out is letting them
        influence IDENTIFICATION, and that is enforced separately: both sit in
        fdr_analysis's scoring drop list. This test pins that split, because
        normalizing a reported column is harmless while scoring on it is not.
        """
        from src.quant_normalization import MS2_QUANT_COLS
        self.assertIn(QUANT_COL, MS2_QUANT_COLS)

        # never emitted at all
        for gone in ("coeff_ChannelUnique_plex_Area",
                     "coeff_ChannelUnique_ConsensusPlexArea",
                     "coeff_plex_Area", "coeff_ConsensusPlexArea"):
            self.assertNotIn(gone, MS2_QUANT_COLS)

        # reported and normalized, but must never reach the scorer
        import inspect
        import src.fdr_analysis as fdr
        drop_src = inspect.getsource(fdr.score_precursors)
        for reported_not_scored in ("coeff_Area", "coeff_ChannelUnique_Area"):
            self.assertIn(reported_not_scored, MS2_QUANT_COLS)
            self.assertIn(reported_not_scored, drop_src)

    def test_default_is_on(self):
        from src.default_dict import default_dict
        self.assertTrue(default_dict["ms2_channel_unique"]["default"])
        self.assertEqual(
            default_dict["ms2_channel_unique_qvalue"]["default"], 0.01)


class TestBaseFragKey(unittest.TestCase):
    def test_strips_iso_suffix(self):
        self.assertEqual(base_frag_key("b3_1_iso1"), "b3_1")
        self.assertEqual(base_frag_key("b3_1_iso12"), "b3_1")

    def test_leaves_monoisotopic_names_alone(self):
        self.assertEqual(base_frag_key("b3_1"), "b3_1")
        self.assertEqual(base_frag_key("y10-H2O_2"), "y10-H2O_2")


class TestFragmentIsotopePattern(unittest.TestCase):
    """Physical sanity of the theoretical envelope this whole module rests on."""

    def setUp(self):
        self.tag = read_json_to_massTag(MASS_TAGS_DIR, "PSMtag_9plex.json")

    def test_monoisotopic_offset_and_relative_intensity_are_one(self):
        off, ri = fragment_isotope_pattern(
            "E(PSMtag_9plex-0)MAAAPPGR", "b3_1", self.tag, n_iso=4)
        self.assertAlmostEqual(off[0], 0.0)
        self.assertAlmostEqual(ri[0], 1.0)

    def test_offsets_increase_and_relative_intensity_decays(self):
        off, ri = fragment_isotope_pattern(
            "E(PSMtag_9plex-0)MAAAPPGR", "b3_1", self.tag, n_iso=5)
        self.assertTrue(np.all(np.diff(off) > 0))
        self.assertTrue(np.all(np.diff(ri) < 0))

    def test_channel_step_nearly_matches_two_isotope_orders(self):
        """The whole point of this module: the 9-plex tag's per-step delta
        (2.0067 Da) is close enough to 2x the natural isotope spacing that a
        channel's own M+2 peak sits within a few mDa of the NEXT channel's
        monoisotopic target -- confirmed here against the tag's own JSON
        delta rather than a hard-coded number, so this fails loudly if the
        tag definition ever changes."""
        off, _ = fragment_isotope_pattern(
            "E(PSMtag_9plex-0)MAAAPPGR", "b3_1", self.tag, n_iso=5)
        step_0_to_2 = self.tag.mass_dict["PSMtag_9plex-2"] - \
            self.tag.mass_dict["PSMtag_9plex-0"]
        self.assertLess(abs(off[2] - step_0_to_2), 0.01)

    def test_cache_returns_identical_pattern(self):
        cache = {}
        off1, ri1 = fragment_isotope_pattern(
            "E(PSMtag_9plex-4)MAAAPPGR", "b3_1", self.tag, n_iso=4, cache=cache)
        self.assertTrue(cache)
        off2, ri2 = fragment_isotope_pattern(
            "E(PSMtag_9plex-4)MAAAPPGR", "b3_1", self.tag, n_iso=4, cache=cache)
        np.testing.assert_array_equal(off1, off2)
        np.testing.assert_array_equal(ri1, ri2)


class TestDeconvolveFragmentGroup(unittest.TestCase):
    """Per-fragment-key joint solve -- a building block, superseded as the
    primary mechanism by ``cascading_channel_fit`` (see that class below and
    ms2_isotope_xtalk's module docstring for why: this alone can't fit a
    whole channel's several fragments together without either re-deriving
    each key's siblings independently or double-counting library weight)."""

    def setUp(self):
        self.tag = read_json_to_massTag(MASS_TAGS_DIR, "PSMtag_9plex.json")
        self.seq0 = "E(PSMtag_9plex-0)MAAAPPGR"
        self.seq2 = "E(PSMtag_9plex-2)MAAAPPGR"
        self.mz0 = 644.2545167
        self.mz2 = self.mz0 + self.tag.mass_dict["PSMtag_9plex-2"] - \
            self.tag.mass_dict["PSMtag_9plex-0"]

    def test_recovers_true_intensity_under_synthetic_contamination(self):
        _, ri0 = fragment_isotope_pattern(self.seq0, "b3_1", self.tag, n_iso=5)
        true0, true2 = 1000.0, 500.0
        contamination = true0 * ri0[2]
        # A plain ppm-window sum would report obs2 as channel 2's own
        # intensity -- this is exactly the bug the module fixes.
        obs0, obs2 = true0, true2 + contamination
        self.assertGreater(contamination, 50.0)  # the effect must be non-trivial

        corrected = deconvolve_fragment_group(
            [self.seq0, self.seq2], "b3_1", [self.mz0, self.mz2],
            [obs0, obs2], self.tag, n_iso=5)
        np.testing.assert_allclose(corrected, [true0, true2], atol=1e-6)

    def test_naive_extraction_would_have_overstated_the_later_channel(self):
        _, ri0 = fragment_isotope_pattern(self.seq0, "b3_1", self.tag, n_iso=5)
        true0, true2 = 1000.0, 500.0
        obs2_naive = true2 + true0 * ri0[2]
        self.assertGreater(obs2_naive, true2 * 1.1)

    def test_silent_earlier_channel_contaminates_nothing(self):
        """If channel 0 has no real signal, it can contribute no isotope
        tail either -- contamination is deterministic given the earlier
        channel's true intensity, never an independent free variable."""
        corrected = deconvolve_fragment_group(
            [self.seq0, self.seq2], "b3_1", [self.mz0, self.mz2],
            [0.0, 500.0], self.tag, n_iso=5)
        np.testing.assert_allclose(corrected, [0.0, 500.0], atol=1e-6)

    def test_single_channel_is_untouched(self):
        out = deconvolve_fragment_group(
            [self.seq0], "b3_1", [self.mz0], [777.0], self.tag, n_iso=5)
        np.testing.assert_array_equal(out, [777.0])

    def test_later_channel_cannot_contaminate_an_earlier_one(self):
        """Isotope tails only run up in mass -- channel 2 must not be allowed
        to explain any of channel 0's own (earlier, lower-mass) signal."""
        corrected = deconvolve_fragment_group(
            [self.seq0, self.seq2], "b3_1", [self.mz0, self.mz2],
            [1000.0, 10000.0], self.tag, n_iso=5)
        # channel 0 sees none of channel 2's huge intensity
        self.assertAlmostEqual(corrected[0], 1000.0, places=3)


class TestCascadingChannelFit(unittest.TestCase):
    """The mechanism actually wired into production: channels processed in
    ascending mass order, each using lower-mass siblings' ALREADY-computed
    coefficients (known constants, not free unknowns) to subtract
    contamination before its own jmod_coefficient refit."""

    def setUp(self):
        self.tag = read_json_to_massTag(MASS_TAGS_DIR, "PSMtag_9plex.json")
        self.seq0 = "E(PSMtag_9plex-0)MAAAPPGR"
        self.seq2 = "E(PSMtag_9plex-2)MAAAPPGR"
        self.mz0 = 644.2545167
        self.mz2 = self.mz0 + self.tag.mass_dict["PSMtag_9plex-2"] - \
            self.tag.mass_dict["PSMtag_9plex-0"]
        self.names = {0: np.array(["b3_1"]), 1: np.array(["b3_1"])}
        self.mzs = {0: np.array([self.mz0]), 1: np.array([self.mz2])}
        self.libs = {0: np.array([1.0]), 1: np.array([1.0])}
        self.masks = {0: np.array([True]), 1: np.array([True])}
        self.fracs = {0: 1.0, 1: 1.0}
        self.seqs = {0: self.seq0, 1: self.seq2}
        self.group_rows = [0, 1]

    def test_first_channel_with_no_known_contaminators_matches_jmod_coefficient(self):
        """No known_coeffs entries yet (this channel is processed first) ->
        nothing gets subtracted -> exact match to the plain formula."""
        true0 = 1000.0
        extract0 = lambda mzarr: np.array([true0])
        c0 = cascading_channel_fit(
            0, self.group_rows, self.seqs, self.names, self.mzs, self.libs,
            self.masks, self.fracs, {}, extract0, self.tag, n_iso=5,
            mz_tol_da=0.015, cache={})
        self.assertAlmostEqual(c0, jmod_coefficient(
            self.libs[0], np.array([true0]), self.fracs[0], mask=self.masks[0]))
        self.assertAlmostEqual(c0, true0, places=6)

    def test_recovers_true_intensity_given_the_contaminators_known_coefficient(self):
        _, ri0 = fragment_isotope_pattern(self.seq0, "b3_1", self.tag, n_iso=5)
        true0, true2 = 1000.0, 500.0
        contamination = true0 * ri0[2]
        self.assertGreater(contamination, 50.0)  # non-trivial effect

        known = {0: true0}  # channel 0 already fitted from its OWN scan
        extract2 = lambda mzarr: np.array([true2 + contamination])
        c2 = cascading_channel_fit(
            1, self.group_rows, self.seqs, self.names, self.mzs, self.libs,
            self.masks, self.fracs, known, extract2, self.tag, n_iso=5,
            mz_tol_da=0.015, cache={})
        self.assertAlmostEqual(c2, true2, places=4)

    def _spectrum(self, amounts):
        """extract_fn over a spectrum holding each channel's full envelope.

        ``amounts`` maps channel label -> true amount.
        """
        peaks = []
        for lbl, amt in amounts.items():
            seq = f"E(PSMtag_9plex-{lbl})MAAAPPGR"
            mz = self.mz0 + (self.tag.mass_dict[f"PSMtag_9plex-{lbl}"]
                             - self.tag.mass_dict["PSMtag_9plex-0"])
            off, rel = fragment_isotope_pattern(seq, "b3_1", self.tag, n_iso=5)
            for o in range(len(off)):
                peaks.append((mz + off[o], amt * rel[o]))
        pmz = np.array([p[0] for p in peaks])
        pin = np.array([p[1] for p in peaks])
        return lambda a: np.array(
            [pin[np.abs(pmz - m) <= 0.005].sum() for m in a])



    def test_clean_spectrum_with_no_contamination_is_left_alone(self):
        """Nothing at any lower rung -> nothing to subtract -> the original
        formula, unchanged."""
        true2 = 500.0
        extract = self._spectrum({"2": true2})
        c2 = cascading_channel_fit(
            1, self.group_rows, self.seqs, self.names, self.mzs, self.libs,
            self.masks, self.fracs, {}, extract, self.tag, n_iso=5,
            mz_tol_da=0.015, cache={})
        self.assertAlmostEqual(c2, jmod_coefficient(
            self.libs[1], extract(self.mzs[1]), self.fracs[1],
            mask=self.masks[1]), places=6)


class TestIsotopeCorrectEndToEnd(unittest.TestCase):
    """add_channel_unique_coeff(isotope_correct=True) against a fake scan."""

    class FakeSpec:
        def __init__(self, mz, intens, prec_mz=500.0, scan_num=1, rt=20.0):
            order = np.argsort(mz)
            self.mz = np.asarray(mz)[order]
            self.intens = np.asarray(intens)[order]
            self.prec_mz = prec_mz
            self.scan_num = scan_num
            self.RT = rt

    class FakeSpectra:
        def __init__(self, scan):
            self.ms2scans = [scan]
            self.scan_pos = {scan.scan_num: (2, 0)}

    def setUp(self):
        self.tag = read_json_to_massTag(MASS_TAGS_DIR, "PSMtag_9plex.json")
        self.seq0 = "E(PSMtag_9plex-0)MAAAPPGR"
        self.seq2 = "E(PSMtag_9plex-2)MAAAPPGR"
        self.mz0 = 644.2545167
        self.mz2 = self.mz0 + self.tag.mass_dict["PSMtag_9plex-2"] - \
            self.tag.mass_dict["PSMtag_9plex-0"]
        _, self.ri0 = fragment_isotope_pattern(self.seq0, "b3_1", self.tag, n_iso=5)

    def _frame(self):
        rows = []
        for lab, seq, mz in (("0", self.seq0, self.mz0), ("2", self.seq2, self.mz2)):
            rows.append({
                "untag_prec": "EMAAAPPGR_1", "time_channel": 0.0,
                "channel": f"0_{lab}", "window_mz": 500.0, "rt": 20.0,
                "BestChannel_Qvalue": 0.001, "is_decoy": False,
                "frag_names": ["b3_1"], "frag_mz": [mz], "frag_int": [1.0],
                "frac_lib_int": 1.0, "seq": seq, "spec_id": 1,
            })
        return pd.DataFrame(rows)

    def test_isotope_correction_recovers_the_contaminated_channel(self):
        true0, true2 = 1000.0, 500.0
        contamination = true0 * self.ri0[2]
        scan = self.FakeSpec([self.mz0, self.mz2],
                             [true0, true2 + contamination])
        spectra = self.FakeSpectra(scan)

        raw = add_channel_unique_coeff(self._frame(), spectra=spectra,
                                       plexDIA=True, q_gate=0.01)
        corrected = add_channel_unique_coeff(
            self._frame(), spectra=spectra, plexDIA=True, q_gate=0.01,
            tag=self.tag, isotope_correct=True)

        raw_ch2 = raw.loc[raw["channel"] == "0_2", QUANT_COL].iloc[0]
        corr_ch2 = corrected.loc[corrected["channel"] == "0_2", QUANT_COL].iloc[0]

        # Uncorrected inherits the full contamination; corrected recovers
        # the true value net of channel 0's isotope tail.
        self.assertAlmostEqual(raw_ch2, true2 + contamination, places=2)
        self.assertAlmostEqual(corr_ch2, true2, places=2)

    def test_isotope_correct_without_a_tag_falls_back_quietly(self):
        scan = self.FakeSpec([self.mz0, self.mz2], [1000.0, 500.0])
        spectra = self.FakeSpectra(scan)
        out = add_channel_unique_coeff(self._frame(), spectra=spectra,
                                       plexDIA=True, q_gate=0.01,
                                       isotope_correct=True, tag=None)
        self.assertIn(QUANT_COL, out.columns)

    def test_isotope_correction_is_unconditional(self):
        """Not a flag any more.  Single-variable production A/B on JD0588
        (baseline_run vs area_run, one flag apart) gives sample-level mean
        |error| 0.185 -> 0.134 and slope 0.841 -> 0.902.  No configuration was
        found where turning it off was better, so it is hardcoded and the
        worker defaults match production."""
        from src.default_dict import default_dict
        self.assertNotIn("ms2_channel_unique_isotope_correct", default_dict)
        with open("src/fdr_analysis.py") as fh:
            self.assertIn("isotope_correct=True,", fh.read())
        import inspect
        from src.ms2_common_apex import add_common_apex_coeff
        self.assertIs(
            inspect.signature(add_common_apex_coeff)
            .parameters["isotope_correct"].default, True)

    def test_unparseable_fragment_key_falls_back_instead_of_crashing(self):
        """Real ungated candidate rows include fragment keys this module's
        parser was never exercised against (a regression: split_frag_name
        choked on one and crashed a full search). One bad key must not take
        down the whole row -- it should fail open to the raw extraction."""
        # Both rows share the SAME (unparseable) key at their own distinct
        # m/z, so channel_unique_masks still calls it channel-unique and the
        # correction finds >=2 siblings for it -- reaching, and needing to
        # survive, the parse failure.
        df = self._frame()
        df["frag_names"] = df["frag_names"].apply(
            lambda _: ["not_a_valid_fragment_key_at_all"])
        scan = self.FakeSpec([self.mz0, self.mz2], [1000.0, 500.0])
        spectra = self.FakeSpectra(scan)
        out = add_channel_unique_coeff(df, spectra=spectra, plexDIA=True,
                                       q_gate=0.01, tag=self.tag,
                                       isotope_correct=True)
        self.assertIn(QUANT_COL, out.columns)
        self.assertTrue(np.isfinite(out[QUANT_COL]).any())


class TestAmplitudeFromCoeff(unittest.TestCase):
    """Recovering the per-library-unit amplitude behind a coefficient.

    Subtracting contamination needs A (obs ~= A*lib), and A is NOT
    coeff*f/total: jmod_coefficient carries an unmatched-library penalty row
    with no physical counterpart. Getting this wrong made the isotope
    correction subtract a small fraction of the real contamination at every
    frac_lib_int below 1, which is nearly every real precursor.
    """

    def setUp(self):
        self.lib = np.array([1.0, 0.8, 0.6, 0.5, 0.9, 0.7, 0.4, 0.3])
        self.mask = np.array([True] * 4 + [False] * 4)

    def test_round_trips_at_every_frac_lib_int(self):
        for f in (1.0, 0.9, 0.74, 0.6, 0.5, 0.35, 0.2):
            for amp in (100.0, 1000.0, 54321.0):
                c = jmod_coefficient(self.lib, amp * self.lib, f,
                                     mask=self.mask)
                got = amplitude_from_coeff(c, self.lib, self.mask, f)
                self.assertAlmostEqual(got / amp, 1.0, places=9,
                                       msg=f"frac_lib_int={f}, amp={amp}")

    def test_differs_from_the_naive_ratio_whenever_frac_below_one(self):
        """Guards the actual bug: coeff*f/total silently under-reports."""
        amp, f = 1000.0, 0.5
        c = jmod_coefficient(self.lib, amp * self.lib, f, mask=self.mask)
        naive = c * f / self.lib.sum()
        self.assertLess(naive, 0.25 * amp)          # was ~13x too small
        self.assertAlmostEqual(
            amplitude_from_coeff(c, self.lib, self.mask, f) / amp, 1.0,
            places=9)

    def test_degenerate_inputs_give_nan_not_a_wrong_number(self):
        for bad in (np.nan, np.inf):
            self.assertTrue(np.isnan(amplitude_from_coeff(
                bad, self.lib, self.mask, 0.5)))
        self.assertTrue(np.isnan(amplitude_from_coeff(
            1.0, np.zeros(4), np.ones(4, bool), 0.5)))


class TestOptionalMs2FlagsAreActuallyWired(unittest.TestCase):
    """Every optional MS2 flag must reach the worker that implements it.

    Three separate bugs in this family shipped silently: 'seq' missing from
    ms2_prescoring.SCALARS, then 'mz'/'z' missing from it, then
    ms2_fragment_purity_diag never passed from fdr_analysis. In all three the
    run completed with no error and simply produced nothing -- the failure mode
    is invisible at runtime, so it has to be caught here.
    """

    def _src(self, obj):
        import inspect
        return inspect.getsource(obj)

    def test_prescoring_slim_frame_carries_what_the_workers_need(self):
        from src.ms2_prescoring import SCALARS
        # seq: per-channel tag composition for the isotope envelope
        # mz/z: precursor m/z + charge for the isolation-window truncation
        for col in ("seq", "mz", "z", "frac_lib_int", "window_mz", "spec_id"):
            self.assertIn(col, SCALARS, msg=f"{col} would be dropped silently")

    def test_isotope_and_purity_flags_reach_add_prescoring_ms2(self):
        import src.fdr_analysis as fdr
        src = self._src(fdr)
        self.assertIn("add_prescoring_ms2", src)
        for kw in ("isotope_correct=", "n_iso=", "purity_diag=", "tag=",
                   "min_siblings=", "weighted_apex=",
                   "n_adjacent="):
            self.assertIn(kw, src, msg=f"{kw} never passed from fdr_analysis")

    def test_prescoring_forwards_them_to_the_workers(self):
        import src.ms2_prescoring as pre
        src = self._src(pre.add_prescoring_ms2)
        for kw in ("isotope_correct=isotope_correct", "purity_diag=purity_diag",
                   "tag=tag",
                   "min_siblings=min_siblings",
                   "weighted_apex=weighted_apex", "n_adjacent=n_adjacent"):
            self.assertIn(kw, src, msg=f"{kw} not forwarded")
        # the areas need the correction too, or they disagree with the
        # single-scan value they are supposed to be the area of
        self.assertIn("add_common_apex_coeff", src)
        # and every shared flag must reach BOTH workers, not just one --
        for kw in ("isotope_correct=isotope_correct",
                   "min_siblings=min_siblings",):
            self.assertGreaterEqual(
                src.count(kw), 2,
                msg=f"{kw} reaches only one of the two MS2 workers")

    def test_purity_diagnostic_columns_are_reported_but_never_scored(self):
        from src.ms2_xchannel_lib import PURITY_DIAG, PURITY_SWEEP
        import src.fdr_analysis as fdr
        drop = self._src(fdr.score_precursors)
        for c in tuple(PURITY_DIAG) + tuple(PURITY_SWEEP):
            self.assertIn(c, drop,
                          msg=f"{c} would become a scoring feature and change IDs")


class TestConsolidatedDefaults(unittest.TestCase):
    """The shipped default is the best-VALIDATED configuration, not the most
    featureful one: amplitude fix + window truncation + area correction, with
    the ladder off (never demonstrated a benefit, worst variant in sparse
    groups) and no sibling guard unless asked for."""

    def test_purity_weighting_is_not_wired_in(self):
        """Purity weighting was never validated out-of-sample: kappa=4 came
        from a single-dataset sweep on JD0588 chosen on sd rather than accuracy
        (kappa=1 had the better mean error), and was never tested on JD1187.
        It must not ship as a default, so nothing may pass purity_kappa."""
        from src.default_dict import default_dict
        self.assertNotIn("ms2_fragment_purity_weight", default_dict)
        for mod in ("src/fdr_analysis.py", "src/ms2_prescoring.py"):
            with open(mod) as fh:
                self.assertNotIn("purity_kappa", fh.read(), mod)

    def test_area_adjacent_scans_is_settable_and_defaults_to_one(self):
        """+/-1 scan measured best on JD1187 (median per-precursor error
        1.206 -> 1.122, slope 1.042 -> 1.027) while 0 is better on JD0588's
        18 Th windows, where every extra scan hurts both metrics. No single
        value serves both geometries, so it must be a flag."""
        from src.default_dict import default_dict
        e = default_dict["ms2_area_adjacent_scans"]
        self.assertEqual(int(e["default"]), 1)
        self.assertTrue(e["takes_value"])
        with open("src/fdr_analysis.py") as fh:
            self.assertIn("ms2_area_adjacent_scans", fh.read())


    def test_min_siblings_backs_off_entirely(self):
        tag = read_json_to_massTag(MASS_TAGS_DIR, "PSMtag_9plex.json")
        seqs = {0: "E(PSMtag_9plex-0)MAAAPPGR", 1: "E(PSMtag_9plex-2)MAAAPPGR"}
        mz0 = 644.2545167
        mzs = {0: np.array([mz0]),
               1: np.array([mz0 + tag.mass_dict["PSMtag_9plex-2"]
                            - tag.mass_dict["PSMtag_9plex-0"]])}
        names = {0: np.array(["b3_1"]), 1: np.array(["b3_1"])}
        libs = {0: np.array([1.0]), 1: np.array([1.0])}
        masks = {0: np.array([True]), 1: np.array([True])}
        fracs = {0: 1.0, 1: 1.0}
        extract = lambda a: np.array([500.0] * len(a))  # noqa: E731
        plain = jmod_coefficient(libs[1], extract(mzs[1]), fracs[1],
                                 mask=masks[1])
        # group of 2 < threshold 7 -> no correction attempted at all
        got = cascading_channel_fit(
            1, [0, 1], seqs, names, mzs, libs, masks, fracs, {0: 4000.0},
            extract, tag, n_iso=5, mz_tol_da=0.015, cache={},
            min_siblings=7)
        self.assertAlmostEqual(got, plain, places=6)
        # ... and with the guard lifted it DOES correct, so the guard is what
        # made the difference rather than something else being broken
        got2 = cascading_channel_fit(
            1, [0, 1], seqs, names, mzs, libs, masks, fracs, {0: 4000.0},
            extract, tag, n_iso=5, mz_tol_da=0.015, cache={}, min_siblings=0)
        self.assertLess(got2, plain)


class TestWeightedCoeffUsesReferenceImpl(unittest.TestCase):
    """jmod_coefficient is rebound to a fast variant inside
    add_xchannel_features, and that variant takes no ``weights``.  Any weighted
    call must therefore use the reference implementation -- this shipped as a
    TypeError that killed a whole run."""

    def test_fast_path_is_not_called_with_weights(self):
        import inspect
        import src.ms2_xchannel_lib as xch
        src = inspect.getsource(xch.add_xchannel_features)
        for line in src.splitlines():
            if "weights=" in line:
                self.assertNotIn("jmod_coefficient(", line,
                                 msg="weighted call routed to the fast path")

    def test_reference_impl_accepts_weights(self):
        import inspect
        from src.ms2_unique_quant import jmod_coefficient
        self.assertIn("weights",
                      inspect.signature(jmod_coefficient).parameters)


class TestIsolatedEnvelopeFraction(unittest.TestCase):
    """How much of a contaminating channel's reachable material the isolation
    window admits.  BOTH window edges matter, and the LOWER one is what broke
    K-terminating peptides in narrow windows."""

    FRAG_MZ, FRAG_Z, PREC_Z = 700.0, 1, 2
    STEP = 2.0067096756          # one tag step
    C13 = 1.0033548

    def _f(self, prec_mz, lo, hi, **kw):
        return isolated_envelope_fraction(prec_mz, self.PREC_Z, lo, hi,
                                          self.FRAG_MZ, self.FRAG_Z,
                                          cache={}, **kw)

    def test_wide_window_admits_everything(self):
        p = 600.0
        for hi in (p + 2.6, p + 8.75, p + 17.5):
            self.assertGreater(self._f(p, p - 9.0, hi, order=2), 0.999)

    def test_narrow_window_truncates_from_above_and_is_monotone(self):
        p = 600.0
        vals = [self._f(p, p - 1.0, p + d, order=0) for d in (2.0, 1.0, 0.05)]
        self.assertTrue(all(0.0 <= v <= 1.0 for v in vals))
        self.assertGreater(vals[0], vals[1])
        self.assertGreater(vals[1], vals[2])

    def test_one_tag_peptide_is_barely_truncated(self):
        """R-terminating: channels are ONE tag step apart, and that step is
        2 x 13C, so the contaminator's q=2 isotopologue lands exactly on the
        target's own precursor -- inside the window by construction."""
        z, step = self.PREC_Z, self.STEP
        pt = 500.0 + step / z          # target, one tag step up
        v = self._f(pt, pt - 1.0, pt + 1.0, prec_mz_contaminator=500.0, order=2)
        self.assertGreater(v, 0.85)

    def test_two_tag_peptide_is_heavily_truncated(self):
        """K-terminating: channels are TWO tag steps apart, so the
        contaminator's dominant q=2 isotopologue falls BELOW the window and is
        never fragmented.  Assuming full admission over-subtracts several-fold
        -- the reason K-terminating peptides got worse under the correction in
        2.0 m/z windows while improving in 18 m/z ones."""
        z, step = self.PREC_Z, self.STEP
        pt = 500.0 + 2 * step / z      # target, TWO tag steps up
        v = self._f(pt, pt - 1.0, pt + 1.0, prec_mz_contaminator=500.0, order=2)
        self.assertLess(v, 0.45)
        # and a wide window puts it back
        w = self._f(pt, pt - 9.0, pt + 9.0, prec_mz_contaminator=500.0, order=2)
        self.assertGreater(w, 0.95)

    def test_missing_inputs_fall_back_to_no_truncation(self):
        p = 600.0
        for kw in (dict(prec_mz=None), dict(prec_z=None), dict(hi=None),
                   dict(prec_mz=np.nan), dict(hi=np.nan), dict(prec_z=0)):
            base = dict(prec_mz=p, prec_z=self.PREC_Z, lo=p - 9.0, hi=p + 9.0)
            base.update(kw)
            v = isolated_envelope_fraction(base["prec_mz"], base["prec_z"],
                                           base["lo"], base["hi"],
                                           self.FRAG_MZ, self.FRAG_Z, cache={})
            self.assertEqual(v, 1.0, msg=f"{kw}")

    def test_result_is_always_a_valid_fraction(self):
        p = 600.0
        for hi in np.linspace(p - 1.0, p + 25.0, 60):
            v = self._f(p, p - 1.0, float(hi), order=2)
            self.assertTrue(0.0 <= v <= 1.0, msg=f"hi={hi} -> {v}")


class TestCascadeUnderRealisticFracLibInt(unittest.TestCase):
    """The full 9-plex cascade must recover truth at real frac_lib_int.

    The earlier tests all used frac_lib_int=1.0 with an all-True mask, which
    is exactly the one configuration where the amplitude bug cancels -- so
    they passed while production subtracted ~2% of the contamination.
    """

    LABELS = [0, 2, 4, 6, 8, 10, 12, 14, 16]
    TRUTH = {0: 1.0, 2: 4.0, 4: 0.5, 6: 4.0, 8: 1.0,
             10: 0.5, 12: 0.5, 14: 1.0, 16: 4.0}      # JD1187 A/B/C design
    FRAGS = ["b3_1", "b4_1", "b5_1", "b6_1", "b7_1", "b8_1"]
    LIB = np.array([1.0, 0.8, 0.6, 0.5, 0.9, 0.7])
    MASK = np.array([True] * 4 + [False] * 2)
    MZ0 = {"b3_1": 644.2545167, "b4_1": 715.2916, "b5_1": 786.3287,
           "b6_1": 857.3658, "b7_1": 928.4029, "b8_1": 999.4400}

    def _run(self, frac):
        tag = read_json_to_massTag(MASS_TAGS_DIR, "PSMtag_9plex.json")
        m0 = tag.mass_dict["PSMtag_9plex-0"]
        seqs, mzs, names, libs, masks, fracs = {}, {}, {}, {}, {}, {}
        for r, c in enumerate(self.LABELS):
            seqs[r] = f"E(PSMtag_9plex-{c})MAAAPPGR"
            sh = tag.mass_dict[f"PSMtag_9plex-{c}"] - m0
            mzs[r] = np.array([self.MZ0[f] + sh for f in self.FRAGS])
            names[r] = np.array(self.FRAGS)
            libs[r] = self.LIB.copy()
            masks[r] = self.MASK.copy()
            fracs[r] = frac
        peaks = []
        for r, c in enumerate(self.LABELS):
            for k, f in enumerate(self.FRAGS):
                off, rel = fragment_isotope_pattern(seqs[r], f, tag, n_iso=5)
                for o in range(len(off)):
                    peaks.append((mzs[r][k] + off[o],
                                  self.TRUTH[c] * 1000.0 * self.LIB[k] * rel[o]))
        pmz = np.array([p[0] for p in peaks])
        pin = np.array([p[1] for p in peaks])
        extract = lambda a: np.array(  # noqa: E731
            [pin[np.abs(pmz - m) <= 0.01].sum() for m in a])

        known, errs, raw_errs = {}, [], []
        for r, c in enumerate(self.LABELS):
            got = cascading_channel_fit(
                r, list(range(9)), seqs, names, mzs, libs, masks, fracs,
                known, extract, tag, n_iso=5, mz_tol_da=0.015, cache={})
            known[r] = got
            ref = jmod_coefficient(libs[r], self.TRUTH[c] * 1000.0 * self.LIB,
                                   fracs[r], mask=masks[r])
            raw = jmod_coefficient(libs[r], extract(mzs[r]), fracs[r],
                                   mask=masks[r])
            errs.append(abs(np.log2(got / ref)))
            raw_errs.append(abs(np.log2(raw / ref)))
        return max(errs), max(raw_errs)

    def test_recovers_truth_at_realistic_frac_lib_int(self):
        for frac in (1.0, 0.74, 0.5, 0.35):
            corrected, uncorrected = self._run(frac)
            # the artefact is real and large before correction ...
            self.assertGreater(uncorrected, 1.0, msg=f"frac={frac}")
            # ... and essentially gone after, at EVERY frac_lib_int
            self.assertLess(corrected, 0.02, msg=f"frac={frac}")


if __name__ == "__main__":
    unittest.main()


class TestLadderRemoved(unittest.TestCase):
    """The spectrum-estimated 'ladder' rung was removed after measurement.

    With the fitted-row path working (it had been silently crashing on a
    list-vs-dict `masks` argument), ladder-on vs ladder-off left 94% of cells
    bit-identical and every scorable channel equal to three decimals. It bought
    nothing for an extra code path, an extra flag, and a silent-failure surface
    -- its flag never even reached the code, because argparse named the
    attribute after `--no_ms2_channel_unique_ladder` while the reader asked for
    `ms2_channel_unique_ladder`."""

    def test_no_ladder_anywhere(self):
        import glob
        for f in glob.glob("src/*.py"):
            with open(f) as fh:
                txt = fh.read()
            self.assertNotIn("use_ladder", txt, f)
            self.assertNotIn("ms2_channel_unique_ladder", txt, f)

    def test_absent_contaminator_is_simply_skipped(self):
        """No fitted row for a rung -> it contributes nothing, rather than
        being estimated off the spectrum."""
        from src.ms2_isotope_xtalk import cascading_channel_fit
        import inspect
        self.assertNotIn("use_ladder",
                         inspect.signature(cascading_channel_fit).parameters)
