"""
Tests for functions in preliminary_search.py
"""

#  Copyright (c) 2026 Parallel Squared Technology Institute
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#          http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import os
import numpy as np
import pytest
import polars as pl

from src.utils.io.load_files import SpectrumFile
from src.models.spec_lib.spec_lib import loadSpecLib, create_decoy_lib
from src.models.spec_lib.library_store import SpectrumLibraryStore
from src.preliminary_search import (
    _most_intense_ms1_peak,
    create_fragment_library_map,
    fit_with_features,
    hellinger_score_polars_udf,
    library_entries_for,
    lookup_ms1_data,
    peptide_to_mod_array,
    scribe_score_polars_udf,
)


# Path to test data files
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
TEST_MZML = os.path.join(DATA_DIR, 'test_mode_filtered.mzML')
TEST_LIBRARY = os.path.join(DATA_DIR, 'filtered_library.tsv')


@pytest.fixture(scope="module")
def dia_spectra():
    """Load DIA spectra from test mzML file."""
    return SpectrumFile(TEST_MZML)


@pytest.fixture(scope="module")
def library_spectra():
    """Load library spectra from test TSV file."""
    return loadSpecLib(TEST_LIBRARY)[0]


class _Scan:
    def __init__(self, mz, intens, mobility=None, scan_num=1, im_lo=None, im_hi=None):
        self.mz = np.asarray(mz, float)
        self.intens = np.asarray(intens, float)
        self.mobility = None if mobility is None else np.asarray(mobility, float)
        self.scan_num = scan_num
        self.im_lo, self.im_hi = im_lo, im_hi


class TestMostIntenseMs1Peak:
    """The precursor peak is the most intense within the ppm tolerance (and the
    MS2 spectrum's mobility band), not the nearest in m/z."""
    # 500 +/- 20 ppm is 499.99..500.01
    mz = [499.985, 499.995, 500.0002, 500.004, 500.02]
    intens = [1e6, 50.0, 10.0, 300.0, 1e6]

    def test_most_intense_within_tolerance_not_nearest(self):
        assert _most_intense_ms1_peak(_Scan(self.mz, self.intens), 500.0, 20) == 3

    def test_none_within_tolerance(self):
        assert _most_intense_ms1_peak(_Scan(self.mz, self.intens), 600.0, 20) is None

    def test_band_excludes_ions_at_other_mobilities(self):
        scan = _Scan(self.mz, self.intens, mobility=[1.0, 1.0, 1.0, 0.7, 1.0])
        assert _most_intense_ms1_peak(scan, 500.0, 20, 0.95, 1.05) == 1
        assert _most_intense_ms1_peak(scan, 500.0, 20, 0.60, 0.65) is None

    def test_no_band_or_no_mobility_ignores_mobility(self):
        scan = _Scan(self.mz, self.intens, mobility=[1.0, 1.0, 1.0, 0.7, 1.0])
        assert _most_intense_ms1_peak(scan, 500.0, 20) == 3
        assert _most_intense_ms1_peak(_Scan(self.mz, self.intens), 500.0, 20, 0.95, 1.05) == 3

    def test_ties_go_to_the_lowest_mz(self):
        assert _most_intense_ms1_peak(_Scan([499.999, 500.001], [5.0, 5.0]), 500.0, 20) == 0


class TestLookupMs1Data:
    class _Spectra:
        def __init__(self, ms1, ms2):
            self.ms1, self.ms2scans, self.ms2_by_id = ms1, [ms2], {7: 0}

        def get_nearest_ms1_for_scan(self, spec_id):
            return self.ms1

    def test_reports_the_band_gated_most_intense_peak(self):
        ms1 = _Scan([499.995, 500.004], [50.0, 300.0], mobility=[1.0, 0.7], scan_num=3)
        ms2 = _Scan([], [], im_lo=0.95, im_hi=1.05)
        out = lookup_ms1_data(pl.DataFrame({"spec_id": [7], "mz": [500.0]}),
                              self._Spectra(ms1, ms2), 20)
        assert out.to_dicts() == [{"Ms1_spec_id": 3, "closest_peak_mz_ms1": 499.995,
                                   "closest_peak_intensity_ms1": 50.0, "precursor_mobility": 1.0}]

    def test_native_id_spec_ids(self):
        """mzML PSMs carry the native ID string, not the scan number."""
        ms1 = _Scan([500.001], [50.0], scan_num=3)
        out = lookup_ms1_data(
            pl.DataFrame({"spec_id": ["controllerType=0 controllerNumber=1 scan=7"], "mz": [500.0]}),
            self._Spectra(ms1, _Scan([], [])), 20)
        assert out["closest_peak_mz_ms1"][0] == 500.001

    def test_no_peak_in_window_gives_the_placeholder(self):
        ms1 = _Scan([500.5], [50.0], mobility=[1.0], scan_num=3)
        ms2 = _Scan([], [], im_lo=0.95, im_hi=1.05)
        out = lookup_ms1_data(pl.DataFrame({"spec_id": [7], "mz": [500.0]}),
                              self._Spectra(ms1, ms2), 20)
        assert out.to_dicts() == [{"Ms1_spec_id": 3, "closest_peak_mz_ms1": 0.0,
                                   "closest_peak_intensity_ms1": 0.0, "precursor_mobility": None}]

    def test_no_ms1_scan_gives_the_placeholder_without_a_scan(self):
        spectra = self._Spectra(None, _Scan([], []))
        out = lookup_ms1_data(pl.DataFrame({"spec_id": [7], "mz": [500.0]}), spectra, 20)
        assert out.to_dicts() == [{"Ms1_spec_id": None, "closest_peak_mz_ms1": 0.0,
                                   "closest_peak_intensity_ms1": 0.0, "precursor_mobility": None}]
        assert out.schema == {"Ms1_spec_id": pl.Int64, "closest_peak_mz_ms1": pl.Float64,
                              "closest_peak_intensity_ms1": pl.Float64,
                              "precursor_mobility": pl.Float64}

    def test_matches_the_per_psm_lookup(self):
        """Many PSMs over several float32 MS1 scans, bands and ties: the same
        peak as _most_intense_ms1_peak for every row, in row order."""
        from src.utils.io.load_files import PEAK_MZ_DTYPE, PEAK_INT_DTYPE, PEAK_MOB_DTYPE
        rng = np.random.default_rng(0)

        class Spectra:
            pass

        sp = Spectra()
        ms1 = []
        for k in range(4):
            mz = np.sort(rng.uniform(499.9, 500.1, 400)).astype(PEAK_MZ_DTYPE)
            intens = rng.integers(1, 20, 400).astype(PEAK_INT_DTYPE)       # many ties
            mob = rng.uniform(0.6, 1.4, 400).astype(PEAK_MOB_DTYPE)
            ms1.append(_Scan(mz, intens, mobility=mob, scan_num=100 + k))
            ms1[-1].mz, ms1[-1].intens, ms1[-1].mobility = mz, intens, mob
        sp.ms2scans, sp.ms2_by_id, nearest = [], {}, {}
        for s_num in range(1, 41):
            lo = rng.uniform(0.6, 1.2)
            band = (lo, lo + 0.1) if s_num % 3 else (None, None)
            sp.ms2_by_id[s_num] = len(sp.ms2scans)
            sp.ms2scans.append(_Scan([], [], im_lo=band[0], im_hi=band[1]))
            nearest[s_num] = None if s_num == 40 else ms1[s_num % 4]
        sp.get_nearest_ms1_for_scan = lambda sid: nearest[sid]

        spec_ids = rng.integers(1, 41, 3000)
        theo = rng.uniform(499.95, 500.05, 3000)
        out = lookup_ms1_data(pl.DataFrame({"spec_id": spec_ids, "mz": theo}), sp, 20)

        for row, (sid, t) in enumerate(zip(spec_ids.tolist(), theo.tolist())):
            scan = nearest[sid]
            got = out.row(row, named=True)
            if scan is None:
                assert got["Ms1_spec_id"] is None and got["closest_peak_mz_ms1"] == 0.0
                continue
            ms2 = sp.ms2scans[sp.ms2_by_id[sid]]
            idx = _most_intense_ms1_peak(scan, t, 20, ms2.im_lo, ms2.im_hi)
            assert got["Ms1_spec_id"] == scan.scan_num
            if idx is None:
                assert got["closest_peak_mz_ms1"] == 0.0 and got["precursor_mobility"] is None
            else:
                assert got["closest_peak_mz_ms1"] == float(scan.mz[idx])
                assert got["closest_peak_intensity_ms1"] == float(scan.intens[idx])
                assert got["precursor_mobility"] == float(scan.mobility[idx])


class TestPeptideToModArray:
    """Slot 0 is the N-terminus, slots 1..n the residues, the last the C-terminus."""
    masses = {"UniMod:4": 57.021464, "tag": 300.0}

    def test_mods_in_front_of_the_first_residue_go_in_the_N_terminal_slot(self):
        assert peptide_to_mod_array("(tag)PEK(tag)", self.masses) == [300.0, 0.0, 0.0, 300.0, 0.0]

    def test_a_first_residue_keeps_its_own_mods(self):
        assert peptide_to_mod_array("(tag)K(tag)EK", self.masses) == [300.0, 300.0, 0.0, 0.0, 0.0]
        assert peptide_to_mod_array("C(UniMod:4)EK", self.masses) == [0.0, 57.021464, 0.0, 0.0, 0.0]


class TestFitWithFeatures:
    def test_fit_with_features_returns_dataframe(self, dia_spectra, library_spectra):
        """Test that fit_with_features returns a Polars DataFrame."""
        result = fit_with_features(
            dia_spectra=dia_spectra,
            library_spectra=library_spectra,
            mass_tag=None,
            SILAC=None,
            ms1_ppm_error=20,
            ms2_ppm_error=10
        )

        assert isinstance(result, pl.DataFrame)

    def test_fit_with_features_has_expected_columns(self, dia_spectra, library_spectra):
        """Test that the result DataFrame contains expected columns."""
        result = fit_with_features(
            dia_spectra=dia_spectra,
            library_spectra=library_spectra,
            mass_tag=None,
            SILAC=None,
            ms1_ppm_error=20,
            ms2_ppm_error=10
        )

        expected_columns = [
            'seq',
            'z',
            'spectral_contrast_angle',
            'scribe_score',
            'hellinger_score',
            'matched_lib_pct',
            'ppm_error_ms1',
            'lib_rt'
        ]

        for col in expected_columns:
            assert col in result.columns, f"Expected column '{col}' not found in result"

    def test_fit_with_features_non_empty_results(self, dia_spectra, library_spectra):
        """Test that fit_with_features returns non-empty results."""
        result = fit_with_features(
            dia_spectra=dia_spectra,
            library_spectra=library_spectra,
            mass_tag=None,
            SILAC=None,
            ms1_ppm_error=20,
            ms2_ppm_error=10
        )

        assert len(result) > 0, "Expected non-empty results"

    def test_fit_with_features_spectral_scores_in_range(self, dia_spectra, library_spectra):
        """Test that spectral contrast angles are in valid range [0, 1]."""
        result = fit_with_features(
            dia_spectra=dia_spectra,
            library_spectra=library_spectra,
            mass_tag=None,
            SILAC=None,
            ms1_ppm_error=20,
            ms2_ppm_error=10
        )

        spectral_angles = result['spectral_contrast_angle'].to_numpy()
        assert all(0 <= x <= 1 for x in spectral_angles), \
            "Spectral contrast angles should be between 0 and 1"

    def test_fit_with_features_matched_lib_pct_in_range(self, dia_spectra, library_spectra):
        """Test that matched library percentage is in valid range [0, 100] or -999 for errors."""
        result = fit_with_features(
            dia_spectra=dia_spectra,
            library_spectra=library_spectra,
            mass_tag=None,
            SILAC=None,
            ms1_ppm_error=20,
            ms2_ppm_error=10
        )

        matched_pct = result['matched_lib_pct'].to_numpy()
        valid_values = all((0 <= x <= 100) or (x == -999.0) for x in matched_pct)
        assert valid_values, \
            "Matched library percentage should be between 0 and 100, or -999 for errors"


def _keys(entries):
    return [(v["mod_seq"], v["prec_z"]) for v in entries]


def _small_library(keys):
    """A store with one entry per (mod_seq, charge) in *keys*, in that order."""
    from src.utils.misc_functions import frag_to_peak
    frags = {'y3_1': [350.2, 1.0], 'b2_1': [200.1, 0.5]}
    spectrum, ordered_frags = frag_to_peak(frags, return_frags=True)
    return SpectrumLibraryStore.from_dict({
        (seq, z): {'mod_seq': seq, 'seq': seq, 'prec_mz': 400.0 + i, 'prec_z': z, 'iRT': 10.0 + i,
                   'frags': frags, 'spectrum': spectrum, 'ordered_frags': ordered_frags}
        for i, (seq, z) in enumerate(keys)})


class TestLibraryEntriesFor:
    """The entries fit_with_features reads for the peptides its search hit.  Dicts built
    from them must hold what dicts built from the whole library do for those peptides."""

    def test_every_charge_of_each_hit_in_library_order(self):
        # ACD at two charges, with another peptide between them
        library = _small_library([("ACD", 2.0), ("EFGH", 3.0), ("ACD", 3.0), ("KLM", 2.0)])
        entries = library_entries_for(library, ["KLM", "ACD"])
        assert _keys(entries) == [("ACD", 2.0), ("ACD", 3.0), ("KLM", 2.0)]
        assert _keys(entries) == _keys(v for v in library.values() if v["mod_seq"] in ("KLM", "ACD"))

    def test_a_target_view_leaves_out_decoys(self, library_spectra):
        combined = create_decoy_lib(library_spectra, rules="rev")
        decoy_seq = combined.mod_seq[combined.n_targets]
        target_seq = combined.mod_seq[0]
        entries = library_entries_for(combined.target_view(), [decoy_seq, target_seq])
        assert {seq for seq, _ in _keys(entries)} == {target_seq}

    def test_peptides_not_in_the_library_are_ignored(self, library_spectra):
        assert library_entries_for(library_spectra, ["NOTAPEPTIDE"]) == []


LIB_KEY = ("PEPTIDEK", 2)


def _lib(intensities, losses=None):
    """A library map of y-ions, charge 1, ordinals 1..n.

    Keys are (ion, ordinal, charge, loss); loss defaults to 0 (no loss).
    """
    losses = losses if losses is not None else [0] * len(intensities)
    return {LIB_KEY: {("Y", i + 1, 1, losses[i]): v
                      for i, v in enumerate(intensities)}}


def _row(intensities, ordinals=None, losses=None):
    """An observed row shaped like the polars struct the UDFs receive."""
    n = len(intensities)
    return {
        "seq": LIB_KEY[0],
        "z": LIB_KEY[1],
        "frag_charges": [1] * n,
        "frag_kinds": ["y"] * n,
        "frag_fragment_ordinals": ordinals or list(range(1, n + 1)),
        "frag_intensities": list(intensities),
        "frag_losses": list(losses) if losses is not None else [0] * n,
    }


class TestLossIsPartOfFragmentIdentity:
    """A neutral loss makes a different ion: y7 and y7-H2O sit ~18 Da apart, so
    they must not align against each other. peppy_sage indexes them separately;
    these pin the Python side to the same rule."""

    def test_loss_variants_do_not_collide_in_the_library_map(self):
        entry = {
            'mod_seq': LIB_KEY[0], 'prec_z': LIB_KEY[1],
            'frags': {'y7_1': (800.0, 100.0), 'y7-H2O_1': (782.0, 30.0)},
        }
        peptide_map = create_fragment_library_map([entry])[LIB_KEY]
        # One key each, not one key overwritten by the other.
        assert peptide_map == {("Y", 7, 1, 0): 100.0, ("Y", 7, 1, 1): 30.0}

    def test_observed_loss_ion_does_not_match_the_backbone_ion(self):
        # Library holds only y1 (no loss); the observed y1 carries an H2O loss.
        # They are different ions, so this must not score as a match.
        lib = _lib([100.0])
        same = scribe_score_polars_udf(_row([100.0]), lib)
        lossy = scribe_score_polars_udf(_row([100.0], losses=[1]), lib)
        assert same == 25.0
        assert lossy != same


class TestScribeAndHellScores:
    """The two spectral-difference scores. They differ in the order of the sqrt and the
    normalization, and in whether the sum covers the full library or matched ions only.
    Both differences are silent, so they are pinned here."""

    def test_order_of_operations_differs(self):
        # sum(sqrt(I)) != sqrt(sum(I)) for unequal intensities, so normalize-then-root
        # and root-then-normalize give different vectors.
        lib, row = _lib([100.0, 50.0, 10.0]), _row([90.0, 60.0, 20.0])
        assert scribe_score_polars_udf(row, lib) != pytest.approx(
            hellinger_score_polars_udf(row, lib))

    def test_scribe_matches_equation_1(self):
        lib, row = _lib([100.0, 50.0, 10.0]), _row([90.0, 60.0, 20.0])
        a, b = np.sqrt([90.0, 60.0, 20.0]), np.sqrt([100.0, 50.0, 10.0])
        expected = -np.log(np.sum((a / a.sum() - b / b.sum()) ** 2))
        assert scribe_score_polars_udf(row, lib) == pytest.approx(expected)

    def test_hellinger_score_unchanged_by_rename(self):
        # Pins the incumbent so the rename and refactor cannot move the baseline.
        lib, row = _lib([100.0, 50.0, 10.0]), _row([90.0, 60.0, 20.0])
        a, b = np.array([90.0, 60.0, 20.0]), np.array([100.0, 50.0, 10.0])
        expected = -np.log(
            np.sum((np.sqrt(a / a.sum()) - np.sqrt(b / b.sum())) ** 2))
        assert hellinger_score_polars_udf(row, lib) == pytest.approx(expected)

    def test_support_differs_on_missing_library_ions(self):
        # 4 library ions, 2 observed. Scribe ignores the absent pair; hell charges for
        # them, so hell must score lower.
        lib, row = _lib([100.0, 50.0, 40.0, 30.0]), _row([100.0, 50.0], [1, 2])
        assert scribe_score_polars_udf(row, lib) > hellinger_score_polars_udf(row, lib)

    def test_perfect_match_is_capped(self):
        lib, row = _lib([100.0, 50.0, 10.0]), _row([100.0, 50.0, 10.0])
        assert scribe_score_polars_udf(row, lib) == 25.0
        assert hellinger_score_polars_udf(row, lib) == 25.0

    def test_sparse_clean_match_scores_at_the_cap(self):
        # DELIBERATE: on a matched-only support, 2 of 20 ions at the library's ratios
        # is a perfect match. Accepted because matched_lib_pct carries coverage
        # separately. Do not add a minimum-ion floor without revisiting that.
        lib = _lib([100.0, 50.0] + [10.0] * 18)
        assert scribe_score_polars_udf(_row([100.0, 50.0], [1, 2]), lib) == 25.0

    def test_no_matched_ions_returns_sentinel(self):
        assert scribe_score_polars_udf(_row([100.0], [7]), _lib([100.0, 50.0])) == -999.0

    def test_missing_library_entry_returns_sentinel(self):
        row = _row([100.0, 50.0])
        assert scribe_score_polars_udf(row, {}) == -999.0
        assert hellinger_score_polars_udf(row, {}) == -999.0

    def test_negative_intensity_does_not_produce_nan(self):
        # Guards the clip: scribe roots before normalizing, so a negative intensity
        # would otherwise yield NaN and pass silently through empirical_fit's filter.
        lib, row = _lib([100.0, 50.0, 10.0]), _row([90.0, -5.0, 20.0])
        assert not np.isnan(scribe_score_polars_udf(row, lib))


class TestScorePsms:
    """The compiled scores match the per-row UDFs (up to summation order)."""

    def test_matches_the_row_udfs(self):
        from functools import partial
        from src.preliminary_search import (score_psms, spectral_angle_polars_udf, hellinger_score_polars_udf,
                                            scribe_score_polars_udf, matched_lib_intensity_pct_udf)
        rng = np.random.default_rng(3)
        fmap = {}
        for p in range(40):
            keys = {(str(rng.choice(["B", "Y"])), int(rng.integers(1, 12)), int(rng.integers(1, 3)), int(rng.choice([0, 0, 1])))
                    for _ in range(rng.integers(1, 15))}
            fmap[(f"PEP{p}K", float(2 + p % 2))] = {k: float(rng.choice([0.0, rng.uniform(0, 1)])) for k in keys}
        rows = []
        for i in range(600):
            p = int(rng.integers(0, 45))                          # 40+ have no library entry
            n = int(rng.integers(0, 12))
            rows.append(dict(seq=f"PEP{p}K", z=2 + p % 2,
                             frag_charges=[int(x) for x in rng.integers(1, 3, n)],
                             frag_kinds=[str(x) for x in rng.choice(["B", "Y", "b", "y", "C"], n)],
                             frag_fragment_ordinals=[int(x) for x in rng.integers(1, 12, n)],
                             frag_intensities=[float(x) for x in rng.choice([0.0, 5.0, 17.5, 300.0], n)],
                             frag_losses=[int(x) for x in rng.choice([0, 0, 1], n)]))
        df = pl.DataFrame(rows, schema={"seq": pl.String, "z": pl.Int32, "frag_charges": pl.List(pl.Int32),
                                        "frag_kinds": pl.List(pl.String), "frag_fragment_ordinals": pl.List(pl.Int32),
                                        "frag_intensities": pl.List(pl.Float32), "frag_losses": pl.List(pl.Int32)})
        got = {s.name: s.to_numpy() for s in score_psms(df, fmap, n_threads=3)}
        cols = ["frag_charges", "frag_kinds", "frag_fragment_ordinals", "frag_intensities", "frag_losses", "seq", "z"]
        for name, udf, tol in (("spectral_contrast_angle", spectral_angle_polars_udf, 1e-5),
                               ("hellinger_score", hellinger_score_polars_udf, 1e-12),
                               ("scribe_score", scribe_score_polars_udf, 1e-12),
                               ("matched_lib_pct", matched_lib_intensity_pct_udf, 0.0)):
            ref = df.select(pl.struct(cols).map_elements(partial(udf, fragment_library_map=fmap),
                                                         return_dtype=pl.Float64)).to_series().to_numpy()
            assert np.array_equal(ref == -999.0, got[name] == -999.0), name
            np.testing.assert_allclose(got[name], ref, rtol=0, atol=tol, err_msg=name)
