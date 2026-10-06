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
    fit_with_features,
    hellinger_score_polars_udf,
    library_entries_for,
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


def _lib(intensities):
    """A library map of y-ions, charge 1, ordinals 1..n."""
    return {LIB_KEY: {("Y", i + 1, 1): v for i, v in enumerate(intensities)}}


def _row(intensities, ordinals=None):
    """An observed row shaped like the polars struct the UDFs receive."""
    n = len(intensities)
    return {
        "seq": LIB_KEY[0],
        "z": LIB_KEY[1],
        "frag_charges": [1] * n,
        "frag_kinds": ["y"] * n,
        "frag_fragment_ordinals": ordinals or list(range(1, n + 1)),
        "frag_intensities": list(intensities),
    }


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
