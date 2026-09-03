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
from src.models.spec_lib.spec_lib import loadSpecLib
from src.preliminary_search import (
    fit_with_features,
    hellinger_score_polars_udf,
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
