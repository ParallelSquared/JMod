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
#  """

import os
import pytest
import polars as pl

from src.utils.io.load_files import SpectrumFile
from src.models.spec_lib.spec_lib import loadSpecLib
from src.preliminary_search import fit_with_features


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
