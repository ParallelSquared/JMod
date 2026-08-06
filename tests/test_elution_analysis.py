"""
Tests for functions in elution_analysis.py
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

import numpy as np
import polars as pl
import pytest

from src.elution_analysis import (
    _gaussian,
    _find_adjacent_clusters,
    _compute_scan_gap_mode,
    calculate_elution_width,
)


class TestGaussian:
    def test_gaussian_peak_at_mu(self):
        """Gaussian should have maximum at mu."""
        x = np.array([0.0])
        result = _gaussian(x, amplitude=1.0, mu=0.0, sigma=1.0)
        assert result[0] == 1.0

    def test_gaussian_symmetric(self):
        """Gaussian should be symmetric around mu."""
        x_left = np.array([-1.0])
        x_right = np.array([1.0])
        result_left = _gaussian(x_left, amplitude=1.0, mu=0.0, sigma=1.0)
        result_right = _gaussian(x_right, amplitude=1.0, mu=0.0, sigma=1.0)
        np.testing.assert_almost_equal(result_left, result_right)

    def test_gaussian_amplitude_scaling(self):
        """Gaussian peak should scale with amplitude."""
        x = np.array([0.0])
        result = _gaussian(x, amplitude=2.0, mu=0.0, sigma=1.0)
        assert result[0] == 2.0

    def test_gaussian_array_input(self):
        """Gaussian should handle array inputs."""
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        result = _gaussian(x, amplitude=1.0, mu=0.0, sigma=1.0)
        assert len(result) == 5
        assert result[2] == 1.0  # Peak at center
        assert result[0] < result[2]  # Tails lower than peak


class TestFindAdjacentClusters:
    def test_empty_input(self):
        """Empty input should return empty list."""
        result = _find_adjacent_clusters([], max_scan_gap=5)
        assert result == []

    def test_single_scan(self):
        """Single scan should return one cluster with one element."""
        result = _find_adjacent_clusters([(100, 10.0)], max_scan_gap=5)
        assert len(result) == 1
        assert len(result[0]) == 1

    def test_adjacent_scans(self):
        """Adjacent scans within gap should be in same cluster."""
        scans_rts = [(100, 10.0), (102, 10.1), (104, 10.2)]
        result = _find_adjacent_clusters(scans_rts, max_scan_gap=5)
        assert len(result) == 1
        assert len(result[0]) == 3

    def test_separate_clusters(self):
        """Scans with large gaps should be in separate clusters."""
        scans_rts = [(100, 10.0), (102, 10.1), (200, 20.0), (202, 20.1)]
        result = _find_adjacent_clusters(scans_rts, max_scan_gap=5)
        assert len(result) == 2
        assert len(result[0]) == 2
        assert len(result[1]) == 2

    def test_unsorted_input(self):
        """Function should handle unsorted input."""
        scans_rts = [(200, 20.0), (100, 10.0), (102, 10.1)]
        result = _find_adjacent_clusters(scans_rts, max_scan_gap=5)
        assert len(result) == 2

    def test_duplicate_scans(self):
        """Duplicate scans should be aggregated."""
        scans_rts = [(100, 10.0), (100, 10.1), (102, 10.2)]
        result = _find_adjacent_clusters(scans_rts, max_scan_gap=5)
        assert len(result) == 1
        # Should have 2 unique scans
        assert len(result[0]) == 2


class TestComputeScanGapMode:
    def test_uniform_gaps(self):
        """Mode should be the consistent gap size."""
        df = pl.DataFrame({
            'file_id': [1, 1, 1, 1, 1],
            'seq': ['A', 'A', 'A', 'A', 'A'],
            'scan': [100, 110, 120, 130, 140],
            'rt': [10.0, 11.0, 12.0, 13.0, 14.0],
        })
        result = _compute_scan_gap_mode(df)
        assert result == 10

    def test_mixed_gaps(self):
        """Mode should be the most common gap."""
        df = pl.DataFrame({
            'file_id': [1, 1, 1, 1, 1, 1],
            'seq': ['A', 'A', 'A', 'A', 'A', 'A'],
            'scan': [100, 110, 120, 130, 140, 155],  # gaps: 10, 10, 10, 10, 15
            'rt': [10.0, 11.0, 12.0, 13.0, 14.0, 15.5],
        })
        result = _compute_scan_gap_mode(df)
        assert result == 10

    def test_single_scan_peptide(self):
        """Single scan peptides should be skipped, return default."""
        df = pl.DataFrame({
            'file_id': [1, 2, 3],
            'seq': ['A', 'B', 'C'],
            'scan': [100, 200, 300],
            'rt': [10.0, 20.0, 30.0],
        })
        result = _compute_scan_gap_mode(df)
        assert result == 25  # Default fallback

    def test_multiple_peptides(self):
        """Mode should consider gaps from all peptides."""
        df = pl.DataFrame({
            'file_id': [1, 1, 1, 1, 1, 1],
            'seq': ['A', 'A', 'A', 'B', 'B', 'B'],
            'scan': [100, 120, 140, 200, 220, 240],  # All gaps are 20
            'rt': [10.0, 12.0, 14.0, 20.0, 22.0, 24.0],
        })
        result = _compute_scan_gap_mode(df)
        assert result == 20


class TestCalculateElutionWidth:
    @pytest.fixture
    def valid_df(self):
        """Create a valid DataFrame for testing."""
        return pl.DataFrame({
            'file_id': [1] * 10,
            'spec_name': [f'scan={100 + i*10}' for i in range(10)],
            'rt': [10.0 + i * 0.1 for i in range(10)],
            'z': [2] * 10,
            'seq': ['PEPTIDE'] * 10,
            'closest_peak_intensity_ms1': [1000.0 + i * 100 for i in range(10)]
        })

    def test_missing_columns(self):
        """Should raise ValueError for missing columns."""
        df = pl.DataFrame({'file_id': [1], 'rt': [10.0]})
        with pytest.raises(ValueError, match="Missing required columns"):
            calculate_elution_width(df)

    def test_returns_tuple(self, valid_df):
        """Should return tuple of (fwhm, sigma, df)."""
        result = calculate_elution_width(valid_df)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_fwhm_positive(self, valid_df):
        """FWHM should be positive."""
        fwhm, sigma, df = calculate_elution_width(valid_df)
        assert fwhm > 0
        assert sigma > 0

    def test_fwhm_sigma_relationship(self, valid_df):
        """FWHM should be approximately 2.355 * sigma."""
        fwhm, sigma, df = calculate_elution_width(valid_df)
        np.testing.assert_almost_equal(fwhm, 2.355 * sigma, decimal=5)

    def test_adds_cluster_size_column(self, valid_df):
        """Should add cluster_size column to output DataFrame."""
        fwhm, sigma, df = calculate_elution_width(valid_df)
        assert 'cluster_size' in df.columns

    def test_cluster_size_values(self, valid_df):
        """Cluster sizes should be positive integers."""
        fwhm, sigma, df = calculate_elution_width(valid_df)
        cluster_sizes = df['cluster_size'].to_list()
        assert all(cs >= 0 for cs in cluster_sizes)

    def test_empty_after_filter(self):
        """Should raise ValueError if no valid PSMs after filtering."""
        df = pl.DataFrame({
            'file_id': [1],
            'spec_name': ['invalid'],  # No scan= pattern
            'rt': [10.0],
            'z': [2],
            'seq': ['PEPTIDE'],
            'closest_peak_intensity_ms1': [1000.0]
        })
        with pytest.raises(ValueError, match="No valid PSMs"):
            calculate_elution_width(df)

    def test_multiple_peptides(self):
        """Should handle multiple different peptides."""
        df = pl.DataFrame({
            'file_id': [1] * 20,
            'spec_name': [f'scan={100 + i*10}' for i in range(10)] +
                        [f'scan={500 + i*10}' for i in range(10)],
            'rt': [10.0 + i * 0.1 for i in range(10)] +
                 [50.0 + i * 0.1 for i in range(10)],
            'z': [2] * 20,
            'seq': ['PEPTIDEA'] * 10 + ['PEPTIDEB'] * 10,
            'closest_peak_intensity_ms1': [1000.0] * 20
        })
        fwhm, sigma, result_df = calculate_elution_width(df)
        assert fwhm > 0
        assert sigma > 0
