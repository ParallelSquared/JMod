"""
This Source Code Form is subject to the terms of the Oxford Nanopore
Technologies, Ltd. Public License, v. 1.0.  Full licence can be found
at https://github.com/ParallelSquared/JMod/blob/main/LICENSE.txt
"""

"""
Tests for functions in spectral_fitting.py
"""
import pytest
import numpy as np

# Import the functions we want to test
try:
    from src.utils.spectral_similarity_metrics import (
        get_scribe, get_residuals, gof_stat, get_manhattan_distance,
        get_closest_ms1, max_matched_residual
    )
    from src.spectral_fitting import preprocess_dia_spectrum, filter_candidates_by_window
except ImportError:
    # If direct import fails, we might need to adjust based on actual module structure
    pass


# Simple class to represent MS1 spectrum for testing
class MS1Spectrum:
    def __init__(self, rt):
        self.RT = rt


class TestGetScribe:
    """Test cases for the get_scribe function"""
    
    def test_get_scribe_basic(self):
        """Test basic SCRIBE score calculation"""
        # Setup test data
        row_idx_split = [np.array([0, 1, 2]), np.array([3, 4])]
        col_idx_split = [np.array([0, 0, 0]), np.array([1, 1])]
        prec_val_split = [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0])]
        val_obs = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
        
        #Precursor 1 
        #(1.0, 2.0, 3.0) -> sqrt(1.0), sqrt(2.0), sqrt(3.0)
        #(1.5, 2.5, 3.5) -> sqrt(1.5), sqrt(2.5), sqrt(3.5)
        #Precursor 2
        #(4.0, 5.0) -> sqrt(4.0), sqrt(5.0)
        #(4.5, 5.5) -> sqrt(4.5), sqrt(5.5)
        # Calculate SCRIBE scores
        # Should return array with 2 scores (one per precursor)
        
        result = get_scribe(row_idx_split, col_idx_split, prec_val_split, val_obs)
        
        # Should return array with 2 scores (one per precursor)
        assert len(result) == 2
        assert isinstance(result, np.ndarray)
        assert np.all(result >= 0)  # SCRIBE scores should be non-negative

        a_norm = np.sqrt(1.0) + np.sqrt(2.0) + np.sqrt(3.0)
        i_norm = np.sqrt(1.5) + np.sqrt(2.5) + np.sqrt(3.5)
        expected_val = ((np.sqrt(1.0)/a_norm) - (np.sqrt(1.5)/i_norm))**2 + ((np.sqrt(2.0)/a_norm) - (np.sqrt(2.5)/i_norm))**2 + ((np.sqrt(3.0)/a_norm) - (np.sqrt(3.5)/i_norm))**2 
        assert np.abs(result[0] - expected_val) < 1e-10
    
    def test_get_scribe_empty_input(self):
        """Test SCRIBE calculation with empty input"""
        row_idx_split = []
        col_idx_split = []
        prec_val_split = []
        val_obs = np.array([])
        
        result = get_scribe(row_idx_split, col_idx_split, prec_val_split, val_obs)
        
        assert len(result) == 0
        assert isinstance(result, np.ndarray)
    
    def test_get_scribe_single_precursor(self):
        """Test SCRIBE calculation with single precursor"""
        row_idx_split = [np.array([0, 1])]
        col_idx_split = [np.array([0, 0])]
        prec_val_split = [np.array([1.0, 4.0])]
        val_obs = np.array([1.0, 4.0])
        
        result = get_scribe(row_idx_split, col_idx_split, prec_val_split, val_obs)
        
        assert len(result) == 1
        # Perfect match should give SCRIBE score of 0
        assert np.isclose(result[0], 0.0, atol=1e-10)
    
    def test_get_scribe_different_intensities(self):
        """Test SCRIBE calculation with different intensity patterns"""
        row_idx_split = [np.array([0, 1]), np.array([2, 3])]
        col_idx_split = [np.array([0, 0]), np.array([1, 1])]
        prec_val_split = [np.array([1.0, 1.0]), np.array([10.0, 10.0])]
        val_obs = np.array([1.0, 1.0, 10.0, 10.0])
        
        result = get_scribe(row_idx_split, col_idx_split, prec_val_split, val_obs)
        
        assert len(result) == 2
        # Both should have perfect matches, so scores should be close to 0
        assert np.allclose(result, 0.0, atol=1e-10)


class TestGetResiduals:
    """Test cases for the get_residuals function"""
    
    def test_get_residuals_basic(self):
        """Test basic residual calculation"""
        # Setup test data
        ref_sparse_val = [np.array([1.0, 2.0])]
        ref_sparse_row = [np.array([0, 1])]
        ref_sparse_col = [np.array([0, 0])]
        decoy_sparse_val = [np.array([3.0])]
        decoy_sparse_row = [np.array([2])]
        decoy_sparse_col = [np.array([0])]
        val_obs = np.array([2.0, 4.0, 6.0])
        coeffs = np.array([2.0, 1.0])
        ref_spec_offset = 0
        decoy_spec_offset = 1
        
        residuals, y_pred = get_residuals(
            ref_sparse_val, ref_sparse_row, ref_sparse_col,
            decoy_sparse_val, decoy_sparse_row, decoy_sparse_col,
            val_obs, coeffs, ref_spec_offset, decoy_spec_offset
        )
        test_residuals = val_obs - np.array([2.0*1.0, 2.0*2.0, 1.0*3.0])
        expected_y_pred = np.array([2.0*1.0, 2.0*2.0, 1.0*3.0])  # coeffs * values
        assert len(residuals) == len(val_obs)
        assert len(y_pred) == len(val_obs)
        assert isinstance(residuals, np.ndarray)
        assert isinstance(y_pred, np.ndarray)
        assert np.allclose(expected_y_pred - y_pred, 0.0, atol=1e-10)
        assert np.allclose(test_residuals - residuals, 0.0, atol=1e-10)
    
    def test_get_residuals_empty_decoy(self):
        """Test residual calculation with no decoy data"""
        ref_sparse_val = [np.array([1.0, 2.0])]
        ref_sparse_row = [np.array([0, 1])]
        ref_sparse_col = [np.array([0, 0])]
        decoy_sparse_val = []
        decoy_sparse_row = []
        decoy_sparse_col = []
        val_obs = np.array([2.0, 4.0])
        coeffs = np.array([1.0])
        ref_spec_offset = 0
        decoy_spec_offset = 1
        
        residuals, y_pred = get_residuals(
            ref_sparse_val, ref_sparse_row, ref_sparse_col,
            decoy_sparse_val, decoy_sparse_row, decoy_sparse_col,
            val_obs, coeffs, ref_spec_offset, decoy_spec_offset
        )
        
        assert len(residuals) == len(val_obs)
        assert len(y_pred) == len(val_obs)
        # Check that predictions are calculated correctly
        expected_y_pred = np.array([1.0, 2.0])  # coeff * values
        np.testing.assert_array_almost_equal(y_pred, expected_y_pred)
    
    def test_get_residuals_perfect_fit(self):
        """Test residual calculation with perfect fit"""
        ref_sparse_val = [np.array([1.0, 2.0])]
        ref_sparse_row = [np.array([0, 1])]
        ref_sparse_col = [np.array([0, 0])]
        decoy_sparse_val = []
        decoy_sparse_row = []
        decoy_sparse_col = []
        val_obs = np.array([2.0, 4.0])
        coeffs = np.array([2.0])
        ref_spec_offset = 0
        decoy_spec_offset = 1
        
        residuals, y_pred = get_residuals(
            ref_sparse_val, ref_sparse_row, ref_sparse_col,
            decoy_sparse_val, decoy_sparse_row, decoy_sparse_col,
            val_obs, coeffs, ref_spec_offset, decoy_spec_offset
        )
        
        # Perfect fit should give zero residuals
        expected_residuals = np.array([0.0, 0.0])
        np.testing.assert_array_almost_equal(residuals, expected_residuals)


class TestGofStat:
    """Test cases for the gof_stat function"""
    
    def test_gof_stat_basic(self):
        """Test basic goodness-of-fit calculation"""
        row_idx_split = [np.array([0, 1])]
        col_idx_split = [np.array([0, 0])]
        val_split = [np.array([1.0, 2.0])]
        residuals = np.array([0.1, 0.2])
        val_obs = np.array([1.0, 2.0])
        coeffs = np.array([1.0])
        offset = 0
        
        result, max_unmatched, max_matched = gof_stat(
            row_idx_split, col_idx_split, val_split,
            residuals, val_obs, coeffs, offset
        )
        
        assert len(result) == 1
        assert len(max_unmatched) == 1
        assert len(max_matched) == 1
        assert isinstance(result, np.ndarray)
        assert isinstance(max_unmatched, np.ndarray)
        assert isinstance(max_matched, np.ndarray)
    
    def test_gof_stat_empty_input(self):
        """Test goodness-of-fit calculation with empty input"""
        row_idx_split = []
        col_idx_split = []
        val_split = []
        residuals = np.array([])
        val_obs = np.array([])
        coeffs = np.array([])
        offset = 0
        
        result, max_unmatched, max_matched = gof_stat(
            row_idx_split, col_idx_split, val_split,
            residuals, val_obs, coeffs, offset
        )
        
        assert len(result) == 0
        assert len(max_unmatched) == 0
        assert len(max_matched) == 0
    
    def test_gof_stat_zero_fitted_peaks(self):
        """Test goodness-of-fit calculation when fitted peaks sum to zero"""
        row_idx_split = [np.array([0])]
        col_idx_split = [np.array([0])]
        val_split = [np.array([1.0])]
        residuals = np.array([0.1])
        val_obs = np.array([1.0])
        coeffs = np.array([0.0])  # Zero coefficient
        offset = 0
        
        result, max_unmatched, max_matched = gof_stat(
            row_idx_split, col_idx_split, val_split,
            residuals, val_obs, coeffs, offset
        )
        
        # Should handle zero fitted peaks gracefully
        assert len(result) == 1
        assert np.isfinite(result[0])
    
    def test_gof_stat_matched_vs_unmatched(self):
        """Test goodness-of-fit distinguishes between matched and unmatched peaks"""
        row_idx_split = [np.array([0, 1])]
        col_idx_split = [np.array([0, 0])]
        val_split = [np.array([1.0, 2.0])]
        residuals = np.array([0.1, 0.2])
        val_obs = np.array([1.0, 1e-7])  # Second peak is essentially unmatched
        coeffs = np.array([1.0])
        offset = 0
        
        result, max_unmatched, max_matched = gof_stat(
            row_idx_split, col_idx_split, val_split,
            residuals, val_obs, coeffs, offset
        )
        
        # Should distinguish between matched and unmatched peaks
        assert len(result) == 1
        assert max_matched[0] != max_unmatched[0]


class TestGetManhattanDistance:
    """Test cases for the get_manhattan_distance function"""
    
    def test_get_manhattan_distance_basic(self):
        """Test basic Manhattan distance calculation"""
        row_idx_split = [np.array([0, 1])]
        col_idx_split = [np.array([0, 0])]
        prec_val_split = [np.array([1.0, 2.0])]
        val_obs = np.array([1.5, 2.5])
        y_pred = np.array([1.2, 2.3])
        
        manhattan_distances, spectral_contrasts = get_manhattan_distance(
            row_idx_split, col_idx_split, prec_val_split, val_obs, y_pred
        )
        
        assert len(manhattan_distances) == 1
        assert len(spectral_contrasts) == 1
        assert isinstance(manhattan_distances, np.ndarray)
        assert isinstance(spectral_contrasts, np.ndarray)
        assert np.all(np.isfinite(manhattan_distances))
        assert np.all(np.isfinite(spectral_contrasts))
    
    def test_get_manhattan_distance_empty_input(self):
        """Test Manhattan distance calculation with empty input"""
        row_idx_split = []
        col_idx_split = []
        prec_val_split = []
        val_obs = np.array([])
        y_pred = np.array([])
        
        manhattan_distances, spectral_contrasts = get_manhattan_distance(
            row_idx_split, col_idx_split, prec_val_split, val_obs, y_pred
        )
        
        assert len(manhattan_distances) == 0
        assert len(spectral_contrasts) == 0
    
    def test_get_manhattan_distance_perfect_match(self):
        """Test Manhattan distance calculation with perfect match"""
        row_idx_split = [np.array([0, 1])]
        col_idx_split = [np.array([0, 0])]
        prec_val_split = [np.array([1.0, 2.0])]
        val_obs = np.array([1.0, 2.0])
        y_pred = np.array([1.0, 2.0])  # Perfect prediction
        
        manhattan_distances, spectral_contrasts = get_manhattan_distance(
            row_idx_split, col_idx_split, prec_val_split, val_obs, y_pred
        )
        
        # Perfect match should give very high Manhattan distance (low log value becomes high)
        assert len(manhattan_distances) == 1
        assert manhattan_distances[0] == np.finfo(np.float32).min  # Perfect fit case
        assert np.isclose(spectral_contrasts[0], 1.0)  # Perfect correlation
    
    def test_get_manhattan_distance_zero_observed(self):
        """Test Manhattan distance calculation when observed values are zero"""
        row_idx_split = [np.array([0, 1])]
        col_idx_split = [np.array([0, 0])]
        prec_val_split = [np.array([1.0, 2.0])]
        val_obs = np.array([0.0, 0.0])  # No observed intensity
        y_pred = np.array([1.0, 2.0])
        
        manhattan_distances, spectral_contrasts = get_manhattan_distance(
            row_idx_split, col_idx_split, prec_val_split, val_obs, y_pred
        )
        
        # Zero observed should be handled as bad fit
        assert len(manhattan_distances) == 1
        assert manhattan_distances[0] == np.finfo(np.float32).max  # Bad fit case
        assert spectral_contrasts[0] == 0.0
    
    def test_get_manhattan_distance_multiple_precursors(self):
        """Test Manhattan distance calculation with multiple precursors"""
        row_idx_split = [np.array([0, 1]), np.array([2, 3])]
        col_idx_split = [np.array([0, 0]), np.array([1, 1])]
        prec_val_split = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
        val_obs = np.array([1.1, 2.1, 3.1, 4.1])
        y_pred = np.array([1.05, 2.05, 3.05, 4.05])
        
        manhattan_distances, spectral_contrasts = get_manhattan_distance(
            row_idx_split, col_idx_split, prec_val_split, val_obs, y_pred
        )
        
        assert len(manhattan_distances) == 2
        assert len(spectral_contrasts) == 2
        # Both should have reasonable values
        assert np.all(np.isfinite(manhattan_distances))
        assert np.all(np.isfinite(spectral_contrasts))
        assert np.all(spectral_contrasts >= 0)
        assert np.all(spectral_contrasts <= 1)
    
    def test_get_manhattan_distance_spectral_contrast_bounds(self):
        """Test that spectral contrast values are properly bounded"""
        row_idx_split = [np.array([0, 1])]
        col_idx_split = [np.array([0, 0])]
        prec_val_split = [np.array([1.0, 2.0])]
        val_obs = np.array([1.0, 2.0])
        y_pred = np.array([1.0, 2.0])
        
        manhattan_distances, spectral_contrasts = get_manhattan_distance(
            row_idx_split, col_idx_split, prec_val_split, val_obs, y_pred
        )
        
        # Spectral contrast should be between 0 and 1
        assert np.all(spectral_contrasts >= 0)
        assert np.all(spectral_contrasts <= 1)


class TestGetClosestMS1:
    """Test cases for the get_closest_ms1 function"""
    
    def test_get_closest_ms1_basic(self):
        """Test finding closest MS1 spectrum by retention time"""
        # Create MS1 spectra with RT attributes
        ms1_spectra = [MS1Spectrum(10.0), MS1Spectrum(15.0), MS1Spectrum(20.0), MS1Spectrum(25.0)]
        
        # Test finding closest to RT=16.0 (should return spectrum at RT=15.0)
        result = get_closest_ms1(16.0, ms1_spectra)
        assert result.RT == 15.0
        
        # Test finding closest to RT=22.0 (should return spectrum at RT=20.0)
        result = get_closest_ms1(22.0, ms1_spectra)
        assert result.RT == 20.0
    
    def test_get_closest_ms1_edge_cases(self):
        """Test edge cases for finding closest MS1 spectrum"""
        ms1_spectra = [MS1Spectrum(10.0), MS1Spectrum(20.0)]
        
        # Test RT before all spectra
        result = get_closest_ms1(5.0, ms1_spectra)
        assert result.RT == 10.0
        
        # Test RT after all spectra
        result = get_closest_ms1(30.0, ms1_spectra)
        assert result.RT == 20.0
        
        # Test exact match
        result = get_closest_ms1(10.0, ms1_spectra)
        assert result.RT == 10.0
    
    def test_get_closest_ms1_single_spectrum(self):
        """Test with only one MS1 spectrum"""
        ms1_spectra = [MS1Spectrum(15.0)]
        
        result = get_closest_ms1(20.0, ms1_spectra)
        assert result.RT == 15.0


class TestMaxMatchedResidual:
    """Test cases for the max_matched_residual function"""
    
    def test_max_matched_residual_basic(self):
        """Test finding maximum residual for each precursor"""
        row_idx_split = [np.array([0, 1, 2]), np.array([3, 4])]
        residuals = np.array([0.1, 0.3, 0.2, 0.5, 0.4])
        
        result = max_matched_residual(row_idx_split, residuals)
        
        assert len(result) == 2
        # First precursor uses indices [0, 1, 2] -> residuals [0.1, 0.3, 0.2]
        # Max of values: 0.3
        assert result[0] == 0.3
        # Second precursor uses indices [3, 4] -> residuals [0.5, 0.4]
        # Max of values: 0.5
        assert result[1] == 0.5
    
    def test_max_matched_residual_empty_input(self):
        """Test with empty input"""
        row_idx_split = []
        residuals = np.array([])
        
        result = max_matched_residual(row_idx_split, residuals)
        
        assert len(result) == 0
        assert isinstance(result, np.ndarray)
    
    def test_max_matched_residual_negative_values(self):
        """Test with negative residual values"""
        row_idx_split = [np.array([0, 1]), np.array([2, 3])]
        residuals = np.array([-0.5, -0.1, -0.3, -0.2])
        
        result = max_matched_residual(row_idx_split, residuals)
        
        assert len(result) == 2
        # First precursor uses indices [0, 1] -> residuals [-0.5, -0.1]
        # Max of values: -0.1
        assert result[0] == -0.1
        # Second precursor uses indices [2, 3] -> residuals [-0.3, -0.2]
        # Max of values: -0.2
        assert result[1] == -0.2
    
    def test_max_matched_residual_single_precursor(self):
        """Test with single precursor"""
        row_idx_split = [np.array([0, 1, 2])]
        residuals = np.array([0.2, 0.1, 0.3])
        
        result = max_matched_residual(row_idx_split, residuals)
        
        assert len(result) == 1
        # Single precursor uses indices [0, 1, 2] -> residuals [0.2, 0.1, 0.3]
        # Max of values: 0.3
        assert result[0] == 0.3


class TestAccuracy:
    """Accuracy tests for spectral fitting functions"""
    
    def test_scribe_score_accuracy(self):
        """Test SCRIBE score calculation accuracy with known values"""
        # Test case with known SCRIBE score
        row_idx_split = [np.array([0, 1])]
        col_idx_split = [np.array([0, 0])]
        prec_val_split = [np.array([0.25, 0.75])]  # 25% and 75% of total intensity
        val_obs = np.array([0.4, 0.6])  # 40% and 60% of total intensity
        
        result = get_scribe(row_idx_split, col_idx_split, prec_val_split, val_obs)
        
        # Calculate expected SCRIBE score manually
        # SCRIBE = sum((sqrt(h_i)/sum(sqrt(h)) - sqrt(x_i)/sum(sqrt(x)))^2)
        h_sqrt = np.sqrt(prec_val_split[0])
        x_sqrt = np.sqrt(val_obs[[0, 1]])
        h_sqrt_sum = np.sum(h_sqrt)
        x_sqrt_sum = np.sum(x_sqrt)
        expected_scribe = np.sum((h_sqrt/h_sqrt_sum - x_sqrt/x_sqrt_sum)**2)
        
        np.testing.assert_allclose(result[0], expected_scribe, rtol=1e-10)
    
    def test_residuals_calculation_accuracy(self):
        """Test residual calculation accuracy with precise values"""
        # Setup precise test case
        ref_sparse_val = [np.array([2.0, 3.0]), np.array([1.5])]
        ref_sparse_row = [np.array([0, 1]), np.array([2])]
        ref_sparse_col = [np.array([0, 0]), np.array([1])]
        decoy_sparse_val = []
        decoy_sparse_row = []
        decoy_sparse_col = []
        val_obs = np.array([5.0, 7.5, 3.0])
        coeffs = np.array([2.0, 1.5])
        
        residuals, y_pred = get_residuals(
            ref_sparse_val, ref_sparse_row, ref_sparse_col,
            decoy_sparse_val, decoy_sparse_row, decoy_sparse_col,
            val_obs, coeffs, 0, 2
        )
        
        # Expected predictions: [2.0*2.0, 3.0*2.0, 1.5*1.5] = [4.0, 6.0, 2.25]
        expected_y_pred = np.array([4.0, 6.0, 2.25])
        expected_residuals = val_obs - expected_y_pred  # [1.0, 1.5, 0.75]
        
        np.testing.assert_allclose(y_pred, expected_y_pred, rtol=1e-10)
        np.testing.assert_allclose(residuals, expected_residuals, rtol=1e-10)
    
    def test_manhattan_distance_accuracy(self):
        """Test Manhattan distance calculation accuracy"""
        row_idx_split = [np.array([0, 1, 2])]
        col_idx_split = [np.array([0, 0, 0])]
        prec_val_split = [np.array([1.0, 2.0, 3.0])]
        val_obs = np.array([1.2, 2.1, 2.9])
        y_pred = np.array([1.1, 2.0, 3.1])
        
        manhattan_distances, spectral_contrasts = get_manhattan_distance(
            row_idx_split, col_idx_split, prec_val_split, val_obs, y_pred
        )
        
        # Calculate expected Manhattan distance manually
        manhattan_sum = np.sum(np.abs(y_pred[[0,1,2]] - val_obs[[0,1,2]]))  # |1.1-1.2| + |2.0-2.1| + |3.1-2.9| = 0.4
        x_sum = np.sum(val_obs[[0,1,2]])  # 1.2 + 2.1 + 2.9 = 6.2
        expected_manhattan = -np.log2(manhattan_sum / x_sum)
        
        np.testing.assert_allclose(manhattan_distances[0], expected_manhattan, rtol=1e-6)
        
        # Check spectral contrast calculation (cosine similarity)
        u2_sum = np.sum(y_pred[[0,1,2]]**2)
        v2_sum = np.sum(val_obs[[0,1,2]]**2)
        uv_sum = np.sum(y_pred[[0,1,2]] * val_obs[[0,1,2]])
        expected_contrast = uv_sum / (np.sqrt(u2_sum) * np.sqrt(v2_sum))
        
        np.testing.assert_allclose(spectral_contrasts[0], expected_contrast, rtol=1e-6)
    
    def test_gof_stat_accuracy(self):
        """Test goodness-of-fit calculation accuracy"""
        row_idx_split = [np.array([0, 1])]
        col_idx_split = [np.array([0, 0])]
        val_split = [np.array([2.0, 3.0])]
        residuals = np.array([0.5, -0.3])
        val_obs = np.array([2.5, 2.7])
        coeffs = np.array([1.0])
        offset = 0
        
        result, max_unmatched, max_matched = gof_stat(
            row_idx_split, col_idx_split, val_split,
            residuals, val_obs, coeffs, offset
        )
        
        # Calculate expected values manually
        sum_of_residuals = abs(0.5) + abs(-0.3)  # 0.8
        sum_of_fitted_peaks = abs(1.0 * 2.0) + abs(1.0 * 3.0)  # 5.0
        expected_gof = np.log2(sum_of_residuals / sum_of_fitted_peaks)
        
        np.testing.assert_allclose(result[0], expected_gof, rtol=1e-10)
        
        # Both values have significant observed intensity, so max_matched should be populated
        expected_max_matched = np.log2(max(abs(0.5), abs(-0.3)) / (sum_of_fitted_peaks + 1e-10) + 1e-10)
        np.testing.assert_allclose(max_matched[0], expected_max_matched, rtol=1e-6)


class TestIntegration:
    """Integration tests for spectral fitting functions"""
    
    def test_workflow_integration(self):
        """Test that functions work together in a typical workflow"""
        # Setup realistic test data
        row_idx_split = [np.array([0, 1, 2]), np.array([3, 4])]
        col_idx_split = [np.array([0, 0, 0]), np.array([1, 1])]
        prec_val_split = [np.array([0.3, 0.5, 0.2]), np.array([0.6, 0.4])]
        val_obs = np.array([0.35, 0.45, 0.25, 0.55, 0.35])
        coeffs = np.array([1.2, 0.8])
        
        # Test SCRIBE scores
        scribe_scores = get_scribe(row_idx_split, col_idx_split, prec_val_split, val_obs)
        assert len(scribe_scores) == 2
        
        # Test residuals calculation
        residuals, y_pred = get_residuals(
            prec_val_split, row_idx_split, col_idx_split,
            [], [], [],  # No decoy data
            val_obs, coeffs, 0, 2
        )
        assert len(residuals) == len(val_obs)
        assert len(y_pred) == len(val_obs)
        
        # Test goodness-of-fit
        gof_stats, max_unmatched, max_matched = gof_stat(
            row_idx_split, col_idx_split, prec_val_split,
            residuals, val_obs, coeffs, 0
        )
        assert len(gof_stats) == 2
        
        # Test Manhattan distance
        manhattan_distances, spectral_contrasts = get_manhattan_distance(
            row_idx_split, col_idx_split, prec_val_split, val_obs, y_pred
        )
        assert len(manhattan_distances) == 2
        assert len(spectral_contrasts) == 2
        
        # All results should be finite
        assert np.all(np.isfinite(scribe_scores))
        assert np.all(np.isfinite(residuals))
        assert np.all(np.isfinite(y_pred))
        assert np.all(np.isfinite(gof_stats))
        assert np.all(np.isfinite(manhattan_distances))
        assert np.all(np.isfinite(spectral_contrasts))


class TestPreprocessDiaSpectrum:
    """Test cases for the preprocess_dia_spectrum function"""
    
    def test_basic_spectrum_preprocessing(self):
        """Test basic spectrum preprocessing with simple peaks"""
        # Create test spectrum with 3 distinct peaks
        spectrum = np.array([
            [100.0, 50.0],
            [200.0, 100.0],
            [300.0, 75.0]
        ])
        mz_tol = 10e-6  # 10 ppm as fraction
        
        merged, breaks, centers = preprocess_dia_spectrum(spectrum, mz_tol)
        
        # Should have same number of peaks (no merging needed)
        assert merged.shape[0] == 3
        assert len(breaks) == 6  # 2 boundaries per peak
        assert len(centers) == 3  # One center per peak
        
        # Check intensities are preserved
        np.testing.assert_array_equal(merged[:, 1], spectrum[:, 1])
        
    def test_peak_merging(self):
        """Test that nearby peaks are properly merged"""
        # Create spectrum with overlapping peaks
        # At 100.0 m/z with 10 ppm tolerance, the window is 0.001 m/z
        spectrum = np.array([
            [100.0, 50.0],
            [100.0005, 30.0],  # Should merge with first peak (within 10 ppm)
            [100.002, 20.0],   # Should NOT merge (outside 10 ppm)
            [200.0, 100.0]
        ])
        mz_tol = 10e-6  # 10 ppm as fraction
        
        merged, breaks, centers = preprocess_dia_spectrum(spectrum, mz_tol)
        
        # Should have 3 peaks after merging
        assert merged.shape[0] == 3
        
        # First merged peak should have summed intensity
        assert merged[0, 1] == 80.0  # 50 + 30
        
        # Check m/z values
        assert merged[0, 0] == 100.0  # First m/z preserved
        assert merged[1, 0] == 100.002
        assert merged[2, 0] == 200.0
        
    def test_centroid_breaks_calculation(self):
        """Test that centroid breaks are correctly calculated"""
        spectrum = np.array([
            [100.0, 50.0],
            [200.0, 100.0]
        ])
        mz_tol = 10e-6  # 10 ppm as fraction
        
        merged, breaks, centers = preprocess_dia_spectrum(spectrum, mz_tol)
        
        # Check break calculations
        # For 100 m/z at 10 ppm: tolerance = 100 * 10e-6 = 0.001
        # For 200 m/z at 10 ppm: tolerance = 200 * 10e-6 = 0.002
        expected_breaks = np.array([
            100.0 - 100.0 * 10e-6,  # Lower bound of first peak
            100.0 + 100.0 * 10e-6,  # Upper bound of first peak
            200.0 - 200.0 * 10e-6,  # Lower bound of second peak
            200.0 + 200.0 * 10e-6   # Upper bound of second peak
        ])
        np.testing.assert_allclose(breaks, expected_breaks, rtol=1e-6)
        
        # Check bin centers
        expected_centers = np.array([100.0, 200.0])
        np.testing.assert_allclose(centers, expected_centers, rtol=1e-6)
        
    def test_empty_spectrum(self):
        """Test handling of empty spectrum"""
        spectrum = np.array([]).reshape(0, 2)
        mz_tol = 10e-6  # 10 ppm as fraction
        
        merged, breaks, centers = preprocess_dia_spectrum(spectrum, mz_tol)
        
        assert merged.shape[0] == 0
        assert len(breaks) == 0
        assert len(centers) == 0
        
    def test_single_peak(self):
        """Test preprocessing with single peak"""
        spectrum = np.array([[150.0, 75.0]])
        mz_tol = 20e-6  # 20 ppm as fraction
        
        merged, breaks, centers = preprocess_dia_spectrum(spectrum, mz_tol)
        
        assert merged.shape == (1, 2)
        assert len(breaks) == 2
        assert len(centers) == 1
        assert centers[0] == 150.0
        
    def test_complex_merging_scenario(self):
        """Test complex scenario with multiple merge groups"""
        # Create peaks that form distinct merge groups
        spectrum = np.array([
            # Group 1: These should merge
            [100.0, 10.0],
            [100.0001, 20.0],
            [100.0002, 15.0],
            # Group 2: Separate peak
            [100.01, 30.0],
            # Group 3: These should merge
            [200.0, 40.0],
            [200.0001, 25.0]
        ])
        mz_tol = 5e-6  # 5 ppm as fraction - tighter tolerance
        
        merged, breaks, centers = preprocess_dia_spectrum(spectrum, mz_tol)
        
        # Should have 3 merged peaks
        assert merged.shape[0] == 3
        
        # Check merged intensities
        assert merged[0, 1] == 45.0  # 10 + 20 + 15
        assert merged[1, 1] == 30.0  # No merge
        assert merged[2, 1] == 65.0  # 40 + 25
        
    def test_intensity_filtering(self):
        """Test that zero-intensity peaks are filtered out"""
        spectrum = np.array([
            [100.0, 50.0],
            [200.0, 0.0],  # Zero intensity
            [300.0, 75.0]
        ])
        mz_tol = 10e-6  # 10 ppm as fraction
        
        merged, breaks, centers = preprocess_dia_spectrum(spectrum, mz_tol)
        
        # Should only have 2 peaks (zero intensity filtered)
        assert merged.shape[0] == 2
        assert 200.0 not in merged[:, 0]


class TestFilterCandidatesByWindow:
    """Test cases for the filter_candidates_by_window function"""
    
    def test_basic_mass_window_filtering(self):
        """Test basic mass window filtering without RT"""
        # Create test data
        rt_mz = np.array([
            [10.0, 500.0],   # Within window
            [10.0, 500.5],   # Within window
            [10.0, 502.0],   # Outside window
            [10.0, 498.0]    # Outside window (assuming windowWidth=2)
        ])
        all_keys = ['pep1', 'pep2', 'pep3', 'pep4']
        prec_mz = 500.25
        prec_rt = 10.0
        windowWidth = 2.0  # +/- 1.0 m/z
        
        window_idxs, candidates = filter_candidates_by_window(
            rt_mz, all_keys, prec_mz, prec_rt, windowWidth
        )
        
        # Should include only peptides within mass window
        assert len(window_idxs) == 2
        assert list(window_idxs) == [0, 1]
        assert candidates == ['pep1', 'pep2']
    
    def test_rt_filtering(self):
        """Test filtering with retention time constraints"""
        rt_mz = np.array([
            [10.0, 500.0],   # Within mass and RT
            [10.5, 500.0],   # Within mass and RT
            [11.0, 500.0],   # Within mass, outside RT
            [9.0, 500.0]     # Within mass, outside RT
        ])
        all_keys = ['pep1', 'pep2', 'pep3', 'pep4']
        prec_mz = 500.0
        prec_rt = 10.0
        windowWidth = 2.0
        rt_tol = 0.6
        
        window_idxs, candidates = filter_candidates_by_window(
            rt_mz, all_keys, prec_mz, prec_rt, windowWidth,
            rt_filter=True, rt_tol=rt_tol
        )
        
        # Should include only peptides within both mass and RT windows
        assert len(window_idxs) == 2
        assert list(window_idxs) == [0, 1]
        assert candidates == ['pep1', 'pep2']
    
    def test_ms1_mz_filtering(self):
        """Test filtering with MS1 m/z instead of precursor m/z"""
        rt_mz = np.array([
            [10.0, 500.0],
            [10.0, 500.005],  # Within 10 ppm of 500
            [10.0, 500.01],   # Outside 10 ppm of 500
            [10.0, 499.99]    # Outside 10 ppm of 500
        ])
        all_keys = ['pep1', 'pep2', 'pep3', 'pep4']
        ms1_mz = 500.0
        ms1_tol = 10e-6  # 10 ppm
        
        window_idxs, candidates = filter_candidates_by_window(
            rt_mz, all_keys, 0, 0, 0,  # These are ignored when ms1_mz is provided
            ms1_mz=ms1_mz, ms1_tol=ms1_tol
        )
        
        # Should include only peptides within MS1 tolerance
        assert len(window_idxs) == 2
        assert list(window_idxs) == [0, 1]
        assert candidates == ['pep1', 'pep2']
    
    def test_empty_results(self):
        """Test when no candidates pass filtering"""
        rt_mz = np.array([
            [10.0, 600.0],
            [10.0, 700.0]
        ])
        all_keys = ['pep1', 'pep2']
        prec_mz = 500.0
        windowWidth = 10.0  # Even with wide window, nothing matches
        
        window_idxs, candidates = filter_candidates_by_window(
            rt_mz, all_keys, prec_mz, 10.0, windowWidth
        )
        
        assert len(window_idxs) == 0
        assert candidates == []
    
    def test_error_handling(self):
        """Test that appropriate errors are raised for missing parameters"""
        rt_mz = np.array([[10.0, 500.0]])
        all_keys = ['pep1']
        
        # Test missing ms1_tol when ms1_mz is provided
        with pytest.raises(ValueError, match="ms1_tol must be provided"):
            filter_candidates_by_window(
                rt_mz, all_keys, 500.0, 10.0, 1.0,
                ms1_mz=500.0
            )
        
        # Test missing rt_tol when rt_filter is True
        with pytest.raises(ValueError, match="rt_tol must be provided"):
            filter_candidates_by_window(
                rt_mz, all_keys, 500.0, 10.0, 1.0,
                rt_filter=True
            )
    
    def test_all_candidates_pass(self):
        """Test when all candidates pass filtering"""
        rt_mz = np.array([
            [10.0, 500.0],
            [10.0, 500.1],
            [10.0, 500.2]
        ])
        all_keys = ['pep1', 'pep2', 'pep3']
        prec_mz = 500.1
        windowWidth = 1.0  # Wide enough to include all
        
        window_idxs, candidates = filter_candidates_by_window(
            rt_mz, all_keys, prec_mz, 10.0, windowWidth
        )
        
        assert len(window_idxs) == 3
        assert list(window_idxs) == [0, 1, 2]
        assert candidates == ['pep1', 'pep2', 'pep3']
    
    def test_boundary_conditions(self):
        """Test peptides exactly at window boundaries"""
        rt_mz = np.array([
            [10.0, 499.49],  # Outside (0.51 away)
            [10.0, 499.51],  # Inside (0.49 away)
            [10.0, 500.0],   # Center (0.0 away)
            [10.0, 500.49],  # Inside (0.49 away)
            [10.0, 500.51]   # Outside (0.51 away)
        ])
        all_keys = ['pep1', 'pep2', 'pep3', 'pep4', 'pep5']
        prec_mz = 500.0
        windowWidth = 1.0  # +/- 0.5
        
        window_idxs, candidates = filter_candidates_by_window(
            rt_mz, all_keys, prec_mz, 10.0, windowWidth
        )
        
        # Should include only those strictly within boundaries (< not <=)
        assert len(window_idxs) == 3
        assert list(window_idxs) == [1, 2, 3]
        assert candidates == ['pep2', 'pep3', 'pep4']


if __name__ == "__main__":
    pytest.main([__file__])
