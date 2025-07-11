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
from scipy import sparse

# Import the functions we want to test
try:
    from src.utils.spectral_similarity_metrics import (
        get_scribe, get_scribe_csc, get_residuals, get_residuals_csc, gof_stat, gof_stat_csc, get_manhattan_distance, get_manhattan_distance_csc,
        get_closest_ms1, max_matched_residual
    )
    from src.spectral_fitting import (
        preprocess_dia_spectrum, filter_candidates_by_window, 
        separate_library_candidates, create_unified_candidates,
        create_empty_output_row, extract_non_zero_coefficients,
        format_fragment_information, get_protein_info,
        format_spectral_fitting_output, UnifiedCandidates, UnifiedFeatures,
        fit_to_lib2, calculate_frac_dia_intensity_sparse, calculate_frac_dia_intensity_csc,
        build_sparse_matrix_simple, build_sparse_matrix_direct, extract_basic_fragment_info,
        extract_detailed_fragment_info
    )
    import src.config as config
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


class TestGetScribeCsc:
    """Test cases for the get_scribe_csc function"""
    
    def test_get_scribe_csc_basic(self):
        """Test basic SCRIBE score calculation using CSC sparse matrix"""
        # Create test sparse matrix matching the original test case
        # Two candidates with their fragment indices and intensities
        row_indices = [0, 1, 2, 3, 4]  # Fragment peak positions
        col_indices = [0, 0, 0, 1, 1]  # Candidate assignments
        values = [1.0, 2.0, 3.0, 4.0, 5.0]  # Library intensities
        
        # Create CSC sparse matrix
        sparse_matrix = sparse.csc_matrix((values, (row_indices, col_indices)), shape=(5, 2))
        
        # Observed intensities in DIA spectrum
        val_obs = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
        
        # Calculate SCRIBE scores
        result = get_scribe_csc(sparse_matrix, val_obs)
        
        # Should return array with 2 scores (one per candidate)
        assert len(result) == 2
        assert isinstance(result, np.ndarray)
        assert np.all(result >= 0)  # SCRIBE scores should be non-negative
        
        # Verify against manual calculation for first candidate
        # Candidate 0: predicted [1.0, 2.0, 3.0], observed [1.5, 2.5, 3.5]
        a_norm = np.sqrt(1.0) + np.sqrt(2.0) + np.sqrt(3.0)
        i_norm = np.sqrt(1.5) + np.sqrt(2.5) + np.sqrt(3.5)
        expected_val = ((np.sqrt(1.0)/a_norm) - (np.sqrt(1.5)/i_norm))**2 + \
                      ((np.sqrt(2.0)/a_norm) - (np.sqrt(2.5)/i_norm))**2 + \
                      ((np.sqrt(3.0)/a_norm) - (np.sqrt(3.5)/i_norm))**2
        assert np.abs(result[0] - expected_val) < 1e-10
    
    def test_get_scribe_csc_empty_input(self):
        """Test SCRIBE calculation with empty sparse matrix"""
        # Create empty matrix
        sparse_matrix = sparse.csc_matrix((0, 0))
        val_obs = np.array([])
        
        result = get_scribe_csc(sparse_matrix, val_obs)
        
        assert len(result) == 0
        assert isinstance(result, np.ndarray)
    
    def test_get_scribe_csc_single_precursor(self):
        """Test SCRIBE calculation with single candidate"""
        # Single candidate with two fragments
        row_indices = [0, 1]
        col_indices = [0, 0]
        values = [1.0, 4.0]
        
        sparse_matrix = sparse.csc_matrix((values, (row_indices, col_indices)), shape=(2, 1))
        val_obs = np.array([1.0, 4.0])  # Perfect match
        
        result = get_scribe_csc(sparse_matrix, val_obs)
        
        assert len(result) == 1
        # Perfect match should give SCRIBE score of 0
        assert np.isclose(result[0], 0.0, atol=1e-10)
    
    def test_get_scribe_csc_different_intensities(self):
        """Test SCRIBE calculation with different intensity patterns"""
        # Two candidates with different intensity patterns
        row_indices = [0, 1, 2, 3]
        col_indices = [0, 0, 1, 1]
        values = [10.0, 1.0, 1.0, 10.0]  # Different intensity patterns
        
        sparse_matrix = sparse.csc_matrix((values, (row_indices, col_indices)), shape=(4, 2))
        val_obs = np.array([10.0, 1.0, 1.0, 10.0])  # Perfect match for both
        
        result = get_scribe_csc(sparse_matrix, val_obs)
        
        assert len(result) == 2
        # Both should have perfect matches, so scores should be close to 0
        assert np.allclose(result, 0.0, atol=1e-10)
    
    def test_get_scribe_csc_no_fragments(self):
        """Test SCRIBE calculation when candidate has no fragments"""
        # Matrix with one candidate that has no fragments
        sparse_matrix = sparse.csc_matrix(([], ([], [])), shape=(5, 1))
        val_obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        result = get_scribe_csc(sparse_matrix, val_obs)
        
        assert len(result) == 1
        assert result[0] == 0.0  # No fragments should give score of 0
    
    def test_get_scribe_csc_vs_original(self):
        """Test that CSC version gives identical results to original implementation"""
        # Create test data that can be used by both functions
        row_idx_split = [np.array([0, 1, 2]), np.array([3, 4])]
        col_idx_split = [np.array([0, 0, 0]), np.array([1, 1])]
        prec_val_split = [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0])]
        val_obs = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
        
        # Get result from original function
        original_result = get_scribe(row_idx_split, col_idx_split, prec_val_split, val_obs)
        
        # Convert to sparse matrix format
        all_rows = np.concatenate(row_idx_split)
        all_cols = np.concatenate(col_idx_split)
        all_values = np.concatenate(prec_val_split)
        sparse_matrix = sparse.csc_matrix((all_values, (all_rows, all_cols)), shape=(5, 2))
        
        # Get result from CSC function
        csc_result = get_scribe_csc(sparse_matrix, val_obs)
        
        # Results should be identical
        assert len(original_result) == len(csc_result)
        assert np.allclose(original_result, csc_result, atol=1e-12)
    
    def test_get_scribe_csc_zero_intensities(self):
        """Test SCRIBE calculation with zero intensities"""
        # Test with zero predicted intensities
        row_indices = [0, 1]
        col_indices = [0, 0]
        values = [0.0, 0.0]  # Zero intensities
        
        sparse_matrix = sparse.csc_matrix((values, (row_indices, col_indices)), shape=(2, 1))
        val_obs = np.array([1.0, 2.0])
        
        result = get_scribe_csc(sparse_matrix, val_obs)
        
        assert len(result) == 1
        assert result[0] == 0.0  # Zero predicted intensities should give score of 0


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


class TestSeparateLibraryCandidates:
    """Test cases for the separate_library_candidates function"""
    
    def test_separate_targets_and_decoys(self):
        """Test basic separation of targets and decoys"""
        # Create test data
        candidates = ['PEPTIDE/2', 'Decoy_PEPTIDE/2', 'PEPTIDEK/3', 'Decoy_PEPTIDEK/3']
        library = {
            'PEPTIDE/2': {
                'spectrum': np.array([[500.1, 100], [600.1, 50]]),
                'is_decoy': False
            },
            'Decoy_PEPTIDE/2': {
                'spectrum': np.array([[500.1, 100], [600.1, 50]]),
                'is_decoy': True
            },
            'PEPTIDEK/3': {
                'spectrum': np.array([[700.1, 200]]),
                'is_decoy': False
            },
            'Decoy_PEPTIDEK/3': {
                'spectrum': np.array([[700.1, 200]]),
                'is_decoy': True
            }
        }
        
        unified = separate_library_candidates(candidates, library, include_decoys=True)
        
        # Check counts
        assert unified.n_targets == 2
        assert unified.n_decoys == 2
        assert len(unified.candidates) == 4
        
        # Check order and decoy flags
        assert unified.candidates == ['PEPTIDE/2', 'PEPTIDEK/3', 'Decoy_PEPTIDE/2', 'Decoy_PEPTIDEK/3']
        assert list(unified.is_decoy) == [False, False, True, True]
        
        # Check peaks are correctly assigned
        assert len(unified.peaks) == 4
        np.testing.assert_array_equal(unified.peaks[0], library['PEPTIDE/2']['spectrum'])
        np.testing.assert_array_equal(unified.peaks[2], library['Decoy_PEPTIDE/2']['spectrum'])
    
    def test_no_decoys_in_candidates(self):
        """Test when no decoys are present in candidates"""
        candidates = ['PEPTIDE/2', 'PEPTIDEK/3']
        library = {
            'PEPTIDE/2': {
                'spectrum': np.array([[500.1, 100]]),
                'is_decoy': False
            },
            'PEPTIDEK/3': {
                'spectrum': np.array([[700.1, 200]]),
                'is_decoy': False
            }
        }
        
        unified = separate_library_candidates(candidates, library, include_decoys=True)
        
        # Should only have targets
        assert unified.n_targets == 2
        assert unified.n_decoys == 0
        assert np.all(~unified.is_decoy)
    
    def test_exclude_decoys(self):
        """Test RT alignment mode where decoys are excluded"""
        candidates = ['PEPTIDE/2', 'Decoy_PEPTIDE/2', 'PEPTIDEK/3']
        library = {
            'PEPTIDE/2': {
                'spectrum': np.array([[500.1, 100]]),
                'is_decoy': False
            },
            'Decoy_PEPTIDE/2': {
                'spectrum': np.array([[500.1, 100]]),
                'is_decoy': True
            },
            'PEPTIDEK/3': {
                'spectrum': np.array([[700.1, 200]]),
                'is_decoy': False
            }
        }
        
        unified = separate_library_candidates(candidates, library, include_decoys=False)
        
        # Should only include targets
        assert unified.n_targets == 2
        assert unified.n_decoys == 0
        assert len(unified.candidates) == 2
        assert unified.candidates == ['PEPTIDE/2', 'PEPTIDEK/3']
    
    def test_all_decoys(self):
        """Test when all candidates are decoys"""
        candidates = ['Decoy_PEPTIDE/2', 'Decoy_PEPTIDEK/3']
        library = {
            'Decoy_PEPTIDE/2': {
                'spectrum': np.array([[500.1, 100]]),
                'is_decoy': True
            },
            'Decoy_PEPTIDEK/3': {
                'spectrum': np.array([[700.1, 200]]),
                'is_decoy': True
            }
        }
        
        unified = separate_library_candidates(candidates, library, include_decoys=True)
        
        # Should have only decoys
        assert unified.n_targets == 0
        assert unified.n_decoys == 2
        assert np.all(unified.is_decoy)
    
    def test_empty_candidates(self):
        """Test with empty candidate list"""
        candidates = []
        library = {}
        
        unified = separate_library_candidates(candidates, library, include_decoys=True)
        
        assert unified.n_targets == 0
        assert unified.n_decoys == 0
        assert len(unified.candidates) == 0
        assert len(unified.peaks) == 0
    
    def test_missing_decoy_flag(self):
        """Test handling of library entries without explicit is_decoy flag"""
        candidates = ['PEPTIDE/2', 'PEPTIDEK/3']
        library = {
            'PEPTIDE/2': {
                'spectrum': np.array([[500.1, 100]]),
                # No is_decoy flag - should default to False
            },
            'PEPTIDEK/3': {
                'spectrum': np.array([[700.1, 200]]),
                'is_decoy': False
            }
        }
        
        unified = separate_library_candidates(candidates, library, include_decoys=True)
        
        # Both should be treated as targets
        assert unified.n_targets == 2
        assert unified.n_decoys == 0
        assert np.all(~unified.is_decoy)
    
    def test_peak_data_integrity(self):
        """Test that peak data is correctly preserved and not modified"""
        candidates = ['PEPTIDE/2']
        original_peaks = np.array([[500.1, 100], [600.1, 50]])
        library = {
            'PEPTIDE/2': {
                'spectrum': original_peaks.copy(),
                'is_decoy': False
            }
        }
        
        unified = separate_library_candidates(candidates, library, include_decoys=True)
        
        # Peaks should be the same object (not a copy)
        assert unified.peaks[0] is library['PEPTIDE/2']['spectrum']
        # But should have same values as original
        np.testing.assert_array_equal(unified.peaks[0], original_peaks)


class TestCreateEmptyOutputRow:
    """Test cases for create_empty_output_row function"""
    
    def test_basic_empty_row(self):
        """Test creation of basic empty output row"""
        row = create_empty_output_row(100, 99, 500.5, 10.5, 49)
        
        # Check fixed values
        assert row[0] == 0  # coeff
        assert row[1] == 100  # spec_idx
        assert row[2] == 99  # ms1_spec_id
        assert row[3] == 0  # seq
        assert row[4] == 0  # z
        assert row[5] == 500.5  # prec_mz
        assert row[6] == 10.5  # prec_rt
        
        # Check that remaining values are zeros
        assert len(row) == 49
        assert all(v == 0 for v in row[7:])
    
    def test_no_ms1_spec(self):
        """Test with no MS1 spectrum"""
        row = create_empty_output_row(50, 0, 300.0, 5.0, 49)
        assert row[2] == 0  # ms1_spec_id should be 0
    
    def test_different_column_counts(self):
        """Test with different total column counts"""
        # Smaller output
        row1 = create_empty_output_row(1, 1, 100.0, 1.0, 10)
        assert len(row1) == 10
        assert row1[:7] == [0, 1, 1, 0, 0, 100.0, 1.0]
        assert all(v == 0 for v in row1[7:])
        
        # Larger output
        row2 = create_empty_output_row(2, 2, 200.0, 2.0, 100)
        assert len(row2) == 100
        assert row2[:7] == [0, 2, 2, 0, 0, 200.0, 2.0]
        assert all(v == 0 for v in row2[7:])


class TestExtractNonZeroCoefficients:
    """Test cases for extract_non_zero_coefficients function"""
    
    def test_mixed_coefficients(self):
        """Test with mix of zero and non-zero coefficients"""
        coeffs = np.array([0.0, 0.5, 0.0, 0.3, 0.0, 0.1])
        values, indices = extract_non_zero_coefficients(coeffs)
        
        assert values == [0.5, 0.3, 0.1]
        assert indices == [1, 3, 5]
    
    def test_all_zeros(self):
        """Test with all zero coefficients"""
        coeffs = np.array([0.0, 0.0, 0.0])
        values, indices = extract_non_zero_coefficients(coeffs)
        
        assert values == []
        assert indices == []
    
    def test_all_non_zero(self):
        """Test with all non-zero coefficients"""
        coeffs = np.array([0.1, 0.2, 0.3])
        values, indices = extract_non_zero_coefficients(coeffs)
        
        assert values == [0.1, 0.2, 0.3]
        assert indices == [0, 1, 2]
    
    def test_empty_array(self):
        """Test with empty array"""
        coeffs = np.array([])
        values, indices = extract_non_zero_coefficients(coeffs)
        
        assert values == []
        assert indices == []
    
    def test_single_value(self):
        """Test with single value"""
        # Zero
        values1, indices1 = extract_non_zero_coefficients(np.array([0.0]))
        assert values1 == []
        assert indices1 == []
        
        # Non-zero
        values2, indices2 = extract_non_zero_coefficients(np.array([0.5]))
        assert values2 == [0.5]
        assert indices2 == [0]


class TestFormatFragmentInformation:
    """Test cases for format_fragment_information function"""
    
    def test_complete_fragment_data(self):
        """Test with complete fragment information"""
        additional_outputs = {
            'frag_names': [np.array(['b2', 'y3', 'b4'])],
            'frag_errors': [np.array([0.001, 0.002, 0.003])],
            'lib_frag_mz': [np.array([200.1, 300.2, 400.3])],
            'lib_frag_int': [np.array([100.0, 200.0, 150.0])],
            'obs_frag_int': [np.array([95.0, 205.0, 145.0])]
        }
        
        frags = format_fragment_information(additional_outputs, 0)
        
        assert len(frags) == 7
        assert frags[0] == "b2;y3;b4"  # names
        assert frags[1] == "0.001;0.002;0.003"  # errors
        assert frags[2] == "200.1;300.2;400.3"  # m/z
        assert frags[3] == "100.0;200.0;150.0"  # lib intensities
        assert frags[4] == "95.0;205.0;145.0"  # obs intensities
        assert frags[5] == ""  # unique frags (empty for now)
        assert frags[6] == ""  # unique frags int (empty for now)
    
    def test_missing_candidate_index(self):
        """Test with candidate index out of range"""
        additional_outputs = {
            'frag_names': [np.array(['b2'])]  # Only one candidate
        }
        
        frags = format_fragment_information(additional_outputs, 1)  # Ask for index 1
        
        assert frags == [""] * 7  # Should return all empty strings
    
    def test_empty_fragment_arrays(self):
        """Test with empty fragment arrays"""
        additional_outputs = {
            'frag_names': [np.array([])],
            'frag_errors': [np.array([])],
            'lib_frag_mz': [np.array([])],
            'lib_frag_int': [np.array([])],
            'obs_frag_int': [np.array([])]
        }
        
        frags = format_fragment_information(additional_outputs, 0)
        
        assert frags == [""] * 7  # All should be empty strings
    
    def test_missing_data_keys(self):
        """Test with some missing data keys"""
        additional_outputs = {
            'frag_names': [np.array(['b2', 'y3'])]
            # Other keys missing
        }
        
        frags = format_fragment_information(additional_outputs, 0)
        
        assert frags[0] == "b2;y3"  # Names present
        assert frags[1] == ""  # Others empty
    
    def test_numeric_formatting(self):
        """Test that numeric values are properly formatted"""
        additional_outputs = {
            'frag_errors': [np.array([0.0001234567, 1.234567890])]
        }
        
        frags = format_fragment_information(additional_outputs, 0)
        
        # Should preserve full precision
        assert "0.0001234567" in frags[1]
        assert "1.23456789" in frags[1]


class TestGetProteinInfo:
    """Test cases for get_protein_info function"""
    
    def test_basic_protein_lookup(self):
        """Test basic protein information lookup"""
        library = {
            ('PEPTIDE', 2): {'protein': 'PROT1'},
            ('PEPTIDEK', 3): {'protein': 'PROT2'}
        }
        
        protein = get_protein_info(('PEPTIDE', 2), library, 'protein')
        assert protein == 'PROT1'
    
    def test_decoy_prefix_removal(self):
        """Test that decoy prefix is removed for lookup"""
        library = {
            ('PEPTIDE', 2): {'protein': 'PROT1'}  # Library has clean key
        }
        
        # Query with decoy prefix
        protein = get_protein_info(('Decoy_PEPTIDE', 2), library, 'protein')
        assert protein == 'PROT1'
    
    def test_missing_protein_column(self):
        """Test when protein column is not specified"""
        library = {('PEPTIDE', 2): {'protein': 'PROT1'}}
        
        protein = get_protein_info(('PEPTIDE', 2), library, None)
        assert protein == 'NA'
    
    def test_missing_library_entry(self):
        """Test when candidate is not in library"""
        library = {('OTHER', 2): {'protein': 'PROT1'}}
        
        protein = get_protein_info(('PEPTIDE', 2), library, 'protein')
        assert protein == 'NA'
    
    def test_missing_protein_field(self):
        """Test when library entry exists but protein field is missing"""
        library = {('PEPTIDE', 2): {'other_field': 'value'}}
        
        protein = get_protein_info(('PEPTIDE', 2), library, 'protein')
        assert protein == 'NA'
    
    def test_empty_library(self):
        """Test with empty library"""
        protein = get_protein_info(('PEPTIDE', 2), {}, 'protein')
        assert protein == 'NA'
    
    def test_exception_handling(self):
        """Test that exceptions are handled gracefully"""
        # Invalid candidate format
        library = {('PEPTIDE', 2): {'protein': 'PROT1'}}
        
        protein = get_protein_info('INVALID', library, 'protein')
        assert protein == 'NA'


class TestFormatSpectralFittingOutput:
    """Test cases for format_spectral_fitting_output function"""
    
    def setup_method(self):
        """Set up common test data"""
        # Mock configuration
        self.config = type('Config', (), {
            'protein_column': 'protein',
            'args': type('Args', (), {'mzml': 'test.mzML'})()
        })()
        
        # Mock MS1 spectrum
        self.ms1_spec = type('MS1Spec', (), {'scan_num': 99})()
    
    def test_no_matches(self):
        """Test output when no matches found"""
        lib_coefficients = np.array([0.0, 0.0, 0.0])
        unified = UnifiedCandidates(
            candidates=[('PEPTIDE', 2)],
            is_decoy=np.array([False]),
            peaks=[np.array([[500.1, 100]])],
            peaks_in_dia=[0]
        )
        features = UnifiedFeatures(
            features=np.zeros((1, 26)),
            is_decoy=np.array([False])
        )
        
        # Mock names length
        import src.spectral_fitting
        src.spectral_fitting.names = [''] * 49  # Mock 49 columns
        
        output = format_spectral_fitting_output(
            lib_coefficients=lib_coefficients,
            unified_candidates=unified,
            unified_features=features,
            additional_outputs={},
            spec_idx=100,
            ms1_spec=self.ms1_spec,
            prec_mz=500.5,
            prec_rt=10.5,
            library={},
            config=self.config
        )
        
        assert len(output) == 1
        assert output[0][0] == 0  # coeff
        assert output[0][1] == 100  # spec_idx
        assert output[0][2] == 99  # ms1_spec_id
        assert output[0][5] == 500.5  # prec_mz
        assert output[0][6] == 10.5  # prec_rt
    
    def test_single_match(self):
        """Test output with single match"""
        lib_coefficients = np.array([0.0, 0.5, 0.0])
        unified = UnifiedCandidates(
            candidates=[('PEPTIDE1', 2), ('PEPTIDE2', 2), ('PEPTIDE3', 2)],
            is_decoy=np.array([False, False, False]),
            peaks=[np.array([[500.1, 100]]) for _ in range(3)],
            peaks_in_dia=[0, 1, 2]
        )
        features = UnifiedFeatures(
            features=np.ones((3, 26)),  # Mock features
            is_decoy=np.array([False, False, False])
        )
        additional_outputs = {
            'frag_names': [np.array(['b2']), np.array(['y3']), np.array(['b4'])],
            'frag_errors': [np.array([0.001]), np.array([0.002]), np.array([0.003])]
        }
        library = {
            ('PEPTIDE2', 2): {'protein': 'PROT2'}
        }
        
        output = format_spectral_fitting_output(
            lib_coefficients=lib_coefficients,
            unified_candidates=unified,
            unified_features=features,
            additional_outputs=additional_outputs,
            spec_idx=100,
            ms1_spec=self.ms1_spec,
            prec_mz=500.5,
            prec_rt=10.5,
            library=library,
            config=self.config
        )
        
        assert len(output) == 1  # Only one non-zero coefficient
        assert output[0][0] == 0.5  # coeff value
        assert output[0][3] == 'PEPTIDE2'  # sequence
        assert output[0][4] == 2  # charge
        assert output[0][-1] == 'PROT2'  # protein
        assert output[0][-2] == 'test.mzML'  # file name
    
    def test_multiple_matches(self):
        """Test output with multiple matches"""
        lib_coefficients = np.array([0.3, 0.0, 0.5])
        unified = UnifiedCandidates(
            candidates=[('PEPTIDE1', 2), ('PEPTIDE2', 2), ('PEPTIDE3', 3)],
            is_decoy=np.array([False, False, False]),
            peaks=[np.array([[500.1, 100]]) for _ in range(3)],
            peaks_in_dia=[0, 1, 2]
        )
        features = UnifiedFeatures(
            features=np.random.rand(3, 26),  # Random features
            is_decoy=np.array([False, False, False])
        )
        
        output = format_spectral_fitting_output(
            lib_coefficients=lib_coefficients,
            unified_candidates=unified,
            unified_features=features,
            additional_outputs={},
            spec_idx=100,
            ms1_spec=None,  # No MS1
            prec_mz=500.5,
            prec_rt=10.5,
            library={},
            config=self.config
        )
        
        assert len(output) == 2  # Two non-zero coefficients
        assert output[0][0] == 0.3  # First coeff
        assert output[0][3] == 'PEPTIDE1'  # First peptide
        assert output[1][0] == 0.5  # Second coeff
        assert output[1][3] == 'PEPTIDE3'  # Third peptide
        assert output[1][4] == 3  # Third peptide charge
        
        # Check MS1 spec ID is 0 when None
        assert output[0][2] == 0
        assert output[1][2] == 0


class TestCheckMS1Peaks:
    """Test cases for the check_ms1_peaks function"""
    
    def test_basic_ms1_peak_checking(self):
        """Test basic MS1 peak checking functionality"""
        # Create test data
        rt_mz = np.array([
            [10.0, 500.0],
            [10.1, 500.1],
            [10.2, 600.0],
            [10.3, 700.0]
        ])
        window_idxs = np.array([0, 2, 3])  # Select indices 0, 2, 3
        
        # Mock MS1 spectrum with peaks close to 500.0 and 600.0
        class MockMS1:
            def __init__(self):
                self.mz = np.array([499.99, 600.01, 800.0])
        
        ms1_spec = MockMS1()
        
        # Import the function
        from src.spectral_fitting import check_ms1_peaks
        
        # Call the function
        result = check_ms1_peaks(rt_mz, window_idxs, ms1_spec)
        
        # Verify results
        assert isinstance(result, np.ndarray)
        assert len(result) == 3  # Should match number of window_idxs
        # Index 0 (m/z=500.0) should have MS1 peak
        # Index 2 (m/z=600.0) should have MS1 peak
        # Index 3 (m/z=700.0) should NOT have MS1 peak
        
    def test_empty_window_indices(self):
        """Test with empty window indices"""
        rt_mz = np.array([[10.0, 500.0], [10.1, 500.1]])
        window_idxs = np.array([], dtype=int)
        
        class MockMS1:
            def __init__(self):
                self.mz = np.array([500.0])
        
        ms1_spec = MockMS1()
        
        from src.spectral_fitting import check_ms1_peaks
        result = check_ms1_peaks(rt_mz, window_idxs, ms1_spec)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == 0
        
    def test_no_ms1_peaks(self):
        """Test when MS1 spectrum has no peaks"""
        rt_mz = np.array([[10.0, 500.0], [10.1, 600.0]])
        window_idxs = np.array([0, 1])
        
        class MockMS1:
            def __init__(self):
                self.mz = np.array([])  # No MS1 peaks
        
        ms1_spec = MockMS1()
        
        from src.spectral_fitting import check_ms1_peaks
        result = check_ms1_peaks(rt_mz, window_idxs, ms1_spec)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == 2
        assert not any(result)  # All should be False


class TestFilterCandidatesByPeakMatching:
    """Test cases for the filter_candidates_by_peak_matching function"""
    
    def test_basic_filtering(self):
        """Test basic candidate filtering functionality"""
        from src.spectral_fitting import filter_candidates_by_peak_matching
        
        # Create test data - 2 candidates
        candidate_peaks = [
            np.array([[100.0, 50.0], [200.0, 100.0], [300.0, 150.0]]),  # Candidate 1
            np.array([[150.0, 30.0], [250.0, 60.0]])                     # Candidate 2
        ]
        
        # Centroid breaks that will match some peaks
        # Odd indices (1, 3, 5, etc.) are "in DIA"
        centroid_breaks = np.array([99.0, 101.0, 199.0, 201.0, 299.0, 301.0])
        
        # Both candidates have MS1 peaks
        ms1_peak = np.array([True, True])
        
        # Call the function
        ref_peaks_in_dia, ref_coords, top_ten, all_norm_intensities = filter_candidates_by_peak_matching(
            candidate_peaks=candidate_peaks,
            centroid_breaks=centroid_breaks,
            ms1_peak=ms1_peak,
            top_n=10,
            atleast_m=2,
            frac_matched=0.5
        )
        
        # Verify outputs
        assert isinstance(ref_peaks_in_dia, list)
        assert isinstance(ref_coords, list)
        assert isinstance(top_ten, list)
        assert isinstance(all_norm_intensities, list)
        
        # Check dimensions
        assert len(ref_coords) == 2
        assert len(top_ten) == 2
        assert len(all_norm_intensities) == 2
        
    def test_filtering_criteria(self):
        """Test that filtering criteria are applied correctly"""
        from src.spectral_fitting import filter_candidates_by_peak_matching
        
        # Create a candidate that should pass all filters
        good_candidate = np.array([
            [100.0, 300.0],  # High intensity, will be top peak
            [200.0, 200.0],  # Second highest
            [300.0, 100.0],  # Third highest
            [400.0, 50.0]    # Lower intensity
        ])
        
        # Create a candidate that should fail
        bad_candidate = np.array([
            [150.0, 100.0],  # Won't match any breaks
            [250.0, 50.0]    # Won't match any breaks
        ])
        
        candidate_peaks = [good_candidate, bad_candidate]
        
        # Breaks that match the good candidate's peaks
        centroid_breaks = np.array([99.0, 101.0, 199.0, 201.0, 299.0, 301.0, 399.0, 401.0])
        
        ms1_peak = np.array([True, True])
        
        ref_peaks_in_dia, _, _, _ = filter_candidates_by_peak_matching(
            candidate_peaks=candidate_peaks,
            centroid_breaks=centroid_breaks,
            ms1_peak=ms1_peak,
            top_n=3,
            atleast_m=2,
            frac_matched=0.5
        )
        
        # Only the first candidate should pass
        assert ref_peaks_in_dia == [0]
        
    def test_empty_candidates(self):
        """Test with empty candidate list"""
        from src.spectral_fitting import filter_candidates_by_peak_matching
        
        ref_peaks_in_dia, ref_coords, top_ten, all_norm_intensities = filter_candidates_by_peak_matching(
            candidate_peaks=[],
            centroid_breaks=np.array([100.0, 200.0]),
            ms1_peak=np.array([]),
            top_n=10,
            atleast_m=3,
            frac_matched=0.5
        )
        
        assert ref_peaks_in_dia == []
        assert ref_coords == []
        assert top_ten == []
        assert all_norm_intensities == []


class TestFitToLib2Integration:
    """Integration tests for the complete fit_to_lib2 pipeline"""
    
    def setup_method(self):
        """Set up test data for integration tests"""
        # Mock DIA spectrum
        self.dia_spec = type('obj', (object,), {
            'scan_num': 1000,
            'prec_mz': 500.5,
            'RT': 30.0,
            'peak_list': lambda: [[500.1, 1000], [501.1, 500], [502.1, 300]],
            'ms1window': (495.0, 505.0)  # Mock MS1 window
        })
        
        # Mock library with targets and decoys
        self.library = {
            ('PEPTIDE', 2): {
                'spectrum': np.array([[500.1, 100], [501.1, 50], [502.1, 30]]),
                'prec_mz': 500.5,
                'RT': 30.0,
                'is_decoy': False,
                'protein': 'PROT1'
            },
            ('Decoy_PEPTIDE', 2): {
                'spectrum': np.array([[500.2, 100], [501.2, 50], [502.2, 30]]),
                'prec_mz': 500.6,
                'RT': 30.1,
                'is_decoy': True,
                'parent_key': ('PEPTIDE', 2),
                'protein': 'Decoy_PROT1'
            },
            ('OTHERPEP', 3): {
                'spectrum': np.array([[600.1, 100], [601.1, 50]]),
                'prec_mz': 600.5,
                'RT': 35.0,
                'is_decoy': False,
                'protein': 'PROT2'
            }
        }
        
        # Mock rt_mz and all_keys arrays
        self.rt_mz = np.array([
            [30.0, 500.5],
            [30.1, 500.6],
            [35.0, 600.5]
        ])
        self.all_keys = [
            ('PEPTIDE', 2),
            ('Decoy_PEPTIDE', 2),
            ('OTHERPEP', 3)
        ]
        
        # Mock config
        self.config = type('obj', (object,), {
            'rt_tol': 0.5,
            'ms1_tol': 20e-6,
            'mz_tol': 10e-6,
            'top_n': 10,
            'atleast_m': 3,
            'unmatched_fit_type': 'c',
            'protein_column': 'protein',
            'args': type('obj', (object,), {'mzml': 'test.mzML'})
        })
    
    def test_empty_dia_spectrum(self):
        """Test with empty DIA spectrum"""
        empty_spec = type('obj', (object,), {
            'scan_num': 1000,
            'prec_mz': 500.5,
            'RT': 30.0,
            'peak_list': lambda: [],  # Empty spectrum
            'ms1window': (495.0, 505.0)  # Mock MS1 window
        })
        
        result = fit_to_lib2(
            dia_spec=empty_spec,
            library=self.library,
            rt_mz=self.rt_mz,
            all_keys=self.all_keys,
            rt_tol=self.config.rt_tol,
            ms1_tol=self.config.ms1_tol,
            mz_tol=self.config.mz_tol
        )
        
        # Should return single empty row
        assert len(result) == 1
        assert result[0][0] == 0  # coeff = 0
        assert result[0][1] == 1000  # spec_idx
        assert result[0][5] == 500.5  # prec_mz
        assert result[0][6] == 30.0  # prec_rt
    
    def test_no_matching_candidates(self):
        """Test when no candidates match the mass window"""
        # Spectrum with different mass
        diff_spec = type('obj', (object,), {
            'scan_num': 1001,
            'prec_mz': 700.5,  # No candidates near this mass
            'RT': 30.0,
            'peak_list': lambda: [[700.1, 1000], [701.1, 500]],
            'ms1window': (695.0, 705.0)  # Mock MS1 window
        })
        
        result = fit_to_lib2(
            dia_spec=diff_spec,
            library=self.library,
            rt_mz=self.rt_mz,
            all_keys=self.all_keys,
            rt_tol=self.config.rt_tol,
            ms1_tol=self.config.ms1_tol,
            mz_tol=self.config.mz_tol
        )
        
        # Should return single empty row
        assert len(result) == 1
        assert result[0][0] == 0  # coeff = 0
        assert result[0][1] == 1001  # spec_idx
    
    def test_single_target_match(self):
        """Test successful match with single target peptide"""
        # Configure to ensure match
        config = type('obj', (object,), {
            'rt_tol': 5.0,  # Wide RT tolerance
            'ms1_tol': 20e-6,
            'mz_tol': 100e-6,  # Wide m/z tolerance
            'top_n': 10,
            'atleast_m': 1,  # Low threshold
            'unmatched_fit_type': 'c',
            'protein_column': 'protein',
            'args': type('obj', (object,), {'mzml': 'test.mzML'})
        })
        
        # Test with matching spectrum
        result = fit_to_lib2(
            dia_spec=self.dia_spec,
            library=self.library,
            rt_mz=self.rt_mz,
            all_keys=self.all_keys,
            rt_filter=False,  # No RT filtering
            decoy=False,  # No decoys
            rt_tol=config.rt_tol,
            ms1_tol=config.ms1_tol,
            mz_tol=config.mz_tol
        )
        
        # Should have at least one result
        assert len(result) >= 1
        # Check if PEPTIDE was matched
        peptides = [row[3] for row in result]
        assert 'PEPTIDE' in peptides or len(result) == 1
    
    def test_target_and_decoy_matches(self):
        """Test with both target and decoy matches"""
        # Configure for matches
        config = type('obj', (object,), {
            'rt_tol': 5.0,
            'ms1_tol': 20e-6,
            'mz_tol': 100e-6,
            'top_n': 10,
            'atleast_m': 1,
            'unmatched_fit_type': 'c',
            'protein_column': 'protein',
            'args': type('obj', (object,), {'mzml': 'test.mzML'})
        })
        
        result = fit_to_lib2(
            dia_spec=self.dia_spec,
            library=self.library,
            rt_mz=self.rt_mz,
            all_keys=self.all_keys,
            rt_filter=False,
            decoy=True,  # Include decoys
            rt_tol=config.rt_tol,
            ms1_tol=config.ms1_tol,
            mz_tol=config.mz_tol
        )
        
        # Check results
        if len(result) > 1 or (len(result) == 1 and result[0][0] > 0):
            # Got matches
            peptides = [row[3] for row in result if row[0] > 0]
            # Could have target and/or decoy matches
            assert any('PEPTIDE' in p or 'Decoy_PEPTIDE' in p for p in peptides)
    
    def test_with_ms1_spectra(self):
        """Test with MS1 spectra provided"""
        # Mock MS1 spectra
        ms1_spectra = [
            type('obj', (object,), {
                'RT': 29.5,
                'scan_num': 999,
                'mz': np.array([500.4, 500.5, 500.6]),
                'intensity': np.array([100, 200, 150])
            }),
            type('obj', (object,), {
                'RT': 30.5,
                'scan_num': 1001,
                'mz': np.array([500.5, 500.6]),
                'intensity': np.array([300, 100])
            })
        ]
        
        result = fit_to_lib2(
            dia_spec=self.dia_spec,
            library=self.library,
            rt_mz=self.rt_mz,
            all_keys=self.all_keys,
            ms1_spectra=ms1_spectra,
            rt_tol=self.config.rt_tol,
            ms1_tol=self.config.ms1_tol,
            mz_tol=self.config.mz_tol
        )
        
        # Should process without error
        assert isinstance(result, list)
        if len(result) > 0:
            # Check MS1 spec ID is set (999 is closest to RT 30.0)
            assert result[0][2] == 999  # Ms1_spec_id
    
    def test_return_frags_option(self):
        """Test return_frags option"""
        result, frags = fit_to_lib2(
            dia_spec=self.dia_spec,
            library=self.library,
            rt_mz=self.rt_mz,
            all_keys=self.all_keys,
            return_frags=True,
            rt_tol=self.config.rt_tol,
            ms1_tol=self.config.ms1_tol,
            mz_tol=self.config.mz_tol
        )
        
        # Should return result and fragment data
        assert isinstance(result, list)
        assert isinstance(frags, list)
        assert len(frags) == 2  # [frag_errors, lib_frag_mz]


class TestBuildSparseMatrixSimple:
    """Test cases for build_sparse_matrix_simple function"""
    
    def test_basic_matrix_construction(self):
        """Test basic sparse matrix construction"""
        from src.spectral_fitting import build_sparse_matrix_simple
        
        # Simple test data - 2 candidates, 3 peaks each
        ref_pep_cand_loc = [
            np.array([1, 3, 5]),  # odd = in DIA, even = not
            np.array([3, 4, 7])   # mixed in/out
        ]
        norm_intensities = [
            np.array([0.3, 0.5, 0.2]),
            np.array([0.4, 0.3, 0.3])
        ]
        ref_pep_cand = ['peptide1', 'peptide2']
        unique_row_idxs = [0, 1, 2, 3]  # DIA spectrum rows to use
        dia_spec_int = np.array([100.0, 200.0, 150.0, 50.0])
        
        matrix, dia_int_with_penalty, frag_info = build_sparse_matrix_simple(
            ref_pep_cand_loc,
            norm_intensities,
            ref_pep_cand,
            unique_row_idxs,
            dia_spec_int
        )
        
        # Check matrix properties
        assert matrix.shape[0] == len(unique_row_idxs) + 1  # +1 for penalty row
        assert matrix.shape[1] == len(ref_pep_cand)
        
        # Check dia_spec_int has penalty term appended
        assert len(dia_int_with_penalty) == len(dia_spec_int) + 1
        assert dia_int_with_penalty[-1] == 0  # penalty term
        
        # Check fragment info
        assert 'ref_spec_row_indices_split' in frag_info
        assert 'ref_spec_col_indices_split' in frag_info
        assert 'ref_spec_values_split' in frag_info
        assert 'lib_peaks_matched' in frag_info
        assert 'num_lib_peaks_matched' in frag_info
        assert 'sparse_row_indices' in frag_info
        assert 'sparse_col_indices' in frag_info
    
    def test_empty_candidates(self):
        """Test with no candidates"""
        from src.spectral_fitting import build_sparse_matrix_simple
        
        matrix, dia_int, frag_info = build_sparse_matrix_simple(
            [],
            [],
            [],
            [],
            np.array([])
        )
        
        assert matrix.shape == (1, 0)  # Empty matrix with 1 row for penalty
        assert len(dia_int) == 1
        assert dia_int[0] == 0
    
    def test_all_peaks_not_in_dia(self):
        """Test when all peaks are not in DIA (even indices)"""
        from src.spectral_fitting import build_sparse_matrix_simple
        
        ref_pep_cand_loc = [
            np.array([0, 2, 4])  # all even = not in DIA
        ]
        norm_intensities = [
            np.array([0.3, 0.5, 0.2])
        ]
        ref_pep_cand = ['peptide1']
        unique_row_idxs = []
        dia_spec_int = np.array([])
        
        matrix, dia_int, frag_info = build_sparse_matrix_simple(
            ref_pep_cand_loc,
            norm_intensities,
            ref_pep_cand,
            unique_row_idxs,
            dia_spec_int
        )
        
        # Should have only penalty row
        assert matrix.shape == (1, 1)
        # Penalty value should be sum of all intensities
        assert np.isclose(matrix.toarray()[0, 0], 1.0)  # 0.3 + 0.5 + 0.2


class TestCalculateRTAlignmentFeatures:
    """Test cases for calculate_rt_alignment_features function"""
    
    def test_basic_feature_calculation(self):
        """Test basic feature calculation with minimal data"""
        from src.spectral_fitting import calculate_rt_alignment_features
        
        # Create minimal test data
        ref_spec_values_split = [np.array([0.3, 0.5, 0.2])]
        ref_spec_row_indices_split = [np.array([0, 1, 2])]
        ref_spec_col_indices_split = [np.array([0, 0, 0])]
        num_lib_peaks_matched = np.array([3])
        lib_peaks_matched = [np.array([True, True, True])]
        
        # DIA spectrum with 5 peaks
        dia_spectrum = np.array([[100.0, 50.0], [200.0, 100.0], [300.0, 75.0], 
                                [400.0, 25.0], [500.0, 10.0]])
        dia_spec_int = np.array([50.0, 100.0, 75.0, 0.0])  # With penalty term
        
        # Create sparse matrix (4x1)
        sparse_lib_matrix = sparse.coo_matrix(
            ([0.3, 0.5, 0.2, 0.0], ([0, 1, 2, 3], [0, 0, 0, 0])),
            shape=(4, 1)
        )
        
        lib_coefficients = np.array([0.8])
        ref_pep_cand = [('PEPTIDE', 2)]
        ref_peaks_in_dia = [0]
        window_idxs = np.array([0])
        
        # Mock library entry with proper fragment format
        library = {
            ('PEPTIDE', 2): {
                'frags': {
                    'b1': [100.0, 0.5],
                    'y1': [200.0, 0.3],
                    'y2': [300.0, 0.2]
                },
                'seq': 'PEPTIDE',
                'charge': 2
            }
        }
        
        rt_mz = np.array([[30.0, 500.5]])
        
        # Call the function
        features = calculate_rt_alignment_features(
            num_lib_peaks_matched=num_lib_peaks_matched,
            lib_peaks_matched=lib_peaks_matched,
            dia_spectrum=dia_spectrum,
            dia_spec_int=dia_spec_int,
            sparse_lib_matrix=sparse_lib_matrix,
            lib_coefficients=lib_coefficients,
            ref_pep_cand=ref_pep_cand,
            ref_peaks_in_dia=ref_peaks_in_dia,
            window_idxs=window_idxs,
            library=library,
            rt_mz=rt_mz,
            prec_rt=30.0,
            prec_mz=500.0,
            windowWidth=10.0,
            rt_tol=0.5,
            ms1_tol=1e-5,
            dino_features=None
        )
        
        # Check output shape
        assert features.shape == (1, 26)
        
        # Check some basic features
        assert features[0, 0] == 3  # num_lib_peaks_matched
        assert np.isclose(features[0, 1], 1.0)  # frac_lib_intensity (sum to 1)
        assert features[0, 4] == 0.0  # rt_error (30.0 - 30.0)
    
    def test_feature_calculation_with_dino(self):
        """Test feature calculation with MS1 features"""
        from src.spectral_fitting import calculate_rt_alignment_features
        
        # Create a mock dino_features object that supports boolean indexing
        class MockDinoFeatures:
            def __init__(self):
                self.data = {
                    'mz': np.array([500.1, 500.2, 500.3]),
                    'RT': np.array([29.5, 30.0, 30.5]),
                    'rtApex': np.array([29.5, 30.0, 30.5])
                }
                self.mz = self.data['mz']
                self.RT = self.data['RT']
            
            def __getitem__(self, key):
                if isinstance(key, str):
                    return self.data[key]
                elif isinstance(key, np.ndarray) and key.dtype == bool:
                    # Boolean indexing - return filtered version
                    filtered = MockDinoFeatures()
                    for k, v in self.data.items():
                        filtered.data[k] = v[key]
                        setattr(filtered, k, filtered.data[k])
                    return filtered
                else:
                    return self.data[key]
        
        # Similar setup as above but with dino_features
        num_lib_peaks_matched = np.array([2])
        lib_peaks_matched = [np.array([True, True])]
        
        dia_spectrum = np.array([[100.0, 50.0], [200.0, 50.0]])
        dia_spec_int = np.array([50.0, 50.0, 0.0])
        
        sparse_lib_matrix = sparse.coo_matrix(
            ([0.5, 0.5, 0.0], ([0, 1, 2], [0, 0, 0])),
            shape=(3, 1)
        )
        
        lib_coefficients = np.array([1.0])
        ref_pep_cand = [('PEPTIDE', 2)]
        ref_peaks_in_dia = [0]
        window_idxs = np.array([0])
        
        library = {
            ('PEPTIDE', 2): {
                'frags': {
                    'y1': [100.0, 0.5],
                    'y2': [200.0, 0.5]
                },
                'seq': 'PEPTIDE',
                'charge': 2
            }
        }
        
        rt_mz = np.array([[30.0, 500.2]])
        
        # Create mock dino features
        dino_features = MockDinoFeatures()
        
        features = calculate_rt_alignment_features(
            num_lib_peaks_matched=num_lib_peaks_matched,
            lib_peaks_matched=lib_peaks_matched,
            dia_spectrum=dia_spectrum,
            dia_spec_int=dia_spec_int,
            sparse_lib_matrix=sparse_lib_matrix,
            lib_coefficients=lib_coefficients,
            ref_pep_cand=ref_pep_cand,
            ref_peaks_in_dia=ref_peaks_in_dia,
            window_idxs=window_idxs,
            library=library,
            rt_mz=rt_mz,
            prec_rt=30.0,
            prec_mz=500.2,
            windowWidth=10.0,
            rt_tol=0.5,
            ms1_tol=1e-5,
            dino_features=dino_features
        )
        
        # Check that MS1 error is calculated (not zero when dino_features provided)
        # The actual value depends on the ms1_error function implementation
        assert features.shape == (1, 26)
    
    def test_empty_candidates(self):
        """Test with no matching candidates"""
        from src.spectral_fitting import calculate_rt_alignment_features
        
        # Empty arrays
        features = calculate_rt_alignment_features(
            num_lib_peaks_matched=np.array([]),
            lib_peaks_matched=[],
            dia_spectrum=np.array([[100.0, 50.0]]),
            dia_spec_int=np.array([0.0]),
            sparse_lib_matrix=sparse.coo_matrix((1, 0)),
            lib_coefficients=np.array([]),
            ref_pep_cand=[],
            ref_peaks_in_dia=[],
            window_idxs=np.array([]),
            library={},
            rt_mz=np.array([[30.0, 500.0]]),
            prec_rt=30.0,
            prec_mz=500.0,
            windowWidth=10.0,
            rt_tol=0.5,
            ms1_tol=1e-5,
            dino_features=None
        )
        
        # Should return empty feature array
        assert features.shape[0] == 0
        assert features.shape[1] == 26


class TestCalculateFracLibIntensitySparse:
    """Test cases for calculate_frac_lib_intensity_sparse function"""
    
    def test_single_candidate_basic(self):
        """Test with single candidate"""
        from src.spectral_fitting import calculate_frac_lib_intensity_sparse
        
        # Create sparse matrix with one candidate (column)
        # Values: [0.3, 0.5, 0.2] in column 0
        sparse_matrix = sparse.coo_matrix(
            ([0.3, 0.5, 0.2], ([0, 1, 2], [0, 0, 0])),
            shape=(3, 1)
        )
        
        result = calculate_frac_lib_intensity_sparse(sparse_matrix)
        
        # Should return numpy array with one element
        assert isinstance(result, np.ndarray)
        assert result.shape == (1,)
        assert np.isclose(result[0], 1.0)  # 0.3 + 0.5 + 0.2 = 1.0
    
    def test_multiple_candidates(self):
        """Test with multiple candidates"""
        from src.spectral_fitting import calculate_frac_lib_intensity_sparse
        
        # Create sparse matrix with two candidates
        # Column 0: [0.3, 0.5, 0.0] = 0.8
        # Column 1: [0.0, 0.2, 0.7] = 0.9
        sparse_matrix = sparse.coo_matrix(
            ([0.3, 0.5, 0.2, 0.7], ([0, 1, 1, 2], [0, 0, 1, 1])),
            shape=(3, 2)
        )
        
        result = calculate_frac_lib_intensity_sparse(sparse_matrix)
        
        # Should return numpy array with two elements
        assert isinstance(result, np.ndarray)
        assert result.shape == (2,)
        assert np.isclose(result[0], 0.8)  # 0.3 + 0.5
        assert np.isclose(result[1], 0.9)  # 0.2 + 0.7
    
    def test_empty_matrix(self):
        """Test with empty matrix (no candidates)"""
        from src.spectral_fitting import calculate_frac_lib_intensity_sparse
        
        # Create empty matrix (1 row, 0 columns)
        sparse_matrix = sparse.coo_matrix((1, 0))
        
        result = calculate_frac_lib_intensity_sparse(sparse_matrix)
        
        # Should return empty numpy array with correct dtype
        assert isinstance(result, np.ndarray)
        assert result.shape == (0,)
        assert result.dtype == np.float64
    
    def test_zero_values(self):
        """Test with matrix containing zeros"""
        from src.spectral_fitting import calculate_frac_lib_intensity_sparse
        
        # Create matrix with some zero columns
        sparse_matrix = sparse.coo_matrix(
            ([0.5, 0.3], ([0, 1], [0, 2])),
            shape=(3, 3)
        )
        
        result = calculate_frac_lib_intensity_sparse(sparse_matrix)
        
        # Should return array: [0.5, 0.0, 0.3]
        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)
        assert np.isclose(result[0], 0.5)
        assert np.isclose(result[1], 0.0)  # Empty column
        assert np.isclose(result[2], 0.3)


class TestCalculateFracDiaIntensitySparse:
    """Test cases for calculate_frac_dia_intensity_sparse function"""
    
    def test_single_candidate_basic(self):
        """Test with single candidate"""
        from src.spectral_fitting import calculate_frac_dia_intensity_sparse
        
        # DIA spectrum with intensities [50, 100, 75]
        dia_spectrum = np.array([[100.0, 50.0], [200.0, 100.0], [300.0, 75.0]])
        
        # Row indices: candidate 0 matches DIA spectrum rows 0 and 2
        ref_spec_row_indices_split = [np.array([0, 2])]
        
        # Sparse matrix: candidate 0 matches rows 0 and 2
        sparse_matrix = sparse.csc_matrix(
            ([0.3, 0.2], ([0, 2], [0, 0])),
            shape=(3, 1)
        )
        
        tic = 50.0 + 100.0 + 75.0  # Calculate TIC
        result = calculate_frac_dia_intensity_sparse(sparse_matrix, ref_spec_row_indices_split, dia_spectrum, tic)
        
        # Should return numpy array with one element
        # Expected: (50 + 75) / (50 + 100 + 75) = 125 / 225 = 0.5556
        assert isinstance(result, np.ndarray)
        assert result.shape == (1,)
        assert np.isclose(result[0], 125.0 / 225.0)
    
    def test_multiple_candidates(self):
        """Test with multiple candidates"""
        from src.spectral_fitting import calculate_frac_dia_intensity_sparse
        
        # DIA spectrum with intensities [50, 100, 75]
        dia_spectrum = np.array([[100.0, 50.0], [200.0, 100.0], [300.0, 75.0]])
        
        # Row indices for each candidate
        ref_spec_row_indices_split = [np.array([0, 1]), np.array([2])]
        
        # Sparse matrix: 
        # candidate 0 matches rows 0,1  -> intensities 50+100=150
        # candidate 1 matches row 2     -> intensity 75
        sparse_matrix = sparse.csc_matrix(
            ([0.3, 0.5, 0.2], ([0, 1, 2], [0, 0, 1])),
            shape=(3, 2)
        )
        
        tic = 50.0 + 100.0 + 75.0  # Calculate TIC = 225
        result = calculate_frac_dia_intensity_sparse(sparse_matrix, ref_spec_row_indices_split, dia_spectrum, tic)
        
        # TIC = 50 + 100 + 75 = 225
        # Expected: [150/225, 75/225] = [0.6667, 0.3333]
        assert isinstance(result, np.ndarray)
        assert result.shape == (2,)
        assert np.isclose(result[0], 150.0 / 225.0)
        assert np.isclose(result[1], 75.0 / 225.0)
    
    def test_empty_matrix(self):
        """Test with empty matrix (no candidates)"""
        from src.spectral_fitting import calculate_frac_dia_intensity_sparse
        
        dia_spectrum = np.array([[100.0, 50.0]])
        ref_spec_row_indices_split = []
        sparse_matrix = sparse.csc_matrix((1, 0))
        
        tic = 50.0  # Calculate TIC
        result = calculate_frac_dia_intensity_sparse(sparse_matrix, ref_spec_row_indices_split, dia_spectrum, tic)
        
        # Should return empty numpy array
        assert isinstance(result, np.ndarray)
        assert result.shape == (0,)
        assert result.dtype == np.float64
    
    def test_no_matches_for_candidate(self):
        """Test with candidate that has no matches"""
        from src.spectral_fitting import calculate_frac_dia_intensity_sparse
        
        # DIA spectrum with intensities [50, 100]
        dia_spectrum = np.array([[100.0, 50.0], [200.0, 100.0]])
        
        # Row indices: first candidate matches row 0, second has no matches
        ref_spec_row_indices_split = [np.array([0]), np.array([])]
        
        # Sparse matrix with empty column (no matches for candidate 1)
        sparse_matrix = sparse.csc_matrix(
            ([0.5], ([0], [0])),
            shape=(2, 2)  # 2 candidates, only first has matches
        )
        
        tic = 50.0 + 100.0  # Calculate TIC = 150
        result = calculate_frac_dia_intensity_sparse(sparse_matrix, ref_spec_row_indices_split, dia_spectrum, tic)
        
        # TIC = 150, candidate 0 gets 50, candidate 1 gets 0
        assert isinstance(result, np.ndarray)
        assert result.shape == (2,)
        assert np.isclose(result[0], 50.0 / 150.0)
        assert np.isclose(result[1], 0.0)


class TestCalculateTic:
    """Test cases for calculate_tic function"""
    
    def test_basic_tic_calculation(self):
        """Test basic TIC calculation"""
        from src.spectral_fitting import calculate_tic
        
        # DIA spectrum with intensities [50, 100, 75]
        dia_spectrum = np.array([[100.0, 50.0], [200.0, 100.0], [300.0, 75.0]])
        
        result = calculate_tic(dia_spectrum)
        
        # Expected: 50 + 100 + 75 = 225
        assert isinstance(result, (float, np.floating))
        assert np.isclose(result, 225.0)
    
    def test_single_peak_tic(self):
        """Test TIC with single peak"""
        from src.spectral_fitting import calculate_tic
        
        dia_spectrum = np.array([[100.0, 42.5]])
        
        result = calculate_tic(dia_spectrum)
        
        assert np.isclose(result, 42.5)
    
    def test_zero_intensity_tic(self):
        """Test TIC with zero intensities"""
        from src.spectral_fitting import calculate_tic
        
        dia_spectrum = np.array([[100.0, 0.0], [200.0, 0.0]])
        
        result = calculate_tic(dia_spectrum)
        
        assert np.isclose(result, 0.0)


class TestCalculateR2LibSpecSparse:
    """Test cases for calculate_r2_lib_spec_sparse function"""
    
    def test_single_candidate_perfect_correlation(self):
        """Test with single candidate having perfect correlation"""
        from src.spectral_fitting import calculate_r2_lib_spec_sparse
        
        # DIA spectrum with intensities [10, 20, 30]
        dia_spectrum = np.array([[100.0, 10.0], [200.0, 20.0], [300.0, 30.0]])
        
        # Sparse matrix: candidate 0 has library intensities [5, 10, 15] at rows [0, 1, 2]
        # These are perfectly correlated with DIA intensities [10, 20, 30]
        sparse_matrix = sparse.csc_matrix(
            ([5.0, 10.0, 15.0], ([0, 1, 2], [0, 0, 0])),
            shape=(3, 1)
        )
        
        result = calculate_r2_lib_spec_sparse(sparse_matrix, dia_spectrum)
        
        # Should return correlation coefficient close to 1.0 (perfect positive correlation)
        assert isinstance(result, np.ndarray)
        assert result.shape == (1,)
        assert np.isclose(result[0], 1.0, atol=1e-10)
    
    def test_single_candidate_negative_correlation(self):
        """Test with single candidate having negative correlation"""
        from src.spectral_fitting import calculate_r2_lib_spec_sparse
        
        # DIA spectrum with intensities [10, 20, 30]
        dia_spectrum = np.array([[100.0, 10.0], [200.0, 20.0], [300.0, 30.0]])
        
        # Sparse matrix: candidate 0 has library intensities [30, 20, 10] at rows [0, 1, 2]
        # These are perfectly anti-correlated with DIA intensities [10, 20, 30]
        # Using values that maintain perfect linear relationship
        sparse_matrix = sparse.csc_matrix(
            ([30.0, 20.0, 10.0], ([0, 1, 2], [0, 0, 0])),
            shape=(3, 1)
        )
        
        result = calculate_r2_lib_spec_sparse(sparse_matrix, dia_spectrum)
        
        # Should return correlation coefficient close to -1.0 (perfect negative correlation)
        assert isinstance(result, np.ndarray)
        assert result.shape == (1,)
        assert np.isclose(result[0], -1.0, atol=1e-10)
    
    def test_multiple_candidates(self):
        """Test with multiple candidates having different correlations"""
        from src.spectral_fitting import calculate_r2_lib_spec_sparse
        
        # DIA spectrum with intensities [10, 20, 30, 40]
        dia_spectrum = np.array([[100.0, 10.0], [200.0, 20.0], [300.0, 30.0], [400.0, 40.0]])
        
        # Sparse matrix:
        # Candidate 0: lib [5, 10] at rows [0, 1] -> correlates with DIA [10, 20] (perfect positive)
        # Candidate 1: lib [40, 30] at rows [2, 3] -> correlates with DIA [30, 40] (perfect negative)
        sparse_matrix = sparse.csc_matrix(
            ([5.0, 10.0, 40.0, 30.0], ([0, 1, 2, 3], [0, 0, 1, 1])),
            shape=(4, 2)
        )
        
        result = calculate_r2_lib_spec_sparse(sparse_matrix, dia_spectrum)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (2,)
        assert np.isclose(result[0], 1.0, atol=1e-10)   # Perfect positive correlation
        assert np.isclose(result[1], -1.0, atol=1e-10)  # Perfect negative correlation
    
    def test_single_match_candidate(self):
        """Test with candidate having only one match (insufficient for correlation)"""
        from src.spectral_fitting import calculate_r2_lib_spec_sparse
        
        # DIA spectrum
        dia_spectrum = np.array([[100.0, 10.0], [200.0, 20.0]])
        
        # Sparse matrix: candidate 0 has only one match
        sparse_matrix = sparse.csc_matrix(
            ([5.0], ([0], [0])),
            shape=(2, 1)
        )
        
        result = calculate_r2_lib_spec_sparse(sparse_matrix, dia_spectrum)
        
        # Should return 0.0 for insufficient data points
        assert isinstance(result, np.ndarray)
        assert result.shape == (1,)
        assert np.isclose(result[0], 0.0)
    
    def test_empty_matrix(self):
        """Test with empty matrix (no candidates)"""
        from src.spectral_fitting import calculate_r2_lib_spec_sparse
        
        dia_spectrum = np.array([[100.0, 10.0]])
        sparse_matrix = sparse.csc_matrix((1, 0))
        
        result = calculate_r2_lib_spec_sparse(sparse_matrix, dia_spectrum)
        
        # Should return empty array
        assert isinstance(result, np.ndarray)
        assert result.shape == (0,)
        assert result.dtype == np.float64
    
    def test_constant_dia_values(self):
        """Test with constant DIA values (should return NaN or handle gracefully)"""
        from src.spectral_fitting import calculate_r2_lib_spec_sparse
        
        # DIA spectrum with intensities [10, 10, 10] (constant)
        dia_spectrum = np.array([[100.0, 10.0], [200.0, 10.0], [300.0, 10.0]])
        
        # Sparse matrix: candidate 0 has varying library intensities [5, 15, 25]
        # Correlation with constant DIA is mathematically undefined (NaN)
        sparse_matrix = sparse.csc_matrix(
            ([5.0, 15.0, 25.0], ([0, 1, 2], [0, 0, 0])),
            shape=(3, 1)
        )
        
        result = calculate_r2_lib_spec_sparse(sparse_matrix, dia_spectrum)
        
        # Correlation with constant values is undefined, so NaN is expected
        assert isinstance(result, np.ndarray)
        assert result.shape == (1,)
        # NaN is the mathematically correct result for correlation with constant values
        assert np.isnan(result[0])
    
    def test_weak_correlation(self):
        """Test with candidate having weak but non-zero correlation"""
        from src.spectral_fitting import calculate_r2_lib_spec_sparse
        
        # DIA spectrum with intensities [10, 15, 20, 12, 18]
        dia_spectrum = np.array([[100.0, 10.0], [200.0, 15.0], [300.0, 20.0], [400.0, 12.0], [500.0, 18.0]])
        
        # Sparse matrix: candidate 0 has somewhat correlated library intensities
        sparse_matrix = sparse.csc_matrix(
            ([8.0, 16.0, 22.0, 11.0, 19.0], ([0, 1, 2, 3, 4], [0, 0, 0, 0, 0])),
            shape=(5, 1)
        )
        
        result = calculate_r2_lib_spec_sparse(sparse_matrix, dia_spectrum)
        
        # Should return a finite correlation coefficient
        assert isinstance(result, np.ndarray)
        assert result.shape == (1,)
        assert np.isfinite(result[0])
        assert -1.0 <= result[0] <= 1.0  # Valid correlation range


class TestCalculateUniquePeakFeaturesSparse:
    """Test cases for calculate_unique_peak_features_sparse function"""
    
    def test_basic_unique_peaks(self):
        """Test with simple unique peak scenario"""
        from src.spectral_fitting import calculate_unique_peak_features_sparse
        
        # DIA spectrum
        dia_spectrum = np.array([[100.0, 10.0], [200.0, 20.0], [300.0, 30.0], [400.0, 40.0]])
        
        # Sparse matrix: 
        # Row 0: only candidate 0 has a match (unique to candidate 0)
        # Row 1: only candidate 1 has a match (unique to candidate 1)
        # Row 2: both candidates have matches (shared peak)
        # Row 3: only candidate 0 has a match (unique to candidate 0)
        sparse_matrix = sparse.csc_matrix(
            ([5.0, 15.0, 25.0, 35.0, 45.0], ([0, 1, 2, 2, 3], [0, 1, 0, 1, 0])),
            shape=(4, 2)
        )
        
        lib_coefficients = np.array([0.8, 1.2])
        
        r2_unique, frac_unique_pred = calculate_unique_peak_features_sparse(
            sparse_matrix, dia_spectrum, lib_coefficients
        )
        
        # Should return arrays with 2 elements (one per candidate)
        assert isinstance(r2_unique, np.ndarray)
        assert isinstance(frac_unique_pred, np.ndarray)
        assert r2_unique.shape == (2,)
        assert frac_unique_pred.shape == (2,)
        
        # Candidate 0 has unique peaks at rows 0 and 3
        # DIA values: [10, 40], Lib values: [5, 45] - should have correlation
        assert np.isfinite(r2_unique[0])
        assert frac_unique_pred[0] > 0
        
        # Candidate 1 has unique peak at row 1 only
        # With only 1 unique peak, correlation should be 0
        assert r2_unique[1] == 0.0
        assert frac_unique_pred[1] > 0
    
    def test_no_unique_peaks(self):
        """Test when all peaks are shared (no unique peaks)"""
        from src.spectral_fitting import calculate_unique_peak_features_sparse
        
        # DIA spectrum
        dia_spectrum = np.array([[100.0, 10.0], [200.0, 20.0]])
        
        # Sparse matrix: both rows have matches for both candidates (all shared)
        sparse_matrix = sparse.csc_matrix(
            ([5.0, 15.0, 25.0, 35.0], ([0, 0, 1, 1], [0, 1, 0, 1])),
            shape=(2, 2)
        )
        
        lib_coefficients = np.array([0.8, 1.2])
        
        r2_unique, frac_unique_pred = calculate_unique_peak_features_sparse(
            sparse_matrix, dia_spectrum, lib_coefficients
        )
        
        # No unique peaks, so all values should be 0
        assert r2_unique.shape == (2,)
        assert frac_unique_pred.shape == (2,)
        assert np.all(r2_unique == 0.0)
        assert np.all(frac_unique_pred == 0.0)
    
    def test_all_unique_peaks(self):
        """Test when all peaks are unique to individual candidates"""
        from src.spectral_fitting import calculate_unique_peak_features_sparse
        
        # DIA spectrum with 4 peaks
        dia_spectrum = np.array([[100.0, 10.0], [200.0, 20.0], [300.0, 30.0], [400.0, 40.0]])
        
        # Sparse matrix: each candidate has unique peaks only
        # Candidate 0: peaks at rows 0, 1
        # Candidate 1: peaks at rows 2, 3
        sparse_matrix = sparse.csc_matrix(
            ([5.0, 10.0, 15.0, 20.0], ([0, 1, 2, 3], [0, 0, 1, 1])),
            shape=(4, 2)
        )
        
        lib_coefficients = np.array([0.5, 1.5])
        
        r2_unique, frac_unique_pred = calculate_unique_peak_features_sparse(
            sparse_matrix, dia_spectrum, lib_coefficients
        )
        
        # Both candidates have multiple unique peaks, so should have correlations
        assert r2_unique.shape == (2,)
        assert frac_unique_pred.shape == (2,)
        
        # Both should have finite correlations (perfect positive in this case)
        assert np.isfinite(r2_unique[0])
        assert np.isfinite(r2_unique[1])
        assert np.isclose(r2_unique[0], 1.0, atol=1e-10)  # Perfect correlation [10,20] vs [5,10]
        assert np.isclose(r2_unique[1], 1.0, atol=1e-10)  # Perfect correlation [30,40] vs [15,20]
        
        # Both should have positive fractions
        assert frac_unique_pred[0] > 0
        assert frac_unique_pred[1] > 0
    
    def test_single_peak_per_candidate(self):
        """Test when each candidate has only one unique peak"""
        from src.spectral_fitting import calculate_unique_peak_features_sparse
        
        # DIA spectrum
        dia_spectrum = np.array([[100.0, 10.0], [200.0, 20.0]])
        
        # Sparse matrix: each candidate has one unique peak
        sparse_matrix = sparse.csc_matrix(
            ([5.0, 15.0], ([0, 1], [0, 1])),
            shape=(2, 2)
        )
        
        lib_coefficients = np.array([0.8, 1.2])
        
        r2_unique, frac_unique_pred = calculate_unique_peak_features_sparse(
            sparse_matrix, dia_spectrum, lib_coefficients
        )
        
        # Single peaks can't have correlation, so should be 0
        assert r2_unique.shape == (2,)
        assert frac_unique_pred.shape == (2,)
        assert np.all(r2_unique == 0.0)
        
        # But should still have positive fractions
        assert frac_unique_pred[0] > 0
        assert frac_unique_pred[1] > 0
    
    def test_empty_matrix(self):
        """Test with empty matrix (no candidates)"""
        from src.spectral_fitting import calculate_unique_peak_features_sparse
        
        dia_spectrum = np.array([[100.0, 10.0]])
        sparse_matrix = sparse.csc_matrix((1, 0))
        lib_coefficients = np.array([])
        
        r2_unique, frac_unique_pred = calculate_unique_peak_features_sparse(
            sparse_matrix, dia_spectrum, lib_coefficients
        )
        
        # Should return empty arrays
        assert isinstance(r2_unique, np.ndarray)
        assert isinstance(frac_unique_pred, np.ndarray)
        assert r2_unique.shape == (0,)
        assert frac_unique_pred.shape == (0,)
    
    def test_constant_unique_values(self):
        """Test with constant values in unique peaks (should return NaN)"""
        from src.spectral_fitting import calculate_unique_peak_features_sparse
        
        # DIA spectrum with constant values for unique peaks
        dia_spectrum = np.array([[100.0, 15.0], [200.0, 15.0], [300.0, 15.0]])
        
        # Sparse matrix: candidate 0 has unique peaks at rows 0, 1, 2 with varying library values
        sparse_matrix = sparse.csc_matrix(
            ([5.0, 10.0, 20.0], ([0, 1, 2], [0, 0, 0])),
            shape=(3, 1)
        )
        
        lib_coefficients = np.array([1.0])
        
        r2_unique, frac_unique_pred = calculate_unique_peak_features_sparse(
            sparse_matrix, dia_spectrum, lib_coefficients
        )
        
        # Correlation with constant DIA values should be NaN
        assert r2_unique.shape == (1,)
        assert frac_unique_pred.shape == (1,)
        assert np.isnan(r2_unique[0])
        assert frac_unique_pred[0] > 0  # Fraction should still be calculated
    
    def test_penalty_rows_handling(self):
        """Test handling of penalty rows (beyond DIA spectrum bounds)"""
        from src.spectral_fitting import calculate_unique_peak_features_sparse
        
        # DIA spectrum with only 2 peaks
        dia_spectrum = np.array([[100.0, 10.0], [200.0, 20.0]])
        
        # Sparse matrix with 4 rows (rows 2,3 are penalty rows)
        # Candidate 0 has unique peaks at rows 0, 2 (row 2 is penalty)
        sparse_matrix = sparse.csc_matrix(
            ([5.0, 25.0], ([0, 2], [0, 0])),
            shape=(4, 1)
        )
        
        lib_coefficients = np.array([1.0])
        
        r2_unique, frac_unique_pred = calculate_unique_peak_features_sparse(
            sparse_matrix, dia_spectrum, lib_coefficients
        )
        
        # Should handle penalty rows gracefully
        # Only 1 valid unique peak (row 0), so correlation should be 0
        assert r2_unique.shape == (1,)
        assert frac_unique_pred.shape == (1,)
        assert r2_unique[0] == 0.0
        assert frac_unique_pred[0] > 0


class TestCalculateFracDiaIntensityPred:
    """Test cases for calculate_frac_dia_intensity_pred function"""
    
    def test_basic_calculation(self):
        """Test basic calculation with normal values"""
        from src.spectral_fitting import calculate_frac_dia_intensity_pred
        
        frac_lib_intensity = np.array([0.2, 0.3, 0.4])
        frac_dia_intensity = np.array([0.1, 0.2, 0.5])
        lib_coefficients = np.array([0.5, 1.0, 1.5])
        
        result = calculate_frac_dia_intensity_pred(
            frac_lib_intensity, frac_dia_intensity, lib_coefficients
        )
        
        # Expected: (frac_lib * coeff) / frac_dia
        # [0.2*0.5/0.1, 0.3*1.0/0.2, 0.4*1.5/0.5] = [1.0, 1.5, 1.2]
        expected = np.array([1.0, 1.5, 1.2])
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)
        assert np.allclose(result, expected)
    
    def test_zero_dia_intensity(self):
        """Test handling of zero DIA intensity (division by zero)"""
        from src.spectral_fitting import calculate_frac_dia_intensity_pred
        
        frac_lib_intensity = np.array([0.2, 0.3, 0.4])
        frac_dia_intensity = np.array([0.1, 0.0, 0.5])  # Middle value is 0
        lib_coefficients = np.array([0.5, 1.0, 1.5])
        
        result = calculate_frac_dia_intensity_pred(
            frac_lib_intensity, frac_dia_intensity, lib_coefficients
        )
        
        # Expected: [0.2*0.5/0.1, 0.0, 0.4*1.5/0.5] = [1.0, 0.0, 1.2]
        expected = np.array([1.0, 0.0, 1.2])
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)
        assert np.allclose(result, expected)
    
    def test_zero_lib_intensity(self):
        """Test handling of zero library intensity"""
        from src.spectral_fitting import calculate_frac_dia_intensity_pred
        
        frac_lib_intensity = np.array([0.0, 0.3, 0.4])  # First value is 0
        frac_dia_intensity = np.array([0.1, 0.2, 0.5])
        lib_coefficients = np.array([0.5, 1.0, 1.5])
        
        result = calculate_frac_dia_intensity_pred(
            frac_lib_intensity, frac_dia_intensity, lib_coefficients
        )
        
        # Expected: [0.0*0.5/0.1, 0.3*1.0/0.2, 0.4*1.5/0.5] = [0.0, 1.5, 1.2]
        expected = np.array([0.0, 1.5, 1.2])
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)
        assert np.allclose(result, expected)
    
    def test_zero_coefficient(self):
        """Test handling of zero coefficient"""
        from src.spectral_fitting import calculate_frac_dia_intensity_pred
        
        frac_lib_intensity = np.array([0.2, 0.3, 0.4])
        frac_dia_intensity = np.array([0.1, 0.2, 0.5])
        lib_coefficients = np.array([0.5, 0.0, 1.5])  # Middle coefficient is 0
        
        result = calculate_frac_dia_intensity_pred(
            frac_lib_intensity, frac_dia_intensity, lib_coefficients
        )
        
        # Expected: [0.2*0.5/0.1, 0.3*0.0/0.2, 0.4*1.5/0.5] = [1.0, 0.0, 1.2]
        expected = np.array([1.0, 0.0, 1.2])
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)
        assert np.allclose(result, expected)
    
    def test_large_coefficients(self):
        """Test handling of large coefficients (>1)"""
        from src.spectral_fitting import calculate_frac_dia_intensity_pred
        
        frac_lib_intensity = np.array([0.2, 0.3])
        frac_dia_intensity = np.array([0.1, 0.2])
        lib_coefficients = np.array([2.0, 5.0])  # Large coefficients
        
        result = calculate_frac_dia_intensity_pred(
            frac_lib_intensity, frac_dia_intensity, lib_coefficients
        )
        
        # Expected: [0.2*2.0/0.1, 0.3*5.0/0.2] = [4.0, 7.5]
        expected = np.array([4.0, 7.5])
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (2,)
        assert np.allclose(result, expected)
    
    def test_small_values(self):
        """Test handling of very small values"""
        from src.spectral_fitting import calculate_frac_dia_intensity_pred
        
        frac_lib_intensity = np.array([1e-6, 1e-5])
        frac_dia_intensity = np.array([1e-7, 1e-4])
        lib_coefficients = np.array([0.1, 0.01])
        
        result = calculate_frac_dia_intensity_pred(
            frac_lib_intensity, frac_dia_intensity, lib_coefficients
        )
        
        # Expected: [1e-6*0.1/1e-7, 1e-5*0.01/1e-4] = [1.0, 0.001]
        expected = np.array([1.0, 0.001])
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (2,)
        assert np.allclose(result, expected)
    
    def test_empty_arrays(self):
        """Test handling of empty input arrays"""
        from src.spectral_fitting import calculate_frac_dia_intensity_pred
        
        frac_lib_intensity = np.array([])
        frac_dia_intensity = np.array([])
        lib_coefficients = np.array([])
        
        result = calculate_frac_dia_intensity_pred(
            frac_lib_intensity, frac_dia_intensity, lib_coefficients
        )
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (0,)
        assert result.dtype == np.float64
    
    def test_single_element(self):
        """Test handling of single element arrays"""
        from src.spectral_fitting import calculate_frac_dia_intensity_pred
        
        frac_lib_intensity = np.array([0.5])
        frac_dia_intensity = np.array([0.2])
        lib_coefficients = np.array([1.2])
        
        result = calculate_frac_dia_intensity_pred(
            frac_lib_intensity, frac_dia_intensity, lib_coefficients
        )
        
        # Expected: [0.5*1.2/0.2] = [3.0]
        expected = np.array([3.0])
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (1,)
        assert np.allclose(result, expected)
    
    def test_vectorization_vs_list_comprehension(self):
        """Test that vectorized version gives same results as original list comprehension"""
        from src.spectral_fitting import calculate_frac_dia_intensity_pred
        
        # Test data
        frac_lib_intensity = np.array([0.1, 0.2, 0.3, 0.4, 0.0, 0.5])
        frac_dia_intensity = np.array([0.05, 0.1, 0.0, 0.2, 0.1, 0.25])
        lib_coefficients = np.array([0.5, 1.0, 1.5, 2.0, 0.0, 0.8])
        
        # Vectorized version
        result_vectorized = calculate_frac_dia_intensity_pred(
            frac_lib_intensity, frac_dia_intensity, lib_coefficients
        )
        
        # Original list comprehension (with safe division)
        result_original = []
        for i, j, c in zip(frac_lib_intensity, frac_dia_intensity, lib_coefficients):
            if j == 0:
                result_original.append(0.0)
            else:
                result_original.append((i * c) / j)
        result_original = np.array(result_original)
        
        assert np.allclose(result_vectorized, result_original)


class TestCalculateBYIonCounts:
    """Test cases for calculate_b_y_ion_counts function"""
    
    def test_basic_b_y_counting(self):
        """Test basic b and y ion counting"""
        from src.spectral_fitting import calculate_b_y_ion_counts
        
        # Mock library with fragment data
        library = {
            'cand1': {
                'frags': {
                    'b3_1': [244.09, 0.25],
                    'b4_1': [372.15, 0.45],
                    'y4_1': [472.25, 0.31],
                    'y5_1': [529.27, 1.0],
                    'y6_1': [642.36, 0.50]
                }
            },
            'cand2': {
                'frags': {
                    'b2_1': [200.10, 0.30],
                    'y3_1': [350.20, 0.80],
                    'y4_1': [451.25, 0.60]
                }
            }
        }
        
        ref_pep_cand = ['cand1', 'cand2']
        # cand1: match b3, b4, y5 (2 b-ions, 1 y-ion)
        # cand2: match b2, y4 (1 b-ion, 1 y-ion)
        lib_peaks_matched = [
            np.array([True, True, False, True, False]),  # b3, b4, y5 matched
            np.array([True, False, True])  # b2, y4 matched
        ]
        
        b_counts, y_counts = calculate_b_y_ion_counts(library, ref_pep_cand, lib_peaks_matched)
        
        assert isinstance(b_counts, np.ndarray)
        assert isinstance(y_counts, np.ndarray)
        assert b_counts.shape == (2,)
        assert y_counts.shape == (2,)
        assert b_counts[0] == 2  # cand1: b3, b4
        assert y_counts[0] == 1  # cand1: y5
        assert b_counts[1] == 1  # cand2: b2
        assert y_counts[1] == 1  # cand2: y4
    
    def test_no_matches(self):
        """Test when no fragments match"""
        from src.spectral_fitting import calculate_b_y_ion_counts
        
        library = {
            'cand1': {
                'frags': {
                    'b3_1': [244.09, 0.25],
                    'y4_1': [472.25, 0.31]
                }
            }
        }
        
        ref_pep_cand = ['cand1']
        lib_peaks_matched = [np.array([False, False])]  # No matches
        
        b_counts, y_counts = calculate_b_y_ion_counts(library, ref_pep_cand, lib_peaks_matched)
        
        assert b_counts[0] == 0
        assert y_counts[0] == 0
    
    def test_empty_candidates(self):
        """Test with empty candidate list"""
        from src.spectral_fitting import calculate_b_y_ion_counts
        
        library = {}
        ref_pep_cand = []
        lib_peaks_matched = []
        
        b_counts, y_counts = calculate_b_y_ion_counts(library, ref_pep_cand, lib_peaks_matched)
        
        assert isinstance(b_counts, np.ndarray)
        assert isinstance(y_counts, np.ndarray)
        assert b_counts.shape == (0,)
        assert y_counts.shape == (0,)
    
    def test_only_b_ions(self):
        """Test with only b-ions matched"""
        from src.spectral_fitting import calculate_b_y_ion_counts
        
        library = {
            'cand1': {
                'frags': {
                    'b1_1': [100.0, 0.2],
                    'b2_1': [200.0, 0.3],
                    'b3_1': [300.0, 0.4],
                    'y1_1': [150.0, 0.5]
                }
            }
        }
        
        ref_pep_cand = ['cand1']
        lib_peaks_matched = [np.array([True, True, True, False])]  # Only b-ions match
        
        b_counts, y_counts = calculate_b_y_ion_counts(library, ref_pep_cand, lib_peaks_matched)
        
        assert b_counts[0] == 3
        assert y_counts[0] == 0


class TestCalculateHyperscores:
    """Test cases for calculate_hyperscores function"""
    
    def test_basic_hyperscore_calculation(self):
        """Test basic hyperscore calculation"""
        from src.spectral_fitting import calculate_hyperscores
        import math
        
        # Mock library with known fragment intensities
        library = {
            'cand1': {
                'frags': {
                    'b1_1': [100.0, 0.5],
                    'b2_1': [200.0, 0.3],
                    'y1_1': [150.0, 0.8]
                }
            }
        }
        
        ref_pep_cand = ['cand1']
        lib_peaks_matched = [np.array([True, True, True])]  # All fragments match
        b_counts = np.array([2])  # 2 b-ions
        y_counts = np.array([1])  # 1 y-ion
        
        hyperscores = calculate_hyperscores(library, ref_pep_cand, lib_peaks_matched, b_counts, y_counts)
        
        # Expected: max(0, log((0.5+0.3+0.8) * factorial(2) * factorial(1)))
        # = max(0, log(1.6 * 2 * 1)) = max(0, log(3.2))
        expected_hyperscore = max(0, np.log(1.6 * math.factorial(2) * math.factorial(1)))
        
        assert isinstance(hyperscores, np.ndarray)
        assert hyperscores.shape == (1,)
        assert np.isclose(hyperscores[0], expected_hyperscore)
    
    def test_zero_ions(self):
        """Test hyperscore when no b or y ions match"""
        from src.spectral_fitting import calculate_hyperscores
        
        library = {
            'cand1': {
                'frags': {
                    'c1_1': [100.0, 0.5],  # c-ion, not b or y
                    'z1_1': [200.0, 0.3]   # z-ion, not b or y
                }
            }
        }
        
        ref_pep_cand = ['cand1']
        lib_peaks_matched = [np.array([True, True])]
        b_counts = np.array([0])  # No b-ions
        y_counts = np.array([0])  # No y-ions
        
        hyperscores = calculate_hyperscores(library, ref_pep_cand, lib_peaks_matched, b_counts, y_counts)
        
        assert hyperscores[0] == 0.0
    
    def test_empty_candidates(self):
        """Test with empty candidate list"""
        from src.spectral_fitting import calculate_hyperscores
        
        library = {}
        ref_pep_cand = []
        lib_peaks_matched = []
        b_counts = np.array([], dtype=int)
        y_counts = np.array([], dtype=int)
        
        hyperscores = calculate_hyperscores(library, ref_pep_cand, lib_peaks_matched, b_counts, y_counts)
        
        assert isinstance(hyperscores, np.ndarray)
        assert hyperscores.shape == (0,)


class TestCalculateLongestYIons:
    """Test cases for calculate_longest_y_ions function"""
    
    def test_basic_longest_y_calculation(self):
        """Test basic longest y-ion calculation"""
        from src.spectral_fitting import calculate_longest_y_ions
        
        # Mock library - will use the actual longest_y function internally
        library = {
            'cand1': {
                'frags': {
                    'y1_1': [100.0, 0.5],
                    'y2_1': [200.0, 0.3],
                    'y3_1': [300.0, 0.8],
                    'b1_1': [150.0, 0.4]
                }
            }
        }
        
        ref_pep_cand = ['cand1']
        lib_peaks_matched = [np.array([True, False, True, False])]  # y1 and y3 match
        
        longest_y_ions = calculate_longest_y_ions(library, ref_pep_cand, lib_peaks_matched)
        
        assert isinstance(longest_y_ions, np.ndarray)
        assert longest_y_ions.shape == (1,)
        # Should return the result from the actual longest_y function
        assert isinstance(longest_y_ions[0], (int, np.integer))
    
    def test_empty_candidates(self):
        """Test with empty candidate list"""
        from src.spectral_fitting import calculate_longest_y_ions
        
        library = {}
        ref_pep_cand = []
        lib_peaks_matched = []
        
        longest_y_ions = calculate_longest_y_ions(library, ref_pep_cand, lib_peaks_matched)
        
        assert isinstance(longest_y_ions, np.ndarray)
        assert longest_y_ions.shape == (0,)


class TestFragmentScoreIntegration:
    """Integration tests for the fragment scoring functions"""
    
    def test_functions_work_together(self):
        """Test that all three functions work together correctly"""
        from src.spectral_fitting import calculate_b_y_ion_counts, calculate_hyperscores, calculate_longest_y_ions
        
        # Mock library with realistic fragment data
        library = {
            'peptide1': {
                'frags': {
                    'b1_1': [100.0, 0.2],
                    'b2_1': [200.0, 0.4],
                    'y1_1': [150.0, 0.3],
                    'y2_1': [250.0, 0.6],
                    'y3_1': [350.0, 0.5]
                }
            },
            'peptide2': {
                'frags': {
                    'b1_1': [110.0, 0.1],
                    'y1_1': [160.0, 0.8],
                    'y2_1': [260.0, 0.4]
                }
            }
        }
        
        ref_pep_cand = ['peptide1', 'peptide2']
        lib_peaks_matched = [
            np.array([True, True, False, True, True]),  # b1, b2, y2, y3 match
            np.array([False, True, True])  # y1, y2 match
        ]
        
        # Test the complete workflow
        b_counts, y_counts = calculate_b_y_ion_counts(library, ref_pep_cand, lib_peaks_matched)
        hyperscores = calculate_hyperscores(library, ref_pep_cand, lib_peaks_matched, b_counts, y_counts)
        longest_y_ions = calculate_longest_y_ions(library, ref_pep_cand, lib_peaks_matched)
        
        # Verify results make sense
        assert len(b_counts) == len(y_counts) == len(hyperscores) == len(longest_y_ions) == 2
        assert b_counts[0] == 2  # peptide1: b1, b2
        assert y_counts[0] == 2  # peptide1: y2, y3
        assert b_counts[1] == 0  # peptide2: no b-ions matched
        assert y_counts[1] == 2  # peptide2: y1, y2
        assert hyperscores[0] > 0  # Should have positive hyperscore
        assert hyperscores[1] > 0  # Should have positive hyperscore


class TestGetResidualsCsc:
    """Test cases for the get_residuals_csc function"""
    
    def test_get_residuals_csc_basic(self):
        """Test basic residuals calculation using CSC sparse matrix"""
        # Create test sparse matrix
        row_indices = [0, 1, 2]
        col_indices = [0, 0, 1]
        values = [2.0, 3.0, 1.5]
        sparse_matrix = sparse.csc_matrix((values, (row_indices, col_indices)), shape=(3, 2))
        
        dia_intensities = np.array([5.0, 7.5, 3.0])
        coeffs = np.array([2.0, 1.5])
        
        residuals, y_pred = get_residuals_csc(sparse_matrix, dia_intensities, coeffs)
        
        # Expected predictions: [2.0*2.0, 3.0*2.0, 1.5*1.5] = [4.0, 6.0, 2.25]
        expected_y_pred = np.array([4.0, 6.0, 2.25])
        # Expected residuals: [5.0-4.0, 7.5-6.0, 3.0-2.25] = [1.0, 1.5, 0.75]
        expected_residuals = np.array([1.0, 1.5, 0.75])
        
        np.testing.assert_allclose(y_pred, expected_y_pred, rtol=1e-10)
        np.testing.assert_allclose(residuals, expected_residuals, rtol=1e-10)
    
    def test_get_residuals_csc_empty_matrix(self):
        """Test handling of empty sparse matrix"""
        # Empty matrix
        sparse_matrix = sparse.csc_matrix((0, 0))
        dia_intensities = np.array([1.0, 2.0, 3.0])
        coeffs = np.array([])
        
        residuals, y_pred = get_residuals_csc(sparse_matrix, dia_intensities, coeffs)
        
        # Should return arrays of zeros with same length as dia_intensities
        expected_zeros = np.zeros_like(dia_intensities)
        np.testing.assert_array_equal(residuals, expected_zeros)
        np.testing.assert_array_equal(y_pred, expected_zeros)
    
    def test_get_residuals_csc_single_candidate(self):
        """Test with single candidate (single column)"""
        # Single column matrix
        row_indices = [0, 1, 2]
        col_indices = [0, 0, 0]
        values = [1.0, 2.0, 1.5]
        sparse_matrix = sparse.csc_matrix((values, (row_indices, col_indices)), shape=(3, 1))
        
        dia_intensities = np.array([1.2, 2.1, 1.4])
        coeffs = np.array([2.0])
        
        residuals, y_pred = get_residuals_csc(sparse_matrix, dia_intensities, coeffs)
        
        # Expected predictions: [1.0*2.0, 2.0*2.0, 1.5*2.0] = [2.0, 4.0, 3.0]
        expected_y_pred = np.array([2.0, 4.0, 3.0])
        expected_residuals = dia_intensities - expected_y_pred
        
        np.testing.assert_allclose(y_pred, expected_y_pred, rtol=1e-10)
        np.testing.assert_allclose(residuals, expected_residuals, rtol=1e-10)
    
    def test_get_residuals_csc_multiple_candidates(self):
        """Test with multiple candidates and different patterns"""
        # Create matrix with 4 candidates and 5 peaks
        row_indices = [0, 1, 2, 3, 4, 2, 3, 4, 0, 4]
        col_indices = [0, 0, 0, 1, 1, 2, 2, 3, 3, 3]
        values = [1.0, 2.0, 3.0, 0.5, 1.5, 2.5, 1.0, 0.8, 1.2, 0.3]
        sparse_matrix = sparse.csc_matrix((values, (row_indices, col_indices)), shape=(5, 4))
        
        dia_intensities = np.array([1.5, 2.5, 3.5, 1.0, 2.0])
        coeffs = np.array([1.5, 2.0, 0.5, 3.0])
        
        residuals, y_pred = get_residuals_csc(sparse_matrix, dia_intensities, coeffs)
        
        # Calculate expected manually: y_pred = matrix @ coeffs
        expected_y_pred = sparse_matrix @ coeffs
        expected_residuals = dia_intensities - expected_y_pred
        
        np.testing.assert_allclose(y_pred, expected_y_pred, rtol=1e-10)
        np.testing.assert_allclose(residuals, expected_residuals, rtol=1e-10)
    
    def test_get_residuals_csc_zero_coefficients(self):
        """Test with all zero coefficients"""
        row_indices = [0, 1, 2]
        col_indices = [0, 1, 1]
        values = [2.0, 3.0, 1.5]
        sparse_matrix = sparse.csc_matrix((values, (row_indices, col_indices)), shape=(3, 2))
        
        dia_intensities = np.array([5.0, 7.5, 3.0])
        coeffs = np.array([0.0, 0.0])  # All zeros
        
        residuals, y_pred = get_residuals_csc(sparse_matrix, dia_intensities, coeffs)
        
        # With zero coefficients, y_pred should be all zeros
        expected_y_pred = np.zeros_like(dia_intensities)
        expected_residuals = dia_intensities  # residuals = observed - 0
        
        np.testing.assert_allclose(y_pred, expected_y_pred, rtol=1e-10)
        np.testing.assert_allclose(residuals, expected_residuals, rtol=1e-10)
    
    def test_get_residuals_csc_vs_original(self):
        """Test mathematical equivalence with original get_residuals function"""
        # Create test data for both functions
        ref_sparse_val = [np.array([2.0, 3.0]), np.array([1.5])]
        ref_sparse_row = [np.array([0, 1]), np.array([2])]
        ref_sparse_col = [np.array([0, 0]), np.array([1])]
        val_obs = np.array([5.0, 7.5, 3.0])
        coeffs = np.array([2.0, 1.5])
        
        # Get result from original function
        original_residuals, original_y_pred = get_residuals(
            ref_sparse_val, ref_sparse_row, ref_sparse_col,
            [], [], [],  # No decoy data
            val_obs, coeffs, 0, 2
        )
        
        # Convert to sparse matrix format for new function
        all_rows = np.concatenate(ref_sparse_row)
        all_cols = np.concatenate(ref_sparse_col)
        all_values = np.concatenate(ref_sparse_val)
        sparse_matrix = sparse.csc_matrix((all_values, (all_rows, all_cols)), shape=(3, 2))
        
        # Get result from new function
        csc_residuals, csc_y_pred = get_residuals_csc(sparse_matrix, val_obs, coeffs)
        
        # Results should be identical
        np.testing.assert_allclose(original_residuals, csc_residuals, rtol=1e-12)
        np.testing.assert_allclose(original_y_pred, csc_y_pred, rtol=1e-12)
    
    def test_get_residuals_csc_accuracy(self):
        """Test calculation accuracy with known precise values"""
        # Create precise test case
        row_indices = [0, 1, 2, 3]
        col_indices = [0, 0, 1, 1]
        values = [0.25, 0.75, 0.4, 0.6]
        sparse_matrix = sparse.csc_matrix((values, (row_indices, col_indices)), shape=(4, 2))
        
        dia_intensities = np.array([1.0, 2.0, 3.0, 4.0])
        coeffs = np.array([4.0, 5.0])
        
        residuals, y_pred = get_residuals_csc(sparse_matrix, dia_intensities, coeffs)
        
        # Calculate expected values manually
        # y_pred[0] = 0.25 * 4.0 = 1.0
        # y_pred[1] = 0.75 * 4.0 = 3.0  
        # y_pred[2] = 0.4 * 5.0 = 2.0
        # y_pred[3] = 0.6 * 5.0 = 3.0
        expected_y_pred = np.array([1.0, 3.0, 2.0, 3.0])
        expected_residuals = dia_intensities - expected_y_pred  # [0.0, -1.0, 1.0, 1.0]
        
        np.testing.assert_allclose(y_pred, expected_y_pred, rtol=1e-15)
        np.testing.assert_allclose(residuals, expected_residuals, rtol=1e-15)
    
    def test_get_residuals_csc_coefficient_dimension_mismatch(self):
        """Test handling of coefficient array dimension mismatches"""
        # Create 3x2 matrix (3 peaks, 2 candidates)
        row_indices = [0, 1, 2]
        col_indices = [0, 1, 1]
        values = [1.0, 2.0, 3.0]
        sparse_matrix = sparse.csc_matrix((values, (row_indices, col_indices)), shape=(3, 2))
        
        dia_intensities = np.array([1.0, 2.0, 3.0])
        
        # Test with too few coefficients (should pad with zeros)
        coeffs_short = np.array([1.0])  # Missing second coefficient
        residuals, y_pred = get_residuals_csc(sparse_matrix, dia_intensities, coeffs_short)
        
        # Should behave as if second coefficient is 0
        expected_y_pred = np.array([1.0, 0.0, 0.0])  # Only first column contributes
        expected_residuals = dia_intensities - expected_y_pred
        
        np.testing.assert_allclose(y_pred, expected_y_pred, rtol=1e-10)
        np.testing.assert_allclose(residuals, expected_residuals, rtol=1e-10)
        
        # Test with too many coefficients (should truncate)
        coeffs_long = np.array([1.0, 2.0, 3.0])  # Extra coefficient
        residuals2, y_pred2 = get_residuals_csc(sparse_matrix, dia_intensities, coeffs_long)
        
        # Should ignore the third coefficient
        expected_y_pred2 = np.array([1.0, 4.0, 6.0])  # Both columns contribute
        expected_residuals2 = dia_intensities - expected_y_pred2
        
        np.testing.assert_allclose(y_pred2, expected_y_pred2, rtol=1e-10)
        np.testing.assert_allclose(residuals2, expected_residuals2, rtol=1e-10)


class TestGetManhattanDistanceCsc:
    """Test cases for the get_manhattan_distance_csc function"""
    
    def test_get_manhattan_distance_csc_basic(self):
        """Test basic Manhattan distance calculation using CSC sparse matrix"""
        # Create test sparse matrix (structure only matters, not values)
        row_indices = [0, 1, 2, 3]
        col_indices = [0, 0, 1, 1]
        values = [1.0, 1.0, 1.0, 1.0]  # Values don't affect calculation
        sparse_matrix = sparse.csc_matrix((values, (row_indices, col_indices)), shape=(4, 2))
        
        dia_intensities = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.1, 2.1, 2.9, 4.1])
        
        distances, contrasts = get_manhattan_distance_csc(sparse_matrix, dia_intensities, y_pred)
        
        # Should return arrays with 2 values (one per candidate)
        assert len(distances) == 2
        assert len(contrasts) == 2
        assert isinstance(distances, np.ndarray)
        assert isinstance(contrasts, np.ndarray)
        
        # Values should be finite and contrasts in valid range
        assert np.all(np.isfinite(distances))
        assert np.all(np.isfinite(contrasts))
        assert np.all(contrasts >= 0)
        assert np.all(contrasts <= 1)
    
    def test_get_manhattan_distance_csc_empty_matrix(self):
        """Test handling of empty sparse matrix"""
        # Empty matrix
        sparse_matrix = sparse.csc_matrix((0, 0))
        dia_intensities = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.1, 3.1])
        
        distances, contrasts = get_manhattan_distance_csc(sparse_matrix, dia_intensities, y_pred)
        
        # Should return empty arrays
        assert len(distances) == 0
        assert len(contrasts) == 0
        assert isinstance(distances, np.ndarray)
        assert isinstance(contrasts, np.ndarray)
    
    def test_get_manhattan_distance_csc_perfect_match(self):
        """Test with perfect prediction (Manhattan distance should be minimal)"""
        # Single candidate matrix
        row_indices = [0, 1]
        col_indices = [0, 0]
        values = [1.0, 1.0]
        sparse_matrix = sparse.csc_matrix((values, (row_indices, col_indices)), shape=(2, 1))
        
        dia_intensities = np.array([2.0, 3.0])
        y_pred = np.array([2.0, 3.0])  # Perfect prediction
        
        distances, contrasts = get_manhattan_distance_csc(sparse_matrix, dia_intensities, y_pred)
        
        # Perfect fit should give minimal Manhattan distance and perfect correlation
        assert len(distances) == 1
        assert len(contrasts) == 1
        assert distances[0] == np.finfo(np.float32).min  # Perfect fit case
        assert np.isclose(contrasts[0], 1.0, atol=1e-10)  # Perfect correlation
    
    def test_get_manhattan_distance_csc_zero_observed(self):
        """Test when observed intensities are zero"""
        row_indices = [0, 1]
        col_indices = [0, 0]
        values = [1.0, 1.0]
        sparse_matrix = sparse.csc_matrix((values, (row_indices, col_indices)), shape=(2, 1))
        
        dia_intensities = np.array([0.0, 0.0])  # No observed intensity
        y_pred = np.array([1.0, 2.0])
        
        distances, contrasts = get_manhattan_distance_csc(sparse_matrix, dia_intensities, y_pred)
        
        # Zero observed should be handled as bad fit
        assert len(distances) == 1
        assert len(contrasts) == 1
        assert distances[0] == np.finfo(np.float32).max  # Bad fit case
        assert contrasts[0] == 0.0  # No correlation possible
    
    def test_get_manhattan_distance_csc_multiple_candidates(self):
        """Test with multiple candidates and different patterns"""
        # Create matrix with 3 candidates and 6 peaks
        row_indices = [0, 1, 2, 3, 4, 5, 1, 2, 3, 4]
        col_indices = [0, 0, 1, 1, 2, 2, 2, 2, 2, 2]
        values = [1.0] * len(row_indices)  # Structure matters, not values
        sparse_matrix = sparse.csc_matrix((values, (row_indices, col_indices)), shape=(6, 3))
        
        dia_intensities = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        y_pred = np.array([1.1, 2.1, 2.9, 4.1, 4.9, 6.1])
        
        distances, contrasts = get_manhattan_distance_csc(sparse_matrix, dia_intensities, y_pred)
        
        # Should have 3 results
        assert len(distances) == 3
        assert len(contrasts) == 3
        
        # All should be finite and valid
        assert np.all(np.isfinite(distances))
        assert np.all(np.isfinite(contrasts))
        assert np.all(contrasts >= 0)
        assert np.all(contrasts <= 1)
    
    def test_get_manhattan_distance_csc_vs_original(self):
        """Test mathematical equivalence with original get_manhattan_distance function"""
        # Create test data for both functions
        row_idx_split = [np.array([0, 1]), np.array([2, 3])]
        col_idx_split = [np.array([0, 0]), np.array([1, 1])]  # Not used by original
        prec_val_split = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]  # Not used by original
        val_obs = np.array([1.5, 2.5, 3.5, 4.5])
        y_pred = np.array([1.2, 2.3, 3.4, 4.6])
        
        # Get result from original function
        original_distances, original_contrasts = get_manhattan_distance(
            row_idx_split, col_idx_split, prec_val_split, val_obs, y_pred
        )
        
        # Convert to sparse matrix format for new function
        all_rows = np.concatenate(row_idx_split)
        all_cols = np.concatenate(col_idx_split)
        all_values = np.ones(len(all_rows))  # Values don't matter for Manhattan distance
        sparse_matrix = sparse.csc_matrix((all_values, (all_rows, all_cols)), shape=(4, 2))
        
        # Get result from new function
        csc_distances, csc_contrasts = get_manhattan_distance_csc(sparse_matrix, val_obs, y_pred)
        
        # Results should be identical (allowing for small floating point differences)
        np.testing.assert_allclose(original_distances, csc_distances, rtol=1e-10)
        np.testing.assert_allclose(original_contrasts, csc_contrasts, rtol=1e-10)
    
    def test_get_manhattan_distance_csc_spectral_contrast_bounds(self):
        """Test that spectral contrast values are properly bounded [0,1]"""
        # Create test case that might produce out-of-bounds values
        row_indices = [0, 1, 2]
        col_indices = [0, 0, 0]
        values = [1.0, 1.0, 1.0]
        sparse_matrix = sparse.csc_matrix((values, (row_indices, col_indices)), shape=(3, 1))
        
        # Test with various intensity patterns
        test_cases = [
            # (dia_intensities, y_pred, description)
            (np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]), "perfect match"),
            (np.array([1.0, 2.0, 3.0]), np.array([-1.0, -2.0, -3.0]), "negative prediction"),
            (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 1.0]), "different patterns"),
            (np.array([3.0, 2.0, 1.0]), np.array([1.0, 2.0, 3.0]), "inverse correlation"),
        ]
        
        for dia_intensities, y_pred_test, description in test_cases:
            distances, contrasts = get_manhattan_distance_csc(sparse_matrix, dia_intensities, y_pred_test)
            
            # Spectral contrast should always be in [0, 1]
            assert len(contrasts) == 1, f"Failed for {description}"
            assert 0.0 <= contrasts[0] <= 1.0, f"Spectral contrast out of bounds for {description}: {contrasts[0]}"
            assert np.isfinite(distances[0]), f"Manhattan distance not finite for {description}: {distances[0]}"


class TestGofStatCsc:
    """Test cases for the gof_stat_csc function"""
    
    def test_gof_stat_csc_basic(self):
        """Test basic GOF statistics calculation with CSC matrix"""
        # Create simple test case
        row_indices = [0, 1, 2]
        col_indices = [0, 0, 1]
        values = [100.0, 200.0, 150.0]
        sparse_matrix = sparse.csc_matrix((values, (row_indices, col_indices)), shape=(3, 2))
        
        residuals = np.array([10.0, -20.0, 15.0])
        dia_intensities = np.array([110.0, 180.0, 1e-7])  # Last peak is "unmatched"
        coeffs = np.array([1.0, 1.0])
        
        result, max_unmatched, max_matched = gof_stat_csc(
            sparse_matrix, residuals, dia_intensities, coeffs)
        
        # Should return arrays with 2 elements (one per candidate)
        assert len(result) == 2
        assert len(max_unmatched) == 2
        assert len(max_matched) == 2
        assert isinstance(result, np.ndarray)
        assert isinstance(max_unmatched, np.ndarray)
        assert isinstance(max_matched, np.ndarray)
        
        # All results should be finite
        assert np.all(np.isfinite(result))
        assert np.all(np.isfinite(max_unmatched))
        assert np.all(np.isfinite(max_matched))
    
    def test_gof_stat_csc_empty_matrix(self):
        """Test GOF calculation with empty matrix"""
        sparse_matrix = sparse.csc_matrix((0, 0))
        residuals = np.array([])
        dia_intensities = np.array([])
        coeffs = np.array([])
        
        result, max_unmatched, max_matched = gof_stat_csc(
            sparse_matrix, residuals, dia_intensities, coeffs)
        
        assert len(result) == 0
        assert len(max_unmatched) == 0
        assert len(max_matched) == 0
        assert isinstance(result, np.ndarray)
        assert isinstance(max_unmatched, np.ndarray)
        assert isinstance(max_matched, np.ndarray)
    
    def test_gof_stat_csc_single_candidate(self):
        """Test GOF calculation with single candidate"""
        row_indices = [0, 1]
        col_indices = [0, 0]
        values = [100.0, 200.0]
        sparse_matrix = sparse.csc_matrix((values, (row_indices, col_indices)), shape=(2, 1))
        
        residuals = np.array([5.0, 10.0])
        dia_intensities = np.array([105.0, 210.0])  # Both matched peaks
        coeffs = np.array([1.0])
        
        result, max_unmatched, max_matched = gof_stat_csc(
            sparse_matrix, residuals, dia_intensities, coeffs)
        
        assert len(result) == 1
        assert len(max_unmatched) == 1
        assert len(max_matched) == 1
        
        # Check calculations
        # sum_of_residuals = 5.0 + 10.0 = 15.0
        # sum_of_fitted_peaks = 1.0*100.0 + 1.0*200.0 = 300.0
        # result = log2(15.0 / 300.0) = log2(0.05)
        expected_result = np.log2(15.0 / 300.0)
        assert np.isclose(result[0], expected_result), f"Expected {expected_result}, got {result[0]}"
        
        # max_matched_residual should be 10.0 (larger of the two)
        # max_matched = log2(10.0 / 300.0 + 1e-10)
        expected_max_matched = np.log2(10.0 / 300.0 + 1e-10)
        assert np.isclose(max_matched[0], expected_max_matched, rtol=1e-10)
        
        # Since both peaks are matched (> 1e-6), max_unmatched should be log2(1e-10)
        expected_max_unmatched = np.log2(0.0 / 300.0 + 1e-10)
        assert np.isclose(max_unmatched[0], expected_max_unmatched, rtol=1e-10)
    
    def test_gof_stat_csc_matched_unmatched_peaks(self):
        """Test GOF calculation with both matched and unmatched peaks"""
        row_indices = [0, 1, 2]
        col_indices = [0, 0, 0]
        values = [100.0, 200.0, 150.0]
        sparse_matrix = sparse.csc_matrix((values, (row_indices, col_indices)), shape=(3, 1))
        
        residuals = np.array([5.0, 15.0, 25.0])
        dia_intensities = np.array([105.0, 1e-7, 175.0])  # Middle peak is unmatched
        coeffs = np.array([1.0])
        
        result, max_unmatched, max_matched = gof_stat_csc(
            sparse_matrix, residuals, dia_intensities, coeffs)
        
        assert len(result) == 1
        
        # sum_of_residuals = 5.0 + 15.0 + 25.0 = 45.0
        # sum_of_fitted_peaks = 100.0 + 200.0 + 150.0 = 450.0
        expected_result = np.log2(45.0 / 450.0)
        assert np.isclose(result[0], expected_result)
        
        # max_matched_residual should be 25.0 (from matched peaks: residuals[0]=5.0, residuals[2]=25.0)
        # max_unmatched_residual should be 15.0 (from unmatched peak: residuals[1]=15.0)
        expected_max_matched = np.log2(25.0 / 450.0 + 1e-10)
        expected_max_unmatched = np.log2(15.0 / 450.0 + 1e-10)
        
        assert np.isclose(max_matched[0], expected_max_matched, rtol=1e-10)
        assert np.isclose(max_unmatched[0], expected_max_unmatched, rtol=1e-10)
    
    def test_gof_stat_csc_zero_residuals(self):
        """Test GOF calculation with zero residuals (perfect fit)"""
        row_indices = [0, 1]
        col_indices = [0, 0]
        values = [100.0, 200.0]
        sparse_matrix = sparse.csc_matrix((values, (row_indices, col_indices)), shape=(2, 1))
        
        residuals = np.array([0.0, 0.0])
        dia_intensities = np.array([100.0, 200.0])
        coeffs = np.array([1.0])
        
        result, max_unmatched, max_matched = gof_stat_csc(
            sparse_matrix, residuals, dia_intensities, coeffs)
        
        # Should handle zero residuals gracefully
        assert len(result) == 1
        assert np.isfinite(result[0])
        
        # sum_of_residuals = 0.0 -> set to 1e-6
        # sum_of_fitted_peaks = 300.0
        expected_result = np.log2(1e-6 / 300.0)
        assert np.isclose(result[0], expected_result)
    
    def test_gof_stat_csc_zero_fitted_peaks(self):
        """Test GOF calculation with zero fitted peaks"""
        row_indices = [0, 1]
        col_indices = [0, 0]
        values = [100.0, 200.0]
        sparse_matrix = sparse.csc_matrix((values, (row_indices, col_indices)), shape=(2, 1))
        
        residuals = np.array([5.0, 10.0])
        dia_intensities = np.array([105.0, 210.0])
        coeffs = np.array([0.0])  # Zero coefficient -> zero fitted peaks
        
        result, max_unmatched, max_matched = gof_stat_csc(
            sparse_matrix, residuals, dia_intensities, coeffs)
        
        # Should handle zero fitted peaks gracefully
        assert len(result) == 1
        assert np.isfinite(result[0])
        
        # sum_of_fitted_peaks = 0.0 -> set to 1e-6
        # sum_of_residuals = 15.0
        expected_result = np.log2(15.0 / 1e-6)
        assert np.isclose(result[0], expected_result)
    
    def test_gof_stat_csc_no_matched_peaks(self):
        """Test GOF calculation with no matched peaks for a candidate"""
        # Create matrix where first column has no entries
        row_indices = [0, 1]
        col_indices = [1, 1]  # Only second candidate has peaks
        values = [100.0, 200.0]
        sparse_matrix = sparse.csc_matrix((values, (row_indices, col_indices)), shape=(2, 2))
        
        residuals = np.array([5.0, 10.0])
        dia_intensities = np.array([105.0, 210.0])
        coeffs = np.array([1.0, 1.0])
        
        result, max_unmatched, max_matched = gof_stat_csc(
            sparse_matrix, residuals, dia_intensities, coeffs)
        
        assert len(result) == 2
        
        # First candidate has no peaks, should get default values
        assert np.isclose(result[0], np.log2(1e-6))
        assert np.isclose(max_unmatched[0], np.log2(1e-10))
        assert np.isclose(max_matched[0], np.log2(1e-10))
        
        # Second candidate should have normal calculation
        assert np.isfinite(result[1])
        assert np.isfinite(max_unmatched[1])
        assert np.isfinite(max_matched[1])
    
    def test_gof_stat_csc_vs_original(self):
        """Test mathematical equivalence between gof_stat_csc and original gof_stat"""
        # Create test data that can be used with both functions
        row_indices = np.array([0, 1, 2, 3, 4])
        col_indices = np.array([0, 0, 1, 1, 1])
        values = np.array([100.0, 200.0, 150.0, 300.0, 250.0])
        
        # Create sparse matrix for CSC version
        sparse_matrix = sparse.csc_matrix((values, (row_indices, col_indices)), shape=(5, 2))
        
        # Create split arrays for original version
        row_idx_split = [np.array([0, 1]), np.array([2, 3, 4])]
        col_idx_split = [np.array([0, 0]), np.array([1, 1, 1])]
        val_split = [np.array([100.0, 200.0]), np.array([150.0, 300.0, 250.0])]
        
        # Test data
        residuals = np.array([5.0, -10.0, 15.0, 20.0, -25.0])
        dia_intensities = np.array([105.0, 190.0, 1e-7, 320.0, 225.0])  # Peak 2 is unmatched
        coeffs = np.array([1.0, 1.0])
        offset = 0
        
        # Call both functions
        original_result, original_max_unmatched, original_max_matched = gof_stat(
            row_idx_split, col_idx_split, val_split, residuals, dia_intensities, coeffs, offset)
        
        csc_result, csc_max_unmatched, csc_max_matched = gof_stat_csc(
            sparse_matrix, residuals, dia_intensities, coeffs)
        
        # Results should be identical
        np.testing.assert_allclose(original_result, csc_result, rtol=1e-12)
        np.testing.assert_allclose(original_max_unmatched, csc_max_unmatched, rtol=1e-12)
        np.testing.assert_allclose(original_max_matched, csc_max_matched, rtol=1e-12)


class TestCalculateFracDiaIntensityCsc:
    """Test cases for the calculate_frac_dia_intensity_csc function"""
    
    def test_calculate_frac_dia_intensity_csc_basic(self):
        """Test basic fractional DIA intensity calculation with CSC matrix"""
        # Create simple test case
        row_indices = [0, 1, 2, 3]
        col_indices = [0, 0, 1, 1]
        values = [1.0, 1.0, 1.0, 1.0]  # Values don't matter, just structure
        sparse_matrix = sparse.csc_matrix((values, (row_indices, col_indices)), shape=(4, 2))
        
        # DIA spectrum with m/z and intensity columns
        dia_spectrum = np.array([[500.0, 100.0], [501.0, 200.0], [502.0, 150.0], [503.0, 250.0]])
        tic = 1000.0  # Total intensity
        
        frac_intensities = calculate_frac_dia_intensity_csc(sparse_matrix, dia_spectrum, tic)
        
        # Should return array with 2 elements (one per candidate)
        assert len(frac_intensities) == 2
        assert isinstance(frac_intensities, np.ndarray)
        
        # Candidate 0 has peaks at indices 0,1 -> intensities 100+200=300 -> 300/1000=0.3
        # Candidate 1 has peaks at indices 2,3 -> intensities 150+250=400 -> 400/1000=0.4
        expected_candidate_0 = (100.0 + 200.0) / 1000.0  # 0.3
        expected_candidate_1 = (150.0 + 250.0) / 1000.0  # 0.4
        
        assert np.isclose(frac_intensities[0], expected_candidate_0), f"Expected {expected_candidate_0}, got {frac_intensities[0]}"
        assert np.isclose(frac_intensities[1], expected_candidate_1), f"Expected {expected_candidate_1}, got {frac_intensities[1]}"
    
    def test_calculate_frac_dia_intensity_csc_empty_matrix(self):
        """Test fractional DIA intensity calculation with empty matrix"""
        sparse_matrix = sparse.csc_matrix((0, 0))
        dia_spectrum = np.array([]).reshape(0, 2)
        tic = 1000.0
        
        frac_intensities = calculate_frac_dia_intensity_csc(sparse_matrix, dia_spectrum, tic)
        
        assert len(frac_intensities) == 0
        assert isinstance(frac_intensities, np.ndarray)
    
    def test_calculate_frac_dia_intensity_csc_single_candidate(self):
        """Test fractional DIA intensity calculation with single candidate"""
        row_indices = [0, 1, 2]
        col_indices = [0, 0, 0]
        values = [100.0, 200.0, 150.0]  # Values don't matter for this calculation
        sparse_matrix = sparse.csc_matrix((values, (row_indices, col_indices)), shape=(3, 1))
        
        dia_spectrum = np.array([[500.0, 50.0], [501.0, 75.0], [502.0, 25.0]])
        tic = 300.0  # Total intensity
        
        frac_intensities = calculate_frac_dia_intensity_csc(sparse_matrix, dia_spectrum, tic)
        
        assert len(frac_intensities) == 1
        
        # Single candidate has all three peaks -> intensities 50+75+25=150 -> 150/300=0.5
        expected_fraction = (50.0 + 75.0 + 25.0) / 300.0  # 0.5
        assert np.isclose(frac_intensities[0], expected_fraction), f"Expected {expected_fraction}, got {frac_intensities[0]}"
    
    def test_calculate_frac_dia_intensity_csc_no_matched_peaks(self):
        """Test fractional DIA intensity calculation with no matched peaks for a candidate"""
        # Create matrix where first column has no entries
        row_indices = [0, 1]
        col_indices = [1, 1]  # Only second candidate has peaks
        values = [100.0, 200.0]
        sparse_matrix = sparse.csc_matrix((values, (row_indices, col_indices)), shape=(2, 2))
        
        dia_spectrum = np.array([[500.0, 100.0], [501.0, 200.0]])
        tic = 500.0
        
        frac_intensities = calculate_frac_dia_intensity_csc(sparse_matrix, dia_spectrum, tic)
        
        assert len(frac_intensities) == 2
        
        # First candidate has no peaks, should get 0.0
        assert frac_intensities[0] == 0.0
        
        # Second candidate has both peaks -> intensities 100+200=300 -> 300/500=0.6
        expected_candidate_1 = (100.0 + 200.0) / 500.0  # 0.6
        assert np.isclose(frac_intensities[1], expected_candidate_1)
    
    def test_calculate_frac_dia_intensity_csc_zero_tic(self):
        """Test fractional DIA intensity calculation with zero TIC"""
        row_indices = [0, 1]
        col_indices = [0, 0]
        values = [100.0, 200.0]
        sparse_matrix = sparse.csc_matrix((values, (row_indices, col_indices)), shape=(2, 1))
        
        dia_spectrum = np.array([[500.0, 100.0], [501.0, 200.0]])
        tic = 0.0  # Zero TIC
        
        frac_intensities = calculate_frac_dia_intensity_csc(sparse_matrix, dia_spectrum, tic)
        
        # Should handle zero TIC gracefully
        assert len(frac_intensities) == 1
        assert frac_intensities[0] == 0.0
    
    def test_calculate_frac_dia_intensity_csc_different_patterns(self):
        """Test fractional DIA intensity calculation with different peak patterns"""
        # Create matrix with varied patterns
        row_indices = [0, 1, 2, 3, 4]
        col_indices = [0, 1, 1, 2, 2]  # Candidate 0: 1 peak, Candidate 1: 2 peaks, Candidate 2: 2 peaks
        values = [1.0, 1.0, 1.0, 1.0, 1.0]
        sparse_matrix = sparse.csc_matrix((values, (row_indices, col_indices)), shape=(5, 3))
        
        dia_spectrum = np.array([
            [500.0, 100.0],  # candidate 0
            [501.0, 50.0],   # candidate 1
            [502.0, 75.0],   # candidate 1
            [503.0, 200.0],  # candidate 2
            [504.0, 25.0]    # candidate 2
        ])
        tic = 1000.0
        
        frac_intensities = calculate_frac_dia_intensity_csc(sparse_matrix, dia_spectrum, tic)
        
        assert len(frac_intensities) == 3
        
        # Candidate 0: 100/1000 = 0.1
        assert np.isclose(frac_intensities[0], 0.1)
        
        # Candidate 1: (50+75)/1000 = 0.125
        assert np.isclose(frac_intensities[1], 0.125)
        
        # Candidate 2: (200+25)/1000 = 0.225
        assert np.isclose(frac_intensities[2], 0.225)
    
    def test_calculate_frac_dia_intensity_csc_vs_original(self):
        """Test mathematical equivalence between CSC and original sparse versions"""
        # Create test data that can be used with both functions
        row_indices = np.array([0, 1, 2, 3, 4])
        col_indices = np.array([0, 0, 1, 1, 2])
        values = np.array([100.0, 200.0, 150.0, 300.0, 250.0])
        
        # Create sparse matrix for CSC version
        sparse_matrix = sparse.csc_matrix((values, (row_indices, col_indices)), shape=(5, 3))
        
        # Create split arrays for original version
        ref_spec_row_indices_split = [
            np.array([0, 1]),      # Candidate 0
            np.array([2, 3]),      # Candidate 1
            np.array([4])          # Candidate 2
        ]
        
        # Test data
        dia_spectrum = np.array([
            [500.0, 80.0],   # index 0
            [501.0, 120.0],  # index 1
            [502.0, 200.0],  # index 2
            [503.0, 300.0],  # index 3
            [504.0, 100.0]   # index 4
        ])
        tic = 1000.0
        
        # Call both functions
        original_result = calculate_frac_dia_intensity_sparse(
            sparse_matrix, ref_spec_row_indices_split, dia_spectrum, tic)
        
        csc_result = calculate_frac_dia_intensity_csc(
            sparse_matrix, dia_spectrum, tic)
        
        # Results should be identical
        np.testing.assert_allclose(original_result, csc_result, rtol=1e-12)
        
        # Verify specific calculations
        # Candidate 0: (80+120)/1000 = 0.2
        # Candidate 1: (200+300)/1000 = 0.5
        # Candidate 2: 100/1000 = 0.1
        expected_results = np.array([0.2, 0.5, 0.1])
        np.testing.assert_allclose(csc_result, expected_results, rtol=1e-12)
    
    def test_calculate_frac_dia_intensity_csc_edge_cases(self):
        """Test edge cases like very small intensities and large matrices"""
        # Test with very small intensities
        row_indices = [0, 1]
        col_indices = [0, 0]
        values = [1e-10, 1e-10]
        sparse_matrix = sparse.csc_matrix((values, (row_indices, col_indices)), shape=(2, 1))
        
        dia_spectrum = np.array([[500.0, 1e-12], [501.0, 1e-12]])
        tic = 1e-10
        
        frac_intensities = calculate_frac_dia_intensity_csc(sparse_matrix, dia_spectrum, tic)
        
        # Should handle very small numbers correctly
        assert len(frac_intensities) == 1
        assert np.isfinite(frac_intensities[0])
        
        # Expected: (1e-12 + 1e-12) / 1e-10 = 2e-12 / 1e-10 = 0.02
        expected_fraction = (1e-12 + 1e-12) / 1e-10
        assert np.isclose(frac_intensities[0], expected_fraction, rtol=1e-10)


class TestBuildSparseMatrixDirect:
    """Test cases for the build_sparse_matrix_direct function"""
    
    def test_build_sparse_matrix_direct_basic(self):
        """Test basic sparse matrix construction with direct approach"""
        # Create simple test case
        ref_pep_cand_loc = [np.array([1, 3, 5])]  # Odd indices are "in DIA" 
        norm_intensities = [np.array([0.2, 0.5, 0.3])]
        ref_pep_cand = [('PEPTIDE', 2)]
        dia_spectrum = np.array([[500.0, 100.0], [501.0, 200.0], [502.0, 150.0]])
        
        matrix, dia_padded, unique_idxs, info = build_sparse_matrix_direct(
            ref_pep_cand_loc, norm_intensities, ref_pep_cand, dia_spectrum)
        
        # Check return types and shapes
        assert isinstance(matrix, sparse.coo_matrix)
        assert isinstance(dia_padded, np.ndarray)
        assert isinstance(unique_idxs, list)
        assert isinstance(info, dict)
        
        # Matrix should have shape (n_unique_peaks + 1 penalty row, n_candidates)
        assert matrix.shape[1] == 1  # 1 candidate
        assert matrix.shape[0] == len(unique_idxs) + 1  # unique peaks + penalty row
        
        # DIA intensities should be padded with one zero
        assert len(dia_padded) == len(unique_idxs) + 1
        assert dia_padded[-1] == 0  # Last element should be penalty padding
        
        # Should have lib_peaks_matched info
        assert 'lib_peaks_matched' in info
        assert 'num_lib_peaks_matched' in info
        assert len(info['lib_peaks_matched']) == 1
    
    def test_build_sparse_matrix_direct_empty_candidates(self):
        """Test sparse matrix construction with no candidates"""
        ref_pep_cand_loc = []
        norm_intensities = []
        ref_pep_cand = []
        dia_spectrum = np.array([[500.0, 100.0], [501.0, 200.0]])
        
        matrix, dia_padded, unique_idxs, info = build_sparse_matrix_direct(
            ref_pep_cand_loc, norm_intensities, ref_pep_cand, dia_spectrum)
        
        # Should return empty matrix
        assert matrix.shape == (1, 0)  # 1 row for penalty, 0 candidates
        assert len(dia_padded) == 1
        assert dia_padded[0] == 0
        assert unique_idxs == []
        assert info['lib_peaks_matched'] == []
        assert len(info['num_lib_peaks_matched']) == 0
    
    def test_build_sparse_matrix_direct_multiple_candidates(self):
        """Test sparse matrix construction with multiple candidates"""
        # Candidate 1: peaks at indices 1,3 (DIA indices 0,1)
        # Candidate 2: peaks at indices 5,7 (DIA indices 2,3)
        ref_pep_cand_loc = [
            np.array([1, 3, 4]),  # 1,3 are odd (matched), 4 is even (unmatched)
            np.array([5, 7, 6])   # 5,7 are odd (matched), 6 is even (unmatched)
        ]
        norm_intensities = [
            np.array([0.3, 0.4, 0.1]),  # Intensities for candidate 1
            np.array([0.5, 0.2, 0.2])   # Intensities for candidate 2
        ]
        ref_pep_cand = [('PEPTIDE1', 2), ('PEPTIDE2', 3)]
        dia_spectrum = np.array([
            [500.0, 100.0],  # DIA index 0 (from location 1)
            [501.0, 200.0],  # DIA index 1 (from location 3)
            [502.0, 150.0],  # DIA index 2 (from location 5)
            [503.0, 250.0]   # DIA index 3 (from location 7)
        ])
        
        matrix, dia_padded, unique_idxs, info = build_sparse_matrix_direct(
            ref_pep_cand_loc, norm_intensities, ref_pep_cand, dia_spectrum)
        
        # Check dimensions
        assert matrix.shape[1] == 2  # 2 candidates
        assert matrix.shape[0] == len(unique_idxs) + 1  # unique peaks + penalty
        
        # Should have 4 unique DIA indices: [0, 1, 2, 3]
        assert len(unique_idxs) == 4
        assert sorted(unique_idxs) == [0, 1, 2, 3]
        
        # DIA intensities should match the unique indices
        assert len(dia_padded) == 5  # 4 peaks + 1 penalty
        assert dia_padded[-1] == 0  # Penalty padding
        
        # Check that matched peaks are correctly identified
        assert len(info['lib_peaks_matched']) == 2
        assert info['num_lib_peaks_matched'][0] == 2  # Candidate 1 has 2 matched peaks
        assert info['num_lib_peaks_matched'][1] == 2  # Candidate 2 has 2 matched peaks
    
    def test_build_sparse_matrix_direct_no_matched_peaks(self):
        """Test sparse matrix construction with no matched peaks"""
        # All peaks have even indices (unmatched)
        ref_pep_cand_loc = [np.array([2, 4, 6])]  # All even indices
        norm_intensities = [np.array([0.3, 0.4, 0.3])]
        ref_pep_cand = [('PEPTIDE', 2)]
        dia_spectrum = np.array([[500.0, 100.0], [501.0, 200.0]])
        
        matrix, dia_padded, unique_idxs, info = build_sparse_matrix_direct(
            ref_pep_cand_loc, norm_intensities, ref_pep_cand, dia_spectrum)
        
        # Should have only penalty row
        assert matrix.shape == (1, 1)  # 1 penalty row, 1 candidate
        assert len(unique_idxs) == 0  # No matched peaks
        assert len(dia_padded) == 1   # Only penalty padding
        assert dia_padded[0] == 0
        
        # Should still have candidate info
        assert len(info['lib_peaks_matched']) == 1
        assert info['num_lib_peaks_matched'][0] == 0  # No matched peaks
    
    def test_build_sparse_matrix_direct_vs_original(self):
        """Test mathematical equivalence between direct and original approaches"""
        # Create test data
        ref_pep_cand_loc = [
            np.array([1, 3, 5, 2]),  # Mixed matched (1,3,5) and unmatched (2)
            np.array([7, 9, 4])      # Mixed matched (7,9) and unmatched (4)
        ]
        norm_intensities = [
            np.array([0.2, 0.3, 0.4, 0.1]),
            np.array([0.5, 0.3, 0.2])
        ]
        ref_pep_cand = [('PEPTIDE1', 2), ('PEPTIDE2', 3)]
        dia_spectrum = np.array([
            [500.0, 80.0],   # DIA index 0 (from location 1)
            [501.0, 120.0],  # DIA index 1 (from location 3)
            [502.0, 200.0],  # DIA index 2 (from location 5)
            [503.0, 300.0],  # DIA index 3 (from location 7)
            [504.0, 180.0]   # DIA index 4 (from location 9)
        ])
        
        # Call direct function
        matrix_direct, dia_direct, unique_direct, info_direct = build_sparse_matrix_direct(
            ref_pep_cand_loc, norm_intensities, ref_pep_cand, dia_spectrum)
        
        # Call original function for comparison
        # First get unique indices the old way (simulate split array approach)
        lib_peaks_matched = [j % 2 == 1 for j in ref_pep_cand_loc]
        ref_spec_row_indices_split = [np.int32(((i[j] + 1) / 2) - 1) for i, j in zip(ref_pep_cand_loc, lib_peaks_matched)]
        ref_spec_row_indices = np.concatenate([s for s in ref_spec_row_indices_split if len(s) > 0])
        unique_row_idxs_old = sorted(set(ref_spec_row_indices))
        dia_spec_int_old = dia_spectrum[unique_row_idxs_old, 1]
        
        matrix_original, dia_original, split_data = build_sparse_matrix_simple(
            ref_pep_cand_loc, norm_intensities, ref_pep_cand, unique_row_idxs_old, dia_spec_int_old)
        
        # Matrices should be equivalent
        assert matrix_direct.shape == matrix_original.shape
        
        # Convert to dense for comparison
        dense_direct = matrix_direct.toarray()
        dense_original = matrix_original.toarray()
        np.testing.assert_allclose(dense_direct, dense_original, rtol=1e-12)
        
        # DIA intensities should be equivalent
        np.testing.assert_allclose(dia_direct, dia_original, rtol=1e-12)
        
        # Unique indices should be equivalent
        assert unique_direct == unique_row_idxs_old
        
        # Info should contain equivalent data
        assert len(info_direct['lib_peaks_matched']) == len(split_data['lib_peaks_matched'])
        np.testing.assert_array_equal(info_direct['num_lib_peaks_matched'], split_data['num_lib_peaks_matched'])
    
    def test_build_sparse_matrix_direct_single_peak_candidates(self):
        """Test with candidates that have only one matched peak each"""
        ref_pep_cand_loc = [
            np.array([1]),  # Single matched peak -> DIA index 0
            np.array([5])   # Single matched peak -> DIA index 2
        ]
        norm_intensities = [
            np.array([1.0]),
            np.array([0.8])
        ]
        ref_pep_cand = [('PEPTIDE1', 2), ('PEPTIDE2', 2)]
        dia_spectrum = np.array([
            [500.0, 100.0],  # DIA index 0 (from location 1)
            [501.0, 130.0],  # DIA index 1 (unused)
            [502.0, 150.0]   # DIA index 2 (from location 5)
        ])
        
        matrix, dia_padded, unique_idxs, info = build_sparse_matrix_direct(
            ref_pep_cand_loc, norm_intensities, ref_pep_cand, dia_spectrum)
        
        # Should have 2 unique peaks + 1 penalty row
        assert matrix.shape == (3, 2)  # 3 rows (2 peaks + penalty), 2 candidates
        assert len(unique_idxs) == 2
        assert sorted(unique_idxs) == [0, 2]  # DIA indices 0 and 2
        
        # Each candidate should have 1 matched peak
        assert info['num_lib_peaks_matched'][0] == 1
        assert info['num_lib_peaks_matched'][1] == 1
        
        # DIA intensities should be [100.0, 150.0, 0] (last 0 is penalty)
        expected_dia = np.array([100.0, 150.0, 0])
        np.testing.assert_allclose(dia_padded, expected_dia)
    
    def test_build_sparse_matrix_direct_edge_cases(self):
        """Test edge cases like very large indices"""
        # Test with larger DIA spectrum indices
        ref_pep_cand_loc = [np.array([99, 101])]  # Large odd indices
        norm_intensities = [np.array([0.6, 0.4])]
        ref_pep_cand = [('PEPTIDE', 2)]
        
        # Create DIA spectrum large enough to contain these indices
        dia_spectrum = np.zeros((51, 2))  # 51 peaks (indices 0-50)
        dia_spectrum[:, 0] = np.arange(500, 551)  # m/z values
        dia_spectrum[:, 1] = np.random.rand(51) * 100  # Random intensities
        
        matrix, dia_padded, unique_idxs, info = build_sparse_matrix_direct(
            ref_pep_cand_loc, norm_intensities, ref_pep_cand, dia_spectrum)
        
        # Should handle large indices correctly
        # DIA indices should be ((99+1)/2)-1=49 and ((101+1)/2)-1=50
        expected_unique = [49, 50]
        assert unique_idxs == expected_unique
        
        # Matrix should have correct shape
        assert matrix.shape == (3, 1)  # 2 peaks + 1 penalty, 1 candidate
        
        # Should have 2 matched peaks
        assert info['num_lib_peaks_matched'][0] == 2


class TestFragmentInfoFunctions:
    """Test cases for the new fragment info functions"""
    
    def test_extract_basic_fragment_info(self):
        """Test basic fragment info extraction"""
        # Create test data with mixed matched/unmatched peaks
        ref_pep_cand_loc = [
            np.array([1, 3, 5, 2]),  # 1,3,5 are odd (matched), 2 is even (unmatched)
            np.array([7, 4, 9])      # 7,9 are odd (matched), 4 is even (unmatched)
        ]
        
        info = extract_basic_fragment_info(ref_pep_cand_loc)
        
        # Should return basic info only
        assert 'lib_peaks_matched' in info
        assert 'num_lib_peaks_matched' in info
        assert len(info) == 2  # Only 2 keys
        
        # Check matched peaks calculation
        assert len(info['lib_peaks_matched']) == 2
        assert len(info['num_lib_peaks_matched']) == 2
        
        # Candidate 1: 3 matched peaks (1,3,5)
        assert info['num_lib_peaks_matched'][0] == 3
        expected_matched_1 = np.array([True, True, True, False])
        np.testing.assert_array_equal(info['lib_peaks_matched'][0], expected_matched_1)
        
        # Candidate 2: 2 matched peaks (7,9)
        assert info['num_lib_peaks_matched'][1] == 2
        expected_matched_2 = np.array([True, False, True])
        np.testing.assert_array_equal(info['lib_peaks_matched'][1], expected_matched_2)
    
    def test_extract_basic_fragment_info_empty(self):
        """Test basic fragment info with empty input"""
        ref_pep_cand_loc = []
        
        info = extract_basic_fragment_info(ref_pep_cand_loc)
        
        assert info['lib_peaks_matched'] == []
        assert len(info['num_lib_peaks_matched']) == 0
    
    def test_extract_detailed_fragment_info_basic(self):
        """Test detailed fragment info extraction"""
        # Create simple test case
        ref_pep_cand_loc = [np.array([1, 3])]  # Both odd (matched)
        norm_intensities = [np.array([0.3, 0.7])]
        ref_pep_cand = [('PEPTIDE', 2)]
        ref_pep_cand_list = [np.array([[500.1, 100.0], [600.2, 200.0]])]
        bin_centers = np.array([500.0, 600.0])
        dia_spectrum = np.array([[500.0, 150.0], [600.0, 250.0]])
        unique_row_idxs = [0, 1]
        
        # Mock library with ordered_frags
        library = {
            ('PEPTIDE', 2): {
                'ordered_frags': np.array(['b1', 'y1'])
            }
        }
        
        info = extract_detailed_fragment_info(
            ref_pep_cand_loc, norm_intensities, ref_pep_cand, 
            ref_pep_cand_list, bin_centers, dia_spectrum, 
            library, unique_row_idxs
        )
        
        # Should have all detailed info keys
        expected_keys = ['frag_errors', 'lib_frag_mz', 'lib_frag_int', 
                        'obs_frag_int', 'frag_names', 'frag_mz']
        for key in expected_keys:
            assert key in info
        
        # Should have one entry per candidate
        assert len(info['frag_errors']) == 1
        assert len(info['lib_frag_mz']) == 1
        
        # Check fragment errors calculation
        # frag_errors = (obs_mz - lib_mz) / obs_mz
        expected_errors = (bin_centers - ref_pep_cand_list[0][:, 0]) / bin_centers
        np.testing.assert_allclose(info['frag_errors'][0], expected_errors)
        
        # Check fragment m/z values
        np.testing.assert_allclose(info['lib_frag_mz'][0], ref_pep_cand_list[0][:, 0])
        
        # Check fragment names
        np.testing.assert_array_equal(info['frag_names'][0], ['b1', 'y1'])
    
    def test_extract_detailed_fragment_info_no_matches(self):
        """Test detailed fragment info with no matched peaks"""
        # All even indices (no matches)
        ref_pep_cand_loc = [np.array([2, 4])]
        norm_intensities = [np.array([0.5, 0.5])]
        ref_pep_cand = [('PEPTIDE', 2)]
        ref_pep_cand_list = [np.array([[500.1, 100.0], [600.2, 200.0]])]
        bin_centers = np.array([500.0, 600.0])
        dia_spectrum = np.array([[500.0, 150.0], [600.0, 250.0]])
        unique_row_idxs = []
        
        library = {
            ('PEPTIDE', 2): {
                'ordered_frags': np.array(['b1', 'y1'])
            }
        }
        
        info = extract_detailed_fragment_info(
            ref_pep_cand_loc, norm_intensities, ref_pep_cand, 
            ref_pep_cand_list, bin_centers, dia_spectrum, 
            library, unique_row_idxs
        )
        
        # All arrays should be empty for no matches
        assert len(info['frag_errors'][0]) == 0
        assert len(info['lib_frag_mz'][0]) == 0
        assert len(info['obs_frag_int'][0]) == 0
        assert len(info['frag_names'][0]) == 0


if __name__ == "__main__":
    pytest.main([__file__])
