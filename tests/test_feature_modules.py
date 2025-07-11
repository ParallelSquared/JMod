"""
Unit tests for the refactored feature calculation modules.
"""

import numpy as np
import pytest
from src.features.intensity_features import (
    calculate_num_peaks_matched,
    calculate_fraction_library_intensity,
    calculate_fraction_dia_intensity,
    calculate_fraction_intensity_matched
)
from src.features.error_features import (
    calculate_ms1_error,
    calculate_rt_error,
    calculate_fragment_errors,
    calculate_residual_features
)
from src.features.correlation_features import (
    calculate_r2_all,
    calculate_r2_lib_spec,
    calculate_cosine_similarity,
    calculate_spectral_contrast
)
from src.features.fragment_features import (
    count_b_y_ions,
    calculate_hyperscore,
    find_longest_y_series
)
from src.features.scoring_features import (
    calculate_scribe_score,
    calculate_manhattan_distance,
    calculate_goodness_of_fit
)


class TestIntensityFeatures:
    """Test intensity-based feature calculations."""
    
    def test_num_peaks_matched(self):
        """Test counting matched peaks."""
        lib_peaks_matched = [
            np.array([True, False, True, True]),
            np.array([False, False, True]),
            np.array([])
        ]
        result = calculate_num_peaks_matched(lib_peaks_matched)
        assert np.array_equal(result, [3, 1, 0])
    
    def test_fraction_library_intensity(self):
        """Test fraction of library intensity calculation."""
        spec_values = [
            np.array([0.1, 0.2, 0.3]),
            np.array([0.5]),
            np.array([])
        ]
        result = calculate_fraction_library_intensity(spec_values)
        assert np.allclose(result, [0.6, 0.5, 0.0])
    
    def test_fraction_dia_intensity(self):
        """Test fraction of DIA intensity calculation."""
        row_indices = [
            np.array([0, 2]),
            np.array([1]),
            np.array([])
        ]
        dia_spectrum = np.array([
            [100.0, 10.0],
            [200.0, 20.0],
            [300.0, 30.0]
        ])
        result = calculate_fraction_dia_intensity(row_indices, dia_spectrum)
        assert np.allclose(result, [40.0/60.0, 20.0/60.0, 0.0])
    
    def test_fraction_intensity_matched(self):
        """Test weighted intensity fraction calculation."""
        spec_values = [
            np.array([0.2, 0.3]),
            np.array([0.4])
        ]
        lib_coeffs = np.array([2.0, 0.5])
        result = calculate_fraction_intensity_matched(spec_values, lib_coeffs)
        assert np.allclose(result, [1.0, 0.2])


class TestErrorFeatures:
    """Test error-based feature calculations."""
    
    def test_ms1_error(self):
        """Test MS1 error extraction."""
        ms1_errors = np.array([1.0, 2.0, 3.0, 4.0])
        peaks_in_dia = [0, 2, 3]
        result = calculate_ms1_error(ms1_errors, peaks_in_dia)
        assert np.array_equal(result, [1.0, 3.0, 4.0])
    
    def test_rt_error(self):
        """Test RT error calculation."""
        prec_rt = 10.0
        rt_mz = np.array([
            [9.5, 100.0],
            [10.2, 200.0],
            [11.0, 300.0]
        ])
        window_idxs = np.array([0, 1, 2])
        peaks_in_dia = [0, 2]
        result = calculate_rt_error(prec_rt, rt_mz, window_idxs, peaks_in_dia)
        assert np.allclose(result, [0.5, -1.0])
    
    def test_residual_features(self):
        """Test residual feature calculation."""
        residuals = np.array([0.1, -0.2, 0.3, -0.4, 0.5])
        row_indices = [
            np.array([0, 1]),
            np.array([2, 3, 4]),
            np.array([])
        ]
        max_unmatched, max_matched = calculate_residual_features(residuals, row_indices)
        assert np.allclose(max_unmatched, [0.2, 0.5, 0.0])
        assert np.allclose(max_matched, [0.1, 0.5, 0.0])


class TestCorrelationFeatures:
    """Test correlation and similarity features."""
    
    def test_cosine_similarity(self):
        """Test cosine similarity calculation."""
        vec1 = np.array([1.0, 2.0, 3.0])
        vec2 = np.array([2.0, 4.0, 6.0])
        result = calculate_cosine_similarity(vec1, vec2)
        assert np.isclose(result, 1.0)  # Perfectly aligned vectors
        
        # Orthogonal vectors
        vec3 = np.array([1.0, 0.0])
        vec4 = np.array([0.0, 1.0])
        result2 = calculate_cosine_similarity(vec3, vec4)
        assert np.isclose(result2, 0.0)
    
    def test_spectral_contrast(self):
        """Test spectral contrast calculation."""
        pred = np.array([1.0, 2.0, 3.0])
        obs = np.array([1.0, 2.0, 3.0])
        result = calculate_spectral_contrast(pred, obs)
        assert np.isclose(result, 1.0)  # Perfect match
        
        # Opposite vectors
        pred2 = np.array([1.0, 0.0])
        obs2 = np.array([-1.0, 0.0])
        result2 = calculate_spectral_contrast(pred2, obs2)
        assert np.isclose(result2, -1.0)


class TestFragmentFeatures:
    """Test fragment ion features."""
    
    def test_count_b_y_ions(self):
        """Test counting b and y ions."""
        fragments = np.array(['b2', 'y3', 'b4', 'a2', 'y5', 'c3'])
        b_count, y_count = count_b_y_ions(fragments)
        assert b_count == 2
        assert y_count == 2
    
    def test_find_longest_y_series(self):
        """Test finding longest consecutive y-ion series."""
        # Consecutive series y3, y4, y5
        fragments = np.array(['b2', 'y3', 'y4', 'y5', 'y7', 'y8'])
        result = find_longest_y_series(fragments)
        assert result == 3
        
        # No consecutive series
        fragments2 = np.array(['y1', 'y3', 'y5', 'y7'])
        result2 = find_longest_y_series(fragments2)
        assert result2 == 1
    
    def test_hyperscore(self):
        """Test hyperscore calculation."""
        fragments = np.array(['b2', 'b3', 'y4', 'y5'])
        intensities = np.array([100.0, 200.0, 150.0, 250.0])
        score = calculate_hyperscore(fragments, intensities)
        assert score > 0  # Should be positive


class TestScoringFeatures:
    """Test advanced scoring features."""
    
    def test_manhattan_distance(self):
        """Test Manhattan distance calculation."""
        predicted = np.array([1.0, 2.0, 3.0])
        observed = np.array([1.1, 1.9, 3.2])
        result = calculate_manhattan_distance(predicted, observed)
        # log10(0.4/6.2) ≈ -1.19
        assert -2 < result < -1
    
    def test_goodness_of_fit(self):
        """Test goodness-of-fit calculation."""
        # Small residuals = good fit
        residuals = np.array([0.01, -0.02, 0.01])
        gof = calculate_goodness_of_fit(residuals, 3)
        assert gof > 0.9
        
        # Large residuals = poor fit
        residuals2 = np.array([1.0, -2.0, 3.0])
        gof2 = calculate_goodness_of_fit(residuals2, 3)
        assert gof2 < 0.1
    
    def test_scribe_score(self):
        """Test SCRIBE score calculation."""
        spec_values = np.array([0.3, 0.4, 0.3])
        dia_intensities = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        row_indices = np.array([0, 1, 2])
        score = calculate_scribe_score(spec_values, dia_intensities, row_indices)
        assert 0 <= score <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])