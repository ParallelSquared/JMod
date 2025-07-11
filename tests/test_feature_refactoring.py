"""
Tests for the refactored feature calculation modules.
"""

import numpy as np
import pytest
from dataclasses import dataclass

# Import the new modules
from src.features.basic_features import BasicFeatureInputs, calculate_basic_features
from src.features.ms1_features import MS1FeatureInputs, calculate_ms1_features
from src.features.spectral_features import SpectralFeatureInputs, calculate_spectral_features
from src.features.statistical_features import StatisticalFeatureInputs, calculate_statistical_features
from src.features.feature_aggregator import FeatureCalculator, FeatureCalculatorInputs


class TestBasicFeatures:
    """Test basic feature calculations."""
    
    def test_calculate_basic_features(self):
        """Test basic feature calculation."""
        # Create mock input data
        inputs = BasicFeatureInputs(
            n_candidates=2,
            peaks_in_dia=[0, 1],
            lib_peaks_matched=[np.array([True, True, False]), np.array([True, False])],
            spec_values_split=[np.array([0.5, 0.3, 0.2]), np.array([0.7, 0.3])],
            spec_row_indices_split=[np.array([0, 1, 2]), np.array([3, 4])],
            dia_spectrum=np.array([[0, 10], [1, 20], [2, 30], [3, 15], [4, 25]]),
            dia_total_intensity=100.0,
            lib_coefficients=np.array([0.8, 0.6]),
            ms1_error=np.array([0.1, 0.2]),
            rt_mz=np.array([[1.0, 500], [2.0, 600]]),
            window_idxs=np.array([0, 1]),
            prec_rt=1.5
        )
        
        features = calculate_basic_features(inputs)
        
        # Check shape
        assert features.shape == (2, 7)
        
        # Check feature 0 (num peaks matched)
        assert features[0, 0] == 2  # Two True values
        assert features[1, 0] == 1  # One True value
        
        # Check feature 1 (frac lib intensity)
        assert features[0, 1] == 1.0  # sum([0.5, 0.3, 0.2])
        assert features[1, 1] == 1.0  # sum([0.7, 0.3])
        
        # Check feature 3 (MS1 error)
        assert features[0, 3] == 0.1
        assert features[1, 3] == 0.2
        
        # Check feature 4 (RT error)
        assert features[0, 4] == 0.5  # 1.5 - 1.0
        assert features[1, 4] == -0.5  # 1.5 - 2.0


class TestMS1Features:
    """Test MS1 feature calculations."""
    
    def test_calculate_r_squared(self):
        """Test R² calculation."""
        from src.features.ms1_features import calculate_r_squared
        
        # Perfect correlation
        observed = np.array([1, 2, 3, 4, 5])
        predicted = np.array([1, 2, 3, 4, 5])
        r2 = calculate_r_squared(observed, predicted)
        assert np.isclose(r2, 1.0)
        
        # No correlation
        observed = np.array([1, 2, 3, 4, 5])
        predicted = np.array([5, 5, 5, 5, 5])
        r2 = calculate_r_squared(observed, predicted)
        assert r2 < 0.5
        
        # Empty arrays
        r2 = calculate_r_squared(np.array([]), np.array([]))
        assert r2 == 0.0


class TestSpectralFeatures:
    """Test spectral feature calculations."""
    
    def test_parse_fragment_ions(self):
        """Test fragment ion parsing."""
        from src.features.spectral_features import parse_fragment_ions
        
        frag_names = ['b2', 'y3', 'b4', 'y5', 'a2']
        b_ions, y_ions = parse_fragment_ions(frag_names)
        
        assert b_ions == ['b2', 'b4']
        assert y_ions == ['y3', 'y5']
    
    def test_calculate_longest_y_series(self):
        """Test longest y-ion series calculation."""
        from src.features.spectral_features import calculate_longest_y_series
        
        # Consecutive series
        y_ions = ['y2', 'y3', 'y4', 'y6']
        longest = calculate_longest_y_series(y_ions)
        assert longest == 3  # y2, y3, y4
        
        # Non-consecutive
        y_ions = ['y1', 'y3', 'y5']
        longest = calculate_longest_y_series(y_ions)
        assert longest == 1
        
        # Empty
        longest = calculate_longest_y_series([])
        assert longest == 0


class TestStatisticalFeatures:
    """Test statistical feature calculations."""
    
    def test_calculate_frac_intensity_with_cutoff(self):
        """Test fraction intensity with cutoff."""
        from src.features.statistical_features import calculate_frac_intensity_with_cutoff
        
        # Above cutoff
        result = calculate_frac_intensity_with_cutoff(0.8, 0.2, cutoff=0.1)
        assert result == 0.8
        
        # Below cutoff
        result = calculate_frac_intensity_with_cutoff(0.8, 0.05, cutoff=0.1)
        assert result == 0.0
    
    def test_calculate_fdr(self):
        """Test FDR calculation."""
        from src.features.statistical_features import calculate_fdr
        
        p_values = np.array([0.01, 0.04, 0.03, 0.05, 0.20])
        fdr = calculate_fdr(p_values)
        
        # Check properties of FDR
        assert len(fdr) == len(p_values)
        assert np.all(fdr >= p_values)  # FDR >= p-value
        assert np.all(fdr <= 1.0)  # FDR <= 1


class TestFeatureAggregator:
    """Test feature aggregator."""
    
    def test_aggregate_features(self):
        """Test feature aggregation."""
        from src.features.feature_aggregator import aggregate_features
        
        basic = np.ones((3, 7))
        ms1 = np.ones((3, 5)) * 2
        spectral = np.ones((3, 10)) * 3
        statistical = np.ones((3, 3)) * 4
        
        features = aggregate_features(basic, ms1, spectral, statistical)
        
        assert features.shape == (3, 26)
        assert np.all(features[:, 0:7] == 1)
        assert np.all(features[:, 7:12] == 2)
        assert np.all(features[:, 12:22] == 3)
        assert np.all(features[:, 22:24] == 4)  # 22-23 are from statistical
        assert np.all(features[:, 24] == 0)  # Feature 24 is placeholder
        assert np.all(features[:, 25] == 4)  # Feature 25 is from statistical
    
    def test_feature_calculator_empty(self):
        """Test feature calculator with empty input."""
        inputs = FeatureCalculatorInputs(
            candidates=[],
            peaks_in_dia=[],
            is_decoy_matched=np.array([]),
            spec_values_split=[],
            spec_row_indices_split=[],
            spec_col_indices_split=[],
            lib_peaks_matched=[],
            dia_spectrum=np.array([[0, 0]]),
            prec_rt=0.0,
            lib_coefficients=np.array([]),
            rt_mz=np.array([]),
            window_idxs=np.array([]),
            library={},
            ms1_error_array=None,
            frag_names=None,
            sparse_matrix=None,
            residuals=None,
            y_pred=None
        )
        
        calculator = FeatureCalculator()
        features = calculator.calculate_all_features(inputs)
        
        assert features.shape == (0, 26)
        assert len(calculator.feature_names) == 26


if __name__ == "__main__":
    pytest.main([__file__, "-v"])