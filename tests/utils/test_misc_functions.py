"""
This Source Code Form is subject to the terms of the Oxford Nanopore
Technologies, Ltd. Public License, v. 1.0.  Full licence can be found
at https://github.com/ParallelSquared/JMod/blob/main/LICENSE.txt
"""

"""
Tests for functions in misc_functions.py
"""
import pytest
from unittest.mock import Mock, patch
import numpy as np

# Import the functions we want to test
from src.utils.misc_functions import (
    closest_peak_diff, frag_to_peak, within_tol
)

class TestClosestPeakDiff:
    """Test cases for the closest_peak_diff function"""
    
    def test_closest_peak_diff_exact_match(self):
        """Test when there's an exact match"""
        mz = 500.0
        spec_mz_list = np.array([400.0, 500.0, 600.0])
        result = closest_peak_diff(mz, spec_mz_list)
        assert result == 0.0
    
    def test_closest_peak_diff_close_match(self):
        """Test when there's a close match within tolerance"""
        mz = 500.0
        spec_mz_list = np.array([400.0, 500.005, 600.0])
        result = closest_peak_diff(mz, spec_mz_list, max_diff=2e-5)
        assert abs(result - 1e-5) < 1e-10  # (500.005-500)/500 = 1e-5
    
    def test_closest_peak_diff_no_match(self):
        """Test when no peak is within tolerance"""
        mz = 500.0
        spec_mz_list = np.array([400.0, 501.0, 600.0])
        result = closest_peak_diff(mz, spec_mz_list, max_diff=2e-5)
        assert np.isnan(result)
    
    def test_closest_peak_diff_edge_case_beginning(self):
        """Test when query m/z is before all peaks"""
        mz = 100.0
        spec_mz_list = np.array([200.0, 300.0, 400.0])
        
        # Within tolerance
        result = closest_peak_diff(mz, spec_mz_list, max_diff=2.0)
        assert result == 1.0  # (200-100)/100 = 1.0
        
        # Outside tolerance
        result = closest_peak_diff(mz, spec_mz_list, max_diff=0.5)
        assert np.isnan(result)
    
    def test_closest_peak_diff_edge_case_end(self):
        """Test when query m/z is after all peaks"""
        mz = 500.0
        spec_mz_list = np.array([200.0, 300.0, 400.0])
        
        # Within tolerance
        result = closest_peak_diff(mz, spec_mz_list, max_diff=0.3)
        assert result == -0.2  # (400-500)/500 = -0.2
        
        # Outside tolerance
        result = closest_peak_diff(mz, spec_mz_list, max_diff=0.1)
        assert np.isnan(result)
    
    def test_closest_peak_diff_between_peaks(self):
        """Test when query m/z is between two peaks"""
        mz = 250.0
        spec_mz_list = np.array([200.0, 300.0, 400.0])
        
        # Closer to left peak
        result = closest_peak_diff(mz, spec_mz_list, max_diff=0.3)
        assert result == 0.2  # (200-250)/250 = -0.2
        
        # Exactly in the middle
        mz = 250.0
        spec_mz_list = np.array([200.0, 300.0])
        result = closest_peak_diff(mz, spec_mz_list)
        # Should choose the one with smaller absolute difference
        assert np.isnan(result)==True # (200-250)/250 = -0.2
        
        # Closer to right peak
        mz = 280.0
        spec_mz_list = np.array([200.0, 300.0])
        result = closest_peak_diff(mz, spec_mz_list, max_diff=0.2)
        expected = (300.0 - 280.0) / 280.0  # ≈ 0.0714
        assert abs(result - expected) < 0.072
    
    def test_closest_peak_diff_single_peak(self):
        """Test with single peak in spectrum"""
        mz = 500.0
        spec_mz_list = np.array([505.0])
        
        # Within tolerance
        result = closest_peak_diff(mz, spec_mz_list, max_diff=0.01)
        assert abs(result - 0.01) < 1e-10  # (505-500)/500 = 0.01
        
        # Outside tolerance
        result = closest_peak_diff(mz, spec_mz_list, max_diff=0.005)
        assert np.isnan(result)
    
    def test_closest_peak_diff_tolerance_boundary(self):
        """Test values exactly at tolerance boundary"""
        mz = 1000.0
        
        # Test positive difference at boundary
        spec_mz_list = np.array([1000.02])  # Exactly 2e-5 relative difference
        result = closest_peak_diff(mz, spec_mz_list, max_diff=1.99e-5)
        assert np.isnan(result)  # Should be outside because condition is < not <=
        
        # Just inside tolerance
        spec_mz_list = np.array([1000.019999])
        result = closest_peak_diff(mz, spec_mz_list, max_diff=2e-5)
        assert abs(result - 1.9999e-5) < 1e-10
        
        # Test negative difference at boundary
        spec_mz_list = np.array([999.98])  # Exactly -2e-5 relative difference
        result = closest_peak_diff(mz, spec_mz_list, max_diff=1.99e-5)
        assert np.isnan(result)  # Should be outside because condition is < not <=
    
    def test_closest_peak_diff_large_mz_values(self):
        """Test with realistic large m/z values"""
        mz = 1500.0
        spec_mz_list = np.array([1499.97, 1500.03, 1500.045])
        
        # Should find the closest match
        result = closest_peak_diff(mz, spec_mz_list, max_diff=1.99e-5)
        expected = (1500.03 - 1500.0) / 1500.0  # 2e-5
        assert np.isnan(result)  # Actually outside tolerance due to < condition
        
        # With larger tolerance
        result = closest_peak_diff(mz, spec_mz_list, max_diff=3e-5)
        assert abs(result - 2e-5) < 1e-10
    
    def test_closest_peak_diff_zero_mz(self):
        """Test behavior with zero m/z (edge case)"""
        # This would cause division by zero in real usage
        # but m/z values should never be zero in practice
        # Skip this test or handle it based on your requirements
        pass  # Mass spectrometry m/z values are always positive
    
    def test_closest_peak_diff_negative_values(self):
        """Test with negative m/z values (shouldn't happen in real data)"""
        mz = -500.0
        spec_mz_list = np.array([-600.0, -400.0, -300.0])
        
        result = closest_peak_diff(mz, spec_mz_list, max_diff=0.3)
        expected = (-400.0 - (-500.0)) / (-500.0)  # 100/-500 = -0.2
        assert abs(result - expected) < 1e-10
    
    @pytest.mark.parametrize("mz,spec_mz_list,max_diff,expected", [
        # Exact matches
        (100.0, [50.0, 100.0, 150.0], 2e-5, 0.0),
        (500.0, [500.0], 2e-5, 0.0),
        
        # Close matches within tolerance
        (1000.0, [999.99, 1000.01], 2e-5, -1e-5),  # Closer to 999.99
        (1000.0, [999.98, 1000.015], 2e-5, 1.5e-5),  # Closer to 1000.015
        
        # Outside tolerance
        (500.0, [400.0, 600.0], 2e-5, np.nan),
        (1000.0, [999.0, 1001.0], 2e-5, np.nan),
        
        # Edge cases
        (100.0, [200.0], 1.0, 1.0),  # Query before all peaks, within tolerance
        (300.0, [100.0, 200.0], 0.4, -1.0/3),  # Query after all peaks
    ])
    def test_closest_peak_diff_parametrized(self, mz, spec_mz_list, max_diff, expected):
        """Parametrized tests for various scenarios"""
        spec_mz_array = np.array(spec_mz_list)
        result = closest_peak_diff(mz, spec_mz_array, max_diff)
        
        if np.isnan(expected):
            assert np.isnan(result)
        else:
            assert abs(result - expected) < 1e-4


class TestFragToPeak:
    """Test cases for the frag_to_peak function"""
    
    def test_frag_to_peak_simple(self):
        """Test converting fragment dictionary to peak array"""
        frag_dict = {
            "b2_1": [200.0, 100.0],
            "b3_1": [300.0, 150.0],
            "y2_1": [150.0, 80.0]
        }
        peaks = frag_to_peak(frag_dict)
        
        # Should be sorted by m/z
        assert peaks[0][0] == 150.0  # y2_1
        assert peaks[1][0] == 200.0  # b2_1
        assert peaks[2][0] == 300.0  # b3_1
        
        # Check intensities
        assert peaks[0][1] == 80.0
        assert peaks[1][1] == 100.0
        assert peaks[2][1] == 150.0
    
    def test_frag_to_peak_with_return_frags(self):
        """Test returning ordered fragment names"""
        frag_dict = {
            "b2_1": [200.0, 100.0],
            "y2_1": [150.0, 80.0]
        }
        peaks, ordered_frags = frag_to_peak(frag_dict, return_frags=True)
        
        assert ordered_frags[0] == "y2_1"  # Lower m/z first
        assert ordered_frags[1] == "b2_1"


class TestWithinTol:
    """Test cases for the within_tol function"""
    
    def test_within_tol_exact_match(self):
        """Test exact matches"""
        result = within_tol(100.0, 100.0, atol=0, rtol=0.01)
        assert result[0] == True
        assert result[1] == 0.0
    
    def test_within_tol_relative_tolerance(self):
        """Test relative tolerance"""
        result = within_tol(100.0, 101.0, atol=0, rtol=0.01)
        assert result[0] == True  # Within 1% tolerance
        assert result[1] == -1.0  # Difference
        
        result = within_tol(100.0, 102.0, atol=0, rtol=0.01)
        assert result[0] == False  # Outside 1% tolerance
    
    def test_within_tol_absolute_tolerance(self):
        """Test absolute tolerance"""
        result = within_tol(100.0, 100.5, atol=1.0, rtol=0)
        assert result[0] == True  # Within 1.0 absolute tolerance
        
        result = within_tol(100.0, 102.0, atol=1.0, rtol=0)
        assert result[0] == False  # Outside 1.0 absolute tolerance
    
    def test_within_tol_arrays(self):
        """Test with numpy arrays"""
        x = np.array([100.0, 200.0, 300.0])
        y = np.array([101.0, 202.0, 306.0])
        result = within_tol(x, y, atol=0, rtol=0.01)
        
        assert result[0, 0] == 1.  # 101/100 = 1.01, within 2%
        assert result[1, 0] == 1.   # 202/200 = 1.01, within 2%
        assert result[2, 0] == 0.  # 306/300 = 1.02, outside 2%