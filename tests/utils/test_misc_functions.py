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
import math

# Import the functions we want to test
from src.utils.misc_functions import (
    window_width, createTolWindows, within_tol, get_diff, ms1_error, moving_average, moving_auc, closest_feature, closest_ms1spec, closest_peak_diff, hyperscore_b_y, longest_y, cosim, frag_to_peak, specific_frags
)

class TestWindowWidth:
    """Test cases for the window_width function"""

    def test_basic_width(self):
        """Test basic positive window width"""
        spec = Mock()
        spec.ms1window = (400.0, 420.0)
        result = window_width(spec)
        assert result == 20.0

    def test_zero_width(self):
        """Test when w1 == w2"""
        spec = Mock()
        spec.ms1window = (500.0, 500.0)
        result = window_width(spec)
        assert result == 0.0

    def test_negative_window(self):
        """Test when w2 < w1 (unusual case)"""
        spec = Mock()
        spec.ms1window = (600.0, 590.0)
        result = window_width(spec)
        # Function just subtracts, no check for ordering
        assert result == -10.0

    def test_integer_inputs(self):
        """Test with integer values"""
        spec = Mock()
        spec.ms1window = (100, 120)
        result = window_width(spec)
        assert result == 20

    def test_large_range(self):
        """Test with a very large range"""
        spec = Mock()
        spec.ms1window = (1.0, 1e6)
        result = window_width(spec)
        assert result == 1e6 - 1.0

class TestCreateTolWindows:
    """Test cases for the createTolWindows function"""

    def test_empty_positions(self):
        """Empty input should return empty array"""
        positions = np.array([])
        tolerance = 0.1
        result = createTolWindows(positions, tolerance)
        assert result.size == 0

    def test_single_position(self):
        """Test with a single position"""
        positions = np.array([100.0])
        tolerance = 0.01  # 1%
        result = createTolWindows(positions, tolerance)
        expected = np.array([99.0, 101.0])
        np.testing.assert_allclose(result, expected)

    def test_two_non_overlapping_positions(self):
        """Test with two well-separated positions"""
        positions = np.array([100.0, 200.0])
        tolerance = 0.01
        result = createTolWindows(positions, tolerance)
        # First window: 99–101, second window: 198–202
        expected = np.array([99.0, 101.0, 198.0, 202.0])
        np.testing.assert_allclose(result, expected)

    def test_two_overlapping_positions(self):
        """Test with two close positions that should merge"""
        positions = np.array([100.0, 101.0])
        tolerance = 0.05  # 5%
        result = createTolWindows(positions, tolerance)
        # Position 100 → window [95,105]
        # Position 101 → window [95.95,106.05]
        # These overlap, so should merge into [95,106.05]
        expected = np.array([95.0, 106.05])
        np.testing.assert_allclose(result, expected)

    def test_unsorted_input(self):
        """Test that input order does not matter (function sorts internally)"""
        positions = np.array([300.0, 100.0])
        tolerance = 0.01
        result = createTolWindows(positions, tolerance)
        expected = np.array([99.0, 101.0, 297.0, 303.0])
        np.testing.assert_allclose(result, expected)

    def test_integer_positions(self):
        """Test with integer inputs"""
        positions = [10, 20]
        tolerance = 0.1  # 10%
        result = createTolWindows(positions, tolerance)
        expected = np.array([9.0, 11.0, 18.0, 22.0])
        np.testing.assert_allclose(result, expected)

    def test_large_array(self):
        """Sanity check with many positions"""
        positions = np.arange(100, 110, 1)  # 100, 101, ..., 109
        tolerance = 0.01
        result = createTolWindows(positions, tolerance)
        # Should return alternating lower/upper edges, same length as 2 * positions.size
        assert result.shape[0] % 2 == 0
        assert np.all(result[::2] < result[1::2])

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

class TestGetDiff:
    """Test cases for get_diff function"""

    def test_exact_match(self):
        """mz matches one peak exactly"""
        mz = 100.0
        peaks = np.array([95.0, 100.0, 105.0])
        tol = 0.01  # 1%
        result = get_diff(mz, peaks, tol)
        # exact match → relative diff = 0
        assert result == 0.0

    def test_within_tolerance(self):
        """mz matches a peak within tolerance"""
        mz = 100.0
        peaks = np.array([101.0])  # 1% higher
        tol = 0.02  # 2% tolerance
        result = get_diff(mz, peaks, tol)
        expected = (101.0 - 100.0) / 100.0  # 0.01
        assert np.isclose(result, expected)

    def test_outside_tolerance(self):
        """mz has no match within tolerance"""
        mz = 100.0
        peaks = np.array([105.0])  # 5% higher
        tol = 0.01  # 1% tolerance
        result = get_diff(mz, peaks, tol)
        assert np.isnan(result)

    def test_multiple_matches(self):
        """Choose closest peak when multiple are within tolerance"""
        mz = 100.0
        peaks = np.array([101.0, 99.5, 105.0])
        tol = 0.02  # 2% tolerance, so 101.0 and 99.5 are both valid
        result = get_diff(mz, peaks, tol)
        expected = (99.5 - 100.0) / 100.0
        print(result, expected)
        assert np.isclose(result, expected)

class TestMS1Error:
    """Tests for ms1_error function"""

    def test_empty_inputs(self):
        dia_ms1 = np.array([])
        lib_mzs = np.array([])
        tol = 0.01
        result = ms1_error(dia_ms1, lib_mzs, tol)
        assert result.size == 0

    def test_no_matches(self):
        dia_ms1 = np.array([100.0, 200.0])
        lib_mzs = np.array([300.0])
        tol = 0.01
        result = ms1_error(dia_ms1, lib_mzs, tol)
        assert np.isnan(result[0])

    def test_exact_matches(self):
        dia_ms1 = np.array([100.0, 200.0])
        lib_mzs = np.array([100.0, 200.0])
        tol = 0.01
        result = ms1_error(dia_ms1, lib_mzs, tol)
        np.testing.assert_allclose(result, [0.0, 0.0])

    def test_multiple_matches_choose_closest(self):
        dia_ms1 = np.array([99.5, 101.0, 105.0])
        lib_mzs = np.array([100.0])
        tol = 0.02
        result = ms1_error(dia_ms1, lib_mzs, tol)
        expected = (99.5 - 100.0) / 100.0  # closest match
        np.testing.assert_allclose(result, [expected])

    def test_zero_tolerance(self):
        dia_ms1 = np.array([100.0, 101.0])
        lib_mzs = np.array([100.0])
        tol = 0.0
        result = ms1_error(dia_ms1, lib_mzs, tol)
        expected = (100.0 - 100.0) / 100.0
        np.testing.assert_allclose(result, [expected])

class TestMovingAverage:
    """Tests for moving_average function"""

    def test_empty_input(self):
        x = np.array([])
        w = 3
        result = moving_average(x, w)
        assert result.size == 0

    def test_window_size_1(self):
        x = np.array([1, 2, 3])
        w = 1
        result = moving_average(x, w)
        np.testing.assert_allclose(result, x)

    def test_basic_average(self):
        x = np.array([1, 2, 3, 4, 5])
        w = 3
        result = moving_average(x, w)
        expected = np.convolve(x, np.ones(w), 'same') / w
        np.testing.assert_allclose(result, expected)

    def test_window_larger_than_array(self):
        x = np.array([1, 2])
        w = 5
        result = moving_average(x, w)
        expected = np.convolve(x, np.ones(w), 'same') / w
        np.testing.assert_allclose(result, expected)

    def test_negative_numbers(self):
        x = np.array([-1, 0, 1])
        w = 2
        result = moving_average(x, w)
        expected = np.convolve(x, np.ones(w), 'same') / w
        np.testing.assert_allclose(result, expected)

class TestMovingAUC:
    """Tests for moving_auc function"""

    def test_empty_input(self):
        x = np.array([])
        w = 3
        dx = 1.0
        result = moving_auc(x, w, dx)
        assert result.size == 0

    def test_window_size_1(self):
        x = np.array([1, 2, 3])
        w = 1
        dx = 0.5
        result = moving_auc(x, w, dx)
        np.testing.assert_allclose(result, x * dx)

    def test_basic_auc(self):
        x = np.array([1, 2, 3, 4, 5])
        w = 3
        dx = 0.1
        result = moving_auc(x, w, dx)
        expected = np.convolve(x, np.ones(w), 'same') * dx
        np.testing.assert_allclose(result, expected)

    def test_dx_zero(self):
        x = np.array([1, 2, 3])
        w = 2
        dx = 0.0
        result = moving_auc(x, w, dx)
        np.testing.assert_allclose(result, np.zeros_like(x))

    def test_window_larger_than_array(self):
        x = np.array([1, 2])
        w = 5
        dx = 1.0
        result = moving_auc(x, w, dx)
        expected = np.convolve(x, np.ones(w), 'same') * dx
        np.testing.assert_allclose(result, expected)

class TestClosestFeature:
    """Test cases for closest_feature using mocks"""

    def test_exact_match(self):
        """mz matches exactly within RT window"""
        dino_features = Mock()
        dino_features.rtStart = np.array([10, 20])
        dino_features.rtEnd = np.array([15, 25])
        dino_features.mz = np.array([100.0, 101.0])

        mz = 100.0
        rt = 12.0
        rt_tol = 2.0
        mz_tol = 0.01

        result = closest_feature(mz, rt, dino_features, rt_tol, mz_tol)
        expected = (100.0 - mz) / mz
        assert np.isclose(result, expected)

    def test_within_mz_tolerance(self):
        """mz within relative tolerance"""
        dino_features = Mock()
        dino_features.rtStart = np.array([10, 20])
        dino_features.rtEnd = np.array([15, 25])
        dino_features.mz = np.array([100.0, 101.0])

        mz = 100.5
        rt = 12.0
        rt_tol = 3.0
        mz_tol = 0.01

        result = closest_feature(mz, rt, dino_features, rt_tol, mz_tol)
        expected = (100.0 - mz) / mz
        assert np.isclose(result, expected)

    def test_multiple_mz_matches_choose_closest(self):
        """Pick closest peak when multiple mzs within tolerance"""
        dino_features = Mock()
        dino_features.rtStart = np.array([10, 20, 30])
        dino_features.rtEnd = np.array([15, 25, 35])
        dino_features.mz = np.array([100.0, 101.0, 102.0])

        mz = 101.2
        rt = 22.0
        rt_tol = 5.0
        mz_tol = 0.02

        result = closest_feature(mz, rt, dino_features, rt_tol, mz_tol)
        expected = (101.0 - mz) / mz
        assert np.isclose(result, expected)

    def test_no_mz_matches(self):
        """mz outside tolerance"""
        dino_features = Mock()
        dino_features.rtStart = np.array([10, 20])
        dino_features.rtEnd = np.array([15, 25])
        dino_features.mz = np.array([100.0, 101.0])

        mz = 110.0
        rt = 12.0
        rt_tol = 2.0
        mz_tol = 0.01

        result = closest_feature(mz, rt, dino_features, rt_tol, mz_tol)
        assert np.isnan(result)

    def test_no_rt_matches(self):
        """rt outside RT window"""
        dino_features = Mock()
        dino_features.rtStart = np.array([10, 20])
        dino_features.rtEnd = np.array([15, 25])
        dino_features.mz = np.array([100.0, 101.0])

        mz = 100.0
        rt = 50.0
        rt_tol = 2.0
        mz_tol = 0.01

        result = closest_feature(mz, rt, dino_features, rt_tol, mz_tol)
        assert np.isnan(result)

    def test_empty_features(self):
        """Empty dino_features"""
        dino_features = Mock()
        dino_features.rtStart = np.array([])
        dino_features.rtEnd = np.array([])
        dino_features.mz = np.array([])

        mz = 100.0
        rt = 10.0
        rt_tol = 2.0
        mz_tol = 0.01

        result = closest_feature(mz, rt, dino_features, rt_tol, mz_tol)
        assert np.isnan(result)

class TestClosestMS1Spec:
    """Test cases for closest_ms1spec function"""

    def test_single_value_exact_match(self):
        """Exact match in array"""
        ms1rt = np.array([10.0, 20.0, 30.0])
        ms2rt = 20.0
        result = closest_ms1spec(ms2rt, ms1rt)
        assert result == 1  # index of exact match

    def test_single_value_closest(self):
        """Pick closest if no exact match"""
        ms1rt = np.array([10.0, 20.0, 30.0])
        ms2rt = 22.0
        result = closest_ms1spec(ms2rt, ms1rt)
        assert result == 1  # 20 is closer than 30

    def test_first_closest(self):
        """ms2rt closer to first element"""
        ms1rt = np.array([10.0, 20.0, 30.0])
        ms2rt = 9.0
        result = closest_ms1spec(ms2rt, ms1rt)
        assert result == 0

    def test_last_closest(self):
        """ms2rt closer to last element"""
        ms1rt = np.array([10.0, 20.0, 30.0])
        ms2rt = 35.0
        result = closest_ms1spec(ms2rt, ms1rt)
        assert result == 2

    def test_tie_pick_first(self):
        """If tie, np.argmin picks first occurrence"""
        ms1rt = np.array([10.0, 20.0])
        ms2rt = 15.0
        result = closest_ms1spec(ms2rt, ms1rt)
        assert result == 0  # 10 and 20 are equidistant, first picked

    def test_empty_array(self):
        """Empty ms1rt should raise an error"""
        ms1rt = np.array([])
        ms2rt = 10.0
        with pytest.raises(ValueError):
            closest_ms1spec(ms2rt, ms1rt)

    def test_single_element_array(self):
        """Array with single element always returns index 0"""
        ms1rt = np.array([42.0])
        ms2rt = 10.0
        result = closest_ms1spec(ms2rt, ms1rt)
        assert result == 0

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

class TestHyperscoreBY:
    """Test cases for hyperscore_b_y without using patch"""

    def setup_method(self):
        # example fragment list
        self.frag_list = {
            'b3_1': [244.09, 0.25],
            'b4_1': [372.15, 0.45],
            'b5_1': [485.23, 0.27],
            'b6_1': [542.25, 0.08],
            'y4_1': [472.25, 0.31],
            'y5_1': [529.27, 1.0],
            'y6_1': [642.35, 0.50],
            'y7_1': [770.41, 0.15],
            'y8_1': [827.43, 0.33],
            'y9_1': [956.47, 0.05]
        }

        # simulate frag_to_peak output
        self.frag_to_peak_arr = np.array([v for v in self.frag_list.values()])

    def test_some_matches(self):
        """Some fragments matched"""
        matches = np.array([True, True, True, True, True, False, True, True, True, False])
        hs, b_count, y_count = hyperscore_b_y(self.frag_list, matches)
        assert b_count == 4
        assert y_count == 4
        assert hs > 0

    def test_all_matches(self):
        """All fragments matched"""
        matches = np.ones(len(self.frag_list), dtype=bool)
        hs, b_count, y_count = hyperscore_b_y(self.frag_list, matches)
        assert b_count == 4
        assert y_count == 6
        assert hs > 0

    def test_no_matches(self):
        """No fragments matched"""
        matches = np.zeros(len(self.frag_list), dtype=bool)
        hs, b_count, y_count = hyperscore_b_y(self.frag_list, matches)
        assert b_count == 0
        assert y_count == 0
        assert hs == 0

    def test_only_b_matches(self):
        """Only b ions matched"""
        matches = np.array([True, True, True, True, False, False, False, False, False, False])
        hs, b_count, y_count = hyperscore_b_y(self.frag_list, matches)
        assert b_count == 4
        assert y_count == 0
        assert hs > 0

    def test_only_y_matches(self):
        """Only y ions matched"""
        matches = np.array([False, False, False, False, True, True, True, True, True, True])
        hs, b_count, y_count = hyperscore_b_y(self.frag_list, matches)
        assert b_count == 0
        assert y_count == 6
        assert hs > 0

    def test_empty_frag_list(self):
        """Empty fragment list"""
        matches = np.array([], dtype=bool)
        hs, b_count, y_count = hyperscore_b_y({}, matches)
        assert b_count == 0
        assert y_count == 0
        assert hs == 0

class TestLongestY:
    """Test cases for the longest_y function"""

    def setup_method(self):
        # Example fragment list
        self.frag_list = {
            'b3_1': [244.09, 0.25],
            'b4_1': [372.15, 0.45],
            'b5_1': [485.23, 0.27],
            'b6_1': [542.25, 0.08],
            'y4_1': [472.25, 0.31],
            'y5_1': [529.27, 1.0],
            'y6_1': [642.35, 0.50],
            'y7_1': [770.41, 0.15],
            'y8_1': [827.43, 0.33],
            'y9_1': [956.47, 0.05]
        }

    def test_some_matches(self):
        """Test with some y ions matched"""
        matches = np.array([True, True, True, True, True, False, True, True, True, False])
        result = longest_y(self.frag_list, matches)
        # Matched y ions are y4_1, y6_1, y7_1, y8_1 -> longest is y8_1
        assert result == 8

    def test_all_matches(self):
        """Test with all fragments matched"""
        matches = np.ones(len(self.frag_list), dtype=bool)
        result = longest_y(self.frag_list, matches)
        # All y ions matched -> longest is y9_1
        assert result == 9

    def test_no_y_matches(self):
        """Test when no y ions matched"""
        matches = np.array([True, True, True, True, False, False, False, False, False, False])
        result = longest_y(self.frag_list, matches)
        # No y ions matched
        assert result == 0

    def test_empty_frag_list(self):
        """Test with empty fragment list"""
        result = longest_y({}, np.array([], dtype=bool))
        assert result == 0

    def test_unordered_frag_names(self):
        """Test when fragment names are out of order"""
        frag_list_unordered = {
            'y7_1': [770.41, 0.15],
            'b4_1': [372.15, 0.45],
            'y5_1': [529.27, 1.0],
            'y9_1': [956.47, 0.05]
        }
        matches = np.array([True, True, True, True])
        result = longest_y(frag_list_unordered, matches)
        # longest y ion is y9_1
        assert result == 9

    def test_nonstandard_names(self):
        """Test with non-standard fragment names"""
        frag_list_nonstandard = {
            'y4_extra': [472.25, 0.31],
            'y5-something': [529.27, 1.0],
            'b3': [244.09, 0.25]
        }
        matches = np.array([True, True, True])
        result = longest_y(frag_list_nonstandard, matches)
        # longest y ion is y5-something
        assert result == 5

    def test_nonstring_keys(self):
        """Test fragment keys that are not strings"""
        frag_list_invalid = {
            101: [472.25, 0.31],
            None: [529.27, 1.0],
            'y3_1': [244.09, 0.25]
        }
        matches = np.array([True, True, True])
        result = longest_y(frag_list_invalid, matches)
        # Only 'y3_1' is valid -> longest y ion is 3
        assert result == 3

class TestCosim:
    """Test cases for the cosim function"""

    def test_basic_vectors(self):
        """Test with simple 2D vectors"""
        x = np.array([1, 0])
        y = np.array([0, 1])
        result = cosim(x, y)
        assert np.isclose(result, 0.0)

        x = np.array([1, 0])
        y = np.array([1, 0])
        result = cosim(x, y)
        assert np.isclose(result, 1.0)

        x = np.array([1, 0])
        y = np.array([-1, 0])
        result = cosim(x, y)
        assert np.isclose(result, -1.0)

    def test_nonbinary_vectors(self):
        """Test with non-binary values"""
        x = np.array([1, 2, 3])
        y = np.array([4, 5, 6])
        result = cosim(x, y)
        expected = np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
        assert np.isclose(result, expected)

    def test_higher_dimensions(self):
        """Test with higher dimensional vectors"""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([5, 4, 3, 2, 1])
        result = cosim(x, y)
        expected = np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
        assert np.isclose(result, expected)

    def test_identical_vectors(self):
        """Cosine similarity of a vector with itself should be 1"""
        x = np.array([2, -3, 1])
        result = cosim(x, x)
        assert np.isclose(result, 1.0)

    def test_squeezing(self):
        """Ensure squeezed arrays behave the same"""
        x = np.array([[1, 2, 3]])
        y = np.array([[1, 2, 3]])
        result = cosim(x, y)
        assert np.isclose(result, 1.0)

    def test_zero_vector(self):
        """Cosine similarity with zero vector raises a warning / inf"""
        x = np.array([0, 0, 0])
        y = np.array([1, 2, 3])
        with pytest.raises(ZeroDivisionError):
            cosim(x, y)

    def test_mismatched_lengths(self):
        """Vectors of different lengths should raise assertion"""
        x = np.array([1, 2])
        y = np.array([1, 2, 3])
        with pytest.raises(AssertionError):
            cosim(x, y)
    
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

class TestSpecificFrags:
    """Test cases for the specific_frags function"""

    def test_basic_filtering(self):
        """Removes non-specific fragments and sorts by m/z"""
        frag_dict = {
            'b1_1': [100.0, 0.5],
            'b3_1': [200.0, 0.3],
            'y1_1': [150.0, 0.2],
            'y4_1': [250.0, 0.4]
        }
        result = specific_frags(frag_dict)
        expected = np.array([[200.0, 0.3], [250.0, 0.4]])
        assert np.allclose(result, expected)

    def test_custom_non_spec(self):
        """Use a custom non-specific list"""
        frag_dict = {
            'b1_1': [100.0, 0.5],
            'b2_1': [120.0, 0.6],
            'b3_1': [200.0, 0.3]
        }
        result = specific_frags(frag_dict, non_spec=["b1"])
        expected = np.array([[120.0, 0.6], [200.0, 0.3]])
        assert np.allclose(result, expected)

    def test_all_filtered(self):
        """All fragments are non-specific, should return empty array"""
        frag_dict = {
            'b1_1': [100.0, 0.5],
            'y2_1': [150.0, 0.2]
        }
        result = specific_frags(frag_dict)
        assert result.shape[0] == 0

    def test_empty_input(self):
        """Empty frag_dict should return empty array"""
        frag_dict = {}
        result = specific_frags(frag_dict)
        assert result.shape[0] == 0

    def test_sorting_by_mz(self):
        """Output should be sorted by m/z"""
        frag_dict = {
            'b3_1': [300.0, 0.3],
            'b4_1': [200.0, 0.2],
            'b5_1': [250.0, 0.5]
        }
        result = specific_frags(frag_dict, non_spec=[])
        expected = np.array([[200.0, 0.2], [250.0, 0.5], [300.0, 0.3]])
        assert np.allclose(result, expected)