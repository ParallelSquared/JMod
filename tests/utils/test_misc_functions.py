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
import pandas as pd
import math

# Import the functions we want to test
from src.utils.misc_functions import (
    window_width, createTolWindows, within_tol, get_diff, ms1_error, moving_average, moving_auc, closest_feature, closest_ms1spec, closest_peak_diff, hyperscore_b_y, longest_y, cosim, frag_to_peak, specific_frags, ordered_frags, fragment_cor, unstring_floats
)
from tests.fixtures.test_data import SAMPLE_LIBRARY_ENTRY

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
        assert result == -10.0

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
        expected = np.array([95.0, 106.05])
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

# TODO: this gets changed
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

class TestMovingAverage:
    """Tests for moving_average function"""

    def test_basic_average(self):
        x = np.array([1, 2, 3, 4, 5])
        w = 3
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

class TestClosestFeature:
    """Test cases for closest_feature"""

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

    def test_closest_by_absolute_error(self):
        """Should pick feature with smallest absolute m/z error"""
        dino_features = Mock()
        dino_features.rtStart = np.array([10, 10])
        dino_features.rtEnd = np.array([20, 20])
        # Two candidate features within RT window
        dino_features.mz = np.array([99.5, 101.0])

        mz = 100.0
        rt = 15.0
        rt_tol = 5.0
        mz_tol = 0.05  # wide enough to include both

        # We expect it to pick 99.5 because |99.5-100| < |101-100|
        result = closest_feature(mz, rt, dino_features, rt_tol, mz_tol)
        
        # (99.5 - 100)/100 = -0.005
        assert np.isclose(result, -0.005, atol=1e-6)

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

    def test_pick_closest_positive_vs_negative(self):
        """Should choose the closest RT regardless of direction (absolute difference)"""
        ms1rt = np.array([9.5, 10.5])
        ms2rt = 10.0
        result = closest_ms1spec(ms2rt, ms1rt)
        # |10 - 9.5| = 0.5, |10 - 10.5| = 0.5 → tie, np.argmin picks first (index 0)
        assert result == 0

    def test_pick_negative_side_closer(self):
        """If one side is closer but smaller, should still pick it"""
        ms1rt = np.array([9.6, 11.0])
        ms2rt = 10.0
        result = closest_ms1spec(ms2rt, ms1rt)
        # |10 - 9.6| = 0.4 vs |10 - 11| = 1.0 → pick 9.6 → index 0
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
    
    def test_closest_peak_diff_absolute_error(self):
        """Should pick feature with smallest *absolute* fractional difference"""
        mz = 500.0
        spec_mz_list = np.array([499.75, 500.6])  # -0.0005 vs +0.0012
        # (499.75-500)/500 = -0.0005
        # (500.6-500)/500 = +0.0012
        result = closest_peak_diff(mz, spec_mz_list, max_diff=1e-2)
        assert np.isclose(result, -0.0005, atol=1e-10)

class TestHyperscoreBY:
    """Test cases for hyperscore_b_y without using patch"""

    def setup_method(self):
        self.frag_list = SAMPLE_LIBRARY_ENTRY["frags"]
        self.num_frags = len(self.frag_list)

    def test_some_matches(self):
        matches = np.array([True, True, False, True, True, False])
        hs, b_count, y_count = hyperscore_b_y(self.frag_list, matches)
        assert b_count > 0
        assert y_count > 0
        assert hs > 0

    def test_all_matches(self):
        matches = np.ones(self.num_frags, dtype=bool)
        hs, b_count, y_count = hyperscore_b_y(self.frag_list, matches)
        assert b_count > 0
        assert y_count > 0
        assert hs > 0

    def test_no_matches(self):
        matches = np.zeros(self.num_frags, dtype=bool)
        hs, b_count, y_count = hyperscore_b_y(self.frag_list, matches)
        assert hs == 0
        assert b_count == 0
        assert y_count == 0

class TestLongestY:
    """Test longest_y using sample fragment data"""

    def setup_method(self):
        self.frag_list = SAMPLE_LIBRARY_ENTRY["frags"]

    def test_some_y_matches(self):
        matches = np.array([True, False, False, True, True, False])
        result = longest_y(self.frag_list, matches)
        assert result > 0  # longest y ion index
        assert isinstance(result, int)

    def test_no_y_matches(self):
        matches = np.array([True, True, True, False, False, False])
        result = longest_y(self.frag_list, matches)
        assert result == 0

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

    def test_mismatched_lengths(self):
        """Vectors of different lengths should raise assertion"""
        x = np.array([1, 2])
        y = np.array([1, 2, 3])
        with pytest.raises(AssertionError):
            cosim(x, y)
    

class TestFragToPeak:
    """Test cases for the frag_to_peak function"""

    def test_frag_to_peak_sample_library(self):
        """Test converting SAMPLE_LIBRARY_ENTRY['frags'] to peak array"""
        frag_dict = SAMPLE_LIBRARY_ENTRY["frags"]
        peaks = frag_to_peak(frag_dict)

        mzs = peaks[:, 0]
        assert np.all(np.diff(mzs) > 0), "Peaks not sorted by m/z ascending"

        assert np.isclose(peaks[0][0], 227.1026)
        assert np.isclose(peaks[-1][0], 490.2872)

        expected_intensities = [frag_dict[k][1] for k in sorted(frag_dict, key=lambda x: frag_dict[x][0])]
        np.testing.assert_array_equal(peaks[:, 1], expected_intensities)

    def test_frag_to_peak_with_return_frags_sample_library(self):
        """Test frag_to_peak with return_frags=True using SAMPLE_LIBRARY_ENTRY"""
        frag_dict = SAMPLE_LIBRARY_ENTRY["frags"]
        peaks, ordered_frags = frag_to_peak(frag_dict, return_frags=True)

        mzs_sorted = [frag_dict[f][0] for f in ordered_frags]
        assert mzs_sorted == sorted(mzs_sorted), "Fragment order does not match m/z sorting"

        assert set(ordered_frags) == set(frag_dict.keys()), "Some fragment names missing"

        assert peaks.shape == (len(frag_dict), 2)

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

class TestOrderedFrags:
    """Test cases for ordered_frags function."""

    def test_ordering_from_sample_library(self):
        """Check that fragments from SAMPLE_LIBRARY_ENTRY are sorted by m/z."""
        frag_dict = SAMPLE_LIBRARY_ENTRY["frags"]

        result = ordered_frags(frag_dict)
        mzs = [v[0] for v in result.values()]
        assert np.all(np.diff(mzs) > 0)
        expected_order = [k for k, _ in sorted(frag_dict.items(), key=lambda x: x[1][0])]
        assert list(result.keys()) == expected_order

    def test_single_element(self):
        """Single-element dictionary remains unchanged."""
        frag_dict = {'b3_1': [329.16081698605, 0.25633228]}
        result = ordered_frags(frag_dict)
        assert list(result.keys()) == ['b3_1']
        assert np.allclose(result['b3_1'], [329.16081698605, 0.25633228])

    def test_ties_preserve_order(self):
        """Fragments with identical m/z preserve insertion order."""
        frag_dict = {
            'b1_1': [100.0, 0.5],
            'b2_1': [100.0, 0.2],
            'b3_1': [150.0, 0.1]
        }
        result = ordered_frags(frag_dict)
        keys = list(result.keys())

        assert keys[:2] == ['b1_1', 'b2_1']
        assert keys[2] == 'b3_1'

class TestFragmentCor:
    """Tests for fragment_cor function."""

    def _make_df_from_library(self, library_entry=None):
        """Helper to create a realistic fragment correlation DataFrame."""
        library_entry = SAMPLE_LIBRARY_ENTRY

        frags = library_entry["frags"]
        frag_names = ";".join(frags.keys())
        frag_int = ";".join(str(v[1]) for v in frags.values())
        obs_int = ";".join(str(v[1] * np.random.uniform(0.8, 1.2)) for v in frags.values())

        return pd.DataFrame({
            "frag_names": [frag_names],
            "obs_int": [obs_int],
            "frag_int": [frag_int],
        })

    def test_cosine_similarity(self):
        """Test cosine similarity with real fragment data."""
        df = self._make_df_from_library()
        result = fragment_cor(df, 0, fn="cos")
        assert isinstance(result, float)
        assert 0 <= result <= 1

    def test_pearson_correlation(self):
        """Test Pearson correlation with real fragment data."""
        df = self._make_df_from_library()
        result = fragment_cor(df, 0, fn="pearson")
        assert isinstance(result, float)
        assert -1 <= result <= 1

    def test_empty_shared_fragments(self):
        """Test when no shared fragments exist."""
        data = {
            "frag_names": ["b3_1;b4_1"],
            "obs_int": ["1.0;2.0"],
            "frag_int": ["3.0;4.0"]
        }
        df = pd.DataFrame(data)
        result = fragment_cor(df, 0, fn="cos")
        assert isinstance(result, float)


class TestUnstringFloats:
    """Test cases for unstring_floats function."""

    def test_basic_semicolon_delimited(self):
        """Test basic semicolon-delimited string."""
        result = unstring_floats("1.0;2.0;3.0")
        expected = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(result, expected)

    def test_custom_delimiter(self):
        """Test with custom delimiter."""
        result = unstring_floats("1.5,2.5,3.5", delim=",")
        expected = np.array([1.5, 2.5, 3.5])
        np.testing.assert_array_equal(result, expected)

    def test_single_value(self):
        """Test with single value (no delimiter)."""
        result = unstring_floats("42.0")
        expected = np.array([42.0])
        np.testing.assert_array_equal(result, expected)

    def test_negative_values(self):
        """Test with negative values."""
        result = unstring_floats("-1.0;-2.5;3.0")
        expected = np.array([-1.0, -2.5, 3.0])
        np.testing.assert_array_equal(result, expected)

    def test_scientific_notation(self):
        """Test with scientific notation."""
        result = unstring_floats("1e-5;2.5e3;3.0")
        expected = np.array([1e-5, 2.5e3, 3.0])
        np.testing.assert_array_equal(result, expected)
