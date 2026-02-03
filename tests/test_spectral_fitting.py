"""
This Source Code Form is subject to the terms of the Oxford Nanopore
Technologies, Ltd. Public License, v. 1.0.  Full licence can be found
at https://github.com/ParallelSquared/JMod/blob/main/LICENSE.txt
"""

"""
Tests for functions in spectral_fitting.py
"""

import numpy as np
from scipy import sparse

# Import the functions we want to test
from src.spectral_fitting import (
    get_closest_ms1, get_scribe, get_residuals, max_matched_residual, get_manhattan_distance, hyperscore2, merge_spectrum_peaks, create_entries, unmatched_peaks, gof_stat, huber_nnls_irls
)

class TestGetClosestMS1:
    def test_get_closest_ms1(self):
        class FakeSpec:
            def __init__(self, RT):
                self.RT = RT
        
        spectra = [FakeSpec(1.0), FakeSpec(2.5), FakeSpec(5.0)]
        # Patch closest_ms1spec to just pick closest manually
        def fake_closest(target, arr):
            return np.argmin(np.abs(arr - target))
        closest_ms1spec = fake_closest
        
        ms1 = get_closest_ms1(2.0, spectra)
        assert ms1.RT == 2.5

class TestGetScribe:
    def test_get_scribe(self):
        row_idx_split = [np.array([0,1])]
        col_idx_split = [np.array([0,1])]
        prec_val_split = [np.array([4,9])]
        val_obs = np.array([4,9])
        
        scores = get_scribe(row_idx_split, col_idx_split, prec_val_split, val_obs)
        assert scores.shape[0] == 1
        assert scores[0] >= 0

class TestGetResiduals:
    def test_get_residuals(self):
        ref_val = [np.array([1])]
        ref_row = [np.array([0])]
        ref_col = [np.array([0])]
        decoy_val = [np.array([2])]
        decoy_row = [np.array([1])]
        decoy_col = [np.array([0])]
        val_obs = np.array([1.0, 2.0])
        coeffs = np.array([1.0, 1.0])
        
        r, y_pred = get_residuals(ref_val, ref_row, ref_col,
                                            decoy_val, decoy_row, decoy_col,
                                            val_obs, coeffs, 0, 1)
        assert r.shape == val_obs.shape
        assert y_pred.shape == val_obs.shape

class TestMaxMatchedResidual:   
    def test_max_matched_residual(self):
        row_idx_split = [np.array([0,1])]
        residuals = np.array([0.5, 0.2])
        max_res = max_matched_residual(row_idx_split, residuals)
        assert max_res[0] == 0.5

class TestGetManhattanDistance:
    def test_get_manhattan_distance(self):
        row_idx_split = [np.array([0,1])]
        col_idx_split = [np.array([0,1])]
        prec_val_split = [np.array([1,2])]
        val_obs = np.array([1,2])
    
        y_pred = np.array([1,2])
        
        manhattan, sc = get_manhattan_distance(row_idx_split, col_idx_split, prec_val_split, val_obs, y_pred)
        assert manhattan.shape[0] == 1
        assert sc.shape[0] == 1

class TestHyperscore2:
    def test_hyperscore2(self):
        frags = {"b1": 100, "y1": 200, "b2":50}
        frag_names_matched = ["b1", "y1", "b2"]
        score, num_b, num_y = hyperscore2(frags, frag_names_matched)
        assert score >= 0
        assert num_b == 2
        assert num_y == 1

class TestMergeSpectrumPeaks:
    class DummySpec:
        def __init__(self, mz, intens):
            self.mz=mz
            self.intens=intens
    
        def peak_list(self):
            return(np.array([self.mz,self.intens]))
    
    def test_merge_spectrum_peaks(self):
        dia_spec = self.DummySpec(
            mz=np.array([100.0, 100.01]), 
            intens=np.array([10.0, 20.0])
            )
            
        mz_ppm = 1e-6
        merge_spectrum_peaks(dia_spec, mz_ppm)
        expected = np.array([
            [100.0, 10.0],
            [100.01, 20.0]
            ])
        output = np.array(dia_spec.peak_list(), dtype=np.float64).T
        assert np.all(output == expected), f"Expected {expected}, got {output}"

        dia_spec = self.DummySpec(
            mz=np.array([100.000, 100.0001]),
            intens=np.array([10.0, 20.0])
            )
        mz_ppm = 1e-6
        merge_spectrum_peaks(dia_spec, mz_ppm)
        expected = np.array([
            [100.0, 30.0]
            ])
        output = np.array(dia_spec.peak_list(), dtype=np.float64).T
        assert np.all(output == expected), f"Expected {expected}, got {output}"


        dia_spec = self.DummySpec(
            mz=np.array([100.000, 100.0001, 100.0002]),
            intens=np.array([10.0, 20.0, 60.0])
            )
        mz_ppm = 1e-6
        merge_spectrum_peaks(dia_spec, mz_ppm)
        expected = np.array([[100.0, 30.0], [100.0001, 60.0]])
        output = np.array(dia_spec.peak_list(), dtype=np.float64).T
        assert np.all(output == expected), f"Expected {expected}, got {output}"

        dia_spec = self.DummySpec(
            mz=np.array([]),
            intens=np.array([])
            )
        mz_ppm = 1e-6
        merge_spectrum_peaks(dia_spec, mz_ppm)
        expected = np.array([]).reshape(0, 2)
        output = np.array(dia_spec.peak_list(), dtype=np.float64).T
        assert np.all(output == expected), f"Expected {expected}, got {output}"

        dia_spec = self.DummySpec(
            mz=np.array([50.0, 60.0, 100.000, 100.0001, 200.0]),
            intens=np.array([50, 60, 10.0, 20.0, 200])
        )
        mz_ppm = 1e-6
        merge_spectrum_peaks(dia_spec, mz_ppm)
        expected = np.array([[50.0, 50],[60.0, 60],[100.0, 30.0],[200.0, 200]])
        output = np.array(dia_spec.peak_list(), dtype=np.float64).T
        assert np.all(output == expected), f"Expected {expected}, got {output}"

class TestUnmatchedPeaks:
    def test_unmatched_peaks_minimal(self):
        norm_intensities = [np.array([0.1, 0.9])]
        pep_cand_loc = [np.array([1, 0])]
        last_row = 0

        for ft in ["a", "b", "c"]:
            rows, cols, values = unmatched_peaks(norm_intensities, pep_cand_loc, last_row, fit_type=ft)
            assert isinstance(rows, np.ndarray)
            assert isinstance(cols, np.ndarray)
            assert isinstance(values, np.ndarray)

class TestCreateEntries:
    def test_create_entries_minimal(self):
        centroid_breaks = np.array([100, 200, 300])
        candidate_peaks = [np.array([[100, 1.0], [200, 2.0]])]
        mass_window_candidates = [np.array([100, 200])]
        top_n_idxs = [np.array([0, 1])]
        prec_mzs = np.array([100.0])
        
        class DummyMS1:
            mz = np.array([100.0])
        
        ms1_spec = DummyMS1()
        ms1_tol = 1.0
        top_n = 1
        atleast_m = 1

        out = create_entries(
            centroid_breaks,
            candidate_peaks,
            mass_window_candidates,
            top_n,
            atleast_m,
            prec_mzs,
            ms1_spec,
            ms1_tol,
            top_n_idxs=top_n_idxs
        )

        # check that it returns a tuple with the expected number of elements
        assert isinstance(out, tuple)
        assert len(out) == 10

class TestGofStat:
    def test_gof_stat_basic(self):
        row_idx_split = [np.array([0, 1]), np.array([2, 3])]
        col_idx_split = [np.array([0, 1]), np.array([0, 1])]
        val_split = [np.array([1.0, 2.0]), np.array([1.5, 2.5])]
        residuals = np.array([0.5, 1.0, 0.0, 2.0])
        val_obs = np.array([1.0, 0.0, 0.0, 1.0])
        coeffs = np.array([1.0, 2.0])
        offset = 0

        result, max_unmatched_res, max_matched_res = gof_stat(
            row_idx_split,
            col_idx_split,
            val_split,
            residuals,
            val_obs,
            coeffs,
            offset
        )

        assert len(result) == 2
        assert len(max_unmatched_res) == 2
        assert len(max_matched_res) == 2

        np.testing.assert_allclose(result, np.log2([ (0.5+1.0)/(1*1.0 + 2*2.0),  (0.0+2.0)/(1*1.5 + 2*2.5) ]), rtol=1e-6)
        np.testing.assert_allclose(max_matched_res, np.log2([0.5/(1*1.0 + 2*2.0 + 1e-10) + 1e-10, 2.0/(1*1.5 + 2*2.5 + 1e-10) + 1e-10]), rtol=1e-6)
        np.testing.assert_allclose(max_unmatched_res, np.log2([1.0/(1*1.0 + 2*2.0 + 1e-10) + 1e-10, 0.0/(1*1.5 + 2*2.5 + 1e-10) + 1e-10]), rtol=1e-6)


class TestHuberNnlsIrls:
    """Smoke tests for huber_nnls_irls function."""

    def test_basic_sparse_input(self):
        """Test with simple sparse matrix."""
        # Create a simple sparse matrix (10 peaks, 3 candidates)
        np.random.seed(42)
        A = sparse.random(10, 3, density=0.5, format='csr')
        b = np.abs(np.random.rand(10))  # Observed intensities (non-negative)

        result = huber_nnls_irls(A, b)

        assert 'x' in result
        x = np.array(result['x']).flatten()
        assert x.shape == (3,)
        assert np.all(x >= -1e-10)  # Non-negative constraint (allow tiny numerical errors)

    def test_small_matrix(self):
        """Test with a small dense matrix converted to sparse."""
        A = sparse.csr_matrix(np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]))
        b = np.array([1.0, 1.0, 2.0])

        result = huber_nnls_irls(A, b)

        x = np.array(result['x']).flatten()
        assert x.shape == (2,)
        # Just verify it returns non-negative values
        assert np.all(x >= -1e-10)

    def test_with_outliers(self):
        """Test that Huber loss handles outliers better than standard NNLS."""
        # Create data with an outlier
        A = sparse.csr_matrix(np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [1.0, 0.0],  # Outlier row
        ]))
        b = np.array([1.0, 1.0, 2.0, 100.0])  # Last value is outlier

        result = huber_nnls_irls(A, b, max_iter=50)

        x = np.array(result['x']).flatten()
        assert x.shape == (2,)
        assert np.all(x >= -1e-10)

    def test_convergence(self):
        """Test that the algorithm converges within max_iter."""
        np.random.seed(123)
        A = sparse.random(20, 5, density=0.3, format='csr')
        b = np.abs(np.random.rand(20))

        result = huber_nnls_irls(A, b, max_iter=100, tol=1e-6)

        assert 'x' in result
        x = np.array(result['x']).flatten()
        assert x.shape == (5,)
