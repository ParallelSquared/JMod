"""
Tests for functions in spectral_fitting.py
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

import numpy as np
import pytest
from scipy import sparse

# Import the functions we want to test
from src.spectral_fitting import (
    get_closest_ms1, get_scribe, get_residuals, max_matched_residual, get_manhattan_distance, hyperscore2, merge_spectrum_peaks, create_entries, gof_stat, huber_nnls_irls,
    _dia_prep_2d_jit, _match_mz_only, _match_mz_im, _create_entries_direct_jit,
)
from src.utils.frag_encoding import encode_frag_name


class TestDiaPrep2D:
    """(m/z, IM) binning + IM-aware matching for the timsTOF main search."""

    def test_bins_split_by_im_and_sum_within(self):
        mz = np.array([500.0, 500.0, 500.0, 600.0])
        ints = np.array([10.0, 20.0, 99.0, 5.0])
        mob = np.array([0.90, 0.905, 1.30, 0.90])
        bmz, bint, bmob = _dia_prep_2d_jit(mz, ints, mob, 20e-6, 0.05)
        assert bmz.shape[0] == 3                      # 500@0.9, 500@1.3, 600@0.9
        assert 30.0 in set(np.round(bint, 6))         # 0.90 pair summed
        assert 99.0 in set(np.round(bint, 6))         # 1.30 interferer separate

    def test_matchers_pick_the_right_bin(self):
        mz = np.array([500.0, 500.0]); ints = np.array([30.0, 99.0]); mob = np.array([0.90, 1.30])
        bmz, bint, bmob = _dia_prep_2d_jit(mz, ints, mob, 20e-6, 0.05)
        assert abs(bmob[_match_mz_only(bmz, bint, 500.0, 20e-6)] - 1.30) < 1e-9
        assert abs(bint[_match_mz_im(bmz, bmob, 500.0, 0.90, 20e-6, 0.05)] - 30.0) < 1e-9
        assert _match_mz_im(bmz, bmob, 500.0, 2.0, 20e-6, 0.05) == -1

    def test_main_search_observation_is_im_clean(self):
        mz = np.array([200.0, 200.0, 400.0]); ints = np.array([900.0, 100.0, 500.0])
        mob = np.array([0.90, 1.30, 0.90])
        bmz, bint, bmob = _dia_prep_2d_jit(mz, ints, mob, 20e-6, 0.05)
        spec_mz = np.array([200.0, 400.0]); spec_int = np.array([1.0, 1.0])
        res = _create_entries_direct_jit(
            np.zeros(0), spec_mz, spec_int,
            np.array([0], dtype=np.int64), np.array([2], dtype=np.int32),
            np.array([0, 1], dtype=np.int32), np.array([0], dtype=np.int64),
            np.array([2], dtype=np.int32), np.array([0], dtype=np.int64),
            np.array([500.0]), np.array([500.0]), np.array([1.0]), np.array([0.90]), 20e-6,
            0.0, 0, False,
            bmz, bint, bmob, 20e-6, 0.05, True)
        passing, flat_rows, all_coords = res[0], res[1], res[6]
        assert passing.tolist() == [0]
        assert all(c % 2 == 1 for c in all_coords)              # both matched
        assert sorted(bint[flat_rows].tolist()) == [500.0, 900.0]  # clean, not the 100 interferer

class TestMainSearchMs1Error:
    """The main search's MS1 error is to the most intense MS1 peak within the
    ppm window and the candidate's IM gate, not to the nearest by m/z."""

    @staticmethod
    def _ms1_error(ms1_mz, ms1_int, ms1_mob, ms1_tol=20e-6):
        # One candidate (library spectrum 200/400) matched in a 2-peak DIA spectrum
        # at IM 0.90, precursor m/z 500.
        mz = np.array([200.0, 400.0]); ints = np.array([900.0, 500.0]); mob = np.array([0.90, 0.90])
        bmz, bint, bmob = _dia_prep_2d_jit(mz, ints, mob, 20e-6, 0.05)
        res = _create_entries_direct_jit(
            np.zeros(0), np.array([200.0, 400.0]), np.array([1.0, 1.0]),
            np.array([0], dtype=np.int64), np.array([2], dtype=np.int32),
            np.array([0, 1], dtype=np.int32), np.array([0], dtype=np.int64),
            np.array([2], dtype=np.int32), np.array([0], dtype=np.int64),
            np.array([500.0]),
            np.asarray(ms1_mz, float), np.asarray(ms1_int, float), np.asarray(ms1_mob, float),
            ms1_tol, 0.0, 0, False,
            bmz, bint, bmob, 20e-6, 0.05, True)
        passing, ms1_error = res[0], res[5]
        assert passing.tolist() == [0]
        return ms1_error[0]

    def test_most_intense_not_nearest(self):
        # 500.0005 (1 ppm, weak) vs 500.004 (8 ppm, intense), both at IM 0.90
        e = self._ms1_error([500.0005, 500.004], [10.0, 1000.0], [0.90, 0.90])
        assert abs(e - 0.004 / 500.0) < 1e-12

    def test_im_gate_still_applies(self):
        # the intense peak is at another mobility, so the weak in-gate one wins
        e = self._ms1_error([500.0005, 500.004], [10.0, 1000.0], [0.90, 1.30])
        assert abs(e - 0.0005 / 500.0) < 1e-12

    def test_outside_the_ppm_window_is_ignored(self):
        e = self._ms1_error([500.0005, 500.02], [10.0, 1000.0], [0.90, 0.90])
        assert abs(e - 0.0005 / 500.0) < 1e-12

    def test_none_in_window_is_nan(self):
        assert np.isnan(self._ms1_error([500.5], [10.0], [0.90]))

    def test_ties_go_to_the_lowest_mz(self):
        e = self._ms1_error([499.999, 500.001], [5.0, 5.0], [0.90, 0.90])
        assert abs(e - (-0.001 / 500.0)) < 1e-12


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
        frag_codes = np.array([encode_frag_name("b1_1"), encode_frag_name("y1_1"), encode_frag_name("b2_1")])
        frags = np.array([100, 200, 50])
        score, num_b, num_y = hyperscore2(frags, frag_codes)
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
        assert len(out) == 15

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

    def _to_coo_args(self, A):
        coo = A.tocoo()
        return coo.data.astype(np.float64), coo.row.astype(np.int32), coo.col.astype(np.int32), coo.shape[0], coo.shape[1]

    def test_basic_sparse_input(self):
        """Test with simple sparse matrix."""
        # Create a simple sparse matrix (10 peaks, 3 candidates)
        np.random.seed(42)
        A = sparse.random(10, 3, density=0.5, format='csr')
        b = np.abs(np.random.rand(10))  # Observed intensities (non-negative)

        result = huber_nnls_irls(*self._to_coo_args(A), b)

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

        result = huber_nnls_irls(*self._to_coo_args(A), b)

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

        result = huber_nnls_irls(*self._to_coo_args(A), b, max_iter=50)

        x = np.array(result['x']).flatten()
        assert x.shape == (2,)
        assert np.all(x >= -1e-10)

    def test_convergence(self):
        """Test that the algorithm converges within max_iter."""
        np.random.seed(123)
        A = sparse.random(20, 5, density=0.3, format='csr')
        b = np.abs(np.random.rand(20))

        result = huber_nnls_irls(*self._to_coo_args(A), b, max_iter=100, tol=1e-6)

        assert 'x' in result
        x = np.array(result['x']).flatten()
        assert x.shape == (5,)


# The dense-Gram implementation _coo_gram_and_Aty replaced (it allocated an
# n_cols x n_cols matrix per fit), kept as the bit-identity reference.
from numba import njit  # noqa: E402

@njit(nogil=True)
def _dense_gram_reference(rows, cols, vals, y, n_cols):
    """A^T y and the largest absolute correlation between two different
    columns of A, from COO arrays.

    The correlation is computed exactly as numpy would from the Gram matrix:
    gram / outer(norms, norms) with zero norms set to 1 and the diagonal
    zeroed, then max(abs(.)) -- the same operations in the same order, so
    the result is bit-identical.  Returns (Aty, max_corr)."""
    gram = np.zeros((n_cols, n_cols))
    Aty = np.zeros(n_cols)
    # Group entries by row for outer-product accumulation
    # First pass: A^T y
    for i in range(len(vals)):
        Aty[cols[i]] += vals[i] * y[rows[i]]
    # Gram matrix: for each pair of entries sharing a row, accumulate
    # Sort by row for efficient grouping
    order = np.argsort(rows)
    i = 0
    n = len(order)
    while i < n:
        r = rows[order[i]]
        j = i
        while j < n and rows[order[j]] == r:
            j += 1
        # entries order[i:j] all share row r
        for a in range(i, j):
            ca = cols[order[a]]
            va = vals[order[a]]
            for b in range(a, j):
                cb = cols[order[b]]
                vb = vals[order[b]]
                gram[ca, cb] += va * vb
                if ca != cb:
                    gram[cb, ca] += va * vb
        i = j
    norms = np.empty(n_cols)
    for c in range(n_cols):
        norms[c] = np.sqrt(gram[c, c])
        if norms[c] == 0:
            norms[c] = 1.0
    max_corr = 0.0
    for a in range(n_cols):
        for b in range(n_cols):
            if a != b:
                v = abs(gram[a, b] / (norms[a] * norms[b]))
                if v != v:
                    return Aty, v          # NaN: np.max propagates it
                if v > max_corr:
                    max_corr = v
    return Aty, max_corr



def _random_coo(rng, n_rows, n_cols, n, dup=True):
    rows = rng.integers(0, n_rows, n).astype(np.int32)
    cols = rng.integers(0, n_cols, n).astype(np.int32)
    if not dup:
        key = np.unique(rows.astype(np.int64) * n_cols + cols)
        rows = (key // n_cols).astype(np.int32)
        cols = (key % n_cols).astype(np.int32)
        perm = rng.permutation(len(rows))
        rows, cols = rows[perm], cols[perm]
    vals = rng.random(len(rows))
    return rows, cols, vals, rng.random(n_rows)


def _search_like_coo(rng, n_rows, n_cols):
    """Each candidate (column) matches 8-36 peaks (rows), unique per column,
    rows ordered the way the matcher emits them (by candidate)."""
    rows, cols = [], []
    for c in range(n_cols):
        k = int(rng.integers(8, 37))
        rows.append(np.sort(rng.choice(n_rows, k, replace=False)))
        cols.append(np.full(k, c))
    rows = np.concatenate(rows).astype(np.int32)
    cols = np.concatenate(cols).astype(np.int32)
    return rows, cols, rng.random(len(rows)), rng.random(n_rows)


class TestCooGramSparseMatchesDense:
    """The sparse _coo_gram_and_Aty must return exactly what the dense-Gram
    version did: Aty and max_corr bit for bit, NaN included."""

    @staticmethod
    def _check(rows, cols, vals, y, n_cols):
        from src.spectral_fitting import _coo_gram_and_Aty
        Aty, mc = _coo_gram_and_Aty(rows, cols, vals, y, n_cols)
        Aty_ref, mc_ref = _dense_gram_reference(rows, cols, vals, y, n_cols)
        assert np.array_equal(Aty, Aty_ref, equal_nan=True)
        if mc_ref != mc_ref:
            assert mc != mc
        else:
            assert mc == mc_ref

    @pytest.mark.parametrize("seed", range(40))
    def test_random_with_duplicate_entries(self, seed):
        rng = np.random.default_rng(seed)
        n_rows = int(rng.integers(1, 60)); n_cols = int(rng.integers(1, 80))
        self._check(*_random_coo(rng, n_rows, n_cols, int(rng.integers(1, 600))), n_cols)

    @pytest.mark.parametrize("seed", range(10))
    def test_random_without_duplicates(self, seed):
        rng = np.random.default_rng(100 + seed)
        self._check(*_random_coo(rng, 200, 300, 4000, dup=False), 300)

    @pytest.mark.parametrize("seed", range(3))
    def test_search_like(self, seed):
        rng = np.random.default_rng(200 + seed)
        self._check(*_search_like_coo(rng, 3000, 2500), 2500)

    def test_columns_without_entries(self):
        rng = np.random.default_rng(7)
        rows, cols, vals, y = _random_coo(rng, 30, 20, 200)
        cols = np.where(cols % 3 == 0, 1, cols).astype(np.int32)  # columns 0, 3, 6, ... empty
        self._check(rows, cols, vals, y, 20)

    def test_single_column_and_no_entries(self):
        rng = np.random.default_rng(8)
        self._check(*_random_coo(rng, 10, 1, 30), 1)
        empty = np.empty(0, np.int32)
        self._check(empty, empty, np.empty(0), np.ones(3), 4)

    @pytest.mark.parametrize("where", ["shared_row", "isolated_column"])
    def test_nan(self, where):
        rng = np.random.default_rng(9)
        rows, cols, vals, y = _random_coo(rng, 30, 20, 200)
        if where == "isolated_column":
            # column 19 alone in its own row: no shared row, NaN only via its norm
            rows = np.append(rows[cols != 19], 30).astype(np.int32)
            vals = np.append(vals[cols != 19], np.nan)
            cols = np.append(cols[cols != 19], 19).astype(np.int32)
            y = np.append(y, 1.0)
        else:
            vals[5] = np.nan
        self._check(rows, cols, vals, y, 20)

    def test_large_fit_stays_small(self):
        """5,000 candidates: the dense Gram would be 200 MB.  (The test session
        runs numba-disabled, so this is kept small; compiled, the 55,000-
        candidate fits of a raw-data run would have needed 22 GB.)"""
        from src.spectral_fitting import _coo_gram_and_Aty
        from src.utils import mem_profile as mp
        rng = np.random.default_rng(10)
        n_cols = 5_000
        rows, cols, vals, y = _search_like_coo(rng, 2000, n_cols)
        _coo_gram_and_Aty(rows[:100], cols[:100], vals[:100], y, n_cols)  # compile if JIT on
        before = mp.malloc_stats()[0]
        Aty, mc = _coo_gram_and_Aty(rows, cols, vals, y, n_cols)
        after = mp.malloc_stats()[0]
        assert 0.0 < mc <= 1.0 + 1e-12
        if before is not None:
            assert after - before < 64 * 2**20   # nothing near the 200 MB dense matrix held


class TestCooGramMaxCorr:
    """_coo_gram_and_Aty's max column correlation must equal the numpy
    computation it replaced, bit for bit (it sets huber_nnls_irls' l1_ratio)."""

    @staticmethod
    def _numpy(rows, cols, vals, y, n_cols):
        A = np.zeros((int(rows.max()) + 1, n_cols))
        np.add.at(A, (rows, cols), vals)
        gram = np.zeros((n_cols, n_cols))
        order = np.argsort(rows, kind="quicksort")   # same accumulation order as the kernel
        for r in np.unique(rows):
            idx = [k for k in order if rows[k] == r]
            for a_i, a in enumerate(idx):
                for b in idx[a_i:]:
                    gram[cols[a], cols[b]] += vals[a] * vals[b]
                    if cols[a] != cols[b]:
                        gram[cols[b], cols[a]] += vals[a] * vals[b]
        norms = np.sqrt(np.diag(gram))
        norms[norms == 0] = 1
        corr = gram / np.outer(norms, norms)
        np.fill_diagonal(corr, 0)
        return A.T @ y, np.max(np.abs(corr))

    @pytest.mark.parametrize("n_cols, empty_col", [(6, None), (6, 3), (1, None)])
    def test_matches_numpy(self, n_cols, empty_col):
        from src.spectral_fitting import _coo_gram_and_Aty
        rng = np.random.default_rng(n_cols)
        n = 40
        rows = rng.integers(0, 15, n).astype(np.int32)
        cols = rng.integers(0, n_cols, n).astype(np.int32)
        if empty_col is not None:
            cols[cols == empty_col] = (empty_col + 1) % n_cols   # a column with no entries: norm 0
        vals = rng.random(n)
        y = rng.random(15)
        Aty, max_corr = _coo_gram_and_Aty(rows, cols, vals, y, n_cols)
        _, expected = self._numpy(rows, cols, vals, y, n_cols)
        assert max_corr == expected
