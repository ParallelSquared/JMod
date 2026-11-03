
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


import warnings

from sklearn.linear_model import ElasticNet, Lasso
from sklearn.linear_model._coordinate_descent import enet_path
from scipy.sparse import csc_matrix
from sklearn.exceptions import ConvergenceWarning

# TODO: Investigate why sklearn reports tolerance=0 in ConvergenceWarning
#       (duality gap: ~1e-34, tolerance: 0.000e+00). Both Lasso and ElasticNet
#       have explicit tol set, so this may be a numerical precision issue where
#       the internal scaled tolerance rounds to zero for certain inputs.
warnings.filterwarnings(
    "ignore",
    message="Objective did not converge",
    category=ConvergenceWarning,
    module=r"sklearn\.linear_model\._coordinate_descent",
)

from scipy import sparse
import re
from src.utils.io.read_output import names
import src.config as config
import math

from src.utils.misc_functions import createTolWindows, window_width, feature_list_mz, feature_list_rt, \
hyperscore_b_y, longest_y, closest_ms1spec, closest_peak_diff, cosim,np_pearson_cor
from src.utils.parse_peptides import change_seq, convert_frags
from src.models.spec_lib.spec_lib import frag_to_peak
from src.utils.frag_encoding import is_b_ion, is_y_ion, is_isotope, get_index, decode_frag_names
from numba import njit
import time
import threading
from typing import NamedTuple, Optional

import pyarrow as pa


# ── Thread-local RandomState cache for enet_path ──
# Constructing np.random.RandomState(42) is expensive (~2K GIL samples in profiling).
# Each thread keeps one, reseeded to 42 per spectrum for determinism + thread safety.
_enet_tls = threading.local()

def _get_enet_rng():
    try:
        rng = _enet_tls.rng
    except AttributeError:
        rng = np.random.RandomState(42)
        _enet_tls.rng = rng
    rng.seed(42)
    return rng




@njit(nogil=True)
def _build_y_pred_jit(all_rows, all_cols, all_vals, offsets, coeffs, coeff_offset, y_pred):
    """Build predicted spectrum by accumulating library intensity * coefficient contributions.

    Replaces get_residuals._compute_prediction. For each candidate j, for each
    of its matched fragments k: y_pred[row] += lib_intensity * coeff[col + offset].
    This builds the full predicted spectrum (Ax) across all candidates.

    Args:
        all_rows:      int32[]  — flattened row indices (which DIA peak each fragment maps to)
        all_cols:      int32[]  — flattened col indices (which candidate each fragment belongs to)
        all_vals:      float64[] — flattened normalized library intensities
        offsets:       int32[n+1] — candidate j's fragments are at all_rows[offsets[j]:offsets[j+1]]
        coeffs:        float64[] — fit coefficients from NNLS
        coeff_offset:  int       — offset into coeffs for this candidate set (ref vs decoy)
        y_pred:        float64[] — output array, accumulated in-place (must be pre-zeroed)
    """
    n = len(offsets) - 1
    for j in range(n):
        start = offsets[j]
        end = offsets[j + 1]
        for k in range(start, end):
            # y_pred[peak_row] += normalized_lib_intensity * fit_coefficient
            y_pred[all_rows[k]] += all_vals[k] * coeffs[all_cols[k] + coeff_offset]


@njit(nogil=True)
def _compute_candidate_features_jit(
    all_rows, all_vals, all_cols, offsets,
    val_obs, residuals, y_pred, coeffs, coeff_offset,
    scribe_scores, gof_stats, max_unmatched_res, max_matched_res,
    manhattan_dist, spectral_contrast,
    num_peaks_matched_out, frac_lib_intensity_out, frac_dia_intensity_out,
    tic
):
    """Fused computation of per-candidate spectral features in two passes.

    Replaces four separate Python-loop functions (get_scribe, gof_stat,
    get_manhattan_distance) with a single JIT'd function that iterates over
    each candidate's matched fragments only twice.

    Pass 1: Accumulate sqrt sums needed for Scribe normalization denominators.
    Pass 2: Compute all features in a single iteration over fragments.

    All output arrays are pre-allocated by the caller and written in-place.

    Args:
        all_rows:      int32[]   — flattened row indices (DIA peak index per fragment)
        all_vals:      float64[] — flattened normalized library intensities
        all_cols:      int32[]   — flattened column indices (candidate index per fragment)
        offsets:       int32[n+1] — candidate j owns fragments at [offsets[j], offsets[j+1])
        val_obs:       float64[] — observed DIA spectrum intensities
        residuals:     float64[] — val_obs - y_pred (precomputed)
        y_pred:        float64[] — predicted spectrum from _build_y_pred_jit
        coeffs:        float64[] — NNLS fit coefficients
        coeff_offset:  int       — offset into coeffs for this candidate set
        tic:           float     — total ion current (sum of val_obs)

    Output arrays (all float64[n], written in-place):
        scribe_scores:           Scribe score (Searle et al. 2023) — sum of squared differences
                                 between sqrt-normalized predicted vs observed fragment distributions.
                                 Lower = better match.
        gof_stats:               log2(sum_abs_residuals / sum_fitted_intensity) — goodness of fit.
                                 Lower = better fit.
        max_unmatched_res:       log2(max_residual_at_unobserved_peaks / sum_fitted + eps) —
                                 worst residual at peaks where val_obs ≈ 0 (false signal).
        max_matched_res:         log2(max_residual_at_observed_peaks / sum_fitted + eps) —
                                 worst residual at peaks where val_obs > 0.
        manhattan_dist:          -log2(sum_abs(y_pred - val_obs) / sum_val_obs) — modified
                                 Manhattan distance. Higher = better fit.
        spectral_contrast:       sqrt(uv) / (sqrt(u²) * sqrt(v²)) — cosine similarity between
                                 y_pred and val_obs at matched fragment positions.
        num_peaks_matched_out:   Number of matched fragments (same as len(row_idx_split[j])).
        frac_lib_intensity_out:  Sum of normalized library intensities (np.sum(val_split[j])).
        frac_dia_intensity_out:  Sum of observed intensities at matched peaks / TIC.
    """
    n = len(offsets) - 1
    _F32_MAX = 3.4028235e+38
    _F32_MIN = -3.4028235e+38

    for j in range(n):
        start = offsets[j]
        end = offsets[j + 1]

        # ── Pass 1: Scribe normalization denominators ──
        # h_sqrt_sum = sum(sqrt(lib_intensity)) for this candidate
        # x_sqrt_sum = sum(sqrt(obs_intensity)) at matched peak positions
        # These are the denominators in the Scribe score formula:
        #   score = sum_k( (sqrt(h_k)/h_sqrt_sum - sqrt(x_k)/x_sqrt_sum)^2 )
        h_sqrt_sum = 0.0
        x_sqrt_sum = 0.0
        for k in range(start, end):
            h_sqrt_sum += math.sqrt(all_vals[k])
            x_sqrt_sum += math.sqrt(val_obs[all_rows[k]])

        # ── Pass 2: All per-candidate features in one loop ──
        # Scribe accumulators
        scribe = 0.0
        # GoF accumulators (from gof_stat)
        sum_residuals = 0.0   # sum of |residual| at matched positions
        sum_fitted = 0.0      # sum of |coeff * lib_intensity| at matched positions
        max_unmatched = 0.0   # max |residual| where obs ≈ 0 (unmatched library peak)
        max_matched = 0.0     # max |residual| where obs > 0 (matched peak)
        # Manhattan + spectral contrast accumulators (from get_manhattan_distance)
        x_sum = 0.0           # sum of observed intensities at matched positions
        manhattan = 0.0       # sum of |y_pred - val_obs| at matched positions
        u2_sum = 0.0          # sum of y_pred^2 (for spectral contrast)
        v2_sum = 0.0          # sum of val_obs^2 (for spectral contrast)
        uv_sum = 0.0          # sum of y_pred * val_obs (for spectral contrast)
        # Simple feature accumulators
        val_sum = 0.0         # sum of lib intensities (= frac_lib_intensity)
        dia_sum = 0.0         # sum of obs intensities at matched peaks (for frac_dia_intensity)
        n_matched = 0         # count of matched fragments

        for k in range(start, end):
            row = all_rows[k]     # DIA peak index
            val = all_vals[k]     # normalized library intensity
            col = all_cols[k]     # candidate column index
            obs = val_obs[row]    # observed DIA intensity at this peak
            res = residuals[row]  # obs - predicted (precomputed)
            pred = y_pred[row]    # full predicted intensity at this peak (all candidates)

            # ── Scribe score (Searle et al. 2023, PMID: 36695531) ──
            # Measures divergence between sqrt-normalized library and observed
            # fragment intensity distributions. Lower = more similar spectra.
            if h_sqrt_sum > 0 and x_sqrt_sum > 0:
                h_norm = math.sqrt(val) / h_sqrt_sum   # normalized predicted
                x_norm = math.sqrt(obs) / x_sqrt_sum   # normalized observed
                scribe += (h_norm - x_norm) ** 2

            # ── Goodness-of-fit (from gof_stat) ──
            # Accumulates total absolute residuals and total fitted intensity.
            # Also tracks the single worst residual separately for observed peaks
            # (max_matched) vs unobserved peaks (max_unmatched, where obs ≈ 0
            # means the library predicted a peak the DIA spectrum doesn't have).
            abs_res = abs(res)
            sum_residuals += abs_res
            sum_fitted += abs(coeffs[col + coeff_offset] * val)
            if obs > 1e-6:
                # Observed peak — track worst match
                if abs_res > max_matched:
                    max_matched = abs_res
            elif obs < 1e-6:
                # Unobserved peak — library predicted signal that isn't there
                if abs_res > max_unmatched:
                    max_unmatched = abs_res

            # ── Manhattan distance + spectral contrast (from get_manhattan_distance) ──
            # Manhattan: sum of absolute differences between predicted and observed.
            # Spectral contrast: cosine similarity between y_pred and val_obs vectors
            # at matched positions: cos(θ) = dot(u,v) / (|u| * |v|).
            x_sum += obs
            manhattan += abs(pred - obs)
            u2_sum += pred * pred
            v2_sum += obs * obs
            uv_sum += pred * obs

            # ── Simple features ──
            val_sum += val    # replaces: np.sum(ref_spec_values_split[j])
            dia_sum += obs    # replaces: np.sum(dia_spectrum[row_idx_split[j], 1])
            n_matched += 1    # replaces: np.sum(lib_peaks_matched[j])

        # ── Finalize Scribe ──
        scribe_scores[j] = scribe

        # ── Finalize GoF ──
        # Guard against zero denominators, then log-transform
        if sum_fitted == 0:
            sum_fitted = 1e-6
        if sum_residuals == 0:
            sum_residuals = 1e-6
        gof_stats[j] = math.log2(sum_residuals / sum_fitted)
        max_matched_res[j] = math.log2(max_matched / (sum_fitted + 1e-10) + 1e-10)
        max_unmatched_res[j] = math.log2(max_unmatched / (sum_fitted + 1e-10) + 1e-10)

        # ── Finalize Manhattan + spectral contrast ──
        if x_sum > 0 and manhattan > 0:
            # Normal case: -log2 so higher = better fit
            manhattan_dist[j] = -math.log2(manhattan / x_sum)
            spectral_contrast[j] = math.sqrt(uv_sum) / (math.sqrt(u2_sum) * math.sqrt(v2_sum) + 1e-10)
        elif x_sum == 0:
            # No observed intensity at any matched position — bad fit
            manhattan_dist[j] = _F32_MAX
            spectral_contrast[j] = 0.0
        else:
            # manhattan == 0 means perfect prediction — rare edge case
            manhattan_dist[j] = _F32_MIN
            spectral_contrast[j] = math.sqrt(uv_sum) / (math.sqrt(u2_sum) * math.sqrt(v2_sum) + 1e-10)

        # ── Finalize simple features ──
        num_peaks_matched_out[j] = n_matched
        frac_lib_intensity_out[j] = val_sum
        if tic > 0:
            frac_dia_intensity_out[j] = dia_sum / tic
        else:
            frac_dia_intensity_out[j] = 0.0


# TODO: Investigate inlining hyperscore computation into _compute_candidate_features_jit
#       to avoid a separate pass over fragment data. Would need to pass fragment codes
#       and intensities alongside the sparse matrix arrays.
@njit(nogil=True)
def _compute_hyperscores_jit(all_intensities, all_codes, offsets,
                              hyperscores, b_counts_out, y_counts_out,
                              longest_y_out):
    """Batch-compute hyperscores and ion counts for all candidates in one JIT'd loop.

    Replaces per-candidate calls to hyperscore2 + get_index. For each candidate,
    decodes the packed int32 frag codes using bitwise operations to classify ions
    as b/y and filter out isotopes, then computes:
      hyperscore = max(0, ln(dot_product * b_factorial * y_factorial))

    Bit layout of frag codes (from frag_encoding.py):
      [2:0]   ion_type  — 0=b, 1=y, 2=a, 3=c, 4=x, 5=z
      [10:3]  index     — fragment ordinal (1-255)
      [20:17] iso       — isotope index (0 = monoisotopic, >0 = isotope)

    Args:
        all_intensities: float64[] — flattened fragment intensities for all candidates
        all_codes:       int32[]   — flattened packed frag codes for all candidates
        offsets:         int32[n+1] — candidate j's fragments at [offsets[j], offsets[j+1])
        hyperscores:     float64[n] — output: hyperscore per candidate
        b_counts_out:    float64[n] — output: number of non-isotope b-ions per candidate
        y_counts_out:    float64[n] — output: number of non-isotope y-ions per candidate
        longest_y_out:   float64[n] — output: max fragment index among y-ions per candidate
    """
    # Bit masks and shifts (must match frag_encoding.py)
    ION_MASK = 0x7
    ION_SHIFT = 0
    IDX_MASK = 0xFF
    IDX_SHIFT = 3
    ISO_MASK = 0xF
    ISO_SHIFT = 17
    ION_B = 0
    ION_Y = 1

    n = len(offsets) - 1
    for j in range(n):
        start = offsets[j]
        end = offsets[j + 1]

        num_b = 0
        num_y = 0
        dp = 0.0          # sum of non-isotope fragment intensities
        max_idx = 0        # max fragment index (for longest_y)

        for k in range(start, end):
            code = all_codes[k]
            ion_type = (code >> ION_SHIFT) & ION_MASK
            iso = (code >> ISO_SHIFT) & ISO_MASK
            idx = (code >> IDX_SHIFT) & IDX_MASK

            # Track max fragment index across all ion types
            if idx > max_idx:
                max_idx = idx

            # Only count non-isotope fragments for hyperscore
            if iso == 0:
                dp += all_intensities[k]
                if ion_type == ION_B:
                    num_b += 1
                elif ion_type == ION_Y:
                    num_y += 1

        # hyperscore = max(0, ln(dp * b! * y!))
        # Use log-space to avoid overflow: ln(dp) + ln(b!) + ln(y!)
        if dp > 0 and (num_b > 0 or num_y > 0):
            log_score = math.log(dp) + math.lgamma(num_b + 1) + math.lgamma(num_y + 1)
            hyperscores[j] = max(0.0, log_score)
        else:
            hyperscores[j] = 0.0

        b_counts_out[j] = num_b
        y_counts_out[j] = num_y
        longest_y_out[j] = max_idx


@njit(nogil=True)
def _single_match_lookup_jit(coo_rows, coo_cols, n_rows):
    """Compute boolean lookup of rows matched by exactly one candidate.

    A row (DIA peak) matched by exactly one column (precursor candidate) means
    that peak's intensity is unambiguously explained by a single precursor.
    These uniquely-matched peaks are used to compute frac_unique_pred.

    Replaces: sparse.coo_matrix(...) → np.sum(matrix>0, axis=1)==1 → np.where

    Args:
        coo_rows: int32/int64 array of row indices (COO format)
        coo_cols: int32/int64 array of column indices (COO format)
        n_rows: total number of rows

    Returns:
        bool array of length n_rows — True where row is matched by exactly one precursor.
    """
    # Count distinct columns per row using a seen-column tracker
    # Since values can be duplicated in COO (same row, same col), we need
    # to count distinct columns. Use a simple approach: for each row, track
    # how many distinct columns we've seen.
    col_count = np.zeros(n_rows, dtype=np.int32)
    # Track last seen column per row to avoid counting duplicates from sorted COO
    # But COO isn't necessarily sorted, so use a different approach:
    # First pass: for each (row, col) pair, mark the row as having that col
    # Since we just need count >= 2 vs == 1, we can increment and cap at 2
    last_col = np.full(n_rows, -1, dtype=np.int64)
    for i in range(len(coo_rows)):
        r = coo_rows[i]
        c = coo_cols[i]
        if col_count[r] == 0:
            col_count[r] = 1
            last_col[r] = c
        elif last_col[r] != c:
            col_count[r] = 2  # already >= 2, no need to count further

    result = np.empty(n_rows, dtype=np.bool_)
    for i in range(n_rows):
        result[i] = col_count[i] == 1
    return result


@njit(nogil=True)
def _dia_prep_jit(mz, intens, mobility, mz_tol):
    """Merge nearby DIA peaks and compute centroid breaks + bin centers.

    NOTE: This merging step may be redundant if the input spectra are already
    centroided. The original code merged peaks within mz_tol of each other,
    which could collapse peaks that are close in m/z. If inputs are already
    properly centroided, this step could be skipped.

    ``mobility`` is the per-peak ion mobility (1/K0), parallel to ``mz``; each
    merged bin gets the intensity-weighted mean mobility of the raw peaks in it
    (``merged_mob``). Pass zeros for non-IM data — the caller ignores the output.

    Returns:
        merged_mz: merged peak m/z values
        merged_int: merged peak intensities
        centroid_breaks: sorted lower/upper tolerance bounds (2*n_merged)
        bin_centers: midpoints of each (lower, upper) break pair (n_merged)
        merged_mob: intensity-weighted mean mobility per merged bin (n_merged)
    """
    n = len(mz)

    # ── Step 1: Merge nearby peaks within mz_tol ──
    # Equivalent to: searchsorted(mz + mz_tol*mz, mz) to find merge groups
    upper_bounds = np.empty(n, dtype=np.float64)
    for i in range(n):
        upper_bounds[i] = mz[i] + mz_tol * mz[i]

    # Assign each peak to a merge group via searchsorted into upper_bounds
    merge_idx = np.empty(n, dtype=np.int64)
    for i in range(n):
        # Binary search for leftmost position where upper_bounds[pos] >= mz[i]
        lo = 0
        hi = n
        while lo < hi:
            mid = (lo + hi) >> 1
            if upper_bounds[mid] < mz[i]:
                lo = mid + 1
            else:
                hi = mid
        merge_idx[i] = lo

    # Count unique merge groups
    n_merged = 0
    for i in range(n):
        if i == 0 or merge_idx[i] != merge_idx[i - 1]:
            n_merged += 1

    # Build merged arrays
    merged_mz = np.empty(n_merged, dtype=np.float64)
    merged_int = np.zeros(n_merged, dtype=np.float64)
    merged_mob = np.zeros(n_merged, dtype=np.float64)  # intensity-weighted sum, then /int
    g = -1
    prev_idx = -1
    for i in range(n):
        if merge_idx[i] != prev_idx:
            g += 1
            merged_mz[g] = mz[merge_idx[i]]
            prev_idx = merge_idx[i]
        merged_int[g] += intens[i]
        merged_mob[g] += intens[i] * mobility[i]
    for i in range(n_merged):
        if merged_int[i] > 0.0:
            merged_mob[i] /= merged_int[i]

    # ── Step 2: Compute centroid breaks and bin centers ──
    breaks = np.empty(2 * n_merged, dtype=np.float64)
    for i in range(n_merged):
        breaks[2 * i] = merged_mz[i] - mz_tol * merged_mz[i]
        breaks[2 * i + 1] = merged_mz[i] + mz_tol * merged_mz[i]

    # Sort breaks
    for i in range(1, len(breaks)):
        key = breaks[i]
        j = i - 1
        while j >= 0 and breaks[j] > key:
            breaks[j + 1] = breaks[j]
            j -= 1
        breaks[j + 1] = key

    # Bin centers: mean of each consecutive pair
    bin_centers = np.empty(n_merged, dtype=np.float64)
    for i in range(n_merged):
        bin_centers[i] = (breaks[2 * i] + breaks[2 * i + 1]) * 0.5

    return merged_mz, merged_int, breaks, bin_centers, merged_mob


@njit(nogil=True)
def _dia_prep_2d_jit(mz, intens, mobility, mz_tol, im_tol):
    """Bin DIA peaks by BOTH m/z (within ``mz_tol``) and ion mobility (within
    ``im_tol``), summing intensity within each 2D bin.

    Same-m/z peaks that are IM-separated land in DISTINCT bins, so the summed
    observation for a bin reflects only one IM population (no cross-IM
    contamination), while same-m/z + same-IM peaks share a bin (deconvolved by
    NNLS downstream, exactly like the 1D merge).

    Returns ``(bin_mz, bin_int, bin_mob)``, all sorted ascending by ``bin_mz``.
    ``bin_mz``/``bin_mob`` are intensity-weighted means within the bin.
    """
    n = mz.shape[0]
    if n == 0:
        z = np.empty(0, dtype=np.float64)
        return z, z.copy(), z.copy()

    order = np.argsort(mz)
    smz = mz[order]
    sin = intens[order]
    smo = mobility[order]

    bin_mz = np.empty(n, dtype=np.float64)
    bin_int = np.empty(n, dtype=np.float64)
    bin_mob = np.empty(n, dtype=np.float64)
    nb = 0

    i = 0
    while i < n:
        # m/z group: peaks within mz_tol of the group anchor
        anchor_mz = smz[i]
        mz_lim = anchor_mz + mz_tol * anchor_mz
        jmz = i
        while jmz < n and smz[jmz] <= mz_lim:
            jmz += 1

        # within the group, sub-group by mobility within im_tol
        g = jmz - i
        gmz = smz[i:jmz].copy()
        gin = sin[i:jmz].copy()
        gmo = smo[i:jmz].copy()
        mo_order = np.argsort(gmo)
        gmz = gmz[mo_order]
        gin = gin[mo_order]
        gmo = gmo[mo_order]

        p = 0
        while p < g:
            anchor_mob = gmo[p]
            mob_lim = anchor_mob + im_tol
            wsum = 0.0
            mzw = 0.0
            mow = 0.0
            q = p
            while q < g and gmo[q] <= mob_lim:
                w = gin[q]
                wsum += w
                mzw += w * gmz[q]
                mow += w * gmo[q]
                q += 1
            bin_int[nb] = wsum
            if wsum > 0.0:
                bin_mz[nb] = mzw / wsum
                bin_mob[nb] = mow / wsum
            else:
                bin_mz[nb] = gmz[p]
                bin_mob[nb] = gmo[p]
            nb += 1
            p = q
        i = jmz

    bmz = bin_mz[:nb]
    bint = bin_int[:nb]
    bmob = bin_mob[:nb]
    # sort bins ascending by m/z so the matchers can binary-search
    bo = np.argsort(bmz)
    return bmz[bo], bint[bo], bmob[bo]


@njit(nogil=True)
def _match_mz_only(bin_mz, bin_int, q, mz_tol):
    """Index of the MOST INTENSE bin within ``mz_tol`` of ``q``, or -1."""
    n = bin_mz.shape[0]
    if n == 0:
        return -1
    tol = q * mz_tol
    pos = np.searchsorted(bin_mz, q)
    best = -1
    best_int = -1.0
    i = pos - 1
    while i >= 0 and (q - bin_mz[i]) <= tol:
        if bin_int[i] > best_int:
            best_int = bin_int[i]
            best = i
        i -= 1
    i = pos
    while i < n and (bin_mz[i] - q) <= tol:
        if bin_int[i] > best_int:
            best_int = bin_int[i]
            best = i
        i += 1
    return best


@njit(nogil=True)
def _match_mz_im(bin_mz, bin_mob, q, prec_im, mz_tol, im_tol):
    """Index of the bin within ``mz_tol`` of ``q`` AND ``im_tol`` of ``prec_im``
    (nearest mobility), or -1."""
    n = bin_mz.shape[0]
    if n == 0:
        return -1
    tol = q * mz_tol
    pos = np.searchsorted(bin_mz, q)
    best = -1
    best_d = 1e18
    i = pos - 1
    while i >= 0 and (q - bin_mz[i]) <= tol:
        d = abs(bin_mob[i] - prec_im)
        if d <= im_tol and d < best_d:
            best_d = d
            best = i
        i -= 1
    i = pos
    while i < n and (bin_mz[i] - q) <= tol:
        d = abs(bin_mob[i] - prec_im)
        if d <= im_tol and d < best_d:
            best_d = d
            best = i
        i += 1
    return best


@njit(nogil=True)
def _compute_unique_frac_jit(ref_rows, ref_vals, ref_offsets,
                              unique_lookup, dia_obs, coeffs, coeff_offset,
                              frac_unique_pred_out):
    """Compute frac_unique_pred for each candidate using flat arrays.

    Replaces peaks_not_shared loop + frac_unique_pred list comprehension.
    For each candidate, sums observed and library intensities at DIA peaks
    that are uniquely matched (single candidate), then computes
    frac = (sum_lib / sum_obs) * coefficient.
    """
    n = len(ref_offsets) - 1
    for j in range(n):
        sum_obs = 0.0
        sum_lib = 0.0
        for k in range(ref_offsets[j], ref_offsets[j + 1]):
            row = ref_rows[k]
            if row < len(unique_lookup) and unique_lookup[row]:
                sum_obs += dia_obs[row]
                sum_lib += ref_vals[k]
        if sum_obs > 0.0:
            frac_unique_pred_out[j] = (sum_lib / sum_obs) * coeffs[coeff_offset + j]


@njit(nogil=True)
def _large_coeff_int_pred_jit(all_vals, all_offsets, coeffs,
                               large_coeff_indices, col_to_pos):
    """Compute sum of (lib_intensity_sum * coefficient) for large-coeff candidates.

    Replaces: sum(np.sum(all_values[i]) * lib_coefficients[i] for i in large_coeff_indices)
    """
    total = 0.0
    for k in range(len(large_coeff_indices)):
        col = large_coeff_indices[k]
        pos = col_to_pos[col]
        val_sum = 0.0
        for j in range(all_offsets[pos], all_offsets[pos + 1]):
            val_sum += all_vals[j]
        total += val_sum * coeffs[col]
    return total


@njit(nogil=True)
def _large_coeff_block_jit(coo_rows, coo_cols, coo_vals, n_rows,
                            coeffs, val_obs,
                            all_flat_rows, all_flat_vals, all_flat_offsets,
                            col_to_pos):
    """Replaces the large_coeff block in get_features with a single nogil pass.

    Computes three outputs for candidates with coefficient > 1:
      1. frac_int_matched_pred_sigcoeff: ratio of predicted intensity from
         large-coeff candidates to their matched observed intensity
      2. subset_cosine: cosine similarity between observed and predicted
         spectra at real DIA peaks matched by large-coeff candidates.
         Computed entirely from flat arrays (original DIA peak space), so
         penalty rows (unmatched fragments) are never included.
      3. predicted_spec: A @ coeffs (full predicted spectrum), also used
         for frac_int_pred / frac_int_matched_pred outside this function

    Args:
        coo_rows/coo_cols/coo_vals: COO sparse matrix (library spectra vs DIA peaks).
            Rows = DIA peak indices (re-indexed via rankdata), cols = candidate indices.
        n_rows: number of rows in sparse matrix (includes penalty rows).
        coeffs: NNLS coefficients per candidate column.
        val_obs: observed intensities at original DIA peak indices. Used for
            large_coeff_int_matched and cosine (original DIA peak space, no
            penalty rows).
        all_flat_rows/all_flat_vals/all_flat_offsets: flattened per-candidate
            matched fragment arrays (ref + decoy). Rows are original DIA peak
            indices, vals are normalized library intensities. These only contain
            real matched peaks — no penalty rows.
        col_to_pos: maps candidate column index → position in the flat arrays.
            -1 if the candidate has no entry.

    Returns:
        (predicted_spec, frac, subset_cosine)

    Previously this was ~10 GIL-holding numpy/scipy calls (~4000 py-spy samples):
      np.where, np.isin, np.unique, np.concatenate, np.delete,
      sparse_lib_matrix.toarray(), np.multiply, np.sum, cosim
    """
    n_cols = len(coeffs)

    # ── Step 1: Identify large-coeff columns (coeff > 1) ──
    n_large = 0
    for c in range(n_cols):
        if coeffs[c] > 1.0:
            n_large += 1

    # ── Step 2: Full predicted spectrum A @ coeffs (reused for frac_int_pred) ──
    predicted_spec = np.zeros(n_rows)
    for i in range(len(coo_vals)):
        predicted_spec[coo_rows[i]] += coo_vals[i] * coeffs[coo_cols[i]]

    if n_large == 0:
        return predicted_spec, 0.0, 0.0

    # ── Step 3: frac + cosine in one pass over flat arrays ──
    # Flat arrays contain only real matched DIA peaks (no penalty rows), so
    # the cosine naturally excludes all penalty rows.
    #
    # We accumulate per-peak predicted intensity from large-coeff candidates
    # into val_obs-sized arrays, then compute cosine over touched peaks.
    n_obs = len(val_obs)
    pred_at_peak = np.zeros(n_obs)   # predicted intensity per DIA peak (large coeffs only)
    peak_touched = np.zeros(n_obs, dtype=np.int8)  # which peaks were matched

    large_coeff_int_pred = 0.0
    for c in range(n_cols):
        if coeffs[c] <= 1.0:
            continue
        pos = col_to_pos[c]
        if pos < 0:
            continue
        # Sum library intensity for this candidate (for frac numerator)
        val_sum = 0.0
        for j in range(all_flat_offsets[pos], all_flat_offsets[pos + 1]):
            val_sum += all_flat_vals[j]
            # Accumulate predicted intensity at each matched DIA peak (for cosine)
            row = all_flat_rows[j]
            pred_at_peak[row] += all_flat_vals[j] * coeffs[c]
            peak_touched[row] = 1
        large_coeff_int_pred += val_sum * coeffs[c]

    # Sum observed intensity at matched peaks (for frac denominator)
    large_coeff_int_matched = 0.0
    for r in range(n_obs):
        if peak_touched[r]:
            large_coeff_int_matched += val_obs[r]

    if large_coeff_int_matched == 0.0:
        large_coeff_int_matched = 1.0  # avoid division by zero

    frac = large_coeff_int_pred / large_coeff_int_matched

    # ── Cosine similarity at matched peaks ──
    # Only real DIA peaks contribute — penalty rows are not in flat arrays
    dot_xy = 0.0
    sum_x2 = 0.0
    sum_y2 = 0.0
    for r in range(n_obs):
        if peak_touched[r]:
            x = val_obs[r]
            y = pred_at_peak[r]
            dot_xy += x * y
            sum_x2 += x * x
            sum_y2 += y * y

    denom = np.sqrt(sum_x2) * np.sqrt(sum_y2)
    subset_cosine = dot_xy / denom if denom > 0.0 else 0.0

    return predicted_spec, frac, subset_cosine


@njit(nogil=True)
def _frag_columns_jit(passing, cand_idx, spec_offsets, spec_mz, spec_int, frag_codes,
                      all_coords, frag_offsets, flat_rows, flat_offsets,
                      bin_centers, dia_int, single_match_lookup, peak_idx_lookup):
    """The matched-fragment output columns of each passing candidate, flat.

    For passing candidate k, its matched library peaks (odd coords) pair in
    order with its DIA rows flat_rows[flat_offsets[k]:flat_offsets[k+1]].
    Per matched peak: fragment code, relative m/z error against the DIA bin
    centre, library m/z and intensity, observed intensity; and, for peaks on
    a DIA row only one candidate matched, the library m/z and observed
    intensity again (the unique columns).  Segments are delimited by offsets
    (matched peaks) and unique_offsets (unique peaks).
    """
    n = passing.shape[0]
    offsets = np.zeros(n + 1, dtype=np.int64)
    unique_offsets = np.zeros(n + 1, dtype=np.int64)
    for k in range(n):
        c = passing[k]
        m = 0
        for p in range(frag_offsets[c], frag_offsets[c + 1]):
            if all_coords[p] % 2 == 1:
                m += 1
        if m != flat_offsets[k + 1] - flat_offsets[k]:
            raise ValueError("matched library peaks and DIA rows differ in number")
        offsets[k + 1] = offsets[k] + m
        u = 0
        for t in range(flat_offsets[k], flat_offsets[k + 1]):
            if single_match_lookup[peak_idx_lookup[flat_rows[t]]]:
                u += 1
        unique_offsets[k + 1] = unique_offsets[k] + u
    total = offsets[n]
    codes = np.empty(total, dtype=np.int32)
    errors = np.empty(total, dtype=np.float64)
    lib_mz = np.empty(total, dtype=np.float64)
    lib_int = np.empty(total, dtype=np.float64)
    obs_int = np.empty(total, dtype=np.float64)
    unique_mz = np.empty(unique_offsets[n], dtype=np.float64)
    unique_obs = np.empty(unique_offsets[n], dtype=np.float64)
    for k in range(n):
        c = passing[k]
        first = spec_offsets[cand_idx[c]] - frag_offsets[c]
        out = offsets[k]
        row = flat_offsets[k]
        uout = unique_offsets[k]
        for p in range(frag_offsets[c], frag_offsets[c + 1]):
            if all_coords[p] % 2 == 1:
                lib = first + p
                r = flat_rows[row]
                centre = bin_centers[r]
                mz = np.float64(spec_mz[lib])
                codes[out] = frag_codes[lib]
                errors[out] = (centre - mz) / centre
                lib_mz[out] = mz
                lib_int[out] = spec_int[lib]
                obs_int[out] = dia_int[r]
                if single_match_lookup[peak_idx_lookup[r]]:
                    unique_mz[uout] = mz
                    unique_obs[uout] = dia_int[r]
                    uout += 1
                out += 1
                row += 1
    return codes, errors, lib_mz, lib_int, obs_int, offsets, unique_mz, unique_obs, unique_offsets


@njit(nogil=True)
def _take_segments_jit(values, offsets, sel):
    """Segments sel[0], sel[1], ... of a flat array, with their new offsets."""
    out_offsets = np.zeros(sel.shape[0] + 1, dtype=np.int64)
    for k in range(sel.shape[0]):
        out_offsets[k + 1] = out_offsets[k] + offsets[sel[k] + 1] - offsets[sel[k]]
    out = np.empty(out_offsets[-1], dtype=values.dtype)
    for k in range(sel.shape[0]):
        start = offsets[sel[k]]
        out[out_offsets[k]:out_offsets[k + 1]] = values[start:start + out_offsets[k + 1] - out_offsets[k]]
    return out, out_offsets


@njit(nogil=True)
def _rankdata_dense_jit(x):
    """0-based dense ranking for non-negative integer array.
    Equivalent to scipy.stats.rankdata(x, method='dense') - 1.
    """
    n = len(x)
    if n == 0:
        return np.empty(0, dtype=np.int32)
    max_val = 0
    for i in range(n):
        v = int(x[i])
        if v > max_val:
            max_val = v
    present = np.zeros(max_val + 1, dtype=np.int8)
    for i in range(n):
        present[int(x[i])] = 1
    rank_map = np.empty(max_val + 1, dtype=np.int32)
    rank = 0
    for i in range(max_val + 1):
        if present[i]:
            rank_map[i] = rank
            rank += 1
        else:
            rank_map[i] = -1
    out = np.empty(n, dtype=np.int32)
    for i in range(n):
        out[i] = rank_map[int(x[i])]
    return out


@njit(nogil=True)
def _assemble_features_jit(
    num_lib_peaks_matched, frac_lib_intensity, frac_dia_intensity,
    rel_error, rt_error,
    frac_int_matched, frac_int_pred,
    r2all, r2_lib_spec, r2_unique,
    frac_unique_pred, frac_dia_intensity_pred,
    hyperscores, b_counts, y_counts, longest_y_ions,
    scribe_scores, max_unmatched_residuals, max_matched_residuals,
    gof_stats, manhattan_distances, fitted_spectral_contrasts,
    frac_int_matched_pred, lc_frac, lc_cosine,
    prec_mz, tic, im_error):
    """Assemble the 28-column feature matrix in one nogil pass.

    Replaces np.ones_like * scalar broadcasts (5 allocations) + np.stack of
    28 arrays (~1500 GIL-holding samples). Fills a pre-allocated (n, 28)
    array directly — no intermediate arrays, no GIL.

    Per-candidate arrays (length n) are copied directly into their column.
    Scalar features (frac_int_matched, frac_int_pred, frac_int_matched_pred,
    lc_frac, lc_cosine, tic) are broadcast by filling the column with the value.
    """
    n = len(num_lib_peaks_matched)
    out = np.empty((n, 28), dtype=np.float64)
    for i in range(n):
        out[i, 0] = num_lib_peaks_matched[i]
        out[i, 1] = frac_lib_intensity[i]
        out[i, 2] = frac_dia_intensity[i]
        out[i, 3] = rel_error[i]
        out[i, 4] = rt_error[i]
        out[i, 5] = frac_int_matched       # scalar broadcast
        out[i, 6] = frac_int_pred           # scalar broadcast
        out[i, 7] = r2all[i]
        out[i, 8] = r2_lib_spec[i]
        out[i, 9] = r2_unique[i]
        out[i, 10] = frac_unique_pred[i]
        out[i, 11] = frac_dia_intensity_pred[i]
        out[i, 12] = hyperscores[i]
        out[i, 13] = b_counts[i]
        out[i, 14] = y_counts[i]
        out[i, 15] = longest_y_ions[i]
        out[i, 16] = scribe_scores[i]
        out[i, 17] = max_unmatched_residuals[i]
        out[i, 18] = max_matched_residuals[i]
        out[i, 19] = gof_stats[i]
        out[i, 20] = manhattan_distances[i]
        out[i, 21] = fitted_spectral_contrasts[i]
        out[i, 22] = frac_int_matched_pred  # scalar broadcast
        out[i, 23] = lc_frac                # scalar broadcast
        out[i, 24] = lc_cosine              # scalar broadcast
        out[i, 25] = prec_mz[i]
        out[i, 26] = tic                    # scalar broadcast
        out[i, 27] = im_error[i]
    return out


@njit(nogil=True)
def _assemble_coo_jit(
    ref_rows, ref_cols, ref_vals,
    dec_rows, dec_cols, dec_vals,
    ref_all_coords, ref_all_norm_int, ref_frag_offsets, ref_passing,
    dec_all_coords, dec_all_norm_int, dec_frag_offsets, dec_passing,
    decoy_col_offset,
    dia_intensities,
    lower_limit):
    """Build ranked COO arrays + dia_spec_int from matched + unmatched data (GIL-free).

    Replaces ~14 GIL-acquiring numpy calls (np.append, np.concatenate, np.unique,
    np.sort, rankdata, unmatched_peaks) with a single nogil JIT call. The caller
    does one scipy.sparse.coo_matrix() call with the returned arrays.

    Unmatched penalty: each individual unmatched fragment (coord even, intensity >
    lower_limit) gets its own zero-intensity penalty row (fit_type "c").

    Returns:
        ranked_rows:     int32[]  — dense-ranked row indices for COO
        out_cols:        int32[]  — column indices for COO
        out_vals:        float64[] — values for COO
        dia_spec_int:    float64[n_unique_rows] — observed intensities (b)
        peak_idx_lookup: int32[max_orig_row+1] — maps original DIA row → ranked row
    """
    n_ref_matched = len(ref_rows)
    n_dec_matched = len(dec_rows)
    n_ref = len(ref_passing)
    n_dec = len(dec_passing)

    # Pass 1: count unmatched fragments
    n_ref_unmatched = 0
    for j in range(n_ref):
        i = int(ref_passing[j])
        s = int(ref_frag_offsets[i])
        e = int(ref_frag_offsets[i + 1])
        for k in range(s, e):
            if ref_all_coords[k] % 2 == 0 and ref_all_norm_int[k] > lower_limit:
                n_ref_unmatched += 1

    n_dec_unmatched = 0
    for j in range(n_dec):
        i = int(dec_passing[j])
        s = int(dec_frag_offsets[i])
        e = int(dec_frag_offsets[i + 1])
        for k in range(s, e):
            if dec_all_coords[k] % 2 == 0 and dec_all_norm_int[k] > lower_limit:
                n_dec_unmatched += 1

    n_total = n_ref_matched + n_dec_matched + n_ref_unmatched + n_dec_unmatched

    # Pass 2: fill combined (row, col, val) arrays
    all_rows = np.empty(n_total, dtype=np.int64)
    out_cols = np.empty(n_total, dtype=np.int32)
    out_vals = np.empty(n_total, dtype=np.float64)

    # Matched ref
    pos = 0
    for i in range(n_ref_matched):
        all_rows[pos] = ref_rows[i]
        out_cols[pos] = np.int32(ref_cols[i])
        out_vals[pos] = ref_vals[i]
        pos += 1

    # Matched decoy
    for i in range(n_dec_matched):
        all_rows[pos] = dec_rows[i]
        out_cols[pos] = np.int32(dec_cols[i] + decoy_col_offset)
        out_vals[pos] = dec_vals[i]
        pos += 1

    # Find max matched row for penalty row assignment
    max_matched_row = np.int64(-1)
    for i in range(n_ref_matched):
        if ref_rows[i] > max_matched_row:
            max_matched_row = ref_rows[i]
    for i in range(n_dec_matched):
        if dec_rows[i] > max_matched_row:
            max_matched_row = dec_rows[i]

    # Penalty rows must be above all DIA peak indices so they get 0 in dia_spec_int
    n_dia = len(dia_intensities)
    next_row = max(max_matched_row + 1, np.int64(n_dia))
    for j in range(n_ref):
        i = int(ref_passing[j])
        s = int(ref_frag_offsets[i])
        e = int(ref_frag_offsets[i + 1])
        for k in range(s, e):
            if ref_all_coords[k] % 2 == 0 and ref_all_norm_int[k] > lower_limit:
                all_rows[pos] = next_row
                out_cols[pos] = np.int32(j)
                out_vals[pos] = ref_all_norm_int[k]
                next_row += 1
                pos += 1

    # Unmatched decoy: each fragment gets its own penalty row
    for j in range(n_dec):
        i = int(dec_passing[j])
        s = int(dec_frag_offsets[i])
        e = int(dec_frag_offsets[i + 1])
        for k in range(s, e):
            if dec_all_coords[k] % 2 == 0 and dec_all_norm_int[k] > lower_limit:
                all_rows[pos] = next_row
                out_cols[pos] = np.int32(j + decoy_col_offset)
                out_vals[pos] = dec_all_norm_int[k]
                next_row += 1
                pos += 1

    # Dense ranking of rows
    max_row_val = np.int64(0)
    for i in range(n_total):
        if all_rows[i] > max_row_val:
            max_row_val = all_rows[i]

    present = np.zeros(max_row_val + 1, dtype=np.int8)
    for i in range(n_total):
        present[all_rows[i]] = 1

    peak_idx_lookup = np.full(max_row_val + 1, -1, dtype=np.int32)
    unique_orig_rows = np.empty(max_row_val + 1, dtype=np.int64)
    rank = np.int32(0)
    for i in range(max_row_val + 1):
        if present[i]:
            peak_idx_lookup[i] = rank
            unique_orig_rows[rank] = i
            rank += 1
    n_unique_rows = int(rank)

    # Apply ranking to row indices
    ranked_rows = np.empty(n_total, dtype=np.int32)
    for i in range(n_total):
        ranked_rows[i] = peak_idx_lookup[all_rows[i]]

    # Build dia_spec_int: observed intensities for matched rows, 0 for penalty rows
    dia_spec_int = np.zeros(n_unique_rows, dtype=np.float64)
    n_dia = len(dia_intensities)
    for i in range(n_unique_rows):
        orig_row = unique_orig_rows[i]
        if orig_row < n_dia:
            dia_spec_int[i] = dia_intensities[orig_row]

    return ranked_rows, out_cols, out_vals, dia_spec_int, peak_idx_lookup


@njit(nogil=True)
def _create_entries_core_jit(
    centroid_breaks, all_frag_mz, all_frag_int, frag_offsets,
    all_top_n_local, top_n_offsets,
    prec_mzs, ms1_mz, ms1_tol,
    frac_lib_threshold, atleast_m, match_ms1):
    """JIT core of create_entries: searchsorted + filtering + flat array construction.

    Replaces ~200-400 GIL acquire/release cycles (from individual numpy calls) with
    a single GIL-free JIT call. This is the key enabler for effective multithreading.

    Args:
        centroid_breaks: float64[] — sorted DIA bin edges
        all_frag_mz: float64[] — all candidate fragment m/z, flattened
        all_frag_int: float64[] — all candidate fragment intensities, flattened
        frag_offsets: int32[n+1] — candidate i's frags at [offsets[i], offsets[i+1])
        all_top_n_local: int32[] — top-N indices (local within each candidate)
        top_n_offsets: int32[n+1] — candidate i's top-N at [offsets[i], offsets[i+1])
        prec_mzs: float64[n] — precursor m/z per candidate
        ms1_mz: float64[] — MS1 spectrum m/z (sorted)
        ms1_tol: float64 — relative MS1 tolerance
        frac_lib_threshold: float64 — minimum frac of lib intensity matched
        atleast_m: int — minimum top-N fragments matched
        match_ms1: bool — whether to require MS1 peak presence

    Returns:
        (passing, flat_rows, flat_cols, flat_vals, flat_offsets,
         ms1_error_out, all_coords, all_norm_int)
    """
    n_cands = len(frag_offsets) - 1
    total_frags = int(frag_offsets[n_cands])

    # Step 1: Searchsorted — which DIA bin does each fragment land in?
    # Odd coord = matched a DIA peak, even = unmatched
    all_coords = np.empty(total_frags, dtype=np.int64)
    for i in range(total_frags):
        all_coords[i] = np.searchsorted(centroid_breaks, all_frag_mz[i])

    # Step 2: Top-N match counting per candidate
    top_n_matched = np.zeros(n_cands, dtype=np.int32)
    for i in range(n_cands):
        for k in range(int(top_n_offsets[i]), int(top_n_offsets[i + 1])):
            global_idx = int(frag_offsets[i]) + int(all_top_n_local[k])
            if global_idx < total_frags and all_coords[global_idx] % 2 == 1:
                top_n_matched[i] += 1

    # Step 3: Normalized intensities + fraction of lib intensity matched
    all_norm_int = np.empty(total_frags, dtype=np.float64)
    frac_matched_arr = np.zeros(n_cands, dtype=np.float64)
    for i in range(n_cands):
        s = int(frag_offsets[i])
        e = int(frag_offsets[i + 1])
        int_sum = 0.0
        for k in range(s, e):
            int_sum += all_frag_int[k]
        inv_sum = 1.0 / int_sum if int_sum > 0.0 else 0.0
        for k in range(s, e):
            all_norm_int[k] = all_frag_int[k] * inv_sum
            if all_coords[k] % 2 == 1:
                frac_matched_arr[i] += all_norm_int[k]

    # Step 4: MS1 closest-peak error (vectorized searchsorted per candidate)
    n_ms1 = len(ms1_mz)
    ms1_error = np.full(n_cands, np.nan)
    if n_ms1 > 0:
        for i in range(n_cands):
            q = prec_mzs[i]
            idx = np.searchsorted(ms1_mz, q)
            left = max(0, idx - 1)
            right = min(idx, n_ms1 - 1)
            left_diff = (ms1_mz[left] - q) / q
            right_diff = (ms1_mz[right] - q) / q
            if abs(left_diff) <= abs(right_diff):
                closest = left_diff
            else:
                closest = right_diff
            if abs(closest) <= ms1_tol:
                ms1_error[i] = closest

    # Candidate filtering
    n_passing = 0
    passing = np.empty(n_cands, dtype=np.int32)
    for i in range(n_cands):
        ok = frac_matched_arr[i] > frac_lib_threshold and top_n_matched[i] > atleast_m
        if match_ms1:
            ok = ok and not np.isnan(ms1_error[i])
        if ok:
            passing[n_passing] = i
            n_passing += 1
    passing = passing[:n_passing]

    # Step 5: Build flat output arrays for matched fragments of passing candidates
    flat_offsets = np.zeros(n_passing + 1, dtype=np.int32)
    for j in range(n_passing):
        i = passing[j]
        s = int(frag_offsets[i])
        e = int(frag_offsets[i + 1])
        count = 0
        for k in range(s, e):
            if all_coords[k] % 2 == 1:
                count += 1
        flat_offsets[j + 1] = flat_offsets[j] + count

    total_matched = int(flat_offsets[n_passing])
    flat_rows = np.empty(total_matched, dtype=np.int32)
    flat_cols = np.empty(total_matched, dtype=np.int32)
    flat_vals = np.empty(total_matched, dtype=np.float64)

    for j in range(n_passing):
        i = passing[j]
        s = int(frag_offsets[i])
        e = int(frag_offsets[i + 1])
        pos = int(flat_offsets[j])
        for k in range(s, e):
            if all_coords[k] % 2 == 1:
                flat_rows[pos] = np.int32((all_coords[k] + 1) // 2 - 1)
                flat_cols[pos] = np.int32(j)
                flat_vals[pos] = all_norm_int[k]
                pos += 1

    # MS1 error for passing candidates
    ms1_error_out = np.empty(n_passing, dtype=np.float64)
    for j in range(n_passing):
        ms1_error_out[j] = ms1_error[passing[j]]

    return passing, flat_rows, flat_cols, flat_vals, flat_offsets, ms1_error_out, all_coords, all_norm_int


@njit(nogil=True)
def _create_entries_direct_jit(
    centroid_breaks,
    spec_data_mz, spec_data_int, spec_offsets, spec_lengths,
    topn_data, topn_offsets, topn_lengths,
    cand_indices,
    prec_mzs, ms1_mz, ms1_int, ms1_mob, ms1_tol,
    frac_lib_threshold, atleast_m, match_ms1,
    bin_mz, bin_int, bin_mob, mz_tol, im_tol, has_im,
    frag_names_data, dia_int):
    """Like _create_entries_core_jit but reads directly from library backing arrays.

    Instead of pre-flattened fragment arrays, this takes the library's contiguous
    spectrum_data (split into mz/int columns), spectrum_offsets, spectrum_lengths,
    and a list of candidate indices into those arrays. Eliminates the Python-level
    flattening that was the #1 GIL bottleneck.

    Args:
        centroid_breaks: float64[] — sorted DIA bin edges
        spec_data_mz:  float32[] — library spectrum_mz (all fragment m/z)
        spec_data_int: float32[] — library spectrum_int (all fragment intensities)
        spec_offsets:  int64[n_lib] — start offset in spec_data for each library entry
        spec_lengths:  int32[n_lib] — number of fragments for each library entry
        topn_data:     int32[] — library top_n_data (local fragment indices)
        topn_offsets:  int64[n_lib] — start offset in topn_data for each library entry
        topn_lengths:  int32[n_lib] — number of top-N entries for each library entry
        cand_indices:  int32/int64[] — which library entries are candidates
        prec_mzs:      float64[n_cands] — precursor m/z per candidate
        ms1_mz:        float64[] — MS1 spectrum m/z (sorted)
        ms1_int:       float64[] — MS1 spectrum intensities, parallel to ms1_mz
        ms1_tol:       float64 — relative MS1 tolerance
        frac_lib_threshold: float64 — minimum frac of lib intensity matched
        atleast_m:     int — minimum top-N fragments matched
        match_ms1:     bool — whether to require MS1 peak presence

    Returns:
        Same as _create_entries_core_jit:
        (passing, flat_rows, flat_cols, flat_vals, flat_offsets,
         ms1_error_out, all_coords, all_norm_int, frag_offsets)
    """
    n_cands = len(cand_indices)

    # Count total fragments across all candidates
    total_frags = 0
    for i in range(n_cands):
        total_frags += int(spec_lengths[cand_indices[i]])

    # Build frag_offsets for candidate-local indexing
    frag_offsets = np.empty(n_cands + 1, dtype=np.int32)
    frag_offsets[0] = 0
    for i in range(n_cands):
        frag_offsets[i + 1] = frag_offsets[i] + int(spec_lengths[cand_indices[i]])

    # Step 1: bin each fragment. all_coords uses the parity convention that the
    # rest of this function + _assemble_coo_jit rely on: odd => matched, with DIA
    # bin index = coord // 2; even => unmatched.
    all_coords = np.empty(total_frags, dtype=np.int64)
    all_frag_int = np.empty(total_frags, dtype=np.float64)

    # Per-candidate precursor IM (median of matched fragment mobilities). NaN for
    # the non-IM path or candidates with no m/z match; carried out so downstream
    # stages (e.g. fragment correlations) reuse it instead of re-deriving it.
    prec_im_out = np.full(n_cands, np.nan)

    # library fragment intensities are needed regardless of the matching path
    for i in range(n_cands):
        lib_idx = cand_indices[i]
        src_off = int(spec_offsets[lib_idx])
        src_len = int(spec_lengths[lib_idx])
        dst_off = int(frag_offsets[i])
        for k in range(src_len):
            all_frag_int[dst_off + k] = spec_data_int[src_off + k]

    if has_im and im_tol > 0.0:
        # 2D (m/z, IM) matching. Pass 1: most-intense m/z bin per fragment ->
        # prec_IM = median of matched bins' mobilities. Pass 2: match by m/z AND
        # |bin_mob - prec_IM| <= im_tol; encode bin index as 2*bin+1 (matched).
        mob_buf = np.empty(total_frags, dtype=np.float64)
        for i in range(n_cands):
            lib_idx = cand_indices[i]
            src_off = int(spec_offsets[lib_idx])
            src_len = int(spec_lengths[lib_idx])
            dst_off = int(frag_offsets[i])
            cnt = 0
            for k in range(src_len):
                b1 = _match_mz_only(bin_mz, bin_int, spec_data_mz[src_off + k], mz_tol)
                if b1 >= 0:
                    mob_buf[cnt] = bin_mob[b1]
                    cnt += 1
            if cnt == 0:
                for k in range(src_len):
                    all_coords[dst_off + k] = 0  # even => unmatched
                continue
            prec_im = np.median(mob_buf[:cnt])
            prec_im_out[i] = prec_im
            for k in range(src_len):
                b2 = _match_mz_im(bin_mz, bin_mob, spec_data_mz[src_off + k],
                                  prec_im, mz_tol, im_tol)
                if b2 >= 0:
                    all_coords[dst_off + k] = 2 * b2 + 1  # odd => matched, bin=b2
                else:
                    all_coords[dst_off + k] = 0           # even => unmatched
    else:
        # 1D m/z-only path (mzML / non-IM): searchsorted into centroid_breaks.
        for i in range(n_cands):
            lib_idx = cand_indices[i]
            src_off = int(spec_offsets[lib_idx])
            src_len = int(spec_lengths[lib_idx])
            dst_off = int(frag_offsets[i])
            for k in range(src_len):
                all_coords[dst_off + k] = np.searchsorted(
                    centroid_breaks, spec_data_mz[src_off + k])

    # Step 2: Top-N match counting per candidate
    top_n_matched = np.zeros(n_cands, dtype=np.int32)
    for i in range(n_cands):
        lib_idx = cand_indices[i]
        tn_off = int(topn_offsets[lib_idx])
        tn_len = int(topn_lengths[lib_idx])
        dst_off = int(frag_offsets[i])
        for k in range(tn_len):
            local_idx = int(topn_data[tn_off + k])
            if local_idx < int(spec_lengths[lib_idx]) and all_coords[dst_off + local_idx] % 2 == 1:
                top_n_matched[i] += 1

    # Step 3: Normalized intensities + fraction of lib intensity matched
    all_norm_int = np.empty(total_frags, dtype=np.float64)
    frac_matched_arr = np.zeros(n_cands, dtype=np.float64)
    for i in range(n_cands):
        s = int(frag_offsets[i])
        e = int(frag_offsets[i + 1])
        int_sum = 0.0
        for k in range(s, e):
            int_sum += all_frag_int[k]
        inv_sum = 1.0 / int_sum if int_sum > 0.0 else 0.0
        for k in range(s, e):
            all_norm_int[k] = all_frag_int[k] * inv_sum
            if all_coords[k] % 2 == 1:
                frac_matched_arr[i] += all_norm_int[k]

    # Step 4: MS1 precursor-peak error.
    #
    # IM gate first, intensity as the selector.  MS1 frames are not IM-banded --
    # they carry per-peak mobility across the whole acquisition range -- so a
    # peak is only eligible within ms1_tol of the precursor m/z and within
    # im_tol of this candidate's fragment-derived IM.  Among those, the most
    # intense is the precursor's peak (ties to the lowest m/z): the nearest by
    # m/z is, on uncentroided data, usually a single-count ion whose error
    # reflects the TOF grid rather than the precursor.  Falls back to the m/z
    # window alone when there is no mobility to gate on: non-IM data, or a
    # candidate whose fragments yielded no prec_im.
    n_ms1 = len(ms1_mz)
    ms1_error = np.full(n_cands, np.nan)
    ms1_has_mob = len(ms1_mob) == n_ms1
    if n_ms1 > 0:
        for i in range(n_cands):
            q = prec_mzs[i]
            p_im = prec_im_out[i]
            gate_im = ms1_has_mob and im_tol > 0.0 and p_im == p_im
            tol_da = ms1_tol * q
            best = np.nan
            best_int = -np.inf
            idx = np.searchsorted(ms1_mz, q)
            # Upward (increasing m/z): strictly greater keeps the lower m/z on ties.
            j = idx
            while j < n_ms1 and (ms1_mz[j] - q) <= tol_da:
                if (not gate_im) or abs(ms1_mob[j] - p_im) <= im_tol:
                    if ms1_int[j] > best_int:
                        best_int = ms1_int[j]
                        best = (ms1_mz[j] - q) / q
                j += 1
            # Downward (decreasing m/z): greater-or-equal takes the lower m/z on ties.
            j = idx - 1
            while j >= 0 and (q - ms1_mz[j]) <= tol_da:
                if (not gate_im) or abs(ms1_mob[j] - p_im) <= im_tol:
                    if ms1_int[j] >= best_int:
                        best_int = ms1_int[j]
                        best = (ms1_mz[j] - q) / q
                j -= 1
            ms1_error[i] = best

    # Step 4b: entropy-normalized spectral angle, theta / sqrt(D1 - 1), over the
    # candidate's MONOISOTOPIC library fragments.  theta is the angle between
    # sqrt(library) and sqrt(observed) intensities (unmatched fragments observed
    # as 0); D1 = exp(Shannon entropy of the library intensity shares), the
    # effective number of library ions.  Lower is better.
    ent_angle = np.full(n_cands, np.nan)
    for i in range(n_cands):
        lib_idx = cand_indices[i]
        src_off = int(spec_offsets[lib_idx])
        src_len = int(spec_lengths[lib_idx])
        dst_off = int(frag_offsets[i])
        lib_sum = 0.0
        for k in range(src_len):
            if ((frag_names_data[src_off + k] >> 17) & 0xF) == 0:
                lib_sum += spec_data_int[src_off + k]
        if lib_sum <= 0.0:
            continue
        dot = 0.0
        a2 = 0.0
        b2 = 0.0
        ent = 0.0
        for k in range(src_len):
            if ((frag_names_data[src_off + k] >> 17) & 0xF) != 0:
                continue
            li = spec_data_int[src_off + k]
            if li <= 0.0:
                continue
            p = li / lib_sum
            ent -= p * math.log(p)
            b2 += li
            c = all_coords[dst_off + k]
            if c % 2 == 1:
                ob = dia_int[c // 2]
                if ob > 0.0:
                    dot += math.sqrt(ob) * math.sqrt(li)
                    a2 += ob
        if a2 > 0.0 and b2 > 0.0:
            cosv = dot / math.sqrt(a2 * b2)
            if cosv > 1.0:
                cosv = 1.0
            theta = math.acos(cosv)
        else:
            theta = math.pi / 2.0
        d1m1 = math.exp(ent) - 1.0
        if d1m1 < 1e-6:
            d1m1 = 1e-6
        ent_angle[i] = theta / math.sqrt(d1m1)

    # Candidate filtering
    n_passing = 0
    passing = np.empty(n_cands, dtype=np.int32)
    for i in range(n_cands):
        # TODO: atleast_m is misnamed and inconsistent: this filter is
        # top_n_matched > atleast_m ("at least atleast_m + 1") while fragment_index
        # uses >= ("at least atleast_m").  Rename it to what it is (e.g.
        # min_topn_matched) and make every use the same comparison, so one name
        # means one threshold everywhere.  Switching this filter to >= (at least 3)
        # lost about 1,000 precursors on the timsTOF 200 pg test, so retune when
        # changing it.
        ok = frac_matched_arr[i] > frac_lib_threshold and top_n_matched[i] > atleast_m
        if match_ms1:
            ok = ok and not np.isnan(ms1_error[i])
        if ok:
            passing[n_passing] = i
            n_passing += 1
    passing = passing[:n_passing]

    # Step 5: Build flat output arrays for matched fragments of passing candidates
    flat_offsets = np.zeros(n_passing + 1, dtype=np.int32)
    for j in range(n_passing):
        i = passing[j]
        s = int(frag_offsets[i])
        e = int(frag_offsets[i + 1])
        count = 0
        for k in range(s, e):
            if all_coords[k] % 2 == 1:
                count += 1
        flat_offsets[j + 1] = flat_offsets[j] + count

    total_matched = int(flat_offsets[n_passing])
    flat_rows = np.empty(total_matched, dtype=np.int32)
    flat_cols = np.empty(total_matched, dtype=np.int32)
    flat_vals = np.empty(total_matched, dtype=np.float64)

    for j in range(n_passing):
        i = passing[j]
        s = int(frag_offsets[i])
        e = int(frag_offsets[i + 1])
        pos = int(flat_offsets[j])
        for k in range(s, e):
            if all_coords[k] % 2 == 1:
                flat_rows[pos] = np.int32((all_coords[k] + 1) // 2 - 1)
                flat_cols[pos] = np.int32(j)
                flat_vals[pos] = all_norm_int[k]
                pos += 1

    # MS1 error for passing candidates
    ms1_error_out = np.empty(n_passing, dtype=np.float64)
    for j in range(n_passing):
        ms1_error_out[j] = ms1_error[passing[j]]

    ent_angle_out = np.empty(n_passing, dtype=np.float64)
    topn_out = np.empty(n_passing, dtype=np.int32)
    frac_out = np.empty(n_passing, dtype=np.float64)
    for j in range(n_passing):
        ent_angle_out[j] = ent_angle[passing[j]]
        topn_out[j] = top_n_matched[passing[j]]
        frac_out[j] = frac_matched_arr[passing[j]]

    return passing, flat_rows, flat_cols, flat_vals, flat_offsets, ms1_error_out, all_coords, all_norm_int, frag_offsets, prec_im_out, ent_angle_out, topn_out, frac_out


def _keep_passing(keep, passing, flat_rows, flat_cols, flat_vals, flat_offsets, *per_passing):
    """Keep the passing candidates where *keep* is True, in their order: the flat
    COO arrays are cut to their segments and their columns renumbered, and each
    per-passing array is filtered the same way."""
    kept = np.flatnonzero(keep)
    seg = [np.arange(flat_offsets[j], flat_offsets[j + 1]) for j in kept]
    take = np.concatenate(seg) if seg else np.empty(0, dtype=np.int64)
    lengths = np.diff(flat_offsets)[kept]
    new_offsets = np.zeros(len(kept) + 1, dtype=flat_offsets.dtype)
    np.cumsum(lengths, out=new_offsets[1:])
    new_cols = np.repeat(np.arange(len(kept), dtype=flat_cols.dtype), lengths)
    return (passing[kept], flat_rows[take], new_cols, flat_vals[take], new_offsets,
            *[a[kept] for a in per_passing])


def create_entries_direct(centroid_breaks,
                          spec_data_mz, spec_data_int,
                          spec_offsets, spec_lengths,
                          topn_data, topn_offsets, topn_lengths,
                          candidate_indices,
                          atleast_m,
                          prec_mzs,
                          ms1_spec,
                          ms1_tol,
                          bin_mz=None,
                          bin_int=None,
                          bin_mob=None,
                          mz_tol=0.0,
                          im_tol=0.0,
                          has_im=False,
                          frag_names_data=None,
                          dia_int=None):
    """Like create_entries but reads directly from library backing arrays.

    Eliminates Python-level flattening of candidate_peaks and top_n_idxs
    by passing the library's contiguous storage into the JIT.

    Args:
        centroid_breaks: float64[] — sorted DIA bin edges
        spec_data_mz:  float64[] — library spectrum_mz (contiguous 1D)
        spec_data_int: float64[] — library spectrum_int (contiguous 1D)
        spec_offsets:  int64[] — spectrum_offsets from library
        spec_lengths:  int32[] — spectrum_lengths from library
        topn_data:     int32[] — top_n_data from library
        topn_offsets:  int64[] — top_n_offsets from library
        topn_lengths:  int32[] — top_n_lengths from library
        candidate_indices: int[] — internal library indices for each candidate
        atleast_m: int — minimum top-N fragments matched
        prec_mzs: float64[] — precursor m/z per candidate
        ms1_spec: MS1 spectrum object with .mz (and .mobility when the data carries IM)
        ms1_tol: float64 — relative MS1 tolerance

    Returns:
        (passing, flat_rows, flat_cols, flat_vals, flat_offsets, ms1_error,
         all_coords, all_norm_int, frag_offsets, prec_im, cand_idx):
        passing indexes the candidates that pass; prec_im is per passing
        candidate; cand_idx is candidate_indices as int64.
    """
    n_cands = len(candidate_indices)
    if n_cands == 0:
        return (np.empty(0, np.int64),
                np.empty(0, np.int32), np.empty(0, np.int32), np.empty(0, np.float64),
                np.zeros(1, np.int32),
                np.array([], dtype=np.float64),
                np.empty(0, np.int64), np.empty(0, np.float64),
                np.zeros(1, np.int32), np.empty(0, np.float64), np.empty(0, np.int64),
                np.empty(0, np.float64), np.empty(0, np.int32), np.empty(0, np.float64))

    cand_idx_arr = np.asarray(candidate_indices, dtype=np.int64)

    passing, flat_rows, flat_cols, flat_vals, flat_offsets, ms1_error_out, \
        all_coords, all_norm_int, frag_offsets, prec_im_out, ent_angle_out, topn_out, frac_out = \
        _create_entries_direct_jit(
            np.ascontiguousarray(centroid_breaks, dtype=np.float64),
            spec_data_mz, spec_data_int,
            spec_offsets, spec_lengths,
            topn_data, topn_offsets, topn_lengths,
            cand_idx_arr,
            np.ascontiguousarray(prec_mzs, dtype=np.float64),
            np.ascontiguousarray(ms1_spec.mz, dtype=np.float64),
            np.ascontiguousarray(ms1_spec.intens, dtype=np.float64),
            np.ascontiguousarray(ms1_spec.mobility, dtype=np.float64)
            if getattr(ms1_spec, "mobility", None) is not None
            else np.empty(0, dtype=np.float64),
            float(ms1_tol),
            float(config.args.lib_frac), int(atleast_m),
            not config.args.no_ms1_req,
            np.ascontiguousarray(
                bin_mz if bin_mz is not None else np.zeros(0), dtype=np.float64),
            np.ascontiguousarray(
                bin_int if bin_int is not None else np.zeros(0), dtype=np.float64),
            np.ascontiguousarray(
                bin_mob if bin_mob is not None else np.zeros(0), dtype=np.float64),
            float(mz_tol), float(im_tol), bool(has_im),
            np.ascontiguousarray(frag_names_data if frag_names_data is not None
                                 else np.zeros(len(spec_data_mz), dtype=np.int32)),
            np.ascontiguousarray(dia_int if dia_int is not None else np.zeros(0), dtype=np.float64))

    passing = passing.astype(np.int64)
    return (passing,
            flat_rows, flat_cols, flat_vals, flat_offsets, ms1_error_out,
            all_coords, all_norm_int, frag_offsets,
            prec_im_out[passing].astype(np.float64), cand_idx_arr,
            ent_angle_out, topn_out, frac_out)


def _flatten_splits(row_idx_split, col_idx_split, val_split):
    """Concatenate per-candidate split arrays into flat arrays with an offset table.

    The split arrays (lists of variable-length numpy arrays, one per candidate)
    can't be passed to numba directly. This flattens them into contiguous arrays
    with an offset table so candidate j's data is at flat[offsets[j]:offsets[j+1]].

    Args:
        row_idx_split: list of int arrays — DIA peak indices per candidate
        col_idx_split: list of int arrays — candidate column indices per candidate
        val_split:     list of float arrays — normalized lib intensities per candidate

    Returns:
        all_rows:  int32[]   — concatenated row indices
        all_vals:  float64[] — concatenated intensity values
        all_cols:  int32[]   — concatenated column indices
        offsets:   int32[n+1] — offset table, candidate j spans [offsets[j], offsets[j+1])
    """
    n = len(row_idx_split)
    if n == 0:
        return (np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.int32), np.zeros(1, dtype=np.int32))
    all_rows = np.concatenate(row_idx_split).astype(np.int32)
    all_vals = np.concatenate(val_split).astype(np.float64)
    all_cols = np.concatenate(col_idx_split).astype(np.int32)
    offsets = np.zeros(n + 1, dtype=np.int32)
    for i in range(n):
        offsets[i + 1] = offsets[i] + len(row_idx_split[i])
    return all_rows, all_vals, all_cols, offsets


def _lasso_nnls(A, b, sample_weight=None):
    """
    NNLS via sklearn Lasso with near-zero L1 penalty, positive=True.

    Args:
        A: Sparse matrix (n_peaks x n_candidates), will be converted to CSR
        b: 1D array of observed intensities
        sample_weight: Optional per-sample weights for weighted least squares

    Returns:
        1D numpy array of non-negative coefficients
    """
    A_csr = A.tocsr() if not sparse.issparse(A) or A.format != 'csr' else A

    # Normalize b so tol is scale-independent across spectra
    s = np.max(np.abs(b)) or 1.0

    model = Lasso(
        alpha=1e-5 / s,
        positive=True,
        fit_intercept=False,
        selection='random',
        random_state=42,
        tol=1e-4,
        max_iter=20000,
    )
    model.fit(A_csr, b / s, sample_weight=sample_weight)
    return model.coef_ * s


# ── JIT helpers for huber_nnls_irls (replace scipy sparse ops to release GIL) ──

@njit(nogil=True)
def _coo_gram_and_Aty(rows, cols, vals, y, n_cols):
    """A^T y and the largest absolute correlation between two different
    columns of A, from COO arrays.  Returns (Aty, max_corr).

    Bit-identical to forming the dense Gram matrix (gram / outer(norms, norms),
    zero norms set to 1, diagonal zeroed, max(abs(.))), without allocating it:
    that matrix is n_cols^2 float64 -- 22 GB for one 55,000-candidate fit.
    Only columns that share a row have a non-zero Gram entry, so each column
    accumulates its Gram row against later columns in an O(n_cols) scratch
    vector, adding the same products in the same order as the dense build
    (rows ascending, then row-sorted position, the earlier entry's value
    first), and duplicate (row, column) entries included.  Memory is
    O(nnz + n_cols).
    """
    Aty = np.zeros(n_cols)
    for i in range(len(vals)):
        Aty[cols[i]] += vals[i] * y[rows[i]]

    n = len(vals)
    # Entries grouped by row, in the dense build's order (same argsort).
    order = np.argsort(rows)
    row_lo = np.empty(n, np.int64)   # [row_lo, row_hi): sorted span of the entry's row
    row_hi = np.empty(n, np.int64)
    i = 0
    while i < n:
        r = rows[order[i]]
        j = i
        while j < n and rows[order[j]] == r:
            j += 1
        for t in range(i, j):
            row_lo[t] = i
            row_hi[t] = j
        i = j
    s_cols = np.empty(n, np.int64)
    s_vals = np.empty(n)
    for t in range(n):
        s_cols[t] = cols[order[t]]
        s_vals[t] = vals[order[t]]

    # Each column's sorted positions, ascending (so rows ascending).
    col_ptr = np.zeros(n_cols + 1, np.int64)
    for t in range(n):
        col_ptr[s_cols[t] + 1] += 1
    for c in range(n_cols):
        col_ptr[c + 1] += col_ptr[c]
    col_pos = np.empty(n, np.int64)
    fill = col_ptr[:-1].copy()
    for t in range(n):
        c = s_cols[t]
        col_pos[fill[c]] = t
        fill[c] += 1

    # Diagonal: per row, pairs (p, q), p <= q, both in the column.
    norms = np.empty(n_cols)
    for c in range(n_cols):
        g = 0.0
        k = col_ptr[c]
        end = col_ptr[c + 1]
        while k < end:
            m = k
            while m < end and row_lo[col_pos[m]] == row_lo[col_pos[k]]:
                m += 1
            for u in range(k, m):
                vp = s_vals[col_pos[u]]
                for w in range(u, m):
                    g += vp * s_vals[col_pos[w]]
            k = m
        norms[c] = np.sqrt(g)
        if norms[c] == 0:
            norms[c] = 1.0
    # A NaN norm makes the dense ratio NaN for every pair with that column,
    # shared row or not, so the dense scan returns NaN.
    if n_cols > 1:
        for c in range(n_cols):
            if norms[c] != norms[c]:
                return Aty, norms[c]

    # Off-diagonal: for column a, every later column b sharing a row.
    acc = np.zeros(n_cols)
    seen = np.zeros(n_cols, np.bool_)
    touched = np.empty(n_cols, np.int64)
    max_corr = 0.0
    for a in range(n_cols):
        n_touched = 0
        k = col_ptr[a]
        end = col_ptr[a + 1]
        while k < end:
            lo = row_lo[col_pos[k]]
            hi = row_hi[col_pos[k]]
            m = k
            while m < end and row_lo[col_pos[m]] == lo:
                m += 1
            # Positions of column a in this row: col_pos[k:m].  Visit the row's
            # pairs (p, q), p < q, that involve a, in the dense order.
            for p in range(lo, hi):
                cp = s_cols[p]
                vp = s_vals[p]
                if cp == a:
                    for q in range(p + 1, hi):
                        cq = s_cols[q]
                        if cq > a:
                            acc[cq] += vp * s_vals[q]
                            if not seen[cq]:
                                seen[cq] = True
                                touched[n_touched] = cq
                                n_touched += 1
                elif cp > a:
                    for u in range(k, m):
                        q = col_pos[u]
                        if q > p:
                            acc[cp] += vp * s_vals[q]
                            if not seen[cp]:
                                seen[cp] = True
                                touched[n_touched] = cp
                                n_touched += 1
            k = m
        for x in range(n_touched):
            b = touched[x]
            v = abs(acc[b] / (norms[a] * norms[b]))
            if v != v:
                return Aty, v          # NaN: np.max propagates it
            if v > max_corr:
                max_corr = v
            acc[b] = 0.0
            seen[b] = False
    return Aty, max_corr


@njit(nogil=True)
def _coo_Ax(rows, cols, vals, x, n_rows):
    """Compute A @ x from COO arrays. Returns dense (n_rows,) vector."""
    result = np.zeros(n_rows)
    for i in range(len(vals)):
        result[rows[i]] += vals[i] * x[cols[i]]
    return result


@njit(nogil=True)
def _coo_to_csc_arrays(rows, cols, vals, n_rows, n_cols):
    """Convert COO to CSC arrays (data, indices, indptr).
    Returns (data, indices, indptr) suitable for scipy csc_matrix constructor."""
    # Count entries per column
    col_counts = np.zeros(n_cols, dtype=np.int64)
    for i in range(len(cols)):
        col_counts[cols[i]] += 1
    # Build indptr
    indptr = np.zeros(n_cols + 1, dtype=np.int64)
    for c in range(n_cols):
        indptr[c + 1] = indptr[c] + col_counts[c]
    # Fill data and indices
    data = np.empty(len(vals))
    indices = np.empty(len(vals), dtype=np.int64)
    pos = np.zeros(n_cols, dtype=np.int64)  # current write position per column
    for i in range(len(vals)):
        c = cols[i]
        dest = indptr[c] + pos[c]
        data[dest] = vals[i]
        indices[dest] = rows[i]
        pos[c] += 1
    return data, indices, indptr


@njit(nogil=True)
def _compute_mad_cutoff(residuals, b, c):
    """Compute MAD-based cutoff from over-predicted observed peaks.
    Returns cutoff value (scalar)."""
    # Collect abs residuals where over-predicted and observed
    count = 0
    for i in range(len(residuals)):
        if residuals[i] > 0.0 and b[i] > 0.0:
            count += 1
    if count == 0:
        return 1.0
    abs_r = np.empty(count)
    k = 0
    for i in range(len(residuals)):
        if residuals[i] > 0.0 and b[i] > 0.0:
            abs_r[k] = abs(residuals[i])
            k += 1
    med = np.median(abs_r)
    deviations = np.empty(count)
    for i in range(count):
        deviations[i] = abs(abs_r[i] - med)
    mad = np.median(deviations)
    if mad > 0.0:
        return c * (mad / 0.6745)
    return 1.0


@njit(nogil=True)
def _irls_weights(residuals, b, cutoff):
    """Compute Tukey biweight weights for under-predicted observed peaks.
    Returns (new_weights, max_change_from_ones)."""
    n = len(residuals)
    new_weights = np.ones(n)
    max_change = 0.0
    for i in range(n):
        if residuals[i] < 0.0 and b[i] > 0.0:
            abs_r = -residuals[i]  # residuals[i] < 0, so abs = -residuals[i]
            u = abs_r / cutoff
            if u <= 1.0:
                w = (1.0 - u * u) ** 2
            else:
                w = 0.0
            new_weights[i] = w
            change = abs(1.0 - w)
            if change > max_change:
                max_change = change
    return new_weights, max_change


def huber_nnls_irls(coo_vals, coo_rows, coo_cols, n_rows, n_cols, b,
                    max_iter=1, tol=1e-4, c=4.685):
    """
    Asymmetric IRLS-weighted NNLS with Tukey biweight on under-predicted peaks.

    Accepts flat COO arrays instead of scipy sparse to avoid GIL-holding format
    conversions. All pre-solver math is JIT'd (nogil=True); only enet_path
    requires a scipy CSC matrix.

    First pass: unweighted NNLS.
    Subsequent passes:
      - Under-prediction at observed peaks (residuals < 0, b > 0):
        Tukey bisquare weights — smoothly downweights large under-predictions,
        completely rejects beyond c * MAD. This is forgiving because other
        peptides in the mixture can explain the extra observed signal.
      - All other peaks: weight = 1.0 (full penalty for over-prediction
        and false signal at zeros).

    Args:
        coo_vals: Non-zero values (flat array)
        coo_rows: Row indices (flat array)
        coo_cols: Column indices (flat array)
        n_rows: Number of rows in sparse matrix
        n_cols: Number of columns in sparse matrix
        b: Observed intensities (n_rows,)
        max_iter: Maximum IRLS iterations
        tol: Convergence tolerance on weight changes
        c: Tukey biweight tuning constant (multiples of MAD)

    Returns:
        dict with 'x': coefficients, 'weights': final sample weights
    """
    # Normalize b so tol is scale-independent
    s = float(np.max(np.abs(b)) or 1.0)
    y = b / s

    # A^T y and the max column correlation in one JIT pass (nogil)
    Aty, max_corr = _coo_gram_and_Aty(coo_rows, coo_cols, coo_vals, y, n_cols)

    # Data-driven regularization
    alpha_max = np.max(np.abs(Aty)) / n_rows
    alpha = alpha_max * 1e-4  # dynamic range within a spectrum is roughly 1,000x

    # Data-driven l1_ratio from max pairwise column correlation
    l1_ratio = min(max(1 - max_corr ** 2, 0.1), 0.9)

    # Build CSC once from COO via JIT (nogil) — only format enet_path accepts.
    # Replaces COO→CSR→CSC conversion chain that held the GIL.
    # One matrix for both solves: the first reads X_w as built (its data equals
    # csc_data); the IRLS loop then reweights X_w.data in place from csc_data.
    csc_data, csc_indices, csc_indptr = _coo_to_csc_arrays(
        coo_rows, coo_cols, coo_vals, n_rows, n_cols)
    X_w = csc_matrix((csc_data.copy(), csc_indices, csc_indptr), shape=(n_rows, n_cols))

    coef = np.zeros(n_cols)
    _enet_path = enet_path.__wrapped__

    # Initial solve with uniform weights
    weights = np.ones(n_rows)
    _rng = _get_enet_rng()  # thread-local, reseeded to 42 (enet_path mutates rng state)
    _, coef_path, _, n_iters = _enet_path(
        X_w, y,
        l1_ratio=l1_ratio,
        alphas=[alpha],
        positive=True,
        coef_init=coef,
        check_input=False,
        precompute=False,  # sparse input: _pre_fit would only set this, after allocating
        return_n_iter=True,
        max_iter=20000,
        tol=1e-3,
        selection='random',
        random_state=_rng,
    )
    coef = coef_path[:, 0]
    initial_n_iter = n_iters[0]

    x = coef * s
    # Residuals via JIT (nogil) instead of A_csr.dot(x) which holds GIL
    residuals = _coo_Ax(coo_rows, coo_cols, coo_vals, x, n_rows) - b

    # Compute cutoff from over-predicted observed peaks (nogil JIT)
    cutoff = _compute_mad_cutoff(residuals, b, c)

    for _ in range(max_iter):
        x = coef * s
        residuals = _coo_Ax(coo_rows, coo_cols, coo_vals, x, n_rows) - b

        # JIT'd weight computation (nogil) — replaces boolean indexing + np.where
        new_weights, max_change = _irls_weights(residuals, b, cutoff)

        # Check convergence
        if max_change < tol:
            break
        weights = new_weights
        weights *= weights.size / weights.sum()

        # Scale CSC data by sqrt(weights) per row — mutate in-place to avoid
        # rebuilding the csc_matrix object each iteration
        sw = np.sqrt(weights)
        X_w.data[:] = csc_data * np.take(sw, csc_indices)
        y_w = y * sw

        # Warm-started from previous coefficients
        _, coef_path, _, n_iters = _enet_path(
            X_w, y_w,
            l1_ratio=l1_ratio,
            alphas=[alpha],
            positive=True,
            coef_init=coef,
            check_input=False,
            precompute=False,
            return_n_iter=True,
            max_iter=20000,
            tol=1e-3,
            selection='random',
            random_state=_rng,
        )
        coef = coef_path[:, 0]

    return {'x': coef * s, 'weights': weights,
            'initial_n_iter': initial_n_iter, 'robust_n_iter': n_iters[0],
            'alpha_max': alpha_max, 'l1_ratio': l1_ratio}


def get_closest_ms1(prec_rt, ms1_spectra, ms1_rt=None):
    if ms1_rt is None:
        ms1_rt = np.array([i.RT for i in ms1_spectra])
    closest_ms1_scan_idx = closest_ms1spec(prec_rt, ms1_rt)
    ms1_spec = ms1_spectra[closest_ms1_scan_idx]
    return ms1_spec

def get_scribe(
    row_idx_split,
    col_idx_split,
    prec_val_split,
    val_obs
):
    """
    Calculate Scribe scores for each precursor (Searle, Shannon, Wilburn, 2023, PMID: 36695531)
    
    This function computes the Scribe score, which measures spectral similarity by comparing
    the normalized distribution of fragment ion intensities between predicted and observed spectra.
    Lower scores indicate better matches.
    
    Args:
        row_idx_split (list): List of arrays containing row indices for each precursor's fragments.
        col_idx_split (list): List of arrays containing column indices for each precursor.
        prec_val_split (list): List of arrays containing predicted intensity values for each precursor's fragments.
        val_obs (numpy.ndarray): Array of observed intensity values.
        
    Returns:
        numpy.ndarray: Array of SCRIBE scores for each precursor, one score per precursor.
    """
    n = len(row_idx_split)
    if n > 0:
        #Sum of sqrt of predicted fragment intensities for each precursor/column
        h_sqrt_sum = np.zeros(n)
        #Sum of sqrt of observed fragment intensities for each precursor/column
        x_sqrt_sum = np.zeros(n)
        scribe_scores = np.zeros(n)
        for j in range(n):
            for (i, val) in zip(row_idx_split[j], prec_val_split[j]):
                h_sqrt_sum[j] += np.sqrt(val)
                x_sqrt_sum[j] += np.sqrt(val_obs[i])
        
        for j in range(n):
            for (i, val) in zip(row_idx_split[j], prec_val_split[j]):
                scribe_scores[j] += (
                    (np.sqrt(val)/h_sqrt_sum[j]) - 
                    (np.sqrt(val_obs[i])/x_sqrt_sum[j])
                )**2

        return scribe_scores
    else:
        return np.zeros(0)

def get_residuals(
    ref_sparse_val,  # sparse values for reference data
    ref_sparse_row,  # sparse rows for reference data
    ref_sparse_col,  # sparse cols for reference data
    decoy_sparse_val,  # sparse values for decoy data
    decoy_sparse_row,  # sparse rows for decoy data
    decoy_sparse_col,  # sparse cols for decoy data
    val_obs,  # observed values. the 'b' in Ax = b
    coeffs,  # coefficients. the 'x' in Ax = b
    ref_spec_offset,
    decoy_spec_offset,
):
    """
    Calculate residuals (Ax - b) and prediction values for both reference and decoy data.
    
    This function computes the predicted values by multiplying sparse matrix representations 
    of reference and decoy data by the coefficient vector, then calculates residuals 
    as the difference between observed and predicted values.
    
    Args:
        ref_sparse_val (list): List of arrays with sparse values for reference data.
        ref_sparse_row (list): List of arrays with sparse row indices for reference data.
        ref_sparse_col (list): List of arrays with sparse column indices for reference data.
        decoy_sparse_val (list): List of arrays with sparse values for decoy data.
        decoy_sparse_row (list): List of arrays with sparse row indices for decoy data.
        decoy_sparse_col (list): List of arrays with sparse column indices for decoy data.
        val_obs (numpy.ndarray): Observed values (the 'b' in Ax = b).
        coeffs (numpy.ndarray): Coefficients from the fit (the 'x' in Ax = b).
        
    Returns:
        tuple: A tuple containing:
            - residuals (numpy.ndarray): Residuals between observed and predicted values.
            - y_pred (numpy.ndarray): Predicted values calculated as A*x.
    """
    
    def _compute_prediction(sparse_val, sparse_row, sparse_col, coeff_array, y_pred, offset):
        """Helper function to compute predictions for a set of sparse data"""
        for j in range(len(sparse_row)):
            for row, col, val in zip(sparse_row[j], sparse_col[j], sparse_val[j]):
                y_pred[row] += val * coeff_array[col+offset]
        return y_pred
    
    coeffs = np.asarray(coeffs).ravel()
    N = len(val_obs)  # Number of rows in the sparse matrix (A)
    
    # Initialize prediction array
    y_pred = np.zeros(N)
    
    # Compute predictions for reference data
    y_pred = _compute_prediction(ref_sparse_val, ref_sparse_row, ref_sparse_col, coeffs, y_pred, ref_spec_offset)
    
    # Add predictions for decoy data
    y_pred = _compute_prediction(decoy_sparse_val, decoy_sparse_row, decoy_sparse_col, coeffs, y_pred, decoy_spec_offset)
    
    # Compute residuals
    #r = np.zeros_like(y_pred)
    
    # Residuals for matched peaks (where we have observations)
    r = val_obs - y_pred

    return r, y_pred

def max_matched_residual(
    row_idx_split,
    residuals
):
    """
    Find the maximum residual for each precursor's matched peaks.
    
    This function finds the largest residual value among the matched peaks
    for each precursor, which can indicate the worst-fit fragment.
    
    Args:
        row_idx_split (list): List of arrays containing row indices for each precursor's fragments.
        residuals (numpy.ndarray): Array of residuals between observed and predicted values.
        
    Returns:
        numpy.ndarray: Array of maximum residual values for each precursor.
    """
    n = len(row_idx_split)
    if n > 0:
        max_matched_residuals = np.zeros(n)
        for j in range(n):
            for (i, val) in zip(row_idx_split[j], residuals):
                if val > max_matched_residuals[j]:
                    max_matched_residuals[j] = val
        return max_matched_residuals
    else:
        return np.zeros(0)

def gof_stat(
    row_idx_split,
    col_idx_split,
    val_split,
    residuals,
    val_obs,
    coeffs,
    offset
):

    """
    Calculate goodness-of-fit statistics and maximum residuals for each precursor.
    
    This function computes several metrics to assess fit quality:
    1. Overall goodness-of-fit statistic based on sum of residuals to sum of fitted peaks
    2. Maximum residual for matched peaks (peaks with observed intensity)
    3. Maximum residual for unmatched peaks (peaks with near-zero observed intensity)
    
    All metrics are log-transformed and normalized by the sum of fitted peaks.
    
    Args:
        row_idx_split (list): List of arrays containing row indices for each precursor's fragments.
        col_idx_split (list): List of arrays containing column indices for each precursor.
        val_split (list): List of arrays containing predicted intensity values for each precursor's fragments.
        residuals (numpy.ndarray): Array of residuals between observed and predicted values.
        val_obs (numpy.ndarray): Array of observed intensity values.
        coeffs (numpy.ndarray): Coefficients from the fit.
        
    Returns:
        tuple: A tuple containing:
            - result (numpy.ndarray): Goodness-of-fit score for each precursor (log2 of residuals/fitted).
            - max_unmatched_residuals (numpy.ndarray): Maximum residual for unmatched peaks, normalized and log-transformed.
            - max_matched_residuals (numpy.ndarray): Maximum residual for matched peaks, normalized and log-transformed.
    """
    coeffs = np.asarray(coeffs).ravel()
    n = len(row_idx_split)
    if n > 0:
        sum_of_residuals = np.zeros(n)
        sum_of_fitted_peaks = np.zeros(n)
        result = np.zeros(n)
        max_unmatched_residuals = np.zeros(n)
        max_matched_residuals = np.zeros(n)
        for j in range(n):
            max_unmatched_residual = 0.0
            max_matched_residual = 0.0
            for (row_idx, col_idx, val) in zip(row_idx_split[j], col_idx_split[j], val_split[j]):
                r = abs(residuals[row_idx])
                sum_of_residuals[j] += r
                sum_of_fitted_peaks[j] += abs(coeffs[col_idx+offset]*val)
                if (val_obs[row_idx] > 1e-6):
                    if r > max_matched_residual:
                        max_matched_residual = r
                elif (val_obs[row_idx] < 1e-6):
                    if r > max_unmatched_residual:
                        max_unmatched_residual = r
            max_unmatched_residuals[j] = max_unmatched_residual
            max_matched_residuals[j] = max_matched_residual

        #Handle bad values         
        for j in range(n):
            if sum_of_fitted_peaks[j] == 0:
                sum_of_fitted_peaks[j] = 1e-6
            if sum_of_residuals[j] == 0:
                sum_of_residuals[j] = 1e-6  # Perfect agreement (no residuals, no signal)
            result[j] = np.log2(sum_of_residuals[j] / sum_of_fitted_peaks[j])
            max_matched_residuals[j] = np.log2(max_matched_residuals[j]/(sum_of_fitted_peaks[j] + 1e-10) + 1e-10)
            max_unmatched_residuals[j] = np.log2(max_unmatched_residuals[j]/(sum_of_fitted_peaks[j] + 1e-10) + 1e-10)
        return result, max_unmatched_residuals, max_matched_residuals 
    else:
        return np.zeros(0), np.zeros(0), np.zeros(0)

def get_manhattan_distance(
    row_idx_split,
    col_idx_split,
    prec_val_split,
    val_obs,
    y_pred  # Changed from coeffs to y_pred
):
    """
    Calculate fit metrics between predicted and observed fragment intensity values.
    
    This function computes two metrics for each precursor:
    1. Modified Manhattan distance: Sum of absolute differences between predicted and observed 
       values, normalized by sum of observed values and log-transformed. Higher (less negative) 
       values indicate better fits.
    2. Spectral contrast angle: Spectral contrast between model (Ax) and observed (b) internsities for 
    the fragments matching each respective precursor
    
    Parameters
    ----------
    row_idx_split : list of numpy.ndarray
        List of arrays containing row indices for each precursor's fragments.
    col_idx_split : list of numpy.ndarray
        List of arrays containing column indices for each precursor.
    prec_val_split : list of numpy.ndarray
        List of arrays containing predicted intensity values for each precursor's fragments.
    val_obs : numpy.ndarray
        Array of observed intensity values.
    y_pred : numpy.ndarray
        Array of predicted intensity values after applying model coefficients.
    
    Returns
    -------
    manhattan_distances : numpy.ndarray
        Array of modified Manhattan distances for each precursor, with higher values 
        indicating better fits.
    fitted_spectral_contrast : numpy.ndarray
        Array of spectral contrast angles for each precursor
    
    Notes
    -----
    - Edge cases are handled: when sum of observed values is zero (bad fit) or 
      Manhattan distance is zero (perfect fit).
    - The col_idx_split parameter is not used in the current implementation.
    """
    n = len(row_idx_split)
    N = len(val_obs)
    if (n > 0) & (N > 0):
        manhattan_distances = np.zeros(n)
        fitted_spectral_contrast = np.zeros(n)

        x_sums = np.zeros(n)
        
        for j in range(n):
            u2_sum, v2_sum, uv_sum = 0.0, 0.0, 0.0
            for i, row in enumerate(row_idx_split[j]):
                # Sum observed intensities for normalization
                x_sums[j] += val_obs[row]
                # Calculate Manhattan distance using predicted values
                manhattan_distances[j] += abs(y_pred[row] - val_obs[row])
                u2_sum += y_pred[row]**2 
                v2_sum += val_obs[row]**2
                uv_sum += y_pred[row] * val_obs[row]
            # Normalize and transform

            if x_sums[j] > 0 and manhattan_distances[j] > 0:
                manhattan_distances[j] = -np.log2(manhattan_distances[j] / x_sums[j])
                fitted_spectral_contrast[j] = np.sqrt(uv_sum)/(np.sqrt(u2_sum) * np.sqrt(v2_sum) + 1e-10)
            else:
                # Handle edge cases
                if x_sums[j] == 0:
                    manhattan_distances[j] = np.finfo(np.float32).max  # Bad fit
                    fitted_spectral_contrast[j] = 0.0
                else:  # manhattan_distances[j] == 0
                    manhattan_distances[j] = np.finfo(np.float32).min  # Perfect fit
                    fitted_spectral_contrast[j] = np.sqrt(uv_sum)/(np.sqrt(u2_sum) * np.sqrt(v2_sum) + 1e-10)
                
        return manhattan_distances, fitted_spectral_contrast
    else:
        return np.zeros(0), np.zeros(0)

def hyperscore2(frag_intensities, frag_codes):
    codes = np.asarray(frag_codes)
    not_iso = ~is_isotope(codes)
    num_b = int(np.sum(is_b_ion(codes) & not_iso))
    num_y = int(np.sum(is_y_ion(codes) & not_iso))
    dp = np.sum(frag_intensities[not_iso])
    return max(0, np.log(dp * math.factorial(num_b) * math.factorial(num_y))), num_b, num_y
    
@njit(nogil=True, error_model="numpy")
def _get_features_jit(rt_mz, prec_im,
                      ref_rows, ref_vals, ref_cols, ref_offsets,
                      dec_rows, dec_vals, dec_cols, dec_offsets,
                      dia_spectrum, prec_rt, dia_spec_int, coeffs,
                      sparse_row_indices, sparse_col_indices, sparse_values,
                      matched_lib_int, matched_codes, matched_offsets,
                      ms1_error, ref_spec_offset, decoy_spec_offset, unique_lookup):
    """get_features in one GIL-free call.  The same steps as the numpy code it
    replaced; only the sums round differently (numba sums sequentially, numpy
    pairwise).  error_model="numpy": a zero TIC gives inf/nan, not an exception."""
    val_obs = dia_spectrum[:, 1]
    tic = np.sum(val_obs)

    # Combined flat arrays (ref + decoy) for the large-coeff block
    n_ref = len(ref_offsets) - 1
    n_dec = len(dec_offsets) - 1
    if n_dec > 0:
        all_rows = np.concatenate((ref_rows, dec_rows))
        all_vals = np.concatenate((ref_vals, dec_vals))
        all_offsets = np.empty(n_ref + n_dec + 1, dtype=ref_offsets.dtype)
        all_offsets[:n_ref + 1] = ref_offsets
        for k in range(n_dec):
            all_offsets[n_ref + 1 + k] = dec_offsets[k + 1] + ref_offsets[n_ref]
    else:
        all_rows = ref_rows
        all_vals = ref_vals
        all_offsets = ref_offsets.copy()
    n_total = n_ref + n_dec
    max_col = max(ref_spec_offset + n_ref, decoy_spec_offset + n_dec) if n_total > 0 else 0
    col_to_pos = np.full(max_col, -1, dtype=np.int32)
    for k in range(n_ref):
        col_to_pos[ref_spec_offset + k] = k
    for k in range(n_dec):
        col_to_pos[decoy_spec_offset + k] = n_ref + k

    y_pred = np.zeros(len(val_obs))
    if len(ref_rows) > 0:
        _build_y_pred_jit(ref_rows, ref_cols, ref_vals, ref_offsets, coeffs, ref_spec_offset, y_pred)
    if len(dec_rows) > 0:
        _build_y_pred_jit(dec_rows, dec_cols, dec_vals, dec_offsets, coeffs, decoy_spec_offset, y_pred)
    residuals = val_obs - y_pred

    n = n_ref
    scribe_scores = np.zeros(n)
    gof_stats = np.zeros(n)
    max_unmatched_residuals = np.zeros(n)
    max_matched_residuals = np.zeros(n)
    manhattan_distances = np.zeros(n)
    fitted_spectral_contrasts = np.zeros(n)
    num_lib_peaks_matched = np.zeros(n)
    frac_lib_intensity = np.zeros(n)
    frac_dia_intensity = np.zeros(n)
    if len(ref_rows) > 0:
        _compute_candidate_features_jit(
            ref_rows, ref_vals, ref_cols, ref_offsets,
            val_obs, residuals, y_pred, coeffs, ref_spec_offset,
            scribe_scores, gof_stats, max_unmatched_residuals, max_matched_residuals,
            manhattan_distances, fitted_spectral_contrasts,
            num_lib_peaks_matched, frac_lib_intensity, frac_dia_intensity,
            tic)

    m = rt_mz.shape[0]
    rel_error = np.empty(len(ms1_error))
    for k in range(len(ms1_error)):
        rel_error[k] = abs(ms1_error[k]) if not np.isnan(ms1_error[k]) else -1.0
    rt_error = np.empty(m)
    im_error = np.empty(m)
    for k in range(m):
        rt_error[k] = prec_rt - rt_mz[k, 0]
        im_error[k] = prec_im[k] - rt_mz[k, 2]
    sum_dia_spec_int = np.sum(dia_spec_int)
    frac_int_matched = sum_dia_spec_int / tic

    n_coo_rows = sparse_row_indices.max() + 1 if len(sparse_row_indices) > 0 else 0
    predicted_spec_full, lc_frac, lc_cosine = _large_coeff_block_jit(
        sparse_row_indices, sparse_col_indices, sparse_values, n_coo_rows,
        coeffs, val_obs, all_rows, all_vals, all_offsets, col_to_pos)
    predicted_spec = predicted_spec_full[:-1]   # drop penalty row (unmatched-fragment intensity)

    r2all = np.zeros(m)
    r2_lib_spec = np.zeros(m)
    r2_unique = np.zeros(m)
    frac_unique_pred = np.zeros(n)
    if len(ref_rows) > 0:
        _compute_unique_frac_jit(ref_rows, ref_vals, ref_offsets,
                                 unique_lookup, val_obs, coeffs, ref_spec_offset,
                                 frac_unique_pred)
    frac_dia_intensity_pred = np.empty(n)
    for k in range(n):
        d = frac_dia_intensity[k] if frac_dia_intensity[k] > 0 else 1.0
        frac_dia_intensity_pred[k] = (frac_lib_intensity[k] * coeffs[ref_spec_offset + k]) / d

    sum_predicted = np.sum(predicted_spec)
    frac_int_pred = sum_predicted / tic if tic > 0 else 0.0
    frac_int_matched_pred = sum_predicted / sum_dia_spec_int if sum_dia_spec_int > 0 else 0.0

    hyperscores = np.zeros(n)
    b_counts = np.zeros(n)
    y_counts = np.zeros(n)
    longest_y_ions = np.zeros(n)
    if n > 0:
        _compute_hyperscores_jit(matched_lib_int, matched_codes, matched_offsets.astype(np.int32),
                                 hyperscores, b_counts, y_counts, longest_y_ions)

    return _assemble_features_jit(
        num_lib_peaks_matched, frac_lib_intensity, frac_dia_intensity,
        rel_error, rt_error, frac_int_matched, frac_int_pred,
        r2all, r2_lib_spec, r2_unique, frac_unique_pred, frac_dia_intensity_pred,
        hyperscores, b_counts, y_counts, longest_y_ions,
        scribe_scores, max_unmatched_residuals, max_matched_residuals,
        gof_stats, manhattan_distances, fitted_spectral_contrasts,
        frac_int_matched_pred, lc_frac, lc_cosine,
        rt_mz[:, 1].copy(), tic, im_error)


#@profile
def get_features(
    rt_mz,
    prec_im,
    ref_rows, ref_vals, ref_cols, ref_offsets,
    dec_rows, dec_vals, dec_cols, dec_offsets,
    dia_spectrum,
    prec_rt,
    dia_spec_int,
    lib_coefficients,
    sparse_row_indices,
    sparse_col_indices,
    sparse_values,
    matched_lib_int,
    matched_codes,
    matched_offsets,
    ms1_error,
    ref_spec_offset,
    decoy_spec_offset,
    unique_lookup_dia=None):

    """The 28 per-candidate features, as _assemble_features_jit lays them out.

    All of it runs in _get_features_jit, so a candidate group crosses the GIL
    once.  unique_lookup_dia is computed here only if the caller has none.
    """
    if unique_lookup_dia is None:
        _n_sm = int(sparse_row_indices.max()) + 1 if len(sparse_row_indices) > 0 else 0
        unique_lookup_dia = _single_match_lookup_jit(sparse_row_indices, sparse_col_indices, _n_sm)
    return _get_features_jit(
        np.ascontiguousarray(rt_mz, dtype=np.float64), np.asarray(prec_im, dtype=np.float64).ravel(),
        ref_rows, ref_vals, ref_cols, ref_offsets,
        dec_rows, dec_vals, dec_cols, dec_offsets,
        dia_spectrum, float(prec_rt), dia_spec_int,
        np.asarray(lib_coefficients, dtype=np.float64).ravel(),
        sparse_row_indices, sparse_col_indices, sparse_values,
        matched_lib_int, matched_codes, matched_offsets,
        np.asarray(ms1_error, dtype=np.float64),
        int(ref_spec_offset), int(decoy_spec_offset), unique_lookup_dia)

def _batch_closest_peak_diff(query_mzs, ref_mzs, max_diff):
    """Vectorized closest_peak_diff for an array of query m/z values.

    For each query m/z, finds the nearest peak in ref_mzs (sorted) and returns
    the relative difference (ref - query) / query. Returns NaN for queries
    where no ref peak is within max_diff relative tolerance.

    Replaces N individual closest_peak_diff() calls with one searchsorted.
    """
    query_mzs = np.asarray(query_mzs, dtype=np.float64)
    if len(query_mzs) == 0:
        return np.array([], dtype=np.float64)
    n_ref = len(ref_mzs)

    # searchsorted gives insertion point; nearest peak is either left or right neighbor
    idxs = np.searchsorted(ref_mzs, query_mzs)
    left_idx = np.clip(idxs - 1, 0, n_ref - 1)
    right_idx = np.clip(idxs, 0, n_ref - 1)

    left_diff = ref_mzs[left_idx] - query_mzs
    right_diff = ref_mzs[right_idx] - query_mzs

    # Pick whichever neighbor is closer
    closest_diff = np.where(np.abs(left_diff) <= np.abs(right_diff),
                            left_diff, right_diff) / query_mzs

    # NaN for anything outside tolerance
    closest_diff[np.abs(closest_diff) > max_diff] = np.nan
    return closest_diff


# TODO: remove dead code — `create_entries` (and its `_create_entries_jit`) is
# superseded by `create_entries_direct` and has no callers.
def create_entries(centroid_breaks,
                   candidate_peaks,
                   mass_window_candidates,
                   top_n,atleast_m,
                   prec_mzs,
                   ms1_spec,
                   ms1_tol,
                   spec_frags=None,
                   top_n_idxs=None
                   ):

    n_cands = len(candidate_peaks)
    if n_cands == 0:
        return ([], [], [], [],
                np.empty(0, np.int32), np.empty(0, np.int32), np.empty(0, np.float64),
                np.zeros(1, np.int32), [], [],
                np.array([], dtype=np.float64),
                np.empty(0, np.int64), np.empty(0, np.float64),
                np.zeros(1, np.int32), np.empty(0, np.int32))

    # Flatten candidate spectra into contiguous arrays for JIT
    stacked = np.concatenate(candidate_peaks)
    all_frag_mz = np.ascontiguousarray(stacked[:, 0])
    all_frag_int = np.ascontiguousarray(stacked[:, 1])
    frag_lengths = np.array([M.shape[0] for M in candidate_peaks], dtype=np.int32)
    frag_offsets = np.empty(n_cands + 1, dtype=np.int32)
    frag_offsets[0] = 0
    np.cumsum(frag_lengths, out=frag_offsets[1:])

    # Flatten top-N indices (local positions within each candidate)
    all_top_n_local = np.concatenate(top_n_idxs).astype(np.int32)
    top_n_lengths = np.array([len(idxs) for idxs in top_n_idxs], dtype=np.int32)
    top_n_offsets = np.empty(n_cands + 1, dtype=np.int32)
    top_n_offsets[0] = 0
    np.cumsum(top_n_lengths, out=top_n_offsets[1:])

    # JIT core: searchsorted + filtering + flat array construction (nogil=True)
    passing, flat_rows, flat_cols, flat_vals, flat_offsets, ms1_error_out, all_coords, all_norm_int = \
        _create_entries_core_jit(
            np.ascontiguousarray(centroid_breaks, dtype=np.float64),
            all_frag_mz, all_frag_int, frag_offsets,
            all_top_n_local, top_n_offsets,
            np.ascontiguousarray(prec_mzs, dtype=np.float64),
            np.ascontiguousarray(ms1_spec.mz, dtype=np.float64),
            float(ms1_tol),
            float(config.args.lib_frac), int(atleast_m),
            not config.args.no_ms1_req)

    # Reconstruct Python lists from JIT output
    peaks_in_dia = passing.tolist()
    pep_cand_loc = [all_coords[frag_offsets[i]:frag_offsets[i + 1]] for i in peaks_in_dia]
    pep_cand_list = [candidate_peaks[i] for i in peaks_in_dia]
    pep_cand = [mass_window_candidates[i] for i in peaks_in_dia]
    norm_intensities = [all_norm_int[frag_offsets[i]:frag_offsets[i + 1]] for i in peaks_in_dia]
    lib_peaks_matched = [pep_cand_loc[j] % 2 == 1 for j in range(len(peaks_in_dia))]

    return (peaks_in_dia,
            pep_cand,
            pep_cand_loc,
            pep_cand_list,
            flat_rows, flat_cols, flat_vals, flat_offsets, norm_intensities, lib_peaks_matched, ms1_error_out,
            all_coords, all_norm_int, frag_offsets, passing)


#@profile
class SpectrumHits(NamedTuple):
    """fit_to_lib2's output for one spectrum, as columns: one row per library
    entry with a non-zero coefficient, targets then decoys.

    The fragment columns are flat, row k's segment delimited by
    frag_offsets[k:k+2] (unique_offsets for the unique columns).  Scalars are
    the spectrum's, shared by every row.  hits_to_arrow turns a batch into
    the main-search parquet table.
    """
    coeff: np.ndarray
    spec_id: int
    ms1_spec_id: int
    seq: list
    z: np.ndarray
    time_channel: Optional[np.ndarray]
    window_mz: float
    rt: float
    prec_im: np.ndarray
    features: np.ndarray
    frag_codes: np.ndarray
    frag_errors: np.ndarray
    frag_mz: np.ndarray
    frag_int: np.ndarray
    obs_int: np.ndarray
    frag_offsets: np.ndarray
    unique_frag_mz: np.ndarray
    unique_obs_int: np.ndarray
    unique_offsets: np.ndarray
    file_name: str
    protein: list
    is_decoy: np.ndarray


def _concat_offsets(offsets_list):
    """One offset array for segments laid end to end."""
    out = [offsets_list[0]]
    for o in offsets_list[1:]:
        out.append(o[1:] + out[-1][-1])
    return np.concatenate(out)


def hits_to_arrow(hits, pa_schema):
    """A batch of SpectrumHits as one Arrow table in *pa_schema* (the
    main-search parquet schema, with or without time_channel)."""
    n = np.array([len(h.coeff) for h in hits])

    def per_spectrum(field, dtype):
        return np.repeat(np.array([getattr(h, field) for h in hits], dtype=dtype), n)

    def stacked(field, dtype):
        return np.concatenate([getattr(h, field) for h in hits]).astype(dtype, copy=False)

    def list_column(field, offsets_field):
        return pa.LargeListArray.from_arrays(
            _concat_offsets([getattr(h, offsets_field) for h in hits]),
            np.concatenate([getattr(h, field) for h in hits]))

    features = np.concatenate([h.features for h in hits]).astype(np.float32)
    columns = [stacked("coeff", np.float32),
               per_spectrum("spec_id", np.int32),
               per_spectrum("ms1_spec_id", np.int32),
               pa.array([s for h in hits for s in h.seq], type=pa.large_string()),
               stacked("z", np.float32)]
    if hits[0].time_channel is not None:
        columns.append(stacked("time_channel", np.float32))
    columns += [per_spectrum("window_mz", np.float64).astype(np.float32),
                per_spectrum("rt", np.float64).astype(np.float32),
                stacked("prec_im", np.float32)]
    columns += [np.ascontiguousarray(features[:, j]) for j in range(features.shape[1])]
    columns += [list_column("frag_codes", "frag_offsets"),
                list_column("frag_errors", "frag_offsets"),
                list_column("frag_mz", "frag_offsets"),
                list_column("frag_int", "frag_offsets"),
                list_column("obs_int", "frag_offsets"),
                list_column("unique_frag_mz", "unique_offsets"),
                list_column("unique_obs_int", "unique_offsets"),
                pa.array([h.file_name for h, k in zip(hits, n) for _ in range(k)], type=pa.large_string()),
                pa.array([p for h in hits for p in h.protein], type=pa.large_string()),
                np.concatenate([h.is_decoy for h in hits])]
    if len(columns) != len(pa_schema):
        raise ValueError(f"{len(columns)} columns for a {len(pa_schema)}-column schema")
    return pa.Table.from_arrays(
        [c if isinstance(c, pa.Array) else pa.array(c, type=f.type) for c, f in zip(columns, pa_schema)],
        schema=pa_schema)


def protein_column(library):
    """The library's config.protein_column, or None when it is not filled in
    (judged by the first entry): fit_to_lib2 then writes "NA"."""
    first = library.key_to_idx[next(iter(library.key_to_idx))]
    column = {'protein_group': library.protein_group, 'protein_name': library.protein_name,
              'genes': library.genes, 'UniprotID': library.uniprot_id}.get(config.protein_column)
    return column if column is not None and column[first] not in (None, '') else None


class _StepClock:
    """Per-step wall time and output bytes for one fit_to_lib2 call.

    ``lap(step, *arrays)`` charges the time since the previous lap to *step*
    and adds the arrays' sizes to its bytes; repeated steps (targets, then
    decoys) accumulate.  Everything a step produces stays alive until the fit
    returns, so the sum of the steps' bytes is the fit's working set (plus
    the solver's internal copies, estimated separately).  A no-op when
    *stats* is None.
    """
    __slots__ = ("stats", "t")

    def __init__(self, stats):
        self.stats = stats
        self.t = time.perf_counter()

    def lap(self, step, *arrays):
        if self.stats is None:
            return
        now = time.perf_counter()
        st = self.stats
        st[step + "_s"] = st.get(step + "_s", 0.0) + (now - self.t)
        self.t = now
        nbytes = 0
        for a in arrays:
            if isinstance(a, np.ndarray):
                nbytes += a.nbytes
            elif isinstance(a, (list, tuple)):
                nbytes += sum(x.nbytes for x in a if isinstance(x, np.ndarray))
        st[step + "_bytes"] = st.get(step + "_bytes", 0) + nbytes


def fit_to_lib2(dia_spec,
                library,
                rt_mz,
                all_keys,
                rt_tol,
                ms1_tol,
                mz_tol,
                im_tol,
                im_accuracy,
                file_name,
                dino_features=None,rt_filter=False,ms1_mz=None,
               ms1_spectra = None,
               decoy=False,
               output_folder=None,
               frag_index=None,
               ms1_rt=None,
               im_bin_ms1=None,
               proteins=None,
               stats=None):
    # spec_idx,dia_spec,library = inputs
    # stats: optional dict; when given, filled with this fit's sizes and the
    # time and output bytes of each step (see _StepClock and main_search).
    clock = _StepClock(stats)
    # im_tol: fragment IM tolerance (the run's fitted precision).
    # im_accuracy: widening of the IM candidate gate (the run's fitted library-IM
    # accuracy), or None on data without ion mobility (and timeplex), where
    # there is no gate.
    # file_name: the run's data file, written to each result row's file_name.
    
    spec_idx=dia_spec.scan_num

    top_n=config.top_n
    atleast_m=config.args.atleast_m
    spec = dia_spec
    dia_spectrum = np.stack(spec.peak_list(),1)
    if stats is not None:
        stats["raw_peaks"] = dia_spectrum.shape[0]
    prec_mz = spec.prec_mz
    prec_rt = spec.RT
    windowWidth = window_width(dia_spec)
    if ms1_spectra is not None:
        if im_bin_ms1 is not None and getattr(spec, 'im_lo', None) is not None:
            im_key = (spec.im_lo, spec.im_hi)
            if im_key in im_bin_ms1:
                _rt_arr, _idx_arr = im_bin_ms1[im_key]
                _pos = np.searchsorted(_rt_arr, prec_rt)
                if _pos == 0:
                    ms1_spec = ms1_spectra[_idx_arr[0]]
                elif _pos == len(_rt_arr):
                    ms1_spec = ms1_spectra[_idx_arr[-1]]
                else:
                    _before, _after = _rt_arr[_pos - 1], _rt_arr[_pos]
                    ms1_spec = ms1_spectra[_idx_arr[_pos - 1] if abs(prec_rt - _before) < abs(prec_rt - _after) else _idx_arr[_pos]]
            else:
                ms1_spec = get_closest_ms1(prec_rt, ms1_spectra, ms1_rt=ms1_rt)
        else:
            ms1_spec = get_closest_ms1(prec_rt, ms1_spectra, ms1_rt=ms1_rt)
    lib_coefficients = []

    # Per-peak ion mobility (timsTOF); zeros for non-IM data (merged_mob unused).
    _has_im = getattr(spec, "mobility", None) is not None
    _im_tol = im_tol if _has_im else 0.0
    if _has_im and _im_tol > 0.0:
        # timsTOF: bin DIA peaks by (m/z, IM) so same-m/z peaks at different
        # mobility stay separate; the summed bin intensity is the NNLS observation.
        _dia_mob = np.ascontiguousarray(spec.mobility, dtype=np.float64)
        _bin_mz, _bin_int, _bin_mob = _dia_prep_2d_jit(
            dia_spectrum[:, 0].copy(), dia_spectrum[:, 1].copy(), _dia_mob,
            mz_tol, _im_tol)
        dia_spectrum = np.stack([_bin_mz, _bin_int], axis=1)
        # centroid_breaks unused on the IM path (2D matcher uses bin_mz directly)
        centroid_breaks = np.zeros(0, dtype=np.float64)
        bin_centers = _bin_mz
    else:
        _dia_mob = np.zeros(dia_spectrum.shape[0], dtype=np.float64)
        merged_mz, merged_int, centroid_breaks, bin_centers, _merged_mob = _dia_prep_jit(
            dia_spectrum[:, 0].copy(), dia_spectrum[:, 1].copy(), _dia_mob, mz_tol)
        dia_spectrum = np.stack([merged_mz, merged_int], axis=1)
        _bin_mz = _bin_int = _bin_mob = None

    if stats is not None:
        stats["binned_peaks"] = dia_spectrum.shape[0]
    clock.lap("bin_peaks", dia_spectrum, _dia_mob, _bin_mz, _bin_int, _bin_mob)

    # Get candidates via fragment index or fallback to m/z + RT window
    # Single query returns both target and decoy candidates from unified index
    # IM admission window: a library precursor is only a candidate for this
    # spectrum if its aligned IM is compatible with the spectrum's mobility band,
    # widened by the library-IM accuracy.  Inert (-inf, +inf) whenever the
    # spectrum has no band, the library has no IM, or no alignment was fitted --
    # in which case rt_mz has no IM column and every value is NaN anyway.
    _im_gate_lo, _im_gate_hi = -np.inf, np.inf
    if (rt_mz.shape[1] > 2 and getattr(spec, "im_lo", None) is not None
            and im_accuracy is not None):
        _im_gate_lo = spec.im_lo - im_accuracy
        _im_gate_hi = spec.im_hi + im_accuracy

    if frag_index is not None and not ms1_mz:
        win_lo = prec_mz - windowWidth / 2
        win_hi = prec_mz + windowWidth / 2
        all_window_idxs = frag_index.query(
            dia_spectrum[:, 0], win_lo, win_hi,
            prec_rt, rt_tol, atleast_m,
            im_gate_lo=_im_gate_lo, im_gate_hi=_im_gate_hi
        )
    else:
        if ms1_mz:
            _bool = (np.abs(rt_mz[:,1]-ms1_mz)/ms1_mz)<ms1_tol
        else:
            if rt_filter:
                _bool = np.logical_and(np.abs(rt_mz[:,1]-prec_mz)<(windowWidth/2),np.abs(rt_mz[:,0]-prec_rt)<rt_tol)
            else:
                _bool = np.abs(rt_mz[:,1]-prec_mz)<(windowWidth/2)
        if np.isfinite(_im_gate_lo):
            # Mirror the index gate. NaN IM (library without mobility) is not a
            # rejection, so it is explicitly admitted rather than compared.
            _im = rt_mz[:, 2]
            _bool = np.logical_and(
                _bool,
                np.isnan(_im) | ((_im >= _im_gate_lo) & (_im <= _im_gate_hi)),
            )
        all_window_idxs = np.where(_bool)[0]

    if stats is not None:
        stats["candidates"] = len(all_window_idxs)
    clock.lap("query", all_window_idxs)

    # Split into target and decoy candidates
    # TODO: unify target/decoy processing paths to eliminate this split
    n_targets = library.n_targets
    target_mask = all_window_idxs < n_targets
    window_idxs = all_window_idxs[target_mask]
    decoy_window_idxs = all_window_idxs[~target_mask] if decoy else np.empty(0, dtype=all_window_idxs.dtype)

    # Entry arrays every candidate group reads
    _lib_arrays = dict(spec_data_mz=library.spectrum_mz,
                       spec_data_int=library.spectrum_int,
                       spec_offsets=library.spectrum_offsets,
                       spec_lengths=library.spectrum_lengths,
                       topn_data=library.top_n_data,
                       topn_offsets=library.top_n_offsets,
                       topn_lengths=library.top_n_lengths)
    _match_args = dict(centroid_breaks=centroid_breaks, atleast_m=atleast_m,
                       ms1_spec=ms1_spec, ms1_tol=ms1_tol,
                       bin_mz=_bin_mz, bin_int=_bin_int, bin_mob=_bin_mob,
                       mz_tol=mz_tol, im_tol=_im_tol, has_im=_has_im,
                       frag_names_data=library.frag_names_data,
                       dia_int=dia_spectrum[:, 1])

    # window_idxs are library indices: the fragment index returns positions in
    # all_keys, which is list(library)
    ref_passing, ref_flat_rows, ref_flat_cols, ref_flat_vals, ref_flat_offsets, ref_ms1_error, \
        ref_all_coords, ref_all_norm_int, ref_frag_offsets, ref_prec_im, ref_cand_idx, \
        ref_ent_angle, ref_topn, ref_frac = create_entries_direct(
            candidate_indices=window_idxs, prec_mzs=rt_mz[:,1][window_idxs],
            **_lib_arrays, **_match_args)
    clock.lap("match_targets", ref_passing, ref_flat_rows, ref_flat_cols, ref_flat_vals,
              ref_flat_offsets, ref_ms1_error, ref_all_coords, ref_all_norm_int,
              ref_frag_offsets, ref_prec_im, ref_cand_idx)

    ### Decoys, from the same index query (their m/z offset is already in rt_mz)
    if decoy:
        decoy_mz = rt_mz[:,1][decoy_window_idxs]
        decoy_passing, decoy_flat_rows, decoy_flat_cols, decoy_flat_vals, decoy_flat_offsets, decoy_ms1_error, \
            dec_all_coords, dec_all_norm_int, dec_frag_offsets, dec_prec_im, dec_cand_idx, \
            dec_ent_angle, dec_topn, dec_frac = create_entries_direct(
                candidate_indices=decoy_window_idxs, prec_mzs=decoy_mz,
                **_lib_arrays, **_match_args)
        clock.lap("match_decoys", decoy_passing, decoy_flat_rows, decoy_flat_cols,
                  decoy_flat_vals, decoy_flat_offsets, decoy_ms1_error, dec_all_coords,
                  dec_all_norm_int, dec_frag_offsets, dec_prec_im, dec_cand_idx)

    # Per-spectrum cap on candidates entering the fit (--max_candidates).
    # Targets and decoys are ranked together, by matched top-N ions then the
    # matched fraction of library intensity, with the library index breaking ties
    # so the choice is deterministic.  Same rule for both, so it is symmetric.
    _cap = getattr(config.args, "max_candidates", None)
    _n_dec = len(decoy_passing) if decoy else 0
    if _cap is not None and len(ref_passing) + _n_dec > _cap:
        _topn = np.concatenate([ref_topn, dec_topn]) if decoy else ref_topn
        _frac = np.concatenate([ref_frac, dec_frac]) if decoy else ref_frac
        _lib = (np.concatenate([window_idxs[ref_passing], decoy_window_idxs[decoy_passing]])
                if decoy else window_idxs[ref_passing]).astype(np.int64)
        _order = np.lexsort((_lib, -_frac, -_topn.astype(np.int64)))
        _keep = np.zeros(len(_order), dtype=bool)
        _keep[_order[:_cap]] = True
        _kr, _kd = _keep[:len(ref_passing)], _keep[len(ref_passing):]
        (ref_passing, ref_flat_rows, ref_flat_cols, ref_flat_vals, ref_flat_offsets,
         ref_ms1_error, ref_prec_im, ref_ent_angle) = _keep_passing(
            _kr, ref_passing, ref_flat_rows, ref_flat_cols, ref_flat_vals, ref_flat_offsets,
            ref_ms1_error, ref_prec_im, ref_ent_angle)
        if decoy:
            (decoy_passing, decoy_flat_rows, decoy_flat_cols, decoy_flat_vals, decoy_flat_offsets,
             decoy_ms1_error, dec_prec_im, dec_ent_angle) = _keep_passing(
                _kd, decoy_passing, decoy_flat_rows, decoy_flat_cols, decoy_flat_vals,
                decoy_flat_offsets, decoy_ms1_error, dec_prec_im, dec_ent_angle)

    decoy_col_offset = int(np.max(ref_flat_cols)) + 1 if len(ref_flat_rows) > 0 else 0

    if not (len(ref_flat_rows) > 0 or (decoy and len(decoy_flat_rows) > 0)):
        return None

    # Single JIT call replaces ~14 GIL-acquiring numpy/scipy calls
    # (np.append ×6, np.concatenate ×3, np.unique, np.sort, rankdata, unmatched_peaks ×2)
    _has_dec = decoy and len(decoy_flat_rows) > 0
    _dec_rows = decoy_flat_rows if _has_dec else np.empty(0, np.int32)
    _dec_cols = decoy_flat_cols if _has_dec else np.empty(0, np.int32)
    _dec_vals = decoy_flat_vals if _has_dec else np.empty(0, np.float64)
    _dec_coords = dec_all_coords if _has_dec else np.empty(0, np.int64)
    _dec_norm_int = dec_all_norm_int if _has_dec else np.empty(0, np.float64)
    _dec_frag_off = dec_frag_offsets if _has_dec else np.zeros(1, np.int32)
    _dec_passing = decoy_passing if _has_dec else np.empty(0, np.int32)

    sparse_row_indices, sparse_col_indices, sparse_values, dia_spec_int, peak_idx_lookup = \
        _assemble_coo_jit(
            ref_flat_rows, ref_flat_cols, ref_flat_vals,
            _dec_rows, _dec_cols, _dec_vals,
            ref_all_coords, ref_all_norm_int, ref_frag_offsets, ref_passing,
            _dec_coords, _dec_norm_int, _dec_frag_off, _dec_passing,
            decoy_col_offset,
            dia_spectrum[:, 1],
            1e-10)

    # Fit lib spectra to observed spectra (Huber loss via IRLS)
    # Pass flat COO arrays directly — avoids constructing scipy sparse matrix
    _n_coo_rows = len(dia_spec_int)
    _n_coo_cols = int(sparse_col_indices.max()) + 1 if len(sparse_col_indices) > 0 else 0
    if stats is not None:
        stats["rows"] = _n_coo_rows
        stats["columns"] = _n_coo_cols
        stats["nonzeros"] = len(sparse_values)
    clock.lap("assemble", sparse_row_indices, sparse_col_indices, sparse_values,
              dia_spec_int, peak_idx_lookup)
    fit_results = huber_nnls_irls(sparse_values, sparse_row_indices, sparse_col_indices,
                                  _n_coo_rows, _n_coo_cols, dia_spec_int)
    lib_coefficients = fit_results['x']
    clock.lap("solve", lib_coefficients, fit_results.get('weights'))
    if stats is not None:
        # Freed before huber_nnls_irls returns, so not in the lap above:
        # _coo_gram_and_Aty's scratch (six per-entry and six per-column
        # arrays), and the CSC matrix, its reweighted copy and enet_path's
        # vectors.
        stats["solve_gram_bytes"] = 48 * len(sparse_values) + 49 * _n_coo_cols
        stats["solve_internal_bytes"] = (stats["solve_gram_bytes"]
                                         + len(sparse_values) * (8 + 4 + 8)
                                         + (_n_coo_rows + _n_coo_cols) * 8 * 4)

    # Compute single-matched rows via JIT (replaces sparse.coo_matrix + np.sum(matrix>0,1)==1)
    _sm_max = int(sparse_row_indices.max()) + 1 if len(sparse_row_indices) > 0 else 0
    single_match_lookup = _single_match_lookup_jit(sparse_row_indices, sparse_col_indices, _sm_max)

    # Map re-indexed single-match status back to original DIA peak indices
    n_dia = dia_spectrum.shape[0]
    unique_lookup_dia = np.zeros(n_dia, dtype=bool)
    _orig_rows = np.where(peak_idx_lookup >= 0)[0]
    _orig_rows = _orig_rows[_orig_rows < n_dia]
    _reindexed = peak_idx_lookup[_orig_rows]
    _valid_ri = _reindexed < _sm_max
    _is_single = single_match_lookup[_reindexed[_valid_ri]]
    unique_lookup_dia[_orig_rows[_valid_ri][_is_single]] = True

    # Matched-fragment output columns, flat per candidate group
    _dia_int = np.ascontiguousarray(dia_spectrum[:, 1])
    def _frag_columns(passing, cand_idx, coords, frag_offsets, flat_rows, flat_offsets):
        return _frag_columns_jit(passing, cand_idx, library.spectrum_offsets,
                                 library.spectrum_mz, library.spectrum_int, library.frag_names_data,
                                 coords, frag_offsets, flat_rows, flat_offsets,
                                 bin_centers, _dia_int, single_match_lookup, peak_idx_lookup)
    clock.lap("unique_peaks", single_match_lookup, unique_lookup_dia)
    ref_frags = _frag_columns(ref_passing, ref_cand_idx, ref_all_coords, ref_frag_offsets,
                              ref_flat_rows, ref_flat_offsets)
    clock.lap("fragment_columns", ref_frags)

    # Build decoy flat arrays for get_features (empty if no decoys)
    _dec_offsets = decoy_flat_offsets if (decoy and len(decoy_flat_offsets) > 1) else np.zeros(1, np.int32)
    features = get_features(rt_mz[window_idxs[ref_passing]],
                            ref_prec_im,
                            ref_flat_rows, ref_flat_vals, ref_flat_cols, ref_flat_offsets,
                            _dec_rows, _dec_vals, _dec_cols, _dec_offsets,
                            dia_spectrum,
                            prec_rt,
                            dia_spec_int,
                            lib_coefficients,
                            sparse_row_indices,
                            sparse_col_indices,
                            sparse_values,
                            ref_frags[3], ref_frags[0], ref_frags[5],
                            ref_ms1_error,
                            0,
                            decoy_col_offset,
                            unique_lookup_dia=unique_lookup_dia)
    # 29th column: entropy-normalized spectral angle (_create_entries_direct_jit)
    features = np.column_stack((features, ref_ent_angle))
    clock.lap("features", features)

    if decoy:
        decoy_frags = _frag_columns(decoy_passing, dec_cand_idx, dec_all_coords, dec_frag_offsets,
                                    decoy_flat_rows, decoy_flat_offsets)
        clock.lap("fragment_columns", decoy_frags)
        # Column 2 (aligned library IM) has to be carried through here too --
        # this array is rebuilt rather than sliced, so it would otherwise be
        # 2 columns wide and the IM error feature would fail on decoys.
        decoy_features = get_features(np.stack([rt_mz[decoy_window_idxs[decoy_passing],0],
                                                decoy_mz[decoy_passing],
                                                rt_mz[decoy_window_idxs[decoy_passing],2]],1),
                                      dec_prec_im,
                                      decoy_flat_rows, decoy_flat_vals, decoy_flat_cols, decoy_flat_offsets,
                                      ref_flat_rows, ref_flat_vals, ref_flat_cols, ref_flat_offsets,
                                      dia_spectrum,
                                      prec_rt,
                                      dia_spec_int,
                                      lib_coefficients,
                                      sparse_row_indices,
                                      sparse_col_indices,
                                      sparse_values,
                                      decoy_frags[3], decoy_frags[0], decoy_frags[5],
                                      decoy_ms1_error,
                                      decoy_col_offset,
                                      0,
                                      unique_lookup_dia=unique_lookup_dia)
        decoy_features = np.column_stack((decoy_features, dec_ent_angle))
        clock.lap("features", decoy_features)

    # Rows: the candidates with a non-zero coefficient, targets then decoys.
    # Columns of lib_coefficients are the target candidates then the decoys
    # from decoy_col_offset, so a non-zero column is also the candidate's row
    # in the stacked features and fragment columns.
    nz = np.flatnonzero(lib_coefficients != 0)
    if len(nz) == 0:
        return None
    n_ref = len(ref_passing)
    nz_target = nz[nz < n_ref]
    lib_idxs = window_idxs[ref_passing[nz_target]]
    prec_im_rows = ref_prec_im[nz_target]
    if decoy:
        nz_decoy = nz[nz >= decoy_col_offset] - decoy_col_offset
        lib_idxs = np.concatenate([lib_idxs, decoy_window_idxs[decoy_passing[nz_decoy]]])
        prec_im_rows = np.concatenate([prec_im_rows, dec_prec_im[nz_decoy]])
        all_features = np.concatenate((features, decoy_features))
        frags = [np.concatenate([t, d]) for t, d in zip(ref_frags, decoy_frags)]
        for o, base in ((5, ref_frags[5][-1]), (8, ref_frags[8][-1])):
            frags[o] = np.concatenate([ref_frags[o], decoy_frags[o][1:] + base])
    else:
        all_features = features
        frags = list(ref_frags)
    codes, errors, lib_mz, lib_int, obs_int, offsets, unique_mz, unique_obs, unique_offsets = frags
    codes, row_offsets = _take_segments_jit(codes, offsets, nz)
    errors, _ = _take_segments_jit(errors, offsets, nz)
    lib_mz, _ = _take_segments_jit(lib_mz, offsets, nz)
    lib_int, _ = _take_segments_jit(lib_int, offsets, nz)
    obs_int, _ = _take_segments_jit(obs_int, offsets, nz)
    unique_mz, unique_row_offsets = _take_segments_jit(unique_mz, unique_offsets, nz)
    unique_obs, _ = _take_segments_jit(unique_obs, unique_offsets, nz)

    keys = [all_keys[i] for i in lib_idxs]

    # Decoy entries have protein info copied from their parent target
    if proteins is not None:
        pool = proteins.pool
        protein = [pool[c] for c in proteins.codes[lib_idxs].tolist()]
    else:
        protein = ["NA"] * len(nz)

    hits = SpectrumHits(
        coeff=lib_coefficients[nz],
        spec_id=spec_idx,
        ms1_spec_id=ms1_spec.scan_num,
        seq=[k[0] for k in keys],
        z=np.array([k[1] for k in keys], dtype=np.float64),
        time_channel=(np.array([k[2] for k in keys], dtype=np.float64)
                      if config.args.timeplex else None),
        window_mz=prec_mz,
        rt=prec_rt,
        prec_im=prec_im_rows,
        features=all_features[nz],
        frag_codes=codes, frag_errors=errors, frag_mz=lib_mz, frag_int=lib_int, obs_int=obs_int,
        frag_offsets=row_offsets,
        unique_frag_mz=unique_mz, unique_obs_int=unique_obs, unique_offsets=unique_row_offsets,
        file_name=file_name,
        protein=protein,
        is_decoy=np.arange(len(nz)) >= len(nz_target),
    )
    clock.lap("output")  # its bytes are the results', measured by the caller
    return hits


# #@profile
def fit_to_lib(dia_spec,library,rt_mz,all_keys,
               rt_tol,
               ms1_tol,
               mz_tol,
               file_name,
               dino_features=None,
               rt_filter=False,
               ms1_mz=None,
               ms1_spectra = None,
               return_frags = False,
               frac_matched = 0.5):
    # Only the timeplex first search calls this.  ptinnls is not a JMod
    # dependency: it needs cvxopt, which has no free-threaded wheels
    import ptinnls as sparse_nnls
    # spec_idx,dia_spec,library = inputs
    
    spec_idx=dia_spec.scan_num
    
    # mz_tol = config.mz_tol
    # rt_tol = min(config.rt_tol,config.opt_rt_tol)
    # ms1_tol = min(config.ms1_tol,config.opt_ms1_tol)
    top_n=config.top_n
    atleast_m=config.args.atleast_m
    spec = dia_spec#spectra.ms2scans[spec_idx]
    dia_spectrum = np.stack(spec.peak_list(),1)
    prec_mz = spec.prec_mz
    prec_rt = spec.RT
    # spec_idx = spec.id
    
    windowWidth = window_width(dia_spec)
    
    
    if ms1_spectra is not None:
        ms1_rt = np.array([i.RT for i in ms1_spectra])
        closest_ms1_scan_idx = closest_ms1spec(prec_rt, ms1_rt)
        ms1_spec = ms1_spectra[closest_ms1_scan_idx]
    
    
    lib_coefficients = []
   
    if ms1_mz:
        _bool = (np.abs(rt_mz[:,1]-ms1_mz)/ms1_mz)<ms1_tol
        
    else:
        if rt_filter:
            _bool = np.logical_and(np.abs(rt_mz[:,1]-prec_mz)<(windowWidth/2),np.abs(rt_mz[:,0]-prec_rt)<rt_tol)
        else:
            _bool = np.abs(rt_mz[:,1]-prec_mz)<(windowWidth/2)
            
    window_idxs = np.where(_bool)[0]        
        
        
        
    ### match lib spec to features
    if dino_features is not None:
        filtered_dino = feature_list_mz(feature_list_rt(dino_features,prec_rt,rt_tol=rt_tol),
                                        prec_mz,windowWidth)
        window_edges = createTolWindows(filtered_dino.mz, tolerance=ms1_tol)
        window_idxs = window_idxs[np.where((np.searchsorted(window_edges,rt_mz[window_idxs,1])%2)==1)[0]]
        
    
    mass_window_candidates = [all_keys[i] for i in window_idxs] 
    candidate_peaks = [library[i]['spectrum'] for i in mass_window_candidates]
    
    # # filter possible lib entries for windows.. NB: DONT LIKE HOW I DO SAME LOOP TWICE
    # candidate_lib = [spectrum for key,spectrum in library.items() if spectrum["prec_mz"]>spec.ms1window[0] and spectrum["prec_mz"]<spec.ms1window[1]]
    # mass_window_candidates = [key for key,spectrum in library.items() if spectrum["prec_mz"]>spec.ms1window[0] and spectrum["prec_mz"]<spec.ms1window[1]]
    # # list of peaks from each candiate pep
    # # candidate_peaks = [SpecLib.frag_to_peak(i["frags"]) for i in candidate_lib]
    # candidate_peaks = [i["spectrum"] for i in candidate_lib]
    
    
    
    ###### Process dia spectrum

    # TODO: merge_spectrum_peaks already pre-merges all spectra — this may be redundant
    # what are the first indices of peaks grouped by tolerance
    merged_coords_idxs = np.searchsorted(dia_spectrum[:,0]+mz_tol*dia_spectrum[:,0],dia_spectrum[:,0])
    
    # what are the first mz of these peak groups
    merged_coords = dia_spectrum[np.unique(merged_coords_idxs),0]
    # print(merged_coords)
    
    
    # NB - should we not sum the intensities?????
    # merged_intensities = [np.mean(dia_spectrum[np.where(merged_coords_idxs==i)[0],1]) for i in np.unique(merged_coords_idxs)]
    #merged_intensities = np.zeros(len((merged_coords_idxs)))
    #for j,val in zip(merged_coords_idxs,dia_spectrum[:,1]):
    #    merged_intensities[j]+=val
    merged_intensities = np.bincount(merged_coords_idxs, weights=dia_spectrum[:, 1])
    merged_intensities = merged_intensities[merged_intensities != 0]

    #merged_intensities = [np.mean(dia_spectrum[merged_coords_idxs==i,1]) for i in np.unique(merged_coords_idxs)]
    #merged_intensities = merged_intensities[merged_intensities!=0]
    
    #update spectrum to new values (note mz remains first in group as this will eventually be rounded)
    dia_spectrum = np.array((merged_coords,merged_intensities)).transpose()
    # print(dia_spectrum)
    
    #get window edge positions each side of peaks in observed spectra (NB the tolerance is now about the first peak in the group not the middile)
    centroid_breaks = np.concatenate((dia_spectrum[:,0]-mz_tol*dia_spectrum[:,0],dia_spectrum[:,0]+mz_tol*dia_spectrum[:,0]))
    centroid_breaks = np.sort(centroid_breaks)
    bin_centers = np.mean(np.stack((centroid_breaks[::2],centroid_breaks[1::2]),1),1)
    
    # if "spec_frags" in library[all_keys[0]].keys():
    #     spec_peaks = [library[i]['spec_frags'] for i in mass_window_candidates]
    #     ref_coords = [np.searchsorted(centroid_breaks,M[:,0]) for M in candidate_peaks]
    #     spec_ref_coords = [np.searchsorted(centroid_breaks,M[:,0]) for M in spec_peaks]
    #     top_ten = [np.searchsorted(centroid_breaks,M[np.argsort(-M[:,1])[0:min(top_n,M.shape[0])],0]) for M in spec_peaks]
       
    #     ## Filter precursors based on resp. MS1 peak
    #     ms1_peak = ~np.isnan([closest_peak_diff(mz,ms1_spec.mz) for mz in rt_mz[window_idxs,1]])
        
    #     # does the top ten peaks fall between centroid breaks? i.e. odd numbers (%2==1), 
    #     ref_peaks_in_dia = [i for i in range(len(spec_peaks)) if len([a for a in top_ten[i] if a%2 ==1])>atleast_m]
        
    #     all_norm_intensities = [M[:,1]/sum(M[:,1]) for M in spec_peaks]
    #     # ref_peaks_in_dia = [i for i in range(len(candidate_peaks)) if np.sum(all_norm_intensities[i][(spec_ref_coords[i]%2)==1])>0.5 and np.sum(top_ten[i]%2)>atleast_m and ms1_peak[i]]
    #     ref_peaks_in_dia = [i for i in range(len(candidate_peaks)) if np.sum(top_ten[i]%2)>atleast_m]
    #     # ref_peaks_in_dia = [i for i in range(len(spec_peaks)) if np.sum(all_norm_intensities[i][(spec_ref_coords[i]%2)==1])>0.5 and np.sum(top_ten[i]%2)>atleast_m and ms1_peak[i] and top_ten[i][0]%2==1 and np.sum(top_ten[i][:3]%2==1)>=2]
    #     # ref_peaks_in_dia = [i for i in range(len(candidate_peaks)) if  np.sum(top_ten[i]%2)>atleast_m and ms1_peak[i]]
    
    
    # else:
    # lib_idx=0
    # M = candidate_peaks[lib_idx]
    ref_coords = [np.searchsorted(centroid_breaks,M[:,0]) for M in candidate_peaks]
    top_ten = [np.searchsorted(centroid_breaks,M[np.argsort(-M[:,1])[0:min(top_n,M.shape[0])],0]) for M in candidate_peaks]
    
    ## Filter precursors based on resp. MS1 peak
    ms1_peak = ~np.isnan([closest_peak_diff(mz,ms1_spec.mz) for mz in rt_mz[window_idxs,1]])
    
    # does the top ten peaks fall between centroid breaks? i.e. odd numbers (%2==1), 
    ref_peaks_in_dia = [i for i in range(len(candidate_peaks)) if len([a for a in top_ten[i] if a%2 ==1])>atleast_m]
    prop_ref_peaks_in_dia = [len([a for a in top_ten[i] if a%2 ==1])/candidate_peaks[i].shape[0] for i in range(len(candidate_peaks))]
    
    all_norm_intensities = [M[:,1]/sum(M[:,1]) for M in candidate_peaks]
    # ref_peaks_in_dia = [i for i in range(len(candidate_peaks)) if np.sum(all_norm_intensities[i][(ref_coords[i]%2)==1])>0.5 and np.sum(top_ten[i]%2)>atleast_m and ms1_peak[i]]
    ref_peaks_in_dia = [i for i in range(len(candidate_peaks)) if np.sum(all_norm_intensities[i][(ref_coords[i]%2)==1])>frac_matched and np.sum(top_ten[i]%2)>atleast_m and ms1_peak[i] and top_ten[i][0]%2==1 and np.sum(top_ten[i][:3]%2==1)>=2]
    # ref_peaks_in_dia = [i for i in range(len(candidate_peaks)) if  np.sum(top_ten[i]%2)>atleast_m and ms1_peak[i]]
    
    # print(len(ref_peaks_in_dia))
    
    # filter database further to those that match the required num peaks
    ref_pep_cand_loc = [ref_coords[i] for i in ref_peaks_in_dia]
    ref_pep_cand_list = [candidate_peaks[i] for i in ref_peaks_in_dia]
    # ref_pep_cand = [candidate_lib[i]["seq"] for i in ref_peaks_in_dia] # Nb this is modified seq!!
    ref_pep_cand = [mass_window_candidates[i] for i in ref_peaks_in_dia] # Nb this is modified seq!!
    
    norm_intensities = [M[:,1]/sum(M[:,1]) for M in ref_pep_cand_list]


    ########## Update
    # lib peaks that match
    lib_peaks_matched = [j%2==1 for j in ref_pep_cand_loc]
    
    # name these something different so can be accessed later
    ref_spec_row_indices_split = [np.int32(((i[j]+1)/2)-1) for i,j in zip(ref_pep_cand_loc,lib_peaks_matched)] # NB these are floats
    num_lib_peaks_matched = np.array([np.sum(i) for i in lib_peaks_matched]) #f1
    ref_spec_col_indices_split = [np.array([idx]*i) for idx,i in zip(range(len(ref_pep_cand)),num_lib_peaks_matched)] 
    ref_spec_values_split = [ints[i] for ints,i in zip(norm_intensities,lib_peaks_matched)]
    
    
    
    # ref_spec_row_indices = ((np.array([j for i in ref_pep_cand_loc for j in i if j%2==1])+1)/2)-1 # NB these are floats
    # ref_spec_col_indices = np.array([i for idx in range(len(ref_pep_cand)) for i in [idx]*len([loc for loc in ref_pep_cand_loc[idx] if loc%2==1])])
    # ref_spec_values = np.array([norm_intensities[idx][peak_idx] for idx in range(len(ref_pep_cand)) for peak_idx in range(len(norm_intensities[idx])) if ref_pep_cand_loc[idx][peak_idx]%2==1])
    
    frag_errors = []
    frag_mz = []
    
    
    if len(ref_spec_row_indices_split)>0 and len(ref_spec_col_indices_split)>0 and len(ref_spec_values_split)>0:
        
        #### concatenate the matrix values
        ref_spec_row_indices = np.concatenate(ref_spec_row_indices_split)
        ref_spec_col_indices = np.concatenate(ref_spec_col_indices_split)
        ref_spec_values = np.concatenate(ref_spec_values_split)
        
        frag_errors = [np.array(bin_centers[ref_spec_row_indices_split[i]]-ref_pep_cand_list[i][:,0][lib_peaks_matched[i]])/bin_centers[ref_spec_row_indices_split[i]] for i in range(len(lib_peaks_matched))]
        lib_frag_mz = [ref_pep_cand_list[i][:,0][lib_peaks_matched[i]] for i in range(len(lib_peaks_matched))]
        lib_frag_int = [ref_pep_cand_list[i][:,1][lib_peaks_matched[i]] for i in range(len(lib_peaks_matched))]
        obs_frag_int = [dia_spectrum[ref_spec_row_indices_split[i],1] for i in range(len(ref_spec_row_indices_split))]
        frag_names = [library[i]["ordered_frags"][j] for i,j in zip(ref_pep_cand,lib_peaks_matched)]
        
        frag_mz = [ref_pep_cand_list[i][:,0][lib_peaks_matched[i]] for i in range(len(lib_peaks_matched))]
        
        # what peaks from the spectrum are matched by library peps
        unique_row_idxs = [int(i) for i in set(ref_spec_row_indices)]
        unique_row_idxs.sort()
        
        dia_spec_int = dia_spectrum[unique_row_idxs,1]
        
        # add another term to penalise additional lib peaks
        dia_spec_int = np.append(dia_spec_int,[0]) 
        # find peaks that are bot matched in dia spectrum
        ref_peaks_not_in_dia = np.array([idx for loc_list in ref_pep_cand_loc for idx in range(len(loc_list)) if loc_list[idx]%2==0])
        # get col indices (will just be one for each)
        not_dia_col_indices = np.arange(len(ref_pep_cand))
        num_rows = max(unique_row_idxs)
        # row indices always the last row (num peaks+1)
        not_dia_row_indices = [num_rows+1]*len(not_dia_col_indices)
        # sum peak intensities not in dia spectrum
        not_dia_values = np.array([np.sum([norm_intensities[idx][peak_idx] for peak_idx in range(len(norm_intensities[idx])) if ref_pep_cand_loc[idx][peak_idx]%2==0])
                                  for idx in range(len(norm_intensities))])
    
        sparse_row_indices = np.append(ref_spec_row_indices,not_dia_row_indices)
        sparse_col_indices = np.append(ref_spec_col_indices,not_dia_col_indices)
        sparse_values = np.append(ref_spec_values,not_dia_values)
        
        # some dia peaks are not matched and are therefore ignored
        # below ranks the rows by number therefore removing missing rows
        sparse_row_indices = _rankdata_dense_jit(sparse_row_indices.astype(np.int64))

        # Generate sparse matrix from data
        sparse_lib_matrix = sparse.coo_matrix((sparse_values,(sparse_row_indices,sparse_col_indices)))
        # print("Starting Fit")
        # Fit lib spectra to observed spectra
        fit_results = sparse_nnls.lsqnonneg(sparse_lib_matrix,dia_spec_int,{"show_progress":False})
        lib_coefficients = fit_results['x']
        
        
        ####################################
        ### features 
        frac_lib_intensity = [np.sum(i) for i in ref_spec_values_split] # all ints sum to 1 so these give frac
        tic = np.sum(dia_spectrum[:,1])
        frac_dia_intensity = [np.sum(dia_spectrum[i,1])/tic for i in ref_spec_row_indices_split]
        # mz tol
        if dino_features is not None:
            rel_error_raw = ms1_error(np.array(filtered_dino.mz), rt_mz[window_idxs[ref_peaks_in_dia],1], tol=ms1_tol)
            rel_error = np.where(~np.isnan(rel_error_raw), np.abs(rel_error_raw), -1.0)
        else:
            rel_error = np.full(len(ref_peaks_in_dia), -1.0)
        rt_error = prec_rt-rt_mz[window_idxs[ref_peaks_in_dia],0]
        
        frac_int_matched = np.sum(dia_spec_int)/np.sum(dia_spectrum[:,1])
        predicted_spec = np.squeeze(sparse_lib_matrix*lib_coefficients)[:-1]
        # print(len(dia_spec_int),len(predicted_spec))
        r2all = np_pearson_cor(dia_spec_int[:-1],predicted_spec).statistic
        
        r2_lib_spec = [np_pearson_cor(i,dia_spectrum[j,1]).statistic for i,j in zip(ref_spec_values_split,ref_spec_row_indices_split)]
        
        single_matched_rows = np.where(np.sum(sparse_lib_matrix>0,1)==1)[0]
        peaks_not_shared = [
            np.array([[dia_spectrum[i,1],j] for i,j in zip(dia,lib) if i in single_matched_rows]) for dia,lib in zip(ref_spec_row_indices_split,ref_spec_values_split)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r2_unique = [np_pearson_cor(*i.T).statistic if i.shape[0]>1 else 0 for i in peaks_not_shared ]
            
        frac_unique_pred = [np.divide(*np.sum(i,axis=0)[::-1])*c if i.shape[0]>0 else 0 for i,c in zip(peaks_not_shared,lib_coefficients)] #frac of int matched by unique peaks pred by unique peaks
        
        frac_dia_intensity_pred = [(i*c)/j for i,j,c in zip(frac_lib_intensity,frac_dia_intensity,lib_coefficients)]
        
        #### stack spectrum features
        r2all = np.ones_like(num_lib_peaks_matched)*r2all
        frac_int_matched = np.ones_like(num_lib_peaks_matched)*frac_int_matched
        frac_int_pred = (np.ones_like(num_lib_peaks_matched)*np.sum(predicted_spec))/tic
        frac_int_matched_pred = (np.ones_like(num_lib_peaks_matched)*np.sum(predicted_spec))/np.sum(dia_spec_int)
        large_coeff_indices = np.where(np.array(lib_coefficients)>1)[0] # identify large coeffs
        large_coeff_matched_peaks = np.unique(np.concatenate(([ref_spec_row_indices_split[i] for i in large_coeff_indices]))) # select the peaks matched to these
        large_coeff_int_pred = np.sum([np.sum(ref_spec_values_split[i])*list(lib_coefficients)[i] for i in large_coeff_indices]) # sum the intensity predicted
        large_coeff_int_matched = np.sum(dia_spectrum[large_coeff_matched_peaks,1]) # sum the intensity matched
        ## Note: some predictions over-shoot the matched peak so we overestimate this value
        ## Q: Should we report different values for coeffs < 1??
        frac_int_matched_pred_sigcoeff = (np.ones_like(num_lib_peaks_matched)*large_coeff_int_pred)/large_coeff_int_matched # create vals for all peaks
        
                
        subset_row_indices = np.unique(sparse_row_indices[np.where(np.isin(sparse_col_indices,large_coeff_indices))])
        subset_row_indices = np.delete(subset_row_indices,np.where(subset_row_indices==max(subset_row_indices))[0][0])
        large_coeffs = np.squeeze(lib_coefficients) # get the coeffs
        large_coeffs[large_coeffs<1] = 0 # set those <1 to 0
        scaled_matrix = np.multiply(sparse_lib_matrix.toarray(),large_coeffs)#scale the matrix
        subset_pred_spec = np.sum(scaled_matrix,1)
        subset_cosine = cosim(dia_spec_int[subset_row_indices],subset_pred_spec[subset_row_indices])
        large_coeff_cosine = np.ones_like(num_lib_peaks_matched)*subset_cosine
        hyperscores, b_counts, y_counts = map(list, zip(*[hyperscore_b_y(library[i]["frags"],j) for i,j in zip(ref_pep_cand,lib_peaks_matched)]))
        longest_y_ions = [longest_y(library[i]["frags"],j) for i,j in zip(ref_pep_cand,lib_peaks_matched)]

        scribe_scores = get_scribe(
            ref_spec_row_indices_split,
            ref_spec_col_indices_split,
            ref_spec_values_split,
            dia_spectrum[:,1]
        )
    
        residuals, y_pred = get_residuals(
            ref_spec_values_split,
            ref_spec_row_indices_split,
            ref_spec_col_indices_split,
            [],
            [],
            [],
            dia_spectrum[:,1],
            lib_coefficients,
            0,
            0
        )
        # Then use y_pred for the manhattan distance
        manhattan_distances, fitted_spectral_contrasts = get_manhattan_distance(
            ref_spec_row_indices_split,
            ref_spec_col_indices_split,
            ref_spec_values_split,
            dia_spectrum[:,1],
            y_pred
        )

        gof_stats, max_unmatched_residuals, max_matched_residuals = gof_stat(
            ref_spec_row_indices_split,
            ref_spec_col_indices_split,
            ref_spec_values_split,
            residuals,
            dia_spectrum[:,1],
            lib_coefficients,
            0
        )

        features = np.stack([num_lib_peaks_matched,
                            frac_lib_intensity,
                            frac_dia_intensity,
                            rel_error,
                            rt_error,
                            frac_int_matched,
                            frac_int_pred,
                            r2all,
                            r2_lib_spec,
                            r2_unique,
                            frac_unique_pred,
                            frac_dia_intensity_pred,
                            hyperscores,
                            b_counts, 
                            y_counts,
                            longest_y_ions,
                            scribe_scores,
                            max_unmatched_residuals,
                            max_matched_residuals,
                            gof_stats,
                            manhattan_distances,
                            fitted_spectral_contrasts,
                            frac_int_matched_pred,
                            frac_int_matched_pred_sigcoeff,
                            large_coeff_cosine,
                            rt_mz[:,1][window_idxs[ref_peaks_in_dia]]
                                ],-1)
        
        
        ####################################
            
    #Select non-zero coeffs
    # Note: many coeffs are non-zero but essentially zero!! Perhaps set less than 1e-7??
    non_zero_coeffs = [c for c in lib_coefficients if c!=0]
    non_zero_coeffs_idxs = [i for i,c in enumerate(lib_coefficients) if c!=0]
    
    output = [[0,spec_idx,0,0,prec_mz,prec_rt,*np.zeros(len(names)-6)]]
    
    if len(non_zero_coeffs)>0:
        lib_spec_ids = [ref_pep_cand[i] for i in range(len(ref_pep_cand)) if lib_coefficients[i] != 0]
        all_spec_ids = lib_spec_ids
        all_features = features
        all_ms2_frags = [[";".join(map(str,j)) for j in i] for i in zip(frag_names,
                                                                        frag_errors,
                                                                        lib_frag_mz,
                                                                        lib_frag_int,
                                                                        obs_frag_int)]
        
        return_prot = config.protein_column in library[next(iter(library))]
        output = [[non_zero_coeffs[i],
                   spec_idx,
                   ms1_spec.scan_num,
                   all_spec_ids[i][0],
                   all_spec_ids[i][1],
                   prec_mz,
                   prec_rt,
                   *all_features[j],
                   *all_ms2_frags[j],
                   file_name,
                   library[(re.sub("Decoy_","",all_spec_ids[i][0]),all_spec_ids[i][1])][config.protein_column] if return_prot else "NA" ]
                   for i,j in zip(range(len(non_zero_coeffs)),non_zero_coeffs_idxs)]
        
        # output = [[non_zero_coeffs[i],
        #            spec_idx,
        #            lib_spec_ids[i][0],
        #            lib_spec_ids[i][1],
        #            prec_mz,
        #            prec_rt,
        #            *features[j]] for i,j in zip(range(len(non_zero_coeffs)),non_zero_coeffs_idxs)]
    
    if return_frags:
        return output, [frag_errors,frag_mz]
    else:
        return output


# def fit_to_lib_decoy(dia_spec,library,rt_mz,all_keys,dino_features=None,rt_filter=False,ms1_mz=None,mz_func = np.array, # mz_func is calibration function - default is just keeping values the same,
#                ms1_spectra = None,
#                rt_tol = config.rt_tol,
#                ms1_tol = config.ms1_tol,
#                mz_tol = config.mz_tol):
#     #print("AAAAAAAAA")
#     spec_idx=dia_spec.scan_num
    
#     # mz_tol = config.mz_tol
#     # rt_tol = min(config.rt_tol,config.opt_rt_tol)
#     # ms1_tol = min(config.ms1_tol,config.opt_ms1_tol)
#     top_n=config.top_n
#     atleast_m=config.args.atleast_m

#     spec = dia_spec#spectra.ms2scans[spec_idx]
#     dia_spectrum = np.stack(spec.peak_list(),1)
#     prec_mz = spec.prec_mz
#     prec_rt = spec.RT
#     # spec_idx = spec.id
    
#     windowWidth = window_width(dia_spec)
    
#     if ms1_spectra is not None:
#         ms1_rt = np.array([i.RT for i in ms1_spectra])
#         closest_ms1_scan_idx = closest_ms1spec(prec_rt, ms1_rt)
#         ms1_spec = ms1_spectra[closest_ms1_scan_idx]
    
    
    
#     lib_coefficients = []
   
#     if ms1_mz:
#         _bool = np.abs(rt_mz[:,1]-ms1_mz)<ms1_tol
        
#     else:
#         if rt_filter:
#             _bool = np.logical_and(np.abs(rt_mz[:,1]-prec_mz)<(windowWidth/2),np.abs(rt_mz[:,0]-prec_rt)<rt_tol)
#         else:
#             _bool = np.abs(rt_mz[:,1]-prec_mz)<(windowWidth/2)
            
#     window_idxs = np.where(_bool)[0]        
        
        
        
#     ### match lib spec to features
#     if dino_features is not None:
#         filtered_dino = feature_list_mz(feature_list_rt(dino_features,prec_rt,rt_tol=rt_tol),
#                                         prec_mz,windowWidth)
#         window_edges = createTolWindows(filtered_dino.mz, tolerance=ms1_tol)
#         window_idxs = window_idxs[np.where((np.searchsorted(window_edges,rt_mz[window_idxs,1])%2)==1)[0]]
        
    
#     mass_window_candidates = [all_keys[i] for i in window_idxs]
#     candidate_peaks = [library[i]['spectrum'] for i in mass_window_candidates]
    
    
#     ###### Process dia spectrum 
    
#     # what are the first indices of peaks grouped by tolerance
#     merged_coords_idxs = np.searchsorted(dia_spectrum[:,0]+mz_tol*dia_spectrum[:,0],dia_spectrum[:,0])
    
#     # what are the first mz of these peak groups
#     merged_coords = dia_spectrum[np.unique(merged_coords_idxs),0]
#     # print(merged_coords)
    
    
#     # NB - should we not sum the intensities?????
#     # merged_intensities = [np.mean(dia_spectrum[np.where(merged_coords_idxs==i)[0],1]) for i in np.unique(merged_coords_idxs)]
#     #merged_intensities = np.zeros(len((merged_coords_idxs)))
#     #for j,val in zip(merged_coords_idxs,dia_spectrum[:,1]):
#     #    merged_intensities[j]+=val
#     #merged_intensities = [np.mean(dia_spectrum[merged_coords_idxs==i,1]) for i in np.unique(merged_coords_idxs)]
#     #merged_intensities = merged_intensities[merged_intensities!=0]
#     merged_intensities = np.bincount(merged_coords_idxs, weights=dia_spectrum[:, 1])
#     merged_intensities = merged_intensities[merged_intensities != 0]
    
#     #update spectrum to new values (note mz remains first in group as this will eventually be rounded)
#     dia_spectrum = np.array((merged_coords,merged_intensities)).transpose()
#     # print(dia_spectrum)
    
#     #get window edge positions each side of peaks in observed spectra (NB the tolerance is now about the first peak in the group not the middile)
#     centroid_breaks = np.concatenate((dia_spectrum[:,0]-mz_tol*dia_spectrum[:,0],dia_spectrum[:,0]+mz_tol*dia_spectrum[:,0]))
#     centroid_breaks = np.sort(centroid_breaks)
#     bin_centers = np.mean(np.stack((centroid_breaks[::2],centroid_breaks[1::2]),1),1)
    
#     # lib_idx=0
#     # M = candidate_peaks[lib_idx]
#     ref_coords = [np.searchsorted(centroid_breaks,M[:,0]) for M in candidate_peaks]
#     top_ten = [np.searchsorted(centroid_breaks,M[np.argsort(-M[:,1])[0:min(top_n,M.shape[0])],0]) for M in candidate_peaks]
    
#     ## Filter precursors based on resp. MS1 peak
#     ms1_peak = ~np.isnan([closest_peak_diff(mz,ms1_spec.mz) for mz in rt_mz[window_idxs,1]])
     
    
#     # does the top ten peaks fall between centroid breaks? i.e. odd numbers (%2==1), 
#     # ref_peaks_in_dia = [i for i in range(len(candidate_peaks)) if len([a for a in top_ten[i] if a%2 ==1])>atleast_m]
#     # print(ref_peaks_in_dia)
#     all_norm_intensities = [M[:,1]/sum(M[:,1]) for M in candidate_peaks]
#     # ref_peaks_in_dia = [i for i in range(len(candidate_peaks)) if np.sum(all_norm_intensities[i][(ref_coords[i]%2)==1])>0.5 and np.sum(top_ten[i]%2)>atleast_m]
#     ref_peaks_in_dia = [i for i in range(len(candidate_peaks)) if np.sum(all_norm_intensities[i][(ref_coords[i]%2)==1])>0.5 and np.sum(top_ten[i]%2)>atleast_m and ms1_peak[i]]
#     # print(ref_peaks_in_dia)
    
#     prop_ref_peaks_in_dia = [len([a for a in top_ten[i] if a%2 ==1])/candidate_peaks[i].shape[0] for i in range(len(candidate_peaks))]
    
#     # print(len(ref_peaks_in_dia))
    
#     # filter database further to those that match the required num peaks
#     ref_pep_cand_loc = [ref_coords[i] for i in ref_peaks_in_dia]
#     ref_pep_cand_list = [candidate_peaks[i] for i in ref_peaks_in_dia]
#     # ref_pep_cand = [candidate_lib[i]["seq"] for i in ref_peaks_in_dia] # Nb this is modified seq!!
#     ref_pep_cand = [mass_window_candidates[i] for i in ref_peaks_in_dia] # Nb this is modified seq!!
    
#     norm_intensities = [M[:,1]/sum(M[:,1]) for M in ref_pep_cand_list]


#     ########## Update
#     # lib peaks that match
#     lib_peaks_matched = [j%2==1 for j in ref_pep_cand_loc]
    
#     # name these something different so can be accessed later
#     ref_spec_row_indices_split = [np.int32(((i[j]+1)/2)-1) for i,j in zip(ref_pep_cand_loc,lib_peaks_matched)] # NB these are floats
#     num_lib_peaks_matched = np.array([np.sum(i) for i in lib_peaks_matched]) #f1
#     ref_spec_col_indices_split = [np.array([idx]*i) for idx,i in zip(range(len(ref_pep_cand)),num_lib_peaks_matched)] 
#     ref_spec_values_split = [ints[i] for ints,i in zip(norm_intensities,lib_peaks_matched)]
    
    
    
    
#     ### Generate eqivalent Decoy spectra
    
#     mass_window_decoy_candidates = [("Decoy_"+i[0],i[1]) for i in mass_window_candidates] 
#     converted_seqs = [change_seq(i[0]) for i in mass_window_candidates]
#     decoy_mz = np.array([mass.fast_mass(i, charge=j[1]) for i,j in zip(converted_seqs, mass_window_candidates)])
#     converted_frags = [convert_frags(i[0], library[i]["frags"]) for i in mass_window_candidates]
#     candidate_decoy_peaks = [frag_to_peak(i) for i in converted_frags]
    
#     ## Decoy equiv
#     decoy_coords = [np.searchsorted(centroid_breaks,M[:,0]) for M in candidate_decoy_peaks]
#     top_ten_decoy = [np.searchsorted(centroid_breaks,M[np.argsort(-M[:,1])[0:min(top_n,M.shape[0])],0]) for M in candidate_decoy_peaks]
#     # decoy_peaks_in_dia = [i for i in range(len(candidate_decoy_peaks)) if len([a for a in top_ten_decoy[i] if a%2 ==1])>atleast_m]
#     all_norm_decoy_intensities = [M[:,1]/sum(M[:,1]) for M in candidate_decoy_peaks]
#     decoy_ms1_peak = ~np.isnan([closest_peak_diff(mz,ms1_spec.mz) for mz in decoy_mz])
#     decoy_peaks_in_dia = [i for i in range(len(candidate_peaks)) if np.sum(all_norm_decoy_intensities[i][(decoy_coords[i]%2)==1])>0.5 and np.sum(top_ten_decoy[i]%2)>atleast_m and decoy_ms1_peak[i]]
    
#     decoy_pep_cand_loc = [decoy_coords[i] for i in decoy_peaks_in_dia]
#     decoy_pep_cand_list = [candidate_decoy_peaks[i] for i in decoy_peaks_in_dia]
#     decoy_pep_cand = [mass_window_decoy_candidates[i] for i in decoy_peaks_in_dia] # Nb this is modified seq!!
    
#     norm_decoy_intensities = [M[:,1]/sum(M[:,1]) for M in decoy_pep_cand_list]
    
#     decoy_lib_peaks_matched = [j%2==1 for j in decoy_pep_cand_loc]
    
#     decoy_spec_row_indices_split = [np.int32(((i[j]+1)/2)-1) for i,j in zip(decoy_pep_cand_loc,decoy_lib_peaks_matched)] # NB these are floats
#     num_decoy_peaks_matched = np.array([np.sum(i) for i in decoy_lib_peaks_matched]) #f1
#     decoy_spec_col_indices_split = [np.array([idx]*i,dtype=int) for idx,i in zip(range(len(decoy_pep_cand)),num_decoy_peaks_matched)] 
#     decoy_spec_values_split = [ints[i] for ints,i in zip(norm_decoy_intensities,decoy_lib_peaks_matched)]
    
#     frag_errors = []
#     frag_mz = []
#     decoy_frag_errors = []
#     decoy_frag_mz = []
    
#     if len(ref_spec_row_indices_split)>0 and len(ref_spec_col_indices_split)>0 and len(ref_spec_values_split)>0:
        
#         #### concatenate the matrix values
#         ref_spec_row_indices = np.concatenate(ref_spec_row_indices_split)
#         ref_spec_col_indices = np.concatenate(ref_spec_col_indices_split)
#         ref_spec_values = np.concatenate(ref_spec_values_split)
        
#         frag_errors = [np.array(bin_centers[ref_spec_row_indices_split[i]]-ref_pep_cand_list[i][:,0][lib_peaks_matched[i]])/bin_centers[ref_spec_row_indices_split[i]] for i in range(len(lib_peaks_matched))]
#         frag_mz = [ref_pep_cand_list[i][:,0][lib_peaks_matched[i]] for i in range(len(lib_peaks_matched))]
        
        
#         if len(decoy_spec_row_indices_split)>0:
#             decoy_spec_row_indices = np.concatenate(decoy_spec_row_indices_split)
#             decoy_spec_col_indices = np.concatenate(decoy_spec_col_indices_split)+max(ref_spec_col_indices)+1
#             decoy_spec_values = np.concatenate(decoy_spec_values_split)
#             decoy_frag_errors = [np.array(bin_centers[decoy_spec_row_indices_split[i]]-decoy_pep_cand_list[i][:,0][decoy_lib_peaks_matched[i]])/bin_centers[decoy_spec_row_indices_split[i]] for i in range(len(decoy_lib_peaks_matched))]
#             decoy_frag_mz = [decoy_pep_cand_list[i][:,0][decoy_lib_peaks_matched[i]] for i in range(len(decoy_lib_peaks_matched))]
            
#         else:
#             decoy_spec_row_indices=np.array([],dtype=int)
#             decoy_spec_col_indices=np.array([],dtype=int)
#             decoy_spec_values=np.array([],dtype=int)
        
#         # what peaks from the spectrum are matched by library peps
#         # unique_row_idxs = [int(i) for i in set(np.concatenate([ref_spec_row_indices,decoy_spec_row_indices]))]
#         # unique_row_idxs.sort()
#         unique_row_idxs = np.unique(np.concatenate((ref_spec_row_indices,decoy_spec_row_indices)))
#         unique_row_idxs = np.array(np.sort(unique_row_idxs),dtype=int)
        
#         dia_spec_int = dia_spectrum[unique_row_idxs,1]
        
#         # add another term to penalise additional lib peaks
#         dia_spec_int = np.append(dia_spec_int,[0]) 
#         # find peaks that are bot matched in dia spectrum
#         ref_peaks_not_in_dia = np.array([idx for loc_list in ref_pep_cand_loc for idx in range(len(loc_list)) if loc_list[idx]%2==0])
#         # get col indices (will just be one for each)
#         not_dia_col_indices = np.arange(len(ref_pep_cand))
#         num_rows = max(unique_row_idxs)
#         # row indices always the last row (num peaks+1)
#         not_dia_row_indices = [num_rows+1]*len(not_dia_col_indices)
#         # sum peak intensities not in dia spectrum
#         not_dia_values = np.array([np.sum([norm_intensities[idx][peak_idx] for peak_idx in range(len(norm_intensities[idx])) if ref_pep_cand_loc[idx][peak_idx]%2==0])
#                                   for idx in range(len(norm_intensities))])
    
#         ref_sparse_row_indices = np.append(ref_spec_row_indices,not_dia_row_indices)
#         ref_sparse_col_indices = np.append(ref_spec_col_indices,not_dia_col_indices)
#         ref_sparse_values = np.append(ref_spec_values,not_dia_values)
        
#         ### Decoy
#         decoy_peaks_not_in_dia = np.array([idx for loc_list in decoy_pep_cand_loc for idx in range(len(loc_list)) if loc_list[idx]%2==0])
#         decoy_not_dia_col_indices = np.arange(len(decoy_pep_cand))
#         num_rows = max(unique_row_idxs)
#         decoy_not_dia_row_indices = [num_rows+1]*len(decoy_not_dia_col_indices)
#         decoy_not_dia_values = np.array([np.sum([norm_decoy_intensities[idx][peak_idx] for peak_idx in range(len(norm_decoy_intensities[idx])) if decoy_pep_cand_loc[idx][peak_idx]%2==0])
#                                   for idx in range(len(norm_decoy_intensities))])
    
#         decoy_sparse_row_indices = np.append(decoy_spec_row_indices,decoy_not_dia_row_indices)
#         decoy_sparse_col_indices = np.append(decoy_spec_col_indices,decoy_not_dia_col_indices+max(ref_spec_col_indices)+1)
#         decoy_sparse_values = np.append(decoy_spec_values,decoy_not_dia_values)
        
        
#         sparse_row_indices = np.concatenate((ref_sparse_row_indices,decoy_sparse_row_indices))
#         sparse_col_indices = np.concatenate((ref_sparse_col_indices,decoy_sparse_col_indices))
#         sparse_values = np.concatenate((ref_sparse_values,decoy_sparse_values))
        
#         # some dia peaks are not matched and are therefore ignored
#         # below ranks the rows by number therefore removing missing rows
#         sparse_row_indices = stats.rankdata(sparse_row_indices,method="dense").astype(int)-1
        
#         # Generate sparse matrix from data
#         sparse_lib_matrix = sparse.coo_matrix((sparse_values,(sparse_row_indices,sparse_col_indices)))

#         # Fit lib spectra to observed spectra
#         fit_results = sparse_nnls.lsqnonneg(sparse_lib_matrix,dia_spec_int,{"show_progress":False})
#         lib_coefficients = fit_results['x']
        
        
        
#         ####################################
#         ### features 
#         frac_lib_intensity = [np.sum(i) for i in ref_spec_values_split] # all ints sum to 1 so these give frac
#         tic = np.sum(dia_spectrum[:,1])
#         frac_dia_intensity = [np.sum(dia_spectrum[i,1])/tic for i in ref_spec_row_indices_split]
#         # mz tol
#         if ms1_spectra is not None:
#             rel_error = np.array([closest_peak_diff(mz,ms1_spec.mz) for mz in rt_mz[window_idxs[ref_peaks_in_dia],1]])
#         elif dino_features is not None:
#             rel_error = ms1_error(np.array(filtered_dino.mz), rt_mz[window_idxs[ref_peaks_in_dia],1], tol=ms1_tol)
#         else:
#             rel_error = np.zeros(len(ref_peaks_in_dia))
#         rt_error = prec_rt-rt_mz[window_idxs[ref_peaks_in_dia],0]
        
#         frac_int_matched = np.sum(dia_spec_int)/np.sum(dia_spectrum[:,1])
#         predicted_spec = np.squeeze(sparse_lib_matrix*lib_coefficients)[:-1]
#         with warnings.catch_warnings():
#             warnings.simplefilter("ignore")
#             r2all = np_pearson_cor(dia_spec_int[:-1],predicted_spec).statistic
        
#             r2_lib_spec = [np_pearson_cor(i,dia_spectrum[j,1]).statistic for i,j in zip(ref_spec_values_split,ref_spec_row_indices_split)]
        
#         single_matched_rows = np.where(np.sum(sparse_lib_matrix>0,1)==1)[0]
#         peaks_not_shared = [np.array([[dia_spectrum[i,1],j] for i,j in zip(dia,lib) if i in single_matched_rows]) for dia,lib in zip(ref_spec_row_indices_split,ref_spec_values_split)]
#         with warnings.catch_warnings():
#             warnings.simplefilter("ignore")
#             r2_unique = [np_pearson_cor(*i.T).statistic if i.shape[0]>1 else 0 for i in peaks_not_shared ]
#         frac_unique_pred = [np.divide(*np.sum(i,axis=0)[::-1])*c if i.shape[0]>0 else 0 for i,c in zip(peaks_not_shared,lib_coefficients)] #frac of int matched by unique peaks pred by unique peaks
        
#         frac_dia_intensity_pred = [(i*c)/j for i,j,c in zip(frac_lib_intensity,frac_dia_intensity,lib_coefficients)]
        
#         #### stack spectrum features
#         r2all = np.ones_like(num_lib_peaks_matched)*r2all
#         frac_int_matched = np.ones_like(num_lib_peaks_matched)*frac_int_matched
#         frac_int_pred = (np.ones_like(num_lib_peaks_matched)*np.sum(predicted_spec))/tic
#         frac_int_matched_pred = (np.ones_like(num_lib_peaks_matched)*np.sum(predicted_spec))/np.sum(dia_spec_int)
#         large_coeff_indices = np.where(np.array(lib_coefficients)>1)[0] # identify large coeffs
#         large_coeff_matched_peaks = np.unique(np.concatenate(([(ref_spec_row_indices_split+decoy_spec_row_indices_split)[i] for i in large_coeff_indices]))) # select the peaks matched to these
#         large_coeff_int_pred = np.sum([np.sum((ref_spec_values_split+decoy_spec_values_split)[i])*list(lib_coefficients)[i] for i in large_coeff_indices]) # sum the intensity predicted
#         large_coeff_int_matched = np.sum(dia_spectrum[large_coeff_matched_peaks,1]) # sum the intensity matched
#         ## Note: some predictions over-shoot the matched peak so we overestimate this value
#         ## Q: Should we report different values for coeffs < 1??
#         frac_int_matched_pred_sigcoeff = (np.ones_like(num_lib_peaks_matched)*large_coeff_int_pred)/large_coeff_int_matched # create vals for all peaks
        
        
#         subset_row_indices = np.unique(sparse_row_indices[np.where(np.isin(sparse_col_indices,large_coeff_indices))])
#         subset_row_indices = np.delete(subset_row_indices,np.where(subset_row_indices==max(subset_row_indices))[0][0])
#         large_coeffs = np.squeeze(lib_coefficients) # get the coeffs
#         large_coeffs[large_coeffs<1] = 0 # set those <1 to 0
#         scaled_matrix = np.multiply(sparse_lib_matrix.toarray(),large_coeffs)#scale the matrix
#         subset_pred_spec = np.sum(scaled_matrix,1)
#         subset_cosine = cosim(dia_spec_int[subset_row_indices],subset_pred_spec[subset_row_indices])
#         large_coeff_cosine = np.ones_like(num_lib_peaks_matched)*subset_cosine
        
#         hyperscores, b_counts, y_counts = map(list, zip(*[hyperscore_b_y(library[i]["frags"],j) for i,j in zip(ref_pep_cand,lib_peaks_matched)]))
#         longest_y_ions = [longest_y(library[i]["frags"],j) for i,j in zip(ref_pep_cand,lib_peaks_matched)]
#         features = np.stack([num_lib_peaks_matched,
#                               frac_lib_intensity,
#                               frac_dia_intensity,
#                               rel_error,
#                               rt_error,
#                               frac_int_matched,
#                               frac_int_pred,
#                               r2all,
#                               r2_lib_spec,
#                               r2_unique,
#                               frac_unique_pred,
#                               frac_dia_intensity_pred,
#                               hyperscores,
#                               b_counts,
#                               y_counts,
#                               longest_y_ions,
#                               frac_int_matched_pred,
#                               frac_int_matched_pred_sigcoeff,
#                               large_coeff_cosine
#                                 ],-1)
        
        
#         ####################################
#         ####################################
#         ### DECOY features 
        
#         frac_lib_intensity = [np.sum(i) for i in decoy_spec_values_split] # all ints sum to 1 so these give frac
#         tic = np.sum(dia_spectrum[:,1])
#         frac_dia_intensity = [np.sum(dia_spectrum[i,1])/tic for i in decoy_spec_row_indices_split]
#         # mz tol
#         if ms1_spectra is not None:
#             rel_error = np.array([closest_peak_diff(mz,ms1_spec.mz) for mz in rt_mz[window_idxs[decoy_peaks_in_dia],1]])
#         elif dino_features is not None:
#             rel_error = ms1_error(np.array(filtered_dino.mz), mz_func(np.array(decoy_mz)[decoy_peaks_in_dia]), tol=ms1_tol)
#         else:
#             rel_error = np.zeros(len(decoy_peaks_in_dia))
#         rt_error = prec_rt-rt_mz[window_idxs[decoy_peaks_in_dia],0] #NB this is not a true reflection of rt
        
#         frac_int_matched = np.sum(dia_spec_int)/np.sum(dia_spectrum[:,1])
#         predicted_spec = np.squeeze(sparse_lib_matrix*lib_coefficients)[:-1]
        
#         with warnings.catch_warnings():
#             warnings.simplefilter("ignore")
            
#             r2all = np_pearson_cor(dia_spec_int[:-1],predicted_spec).statistic
#             r2_lib_spec = [np_pearson_cor(i,dia_spectrum[j,1]).statistic for i,j in zip(decoy_spec_values_split,decoy_spec_row_indices_split)]
        
#         single_matched_rows = np.where(np.sum(sparse_lib_matrix>0,1)==1)[0]
#         peaks_not_shared = [np.array([[dia_spectrum[i,1],j] for i,j in zip(dia,lib) if i in single_matched_rows]) for dia,lib in zip(decoy_spec_row_indices_split,decoy_spec_values_split)]
#         with warnings.catch_warnings():
#             warnings.simplefilter("ignore")
#             r2_unique = [np_pearson_cor(*i.T).statistic if i.shape[0]>1 else 0 for i in peaks_not_shared ]
#         frac_unique_pred = [np.divide(*np.sum(i,axis=0)[::-1])*c if i.shape[0]>0 else 0 for i,c in zip(peaks_not_shared,lib_coefficients)] #frac of int matched by unique peaks pred by unique peaks
        
#         frac_dia_intensity_pred = [(i*c)/j for i,j,c in zip(frac_lib_intensity,frac_dia_intensity,lib_coefficients)]
        
#         #### stack spectrum features
#         r2all = np.ones_like(num_decoy_peaks_matched)*r2all
#         frac_int_matched = np.ones_like(num_decoy_peaks_matched)*frac_int_matched
#         frac_int_pred = (np.ones_like(num_decoy_peaks_matched)*np.sum(predicted_spec))/tic
#         frac_int_matched_pred = (np.ones_like(num_decoy_peaks_matched)*np.sum(predicted_spec))/np.sum(dia_spec_int)
#         large_coeff_indices = np.where(np.array(lib_coefficients)>1)[0] # identify large coeffs
#         large_coeff_matched_peaks = np.unique(np.concatenate(([(ref_spec_row_indices_split+decoy_spec_row_indices_split)[i] for i in large_coeff_indices]))) # select the peaks matched to these
#         large_coeff_int_pred = np.sum([np.sum((ref_spec_values_split+decoy_spec_values_split)[i])*list(lib_coefficients)[i] for i in large_coeff_indices]) # sum the intensity predicted
#         large_coeff_int_matched = np.sum(dia_spectrum[large_coeff_matched_peaks,1]) # sum the intensity matched
#         ## Note: some predictions over-shoot the matched peak so we overestimate this value
#         frac_int_matched_pred_sigcoeff = (np.ones_like(num_decoy_peaks_matched)*large_coeff_int_pred)/large_coeff_int_matched # create vals for all peaks
        
#         large_coeff_cosine = np.ones_like(num_decoy_peaks_matched)*subset_cosine
                              
#         hyperscores, b_counts, y_counts = map(list, zip(*[hyperscore_b_y(i,j) for i,j in zip([converted_frags[k] for k in decoy_peaks_in_dia],decoy_lib_peaks_matched)]))
#         longest_y_ions = [longest_y(i,j) for i,j in zip([longest_y(converted_frags[k]) for k in decoy_peaks_in_dia],decoy_lib_peaks_matched)]
#         #print("TEST")
#         decoy_features = np.stack([num_decoy_peaks_matched,
#                               frac_lib_intensity,
#                               frac_dia_intensity,
#                               rel_error,
#                               rt_error,
#                               frac_int_matched,
#                               frac_int_pred,
#                               r2all,
#                               r2_lib_spec,
#                               r2_unique,
#                               frac_unique_pred,
#                               frac_dia_intensity_pred,
#                               hyperscores,
#                               b_counts,
#                               y_counts,
#                               longest_y_ions,
#                               frac_int_matched_pred,
#                               frac_int_matched_pred_sigcoeff,
#                               large_coeff_cosine
#                                 ],-1)
        
#         # all_features.append(features)
#         ####################################
#     #Select non-zero coeffs
#     # Note: many coeffs are non-zero but essentially zero!! Perhaps set less than 1e-7??
#     non_zero_coeffs = [c for c in lib_coefficients if c!=0]
#     non_zero_coeffs_idxs = [i for i,c in enumerate(lib_coefficients) if c!=0]
    
#     output = [[0,spec_idx,ms1_spec.scan_num,0,0,prec_mz,prec_rt,*np.zeros(len(names)-7)]]
    
#     if len(non_zero_coeffs)>0:
#         lib_spec_ids = [ref_pep_cand[i] for i in range(len(ref_pep_cand)) if lib_coefficients[i] != 0]
#         decoy_spec_ids = [decoy_pep_cand[i] for i in range(len(decoy_pep_cand)) if lib_coefficients[int(max(ref_sparse_col_indices))+1+i] != 0]
        
#         all_spec_ids = lib_spec_ids+decoy_spec_ids
#         output = [[non_zero_coeffs[i],
#                    spec_idx,
#                    all_spec_ids[i][0],
#                    all_spec_ids[i][1],
#                    prec_mz,
#                    prec_rt,
#                    *np.concatenate((features,decoy_features))[j]] for i,j in zip(range(len(non_zero_coeffs)),non_zero_coeffs_idxs)]
#         # output = [[non_zero_coeffs[i],spec_idx,lib_spec_ids[i][0],lib_spec_ids[i][1],prec_mz,prec_rt,*features[j]] for i,j in zip(range(len(non_zero_coeffs)),non_zero_coeffs_idxs)]
    
#     return output


def merge_spectrum_peaks(spec, mz_ppm):
    # TODO: this duplicates the merge logic in fit_to_lib2 (line ~790) — consolidate?
    dia_spectrum = np.array(spec.peak_list(), dtype=np.float64).T
    merged_coords_idxs = np.searchsorted(dia_spectrum[:,0]+mz_ppm*dia_spectrum[:,0],dia_spectrum[:,0])
    merged_coords = dia_spectrum[np.unique(merged_coords_idxs),0]
    merged_intensities = np.bincount(merged_coords_idxs, weights=dia_spectrum[:, 1])
    merged_intensities = merged_intensities[merged_intensities != 0]
    spec.mz = merged_coords
    spec.intens = merged_intensities