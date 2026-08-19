
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
import ptinnls as sparse_nnls
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
from pyteomics import mass
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


def _split_flat(flat_arr, offsets):
    """Split a flat array into a list of sub-arrays using an offset table."""
    return [flat_arr[offsets[i]:offsets[i+1]] for i in range(len(offsets)-1)]


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
    prec_mz, tic):
    """Assemble the 27-column feature matrix in one nogil pass.

    Replaces np.ones_like * scalar broadcasts (5 allocations) + np.stack of
    27 arrays (~1500 GIL-holding samples). Fills a pre-allocated (n, 27)
    array directly — no intermediate arrays, no GIL.

    Per-candidate arrays (length n) are copied directly into their column.
    Scalar features (frac_int_matched, frac_int_pred, frac_int_matched_pred,
    lc_frac, lc_cosine, tic) are broadcast by filling the column with the value.
    """
    n = len(num_lib_peaks_matched)
    out = np.empty((n, 27), dtype=np.float64)
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
    prec_mzs, ms1_mz, ms1_tol,
    frac_lib_threshold, atleast_m, match_ms1,
    bin_mz, bin_int, bin_mob, mz_tol, im_tol, has_im):
    """Like _create_entries_core_jit but reads directly from library backing arrays.

    Instead of pre-flattened fragment arrays, this takes the library's contiguous
    spectrum_data (split into mz/int columns), spectrum_offsets, spectrum_lengths,
    and a list of candidate indices into those arrays. Eliminates the Python-level
    flattening that was the #1 GIL bottleneck.

    Args:
        centroid_breaks: float64[] — sorted DIA bin edges
        spec_data_mz:  float64[] — library spectrum_data column 0 (all fragment m/z)
        spec_data_int: float64[] — library spectrum_data column 1 (all fragment intensities)
        spec_offsets:  int64[n_lib] — start offset in spec_data for each library entry
        spec_lengths:  int32[n_lib] — number of fragments for each library entry
        topn_data:     int32[] — library top_n_data (local fragment indices)
        topn_offsets:  int64[n_lib] — start offset in topn_data for each library entry
        topn_lengths:  int32[n_lib] — number of top-N entries for each library entry
        cand_indices:  int32/int64[] — which library entries are candidates
        prec_mzs:      float64[n_cands] — precursor m/z per candidate
        ms1_mz:        float64[] — MS1 spectrum m/z (sorted)
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

    # Step 4: MS1 closest-peak error
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

    return passing, flat_rows, flat_cols, flat_vals, flat_offsets, ms1_error_out, all_coords, all_norm_int, frag_offsets, prec_im_out


def create_entries_direct(centroid_breaks,
                          spec_data_mz, spec_data_int,
                          spec_offsets, spec_lengths,
                          topn_data, topn_offsets, topn_lengths,
                          candidate_indices,
                          mass_window_candidates,
                          atleast_m,
                          prec_mzs,
                          ms1_spec,
                          ms1_tol,
                          bin_mz=None,
                          bin_int=None,
                          bin_mob=None,
                          mz_tol=0.0,
                          im_tol=0.0,
                          has_im=False):
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
        mass_window_candidates: list — candidate keys (for output reconstruction)
        atleast_m: int — minimum top-N fragments matched
        prec_mzs: float64[] — precursor m/z per candidate
        ms1_spec: MS1 spectrum object with .mz attribute
        ms1_tol: float64 — relative MS1 tolerance

    Returns:
        Same tuple as create_entries.
    """
    n_cands = len(candidate_indices)
    if n_cands == 0:
        return ([], [], [], [],
                np.empty(0, np.int32), np.empty(0, np.int32), np.empty(0, np.float64),
                np.zeros(1, np.int32), [], [],
                np.array([], dtype=np.float64),
                np.empty(0, np.int64), np.empty(0, np.float64),
                np.zeros(1, np.int32), np.empty(0, np.int32), [])

    cand_idx_arr = np.asarray(candidate_indices, dtype=np.int64)

    passing, flat_rows, flat_cols, flat_vals, flat_offsets, ms1_error_out, \
        all_coords, all_norm_int, frag_offsets, prec_im_out = \
        _create_entries_direct_jit(
            np.ascontiguousarray(centroid_breaks, dtype=np.float64),
            spec_data_mz, spec_data_int,
            spec_offsets, spec_lengths,
            topn_data, topn_offsets, topn_lengths,
            cand_idx_arr,
            np.ascontiguousarray(prec_mzs, dtype=np.float64),
            np.ascontiguousarray(ms1_spec.mz, dtype=np.float64),
            float(ms1_tol),
            float(config.args.lib_frac), int(atleast_m),
            bool(config.args.no_ms1_req),
            np.ascontiguousarray(
                bin_mz if bin_mz is not None else np.zeros(0), dtype=np.float64),
            np.ascontiguousarray(
                bin_int if bin_int is not None else np.zeros(0), dtype=np.float64),
            np.ascontiguousarray(
                bin_mob if bin_mob is not None else np.zeros(0), dtype=np.float64),
            float(mz_tol), float(im_tol), bool(has_im))

    # Reconstruct Python lists from JIT output — only for passing candidates
    peaks_in_dia = passing.tolist()
    pep_cand_loc = [all_coords[frag_offsets[i]:frag_offsets[i + 1]] for i in peaks_in_dia]
    # Reconstruct (n,2) spectrum arrays only for passing candidates from library arrays
    pep_cand_list = []
    for i in peaks_in_dia:
        lib_idx = cand_idx_arr[i]
        off = int(spec_offsets[lib_idx])
        l = int(spec_lengths[lib_idx])
        pep_cand_list.append(np.stack([spec_data_mz[off:off + l], spec_data_int[off:off + l]], axis=1))
    pep_cand = [mass_window_candidates[i] for i in peaks_in_dia]
    norm_intensities = [all_norm_int[frag_offsets[i]:frag_offsets[i + 1]] for i in peaks_in_dia]
    lib_peaks_matched = [pep_cand_loc[j] % 2 == 1 for j in range(len(peaks_in_dia))]
    # Precursor IM per passing candidate, aligned with pep_cand.
    prec_im_passing = [float(prec_im_out[i]) for i in peaks_in_dia]

    return (peaks_in_dia,
            pep_cand,
            pep_cand_loc,
            pep_cand_list,
            flat_rows, flat_cols, flat_vals, flat_offsets, norm_intensities, lib_peaks_matched, ms1_error_out,
            all_coords, all_norm_int, frag_offsets, passing, prec_im_passing)


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
    """Compute Gram matrix (A^T A) and A^T y from COO arrays in one pass.
    Returns (gram, Aty) where gram is (n_cols, n_cols) and Aty is (n_cols,)."""
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
    return gram, Aty


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

    # Compute Gram matrix and A^T y in one JIT pass (nogil — releases GIL)
    gram, Aty = _coo_gram_and_Aty(coo_rows, coo_cols, coo_vals, y, n_cols)

    # Data-driven regularization
    alpha_max = np.max(np.abs(Aty)) / n_rows
    alpha = alpha_max * 1e-4  # dynamic range within a spectrum is roughly 1,000x

    # Data-driven l1_ratio from max pairwise column correlation
    norms = np.sqrt(np.diag(gram))
    norms[norms == 0] = 1
    corr = gram / np.outer(norms, norms)
    np.fill_diagonal(corr, 0)
    max_corr = np.max(np.abs(corr))
    l1_ratio = min(max(1 - max_corr ** 2, 0.1), 0.9)

    # Build CSC once from COO via JIT (nogil) — only format enet_path accepts.
    # Replaces COO→CSR→CSC conversion chain that held the GIL.
    # Also pre-allocate X_w for IRLS loop — mutate .data in-place to avoid
    # reconstructing the csc_matrix object each iteration.
    csc_data, csc_indices, csc_indptr = _coo_to_csc_arrays(
        coo_rows, coo_cols, coo_vals, n_rows, n_cols)
    A_csc = csc_matrix((csc_data, csc_indices, csc_indptr), shape=(n_rows, n_cols))
    X_w = csc_matrix((csc_data.copy(), csc_indices, csc_indptr), shape=(n_rows, n_cols))

    coef = np.zeros(n_cols)
    _enet_path = enet_path.__wrapped__

    # Initial solve with uniform weights
    weights = np.ones(n_rows)
    _rng = _get_enet_rng()  # thread-local, reseeded to 42 (enet_path mutates rng state)
    _, coef_path, _, n_iters = _enet_path(
        A_csc, y,
        l1_ratio=l1_ratio,
        alphas=[alpha],
        positive=True,
        coef_init=coef,
        check_input=False,
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
    
#@profile
def get_features(
    rt_mz,
    ref_rows, ref_vals, ref_cols, ref_offsets,
    dec_rows, dec_vals, dec_cols, dec_offsets,
    ref_peaks_in_dia,
    dia_spectrum,
    prec_rt,
    window_idxs,
    dia_spec_int,
    lib_coefficients,
    sparse_row_indices,
    sparse_col_indices,
    sparse_values,
    lib_peaks_matched,
    ref_pep_cand,
    all_row_indices,
    all_values,
    prec_frag_intensities,
    ms1_error,
    ref_spec_offset,
    decoy_spec_offset,
    ordered_frag_codes=None,
    unique_lookup_dia=None):

    val_obs = dia_spectrum[:, 1]
    coeffs = np.asarray(lib_coefficients).ravel()
    tic = np.sum(val_obs)

    # Reconstruct split views where needed for per-candidate loops below
    ref_spec_row_indices_split = _split_flat(ref_rows, ref_offsets)
    ref_spec_values_split = _split_flat(ref_vals, ref_offsets)

    # Build combined flat arrays (ref + decoy) for large_coeff JIT
    n_ref = len(ref_offsets) - 1
    n_dec = len(dec_offsets) - 1
    if n_dec > 0:
        _all_flat_rows = np.concatenate([ref_rows, dec_rows])
        _all_flat_vals = np.concatenate([ref_vals, dec_vals])
        _all_flat_offsets = np.concatenate([ref_offsets, dec_offsets[1:] + ref_offsets[-1]])
    else:
        _all_flat_rows = ref_rows
        _all_flat_vals = ref_vals
        _all_flat_offsets = ref_offsets

    # Map sparse matrix column index → position in combined flat arrays
    n_total = n_ref + n_dec
    _max_col = max(int(ref_spec_offset + n_ref), int(decoy_spec_offset + n_dec)) if n_total > 0 else 0
    _col_to_pos = np.full(_max_col, -1, dtype=np.int32)
    if n_ref > 0:
        _col_to_pos[ref_spec_offset:ref_spec_offset + n_ref] = np.arange(n_ref, dtype=np.int32)
    if n_dec > 0:
        _col_to_pos[decoy_spec_offset:decoy_spec_offset + n_dec] = np.arange(n_ref, n_ref + n_dec, dtype=np.int32)

    # ── Step 2: Build predicted spectrum y_pred = A * x ──
    y_pred = np.zeros(len(val_obs))
    if len(ref_rows) > 0:
        _build_y_pred_jit(ref_rows, ref_cols, ref_vals, ref_offsets, coeffs, ref_spec_offset, y_pred)
    if len(dec_rows) > 0:
        _build_y_pred_jit(dec_rows, dec_cols, dec_vals, dec_offsets, coeffs, decoy_spec_offset, y_pred)
    residuals = val_obs - y_pred

    # ── Step 3: Fused per-candidate features ──
    # Replaces three separate functions that each looped over the same fragment data:
    #   - get_scribe       → scribe_scores
    #   - gof_stat         → gof_stats, max_unmatched_residuals, max_matched_residuals
    #   - get_manhattan_distance → manhattan_distances, fitted_spectral_contrasts
    # Also computes simple per-candidate stats that were previously list comprehensions:
    #   - num_lib_peaks_matched  (was: np.array([np.sum(i) for i in lib_peaks_matched]))
    #   - frac_lib_intensity     (was: [np.sum(i) for i in ref_spec_values_split])
    #   - frac_dia_intensity     (was: [np.sum(dia_spectrum[i,1])/tic for i in ...])
    n = len(ref_spec_row_indices_split)
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
            tic
        )

    # mz tol
    rel_error = np.where(~np.isnan(ms1_error), np.abs(ms1_error), -1.0)
    rt_error = prec_rt-rt_mz[:,0]

    frac_int_matched = np.sum(dia_spec_int)/np.sum(dia_spectrum[:,1])

    # Compute predicted spectrum + large_coeff features in one JIT pass (nogil).
    # Replaces: sparse_lib_matrix*lib_coefficients, np.isin, np.unique, .toarray(), cosim, etc.
    _n_coo_rows = int(sparse_row_indices.max()) + 1 if len(sparse_row_indices) > 0 else 0
    predicted_spec_full, _lc_frac, _lc_cosine = _large_coeff_block_jit(
        sparse_row_indices, sparse_col_indices, sparse_values, _n_coo_rows,
        coeffs, val_obs,
        _all_flat_rows, _all_flat_vals, _all_flat_offsets,
        _col_to_pos)
    predicted_spec = predicted_spec_full[:-1]  # drop penalty row (unmatched-fragment intensity)

    r2all = np.zeros_like(rt_error)
    r2_lib_spec = np.zeros_like(rt_error)
    
    # Use precomputed DIA-space lookup from fit_to_lib2 if available
    if unique_lookup_dia is not None:
        _unique_lookup = unique_lookup_dia
    else:
        # Fallback: compute from COO arrays directly
        _n_sm = int(sparse_row_indices.max()) + 1 if len(sparse_row_indices) > 0 else 0
        _unique_lookup = _single_match_lookup_jit(sparse_row_indices, sparse_col_indices, _n_sm)

    r2_unique = np.zeros_like(rt_error)

    # JIT: compute frac_unique_pred directly from flat arrays (GIL-free)
    frac_unique_pred = np.zeros(n)
    if len(ref_rows) > 0:
        _compute_unique_frac_jit(
            ref_rows, ref_vals, ref_offsets,
            _unique_lookup, val_obs, coeffs, ref_spec_offset,
            frac_unique_pred)

    frac_dia_intensity_pred = (frac_lib_intensity * coeffs[ref_spec_offset:ref_spec_offset + n]) / np.where(frac_dia_intensity > 0, frac_dia_intensity, 1.0)

    #### stack spectrum features
    # Compute scalar feature values (broadcast to all candidates below in JIT)
    _sum_predicted = np.sum(predicted_spec)
    _sum_dia_spec_int = np.sum(dia_spec_int)
    _frac_int_matched_scalar = frac_int_matched  # already a scalar
    _frac_int_pred_scalar = _sum_predicted / tic if tic > 0 else 0.0
    _frac_int_matched_pred_scalar = _sum_predicted / _sum_dia_spec_int if _sum_dia_spec_int > 0 else 0.0

    # ── Hyperscores + ion counts ──
    # Replaces per-candidate loop of hyperscore2() + get_index() calls (231K calls
    # total across target+decoy). Batch-processes all candidates in one JIT'd pass.
    if len(prec_frag_intensities) > 0 and ordered_frag_codes is not None:
        # Flatten intensities and codes into contiguous arrays for numba
        all_hyper_int = np.concatenate(prec_frag_intensities).astype(np.float64)
        all_hyper_codes = np.concatenate(ordered_frag_codes).astype(np.int32)
        n_hyper = len(prec_frag_intensities)
        _hyper_lens = np.array([len(x) for x in prec_frag_intensities], dtype=np.int32)
        hyper_offsets = np.zeros(n_hyper + 1, dtype=np.int32)
        np.cumsum(_hyper_lens, out=hyper_offsets[1:])

        hyperscores = np.zeros(n_hyper)
        b_counts = np.zeros(n_hyper)
        y_counts = np.zeros(n_hyper)
        longest_y_ions = np.zeros(n_hyper)

        _compute_hyperscores_jit(
            all_hyper_int, all_hyper_codes, hyper_offsets,
            hyperscores, b_counts, y_counts, longest_y_ions
        )
    else:
        hyperscores = np.zeros_like(num_lib_peaks_matched)
        b_counts = np.zeros_like(num_lib_peaks_matched)
        y_counts = np.zeros_like(num_lib_peaks_matched)
        longest_y_ions = np.zeros_like(num_lib_peaks_matched)

    # ── Assemble feature matrix in one pass (nogil) ──
    # Replaces ~5 np.ones_like*scalar broadcasts + np.stack of 26 arrays.
    # Each row is a feature column; transposed at the end via -1 axis.
    _prec_mz = rt_mz[:, 1].copy()  # contiguous for JIT
    features = _assemble_features_jit(
        num_lib_peaks_matched,          #  0: number of library peaks matched per candidate
        frac_lib_intensity,             #  1: fraction of library intensity matched
        frac_dia_intensity,             #  2: fraction of DIA intensity at matched peaks
        rel_error,                      #  3: relative MS1 m/z error
        rt_error,                       #  4: RT error (observed - calibrated)
        _frac_int_matched_scalar,       #  5: fraction of total DIA intensity matched (scalar → broadcast)
        _frac_int_pred_scalar,          #  6: fraction of TIC predicted by model (scalar → broadcast)
        r2all,                          #  7: R² all (placeholder, currently zeros)
        r2_lib_spec,                    #  8: R² library spectrum (placeholder, currently zeros)
        r2_unique,                      #  9: R² unique (placeholder, currently zeros)
        frac_unique_pred,               # 10: predicted intensity fraction at uniquely-matched peaks
        frac_dia_intensity_pred,        # 11: predicted DIA intensity fraction per candidate
        hyperscores,                    # 12: X!Tandem-style hyperscore
        b_counts,                       # 13: number of b-ions matched
        y_counts,                       # 14: number of y-ions matched
        longest_y_ions,                 # 15: longest consecutive y-ion series
        scribe_scores,                  # 16: scribe score (spectral similarity metric)
        max_unmatched_residuals,        # 17: max residual at unmatched peaks
        max_matched_residuals,          # 18: max residual at matched peaks
        gof_stats,                      # 19: goodness-of-fit statistic (log2 residual/fitted ratio)
        manhattan_distances,            # 20: manhattan distance between predicted and observed
        fitted_spectral_contrasts,      # 21: spectral contrast angle between fitted and observed
        _frac_int_matched_pred_scalar,  # 22: predicted/observed intensity ratio (scalar → broadcast)
        _lc_frac,                       # 23: large-coeff predicted/observed intensity ratio (scalar → broadcast)
        _lc_cosine,                     # 24: large-coeff subset cosine similarity (scalar → broadcast)
        _prec_mz,                       # 25: calibrated precursor m/z
        tic                              # 26: total ion current (scalar → broadcast)
    )
    return features


#@profile



#@profile
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
            bool(config.args.no_ms1_req))

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
def fit_to_lib2(dia_spec,
                library,
                rt_mz,
                all_keys,
                rt_tol,
                ms1_tol,
                mz_tol,
                dino_features=None,rt_filter=False,ms1_mz=None,
               ms1_spectra = None,
               return_frags = False,
               decoy=False,
               output_folder=None,
               frag_index=None,
               ms1_rt=None,
               im_bin_ms1=None):
    # spec_idx,dia_spec,library = inputs
    
    spec_idx=dia_spec.scan_num
    top_n=config.top_n
    atleast_m=config.args.atleast_m
    spec = dia_spec
    dia_spectrum = np.stack(spec.peak_list(),1)
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
    _im_tol = config.opt_im_tol if _has_im else 0.0
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

    # Get candidates via fragment index or fallback to m/z + RT window
    # Single query returns both target and decoy candidates from unified index
    if frag_index is not None and not ms1_mz:
        win_lo = prec_mz - windowWidth / 2
        win_hi = prec_mz + windowWidth / 2
        all_window_idxs = frag_index.query(
            dia_spectrum[:, 0], win_lo, win_hi,
            prec_rt, rt_tol, atleast_m
        )
    else:
        if ms1_mz:
            _bool = (np.abs(rt_mz[:,1]-ms1_mz)/ms1_mz)<ms1_tol
        else:
            if rt_filter:
                _bool = np.logical_and(np.abs(rt_mz[:,1]-prec_mz)<(windowWidth/2),np.abs(rt_mz[:,0]-prec_rt)<rt_tol)
            else:
                _bool = np.abs(rt_mz[:,1]-prec_mz)<(windowWidth/2)
        all_window_idxs = np.where(_bool)[0]

    # Split into target and decoy candidates
    # TODO: unify target/decoy processing paths to eliminate this split
    n_targets = library.n_targets
    target_mask = all_window_idxs < n_targets
    window_idxs = all_window_idxs[target_mask]
    decoy_window_idxs = all_window_idxs[~target_mask] if decoy else np.empty(0, dtype=all_window_idxs.dtype)

    mass_window_candidates = [all_keys[i] for i in window_idxs]
    _ref_idxs = library.resolve_indices(mass_window_candidates)

    spec_frags = None

    ref_peaks_in_dia,\
    ref_pep_cand,\
    ref_pep_cand_loc,\
    ref_pep_cand_list,\
    ref_flat_rows,\
        ref_flat_cols,\
        ref_flat_vals, \
        ref_flat_offsets, \
        norm_intensities, \
        lib_peaks_matched, \
        ref_ms1_error, \
        ref_all_coords, ref_all_norm_int, ref_frag_offsets, ref_passing, ref_prec_im = create_entries_direct(
                                        centroid_breaks=centroid_breaks,
                                        spec_data_mz=library.spectrum_mz,
                                        spec_data_int=library.spectrum_int,
                                        spec_offsets=library.spectrum_offsets,
                                        spec_lengths=library.spectrum_lengths,
                                        topn_data=library.top_n_data,
                                        topn_offsets=library.top_n_offsets,
                                        topn_lengths=library.top_n_lengths,
                                        candidate_indices=_ref_idxs,
                                        mass_window_candidates=mass_window_candidates,
                                        atleast_m=atleast_m,
                                        prec_mzs=rt_mz[:,1][window_idxs],
                                        ms1_spec=ms1_spec,
                                        ms1_tol=ms1_tol,
                                        bin_mz=_bin_mz,
                                        bin_int=_bin_int,
                                        bin_mob=_bin_mob,
                                        mz_tol=mz_tol,
                                        im_tol=_im_tol,
                                        has_im=_has_im)
    # Reconstruct split views where needed downstream
    ref_spec_row_indices_split = _split_flat(ref_flat_rows, ref_flat_offsets)
    ref_spec_col_indices_split = _split_flat(ref_flat_cols, ref_flat_offsets)
    ref_spec_values_split = _split_flat(ref_flat_vals, ref_flat_offsets)


    ### Generate equivalent Decoy spectra
    if decoy:
        # Decoy candidates already extracted from unified index above
        mass_window_decoy_candidates = [all_keys[i] for i in decoy_window_idxs]
        _decoy_idxs = library.resolve_indices(mass_window_decoy_candidates)
        converted_frag_codes = library.get_frag_codes_batch(_decoy_idxs)
        # Decoy m/z offset already pre-applied in rt_mz
        decoy_mz = rt_mz[:,1][decoy_window_idxs]

        decoy_spec_frags = None

        decoy_peaks_in_dia,\
        decoy_pep_cand,\
        decoy_pep_cand_loc,\
        decoy_pep_cand_list,\
        decoy_flat_rows,\
            decoy_flat_cols,\
                decoy_flat_vals, \
                    decoy_flat_offsets, \
                        norm_decoy_intensities, \
                            decoy_lib_peaks_matched, \
                                decoy_ms1_error, \
                                    dec_all_coords, dec_all_norm_int, dec_frag_offsets, dec_passing, dec_prec_im = create_entries_direct(
                                                                    centroid_breaks=centroid_breaks,
                                                                    spec_data_mz=library.spectrum_mz,
                                                                    spec_data_int=library.spectrum_int,
                                                                    spec_offsets=library.spectrum_offsets,
                                                                    spec_lengths=library.spectrum_lengths,
                                                                    topn_data=library.top_n_data,
                                                                    topn_offsets=library.top_n_offsets,
                                                                    topn_lengths=library.top_n_lengths,
                                                                    candidate_indices=_decoy_idxs,
                                                                    mass_window_candidates=mass_window_decoy_candidates,
                                                                    atleast_m=atleast_m,
                                                                    prec_mzs=decoy_mz,
                                                                    ms1_spec=ms1_spec,
                                                                    ms1_tol=ms1_tol,
                                                                    bin_mz=_bin_mz,
                                                                    bin_int=_bin_int,
                                                                    bin_mob=_bin_mob,
                                                                    mz_tol=mz_tol,
                                                                    im_tol=_im_tol,
                                                                    has_im=_has_im)
        # Reconstruct split views where needed downstream
        decoy_spec_row_indices_split = _split_flat(decoy_flat_rows, decoy_flat_offsets)
        decoy_spec_col_indices_split = _split_flat(decoy_flat_cols, decoy_flat_offsets)
        decoy_spec_values_split = _split_flat(decoy_flat_vals, decoy_flat_offsets)


    frag_errors = []
    lib_frag_mz = []
    decoy_col_offset = 0
    
    if len(ref_flat_rows) > 0:

        #### Use flat arrays directly (already concatenated from create_entries)
        ref_spec_row_indices = ref_flat_rows
        ref_spec_col_indices = ref_flat_cols
        ref_spec_values = ref_flat_vals

        frag_errors = [np.array(bin_centers[ref_spec_row_indices_split[i]]-ref_pep_cand_list[i][:,0][lib_peaks_matched[i]])/bin_centers[ref_spec_row_indices_split[i]] for i in range(len(lib_peaks_matched))]
        lib_frag_mz = [ref_pep_cand_list[i][:,0][lib_peaks_matched[i]] for i in range(len(lib_peaks_matched))]
        lib_frag_int = [ref_pep_cand_list[i][:,1][lib_peaks_matched[i]] for i in range(len(lib_peaks_matched))]
        obs_frag_int = [dia_spectrum[ref_spec_row_indices_split[i],1] for i in range(len(ref_spec_row_indices_split))]
        _ref_surv_idxs = library.resolve_indices(ref_pep_cand)
        _ref_surv_frag_codes = library.get_frag_codes_batch(_ref_surv_idxs)
        frag_name_codes = [_ref_surv_frag_codes[i][j] for i,j in enumerate(lib_peaks_matched)]
        frag_matched_intensities = [ref_pep_cand_list[idx][:,1][j] for idx,j in enumerate(lib_peaks_matched)]

        decoy_col_offset = np.max(ref_spec_col_indices)+1
        
    else:
        ref_spec_row_indices=np.array([],dtype=int)
        ref_spec_col_indices=np.array([],dtype=int)
        ref_spec_values=np.array([],dtype=int)
        frag_errors = []#np.array([],dtype=float)
        lib_frag_mz = []#np.array([],dtype=float)
        lib_frag_int = []
        obs_frag_int = []
        frag_name_codes = []
        frag_matched_intensities = []
        
        
    if decoy and len(decoy_flat_rows) > 0:
        decoy_spec_row_indices = decoy_flat_rows
        decoy_spec_col_indices = decoy_flat_cols + decoy_col_offset
        decoy_spec_values = decoy_flat_vals
        decoy_frag_errors = [np.array(bin_centers[decoy_spec_row_indices_split[i]]-decoy_pep_cand_list[i][:,0][decoy_lib_peaks_matched[i]])/bin_centers[decoy_spec_row_indices_split[i]] for i in range(len(decoy_lib_peaks_matched))]
        decoy_lib_frag_mz = [decoy_pep_cand_list[i][:,0][decoy_lib_peaks_matched[i]] for i in range(len(decoy_lib_peaks_matched))]
        decoy_lib_frag_int = [decoy_pep_cand_list[i][:,1][decoy_lib_peaks_matched[i]] for i in range(len(decoy_lib_peaks_matched))]
        decoy_obs_frag_int = [dia_spectrum[decoy_spec_row_indices_split[i],1] for i in range(len(decoy_spec_row_indices_split))]
        decoy_frag_name_codes = [converted_frag_codes[i][decoy_lib_peaks_matched[idx]] for idx,i in enumerate(decoy_peaks_in_dia)]
        decoy_frag_matched_intensities = [decoy_pep_cand_list[idx][:,1][decoy_lib_peaks_matched[idx]] for idx in range(len(decoy_peaks_in_dia))]
    else:
        decoy_spec_row_indices_split=[] ## needs to be improved
        decoy_spec_values_split=[] ## needs to be improved
        decoy_spec_row_indices=np.array([],dtype=int)
        decoy_spec_col_indices=np.array([],dtype=int)
        decoy_spec_values=np.array([],dtype=int)
        decoy_frag_errors = []#np.array([],dtype=float)
        decoy_lib_frag_mz = []#np.array([],dtype=float)
        decoy_lib_frag_int = []
        decoy_obs_frag_int = []
        decoy_frag_name_codes = []
        decoy_frag_matched_intensities = []
        

    if len(ref_flat_rows) > 0 or (decoy and len(decoy_flat_rows) > 0):
        # Single JIT call replaces ~14 GIL-acquiring numpy/scipy calls
        # (np.append ×6, np.concatenate ×3, np.unique, np.sort, rankdata, unmatched_peaks ×2)
        _dec_rows = decoy_flat_rows if (decoy and len(decoy_flat_rows) > 0) else np.empty(0, np.int32)
        _dec_cols = decoy_flat_cols if (decoy and len(decoy_flat_cols) > 0) else np.empty(0, np.int32)
        _dec_vals = decoy_flat_vals if (decoy and len(decoy_flat_vals) > 0) else np.empty(0, np.float64)
        _dec_coords = dec_all_coords if (decoy and len(decoy_flat_rows) > 0) else np.empty(0, np.int64)
        _dec_norm_int = dec_all_norm_int if (decoy and len(decoy_flat_rows) > 0) else np.empty(0, np.float64)
        _dec_frag_off = dec_frag_offsets if (decoy and len(decoy_flat_rows) > 0) else np.zeros(1, np.int32)
        _dec_passing = dec_passing if (decoy and len(decoy_flat_rows) > 0) else np.empty(0, np.int32)

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
        fit_results = huber_nnls_irls(sparse_values, sparse_row_indices, sparse_col_indices,
                                      _n_coo_rows, _n_coo_cols, dia_spec_int)
        lib_coefficients = fit_results['x']

        ####################################
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


        # Build decoy flat arrays for get_features (empty if no decoys)
        _dec_rows = decoy_flat_rows if (decoy and len(decoy_flat_rows) > 0) else np.empty(0, np.int32)
        _dec_vals = decoy_flat_vals if (decoy and len(decoy_flat_vals) > 0) else np.empty(0, np.float64)
        _dec_cols = decoy_flat_cols if (decoy and len(decoy_flat_cols) > 0) else np.empty(0, np.int32)
        _dec_offsets = decoy_flat_offsets if (decoy and len(decoy_flat_offsets) > 1) else np.zeros(1, np.int32)
        features = get_features(rt_mz[window_idxs[ref_peaks_in_dia]],
                                ref_flat_rows, ref_flat_vals, ref_flat_cols, ref_flat_offsets,
                                _dec_rows, _dec_vals, _dec_cols, _dec_offsets,
                                ref_peaks_in_dia,
                                dia_spectrum,
                                prec_rt,
                                window_idxs,
                                dia_spec_int,
                                lib_coefficients,
                                sparse_row_indices,
                                sparse_col_indices,
                                sparse_values,
                                lib_peaks_matched,
                                ref_pep_cand,
                                (ref_spec_row_indices_split+decoy_spec_row_indices_split),
                                (ref_spec_values_split+decoy_spec_values_split),
                                frag_matched_intensities,
                                ref_ms1_error,
                                0,
                                decoy_col_offset,
                                frag_name_codes,
                                unique_lookup_dia=unique_lookup_dia)


        unique_row_indices_split = [single_match_lookup[peak_idx_lookup[i]] for i in ref_spec_row_indices_split]
        unique_frags = [i[j] for i,j in zip(lib_frag_mz,unique_row_indices_split)]
        unique_frags_int = [i[j] for i,j in zip(obs_frag_int,unique_row_indices_split)]

        ####################################
        if decoy:
            decoy_features = get_features(np.stack([rt_mz[decoy_window_idxs[decoy_peaks_in_dia],0],decoy_mz[decoy_peaks_in_dia]],1),
                                            decoy_flat_rows, decoy_flat_vals, decoy_flat_cols, decoy_flat_offsets,
                                            ref_flat_rows, ref_flat_vals, ref_flat_cols, ref_flat_offsets,
                                            decoy_peaks_in_dia,
                                            dia_spectrum,
                                            prec_rt,
                                            decoy_window_idxs,
                                            dia_spec_int,
                                            lib_coefficients,
                                            sparse_row_indices,
                                            sparse_col_indices,
                                            sparse_values,
                                            decoy_lib_peaks_matched,
                                            decoy_pep_cand,
                                            (ref_spec_row_indices_split+decoy_spec_row_indices_split),
                                            (ref_spec_values_split+decoy_spec_values_split),
                                            decoy_frag_matched_intensities,
                                            decoy_ms1_error,
                                            decoy_col_offset,
                                            0,
                                            decoy_frag_name_codes,
                                            unique_lookup_dia=unique_lookup_dia)


            unique_row_indices_split_decoy = [single_match_lookup[peak_idx_lookup[i]] for i in decoy_spec_row_indices_split]
            unique_frags_decoy = [i[j] for i,j in zip(decoy_lib_frag_mz,unique_row_indices_split_decoy)]
            unique_frags_int_decoy = [i[j] for i,j in zip(decoy_obs_frag_int,unique_row_indices_split_decoy)]

        ####################################

    #Select non-zero coeffs
    # Note: many coeffs are non-zero but essentially zero!! Perhaps set less than 1e-7??
    non_zero_coeffs = [c for c in lib_coefficients if c!=0]
    non_zero_coeffs_idxs = [i for i,c in enumerate(lib_coefficients) if c!=0]
    # print(f"N: {len(lib_coefficients)}, {len(non_zero_coeffs)}")
    output = []

    if len(non_zero_coeffs)>0:
        target_spec_ids = [ref_pep_cand[i] for i in range(len(ref_pep_cand)) if lib_coefficients[i] != 0]
        # Precursor IM per hit, filtered by the same non-zero-coeff mask and in the
        # same target-then-decoy order as all_spec_ids.
        target_prec_im = [ref_prec_im[i] for i in range(len(ref_pep_cand)) if lib_coefficients[i] != 0]
        if decoy:
            updated_decoy_offset = decoy_col_offset
            decoy_spec_ids = [decoy_pep_cand[i] for i in range(len(decoy_pep_cand)) if lib_coefficients[updated_decoy_offset+i] != 0]
            decoy_prec_im = [dec_prec_im[i] for i in range(len(decoy_pep_cand)) if lib_coefficients[updated_decoy_offset+i] != 0]

            all_spec_ids = target_spec_ids+decoy_spec_ids
            all_prec_im = target_prec_im+decoy_prec_im
            n_target_hits = len(target_spec_ids)
            all_features = np.concatenate((features,decoy_features))
            # Store raw arrays instead of stringified — parquet handles list columns
            all_ms2_frags = [list(i) for i in zip(frag_name_codes+decoy_frag_name_codes,
                                                  frag_errors+decoy_frag_errors,
                                                  lib_frag_mz+decoy_lib_frag_mz,
                                                  lib_frag_int+decoy_lib_frag_int,
                                                  obs_frag_int+decoy_obs_frag_int,
                                                  unique_frags+unique_frags_decoy,
                                                  unique_frags_int+unique_frags_int_decoy)]


        else:
            all_spec_ids = target_spec_ids
            all_prec_im = target_prec_im
            n_target_hits = len(target_spec_ids)
            all_features = features
            all_ms2_frags = [list(i) for i in zip(frag_name_codes,
                                                  frag_errors,
                                                  lib_frag_mz,
                                                  lib_frag_int,
                                                  obs_frag_int,
                                                  unique_frags,
                                                  unique_frags_int)]

        # Check if protein column is populated (without creating _EntryView)
        _first_idx = library.key_to_idx[next(iter(library.key_to_idx))]
        _prot_field_map = {'protein_group': library.protein_group, 'protein_name': library.protein_name,
                           'genes': library.genes, 'UniprotID': library.uniprot_id}
        _prot_arr = _prot_field_map.get(config.protein_column)
        return_prot = _prot_arr is not None and _prot_arr[_first_idx] not in (None, '')

        # Pre-resolve protein column values for non-zero coefficients
        # Decoy entries have protein info copied from their parent target
        if return_prot:
            _prot_idxs = library.resolve_indices(all_spec_ids)
            _prot_vals = library.get_scalar_batch(_prot_idxs, config.protein_column)

        if config.args.timeplex:
            output = [[non_zero_coeffs[i],
                       spec_idx,
                       ms1_spec.scan_num,
                       all_spec_ids[i][0],
                       all_spec_ids[i][1],
                       all_spec_ids[i][2],
                       prec_mz,
                       prec_rt,
                       all_prec_im[i],
                       *all_features[j],
                       *all_ms2_frags[j],
                       config.args.mzml,
                       _prot_vals[i] if return_prot else "NA",
                       i >= n_target_hits]
                       for i,j in zip(range(len(non_zero_coeffs)),non_zero_coeffs_idxs)]

        else:

            output = [[non_zero_coeffs[i],
                       spec_idx,
                       ms1_spec.scan_num,
                       all_spec_ids[i][0],
                       all_spec_ids[i][1],
                       prec_mz,
                       prec_rt,
                       all_prec_im[i],
                       *all_features[j],
                       *all_ms2_frags[j],
                       config.args.mzml,
                       _prot_vals[i] if return_prot else "NA",
                       i >= n_target_hits]
                       for i,j in zip(range(len(non_zero_coeffs)),non_zero_coeffs_idxs)]
            
        # lib_spec_ids = [ref_pep_cand[i] for i in range(len(ref_pep_cand)) if lib_coefficients[i] != 0]
        # output = [[non_zero_coeffs[i],spec_idx,lib_spec_ids[i][0],lib_spec_ids[i][1],prec_mz,prec_rt,*features[j]] for i,j in zip(range(len(non_zero_coeffs)),non_zero_coeffs_idxs)]
    

    if return_frags:
        return output, [frag_errors,lib_frag_mz]
    else:
        return output


# #@profile
def fit_to_lib(dia_spec,library,rt_mz,all_keys,
               rt_tol,
               ms1_tol,
               mz_tol,
               dino_features=None,
               rt_filter=False,
               ms1_mz=None,
               ms1_spectra = None,
               return_frags = False,
               frac_matched = 0.5):
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
                   config.args.mzml,
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