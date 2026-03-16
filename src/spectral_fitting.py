"""
This Source Code Form is subject to the terms of the Oxford Nanopore
Technologies, Ltd. Public License, v. 1.0.  Full licence can be found
at https://github.com/ParallelSquared/JMod/blob/main/LICENSE.txt
"""


import numpy as np


import warnings
import ptinnls as sparse_nnls
from sklearn.linear_model import ElasticNet, Lasso
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

from scipy import stats
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


@njit
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


@njit
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
@njit
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


@njit
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


@njit
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


def _split_flat(flat_arr, offsets):
    """Split a flat array into a list of sub-arrays using an offset table."""
    return [flat_arr[offsets[i]:offsets[i+1]] for i in range(len(offsets)-1)]


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


def huber_nnls_irls(A, b, max_iter=1, tol=1e-4, c=4.685):
    """
    Asymmetric IRLS-weighted NNLS with Tukey biweight on under-predicted peaks.

    First pass: unweighted NNLS.
    Subsequent passes:
      - Under-prediction at observed peaks (residuals < 0, b > 0):
        Tukey bisquare weights — smoothly downweights large under-predictions,
        completely rejects beyond c * MAD. This is forgiving because other
        peptides in the mixture can explain the extra observed signal.
      - All other peaks: weight = 1.0 (full penalty for over-prediction
        and false signal at zeros).

    Args:
        A: Sparse library matrix (n_peaks x n_candidates)
        b: Observed intensities (n_peaks,)
        max_iter: Maximum IRLS iterations
        tol: Convergence tolerance on weight changes
        c: Tukey biweight tuning constant (multiples of MAD)

    Returns:
        dict with 'x': coefficients, 'weights': final sample weights
    """
    A_csr = A.tocsr() if sparse.issparse(A) else sparse.csr_matrix(A)
    n_peaks = A_csr.shape[0]

    # Normalize b so tol is scale-independent
    s = float(np.max(np.abs(b)) or 1.0)
    y = b / s

    # Data-driven regularization
    alpha_max = np.max(np.abs(A_csr.T.dot(y))) / n_peaks
    alpha = alpha_max * 1e-4 # 1e-4 means that the dynamic range within a spectrum is roughly 1,000x

    # Data-driven l1_ratio from max pairwise column correlation
    gram = (A_csr.T @ A_csr).toarray()
    norms = np.sqrt(np.diag(gram))
    norms[norms == 0] = 1
    corr = gram / np.outer(norms, norms)
    np.fill_diagonal(corr, 0)
    max_corr = np.max(np.abs(corr))
    l1_ratio = max(1 - max_corr ** 2, 0.1)

    model = ElasticNet(
        alpha=alpha,
        l1_ratio=l1_ratio,
        positive=True,
        fit_intercept=False,
        selection='random',
        random_state=42,
        tol=1e-3,
        max_iter=20000,
        warm_start=True,
    )

    # Initial solve with uniform weights
    weights = np.ones(n_peaks)
    model.fit(A_csr, y, sample_weight=weights)
    initial_n_iter = model.n_iter_

    x = model.coef_ * s
    residuals = A_csr.dot(x).ravel() - b

    # Compute cutoff from over-predicted observed peaks, apply to under-predicted
    over_pred = (residuals > 0) & (b > 0)
    if np.any(over_pred):
        abs_r_over = np.abs(residuals[over_pred])
        mad = np.median(np.abs(abs_r_over - np.median(abs_r_over)))
        cutoff = c * (mad / 0.6745) if mad > 0 else 1.0
    else:
        cutoff = 1.0

    for _ in range(max_iter):
        x = model.coef_ * s
        residuals = A_csr.dot(x).ravel() - b

        new_weights = np.ones(n_peaks)

        # Tukey biweight on under-predicted observed peaks
        under_pred = (residuals < 0) & (b > 0)
        if np.any(under_pred):
            abs_r = np.abs(residuals[under_pred])
            u = abs_r / cutoff
            # Bisquare: (1 - u²)² for |u| <= 1, 0 otherwise
            biweight = np.where(u <= 1.0, (1.0 - u**2)**2, 0.0)
            new_weights[under_pred] = biweight

        # Check convergence
        if np.max(np.abs(new_weights - weights)) < tol:
            break
        weights = new_weights
        weights *= weights.size / weights.sum()

        # Warm-started from previous coefficients
        model.fit(A_csr, y, sample_weight=weights)

    return {'x': model.coef_ * s, 'weights': weights,
            'initial_n_iter': initial_n_iter, 'robust_n_iter': model.n_iter_,
            'alpha_max': alpha_max, 'l1_ratio': l1_ratio}


def get_closest_ms1(prec_rt,ms1_spectra):
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
    sparse_lib_matrix,
    sparse_row_indices,
    sparse_col_indices,
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
    # Replaces get_residuals. Accumulates contributions from both ref and decoy
    # candidates into a single predicted intensity array. Each fragment contributes
    # lib_intensity * fit_coefficient to its matched DIA peak position.
    y_pred = np.zeros(len(val_obs))
    if len(ref_rows) > 0:
        _build_y_pred_jit(ref_rows, ref_cols, ref_vals, ref_offsets, coeffs, ref_spec_offset, y_pred)
    if len(dec_rows) > 0:
        _build_y_pred_jit(dec_rows, dec_cols, dec_vals, dec_offsets, coeffs, decoy_spec_offset, y_pred)
    # residuals = observed - predicted (same as get_residuals returned)
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
    rel_error = ms1_error#np.zeros(len(ref_peaks_in_dia))
    rt_error = prec_rt-rt_mz[:,0]
    
    frac_int_matched = np.sum(dia_spec_int)/np.sum(dia_spectrum[:,1])
    predicted_spec = np.squeeze(sparse_lib_matrix*lib_coefficients)[:-1]
    
    # r2all = np_pearson_cor(dia_spec_int[:-1],predicted_spec).statistic
    # r2_lib_spec = [np_pearson_cor(i,dia_spectrum[j,1]).statistic for i,j in zip(ref_spec_values_split,ref_spec_row_indices_split)]
    r2all = np.zeros_like(rt_error)
    r2_lib_spec = np.zeros_like(rt_error)
    
    # Use precomputed DIA-space lookup from fit_to_lib2 if available
    if unique_lookup_dia is not None:
        _unique_lookup = unique_lookup_dia
    else:
        single_matched_rows = np.where(np.sum(sparse_lib_matrix>0,1)==1)[0]
        _max_idx = max((dia.max() for dia in ref_spec_row_indices_split if len(dia) > 0), default=-1) + 1
        _unique_lookup = np.zeros(_max_idx, dtype=bool)
        _unique_lookup[single_matched_rows.ravel()] = True

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
    # r2all = np.ones_like(num_lib_peaks_matched)*r2all
    frac_int_matched = np.ones_like(num_lib_peaks_matched)*frac_int_matched
    frac_int_pred = (np.ones_like(num_lib_peaks_matched)*np.sum(predicted_spec))/tic
    frac_int_matched_pred = (np.ones_like(num_lib_peaks_matched)*np.sum(predicted_spec))/np.sum(dia_spec_int)
    large_coeff_indices = np.where(np.array(lib_coefficients)>1)[0] # identify large coeffs

    if len(large_coeff_indices) == 0: # Guards against spectra where no peptides have coeff > 1
        frac_int_matched_pred_sigcoeff = np.zeros_like(num_lib_peaks_matched)
        large_coeff_cosine = np.zeros_like(num_lib_peaks_matched)

    else: # standard execution
        # Use combined flat arrays for GIL-free computation
        _lc_positions = _col_to_pos[large_coeff_indices]
        _lc_valid = _lc_positions >= 0
        _lc_pos_valid = _lc_positions[_lc_valid]
        large_coeff_matched_peaks = np.unique(np.concatenate([_all_flat_rows[_all_flat_offsets[p]:_all_flat_offsets[p+1]] for p in _lc_pos_valid])) if len(_lc_pos_valid) > 0 else np.empty(0, dtype=np.int32)
        large_coeff_int_pred = _large_coeff_int_pred_jit(_all_flat_vals, _all_flat_offsets, coeffs, large_coeff_indices.astype(np.int32), _col_to_pos)
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
                          rt_mz[:,1],
                          # peaks
                            ],-1)
    return features


#@profile
def unmatched_peaks(norm_intensities,
                    pep_cand_loc,
                    last_row,
                    fit_type="a",
                    lower_limit = 1e-10):
    """
    3 fit_types:
        a: All summed unmatched intensities are fit to a single zero intensity "obs peak"
        b: Summed unmatched intensities of each precursor are fit to their own zero intensity "obs peak"
        c: Each unmatched peak is fit to its own zero intensity "obs peak"
    
    lower_limit:
        if normalized fragment intensity is below this threshold, exclude from fit (default essentially includes all peaks)
        Only applicable to type c
        
    """
    assert fit_type in ["a","b","c"]
    
    # Vectorized: sum unmatched (even-coord) intensities per candidate
    n_cands = len(pep_cand_loc)
    if fit_type in ("a", "b"):
        not_dia_col_indices = np.arange(n_cands)
        not_dia_values = np.array([np.sum(norm_intensities[idx][pep_cand_loc[idx] % 2 == 0])
                                   for idx in range(n_cands)])
        if fit_type == "a":
            not_dia_row_indices = np.full(n_cands, last_row, dtype=int)
        else:
            not_dia_row_indices = last_row + 1 + not_dia_col_indices

    elif fit_type == "c":
        all_unmatched_peaks = [norm_intensities[idx][(pep_cand_loc[idx] % 2 == 0) & (norm_intensities[idx] > lower_limit)]
                               for idx in range(n_cands)]
        num_unmatched_to_fit = [len(i) for i in all_unmatched_peaks]
        not_dia_col_indices = np.concatenate([np.full(cnt, idx, dtype=int) for idx, cnt in enumerate(num_unmatched_to_fit)]) if any(num_unmatched_to_fit) else np.array([], dtype=int)
        not_dia_row_indices = np.arange(sum(num_unmatched_to_fit), dtype=int) + last_row + 1
        not_dia_values = np.concatenate(all_unmatched_peaks) if any(num_unmatched_to_fit) else np.array([], dtype=float)
    
    return not_dia_row_indices, not_dia_col_indices, not_dia_values



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
                np.array([], dtype=np.float64))

    # ── Step 1: Batch searchsorted ──
    # Concatenate all fragment m/z into one flat array so we make a single
    # np.searchsorted call instead of N separate calls (was 30.7% of time).
    # frag_offsets tracks where each candidate's fragments start/end in the flat array.
    frag_lengths = np.array([M.shape[0] for M in candidate_peaks], dtype=np.int32)
    frag_offsets = np.empty(n_cands + 1, dtype=np.int32)
    frag_offsets[0] = 0
    np.cumsum(frag_lengths, out=frag_offsets[1:])
    total_frags = int(frag_offsets[-1])

    all_frag_mz = np.empty(total_frags)
    all_frag_int = np.empty(total_frags)
    for i in range(n_cands):
        s, e = int(frag_offsets[i]), int(frag_offsets[i + 1])
        all_frag_mz[s:e] = candidate_peaks[i][:, 0]
        all_frag_int[s:e] = candidate_peaks[i][:, 1]

    # One searchsorted for all fragments at once
    # Odd result = fragment matched a DIA peak, even = unmatched
    all_coords = np.searchsorted(centroid_breaks, all_frag_mz)

    # ── Step 2: Vectorized top-N match count ──
    # For each candidate, count how many of its top_n most intense fragments
    # landed on odd coords (i.e., matched DIA peaks). Uses reduceat to avoid
    # per-candidate Python loops.
    top_n_lengths = np.array([len(idxs) for idxs in top_n_idxs], dtype=np.int32)
    all_top_n_flat = np.concatenate(
        [frag_offsets[i] + idxs for i, idxs in enumerate(top_n_idxs)])
    top_n_odd = (all_coords[all_top_n_flat] % 2).astype(np.int32)

    top_n_offsets = np.empty(n_cands + 1, dtype=np.int32)
    top_n_offsets[0] = 0
    np.cumsum(top_n_lengths, out=top_n_offsets[1:])
    top_n_matched = np.add.reduceat(top_n_odd, top_n_offsets[:-1])

    # ── Step 3: Vectorized frac_lib_matched check ──
    # For each candidate, compute what fraction of its total library intensity
    # is at matched (odd coord) positions. This checks whether enough of the
    # library spectrum overlaps with observed DIA peaks.
    # Uses np.add.at to accumulate per-candidate sums without Python loops.
    cand_idx = np.repeat(np.arange(n_cands, dtype=np.int32), frag_lengths)

    # Per-candidate total intensity (for normalization)
    int_sums = np.zeros(n_cands)
    np.add.at(int_sums, cand_idx, all_frag_int)

    # Normalized intensity at matched positions, summed per candidate
    all_norm_int = all_frag_int / int_sums[cand_idx]
    matched_mask = all_coords % 2 == 1
    frac_matched = np.zeros(n_cands)
    np.add.at(frac_matched, cand_idx[matched_mask], all_norm_int[matched_mask])

    # ── Step 4: MS1 matching + candidate filtering ──
    # When match_ms1 is enabled, compute MS1 error for all candidates first
    # (needed for filtering). Otherwise defer to after filtering (only compute
    # for survivors) since most candidates get filtered out anyway.
    if config.match_ms1:
        ms1_error = _batch_closest_peak_diff(prec_mzs, ms1_spec.mz, ms1_tol)
        ms1_peak = ~np.isnan(ms1_error)
        passing = ((frac_matched > config.frac_lib_matched)
                   & (top_n_matched > atleast_m)
                   & ms1_peak)
    else:
        passing = ((frac_matched > config.frac_lib_matched)
                   & (top_n_matched > atleast_m))

    peaks_in_dia = np.where(passing)[0].tolist()

    # Deferred MS1 error: only compute for survivors when not filtering by MS1
    if not config.match_ms1:
        survivor_mzs = prec_mzs[peaks_in_dia] if len(peaks_in_dia) > 0 else np.array([])
        ms1_error_survivors = _batch_closest_peak_diff(survivor_mzs, ms1_spec.mz, ms1_tol)

    # ── Step 5: Build per-candidate output arrays for passing candidates ──
    pep_cand_loc = [all_coords[frag_offsets[i]:frag_offsets[i + 1]] for i in peaks_in_dia]
    pep_cand_list = [candidate_peaks[i] for i in peaks_in_dia]
    pep_cand = [mass_window_candidates[i] for i in peaks_in_dia]

    # Reuse pre-computed normalized intensities from the flat array
    norm_intensities = [all_norm_int[frag_offsets[i]:frag_offsets[i + 1]] for i in peaks_in_dia]
    lib_peaks_matched = [pep_cand_loc[j] % 2 == 1 for j in range(len(peaks_in_dia))]

    # Build flat arrays + offset table directly (avoids split→flatten round-trip)
    n_surv = len(peaks_in_dia)
    flat_offsets = np.zeros(n_surv + 1, dtype=np.int32)
    flat_rows_parts = []
    flat_cols_parts = []
    flat_vals_parts = []
    for j in range(n_surv):
        matched = lib_peaks_matched[j]
        rows_j = np.int32(((pep_cand_loc[j][matched] + 1) / 2) - 1)
        vals_j = norm_intensities[j][matched]
        flat_rows_parts.append(rows_j)
        flat_cols_parts.append(np.full(len(rows_j), j, dtype=np.int32))
        flat_vals_parts.append(vals_j)
        flat_offsets[j + 1] = flat_offsets[j] + len(rows_j)

    flat_rows = np.concatenate(flat_rows_parts).astype(np.int32) if flat_rows_parts else np.empty(0, np.int32)
    flat_cols = np.concatenate(flat_cols_parts).astype(np.int32) if flat_cols_parts else np.empty(0, np.int32)
    flat_vals = np.concatenate(flat_vals_parts).astype(np.float64) if flat_vals_parts else np.empty(0, np.float64)

    if config.match_ms1:
        ms1_error_out = ms1_error[peaks_in_dia]
    else:
        ms1_error_out = ms1_error_survivors

    return (peaks_in_dia,
            pep_cand,
            pep_cand_loc,
            pep_cand_list,
            flat_rows, flat_cols, flat_vals, flat_offsets, norm_intensities, lib_peaks_matched, ms1_error_out)


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
               decoy_library=None,
               output_folder=None,
               frag_index=None,
               decoy_frag_index=None):
    # spec_idx,dia_spec,library = inputs
    
    spec_idx=dia_spec.scan_num
    
    # mz_tol = config.mz_tol
    # rt_tol = min(config.rt_tol,config.opt_rt_tol)
    # ms1_tol = min(config.ms1_tol,config.opt_ms1_tol)
    top_n=config.top_n
    atleast_m=config.atleast_m
    spec = dia_spec#spectra.ms2scans[spec_idx]
    dia_spectrum = np.stack(spec.peak_list(),1)
    prec_mz = spec.prec_mz
    prec_rt = spec.RT
    # spec_idx = spec.id
    
    windowWidth = window_width(dia_spec)
    
    
    if ms1_spectra is not None:
        ms1_spec = get_closest_ms1(prec_rt,ms1_spectra)
    
    
    lib_coefficients = []

    ###### Process dia spectrum

    # TODO: merge_spectrum_peaks already pre-merges all spectra — this may be redundant
    # # what are the first indices of peaks grouped by tolerance
    merged_coords_idxs = np.searchsorted(dia_spectrum[:,0]+mz_tol*dia_spectrum[:,0],dia_spectrum[:,0])

    # # what are the first mz of these peak groups
    merged_coords = dia_spectrum[np.unique(merged_coords_idxs),0]

    merged_intensities = np.bincount(merged_coords_idxs, weights=dia_spectrum[:, 1])
    merged_intensities = merged_intensities[merged_intensities != 0]

    # #update spectrum to new values (note mz remains first in group as this will eventually be rounded)
    if dia_spectrum.shape != np.array((merged_coords,merged_intensities)).transpose().shape:
        print("Warning: Shapes dont match in fit_to_lib2")
    dia_spectrum = np.array((merged_coords,merged_intensities)).transpose()

    #get window edge positions each side of peaks in observed spectra (NB the tolerance is now about the first peak in the group not the middile)
    centroid_breaks = np.concatenate((dia_spectrum[:,0]-mz_tol*dia_spectrum[:,0],dia_spectrum[:,0]+mz_tol*dia_spectrum[:,0]))
    centroid_breaks = np.sort(centroid_breaks)
    bin_centers = np.mean(np.stack((centroid_breaks[::2],centroid_breaks[1::2]),1),1)

    # Get candidates via fragment index or fallback to m/z + RT window
    if frag_index is not None and not ms1_mz:
        win_lo = prec_mz - windowWidth / 2
        win_hi = prec_mz + windowWidth / 2
        window_idxs = frag_index.query(
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
        window_idxs = np.where(_bool)[0]

    mass_window_candidates = [all_keys[i] for i in window_idxs]
    candidate_peaks = [library[i]['spectrum'] for i in mass_window_candidates]

    top_n_idxs = [library[i]['top_n'] for i in mass_window_candidates]
    
    
    spec_frags = None
    if "spec_frags" in library[all_keys[0]].keys():
        spec_frags = [library[i]['spec_frags'] for i in mass_window_candidates]
        
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
        ref_ms1_error = create_entries(centroid_breaks=centroid_breaks,
                                        candidate_peaks=candidate_peaks,
                                        mass_window_candidates=mass_window_candidates,
                                        top_n=top_n,
                                        atleast_m=atleast_m,
                                        prec_mzs=rt_mz[:,1][window_idxs],
                                        ms1_spec=ms1_spec,
                                        ms1_tol=ms1_tol,
                                        top_n_idxs=top_n_idxs)
    # Reconstruct split views where needed downstream
    ref_spec_row_indices_split = _split_flat(ref_flat_rows, ref_flat_offsets)
    ref_spec_col_indices_split = _split_flat(ref_flat_cols, ref_flat_offsets)
    ref_spec_values_split = _split_flat(ref_flat_vals, ref_flat_offsets)

    
    ### Generate eqivalent Decoy spectra
    if decoy:
        # Get decoy candidates via fragment index or fallback to same as target
        if decoy_frag_index is not None and not ms1_mz:
            win_lo = prec_mz - windowWidth / 2
            win_hi = prec_mz + windowWidth / 2
            decoy_window_idxs = decoy_frag_index.query(
                dia_spectrum[:, 0], win_lo, win_hi,
                prec_rt, rt_tol, atleast_m
            )
        else:
            decoy_window_idxs = window_idxs
        decoy_mass_window_candidates = [all_keys[i] for i in decoy_window_idxs]

        mass_window_decoy_candidates = [("Decoy_"+i[0],*i[1:]) for i in decoy_mass_window_candidates]
        # print("old")
        # converted_seqs = [change_seq(i[0],config.args.decoy) for i in mass_window_candidates]
        # decoy_mz = np.array([convert_prec_mz(i, z=j[1]) for i,j in zip(converted_seqs, mass_window_candidates)])
        # if config.args.decoy=="rev": ## this will have the same mz as many correct mathces and therefore a really good ms1 isotope corr
        #     decoy_mz -= config.decoy_mz_offset
        # ## NB: Below needs to change to ibcorporate iso frags!!
        # converted_frags = [convert_frags(i[0], library[i]["frags"],config.args.decoy) for i in mass_window_candidates]
        # decoy_sorted_frags = [sorted(converted_frags[i],key = lambda x: converted_frags[i][x][0]) for i in range(len(converted_frags))]
        # if config.args.iso:
        #     candidate_decoy_peaks = [gen_isotopes(i,j) for i,j in zip(converted_seqs,converted_frags)]
        # else:
        #     candidate_decoy_peaks = [frag_to_peak(i) for i in converted_frags]

        # ## if using decoy_library
        # print("new")
        converted_frag_codes = [decoy_library[i]["ordered_frag_codes"] for i in decoy_mass_window_candidates]
        candidate_decoy_peaks = [decoy_library[i]["spectrum"] for i in decoy_mass_window_candidates]
        # decoy_mz = np.array([decoy_library[i]["prec_mz"] for i in mass_window_candidates])
        decoy_mz = rt_mz[:,1][decoy_window_idxs] - config.decoy_mz_offset

        decoy_top_n_idxs = [decoy_library[i]['top_n'] for i in decoy_mass_window_candidates]
        
        decoy_spec_frags = None
        # if "spec_frags" in library[all_keys[0]].keys():
        #     decoy_spec_frags = [specific_frags(i) for i in converted_frags]
        
        # ## Decoy equiv
        # decoy_coords = [np.searchsorted(centroid_breaks,M[:,0]) for M in candidate_decoy_peaks]
        # top_ten_decoy = [np.searchsorted(centroid_breaks,M[np.argsort(-M[:,1])[0:min(top_n,M.shape[0])],0]) for M in candidate_decoy_peaks]
        # # decoy_peaks_in_dia = [i for i in range(len(candidate_decoy_peaks)) if len([a for a in top_ten_decoy[i] if a%2 ==1])>atleast_m]
        # all_norm_decoy_intensities = [M[:,1]/sum(M[:,1]) for M in candidate_decoy_peaks]
        # decoy_ms1_peak = ~np.isnan([closest_peak_diff(mz,ms1_spec.mz) for mz in decoy_mz])
        # # decoy_peaks_in_dia = [i for i in range(len(candidate_peaks)) if np.sum(all_norm_decoy_intensities[i][(decoy_coords[i]%2)==1])>0.5 and np.sum(top_ten_decoy[i]%2)>atleast_m and decoy_ms1_peak[i]]
        # decoy_peaks_in_dia = [i for i in range(len(candidate_peaks)) if np.sum(top_ten_decoy[i]%2)>atleast_m and decoy_ms1_peak[i]]
        
        # decoy_pep_cand_loc = [decoy_coords[i] for i in decoy_peaks_in_dia]
        # decoy_pep_cand_list = [candidate_decoy_peaks[i] for i in decoy_peaks_in_dia]
        # decoy_pep_cand = [mass_window_decoy_candidates[i] for i in decoy_peaks_in_dia] # Nb this is modified seq!!
        
        # norm_decoy_intensities = [M[:,1]/sum(M[:,1]) for M in decoy_pep_cand_list]
        
        # decoy_lib_peaks_matched = [j%2==1 for j in decoy_pep_cand_loc]
        
        # decoy_spec_row_indices_split = [np.int32(((i[j]+1)/2)-1) for i,j in zip(decoy_pep_cand_loc,decoy_lib_peaks_matched)] # NB these are floats
        # num_decoy_peaks_matched = np.array([np.sum(i) for i in decoy_lib_peaks_matched]) #f1
        # decoy_spec_col_indices_split = [np.array([idx]*i,dtype=int) for idx,i in zip(range(len(decoy_pep_cand)),num_decoy_peaks_matched)] 
        # decoy_spec_values_split = [ints[i] for ints,i in zip(norm_decoy_intensities,decoy_lib_peaks_matched)]
        
        
    
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
                                decoy_ms1_error = create_entries(centroid_breaks=centroid_breaks,
                                                                    candidate_peaks=candidate_decoy_peaks,
                                                                    mass_window_candidates=mass_window_decoy_candidates,
                                                                    top_n=top_n,
                                                                    atleast_m=atleast_m,
                                                                    prec_mzs=decoy_mz,
                                                                    ms1_spec=ms1_spec,
                                                                    ms1_tol=ms1_tol,
                                                                    spec_frags=decoy_spec_frags,
                                                                    top_n_idxs=decoy_top_n_idxs)
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
        frag_name_codes = [library[i]["ordered_frag_codes"][j] for i,j in zip(ref_pep_cand,lib_peaks_matched)]
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
        # what peaks from the spectrum are matched by library peps
        unique_row_idxs = np.unique(np.concatenate((ref_spec_row_indices,decoy_spec_row_indices)))
        unique_row_idxs = np.array(np.sort(unique_row_idxs),dtype=int)
        
        
        # # find peaks that are bot matched in dia spectrum
        # ref_peaks_not_in_dia = np.array([idx for loc_list in ref_pep_cand_loc for idx in range(len(loc_list)) if loc_list[idx]%2==0])
        # # get col indices (will just be one for each)
        # not_dia_col_indices = np.arange(len(ref_pep_cand))
        # num_rows = max(unique_row_idxs)
        # # row indices always the last row (num peaks+1)
        # not_dia_row_indices = [num_rows+1]*len(not_dia_col_indices)
        # # sum peak intensities not in dia spectrum
        # not_dia_values = np.array([np.sum([norm_intensities[idx][peak_idx] for peak_idx in range(len(norm_intensities[idx])) if ref_pep_cand_loc[idx][peak_idx]%2==0])
        #                           for idx in range(len(norm_intensities))])
       
        if len(ref_spec_row_indices_split)>0:
            not_dia_row_indices,not_dia_col_indices,not_dia_values = unmatched_peaks(norm_intensities=norm_intensities,
                                                                                     pep_cand_loc=ref_pep_cand_loc,
                                                                                     last_row=max(unique_row_idxs)+1,
                                                                                     fit_type=config.unmatched_fit_type)
        else:
            not_dia_row_indices=np.array([],dtype=np.int32)
            not_dia_col_indices=np.array([],dtype=np.int32)
            not_dia_values=np.array([],dtype=np.int32)
            
        if decoy and len(decoy_spec_row_indices_split)>0:
            decoy_not_dia_row_indices,decoy_not_dia_col_indices,decoy_not_dia_values = unmatched_peaks(norm_intensities=norm_decoy_intensities,
                                                                                                         pep_cand_loc=decoy_pep_cand_loc,
                                                                                                         last_row=max(not_dia_row_indices,default=max(unique_row_idxs)+1), # if all ref are mathched the initial list is empty
                                                                                                         fit_type=config.unmatched_fit_type)
        else:
            decoy_not_dia_row_indices=np.array([],dtype=np.int32)
            decoy_not_dia_col_indices=np.array([],dtype=np.int32)
            decoy_not_dia_values=np.array([],dtype=np.int32)
            
        ref_sparse_row_indices = np.append(ref_spec_row_indices,not_dia_row_indices)
        ref_sparse_col_indices = np.append(ref_spec_col_indices,not_dia_col_indices)
        ref_sparse_values = np.append(ref_spec_values,not_dia_values)
        
        decoy_sparse_row_indices = np.append(decoy_spec_row_indices,decoy_not_dia_row_indices)
        decoy_sparse_col_indices = np.append(decoy_spec_col_indices,decoy_not_dia_col_indices+decoy_col_offset)
        decoy_sparse_values = np.append(decoy_spec_values,decoy_not_dia_values)
        
        
        sparse_row_indices = np.concatenate((ref_sparse_row_indices,decoy_sparse_row_indices))
        sparse_col_indices = np.concatenate((ref_sparse_col_indices,decoy_sparse_col_indices))
        sparse_values = np.concatenate((ref_sparse_values,decoy_sparse_values))
        
        # some dia peaks are not matched and are therefore ignored
        # below ranks the rows by number therefore removing missing rows
        new_row_indices = stats.rankdata(sparse_row_indices,method="dense").astype(int)-1
        # Dense lookup array replaces Python dict for O(1) vectorized index conversion
        _max_row = int(sparse_row_indices.max()) + 1
        peak_idx_lookup = np.full(_max_row, -1, dtype=np.int32)
        peak_idx_lookup[sparse_row_indices] = new_row_indices
        sparse_row_indices =new_row_indices
        
        # Generate sparse matrix from data
        sparse_lib_matrix = sparse.coo_matrix((sparse_values,(sparse_row_indices,sparse_col_indices)))
        
        
        dia_spec_int = dia_spectrum[unique_row_idxs,1]
        
        # add another term to penalise additional lib peaks
        dia_spec_int = np.append(dia_spec_int,[0]*(sparse_lib_matrix.shape[0]-dia_spec_int.shape[0]))

        # Fit lib spectra to observed spectra (Huber loss via IRLS)
        fit_results = huber_nnls_irls(sparse_lib_matrix, dia_spec_int)
        lib_coefficients = fit_results['x']

        if output_folder is not None:
            with open(output_folder + "/fitting_iterations.tsv", "a") as f:
                f.write(f"{spec_idx}\t{sparse_lib_matrix.shape[1]}\t{fit_results['initial_n_iter']}\t{fit_results['robust_n_iter']}\t{fit_results['alpha_max']}\n")
            n_candidates = sparse_lib_matrix.shape[1]
            n_nonzero = int(np.sum(lib_coefficients > 1))
            import os
            diag_path = output_folder + "/elasticnet_diag.tsv"
            write_header = not os.path.exists(diag_path)
            with open(diag_path, "a") as f:
                if write_header:
                    f.write("spec_idx\tn_candidates\tn_coeff_gt_1\talpha_max\tl1_ratio\n")
                f.write(f"{spec_idx}\t{n_candidates}\t{n_nonzero}\t{fit_results['alpha_max']}\t{fit_results['l1_ratio']}\n")


        ####################################
        # Compute single-matched rows ONCE, build DIA-space lookup for get_features
        single_matched_rows = np.where(np.sum(sparse_lib_matrix>0,1)==1)[0]
        _sm_max = int(new_row_indices.max()) + 1 if len(new_row_indices) > 0 else 0
        single_match_lookup = np.zeros(_sm_max, dtype=bool)
        single_match_lookup[single_matched_rows.ravel()] = True

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
                                sparse_lib_matrix,
                                sparse_row_indices,
                                sparse_col_indices,
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
                                            sparse_lib_matrix,
                                            sparse_row_indices,
                                            sparse_col_indices,
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
    if config.args.timeplex:
        output = [[0,spec_idx,ms1_spec.scan_num,0,0,-1,prec_mz,prec_rt,*np.zeros(len(names)-7)]]
    else:
        output = [[0,spec_idx,ms1_spec.scan_num,0,0,prec_mz,prec_rt,*np.zeros(len(names)-7)]]
    
    if len(non_zero_coeffs)>0:
        # Decode int32 frag codes to strings only at output time
        frag_names = [decode_frag_names(codes) for codes in frag_name_codes]
        decoy_frag_names = [decode_frag_names(codes) for codes in decoy_frag_name_codes]

        lib_spec_ids = [ref_pep_cand[i] for i in range(len(ref_pep_cand)) if lib_coefficients[i] != 0]
        if decoy:
            updated_decoy_offset = int(max(ref_sparse_col_indices))+1 if len(ref_sparse_col_indices)>0 else 0
            decoy_spec_ids = [decoy_pep_cand[i] for i in range(len(decoy_pep_cand)) if lib_coefficients[updated_decoy_offset+i] != 0]

            all_spec_ids = lib_spec_ids+decoy_spec_ids
            all_features = np.concatenate((features,decoy_features))
            all_ms2_frags = [[";".join(map(str,j)) for j in i] for i in zip(frag_names+decoy_frag_names,
                                                                            frag_errors+decoy_frag_errors,
                                                                            lib_frag_mz+decoy_lib_frag_mz,
                                                                            lib_frag_int+decoy_lib_frag_int,
                                                                            obs_frag_int+decoy_obs_frag_int,
                                                                            unique_frags+unique_frags_decoy,
                                                                            unique_frags_int+unique_frags_int_decoy)]
            
            
        else:
            all_spec_ids = lib_spec_ids
            all_features = features
            all_ms2_frags = [[";".join(map(str,j)) for j in i] for i in zip(frag_names,
                                                                            frag_errors,
                                                                            lib_frag_mz,
                                                                            lib_frag_int,
                                                                            obs_frag_int,
                                                                            unique_frags,
                                                                            unique_frags_int)]
            
        return_prot = config.protein_column in library[next(iter(library))]
        
        if config.args.timeplex:
            output = [[non_zero_coeffs[i],
                       spec_idx,
                       ms1_spec.scan_num,
                       all_spec_ids[i][0],
                       all_spec_ids[i][1],
                       all_spec_ids[i][2],
                       prec_mz,
                       prec_rt,
                       *all_features[j],
                       *all_ms2_frags[j],
                       config.args.mzml,
                       library[(re.sub("Decoy_","",all_spec_ids[i][0]),all_spec_ids[i][1],all_spec_ids[i][2])][config.protein_column] if return_prot else "NA" ]
                       for i,j in zip(range(len(non_zero_coeffs)),non_zero_coeffs_idxs)]
        
        else:
            
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
    atleast_m=config.atleast_m
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
        sparse_row_indices = stats.rankdata(sparse_row_indices,method="dense").astype(int)-1
        
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
            rel_error = ms1_error(np.array(filtered_dino.mz), rt_mz[window_idxs[ref_peaks_in_dia],1], tol=ms1_tol)
        else:
            rel_error = np.zeros(len(ref_peaks_in_dia))
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
#     atleast_m=config.atleast_m

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