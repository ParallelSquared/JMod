"""Fragment-ion correlation features for XGBoost scoring.

Computes pairwise Pearson correlations between a precursor's fragment ion
intensities across MS2 scans that cover its apex within a ``fwhm_multiplier *
FWHM`` half-width RT window, and summarizes them as scalar features. One row
per row of ``fdc``.

Thread-safety
-------------
Pure function: no module-level mutable state, no shared scratch, no caches.
Hot paths are ``@nb.njit(cache=False, nogil=True)`` kernels that release the
GIL so callers can parallelize over rows or files under free-threaded Python.
"""

from __future__ import annotations

import numpy as np
import numba as nb
import pandas as pd
import tqdm

from src.logger import logger
from src.utils.frag_encoding import is_isotope


FEATURE_COLUMNS = (
    "n_corr_scans",
    "n_corr_frags",
    "mean_frag_corr",
    "median_frag_corr",
    "max_frag_corr",
    "min_frag_corr",
    "std_frag_corr",
    "mean_top3_frag_corr",
    "frac_corr_above_0p5",
    "mean_frag_mean_corr",
    # Top-10 per-fragment mean correlations (sorted descending; -1 = missing)
    *[f"top{i}_frag_mean_corr" for i in range(1, 11)],
    # Top-10 per-fragment sum correlations (same rank order as mean; -1 = missing)
    *[f"top{i}_frag_sum_corr" for i in range(1, 11)],
    # Top-10 pairwise correlations (sorted descending; -1 = missing)
    *[f"top{i}_pair_corr" for i in range(1, 11)],
    # Precursor–fragment correlation (unfragmented precursor in MS2 vs fragment traces)
    "mean_prec_frag_corr",     # mean Pearson r between precursor trace and each fragment
    "max_prec_frag_corr",      # max r
)

_N_RANKED = 10  # how many ranked slots per category

# Number of kernel output slots (everything after n_corr_scans and n_corr_frags)
_N_KERNEL_FEATURES = len(FEATURE_COLUMNS) - 2


# ---------------------------------------------------------------------------
# Python helpers (run once per call, outside the per-row hot loop)
# ---------------------------------------------------------------------------


def _flatten_ms2_peaks(ms2scans):
    """Flatten per-scan (mz, intens) arrays into CSR-style flat arrays.

    Returns
    -------
    peak_mz_flat, peak_int_flat : np.ndarray (float64)
        Concatenated peak arrays in scan order.
    peak_offsets : np.ndarray (int64, shape n_scans+1)
        Scan ``i``'s peaks live in ``peak_mz_flat[offsets[i]:offsets[i+1]]``.
    scan_rt : np.ndarray (float64, shape n_scans)
    scan_win_lo, scan_win_hi : np.ndarray (float64, shape n_scans)
        Isolation-window bounds per scan.
    """
    n = len(ms2scans)
    lengths = np.empty(n, dtype=np.int64)
    scan_rt = np.empty(n, dtype=np.float64)
    scan_win_lo = np.empty(n, dtype=np.float64)
    scan_win_hi = np.empty(n, dtype=np.float64)
    for i, s in enumerate(ms2scans):
        lengths[i] = int(len(s.mz))
        scan_rt[i] = float(s.RT)
        iw = s.isolation_window
        target = float(iw["isolation window target m/z"])
        lower = float(iw["isolation window lower offset"])
        upper = float(iw["isolation window upper offset"])
        scan_win_lo[i] = target - lower
        scan_win_hi[i] = target + upper

    peak_offsets = np.empty(n + 1, dtype=np.int64)
    peak_offsets[0] = 0
    if n > 0:
        np.cumsum(lengths, out=peak_offsets[1:])

    total = int(peak_offsets[-1])
    peak_mz_flat = np.empty(total, dtype=np.float64)
    peak_int_flat = np.empty(total, dtype=np.float64)
    for i, s in enumerate(ms2scans):
        off = int(peak_offsets[i])
        ln = int(lengths[i])
        if ln:
            peak_mz_flat[off:off + ln] = s.mz
            peak_int_flat[off:off + ln] = s.intens

    return peak_mz_flat, peak_int_flat, peak_offsets, scan_rt, scan_win_lo, scan_win_hi


def _build_window_csr(scan_rt, scan_win_lo, scan_win_hi):
    """Group scan indices by isolation window ``(lo, hi)`` and CSR-sort by RT.

    Returns
    -------
    win_lo, win_hi : np.ndarray (float64, shape n_windows)
    win_scan_offsets : np.ndarray (int64, shape n_windows+1)
    win_scan_idx : np.ndarray (int64, shape n_scans)
        Original scan indices, grouped by window and sorted by RT within group.
    win_scan_rt : np.ndarray (float64, shape n_scans)
        Parallel to ``win_scan_idx``, the RTs in sorted order per group.
    """
    n = scan_rt.shape[0]
    groups = {}
    for i in range(n):
        key = (float(scan_win_lo[i]), float(scan_win_hi[i]))
        bucket = groups.get(key)
        if bucket is None:
            bucket = []
            groups[key] = bucket
        bucket.append(i)

    n_windows = len(groups)
    win_lo = np.empty(n_windows, dtype=np.float64)
    win_hi = np.empty(n_windows, dtype=np.float64)
    win_scan_offsets = np.empty(n_windows + 1, dtype=np.int64)
    win_scan_idx = np.empty(n, dtype=np.int64)
    win_scan_rt = np.empty(n, dtype=np.float64)

    win_scan_offsets[0] = 0
    cursor = 0
    for w, (key, idx_list) in enumerate(groups.items()):
        win_lo[w] = key[0]
        win_hi[w] = key[1]
        idx_arr = np.asarray(idx_list, dtype=np.int64)
        rts = scan_rt[idx_arr]
        order = np.argsort(rts)
        sz = idx_arr.shape[0]
        win_scan_idx[cursor:cursor + sz] = idx_arr[order]
        win_scan_rt[cursor:cursor + sz] = rts[order]
        cursor += sz
        win_scan_offsets[w + 1] = cursor

    return win_lo, win_hi, win_scan_offsets, win_scan_idx, win_scan_rt


# ---------------------------------------------------------------------------
# Numba kernels (nogil, cached)
# ---------------------------------------------------------------------------


@nb.njit(cache=False, nogil=True)
def _covering_scans_numba(
    prec_mz,
    apex_rt,
    rt_halfwidth,
    win_lo,
    win_hi,
    win_scan_offsets,
    win_scan_idx,
    win_scan_rt,
    out_scan_idx,
):
    """Fill ``out_scan_idx`` with MS2 scan indices whose window covers
    ``prec_mz`` and whose RT is within ``[apex - hw, apex + hw]``.

    Returns the number of scans written.
    """
    n_windows = win_lo.shape[0]
    rt_lo = apex_rt - rt_halfwidth
    rt_hi = apex_rt + rt_halfwidth
    count = 0
    out_cap = out_scan_idx.shape[0]
    for w in range(n_windows):
        if prec_mz < win_lo[w] or prec_mz > win_hi[w]:
            continue
        start = win_scan_offsets[w]
        end = win_scan_offsets[w + 1]
        if start == end:
            continue
        rts_slice = win_scan_rt[start:end]
        lo_pos = np.searchsorted(rts_slice, rt_lo, side="left")
        hi_pos = np.searchsorted(rts_slice, rt_hi, side="right")
        for k in range(lo_pos, hi_pos):
            if count == out_cap:
                return count
            out_scan_idx[count] = win_scan_idx[start + k]
            count += 1
    return count


@nb.njit(cache=False, nogil=True)
def _match_and_fill_numba(
    peak_mz_flat,
    peak_int_flat,
    peak_offsets,
    scan_idx,
    n_scans_used,
    frag_mz,
    ppm_tol,
    out_matrix,
):
    """Fill ``out_matrix[i, j]`` with the intensity of the peak in scan
    ``scan_idx[i]`` closest to ``frag_mz[j]`` within a relative PPM tolerance,
    or ``0.0`` if no peak is within tolerance.
    """
    n_frags = frag_mz.shape[0]
    for i in range(n_scans_used):
        s = scan_idx[i]
        lo = peak_offsets[s]
        hi = peak_offsets[s + 1]
        n_peaks = hi - lo
        if n_peaks == 0:
            for j in range(n_frags):
                out_matrix[i, j] = 0.0
            continue
        scan_mz = peak_mz_flat[lo:hi]
        scan_in = peak_int_flat[lo:hi]
        for j in range(n_frags):
            q = frag_mz[j]
            pos = np.searchsorted(scan_mz, q)
            if pos == 0:
                cand = 0
            elif pos >= n_peaks:
                cand = n_peaks - 1
            else:
                left_diff = q - scan_mz[pos - 1]
                right_diff = scan_mz[pos] - q
                if left_diff <= right_diff:
                    cand = pos - 1
                else:
                    cand = pos
            diff = scan_mz[cand] - q
            if diff < 0.0:
                diff = -diff
            if diff <= q * ppm_tol:
                out_matrix[i, j] = scan_in[cand]
            else:
                out_matrix[i, j] = 0.0


@nb.njit(cache=False, nogil=True)
def _pairwise_corr_and_features_numba(matrix, out_features):
    """Compute Pearson correlations between columns of ``matrix`` and write
    summary + ranked features into ``out_features`` (shape (23,), float64).

    Output slot layout (aligns with ``FEATURE_COLUMNS[2:]``):
        0  mean_frag_corr
        1  median_frag_corr
        2  max_frag_corr
        3  min_frag_corr
        4  std_frag_corr
        5  mean_top3_frag_corr
        6  frac_corr_above_0p5
        7  mean_frag_mean_corr
        8-17  top1..top10 per-fragment mean correlation (descending; -1 = missing)
        18-27 top1..top10 per-fragment sum correlation (same rank order; -1 = missing)
        28-37 top1..top10 pairwise correlation (descending; -1 = missing)

    The diagonal is never included: only upper-triangle pairs (a < b) are
    computed, so per-fragment means/sums exclude self-correlation. Zero-variance
    columns are excluded from all pair computations.

    Degenerate inputs (< 2 rows, < 2 columns, all zero-variance columns) write
    NaN to summary slots [0:8] and -1 to ranked slots [8:23].
    """
    nan = np.nan
    n_out = out_features.shape[0]
    n_rows, n_cols = matrix.shape
    if n_rows < 2 or n_cols < 2:
        for k in range(8):
            out_features[k] = nan
        for k in range(8, n_out):
            out_features[k] = -1.0
        return

    means = np.zeros(n_cols, dtype=np.float64)
    for j in range(n_cols):
        s = 0.0
        for i in range(n_rows):
            s += matrix[i, j]
        means[j] = s / n_rows

    norms = np.zeros(n_cols, dtype=np.float64)
    for j in range(n_cols):
        s2 = 0.0
        for i in range(n_rows):
            d = matrix[i, j] - means[j]
            s2 += d * d
        norms[j] = np.sqrt(s2)

    max_pairs = n_cols * (n_cols - 1) // 2
    pair_vals = np.empty(max_pairs, dtype=np.float64)
    per_col_sum_r = np.zeros(n_cols, dtype=np.float64)
    per_col_cnt_r = np.zeros(n_cols, dtype=np.int64)
    valid_pairs = 0
    for a in range(n_cols):
        if norms[a] == 0.0:
            continue
        for b in range(a + 1, n_cols):
            if norms[b] == 0.0:
                continue
            cov = 0.0
            for i in range(n_rows):
                cov += (matrix[i, a] - means[a]) * (matrix[i, b] - means[b])
            r = cov / (norms[a] * norms[b])
            if abs(r - 1.0) < 1e-12:
                continue  # same fragment matched twice — skip
            pair_vals[valid_pairs] = r
            valid_pairs += 1
            per_col_sum_r[a] += r
            per_col_cnt_r[a] += 1
            per_col_sum_r[b] += r
            per_col_cnt_r[b] += 1

    if valid_pairs == 0:
        for k in range(8):
            out_features[k] = nan
        for k in range(8, n_out):
            out_features[k] = -1.0
        return

    # --- Summary stats (slots 0-7) ---
    vals = pair_vals[:valid_pairs]
    total = 0.0
    for v in vals:
        total += v
    mean = total / valid_pairs

    mx = vals[0]
    mn = vals[0]
    for v in vals:
        if v > mx:
            mx = v
        if v < mn:
            mn = v

    var = 0.0
    for v in vals:
        d = v - mean
        var += d * d
    var /= valid_pairs
    std = np.sqrt(var)

    sorted_vals = np.sort(vals)
    if valid_pairs % 2 == 1:
        median = sorted_vals[valid_pairs // 2]
    else:
        median = 0.5 * (sorted_vals[valid_pairs // 2 - 1] + sorted_vals[valid_pairs // 2])

    top_k = 3 if valid_pairs >= 3 else valid_pairs
    top_sum = 0.0
    for k in range(top_k):
        top_sum += sorted_vals[valid_pairs - 1 - k]
    mean_top3 = top_sum / top_k

    above = 0
    for v in vals:
        if v > 0.5:
            above += 1
    frac_above = above / valid_pairs

    per_col_mean_sum = 0.0
    per_col_mean_cnt = 0
    for j in range(n_cols):
        if per_col_cnt_r[j] > 0:
            per_col_mean_sum += per_col_sum_r[j] / per_col_cnt_r[j]
            per_col_mean_cnt += 1
    if per_col_mean_cnt > 0:
        mean_frag_mean = per_col_mean_sum / per_col_mean_cnt
    else:
        mean_frag_mean = nan

    out_features[0] = mean
    out_features[1] = median
    out_features[2] = mx
    out_features[3] = mn
    out_features[4] = std
    out_features[5] = mean_top3
    out_features[6] = frac_above
    out_features[7] = mean_frag_mean

    # --- Top-5 per-fragment mean/sum correlations (slots 8-17) ---
    # Compute per-fragment means, then sort descending.
    n_valid_frags = 0
    frag_means = np.empty(n_cols, dtype=np.float64)
    frag_sums = np.empty(n_cols, dtype=np.float64)
    frag_order = np.empty(n_cols, dtype=np.int64)
    for j in range(n_cols):
        if per_col_cnt_r[j] > 0:
            frag_means[n_valid_frags] = per_col_sum_r[j] / per_col_cnt_r[j]
            frag_sums[n_valid_frags] = per_col_sum_r[j]
            frag_order[n_valid_frags] = n_valid_frags
            n_valid_frags += 1
    # Simple insertion sort descending by mean (n_valid_frags is small).
    for i in range(1, n_valid_frags):
        key_m = frag_means[i]
        key_s = frag_sums[i]
        j = i - 1
        while j >= 0 and frag_means[j] < key_m:
            frag_means[j + 1] = frag_means[j]
            frag_sums[j + 1] = frag_sums[j]
            j -= 1
        frag_means[j + 1] = key_m
        frag_sums[j + 1] = key_s
    for k in range(10):
        if k < n_valid_frags:
            out_features[8 + k] = frag_means[k]
            out_features[18 + k] = frag_sums[k]
        else:
            out_features[8 + k] = -1.0
            out_features[18 + k] = -1.0

    # --- Top-10 pairwise correlations (slots 28-37) ---
    # sorted_vals is ascending; read from the end for descending.
    for k in range(10):
        if k < valid_pairs:
            out_features[28 + k] = sorted_vals[valid_pairs - 1 - k]
        else:
            out_features[28 + k] = -1.0


@nb.njit(cache=False, nogil=True)
def _precursor_frag_corr_numba(prec_trace, frag_matrix, out_mean, out_max):
    """Pearson correlation between the precursor intensity trace and each
    fragment column. Writes mean and max r into scalar outputs.

    ``prec_trace`` shape ``(n_scans,)``, ``frag_matrix`` shape ``(n_scans, n_frags)``.
    Columns with zero variance (precursor or fragment) are skipped.
    """
    nan = np.nan
    n_scans, n_frags = frag_matrix.shape
    if n_scans < 2 or n_frags == 0:
        out_mean[0] = nan
        out_max[0] = nan
        return

    # Precursor mean and norm
    p_mean = 0.0
    for i in range(n_scans):
        p_mean += prec_trace[i]
    p_mean /= n_scans
    p_ss = 0.0
    for i in range(n_scans):
        d = prec_trace[i] - p_mean
        p_ss += d * d
    p_norm = np.sqrt(p_ss)
    if p_norm == 0.0:
        out_mean[0] = nan
        out_max[0] = nan
        return

    r_sum = 0.0
    r_max = -2.0
    n_valid = 0
    for j in range(n_frags):
        f_mean = 0.0
        for i in range(n_scans):
            f_mean += frag_matrix[i, j]
        f_mean /= n_scans
        f_ss = 0.0
        for i in range(n_scans):
            d = frag_matrix[i, j] - f_mean
            f_ss += d * d
        f_norm = np.sqrt(f_ss)
        if f_norm == 0.0:
            continue
        cov = 0.0
        for i in range(n_scans):
            cov += (prec_trace[i] - p_mean) * (frag_matrix[i, j] - f_mean)
        r = cov / (p_norm * f_norm)
        r_sum += r
        if r > r_max:
            r_max = r
        n_valid += 1

    if n_valid > 0:
        out_mean[0] = r_sum / n_valid
        out_max[0] = r_max
    else:
        out_mean[0] = nan
        out_max[0] = nan


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_fragment_correlations(
    spectra,
    library,
    fdc,
    fwhm,
    mz_tol,
    fwhm_multiplier=4.0,
    min_obs=3,
    timeplex=False,
):
    """Compute pairwise fragment-ion correlation features for every row of ``fdc``.

    Parameters
    ----------
    spectra : SpectrumFile
        Must expose ``.ms2scans``, a list of :class:`Spectrum` objects with
        ``.mz``, ``.intens``, ``.RT``, and ``.isolation_window``.
    library : SpectrumLibraryStore
        Unified target+decoy library.
    fdc : pandas.DataFrame
        Top-coeff PSM per (seq, z). Must contain columns ``seq``, ``z``, ``rt``,
        ``coeff``.
    fwhm : float
        Global elution FWHM from ``elution_analysis.calculate_elution_width``.
    mz_tol : float
        Relative PPM tolerance for MS2 peak matching (e.g., 20e-6).
    fwhm_multiplier : float, default 4.0
        RT half-width of the correlation window is ``fwhm_multiplier * fwhm``.
    min_obs : int, default 3
        Fragment columns with fewer than this many non-zero entries are dropped
        before correlation.

    Returns
    -------
    pandas.DataFrame
        Indexed to match ``fdc.index``, one column per entry of
        :data:`FEATURE_COLUMNS`. Rows with ``coeff <= 0``, no covering scans,
        or fewer than two surviving fragments receive NaN for the correlation
        features. Does not mutate any input.
    """
    n_rows = len(fdc)
    n_feat = len(FEATURE_COLUMNS)
    out = np.full((n_rows, n_feat), np.nan, dtype=np.float64)
    # Ranked feature columns default to -1 (missing), not NaN.
    # Last 2 columns (precursor-fragment corr) stay NaN.
    out[:, 10:-2] = -1.0

    if n_rows == 0 or fwhm is None or not np.isfinite(fwhm) or fwhm <= 0:
        return pd.DataFrame(out, index=fdc.index, columns=list(FEATURE_COLUMNS))

    ms2scans = spectra.ms2scans
    if len(ms2scans) == 0:
        return pd.DataFrame(out, index=fdc.index, columns=list(FEATURE_COLUMNS))

    logger.info("Computing fragment-ion correlation features")
    (peak_mz_flat, peak_int_flat, peak_offsets,
     scan_rt, scan_win_lo, scan_win_hi) = _flatten_ms2_peaks(ms2scans)
    (win_lo, win_hi, win_scan_offsets,
     win_scan_idx, win_scan_rt) = _build_window_csr(scan_rt, scan_win_lo, scan_win_hi)

    rt_halfwidth = float(fwhm_multiplier) * float(fwhm)

    # Bound for the scan-index scratch buffer: a precursor cannot match more
    # scans than there are MS2 scans total.
    scan_buf = np.empty(len(ms2scans), dtype=np.int64)
    _n_pairwise_feats = _N_KERNEL_FEATURES - 2  # exclude the 2 precursor-frag slots
    feat_scratch = np.empty(_n_pairwise_feats, dtype=np.float64)
    prec_corr_mean = np.empty(1, dtype=np.float64)
    prec_corr_max = np.empty(1, dtype=np.float64)

    seqs = fdc["seq"].to_numpy()
    zs = fdc["z"].to_numpy()
    rts = fdc["rt"].to_numpy(dtype=np.float64)
    coeffs = fdc["coeff"].to_numpy(dtype=np.float64)
    # In timeplex the library is keyed by (seq, z, channel); match that here so
    # the key_to_idx lookup below resolves instead of silently missing every row.
    channels = fdc["time_channel"].to_numpy() if timeplex else None

    key_to_idx = library.key_to_idx
    spec_mz = library.spectrum_mz
    spec_off = library.spectrum_offsets
    spec_len = library.spectrum_lengths
    frag_names = library.frag_names_data
    prec_mz_arr = library.prec_mz

    for row in tqdm.tqdm(range(n_rows), desc="Fragment correlations", unit="prec"):
        if not (coeffs[row] > 0) or not np.isfinite(rts[row]):
            continue
        key = (seqs[row], int(zs[row]), int(channels[row])) if timeplex else (seqs[row], int(zs[row]))
        lib_idx = key_to_idx.get(key)
        if lib_idx is None:
            continue

        off = int(spec_off[lib_idx])
        ln = int(spec_len[lib_idx])
        if ln == 0:
            continue

        frag_mz = spec_mz[off:off + ln]
        frag_codes = frag_names[off:off + ln]
        keep_mask = ~is_isotope(frag_codes)
        if not keep_mask.any():
            continue
        frag_mz_kept = np.ascontiguousarray(frag_mz[keep_mask], dtype=np.float64)

        prec_mz = float(prec_mz_arr[lib_idx])

        n_scans = _covering_scans_numba(
            prec_mz, float(rts[row]), rt_halfwidth,
            win_lo, win_hi, win_scan_offsets, win_scan_idx, win_scan_rt,
            scan_buf,
        )
        out[row, 0] = float(n_scans)
        if n_scans == 0:
            continue

        n_frags_kept = frag_mz_kept.shape[0]
        matrix = np.empty((n_scans, n_frags_kept), dtype=np.float64)
        _match_and_fill_numba(
            peak_mz_flat, peak_int_flat, peak_offsets,
            scan_buf, n_scans, frag_mz_kept, float(mz_tol), matrix,
        )

        nonzero_counts = np.count_nonzero(matrix, axis=0)
        keep_cols = nonzero_counts >= min_obs
        n_frags_final = int(keep_cols.sum())
        out[row, 1] = float(n_frags_final)
        if n_frags_final < 2:
            continue

        sub_matrix = np.ascontiguousarray(matrix[:, keep_cols])
        _pairwise_corr_and_features_numba(sub_matrix, feat_scratch)
        out[row, 2:2 + _n_pairwise_feats] = feat_scratch

        # Precursor–fragment correlation: extract unfragmented precursor
        # intensity from each covering scan and correlate with fragment traces.
        prec_mz_arr_q = np.array([prec_mz], dtype=np.float64)
        prec_matrix = np.empty((n_scans, 1), dtype=np.float64)
        _match_and_fill_numba(
            peak_mz_flat, peak_int_flat, peak_offsets,
            scan_buf, n_scans, prec_mz_arr_q, float(mz_tol), prec_matrix,
        )
        _precursor_frag_corr_numba(
            prec_matrix[:, 0], sub_matrix, prec_corr_mean, prec_corr_max,
        )
        out[row, -2] = prec_corr_mean[0]
        out[row, -1] = prec_corr_max[0]

    return pd.DataFrame(out, index=fdc.index, columns=list(FEATURE_COLUMNS))
