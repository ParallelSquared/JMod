"""Fragment-ion correlation features for XGBoost scoring.

Computes pairwise Pearson correlations between a precursor's fragment ion
intensities across MS2 scans that cover its apex within a ``fwhm_multiplier *
FWHM`` half-width RT window, and summarizes them as scalar features. One row
per row of ``fdc``.

Thread-safety
-------------
No module-level mutable state, no shared scratch. Hot paths are
``@nb.njit(cache=False, nogil=True)`` kernels that release the GIL so callers can
parallelize over rows or files under free-threaded Python.

Setup wraps the scans' existing peak arrays in numba typed lists rather than
concatenating them into flat buffers. The kernels read one scan at a time, so
per-scan contiguity is all they need, and the peaks are never copied: a
~1e9-peak timsTOF .d would otherwise need a second ~18 GB of buffers alongside
the originals. Nothing is mutated, so the per-row work below remains order- and
thread-independent, and setup is safe to run concurrently.
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

from __future__ import annotations

import numpy as np
import numba as nb
import pandas as pd
import tqdm

from src.logger import logger
from src.utils.frag_encoding import is_isotope
from src.utils.io.load_files import (
    PEAK_MZ_DTYPE, PEAK_INT_DTYPE, PEAK_MOB_DTYPE,
)


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
    # Same correlation taken across the ION MOBILITY axis instead of RT: at the
    # apex scan only, each fragment's matched intensity is binned by its peaks'
    # own 1/K0 across the band, giving a mobility profile per fragment.  A real
    # fragment co-migrates with the precursor -- same profile shape -- while an
    # interferer at another mobility does not.  Complements the RT features
    # rather than replacing them.
    "im_prec_frag_mean",       # mean Pearson r(precursor, fragment) over the IM profile
    "im_prec_frag_max",        # max r
    # Full mirror of the RT pairwise block above, computed on the IM profile
    # matrix by the same kernel. Slot order matches _pairwise_corr_and_features_numba.
    "im_mean_frag_corr",
    "im_median_frag_corr",
    "im_max_frag_corr",
    "im_min_frag_corr",
    "im_std_frag_corr",
    "im_mean_top3_frag_corr",
    "im_frac_corr_above_0p5",
    "im_mean_frag_mean_corr",
    *[f"im_top{i}_frag_mean_corr" for i in range(1, 11)],
    *[f"im_top{i}_frag_sum_corr" for i in range(1, 11)],
    *[f"im_top{i}_pair_corr" for i in range(1, 11)],
)

# Explicit output slots. These were previously written by negative index, which
# silently breaks the moment any column is appended.
_IDX_PREC_MEAN = FEATURE_COLUMNS.index("mean_prec_frag_corr")
_IDX_PREC_MAX = FEATURE_COLUMNS.index("max_prec_frag_corr")
_IDX_IM_MEAN = FEATURE_COLUMNS.index("im_prec_frag_mean")
_IDX_IM_MAX = FEATURE_COLUMNS.index("im_prec_frag_max")
# Start of the 38-slot IM pairwise block (same layout as the RT block).
_IDX_IM_PAIRWISE = FEATURE_COLUMNS.index("im_mean_frag_corr")

# Every IM-derived column is a suffix of FEATURE_COLUMNS, starting here.
_N_NON_IM_FEATURES = _IDX_IM_MEAN

# Gaussian width on the mobility axis for the grid-free kernel, as a fraction
# of the fitted IM tolerance. opt_im_precision is the 99% coverage of the Laplace
# core, b*ln(100), and the core SD is b*sqrt(2), so sigma = sqrt(2)/ln(100).
# Expressed as a fraction so it tracks the fitted tolerance instead of being a
# second hand-set constant.
_IM_KERNEL_SIGMA_FRAC = 1.4142135623730951 / 4.605170185988092

_N_RANKED = 10  # how many ranked slots per category

# Number of kernel output slots (everything after n_corr_scans and n_corr_frags)
_N_KERNEL_FEATURES = len(FEATURE_COLUMNS) - 2

# Ranked slots default to -1 (missing); the scalar features stay NaN.
_IDX_RANKED_RT = FEATURE_COLUMNS.index("top1_frag_mean_corr")
_IDX_RANKED_IM = FEATURE_COLUMNS.index("im_top1_frag_mean_corr")
_N_RANKED_SLOTS = 3 * _N_RANKED


def feature_columns(use_im):
    """The columns emitted for a run with or without ion mobility."""
    return list(FEATURE_COLUMNS if use_im else FEATURE_COLUMNS[:_N_NON_IM_FEATURES])


# ---------------------------------------------------------------------------
# Python helpers (run once per call, outside the per-row hot loop)
# ---------------------------------------------------------------------------


def _scan_metadata(ms2scans):
    """Per-scan RT and isolation-window bounds.

    Three arrays of length n_scans, so this is negligible next to the peak data.

    Returns
    -------
    scan_rt : np.ndarray (float64, shape n_scans)
    scan_win_lo, scan_win_hi : np.ndarray (float64, shape n_scans)
    """
    n = len(ms2scans)
    scan_rt = np.empty(n, dtype=np.float64)
    scan_win_lo = np.empty(n, dtype=np.float64)
    scan_win_hi = np.empty(n, dtype=np.float64)
    for i, s in enumerate(ms2scans):
        scan_rt[i] = float(s.RT)
        iw = s.isolation_window
        target = float(iw["isolation window target m/z"])
        lower = float(iw["isolation window lower offset"])
        upper = float(iw["isolation window upper offset"])
        scan_win_lo[i] = target - lower
        scan_win_hi[i] = target + upper

    return scan_rt, scan_win_lo, scan_win_hi


def _build_peak_lists(ms2scans, need_mob):
    """Wrap the existing per-scan peak arrays in numba typed lists.

    The kernels only ever read one scan's peaks at a time, so they need
    per-scan contiguity -- which the scans already have. Handing them typed
    lists therefore costs no per-peak memory at all, where the previous
    CSR-flattening built a second full copy of the run alongside the first: for
    a ~1e9-peak timsTOF .d that was ~18 GB extra, and on Windows (no
    overcommit) it failed outright at np.empty.

    All three ingest paths now normalize to the peak dtypes at read time, so
    ``ascontiguousarray`` is a no-op here and the lists are pure views. It stays
    as a cheap guard: numba needs one concrete element type per list, and a
    single stray dtype would otherwise fail to compile deep inside a kernel.
    """
    peaks_mz = nb.typed.List()
    peaks_int = nb.typed.List()
    peaks_mob = nb.typed.List()

    for s in ms2scans:
        peaks_mz.append(np.ascontiguousarray(s.mz, dtype=PEAK_MZ_DTYPE))
        peaks_int.append(np.ascontiguousarray(s.intens, dtype=PEAK_INT_DTYPE))
        if need_mob:
            mob = getattr(s, "mobility", None)
            if mob is None:
                mob = np.full(len(s.mz), np.nan, dtype=PEAK_MOB_DTYPE)
            peaks_mob.append(np.ascontiguousarray(mob, dtype=PEAK_MOB_DTYPE))

    if not need_mob:
        # Never indexed (the IM kernels only run when mobility exists), but the
        # list still needs a concrete element type for numba to compile against.
        peaks_mob.append(np.empty(0, dtype=PEAK_MOB_DTYPE))

    return peaks_mz, peaks_int, peaks_mob


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
    peaks_mz,
    peaks_int,
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
        scan_mz = peaks_mz[s]
        scan_in = peaks_int[s]
        n_peaks = scan_mz.shape[0]
        if n_peaks == 0:
            for j in range(n_frags):
                out_matrix[i, j] = 0.0
            continue
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
def _match_and_fill_im_numba(
    peaks_mz,
    peaks_int,
    peaks_mob,
    scan_idx,
    n_scans_used,
    frag_mz,
    ppm_tol,
    im_tol,
    prec_im_in,
    out_matrix,
    out_mob,
):
    """Like ``_match_and_fill_numba`` but IM-gated, summing co-matching peaks.

    ``out_matrix[i, j]`` is the SUMMED intensity of every peak in scan
    ``scan_idx[i]`` that lies within ``ppm_tol`` (m/z) of ``frag_mz[j]`` AND
    within ``im_tol`` (1/K0) of the reference precursor IM. Multiple peaks that
    fall inside both tolerances (e.g. an unresolved IM doublet) are added
    together rather than one being chosen — mirroring the 2D-bin summation the
    spectral-fitting stage performs.

    The reference IM is ``prec_im_in`` when finite; otherwise it is derived as
    the median mobility of the nearest-m/z matched peak per (scan, fragment)
    (``out_mob`` is scratch for that derivation). When it cannot be determined
    (no finite mobilities), matching falls back to m/z only — no IM gate.

    Returns the reference IM used (NaN when it could not be determined).
    """
    n_frags = frag_mz.shape[0]

    # ── Reference IM: caller-provided if finite, else median of nearest-m/z
    #    matched mobilities. NaN => fall back to m/z-only (no IM gate). ──
    if prec_im_in == prec_im_in:
        prec_im = prec_im_in
    else:
        for i in range(n_scans_used):
            for j in range(n_frags):
                out_mob[i, j] = np.nan
        n_matched = 0
        for i in range(n_scans_used):
            s = scan_idx[i]
            scan_mz = peaks_mz[s]
            scan_mob = peaks_mob[s]
            n_peaks = scan_mz.shape[0]
            if n_peaks == 0:
                continue
            for j in range(n_frags):
                q = frag_mz[j]
                pos = np.searchsorted(scan_mz, q)
                if pos == 0:
                    cand = 0
                elif pos >= n_peaks:
                    cand = n_peaks - 1
                else:
                    if q - scan_mz[pos - 1] <= scan_mz[pos] - q:
                        cand = pos - 1
                    else:
                        cand = pos
                diff = scan_mz[cand] - q
                if diff < 0.0:
                    diff = -diff
                if diff <= q * ppm_tol:
                    m = scan_mob[cand]
                    if m == m:
                        out_mob[i, j] = m
                        n_matched += 1
        if n_matched == 0:
            prec_im = np.nan
        else:
            mob_buf = np.empty(n_matched, dtype=np.float64)
            c = 0
            for i in range(n_scans_used):
                for j in range(n_frags):
                    m = out_mob[i, j]
                    if m == m:
                        mob_buf[c] = m
                        c += 1
            prec_im = np.median(mob_buf[:c])

    gate = prec_im == prec_im  # finite reference => apply IM gate

    # ── Summation pass: sum all peaks within m/z tol (and, if gating, im_tol) ──
    for i in range(n_scans_used):
        s = scan_idx[i]
        scan_mz = peaks_mz[s]
        scan_in = peaks_int[s]
        scan_mob = peaks_mob[s]
        n_peaks = scan_mz.shape[0]
        if n_peaks == 0:
            for j in range(n_frags):
                out_matrix[i, j] = 0.0
            continue
        for j in range(n_frags):
            q = frag_mz[j]
            tol = q * ppm_tol
            lo_pos = np.searchsorted(scan_mz, q - tol)
            hi_pos = np.searchsorted(scan_mz, q + tol, side="right")
            acc = 0.0
            for p in range(lo_pos, hi_pos):
                if gate:
                    mob = scan_mob[p]
                    if mob == mob and abs(mob - prec_im) <= im_tol:
                        acc += scan_in[p]
                else:
                    acc += scan_in[p]
            out_matrix[i, j] = acc
    return prec_im


@nb.njit(cache=False, nogil=True)
def _summarize_pairs_numba(pair_vals, valid_pairs, per_col_sum_r, per_col_cnt_r,
                           n_cols, out_features):
    """Turn a set of pairwise similarity values into the 38-slot feature block.

    Split out of _pairwise_corr_and_features_numba so the Pearson path and the
    kernel-similarity path produce byte-identical summaries from the same code.
    Slot layout is unchanged; see that function's docstring.
    """
    nan = np.nan
    n_out = out_features.shape[0]
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

    _summarize_pairs_numba(pair_vals, valid_pairs, per_col_sum_r, per_col_cnt_r,
                           n_cols, out_features)


@nb.njit(cache=False, nogil=True)
def _kernel_pairwise_numba(peaks_mz, peaks_int, peaks_mob,
                           scans, n_scans, queries, mz_tol, sigma_im,
                           out_features, prec_out):
    """Grid-free RT x IM similarity between ions, and its feature block.

    No binning.  Fragments of one precursor are recorded in the *same* MS2
    spectra, so their RT coordinates coincide exactly -- peaks are paired only
    within a scan, with no RT kernel.  On the mobility axis, where two peaks of
    the same ion differ by measurement jitter, pairs are weighted by a Gaussian
    of width ``sigma_im``:

        <A,B> = sum_scans sum_{i in A(s)} sum_{j in B(s)} wi*wj*exp(-dIM^2 / 2s^2)
        sim   = <A,B> / sqrt(<A,A> * <B,B>)

    That is a cosine similarity in the kernel's feature space: bounded [0,1],
    invariant to intensity scale, and free of the bin-edge artefacts a grid
    introduces (two peaks 1e-4 apart can straddle a bin boundary while two
    0.005 apart share one).

    ``queries[0]`` is the precursor; ``queries[1:]`` the fragments.  Pairwise
    features are computed over the fragments only; ``prec_out`` receives the
    (mean, max) precursor-vs-fragment similarity.
    """
    n_q = queries.shape[0]
    gram = np.zeros((n_q, n_q), dtype=np.float64)
    inv2s2 = 1.0 / (2.0 * sigma_im * sigma_im)
    p0 = np.empty(n_q, dtype=np.int64)
    p1 = np.empty(n_q, dtype=np.int64)

    for si in range(n_scans):
        sc = scans[si]
        scan_mz = peaks_mz[sc]
        scan_int = peaks_int[sc]
        scan_mob = peaks_mob[sc]
        if scan_mz.shape[0] == 0:
            continue
        for j in range(n_q):
            q = queries[j]
            tol = q * mz_tol
            p0[j] = np.searchsorted(scan_mz, q - tol)
            p1[j] = np.searchsorted(scan_mz, q + tol, side="right")
        for a in range(n_q):
            if p1[a] <= p0[a]:
                continue
            for b in range(a, n_q):
                if p1[b] <= p0[b]:
                    continue
                acc = 0.0
                for i in range(p0[a], p1[a]):
                    mi = scan_mob[i]
                    wi = scan_int[i]
                    for k in range(p0[b], p1[b]):
                        d = mi - scan_mob[k]
                        acc += wi * scan_int[k] * np.exp(-d * d * inv2s2)
                gram[a, b] += acc
                if b != a:
                    gram[b, a] += acc

    n_frag = n_q - 1
    max_pairs = n_frag * (n_frag - 1) // 2
    if max_pairs < 1:
        max_pairs = 1
    pair_vals = np.empty(max_pairs, dtype=np.float64)
    per_col_sum_r = np.zeros(n_frag, dtype=np.float64)
    per_col_cnt_r = np.zeros(n_frag, dtype=np.int64)
    valid_pairs = 0
    for a in range(1, n_q):
        if gram[a, a] <= 0.0:
            continue
        for b in range(a + 1, n_q):
            if gram[b, b] <= 0.0:
                continue
            r = gram[a, b] / np.sqrt(gram[a, a] * gram[b, b])
            if abs(r - 1.0) < 1e-12:
                continue  # same peak matched by two library fragments
            pair_vals[valid_pairs] = r
            valid_pairs += 1
            per_col_sum_r[a - 1] += r
            per_col_cnt_r[a - 1] += 1
            per_col_sum_r[b - 1] += r
            per_col_cnt_r[b - 1] += 1

    _summarize_pairs_numba(pair_vals, valid_pairs, per_col_sum_r, per_col_cnt_r,
                           n_frag, out_features)

    # precursor (index 0) against each fragment
    tot = 0.0
    cnt = 0
    mx = -1.0
    if gram[0, 0] > 0.0:
        for b in range(1, n_q):
            if gram[b, b] <= 0.0:
                continue
            r = gram[0, b] / np.sqrt(gram[0, 0] * gram[b, b])
            tot += r
            cnt += 1
            if r > mx:
                mx = r
    if cnt > 0:
        prec_out[0] = tot / cnt
        prec_out[1] = mx
    else:
        prec_out[0] = np.nan
        prec_out[1] = np.nan


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
    im_tol=0.0,
    prec_im=None,
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
    prec_im : array-like or None, default None
        Per-row precursor ion mobility (1/K0), positionally aligned with ``fdc``,
        carried through from the spectral-fitting stage (median of matched
        fragment mobilities). When finite it is used directly as the IM gate
        center; NaN entries fall back to an in-kernel median of the matched
        mobilities. Ignored for non-IM data.
    min_obs : int, default 3
        Fragment columns with fewer than this many non-zero entries are dropped
        before correlation.

    Returns
    -------
    pandas.DataFrame
        Indexed to match ``fdc.index``, with the columns :func:`feature_columns`
        gives for this run. Rows with ``coeff <= 0``, no covering scans, or fewer
        than two surviving fragments receive NaN for the correlation features.
        Does not mutate ``fdc``, ``library`` or ``spectra``.
    """
    n_rows = len(fdc)

    def _empty(use_im):
        cols = feature_columns(use_im)
        arr = np.full((n_rows, len(cols)), np.nan, dtype=np.float64)
        arr[:, _IDX_RANKED_RT:_IDX_RANKED_RT + _N_RANKED_SLOTS] = -1.0
        if use_im:
            arr[:, _IDX_RANKED_IM:_IDX_RANKED_IM + _N_RANKED_SLOTS] = -1.0
        return pd.DataFrame(arr, index=fdc.index, columns=cols)

    # Nothing to correlate: bail before touching spectra.
    if n_rows == 0 or fwhm is None or not np.isfinite(fwhm) or fwhm <= 0:
        return _empty(False)

    ms2scans = spectra.ms2scans
    if len(ms2scans) == 0:
        return _empty(False)

    # IM-aware matching only when the data has ion mobility and a tolerance is set.
    _use_im = im_tol > 0.0 and getattr(ms2scans[0], "mobility", None) is not None

    columns = feature_columns(_use_im)
    out = np.full((n_rows, len(columns)), np.nan, dtype=np.float64)
    out[:, _IDX_RANKED_RT:_IDX_RANKED_RT + _N_RANKED_SLOTS] = -1.0
    if _use_im:
        out[:, _IDX_RANKED_IM:_IDX_RANKED_IM + _N_RANKED_SLOTS] = -1.0

    logger.info("Computing fragment-ion correlation features")
    scan_rt, scan_win_lo, scan_win_hi = _scan_metadata(ms2scans)
    (win_lo, win_hi, win_scan_offsets,
     win_scan_idx, win_scan_rt) = _build_window_csr(scan_rt, scan_win_lo, scan_win_hi)

    peaks_mz, peaks_int, peaks_mob = _build_peak_lists(ms2scans, _use_im)
    if _use_im:
        logger.info(f"IM-aware fragment correlations (im_tol={im_tol:.5f})")
        # Per-scan IM-bin bounds (parallel to ms2scans / scan indices), used to
        # restrict each precursor's XIC to a single IM bin across RT.
        scan_im_lo = np.array(
            [s.im_lo if getattr(s, "im_lo", None) is not None else np.nan
             for s in ms2scans], dtype=np.float64)
        scan_im_hi = np.array(
            [s.im_hi if getattr(s, "im_hi", None) is not None else np.nan
             for s in ms2scans], dtype=np.float64)

        # Precursor IM carried through from spectral fitting (per fdc row). When
        # absent, prec_im stays NaN and _match_and_fill_im_numba derives it
        # in-kernel from the matched fragment mobilities.
        prec_im_arr = None if prec_im is None else np.asarray(prec_im, dtype=np.float64)

    rt_halfwidth = float(fwhm_multiplier) * float(fwhm)

    # Bound for the scan-index scratch buffer: a precursor cannot match more
    # scans than there are MS2 scans total.
    scan_buf = np.empty(len(ms2scans), dtype=np.int64)
    # Derived from the precursor slot's position, not the total column count, so
    # appending feature columns cannot shift it.
    _n_pairwise_feats = _IDX_PREC_MEAN - 2
    feat_scratch = np.empty(_n_pairwise_feats, dtype=np.float64)
    im_feat_scratch = np.empty(_n_pairwise_feats, dtype=np.float64)
    im_prec_scratch = np.empty(2, dtype=np.float64)
    prec_corr_mean = np.empty(1, dtype=np.float64)
    prec_corr_max = np.empty(1, dtype=np.float64)

    seqs = fdc["seq"].to_numpy()
    zs = fdc["z"].to_numpy()
    rts = fdc["rt"].to_numpy(dtype=np.float64)
    coeffs = fdc["coeff"].to_numpy(dtype=np.float64)

    key_to_idx = library.key_to_idx
    spec_mz = library.spectrum_mz
    spec_off = library.spectrum_offsets
    spec_len = library.spectrum_lengths
    frag_names = library.frag_names_data
    prec_mz_arr = library.prec_mz

    for row in tqdm.tqdm(range(n_rows)):
        if not (coeffs[row] > 0) or not np.isfinite(rts[row]):
            continue
        key = (seqs[row], int(zs[row]))
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

        if _use_im:
            # Precursor IM comes from the fitting stage (carried on fdc); no need
            # to re-derive it here. NaN => in-kernel median fallback at fill time.
            prec_im = float(prec_im_arr[row]) if prec_im_arr is not None else np.nan
            # Pass 2: restrict to ONE covering scan per RT whose IM bin contains
            # prec_im (nearest bin center) — a clean single-bin XIC across RT.
            cov = scan_buf[:n_scans]
            if np.isfinite(prec_im):
                contains = (scan_im_lo[cov] <= prec_im) & (prec_im <= scan_im_hi[cov])
                cov = cov[contains]
            if cov.shape[0] == 0:
                use_scans = scan_buf[:0]
            else:
                centers = 0.5 * (scan_im_lo[cov] + scan_im_hi[cov])
                dist = np.abs(centers - prec_im) if np.isfinite(prec_im) else np.zeros(cov.shape[0])
                order = np.lexsort((dist, scan_rt[cov]))  # by RT, then nearest bin
                _, first = np.unique(scan_rt[cov][order], return_index=True)
                use_scans = np.ascontiguousarray(cov[order][first])
            out[row, 0] = float(use_scans.shape[0])
            if use_scans.shape[0] == 0:
                continue
            n_use = use_scans.shape[0]
            matrix = np.empty((n_use, n_frags_kept), dtype=np.float64)
            _mob_scratch = np.empty((n_use, n_frags_kept), dtype=np.float64)
            # Gate matched peaks by the calibrated IM tolerance: keep a fragment's
            # intensity only when its peak mobility is within `im_tol` of prec_im.
            # Cheap here because use_scans holds one scan per RT (not all bands).
            _match_and_fill_im_numba(
                peaks_mz, peaks_int, peaks_mob,
                use_scans, n_use, frag_mz_kept, float(mz_tol), float(im_tol),
                prec_im, matrix, _mob_scratch,
            )
        else:
            use_scans = scan_buf[:n_scans]
            n_use = n_scans
            matrix = np.empty((n_use, n_frags_kept), dtype=np.float64)
            _match_and_fill_numba(
                peaks_mz, peaks_int,
                use_scans, n_use, frag_mz_kept, float(mz_tol), matrix,
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
        # intensity from the SAME single-bin scan set and correlate with fragments.
        prec_mz_arr_q = np.array([prec_mz], dtype=np.float64)
        prec_matrix = np.empty((n_use, 1), dtype=np.float64)
        if _use_im:
            # Same IM gate as the fragment matrix, so the precursor trace is
            # measured at the fragment-derived mobility.
            _prec_mob_scratch = np.empty((n_use, 1), dtype=np.float64)
            _match_and_fill_im_numba(
                peaks_mz, peaks_int, peaks_mob,
                use_scans, n_use, prec_mz_arr_q, float(mz_tol), float(im_tol),
                prec_im, prec_matrix, _prec_mob_scratch,
            )
        else:
            _match_and_fill_numba(
                peaks_mz, peaks_int,
                use_scans, n_use, prec_mz_arr_q, float(mz_tol), prec_matrix,
            )
        _precursor_frag_corr_numba(
            prec_matrix[:, 0], sub_matrix, prec_corr_mean, prec_corr_max,
        )
        out[row, _IDX_PREC_MEAN] = prec_corr_mean[0]
        out[row, _IDX_PREC_MAX] = prec_corr_max[0]

        # ---- same correlation across the mobility axis, at the apex scan ----
        if _use_im and n_use > 0:
            # Mobility axis is pinned to the apex scan's band: use_scans may pick
            # different bands at different RTs, and profiles are only addable on a
            # common axis.  Peaks falling outside it are clamped to the edge bins.
            apex_pos = int(np.argmin(np.abs(scan_rt[use_scans] - float(rts[row]))))
            apex_scan = int(use_scans[apex_pos])
            band_lo = float(scan_im_lo[apex_scan])
            band_hi = float(scan_im_hi[apex_scan])
            if np.isfinite(band_lo) and np.isfinite(band_hi) and band_hi > band_lo:
                kept_mz = np.ascontiguousarray(frag_mz_kept[keep_cols])
                # Summed across every covering scan rather than the apex alone:
                # one scan gives ~8 sparse bins, and summing over the elution
                # fills them without blurring the mobility axis (a precursor's
                # 1/K0 does not drift with RT).
                # Joint RT x IM grid, neither axis collapsed. Only scans from
                # the apex scan's own band contribute: a precursor's 1/K0 does
                # not drift with RT, so a neighbouring band (picked at an RT
                # where this band held no peaks) is offset by half a width and
                # would land its peaks in the wrong mobility rows.
                _same_band = ((scan_im_lo[use_scans] == band_lo)
                              & (scan_im_hi[use_scans] == band_hi))
                _band_scans = np.ascontiguousarray(use_scans[_same_band])
                _n_band = int(_band_scans.shape[0])
                if _n_band == 0:
                    continue
                # queries[0] is the precursor so the kernel can produce both the
                # pairwise block and the precursor-vs-fragment pair in one pass
                # over the peaks.
                _q = np.empty(kept_mz.shape[0] + 1, dtype=np.float64)
                _q[0] = prec_mz
                _q[1:] = kept_mz
                _sigma_im = _IM_KERNEL_SIGMA_FRAC * float(im_tol)
                if _sigma_im > 0.0:
                    _kernel_pairwise_numba(
                        peaks_mz, peaks_int, peaks_mob,
                        _band_scans, _n_band, _q, float(mz_tol), _sigma_im,
                        im_feat_scratch, im_prec_scratch,
                    )
                    out[row, _IDX_IM_MEAN] = im_prec_scratch[0]
                    out[row, _IDX_IM_MAX] = im_prec_scratch[1]
                    out[row, _IDX_IM_PAIRWISE:_IDX_IM_PAIRWISE + _n_pairwise_feats] = \
                        im_feat_scratch

    return pd.DataFrame(out, index=fdc.index, columns=columns)
