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
import pandas as pd
import polars as pl
import re
import sys
import os
from itertools import product

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.config as config 

import tqdm
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from src.utils.misc_functions import np_pearson_cor
from scipy.interpolate import interp1d
from src.utils import misc_functions as mf
from src import iso_functions as iso

import warnings
from scipy.signal import find_peaks
import logging

from scipy import optimize
import numba as nb



min_int = 1e-3
import csv
from src.logger import logger
from src.channel_fdr import MIN_MS1_POINTS

def area(x, apex_idx, area_radius=1):
    """Sum of ``x`` over the apex and ``area_radius`` scans on either side"""
    return np.sum(x[max(0, apex_idx-area_radius):apex_idx+area_radius+1])


def argmax_within_radius(fitted, start_pos, radius=None, area_radius=1):
    """Pick the index with the highest ``fitted`` value within
    ``start_pos ± radius`` cycles, capped ``area_radius`` positions in from
    either edge so ``area`` always has its full window. When ``radius`` is
    ``None`` the cap is the full interior of ``fitted``.
    """
    n = len(fitted)
    if n < 2 * area_radius + 1:
        return start_pos
    lo = area_radius
    hi = n - 1 - area_radius
    if radius is not None:
        lo = max(lo, start_pos - radius)
        hi = min(hi, start_pos + radius)
    return lo + int(np.argmax(fitted[lo:hi + 1]))


def walk_to_local_max(fitted, start_pos, apex_jitter=None, area_radius=1):
    """Slide the apex from ``start_pos`` toward a local maximum, one cycle at a
    time, while strictly increasing. Capped ``area_radius`` positions in from
    either edge so the picked apex always has the full window for ``area`` to
    sum. When ``apex_jitter`` is given, the walk additionally stops
    once it has moved ``apex_jitter`` cycles from ``start_pos`` in either
    direction.

    Returns the new apex index. Identical to ``start_pos`` when the voted apex
    is already a local maximum.
    """
    n = len(fitted)
    if n < 2 * area_radius + 1:
        return start_pos
    lo_cap = area_radius
    hi_cap = n - 1 - area_radius
    if apex_jitter is not None:
        lo_cap = max(lo_cap, start_pos - apex_jitter)
        hi_cap = min(hi_cap, start_pos + apex_jitter)
    i = start_pos
    while True:
        left_ok = (i - 1 >= lo_cap)
        right_ok = (i + 1 <= hi_cap)
        left_higher = left_ok and (fitted[i - 1] > fitted[i])
        right_higher = right_ok and (fitted[i + 1] > fitted[i])
        if not (left_higher or right_higher):
            return i
        if left_higher and right_higher:
            # Voted apex is in a saddle — step toward the higher side.
            i = i - 1 if fitted[i - 1] >= fitted[i + 1] else i + 1
        elif left_higher:
            i -= 1
        else:
            i += 1


# The MS1 quant's per-scan outputs, one array per ID (numeric list columns in
# the outputs; None where the ID has no MS1 quant)
MS1_LIST_COLUMNS = (["plexfittrace", "plexfit_ps", "plexfittrace_spec_all", "plexfittrace_all",
                     "plexfittrace_ps_all", "all_ms1_specs"]
                    + [f"all_ms1_iso{i}vals" for i in range(config.num_iso_ms1)])
# Its scalar outputs, in the order they are added to the IDs
MS1_SCALAR_COLUMNS = ["plexfitMS1", "plexfitMS1_p", "plex_Area", "plex_bg", "plex_snr", "ms1_apex_scan",
                      "ms1_cor", "traceproduct", "iso_cor", "MS1_Int", "MS1_Area"]
# Precursor groups per batch: each batch is set up, extracted and fitted before
# the next, which bounds the row-intensity array (pairs x rows)
_GROUP_BATCH = 50000


def ms1_cor_channels(all_spectra,
                     filtered_decoy_coeffs,
                     decoy_coeffs,
                     mz_ppm,
                     rt_tol,
                     tag=None,
                     SILAC=None,
                     timeplex=False,
                     num_iso = None,
                     num_iso_r = None,
                     additional_scans = None,
                     vote_sigma = 1.0,
                     im_tol = 0.0,
                     ):
    """The MS1 quant of every row of *filtered_decoy_coeffs*: a pandas frame on
    its index with the MS1_LIST_COLUMNS and MS1_SCALAR_COLUMNS (NaN / None for a
    row with no fit).

    Each precursor group (all channels of one peptide and charge) is fitted at
    every MS1 scan of a window around its voted apex.  The work runs in three
    stages, each parallel, over batches of groups:

    - setup, per group: the apex vote and window, the channels' isotope
      envelopes, the fit rows (FitRows) and the MS2 coefficients at the window's
      scans -- everything that does not change from scan to scan
    - extraction, per MS1 scan: every (group, scan) pair's observed row
      intensities, into one array; the rows' members are binary-searched in the
      scan's m/z-cut, IM-gated peaks, once per pair
    - fitting, per chunk of groups: each pair's NNLS, then the group's outputs.
      Chunks are taken by whichever thread is free, so one slow fit holds up
      only its own chunk.

    Each isotope's trace is its fit row's observed intensity, the summed signal
    of every peak in the row's window (an isotope sharing a row with another
    channel's isotope has the shared row's), min_int where the row is empty.
    """
    window_half_width = 10

    if tag.name != "no_tag":
        logger.info("Fitting tagged channels together")
    else:
        logger.info("Fitting Precursors Individually")

    ms1_spectra = all_spectra.ms1scans
    ms2_spectra = all_spectra.ms2scans
    ms1_spec_idxs = np.array([i.scan_num for i in ms1_spectra])
    ms2_spec_idxs = np.array([i.scan_num for i in ms2_spectra])
    bottom_of_window, top_of_window = np.array([i.ms1window for i in ms2_spectra]).T
    ms2_rt = np.array([i.RT for i in ms2_spectra])

    if timeplex:
        dc_group_keys = ["seq", "z", "time_channel"]
        fdc_group = filtered_decoy_coeffs.groupby(["untag_seq","z","time_channel"])
    else:
        dc_group_keys = ["seq", "z"]
        fdc_group = filtered_decoy_coeffs.groupby(["untag_seq","z"])

    if isinstance(decoy_coeffs, pl.DataFrame):
        # Production path: ``decoy_coeffs`` is a slim polars frame pre-sorted
        # by ``dc_group_keys`` (see read_output.get_large_prec). Build a {key:
        # (start, length)} offsets dict so per-precursor lookups in
        # get_ms2_vals slice a contiguous range. Full pandas materialization of
        # the post-NNLS frame OOMs at ~60GB on multiplexed low-input runs.
        _bounds = (decoy_coeffs.with_row_index("__idx")
                               .group_by(dc_group_keys, maintain_order=True)
                               .agg(pl.col("__idx").min().alias("start"),
                                    pl.col("__idx").len().alias("n")))
        dc_offsets = {
            tuple(row[:-2]): (int(row[-2]), int(row[-1]))
            for row in _bounds.iter_rows()
        }
        # The four columns get_ms2_vals reads, as numpy arrays: each channel's
        # rows are then a zero-copy slice through dc_offsets
        dc_arrays = {c: decoy_coeffs[c].to_numpy() for c in ("spec_id", "coeff", "rt", "Ms1_spec_id")}
        dc_source = (dc_arrays, dc_offsets)
    else:
        # pandas frames: unit-test fixtures
        dc_source = decoy_coeffs.groupby(dc_group_keys)

    group_keys_all = list(fdc_group.groups)
    group_rows = fdc_group.indices
    # The filtered-ID columns each group reads, as numpy arrays; a group's rows
    # are fancy-indexed out of them instead of building a pandas sub-frame.
    fdc_cols = {c: filtered_decoy_coeffs[c].to_numpy()
                for c in ("seq", "mz", "coeff", "Ms1_spec_id", "rt", "prec_im", "protein")
                if c in filtered_decoy_coeffs.columns}
    area_radius = int(config.args.plex_area_scan_radius)
    # Fit a ±fit_radius window around the voted apex: room for each channel's
    # apex to move (±additional_scans with free_apex, ±apex_jitter in the walk)
    # with its full plex_Area window, and never fewer than MIN_MS1_POINTS scans,
    # which the channel FDR's MS1 trace features need
    fit_radius = max(int(config.args.apex_jitter) + area_radius, additional_scans + area_radius,
                     MIN_MS1_POINTS // 2)
    n_threads = max(1, int(config.args.threads))

    def setup(key):
        return _setup_group(key, group_rows, fdc_cols, fdc_group, timeplex, tag, SILAC, ms1_spec_idxs,
                            ms1_spectra, all_spectra, window_half_width, vote_sigma, fit_radius, num_iso,
                            mz_ppm, dc_source, ms2_rt, rt_tol, bottom_of_window, top_of_window, ms2_spec_idxs)

    # Each row's (group, channel), the last group to claim the row's channel key
    owner = {}
    n_rows = len(filtered_decoy_coeffs)
    if timeplex:
        row_keys = list(zip(filtered_decoy_coeffs["seq"], filtered_decoy_coeffs["z"],
                            filtered_decoy_coeffs["time_channel"]))
    else:
        row_keys = list(zip(filtered_decoy_coeffs["seq"], filtered_decoy_coeffs["z"]))
    out = {}          # (group, channel): _channel_outputs

    with ThreadPoolExecutor(max_workers=n_threads) as pool, tqdm.tqdm(total=len(group_keys_all)) as bar:
        for b0 in range(0, len(group_keys_all), _GROUP_BATCH):
            batch = group_keys_all[b0:b0 + _GROUP_BATCH]
            groups = list(pool.map(setup, batch))
            for g, s in enumerate(groups):
                for k, ck in enumerate(s.channel_keys):
                    owner[ck] = (b0 + g, k)
            # Every (group, scan) pair of the batch, group by group
            n_scans = np.array([len(s.scans) for s in groups])
            pair0 = np.concatenate([[0], np.cumsum(n_scans)])
            pair_group = np.repeat(np.arange(len(groups)), n_scans)
            pair_scan = np.concatenate([s.scans for s in groups]).astype(np.int64)
            flat = _FlatRows(groups, im_tol, ms1_spectra)
            Y = np.zeros((len(pair_scan), flat.max_rows))
            by_scan = _pairs_by_scan(pair_scan)
            list(pool.map(lambda item: _extract_scan(all_spectra.get_by_idx(item[0]), item[1], pair_group, flat, Y),
                          by_scan))
            chunks = [range(i, min(i + 64, len(groups))) for i in range(0, len(groups), 64)]
            for done in pool.map(lambda gs: _fit_groups(gs, groups, pair0, Y, num_iso_r, area_radius), chunks):
                for g, channels in done:
                    for k, res in enumerate(channels):
                        out[(b0 + g, k)] = res
            bar.update(len(batch))
            if config.ran_from_GUI:
                logger.info(f"Fitting - {min(100, int(100 * (b0 + len(batch)) / len(group_keys_all)))}%")

    quant = _quant_frame([out.get(owner.get(k)) for k in row_keys], filtered_decoy_coeffs.index)
    return quant


class FitRows:
    """The spectrum-independent half of one precursor group's channel fit.

    Theoretical peaks (of any channel) within mz_ppm of each other share one
    row, so an isotope of one channel that overlaps an isotope of another is
    modelled jointly; the flank positions become rows with no library entry.

    M: design matrix (rows x channels); row_mz: each row's mean member m/z;
    is_flank: the flank rows; mem_lo / mem_hi / row_mem: each row's members'
    m/z windows (row r's are mem_*[row_mem[r]:row_mem[r + 1]]); iso_row[k, j]:
    the row of channel k's isotope j; top2[k]: the rows nearest channel k's two
    most intense isotopes (a channel with no signal in both is left out of the
    fit).
    """
    __slots__ = ("M", "row_mz", "is_flank", "mem_lo", "mem_hi", "row_mem", "iso_row", "top2")

    def __init__(self, ms1_iso_patterns, mz_ppm, flanks):
        K, N = ms1_iso_patterns.shape[:2]
        peaks = [(ms1_iso_patterns[k, j, 0], k, ms1_iso_patterns[k, j, 1], j) for k in range(K) for j in range(N)]
        peaks += [(m, -1, 0.0, -1) for m in flanks]
        peaks.sort(key=lambda t: t[0])
        groups, cur = [], [peaks[0]]
        for t in peaks[1:]:
            if t[0] - cur[-1][0] <= mz_ppm * t[0]:
                cur.append(t)
            else:
                groups.append(cur)
                cur = [t]
        groups.append(cur)
        self.M = np.zeros((len(groups), K))
        self.row_mz = np.zeros(len(groups))
        self.is_flank = np.zeros(len(groups), dtype=bool)
        self.iso_row = np.zeros((K, N), dtype=np.int64)
        lo, hi, row_mem = [], [], [0]
        for r, members in enumerate(groups):
            for m0, k, rel, j in members:
                if k >= 0:
                    self.M[r, k] += rel
                    self.iso_row[k, j] = r
                else:
                    self.is_flank[r] = True
                lo.append(m0 * (1 - mz_ppm))
                hi.append(m0 * (1 + mz_ppm))
            row_mem.append(len(lo))
            self.row_mz[r] = np.mean([m[0] for m in members])
        self.mem_lo, self.mem_hi = np.array(lo, dtype=np.float64), np.array(hi, dtype=np.float64)
        self.row_mem = np.array(row_mem, dtype=np.int64)
        self.top2 = np.array([[int(np.argmin(np.abs(self.row_mz - ms1_iso_patterns[ch, j, 0])))
                               for j in np.argsort(ms1_iso_patterns[ch, :, 1])[-2:]] for ch in range(K)],
                             dtype=np.int64).reshape(K, -1)

    def intensities(self, mz, intens):
        """The observed intensity of each row in one spectrum's (m/z-sorted)
        peaks: every peak within a member's window, each peak counted in the
        first row (by m/z) that takes it."""
        mz = np.ascontiguousarray(mz, dtype=np.float64)
        intens = np.ascontiguousarray(intens, dtype=np.float64)
        y = np.zeros(len(self.M))
        n = mz.size
        _row_sums(mz, intens, n, self.mem_lo, self.mem_hi, self.row_mem, 0, len(self.M), y, 0,
                  np.zeros(n, dtype=np.bool_), np.zeros(n, dtype=np.bool_), np.zeros(n))
        return y


def build_fit_system(ms1_iso_patterns, mz, intens, mz_ppm, flanks):
    """Rows for the channel fit (FitRows) and their observed intensities in one
    spectrum.  Returns the design matrix (rows x channels), the observed row
    intensities, the row m/z and a mask of the flank rows."""
    rows = FitRows(ms1_iso_patterns, mz_ppm, flanks)
    return rows.M.copy(), rows.intensities(mz, intens), rows.row_mz.copy(), rows.is_flank.copy()


def fit_rows(rows, y):
    """NNLS of the channels plus a flat on-grid background, on rows with
    observed intensities *y*.

    In a dense MS1 other peptides' isotope ladders share this group's mass
    defect grid, so every isotope row -- and the empty flank rows -- carries a
    roughly constant extra intensity (on the 200pg 5plex it is ~30% of a
    neighbouring channel's monoisotope).  Without it the fit assigns that
    intensity to the faintest channels and compresses their ratios.  A
    non-negative intercept, anchored by the flank rows, absorbs it.

    Returns (coefficients per channel, observed rows minus background, design
    matrix, row m/z, background per row); channels whose two most intense
    isotopes both had no signal are set to zero.
    """
    K = rows.M.shape[1]
    M = rows.M.copy()
    M[:, ~(y[rows.top2] > 0).any(axis=1)] = 0.0
    A = np.c_[M, np.ones(len(y))]
    coef, _ = optimize.nnls(A, y)
    background = coef[K]
    c = coef[:K]
    c[c < 1] = 0.0
    keep = ~rows.is_flank & np.any(M != 0, axis=1)
    return c, np.clip(y[keep] - background, 0, None), M[keep], rows.row_mz[keep], background


def fit_channels_with_background(ms1_iso_patterns, mz, intens, mz_ppm, flanks):
    """fit_rows on one spectrum's peaks (*mz*, *intens*)."""
    rows = FitRows(ms1_iso_patterns, mz_ppm, flanks)
    return fit_rows(rows, rows.intensities(mz, intens))


class _GroupSetup:
    """One precursor group's spectrum-independent inputs (see _setup_group)."""
    __slots__ = ("scans", "channel_keys", "theo", "rows", "ms2", "mz_lo", "mz_hi", "im")


def _setup_group(key, group_rows, fdc_cols, fdc_group, timeplex, tag, SILAC, ms1_spec_idxs, ms1_spectra,
                 all_spectra, window_half_width, vote_sigma, fit_radius, num_iso, mz_ppm, dc_source, ms2_rt,
                 rt_tol, bottom_of_window, top_of_window, ms2_spec_idxs):
    """The group's fit window (its voted apex ± fit_radius MS1 scans), channel
    keys, isotope envelopes (theo: channels x isotopes relative intensities),
    FitRows, MS2 coefficients interpolated at the window's scans (channels x
    scans), and the m/z range and IM its peaks are cut and gated to."""
    rows = group_rows[key]
    tag_group = {c: a[rows] for c, a in fdc_cols.items()}
    s = _GroupSetup()
    # Observed IM for this precursor; gates the MS1 peaks.  NaN disables the gate.
    s.im = group_prec_im(tag_group)
    prec_seqs, prec_mzs, prec_z, prec_rt, top_ms1_spec_idx, largest_coeff_scans, time_channel = \
        get_seqs_and_mzs(fdc_group, timeplex, tag, key, SILAC, tag_group=tag_group)
    all_scans, _ = minmax_spec_window(largest_coeff_scans, ms1_spec_idxs, ms1_spectra, all_spectra, window_half_width)

    ### Gaussian-weighted voting: each channel contributes its max-coeff scan
    # as a Gaussian centered on that scan, with sigma = vote_sigma cycles
    # (derived from the global elution SD). Adjacent-cycle votes reinforce
    # instead of competing, and outlier channels only contribute weakly far
    # from the consensus region.
    all_scans_arr = np.asarray(all_scans)
    positions = np.arange(len(all_scans_arr), dtype=np.float64)
    votes = np.zeros(len(all_scans_arr), dtype=np.float64)
    g_seq, g_coeff, g_ms1 = tag_group["seq"], tag_group["coeff"], tag_group["Ms1_spec_id"]
    for seq in dict.fromkeys(g_seq.tolist()):      # channels in first-seen order, like .unique()
        ch = np.flatnonzero(g_seq == seq)
        best = ch[np.nanargmax(g_coeff[ch])]       # first max, like idxmax
        s_idx = float(np.searchsorted(all_scans_arr, int(g_ms1[best])))
        votes += float(g_coeff[best]) * np.exp(-0.5 * ((positions - s_idx) / vote_sigma) ** 2)
    voted_apex = int(all_scans_arr[int(np.argmax(votes))])
    apex_idx = int(np.searchsorted(all_scans, voted_apex))
    lo = max(0, apex_idx - fit_radius)
    hi = min(len(all_scans), apex_idx + fit_radius + 1)
    s.scans = np.asarray(all_scans[lo:hi], dtype=np.int64)

    keys, isotopes, ms2 = [], [], []
    for prec_mz, prec_seq in zip(prec_mzs, prec_seqs):
        ms2_vals, _, channel_key = get_ms2_vals(prec_seq, prec_z, prec_rt, time_channel, timeplex, dc_source,
                                                ms2_rt, rt_tol, prec_mz, bottom_of_window, top_of_window,
                                                ms2_spec_idxs)
        keys.append(channel_key)
        ms2.append(build_ms2_interpolator(ms2_vals)(list(all_scans[lo:hi])))
        isotopes.append(compute_isotopes(prec_seq, prec_mz, prec_z, num_iso, [tag, SILAC]))
    s.channel_keys = keys
    s.ms2 = np.array(ms2, dtype=np.float64)
    patterns = np.array([(p.mz, p.intensity) for iso_ in isotopes for p in iso_],
                        dtype=np.float64).reshape(len(isotopes), len(isotopes[0]), 2)
    s.theo = patterns[:, :, 1]
    s.rows = FitRows(patterns, mz_ppm, flank_mz(isotopes))
    # Every channel's isotopes lie in [mz_lo, mz_hi], with the background flanks
    margin = max(1.5, (N_FLANK + 0.5) * ISOTOPE_SPACING_DA / prec_z)
    s.mz_lo = min(prec_mzs) - margin
    s.mz_hi = max(prec_mzs) + (num_iso + 1) * ISOTOPE_SPACING_DA / prec_z + margin
    return s


class _FlatRows:
    """A batch's FitRows and peak cuts as flat arrays, for _extract_scan."""

    def __init__(self, groups, im_tol, ms1_spectra):
        n_rows = np.array([len(s.rows.M) for s in groups])
        self.max_rows = int(n_rows.max()) if len(groups) else 0
        self.row0 = np.concatenate([[0], np.cumsum(n_rows)]).astype(np.int64)
        n_mem = np.array([len(s.rows.mem_lo) for s in groups])
        mem0 = np.concatenate([[0], np.cumsum(n_mem)])[:-1]
        self.row_mem = np.concatenate([s.rows.row_mem[:-1] + m0 for s, m0 in zip(groups, mem0)]
                                      + [[n_mem.sum()]]).astype(np.int64)
        self.mem_lo = np.concatenate([s.rows.mem_lo for s in groups])
        self.mem_hi = np.concatenate([s.rows.mem_hi for s in groups])
        self.mz_lo = np.array([s.mz_lo for s in groups], dtype=np.float64)
        self.mz_hi = np.array([s.mz_hi for s in groups], dtype=np.float64)
        # The IM gate, as im_filtered_peaks applies it with numpy's types: the
        # peaks' mobility minus the group's IM in their common type, its
        # absolute value compared with im_tol in theirs
        mob = getattr(ms1_spectra[0], "mobility", None) if len(ms1_spectra) else None
        ims = [s.im for s in groups]
        self.gate = np.array([mob is not None and im_tol is not None and im_tol > 0.0
                              and im is not None and np.isfinite(im) for im in ims], dtype=np.bool_)
        if mob is not None:
            diff_t = np.result_type(np.asarray(mob).dtype, 0.0)
            self.im = np.array([im if g else 0.0 for im, g in zip(ims, self.gate)], dtype=diff_t)
            cmp_t = np.result_type(diff_t, im_tol) if self.gate.any() else np.float64
            self.im_tol = float(np.asarray(im_tol if self.gate.any() else 0.0, dtype=cmp_t))
        else:
            self.im = np.zeros(len(groups))
            self.im_tol = 0.0


def _pairs_by_scan(pair_scan):
    """[(scan number, the pairs on it)], scans in order."""
    order = np.argsort(pair_scan, kind="stable")
    scans, starts = np.unique(pair_scan[order], return_index=True)
    return list(zip(scans.tolist(), np.split(order, starts[1:])))


def _extract_scan(spec, pairs, pair_group, flat, Y):
    mob = getattr(spec, "mobility", None)
    use_mob = mob is not None and bool(flat.gate[pair_group[pairs]].any())
    mob_arr = np.asarray(mob) if use_mob else np.zeros(0, dtype=flat.im.dtype)
    _extract_pairs(np.asarray(spec.mz), np.asarray(spec.intens), mob_arr, use_mob, pairs.astype(np.int64),
                   pair_group, flat.mz_lo, flat.mz_hi, flat.gate, flat.im.astype(mob_arr.dtype), flat.im_tol,
                   flat.row0, flat.mem_lo, flat.mem_hi, flat.row_mem, Y)


@nb.njit(cache=False, nogil=True)
def _pairwise_sum(a, lo, n):
    """numpy's pairwise summation of a[lo:lo + n] (np.sum of a contiguous
    float64 array), operation for operation, so the sums are bit-identical."""
    if n < 8:
        res = 0.0
        for i in range(n):
            res += a[lo + i]
        return res
    elif n <= 128:
        r0 = a[lo]; r1 = a[lo + 1]; r2 = a[lo + 2]; r3 = a[lo + 3]
        r4 = a[lo + 4]; r5 = a[lo + 5]; r6 = a[lo + 6]; r7 = a[lo + 7]
        i = 8
        while i < n - (n % 8):
            r0 += a[lo + i]; r1 += a[lo + i + 1]; r2 += a[lo + i + 2]; r3 += a[lo + i + 3]
            r4 += a[lo + i + 4]; r5 += a[lo + i + 5]; r6 += a[lo + i + 6]; r7 += a[lo + i + 7]
            i += 8
        res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))
        while i < n:
            res += a[lo + i]
            i += 1
        return res
    else:
        n2 = n // 2
        n2 -= n2 % 8
        return _pairwise_sum(a, lo, n2) + _pairwise_sum(a, lo + n2, n - n2)


@nb.njit(cache=False, nogil=True)
def _row_sums(mz, intens, n, mem_lo, mem_hi, row_mem, r0, r1, out, out0, sel, used, buf):
    """Rows r0..r1's observed intensities in the peaks mz[:n] / intens[:n] into
    out[out0:]: the peaks in any of the row's member windows, less those an
    earlier row took, summed in index order."""
    for j in range(n):
        used[j] = False
    for r in range(r0, r1):
        a_min = n
        b_max = 0
        for m in range(row_mem[r], row_mem[r + 1]):
            a = np.searchsorted(mz[:n], mem_lo[m])
            b = np.searchsorted(mz[:n], mem_hi[m])
            for j in range(a, b):
                sel[j] = True
            a_min = min(a_min, a)
            b_max = max(b_max, b)
        cnt = 0
        for j in range(a_min, b_max):
            if sel[j]:
                sel[j] = False
                if not used[j]:
                    used[j] = True
                    buf[cnt] = intens[j]
                    cnt += 1
        out[out0 + r - r0] = _pairwise_sum(buf, 0, cnt)


@nb.njit(cache=False, nogil=True)
def _extract_pairs(mz, intens, mob, use_mob, pairs, pair_group, mz_lo, mz_hi, gate, im, im_tol,
                   row0, mem_lo, mem_hi, row_mem, Y):
    """Each pair's row intensities in one MS1 scan: the scan's peaks cut to the
    group's m/z range and, where gated, to |mobility - IM| <= im_tol (the cut and
    gate of im_filtered_peaks), then _row_sums."""
    width = 0
    for p in pairs:
        g = pair_group[p]
        width = max(width, np.searchsorted(mz, mz_hi[g]) - np.searchsorted(mz, mz_lo[g]))
    gmz = np.empty(width)
    gint = np.empty(width)
    sel = np.zeros(width, dtype=np.bool_)
    used = np.empty(width, dtype=np.bool_)
    buf = np.empty(width)
    for p in pairs:
        g = pair_group[p]
        a = np.searchsorted(mz, mz_lo[g])
        b = np.searchsorted(mz, mz_hi[g])
        n = 0
        for i in range(a, b):
            if use_mob and gate[g]:
                if not np.float64(abs(mob[i] - im[g])) <= im_tol:     # NaN mobility fails, as in numpy
                    continue
            gmz[n] = np.float64(mz[i])
            gint[n] = np.float64(intens[i])
            n += 1
        _row_sums(gmz, gint, n, mem_lo, mem_hi, row_mem, row0[g], row0[g + 1], Y[p], 0, sel, used, buf)


def _fit_groups(gs, groups, pair0, Y, num_iso_r, area_radius):
    """Fit every pair of the groups *gs*, then each channel's outputs
    (_channel_outputs).  Returns [(group, [outputs per channel])]."""
    done = []
    for g in gs:
        s = groups[g]
        R, K = s.rows.M.shape
        W = len(s.scans)
        C = np.zeros((W, K))
        bg = np.zeros(W)
        ps = np.full(W, np.nan)
        has_fit = np.zeros(W, dtype=bool)
        for w in range(W):
            c, obs, fit_matrix, _, background = fit_rows(s.rows, Y[pair0[g] + w, :R])
            C[w], bg[w] = c, background
            if len(obs):
                has_fit[w] = True
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    ps[w] = np_pearson_cor(np.sum(fit_matrix * c, 1), obs).statistic
        traces = Y[pair0[g]:pair0[g] + W, :R][:, s.rows.iso_row]       # scans x channels x isotopes
        traces = np.where(traces > 0, traces, min_int)
        done.append((g, [_channel_outputs(s, C[:, k].copy(), bg, ps, has_fit, traces[:, k, :], k, num_iso_r,
                                          area_radius) for k in range(K)]))
    return done


def _channel_outputs(s, fitted, bg, ps, has_fit, trace, k, num_iso_r, area_radius):
    """One channel's MS1 quant: (MS1_SCALAR_COLUMNS values, MS1_LIST_COLUMNS
    values).  *fitted* is its coefficient at each scan of the window, *bg* /
    *ps* / *has_fit* the group's background, fit Pearson and whether any row
    was fitted there, *trace* its isotopes' traces (scans x isotopes)."""
    # The window is centered on the group's voted apex.  Each channel's apex:
    # optionally argmax-pick a seed within ±additional_scans of the center,
    # then let the monotonic walk drift from that seed within ±apex_jitter,
    # capped plex_area_scan_radius positions in from the edge so area() always
    # has its full window.
    center = len(fitted) // 2
    if config.args.free_apex:
        seed = argmax_within_radius(fitted, center, radius=int(config.args.additional_scans), area_radius=area_radius)
    else:
        seed = center
    apex_pos = walk_to_local_max(fitted, seed, apex_jitter=int(config.args.apex_jitter), area_radius=area_radius)
    plex_area, plex_bg = area(fitted, apex_pos, area_radius), area(bg, apex_pos, area_radius)
    ps_or_0 = np.where(has_fit, ps, 0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spec_pearsons, iso_ratio, ms1_int = compute_ms1_ms2_cors(s.ms2[k], trace, num_iso_r, s.theo[k])
    # Correlations rescaled from [-1, 1] to [0, 1] before their product
    traceproduct = np.log10((spec_pearsons[0] + 1) / 2 * ((spec_pearsons[1] + 1) / 2)
                            * ((spec_pearsons[2] + 1) / 2) + 1e-6)
    scalars = (np.max(fitted),
               ps_or_0[center],            # the plex fit's Pearson at the voted apex, 0 where nothing was fitted
               plex_area, plex_bg,         # plex_bg: the fitted on-grid background per row over the same scans
               plex_area / max(plex_bg, 1.0),
               s.scans[apex_pos],
               spec_pearsons[0], traceproduct, iso_ratio.statistic, ms1_int,
               np.trapezoid(trace[:, 0], s.scans.astype(float)))
    lists = (fitted, ps_or_0, s.scans, fitted, ps, s.scans) + tuple(trace[:, j] for j in range(trace.shape[1]))
    return scalars, lists


def _quant_frame(per_row, index):
    """The rows' outputs (_channel_outputs, None for a row with no quant) as a
    frame on *index*."""
    n = len(per_row)
    num = np.full((n, len(MS1_SCALAR_COLUMNS)), np.nan)
    lists = np.full((n, len(MS1_LIST_COLUMNS)), None, dtype=object)
    for i, res in enumerate(per_row):
        if res is not None:
            num[i] = res[0]
            for j, v in enumerate(res[1]):
                lists[i, j] = v
    frame = pd.DataFrame({**{c: num[:, j] for j, c in enumerate(MS1_SCALAR_COLUMNS)},
                          **{c: lists[:, j] for j, c in enumerate(MS1_LIST_COLUMNS)}}, index=index)
    return frame[["plexfitMS1", "plexfitMS1_p", "plexfittrace", "plexfit_ps", "plexfittrace_spec_all",
                  "plexfittrace_all", "plexfittrace_ps_all", "plex_Area", "plex_bg", "plex_snr",
                  "ms1_apex_scan", "ms1_cor", "traceproduct", "iso_cor", "MS1_Int", "all_ms1_specs",
                  "MS1_Area"] + [f"all_ms1_iso{i}vals" for i in range(config.num_iso_ms1)]]


def compute_ms1_ms2_cors(ms2, traces, num_iso_r, theoretical):
    """One channel's MS1-MS2 agreement over the fit window.

    *ms2* is its MS2 coefficients interpolated at the window's MS1 scans,
    *traces* its isotopes' MS1 traces (scans x isotopes, monoisotope first),
    *theoretical* its isotope envelope's relative intensities.

    Returns
    -------
    spec_pearsons : list of float
        Pearson of the MS2 trace with the monoisotope and the next num_iso_r isotopes
    iso_ratio : PearsonRResult-like
        Pearson of the theoretical envelope with the observed one at the scan
        of the highest MS2 value
    ms1_int : float
        The least-squares scale of the theoretical envelope onto that observed one
    """
    spec_pearsons = [np_pearson_cor(ms2, traces[:, i]).statistic for i in range(num_iso_r + 1)]
    obs = traces[np.argmax(ms2)]
    iso_ratio = np_pearson_cor(theoretical, obs)
    ms1_int = np.linalg.lstsq(theoretical[:, np.newaxis], obs, rcond=-1)[0][0]
    return spec_pearsons, iso_ratio, ms1_int


def get_seqs_and_mzs(fdc_group, timeplex, tag, key, SILAC, tag_group=None):
    """
    Retrieve sequences, m/z values, charge, retention time, and scan information 
    for a given precursor group, optionally including timeplex channels.

    Parameters
    ----------
    fdc_group : pandas.core.groupby.generic.DataFrameGroupBy
        Grouped FDC DataFrame, either by ('seq', 'z') or ('seq', 'z', 'time_channel') if timeplex.
    timeplex : bool
        Whether the data includes timeplex channels.
    tag : massTag
        A massTag instance
    key : tuple
        The key identifying the group in fdc_group:
        - (seq, z) if not timeplex
        - (seq, z, time_channel) if timeplex
    SILAC : massTag
        A massTag instance
    Returns
    -------
    prec_seqs : tuple of str
        Unstripped sequences for all channels in the group.
    prec_mzs : tuple of float
        m/z values corresponding to each sequence in prec_seqs.
    prec_z : int or float
        Charge of the precursor.
    prec_rt : float
        Retention time of the channel with the highest coefficient.
    top_ms1_spec_idx : int
        MS1 spectrum index of the channel with the highest coefficient.
    largest_coeff_scans : list of int
        List of MS1 spectrum indices for all scans in the group.
    time_channel : int or None
        Time channel index if timeplex is True; otherwise None.
    """

    # tag_group: the group's rows when the caller has already split them out
    # (ms1_cor_channels does, so its worker threads never use the groupby).
    if tag_group is None:
        tag_group = fdc_group.get_group(key)
    # Columns as arrays, so a pandas group and a dict of numpy arrays behave the same
    group_prec_mzs = np.asarray(tag_group["mz"])
    group_prec_seqs = np.asarray(tag_group["seq"])
    group_prec_z = key[1]
    if timeplex:
        time_channel = key[2]
    else:
        time_channel = None
    largest_id = int(np.nanargmax(np.asarray(tag_group["coeff"])))   # skips NaN like Series.argmax
    top_ms1_spec_idx = np.asarray(tag_group["Ms1_spec_id"]).tolist()[largest_id]
    prec_rt = np.asarray(tag_group["rt"]).tolist()[largest_id]
    
    if (tag.name != "no_tag" and SILAC is not None):
        
        all_comb = get_other_channels((group_prec_seqs[largest_id],group_prec_z), group_prec_mzs[largest_id], [tag,SILAC])
        prec_seqs,prec_mzs = tuple(zip(*all_comb))
    elif tag.name != "no_tag":
        all_comb = get_other_channels((group_prec_seqs[largest_id],group_prec_z), group_prec_mzs[largest_id], [tag])
        prec_seqs,prec_mzs = tuple(zip(*all_comb))
    elif SILAC is not None:
        all_comb = get_other_channels((group_prec_seqs[largest_id],group_prec_z), group_prec_mzs[largest_id], [SILAC])
        prec_seqs,prec_mzs = tuple(zip(*all_comb))
    else:
        prec_seqs = (group_prec_seqs[largest_id],)
        prec_mzs = (group_prec_mzs[largest_id],)

    largest_coeff_scans = np.asarray(tag_group["Ms1_spec_id"]).tolist()

    return prec_seqs, prec_mzs, group_prec_z, prec_rt, top_ms1_spec_idx, largest_coeff_scans, time_channel


def minmax_spec_window(largest_coeff_scans, ms1_spec_idxs, ms1_spectra, all_spectra, window_half_width):
    """
    Retrieve the list of MS1 scan and scan idxs within a window of the highest coeff scans

    Parameters
    ----------
    largest_coeff_scans : list of int
        List of MS1 spectrum indices for all scans in the group.
    ms1_spec_idxs : 1d array of int
        Array of all MS1 spec indexes in MS run
    ms1_spectra : list of Spectrum
        Spectrum Object
    all_spectra : SpectrumFile
        SpectrumFile Object containing all Spectrum Objects
    window_half_width: int
        The width around the largest coeff scans that are returned

    Returns
    -------
    all_scans : list of int
        A list of scan idxs
    spectra_subset : list of Spectrum
        A list of corresponding Spectrum Objects

    Notes
    -----
    The function identifies the min and max scan numbers from `largest_coeff_scans`
    and selects a contiguous window of MS1 scans surrounding them,
    bounded by `window_half_width`.

    When IM data is present, filters to spectra sharing the same IM bin as the
    reference scans, using RT-based windowing within that bin.
    """
    # Check if IM data is present
    has_im = len(ms1_spectra) > 0 and getattr(ms1_spectra[0], 'im_lo', None) is not None

    if has_im:
        # Determine target IM bin from reference scan
        ref_scan = largest_coeff_scans[0]
        ref_idx = list(ms1_spec_idxs).index(ref_scan)
        ref_spec = ms1_spectra[ref_idx]
        target_im_lo, target_im_hi = ref_spec.im_lo, ref_spec.im_hi

        # Filter ms1 indices to same IM bin
        im_filtered_idxs = [i for i, s in enumerate(ms1_spectra)
                            if s.im_lo == target_im_lo and s.im_hi == target_im_hi]
        im_filtered_scan_nums = np.array([ms1_spec_idxs[i] for i in im_filtered_idxs])

        # Find min/max within filtered set
        max_scan, min_scan = max(largest_coeff_scans), min(largest_coeff_scans)
        # Find positions within the IM-filtered list
        filtered_list = list(im_filtered_scan_nums)
        if min_scan in filtered_list:
            idx_min = filtered_list.index(min_scan)
        else:
            idx_min = np.searchsorted(im_filtered_scan_nums, min_scan)
        if max_scan in filtered_list:
            idx_max = filtered_list.index(max_scan)
        else:
            idx_max = np.searchsorted(im_filtered_scan_nums, max_scan)

        lo = max(0, idx_min - window_half_width)
        hi = min(len(im_filtered_scan_nums), idx_max + window_half_width + 1)
        scans_each_side = im_filtered_scan_nums[lo:hi]
        all_scans = list(scans_each_side)
        spectra_subset = [all_spectra.get_by_idx(idx) for idx in all_scans]
    else:
        ## max and min of this list
        max_scan, min_scan = max(largest_coeff_scans), min(largest_coeff_scans)
        ms1_list_idx_min = list(ms1_spec_idxs).index(min_scan)
        ms1_list_idx_max = list(ms1_spec_idxs).index(max_scan)
        scans_each_side = np.array(ms1_spec_idxs)[np.arange(max(0,ms1_list_idx_min-window_half_width),min(len(ms1_spectra),ms1_list_idx_max+window_half_width+1))]
        all_scans = list(scans_each_side)
        spectra_subset = [all_spectra.get_by_idx(idx) for idx in all_scans]

    return all_scans, spectra_subset



def _lookup_group(source, key):
    """Return a pandas DataFrame slice for ``key`` or ``None`` if absent.

    Accepts either a pandas DataFrameGroupBy (legacy callers and tests) or a
    ``(polars.DataFrame, offsets_dict)`` tuple. The tuple form avoids
    materializing the full post-NNLS frame as pandas — only the per-precursor
    slice gets converted.
    """
    if isinstance(source, tuple):
        table, offsets = source
        if key not in offsets:
            return None
        start, n = offsets[key]
        if isinstance(table, dict):     # {column: numpy array}: slice each column
            return {c: a[start:start + n] for c, a in table.items()}
        return table.slice(start, n).to_pandas()
    if key not in source.groups:
        return None
    return source.get_group(key)


def get_ms2_vals(prec_seq, prec_z, prec_rt, time_channel, timeplex, grouped_decoy_coeffs, ms2_rt, rt_tol, prec_mz, bottom_of_window, top_of_window, ms2_spec_idxs):
    """
    For a given precursor, get the ms2 JMod coefficients at MS2 scan #s and return this dictionary,
    along with the ms1 scan number corresponding to the highest scan

    Parameters
    ----------
    prec_seq : str
        Precursor non-stripped sequence
    prec_z : int or float
        precursor charge
    prec_rt : float
        Precursor retention time, not currently used
    time_channel : int or str or None
        Precursor time channel
    timeplex : bool
        Bool for timeplex
    grouped_decoy_coeffs : pandas DataFrameGroupBy or (polars.DataFrame, dict)
        Either a pandas groupby (legacy) or a ``(slim_polars_df, offsets)``
        tuple where ``offsets`` maps the channel key to ``(start, length)`` row
        ranges in the polars frame. The tuple form is used in production to
        avoid materializing the full post-NNLS frame as pandas.
    ms2_rt : 1D Numpy array
        Array of retention time of all ms2 scans in MS run
    rt_tol : float
        Retention time tolerance
    prec_mz : float
        Precursor mz
    bottom_of_window: 1D Numpy array
        Array of the bottom of the isolation window for every ms2 scan (same size as ms2_rt)
    top_of_window: 1D Numpy array
        Array of the top of the isolation window for every ms2 scan (same size as ms2_rt)
    ms2_spec_idxs: 1D Numpy array
        Array of the scan index of every ms2 scan (same size as ms2_rt)


    Returns
    -------
    ms2_vals : dict[int] == float
        A dictionary with keys as ms2 scan indexes and values as the JMod MS2 coeff
    highest_ranked_spec : int
        The ms1 scan index from the highest ranked ms2 Jmod coeff
    channel_key : tuple of (str, int) or (str, int, int)
        Tuple of unstripped sequence, charge, and time channel if applicable

    """
    offset = 0

    if timeplex:
        channel_key = (prec_seq,prec_z,time_channel)
    else:
        channel_key = (prec_seq,prec_z)

    ## create dummy
    ms2_vals = {0:0}

    group = _lookup_group(grouped_decoy_coeffs, channel_key)
    if group is not None:

        # use coeff as rank score directly.  Columns read as arrays so a pandas
        # group and a dict of numpy slices behave the same; nanargmax skips NaN
        # like pandas idxmax / Series.argmax.
        group_coeff = np.asarray(group["coeff"])
        group_rt = np.asarray(group["rt"])
        best = int(np.nanargmax(group_coeff))
        highest_ranked_spec = np.asarray(group["Ms1_spec_id"])[best]

        # ms2_rt_bool = np.abs(ms2_rt-prec_rt)<rt_tol
        prec_rt = group_rt[best]
        ms2_window_bool = np.logical_and(prec_mz+offset>bottom_of_window,prec_mz+offset<top_of_window)
        
        min_rt = np.minimum(prec_rt-rt_tol,np.min(group_rt)*.99)
        max_rt = np.maximum(prec_rt+rt_tol,np.max(group_rt)*1.01)
        ms2_rt_bool = np.logical_and(ms2_rt>=min_rt,ms2_rt<=max_rt)
        
        ms2_bool = np.logical_and(ms2_window_bool,ms2_rt_bool)
        possible_ms2_scans = ms2_spec_idxs[ms2_bool]
        ms2_vals = {i:min_int for i in possible_ms2_scans}
    
        # .tolist(): Python scalars, as iterating a pandas Series gives
        for scan,c in zip(np.asarray(group["spec_id"]).tolist(), group_coeff.tolist()):
            ms2_vals[scan]=c
    else:
        highest_ranked_spec = None

    return ms2_vals, highest_ranked_spec, channel_key



def im_filtered_peaks(spec, prec_im, im_tol, mz_lo=None, mz_hi=None):
    """(mz, intens) for this MS1 spectrum, restricted to peaks co-mobile with the precursor.

    MS1 frames are not IM-banded -- they carry per-peak mobility across the whole
    acquisition range -- so matching an isotope by m/z alone regularly picks up a
    co-isolated ion at unrelated mobility.  Filtering the peak list once per
    (spectrum, precursor) keeps the downstream m/z matching kernels unchanged and
    is cheaper than gating per isotope.

    With *mz_lo*/*mz_hi* the peaks are first cut to that m/z range (the
    spectrum is m/z-sorted, so two binary searches); a raw MS1 frame holds
    ~300k ions, of which a precursor group needs a few Th.

    Returns the spectrum's own arrays untouched when there is nothing to gate on
    (non-IM data, no fitted precursor IM, or no tolerance), so the non-IM path is
    bit-identical to before.  The mask preserves m/z order, which the kernels'
    searchsorted calls rely on.
    """
    mz, intens = spec.mz, spec.intens
    mob = getattr(spec, "mobility", None)
    if mz_lo is not None:
        a, b = np.searchsorted(mz, [mz_lo, mz_hi])
        mz, intens = mz[a:b], intens[a:b]
        if mob is not None:
            mob = mob[a:b]
    if mob is None or im_tol is None or not im_tol > 0.0:
        return mz, intens
    if prec_im is None or not np.isfinite(prec_im):
        return mz, intens
    keep = np.abs(np.asarray(mob) - prec_im) <= im_tol
    return mz[keep], intens[keep]


def group_prec_im(tag_group):
    """Median observed IM for a precursor group, or NaN when unavailable.

    All channels of one precursor share a mobility, so a single value per group
    is enough; the median shrugs off the odd NaN or outlier channel.
    """
    if "prec_im" not in tag_group:
        return np.nan
    vals = np.asarray(tag_group["prec_im"], dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    return float(np.median(vals)) if vals.size else np.nan


def build_ms2_interpolator(ms2_vals):
    return interp1d(list(ms2_vals.keys()), np.array(list(ms2_vals.values())), bounds_error=False)   



def compute_isotopes(prec_seq, prec_mz, prec_z, num_iso, tags):
    """
    Gets a list of isotope peaks for a precursor

    Parameters
    ----------
    prec_seq : str
        precursor unstripped Sequence
    prec_mz : float
        percursor_mz
    prec_z : float or int
        precursor charge
    num_iso : int
        number of isotopes to return
    tag : massTag
        massTag instance
    
    Returns
    -------
    isotopes : list of isotope Peak objects (src.isotope_distribution)
        A list of peak objects with p.mz, p.intensity, and p.charge of length num_iso

    Notes
    -------
    Mostly this is just the function iso.precursor_isotopes. However, if a tag doesn't have compositions, 
    (or the mass if otherwise off for some reason?) then the other parts of the code correct and make sure
    isotopes[0] == prec_mz

    """
    isotopes = iso.precursor_isotopes(prec_seq,prec_z,tags,num_iso, decoys=False)

    delta_mz = prec_mz-isotopes[0].mz
    for i in isotopes:
        i.mz+=delta_mz

    return isotopes


def get_ms1_index_of_max(ms2_vals, top_ms1_spec_idx, highest_ranked_spec):
    """
    Return the best MS1 spec idx to be used

    Parameters
    ----------
    ms2_vals : dict[int] = float
        Dictionary with keys of MS1 spec idxs and values of corresponding intensities
    top_ms1_spec_idx : int 
        MS1 spec idx from get_seqs_and_mzs
    highest_ranked_spec : int
        MS1 spec idx from get_ms2_vals
    
    Returns
    -------
    ms1_index_of_max : int
        The max index to be used going forwards

    """

    ## use monoiso ms1 prec mz to find the elution ms1 peak
    if ms2_vals=={0:0}:
        ms1_index_of_max = top_ms1_spec_idx ## should I just use the max of MS1???? Need to look into again
    else:
        ms1_index_of_max = highest_ranked_spec
    return ms1_index_of_max


def filter_all_scans(all_scans, ms1_peak_edge_idxs, all_ms1_vals, all_iso_vals, all_ms2_vals):
    """
    Filter all scans and corresponding values to only those within the MS1 peak edges.
    """
    channel_scans = all_scans[all_scans.index(ms1_peak_edge_idxs[0]):all_scans.index(ms1_peak_edge_idxs[1])+1]
    all_ms1_vals = {i:all_ms1_vals[i] for i in channel_scans}
    all_iso_vals = [{i:iso_vals[i] for i in channel_scans} for iso_vals in all_iso_vals]
    all_ms2_vals = {i:all_ms2_vals[i] for i in channel_scans}

    return channel_scans, all_ms1_vals, all_iso_vals, all_ms2_vals


def select_scans_to_search(top_ms1_spec_idx, all_scans, all_channel_scans, window_half_width):
    "Aggregate a sorted list of all unique scans collected from every channel"
    # idx_of_max =all_scans.index(top_ms1_spec_idx)
    # scans_to_search = np.array(all_scans)[np.arange(max(0,idx_of_max-window_half_width),min(len(all_scans),idx_of_max+window_half_width+1))]
    scans_to_search = np.sort(np.unique(np.concatenate(all_channel_scans)))
    return scans_to_search


def get_other_channels(prec, mz, tags):
    """
    Return the m/z and sequences for all channels of a given precursor.

    Parameters
    ----------
    prec : tuple
        A tuple (unstripped_seq, z) containing the peptide sequence (with tags) 
        and its charge state.
    mz : float
        The m/z value of the precursor in the current channel.
    tag : massTag
        A massTag instance.

    Returns
    -------
    all_prec_comb : list
        A list of tuples with each precursor-channel combination and their m/z:
        [unstripped_seq, float(mz)] for that channel.
    """
    ## check which tags are in use and update
    tags = [tag for tag in tags if len(re.findall(rf"{tag.name}-(\d+)",prec[0]))]
    ## identify what channel the current prec is in
    all_channels = [re.findall(rf"{tag.name}-(\d+)",prec[0]) for tag in tags]
    channels_top = tuple([i[0] for i in all_channels])
    num_tags = [len(i) for i in all_channels]
    
    # Get all channel combinations (one channel per tag)
    channel_options = [tag.channel_names for tag in tags]
    all_combinations = list(product(*channel_options))
    
    all_prec_comb = []
    for combination in all_combinations:
        # break
        # if channels==combination:
        #     all_prec_comb.append([prec[0],mz])
        # else:
        last_channels = channels_top
        new_seq = prec[0]
        new_mz = mz
        # For each tag, replace ALL occurrences with the same channel
        for tag, channel, n, l in zip(tags, combination, num_tags, last_channels):
            # break
            # Pattern to match tag-channel combinations
            pattern = rf'{tag.name}-\d+'
            replacement = f'{tag.name}-{channel}'
            new_seq = re.sub(pattern, replacement, new_seq)
            
            new_mz = new_mz + (n*(tag.mass_dict[tag.name+"-"+channel]-tag.mass_dict[tag.name+"-"+l])/prec[1])
            # print(new_seq, combination,new_mz)
        all_prec_comb.append([new_seq,new_mz])
    
    return all_prec_comb




def get_ms1_peak(x,y,idx,additional_scans):
    """
    Get the peak apex and bounds for a given set of monoisotopic intensities

    Parameters
    ----------
    x : list of int
        A list of MS1 spec idxs (from ms1_vals)
    y : list or array of float
        The moving averages of MS1 spec intensities (from ms1_vals)
    idx : int
        The MS1 spec idx used to determine the peak (important when there is more than one "peak" found)
    additional_scans : int
        The number of scans to add to each side of the peak edges

    Returns
    -------
    ms1_peak_idx : int
        The MS1 index of the peak
    ms1_peak_edge_idxs : 1D numpy array of int (len==2)
        The first and last MS1 spec idx that is a part of the peak
    """
    x = np.array(x)
    y = np.array(y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        peaks,peak_attr= find_peaks(y,width=(None,None))
    
    ## if no peak, return the index of the max
    if len(peaks)==0:
        return x[np.argmax(y)],[x[0],x[-1]]
    
    peak_idxs = x[peaks]
    
    closest_idx = np.argmin(np.abs(idx-peak_idxs))

    peak_idx = peaks[closest_idx]
    peak_edge_idxs = [max(0,peak_attr["left_bases"][closest_idx]-additional_scans),min(len(x)-1,peak_attr["right_bases"][closest_idx]+additional_scans)]
    # peak_edge_idxs = [peak_attr["left_bases"][closest_idx],peak_attr["right_bases"][closest_idx]]
    
    return x[peak_idx],x[peak_edge_idxs]



def moving_average(x, w=4):
    return np.convolve(x, np.ones(w), 'same') / w


def _compute_cooks_d(fit_matrix, obs_peaks, pred_coeff):
    """Compute Cook's distance and residuals for each observation in the NNLS fit."""
    n, p = fit_matrix.shape
    predicted = fit_matrix @ pred_coeff
    residuals = obs_peaks - predicted
    if n <= p or np.sum(residuals**2) == 0:
        return np.zeros(n), residuals
    mse = np.sum(residuals**2) / (n - p)
    try:
        hat_diag = np.einsum('ij,ji->i', fit_matrix,
                             np.linalg.pinv(fit_matrix.T @ fit_matrix) @ fit_matrix.T)
        hat_diag = np.clip(hat_diag, 0, 1 - 1e-12)
        cooks_d = (residuals**2 * hat_diag) / (p * mse * (1 - hat_diag)**2)
    except np.linalg.LinAlgError:
        cooks_d = np.zeros(n)
    return cooks_d, residuals


# Flank positions on each side of a plex group's envelopes, in isotope
# spacings, used to anchor the on-grid background (see fit_rows)
N_FLANK = 3
# Peptide isotope spacing (averagine, as JrMod's ISOTOPE_SPACING_DA), not the
# pure 13C step: the extra mass of M+1 averages over 13C, 15N, 2H and 18O
ISOTOPE_SPACING_DA = 1.00235


def flank_mz(all_iso):
    """m/z of the empty grid positions N_FLANK isotope spacings below the
    lightest monoisotope and above the heaviest last isotope of *all_iso*."""
    z = abs(int(all_iso[0][0].charge)) or 1
    sp = ISOTOPE_SPACING_DA / z
    lo = min(iso[0].mz for iso in all_iso)
    hi = max(p.mz for iso in all_iso for p in iso)
    return [lo - j * sp for j in range(1, N_FLANK + 1)] + [hi + j * sp for j in range(1, N_FLANK + 1)]


@nb.njit(cache=False)
def get_matrix_to_fit_numba(ms1_iso_patterns, group_lengths, dia_spectrum, all_iso_len, mz_ppm):    
    ### we only need to conseider the part of the spectrum that falls within the isotopic envelopes of the channels
    min_isotope = ms1_iso_patterns[:, :, 0].min() - 1
    max_isotope = ms1_iso_patterns[:, :, 0].max() + 1

    mz = dia_spectrum[:, 0]
    lo = np.searchsorted(mz, min_isotope, side="right")
    hi = np.searchsorted(mz, max_isotope, side="left")
    dia_spectrum = dia_spectrum[lo:hi]

    mz_peaks = dia_spectrum[:, 0]

    fdc_idxs = np.repeat(np.arange(all_iso_len), group_lengths)
    all_mz = ms1_iso_patterns[:,:,0].ravel()
    rel_iso_int = ms1_iso_patterns[:, :, 1].ravel()

    # For each library peak, find ALL observed peaks within ppm tolerance,
    # sum their intensities, and record the closest as the canonical row index.
    n_lib = all_mz.size
    n_obs = mz_peaks.size
    lib_peaks_matched = np.empty(n_lib, dtype=np.bool_)
    matched_obs_idx = np.empty(n_lib, dtype=np.int64)
    # Summed observed intensity for each library peak (across all matching obs peaks)
    summed_obs = np.zeros(n_lib, dtype=np.float64)
    obs_int = dia_spectrum[:, 1]

    for i in range(n_lib):
        lib_mz = all_mz[i]
        pos = np.searchsorted(mz_peaks, lib_mz)

        best_idx = -1
        best_dist = np.inf
        intensity_sum = 0.0

        # Scan left from insertion point
        j = pos - 1
        while j >= 0:
            obs_mz = mz_peaks[j]
            dist = abs(lib_mz - obs_mz)
            tol = mz_ppm * obs_mz
            if dist > tol:
                break
            intensity_sum += obs_int[j]
            if dist < best_dist:
                best_dist = dist
                best_idx = j
            j -= 1

        # Scan right from insertion point
        j = pos
        while j < n_obs:
            obs_mz = mz_peaks[j]
            dist = abs(lib_mz - obs_mz)
            tol = mz_ppm * obs_mz
            if dist > tol:
                break
            intensity_sum += obs_int[j]
            if dist < best_dist:
                best_dist = dist
                best_idx = j
            j += 1

        lib_peaks_matched[i] = (best_idx >= 0)
        matched_obs_idx[i] = best_idx
        summed_obs[i] = intensity_sum

    last_unassigned_peak_idx = len(mz_peaks)

    unique_fdcs = (set(fdc_idxs))

    # Group unmatched library peaks by m/z proximity so that overlapping
    # peaks from different channels share a matrix row (just like matched
    # peaks that hit the same observed peak share a row).
    n_unmatched = np.sum(~lib_peaks_matched)

    # Collect indices of unmatched peaks
    unmatched_idx = np.empty(n_unmatched, dtype=np.int64)
    k = 0
    for i in range(n_lib):
        if not lib_peaks_matched[i]:
            unmatched_idx[k] = i
            k += 1

    # Sort unmatched peaks by m/z
    unmatched_mz = np.empty(n_unmatched, dtype=np.float64)
    for k in range(n_unmatched):
        unmatched_mz[k] = all_mz[unmatched_idx[k]]
    sort_order = np.argsort(unmatched_mz)

    # Assign group IDs: consecutive peaks within tolerance share a group
    unmatched_group = np.empty(n_unmatched, dtype=np.int64)
    if n_unmatched > 0:
        current_group = 0
        unmatched_group[sort_order[0]] = 0
        for j in range(1, n_unmatched):
            prev_mz = unmatched_mz[sort_order[j - 1]]
            cur_mz = unmatched_mz[sort_order[j]]
            tol = mz_ppm * prev_mz
            if abs(cur_mz - prev_mz) <= tol:
                unmatched_group[sort_order[j]] = current_group
            else:
                current_group += 1
                unmatched_group[sort_order[j]] = current_group
        num_unmatched_groups = current_group + 1
    else:
        num_unmatched_groups = 0

    # Map group IDs back to full library-peak array
    unmatched_group_for_all = np.empty(n_lib, dtype=np.int64)
    k = 0
    for i in range(n_lib):
        if not lib_peaks_matched[i]:
            unmatched_group_for_all[i] = unmatched_group[k]
            k += 1
        else:
            unmatched_group_for_all[i] = 0  # unused

    n_rows = len(mz_peaks) + num_unmatched_groups
    n_cols = len(unique_fdcs)

    dense_matrix = np.zeros((n_rows, n_cols), dtype=np.float64)

    ms1_peak_coord = np.empty(n_lib, dtype=np.int64)
    for i in range(n_lib):
        if lib_peaks_matched[i]:
            ms1_peak_coord[i] = matched_obs_idx[i]
        else:
            ms1_peak_coord[i] = last_unassigned_peak_idx + unmatched_group_for_all[i]

    # np.add.at(dense_matrix, (ms1_peak_coord, fdc_idxs), rel_iso_int)
    for i in range(rel_iso_int.size):
        dense_matrix[ms1_peak_coord[i], fdc_idxs[i]] += rel_iso_int[i]

    # Build dia_spec_int using summed intensities: for each observed row that
    # was matched by a library peak, replace its intensity with the sum of all
    # observed peaks within tolerance of that library peak.
    dia_spec_int = dia_spectrum[:, 1].copy()
    for i in range(n_lib):
        if lib_peaks_matched[i]:
            row = matched_obs_idx[i]
            if summed_obs[i] > dia_spec_int[row]:
                dia_spec_int[row] = summed_obs[i]

    n_rows = dense_matrix.shape[0]
    if len(dia_spec_int) > n_rows:
        dia_spec_int = dia_spec_int[:n_rows]
    elif len(dia_spec_int) < n_rows:
        dia_spec_int = np.append(dia_spec_int, [0]*(n_rows - dia_spec_int.shape[0]))

    # dia_spec_int = np.append(dia_spec_int,[0]*(dense_matrix.shape[0]-dia_spec_int.shape[0])) 

    return dense_matrix, dia_spec_int, lib_peaks_matched, fdc_idxs, group_lengths

