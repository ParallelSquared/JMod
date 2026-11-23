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

"""Bruker timsTOF .d input: calibration, DIA windows, the raw-ion readers, MS2 IM
banding, and the Bruker SDK location.  ``file_reader.FileReader``
dispatches .d input here (``_load_d``)."""

import os
import sqlite3
import threading
from collections import OrderedDict
from time import perf_counter

import numpy as np
import numba as nb
import polars as pl

from src.utils.errors import JModError
from src.utils.io.load_files import (
    Spectrum, SpectrumFile, PEAK_MZ_DTYPE, PEAK_INT_DTYPE, PEAK_MOB_DTYPE,
)
from src.utils.io.tdf_bin import TdfBinReader, frame_peak_counts
from src.logger import logger


# ---------------------------------------------------------------------------
# MS2 IM bands with each peak stored once.
#
# The IM bands overlap by 50% (stride = width / 2), so almost every peak belongs to
# two adjacent bands.  Copying each band's peaks out doubled the banded MS2 (~69 GB
# on a 20 ng 5plex).  Instead each (frame, window) group keeps one buffer of its
# peaks, ordered by which bands contain them:
#
#     key 2b      peaks in band b only (edges of the band range)
#     key 2b + 1  peaks in bands b and b + 1
#
# and within a key by (m/z, mobility, intensity).  Band b is then exactly the keys
# 2b - 1, 2b and 2b + 1: three consecutive runs of the buffer, each sorted.  A
# band's peak arrays (``BandSpectrum.mz`` / ``intens`` / ``mobility``) are those
# runs merged by (m/z, mobility, intensity) -- the band's peaks in that order, the
# same arrays a per-band copy sorted on that key holds -- built on demand and
# cached per thread for a few bands.  Kernels that look up ions in many bands
# (fragment correlations) read the runs directly instead (``gather_range``).
#
# Membership is the test the per-band copies used: ``lo <= mob < hi`` on the
# float64 mobility, for the two candidate bins of each peak (the 50% overlap,
# ``width = 2 * stride``, puts a peak in at most two).
# ---------------------------------------------------------------------------


@nb.njit(cache=False, nogil=True)
def band_keys(mob_arr, im_bins):
    """Band-membership key per peak (see the module docstring), -1 if in no band."""
    n = mob_arr.shape[0]
    out = np.full(n, -1, dtype=np.int64)
    bin_lo0 = im_bins[0, 0]
    stride = im_bins[1, 0] - im_bins[0, 0]
    n_bins = im_bins.shape[0]
    for i in range(n):
        m = np.float64(mob_arr[i])
        r = np.int64((m - bin_lo0) / stride)
        in_r = r >= 0 and r < n_bins and im_bins[r, 0] <= m and m < im_bins[r, 1]
        q = r - 1
        in_q = q >= 0 and q < n_bins and im_bins[q, 0] <= m and m < im_bins[q, 1]
        if in_r and in_q:
            out[i] = 2 * q + 1
        elif in_r:
            out[i] = 2 * r
        elif in_q:
            out[i] = 2 * q
    return out


@nb.njit(cache=False, nogil=True, inline="always")
def _less(mz, mob, it, a, b):
    """Peak a before peak b in (m/z, mobility, intensity) order."""
    if mz[a] != mz[b]:
        return mz[a] < mz[b]
    if mob[a] != mob[b]:
        return mob[a] < mob[b]
    return it[a] < it[b]


@nb.njit(cache=False, nogil=True)
def _merge3(mz, it, mob, a, a1, b, b1, c, c1):
    """The peaks [a,a1), [b,b1), [c,c1) of a group buffer, each (m/z, mob, int)-
    sorted, merged into new arrays in that order.  ``mob`` may be empty (no
    mobility), only when at most one range is non-empty (then nothing is compared)."""
    n = (a1 - a) + (b1 - b) + (c1 - c)
    omz = np.empty(n, dtype=mz.dtype)
    oit = np.empty(n, dtype=it.dtype)
    omob = np.empty(n if mob.shape[0] else 0, dtype=mob.dtype)
    for k in range(n):
        # which range supplies the next peak (by range, not position: an exhausted
        # range's cursor can equal the next range's start)
        run = -1
        best = -1
        if a < a1:
            run, best = 0, a
        if b < b1 and (best < 0 or _less(mz, mob, it, b, best)):
            run, best = 1, b
        if c < c1 and (best < 0 or _less(mz, mob, it, c, best)):
            run, best = 2, c
        omz[k] = mz[best]
        oit[k] = it[best]
        if mob.shape[0]:
            omob[k] = mob[best]
        if run == 0:
            a += 1
        elif run == 1:
            b += 1
        else:
            c += 1
    return omz, oit, omob


@nb.njit(cache=False, nogil=True)
def merge_runs(mz, it, mob, s0, s1, s2, s3):
    """The band of runs [s0,s1), [s1,s2), [s2,s3), merged (see ``_merge3``)."""
    return _merge3(mz, it, mob, s0, s1, s1, s2, s2, s3)


@nb.njit(cache=False, nogil=True)
def gather_range(gmz, git, gmob, s0, s1, s2, s3, lo, hi):
    """The band's peaks with ``lo <= mz <= hi`` in the band's order.

    Exactly the merged band's ``[searchsorted(mz, lo), searchsorted(mz, hi,
    side='right'))`` slice.  When the matches lie in one run (always, for
    spectra stored the usual way, passed as s0 = s1 = 0, s2 = s3 = n) they are
    returned as views; otherwise the runs' matches are merged (they are few).
    ``gmob`` may be empty (no mobility); the mobility returned is then empty."""
    p0 = s0 + np.searchsorted(gmz[s0:s1], lo)
    p1 = s0 + np.searchsorted(gmz[s0:s1], hi, side="right")
    q0 = s1 + np.searchsorted(gmz[s1:s2], lo)
    q1 = s1 + np.searchsorted(gmz[s1:s2], hi, side="right")
    r0 = s2 + np.searchsorted(gmz[s2:s3], lo)
    r1 = s2 + np.searchsorted(gmz[s2:s3], hi, side="right")
    nonempty = int(p1 > p0) + int(q1 > q0) + int(r1 > r0)
    if nonempty <= 1:
        # one range (or none): a view; the empty ranges add nothing
        a = p0 if p1 > p0 else (q0 if q1 > q0 else r0)
        b = p1 if p1 > p0 else (q1 if q1 > q0 else r1)
        return gmz[a:b], git[a:b], gmob[a:b] if gmob.shape[0] else gmob
    return _merge3(gmz, git, gmob, p0, p1, q0, q1, r0, r1)


@nb.njit(cache=False, nogil=True)
def nearest_peak(gmz, git, gmob, s0, s1, s2, s3, q):
    """The merged band's neighbours of ``q``: (index of the last peak with
    mz < q, of the first with mz >= q) as buffer positions, -1 if none.  The
    merged band's ``pos - 1`` and ``pos`` for ``pos = searchsorted(mz, q)``."""
    pred = -1
    succ = -1
    for k in range(3):
        if k == 0:
            a, b = s0, s1
        elif k == 1:
            a, b = s1, s2
        else:
            a, b = s2, s3
        if b <= a:
            continue
        pos = a + np.searchsorted(gmz[a:b], q)
        if pos > a:
            if pred < 0 or _less(gmz, gmob, git, pred, pos - 1):
                pred = pos - 1
        if pos < b:
            if succ < 0 or _less(gmz, gmob, git, pos, succ):
                succ = pos
    return pred, succ


_tls = threading.local()
_CACHE_BANDS = 64


class BandSpectrum(Spectrum):
    """An MS2 IM band whose peaks are three runs of its (frame, window) group's
    buffer (see the module docstring).  ``mz`` / ``intens`` / ``mobility`` are the
    merged band, built on first use and cached per thread (``_CACHE_BANDS``
    bands), so reading all three costs one merge.  Assigning one of them
    replaces it for this spectrum (the others still read the band); once all
    three are assigned the spectrum no longer refers to the group buffer, so the
    buffer is freed when no band uses it."""

    def __init__(self, group, runs):
        # Spectrum's attributes, without its mz/intens/mobility = None (properties here)
        self.id = None
        self.level = None
        self.RT = None
        self.collision_energy = None
        self.TIC = None
        self.isolation_window = None
        self.im_lo = None
        self.im_hi = None
        self._group = group          # (mz, intens, mob) buffers of the (frame, window) group
        self._runs = runs            # (s0, s1, s2, s3)
        self._own = None             # [mz, intens, mob], each set once assigned

    def _band(self):
        """The merged band's (mz, intens, mob)."""
        s0, s1, s2, s3 = self._runs
        mz, it, mob = self._group
        # one non-empty run: the band is a view of the buffer
        nonempty = (s1 > s0) + (s2 > s1) + (s3 > s2)
        if nonempty <= 1:
            return mz[s0:s3], it[s0:s3], mob[s0:s3]
        cache = getattr(_tls, "bands", None)
        if cache is None:
            cache = _tls.bands = OrderedDict()
        key = id(self)
        hit = cache.get(key)
        if hit is not None and hit[0] is self:
            cache.move_to_end(key)
            return hit[1]
        arrays = merge_runs(mz, it, mob, s0, s1, s2, s3)
        cache[key] = (self, arrays)
        if len(cache) > _CACHE_BANDS:
            cache.popitem(last=False)
        return arrays

    def _arrays(self):
        if self._own is None:
            return self._band()
        if self._group is None:
            return tuple(self._own)
        return tuple(o if o is not None else b for o, b in zip(self._own, self._band()))

    def _assign(self, i, value):
        if self._own is None:
            self._own = [None, None, None]
        self._own[i] = value
        if all(a is not None for a in self._own):
            self._group = self._runs = None      # own arrays only: release the group buffer

    @property
    def mz(self):
        return self._arrays()[0]

    @mz.setter
    def mz(self, value):
        self._assign(0, value)

    @property
    def intens(self):
        return self._arrays()[1]

    @intens.setter
    def intens(self, value):
        self._assign(1, value)

    @property
    def mobility(self):
        return self._arrays()[2]

    @mobility.setter
    def mobility(self, value):
        self._assign(2, value)

# Bruker's timsdata SDK (5.0.4) unpacks to a folder holding exactly two platform
# directories -- win64/ and linux64/ -- alongside include/, examples/ and
# thirdparty/. Bruker publishes no macOS build, so there is deliberately no
# .dylib name and no 32-bit directory here; both would be dead entries.
# macOS has no Bruker build: the Windows library runs under Wine there.
_BRUKER_LIB_NAME = {"win32": "timsdata.dll", "darwin": "timsdata.dll"}
_BRUKER_LIB_DEFAULT = "libtimsdata.so"
_BRUKER_PLATFORM_DIR = {"win32": "win64", "darwin": "win64"}
_BRUKER_PLATFORM_DIR_DEFAULT = "linux64"
# Deep enough to find <sdk_root>/linux64/lib when pointed one level above the
# root; shallow enough that pointing at a large directory stays cheap.
_BRUKER_SDK_MAX_DEPTH = 2


def compute_im_bins(im_lo, im_hi, width=0.05, stride=0.025):
    """Return (N, 2) array of [bin_lo, bin_hi] for overlapping IM bins.

    ``width = 2 * stride`` (50% overlap) is required by ``band_keys``,
    which only tests two candidate bins per peak.
    """
    starts = np.arange(im_lo, im_hi - width + stride * 0.5, stride)
    return np.column_stack([starts, starts + width])


def _load_d(filepath: str, bruker_sdk_path: str = None) -> SpectrumFile:
    """Load timsTOF .d data: the raw ions in analysis.tdf_bin, read uncentroided
    and calibrated with tables computed by Bruker's timsdata (see
    ``_sdk_calibration_tables``).  A peaks.parquet in the .d is ignored."""
    logger.info("Loading Spectra... from .d file")
    d_path = filepath.rstrip("/")

    tdf_path = os.path.join(d_path, "analysis.tdf")
    bin_path = os.path.join(d_path, "analysis.tdf_bin")

    if not os.path.exists(tdf_path):
        raise FileNotFoundError(f"analysis.tdf not found in {d_path}")

    if not os.path.exists(bin_path):
        raise FileNotFoundError(f"{bin_path} not found; there are no peaks to read.")

    logger.info(f"Loading calibration from {d_path}")
    cal = _load_calibration(tdf_path)
    logger.info(f"Reading the raw, uncentroided ions of {d_path} with Bruker's calibration")
    _load_sdk_tables(_sdk_calibration_tables(d_path, bruker_sdk_path), cal)
    dia_lookup = _load_dia_windows(tdf_path, cal["frame_ids"], cal["ms_level"])

    im_bins = compute_im_bins(cal["im_range_lower"], cal["im_range_upper"])
    logger.info(f"Computed {len(im_bins)} overlapping IM bins "
                f"[{cal['im_range_lower']:.4f}, {cal['im_range_upper']:.4f}]")

    sf = _build_spectrum_file(filepath, d_path, cal, dia_lookup, im_bins)

    logger.info(f"Loaded {len(sf.ms1scans)} MS1 spectra, {len(sf.ms2scans)} MS2 spectra")
    return sf


# Bruker's m/z and 1/K0 tables for a .d, cached inside it, as written by
# scripts/bruker_calibration_tables.py.
_SDK_TABLES_DIR = "bruker_calibration"
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_SDK_TABLES_SCRIPT = os.path.join(_REPO_ROOT, "scripts", "bruker_calibration_tables.py")

# macOS: Bruker ships no macOS library, so the Windows timsdata.dll from
# --bruker_sdk_path runs under Wine with a Windows Python, both set up by the user
# (neither is bundled; Wine is LGPL): ``wine`` on PATH, the prefix Wine itself
# picks (WINEPREFIX, default ~/.wine), and ``python`` on that prefix's Windows
# PATH.  The helper script needs only the standard library.
_MAC_WINE_REQUIREMENTS = (
    "Reading .d data on macOS runs Bruker's Windows timsdata library under Wine, "
    "which needs: wine on PATH; and a Windows Python installed in your Wine prefix "
    "(WINEPREFIX, default ~/.wine) and on its PATH, so that 'wine python --version' "
    "works.")


def _wine_path(path: str) -> str:
    """A macOS path as Wine's Z: drive sees it."""
    return "Z:" + os.path.abspath(path).replace("/", "\\")


def _sdk_calibration_tables(d_path: str, bruker_sdk_path: str = None) -> str:
    """Return the folder of Bruker calibration tables for ``d_path``, making it
    on first use.

    The tables are computed by calling Bruker's unmodified timsdata library
    through scripts/bruker_calibration_tables.py: natively with
    ``bruker_sdk_path`` on Windows or Linux, under Wine on a Mac.
    """
    import shutil
    import subprocess
    import sys

    tables_dir = os.path.join(d_path, _SDK_TABLES_DIR)
    if os.path.isfile(os.path.join(tables_dir, "calibration.json")):
        return tables_dir

    # Write beside the target and rename, so an interrupted run cannot leave
    # partial tables that the next run would treat as complete.
    tmp_dir = tables_dir + ".partial"
    if not bruker_sdk_path:
        _fail_bruker(f"Reading {d_path} raw needs Bruker's calibration")
    if sys.platform == "darwin":
        wine = shutil.which("wine")
        if wine is None:
            raise JModError(f"Cannot compute Bruker calibration for {d_path}: wine is not "
                            f"on PATH.\n\n{_MAC_WINE_REQUIREMENTS}")
        cmd = [wine, "python", _wine_path(_SDK_TABLES_SCRIPT),
               _wine_path(bruker_sdk_path), _wine_path(d_path), _wine_path(tmp_dir)]
        env = dict(os.environ, WINEDEBUG="-all")
    else:
        cmd = [sys.executable, _SDK_TABLES_SCRIPT, bruker_sdk_path, d_path, tmp_dir]
        env = None

    logger.info(f"Computing Bruker calibration tables -> {tables_dir}")
    t0 = perf_counter()
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if result.returncode != 0 or not os.path.isfile(
            os.path.join(tmp_dir, "calibration.json")):
        hint = f"\n\n{_MAC_WINE_REQUIREMENTS}" if sys.platform == "darwin" else ""
        raise RuntimeError(
            f"Computing Bruker calibration for {d_path} failed "
            f"(exit {result.returncode}):\n{result.stdout}\n{result.stderr}{hint}")
    os.replace(tmp_dir, tables_dir)
    logger.info(f"Bruker calibration: {result.stdout.strip()} "
                f"({perf_counter() - t0:.0f}s)")
    return tables_dir


def _load_sdk_tables(tables_dir: str, cal: dict) -> None:
    """Swap ``cal``'s estimated m/z and 1/K0 lookups for Bruker's tables.

    The tables replace ``cal["mz_values"]`` (by TOF index) and
    ``cal["mobility_values"]`` (by scan number) one for one, so they must cover
    exactly the same index ranges as the estimates they replace.
    """
    mz_values = np.load(os.path.join(tables_dir, "mz.npy"))
    mobility_values = np.load(os.path.join(tables_dir, "ook0.npy"))
    for name, table, estimate in (("mz", mz_values, cal["mz_values"]),
                                  ("ook0", mobility_values, cal["mobility_values"])):
        if table.dtype != np.float64 or table.shape != estimate.shape:
            raise ValueError(f"{name}.npy in {tables_dir} is {table.dtype} "
                             f"{table.shape}; expected float64 {estimate.shape}")
    cal["mz_values"] = mz_values
    cal["mobility_values"] = mobility_values


def _load_calibration(tdf_path: str) -> dict:
    """Load calibration data from analysis.tdf SQLite database.

    Uses estimation mode (linear interpolation) for m/z and mobility.
    """
    conn = sqlite3.connect(tdf_path)
    try:
        rows = conn.execute(
            "SELECT Key, Value FROM GlobalMetadata WHERE Key IN "
            "('DigitizerNumSamples', 'MzAcqRangeLower', 'MzAcqRangeUpper', "
            "'OneOverK0AcqRangeLower', 'OneOverK0AcqRangeUpper', "
            "'AcquisitionSoftware')"
        ).fetchall()
        meta = {k: v for k, v in rows}

        frame_rows = conn.execute(
            "SELECT Id, Time, NumScans, MsMsType FROM Frames ORDER BY Id"
        ).fetchall()
    finally:
        conn.close()

    frame_ids = np.array([r[0] for r in frame_rows], dtype=np.uint32)
    rt_values = np.array([r[1] for r in frame_rows], dtype=np.float64)
    num_scans = np.array([r[2] for r in frame_rows], dtype=np.int64)
    msms_types = np.array([r[3] for r in frame_rows], dtype=np.int64)

    scan_max_index = int(num_scans.max()) + 1
    tof_max_index = int(meta["DigitizerNumSamples"]) + 1

    # Mobility: linear, reversed (higher scan index = lower mobility)
    mobility_min = float(meta["OneOverK0AcqRangeLower"])
    mobility_max = float(meta["OneOverK0AcqRangeUpper"])
    mobility_values = mobility_max - (
        (mobility_max - mobility_min) / scan_max_index * np.arange(scan_max_index)
    )

    # m/z: quadratic TOF interpolation in sqrt(m/z) space
    mz_min = float(meta["MzAcqRangeLower"])
    mz_max = float(meta["MzAcqRangeUpper"])
    if meta.get("AcquisitionSoftware") == "Bruker otofControl":
        mz_min -= 5
        mz_max += 5
    tof_intercept = np.sqrt(mz_min)
    tof_slope = (np.sqrt(mz_max) - tof_intercept) / tof_max_index
    mz_values = (tof_intercept + tof_slope * np.arange(tof_max_index)) ** 2

    ms_level = np.where(msms_types == 0, 1, 2).astype(np.uint8)

    return {
        "mz_values": mz_values,
        "mobility_values": mobility_values,
        "rt_values": rt_values,
        "frame_ids": frame_ids,
        "ms_level": ms_level,
        "im_range_lower": mobility_min,
        "im_range_upper": mobility_max,
    }


def _load_dia_windows(tdf_path: str, frame_ids: np.ndarray, ms_level: np.ndarray) -> pl.DataFrame:
    """Load DIA window scheme and build a lookup table for MS2 peak assignment.

    Returns a polars DataFrame with columns:
        frame (UInt32), scan_begin (UInt32), scan_end (UInt32),
        prec_mz (Float64), isolation_width (Float64), collision_energy (Float64)
    """
    conn = sqlite3.connect(tdf_path)
    try:
        # DiaFrameMsMsInfo: Frame -> WindowGroup
        info_rows = conn.execute(
            "SELECT Frame, WindowGroup FROM DiaFrameMsMsInfo"
        ).fetchall()

        # DiaFrameMsMsWindows: WindowGroup -> scan ranges + isolation params
        window_rows = conn.execute(
            "SELECT WindowGroup, ScanNumBegin, ScanNumEnd, "
            "IsolationMz, IsolationWidth, CollisionEnergy "
            "FROM DiaFrameMsMsWindows"
        ).fetchall()
    finally:
        conn.close()

    info_df = pl.DataFrame({
        "frame": [r[0] for r in info_rows],
        "window_group": [r[1] for r in info_rows],
    }).cast({"frame": pl.UInt32, "window_group": pl.Int64})

    windows_df = pl.DataFrame({
        "window_group": [r[0] for r in window_rows],
        "scan_begin": [r[1] for r in window_rows],
        "scan_end": [r[2] for r in window_rows],
        "prec_mz": [r[3] for r in window_rows],
        "isolation_width": [r[4] for r in window_rows],
        "collision_energy": [r[5] for r in window_rows],
    }).cast({
        "window_group": pl.Int64,
        "scan_begin": pl.UInt32,
        "scan_end": pl.UInt32,
    })

    # Join: each MS2 frame gets its window definitions
    dia_lookup = info_df.join(windows_df, on="window_group").drop("window_group")
    return dia_lookup


def _drain_sorted(groups):
    """``groups``' items in sorted key order, each removed from the dict as it is
    yielded, so a window's un-banded arrays are released once banded."""
    for key in sorted(groups):
        yield key, groups.pop(key)


def _band_ms2_groups(sf, items, im_bins, scan_counter):
    """Segment each (frame, window) group into IM bands, one Spectrum per band.

    ``items`` yields ``((rt, prec_mz, iso_width, ce), (mz, intens, mob))``, the
    concatenated un-banded peaks of each window, in sorted key order (scan numbers
    follow it): ``_iter_raw_ms2_groups``, which streams them so only one RT's
    un-banded peaks exist at a time.  Appends a
    Spectrum per band to ``sf.ms2scans`` and updates ``sf.ms2_by_id`` /
    ``sf.scan_pos``; returns the next free scan number and the number of windows.

    A peak lies in up to 2 overlapping fixed IM bins but is stored once: the
    bands are ``BandSpectrum`` views of their window's buffer
    (the band layout at the top of this module).
    """
    # TODO: consider compact peak types (TOF index uint32 + scan uint16 +
    # intensity uint16/uint32 with SDK-table lookups, 8-10 B per peak vs 12 B).
    n_windows = 0
    for key, (mz_concat, intens_concat, mob_concat) in items:
        n_windows += 1
        rt_val, prec_mz_val, iso_width, ce = key
        if len(mz_concat) == 0:
            continue

        # Each peak once, in its group's buffer ordered by band membership then
        # (m/z, mobility, intensity); band b is the buffer's keys 2b-1..2b+1
        # (the band layout at the top of this module).
        keys = band_keys(mob_concat, im_bins)
        keep = keys >= 0
        if not keep.any():
            continue
        keys = keys[keep]
        g_mz = mz_concat[keep]
        g_int = intens_concat[keep]
        g_mob = mob_concat[keep]
        order = np.lexsort((g_int, g_mob, g_mz, keys))
        keys = keys[order]
        group = (g_mz[order], g_int[order], g_mob[order])
        del g_mz, g_int, g_mob, order, keep

        half_width = iso_width / 2.0
        b_lo = int(keys[0]) // 2
        b_hi = int(keys[-1]) // 2 + int(keys[-1]) % 2
        # band edges in the buffer, for every candidate band at once
        cand = np.arange(b_lo, b_hi + 1)
        e = np.searchsorted(keys, np.stack([2 * cand - 1, 2 * cand, 2 * cand + 1, 2 * cand + 2]))
        for j, b in enumerate(cand):
            s0, s1, s2, s3 = (int(x) for x in e[:, j])
            if s3 <= s0:
                continue                     # no peak in this band
            spec = BandSpectrum(group, (s0, s1, s2, s3))
            mz_sorted, intens_sorted, _ = spec._arrays()

            spec.id = f"scan={scan_counter}"
            spec.scan_num = scan_counter
            spec.level = 2
            spec.RT = rt_val
            spec.TIC = float(intens_sorted.sum(dtype=np.float64))
            spec.injection_time = 1.0
            spec.collision_energy = ce
            spec.prec_mz = prec_mz_val
            spec.isolation_window = {
                "isolation window target m/z": prec_mz_val,
                "isolation window lower offset": half_width,
                "isolation window upper offset": half_width,
            }
            spec.ms1window = prec_mz_val + np.array([-1, 1]) * half_width
            spec.im_lo = float(im_bins[b, 0])
            spec.im_hi = float(im_bins[b, 1])
            spec.scanwindow = [float(mz_sorted[0]), float(mz_sorted[-1])]
            del mz_sorted, intens_sorted

            idx = len(sf.ms2scans)
            sf.ms2scans.append(spec)
            sf.ms2_by_id[scan_counter] = idx
            sf.scan_pos[scan_counter] = [2, idx]
            scan_counter += 1
    return scan_counter, n_windows


def _frame_tables(cal: dict):
    """RT and MS level indexed by frame ID (level 0: frame not in the calibration table)."""
    max_frame = int(cal["frame_ids"].max())
    rt_by_frame = np.zeros(max_frame + 1, dtype=np.float64)
    ms_level_by_frame = np.zeros(max_frame + 1, dtype=np.uint8)
    for i, fid in enumerate(cal["frame_ids"]):
        rt_by_frame[int(fid)] = cal["rt_values"][i]
        ms_level_by_frame[int(fid)] = cal["ms_level"][i]
    return rt_by_frame, ms_level_by_frame


def _windows_by_frame(dia_lookup: pl.DataFrame) -> dict:
    """frame -> [(scan_begin, scan_end, prec_mz, iso_width, ce)], its DIA windows."""
    windows = {}
    for frame, scan_begin, scan_end, prec_mz, iso_width, ce in dia_lookup.iter_rows():
        windows.setdefault(int(frame), []).append(
            (int(scan_begin), int(scan_end), float(prec_mz), float(iso_width), float(ce)))
    return windows


def _add_window_peaks(groups, rt, windows, scans, mz, intens, mob):
    """Append one frame's peaks to its DIA windows' groups in ``groups``, keyed
    ``(rt, prec_mz, iso_width, ce)``, each a ``([mz], [intens], [mob])`` of chunks."""
    for scan_begin, scan_end, prec_mz, iso_width, ce in windows:
        win = (scans >= scan_begin) & (scans < scan_end)
        if not win.any():
            continue
        g = groups.setdefault((rt, prec_mz, iso_width, ce), ([], [], []))
        g[0].append(mz[win])
        g[1].append(intens[win])
        g[2].append(mob[win])


def _read_raw_frame(reader, fid: int, cal: dict, min_intensity: int):
    """One frame's ions from ``analysis.tdf_bin``, calibrated:
    ``(scans, mz, intens, mob)``, or None if none are above ``min_intensity``.

    Calibration is whatever ``cal["mz_values"]`` / ``cal["mobility_values"]``
    hold.  By default that is estimation mode, the lookups ``_load_calibration``
    builds from ``GlobalMetadata``, which approximate the instrument response
    and so do *not* match the Bruker SDK.
    ``_load_d`` replaces them with tables Bruker's timsdata computed
    (``_load_sdk_tables``), so m/z and 1/K0 match the SDK.

    ``min_intensity`` drops peaks at or below a raw detector count.  It defaults
    to 0 (keep everything), since recovering the signal the centroider discards
    is the point of reading raw.
    """
    scans, tof_indices, intensities = reader.read_frame(fid)
    if min_intensity > 0:
        keep = intensities > min_intensity
        scans = scans[keep]
        tof_indices = tof_indices[keep]
        intensities = intensities[keep]
    if scans.shape[0] == 0:
        return None
    # Indices come from the instrument and should always be in range; clipping
    # keeps a corrupt frame from taking the whole run down.
    mz_values = cal["mz_values"]
    mobility_values = cal["mobility_values"]
    mz = mz_values[np.clip(tof_indices, 0, len(mz_values) - 1)].astype(PEAK_MZ_DTYPE)
    mob = mobility_values[np.clip(scans, 0, len(mobility_values) - 1)].astype(PEAK_MOB_DTYPE)
    return scans, mz, intensities.astype(PEAK_INT_DTYPE), mob


def _read_raw_ms1_groups(d_path: str, cal: dict, min_intensity: int = 0) -> dict:
    """The raw MS1 frames of ``analysis.tdf_bin``: ``rt -> ([mz], [intens], [mob])``,
    one frame per RT, every stored ion (no centroiding).  MS2 is streamed
    separately (``_iter_raw_ms2_groups``).
    """
    rt_by_frame, ms_level_by_frame = _frame_tables(cal)
    logger.info(f"Reading raw MS1 frames from {d_path}/analysis.tdf_bin "
                f"({frame_peak_counts(d_path)} stored peaks)")
    ms1_groups = {}
    kept = 0
    with TdfBinReader(d_path) as reader:
        for frame_id in reader.frame_ids:
            fid = int(frame_id)
            if ms_level_by_frame[fid] != 1:
                continue
            peaks = _read_raw_frame(reader, fid, cal, min_intensity)
            if peaks is None:
                continue
            _scans, mz, intens, mob = peaks
            kept += mz.shape[0]
            g = ms1_groups.setdefault(float(rt_by_frame[fid]), ([], [], []))
            g[0].append(mz)
            g[1].append(intens)
            g[2].append(mob)
    logger.info(f"Read {kept} raw MS1 peaks in {len(ms1_groups)} MS1 frames")
    return ms1_groups


def _iter_raw_ms2_groups(d_path: str, cal: dict, dia_lookup: pl.DataFrame,
                         min_intensity: int = 0):
    """Stream the raw MS2 (frame, window) groups of ``analysis.tdf_bin``, one RT at a time.

    Yields ``(key, (mz, intens, mob))`` with ``key = (rt, prec_mz, iso_width, ce)``
    in sorted key order, the form ``_band_ms2_groups`` consumes.  A (frame,
    window) group comes from a single frame, so every group of an RT is complete
    once that RT's frames are read: frames are visited in RT order (stable, so
    equal-RT frames keep reader order) and each RT's groups are yielded sorted.
    Only one RT's un-banded peaks are held at a time, instead of the whole run's
    (~35 GB on a 20 ng timsTOF .d).
    """
    rt_by_frame, ms_level_by_frame = _frame_tables(cal)
    windows = _windows_by_frame(dia_lookup)
    logger.info(f"Reading raw MS2 frames from {d_path}/analysis.tdf_bin, banding as they are read")
    kept = 0
    n_groups = 0
    with TdfBinReader(d_path) as reader:
        frames = sorted((int(f) for f in reader.frame_ids
                         if ms_level_by_frame[int(f)] == 2 and int(f) in windows),
                        key=lambda f: rt_by_frame[f])
        i = 0
        while i < len(frames):
            rt = float(rt_by_frame[frames[i]])
            groups = {}
            while i < len(frames) and float(rt_by_frame[frames[i]]) == rt:
                fid = frames[i]
                i += 1
                peaks = _read_raw_frame(reader, fid, cal, min_intensity)
                if peaks is None:
                    continue
                scans, mz, intens, mob = peaks
                kept += scans.shape[0]
                _add_window_peaks(groups, rt, windows[fid], scans, mz, intens, mob)
            for key, (mz, intens, mob) in _drain_sorted(groups):
                n_groups += 1
                yield key, (np.concatenate(mz), np.concatenate(intens), np.concatenate(mob))
    logger.info(f"Read {kept} raw MS2 peaks in {n_groups} MS2 windows")


def _build_spectrum_file(filepath: str, raw_d_path: str, cal: dict, dia_lookup: pl.DataFrame,
                         im_bins: np.ndarray, min_intensity: int = 0) -> SpectrumFile:
    """Assign the .d's raw ions to DIA windows and IM bins, and build a SpectrumFile.

    The ions are decoded frame by frame from analysis.tdf_bin and calibrated here
    (see ``_read_raw_frame``).  Each non-empty IM bin produces a separate Spectrum
    object.
    """
    # MS1 now; MS2 is streamed into banding below, one RT at a time, so the
    # whole run's un-banded MS2 peaks are never held at once.
    ms1_groups = _read_raw_ms1_groups(raw_d_path, cal, min_intensity)
    ms2_groups = _iter_raw_ms2_groups(raw_d_path, cal, dia_lookup, min_intensity)

    # Build SpectrumFile from accumulated groups
    sf = SpectrumFile()
    sf.filename = filepath
    sf.has_ion_mobility = True
    sf.scan_pos = {}
    sf.ms1scans = []
    sf.ms2scans = []
    sf.ms1_by_id = {}
    sf.ms2_by_id = {}

    scan_counter = 1

    # Global IM range shared by all MS1 frames, so downstream (im_lo, im_hi)
    # equality checks treat every frame as the same IM context (IM is now
    # resolved per-peak via spec.mobility, not by binning).
    im_lo_global = float(im_bins[:, 0].min())
    im_hi_global = float(im_bins[:, 1].max())

    # MS1 spectra: one per frame (RT), peaks carry per-peak mobility.
    logger.info("Building MS1 spectra (one per frame, per-peak mobility)")
    for rt_val in sorted(ms1_groups.keys()):
        mz_list, intens_list, mob_list = ms1_groups[rt_val]
        mz_concat = np.concatenate(mz_list)
        intens_concat = np.concatenate(intens_list)
        mob_concat = np.concatenate(mob_list)
        if len(mz_concat) == 0:
            continue

        # Sort by m/z (mobility kept parallel so spec.mobility[i] matches mz[i])
        order = np.argsort(mz_concat)
        mz_sorted = mz_concat[order]
        intens_sorted = intens_concat[order]
        mob_sorted = mob_concat[order]

        spec = Spectrum()
        spec.id = f"scan={scan_counter}"
        spec.scan_num = scan_counter
        spec.level = 1
        spec.RT = rt_val
        spec.mz = mz_sorted
        spec.intens = intens_sorted
        spec.mobility = mob_sorted
        spec.TIC = float(intens_sorted.sum(dtype=np.float64))
        spec.injection_time = 1.0
        spec.collision_energy = None
        spec.isolation_window = None
        spec.im_lo = im_lo_global
        spec.im_hi = im_hi_global
        spec.scanwindow = [float(mz_sorted[0]), float(mz_sorted[-1])]

        idx = len(sf.ms1scans)
        sf.ms1scans.append(spec)
        sf.ms1_by_id[scan_counter] = idx
        sf.scan_pos[scan_counter] = [1, idx]
        scan_counter += 1

    # MS2 spectra: segment each (frame, window) into IM bands, one Spectrum per
    # band, retaining per-peak mobility. Overlapping fixed-bin mode (the pre-
    # watershed "overlapping windows"): each peak lands in up to 2 overlapping
    # fixed IM bins (denormalized). Swap the `bands = ...` line to re-enable the
    # data-driven watershed.
    # These bands are provisional: reband_ms2 redraws them once the preliminary
    # search has fitted the IM precision.  Rather than keeping the un-banded peaks
    # in memory for that (tens of GB on a large .d, untouched across the whole
    # initial search), record what it takes to re-read them from the .d.
    sf._raw_d_path = raw_d_path
    sf._min_intensity = min_intensity
    sf._cal = cal
    sf._dia_lookup = dia_lookup
    sf._im_range = (float(im_bins[:, 0].min()), float(im_bins[:, 1].max()))

    logger.info("Building MS2 spectra with overlapping fixed IM bins")
    scan_counter, n_windows = _band_ms2_groups(sf, ms2_groups, im_bins, scan_counter)
    logger.info(
        f"MS2 scans after IM binning: {len(sf.ms2scans)} "
        f"(original windows: {n_windows})"
    )

    sf.build_ms2_to_ms1_map()
    return sf


def reband_ms2(sf: SpectrumFile, width: float) -> bool:
    """Rebuild the MS2 IM bands at a new width, in place.

    The bands built at load time use a hardcoded width, chosen before anything
    about the data's mobility resolution is known.  Once the IM precision has
    been fitted from the preliminary search, the bands can be redrawn to match
    it.  Call between the preliminary search and the main search.

    ``width`` is the full band width; bins overlap by 50%, so the stride is
    ``width / 2``.  That relationship is required by ``band_keys``,
    which only ever tests two candidate bins per peak.

    The un-banded peaks are not kept in memory for this -- they are streamed again
    from the .d, one RT at a time, at the cost of the initial MS2 read.

    Returns True if the bands were rebuilt, False if there was nothing to do
    (non-IM data, i.e. anything but the .d path).
    """
    im_range = getattr(sf, "_im_range", None)
    if getattr(sf, "_raw_d_path", None) is None or im_range is None:
        logger.info("Re-banding MS2: not a .d acquisition; skipping")
        return False
    if not np.isfinite(width) or width <= 0:
        logger.info(f"Re-banding MS2: invalid width {width}; skipping")
        return False

    im_lo, im_hi = im_range
    new_bins = compute_im_bins(im_lo, im_hi, width=width, stride=width / 2.0)
    if len(new_bins) == 0:
        logger.info(f"Re-banding MS2: width {width:.5f} yields no bins over "
                    f"[{im_lo:.4f}, {im_hi:.4f}]; keeping existing bands")
        return False

    n_before = len(sf.ms2scans)

    # Drop the old MS2 spectra and their index entries *before* re-reading, so the
    # old bands are freed rather than sitting alongside the un-banded peaks and the
    # new bands. MS1 keeps its scan numbers, so MS2 renumbering restarts above the
    # highest MS1 number to stay disjoint -- scan_pos is a single dict shared by
    # both levels.
    for scan_num in list(sf.ms2_by_id):
        sf.scan_pos.pop(scan_num, None)
    sf.ms2scans = []
    sf.ms2_by_id = {}
    # The flattened MS2 peak arrays (if any) describe the bands we just dropped.
    sf._ms2_flat = None
    scan_counter = (max(sf.ms1_by_id) + 1) if sf.ms1_by_id else 1

    logger.info("Re-banding MS2: re-reading un-banded peaks")
    _band_ms2_groups(sf, _iter_raw_ms2_groups(sf._raw_d_path, sf._cal, sf._dia_lookup,
                                              sf._min_intensity),
                     new_bins, scan_counter)
    sf.build_ms2_to_ms1_map()

    logger.info(
        f"Re-banded MS2 at width {width:.5f} (stride {width / 2.0:.5f}): "
        f"{len(new_bins)} bins over [{im_lo:.4f}, {im_hi:.4f}], "
        f"{n_before} -> {len(sf.ms2scans)} MS2 spectra"
    )
    return True


_BRUKER_REMEDIATION = (
    "Point --bruker_sdk_path at the unzipped Bruker timsdata SDK folder (the one "
    "containing win64/ and linux64/), at the platform subfolder itself, or directly "
    "at the library file. You can also set \"bruker_sdk_path\" in JMod/data/settings.json."
)


def _bruker_lib_name(platform):
    return _BRUKER_LIB_NAME.get(platform, _BRUKER_LIB_DEFAULT)


def _fail_bruker(message):
    """Raise a JModError for a bad --bruker_sdk_path, with how to fix it."""
    full = f"{message}\n\n{_BRUKER_REMEDIATION}"
    raise JModError(full)


def _safe_iterdir(path):
    """Children of ``path``, or nothing if it cannot be read.

    An unreadable sibling directory should not abort the search.
    """
    try:
        return list(path.iterdir())
    except (OSError, PermissionError):
        return []


def _search_bruker_lib(root, lib_name, max_depth):
    """Breadth-first search for ``lib_name`` under ``root``, bounded to max_depth.

    Breadth-first so a shallower hit always beats a deeper one, and sorted at every
    level so the result never depends on filesystem iteration order. Depth is
    capped, which also bounds any symlink cycle without tracking visited paths.
    """
    level = [root]
    for _ in range(max_depth):
        children = sorted(
            (c for parent in level for c in _safe_iterdir(parent) if c.is_dir()),
            key=str,
        )
        for child in children:
            candidate = child / lib_name
            if candidate.is_file():
                return candidate
        level = children
    return None


def resolve_bruker_sdk_path(sdk_path, *, platform=None, strict=True):
    """Resolve ``--bruker_sdk_path`` to a concrete timsdata library file.

    Returns None if and only if nothing was specified (None or blank); .d input
    then fails, since it is always read raw with the SDK's calibration. Otherwise
    returns an absolute path to an existing library file -- never a directory.
    Raises JModError when a path was given but no library could be found: a
    specified-but-unresolvable SDK is an error, not a reason to silently
    approximate. Pass ``strict=False`` to get None instead -- for callers merely
    testing a candidate path, such as the GUI's folder picker.

    The library name and platform directory follow ``sys.platform``, so this
    picks timsdata.dll/win64 on Windows and libtimsdata.so/linux64 elsewhere.
    ``platform`` is injectable purely so tests can cover all three.

    Resolving all the way to a file means peppy_sage takes its ``p.is_file()``
    fast path, so its own non-recursive directory search never runs and this is
    the single source of truth for where the library lives.
    """
    import sys
    from pathlib import Path

    platform = platform or sys.platform

    def _fail(message):
        if strict:
            _fail_bruker(message)
        return None

    if sdk_path is None:
        return None
    # Strip quotes left over from shell copy-paste of a path with spaces.
    raw = str(sdk_path).strip().strip("\"'").strip()
    if not raw:
        return None

    lib_name = _bruker_lib_name(platform)
    p = Path(raw).expanduser()

    # An explicit file is taken at the user's word, whatever it is named -- this
    # covers versioned or renamed copies such as libtimsdata.so.2.21.
    if p.is_file():
        return str(p.resolve())

    if not p.is_dir():
        return _fail(
            f"Bruker timsdata SDK path does not exist: {raw}\n"
            f"(this may have come from --bruker_sdk_path, or from "
            f"\"bruker_sdk_path\" in JMod/data/settings.json)"
        )

    direct = p / lib_name
    if direct.is_file():
        return str(direct.resolve())

    # The real SDK layout, checked by exact name before any general search: the
    # SDK also ships a thirdparty/ folder, so a name-agnostic walk is not the
    # thing to rely on first.
    platform_dir = _BRUKER_PLATFORM_DIR.get(platform, _BRUKER_PLATFORM_DIR_DEFAULT)
    candidate = p / platform_dir / lib_name
    if candidate.is_file():
        return str(candidate.resolve())

    found = _search_bruker_lib(p, lib_name, _BRUKER_SDK_MAX_DEPTH)
    if found is not None:
        return str(found.resolve())

    return _fail(
        f"No Bruker timsdata library ({lib_name}) found in {raw} or in its "
        f"subdirectories (searched {_BRUKER_SDK_MAX_DEPTH} levels deep).\n"
        f"The Bruker SDK zip unpacks to a folder containing win64/timsdata.dll "
        f"and linux64/libtimsdata.so -- point at that folder, or at {lib_name} "
        f"directly."
    )


def resolve_bruker_setting(cli_value):
    """Resolve the Bruker SDK path, persisting a CLI value once it validates.

    A CLI arg wins and is stored; otherwise the stored setting is used. Both are
    validated. Returns the resolved library file, or None when nothing was set.
    """
    from src.utils.gui_utils import load_settings, save_settings

    settings = load_settings()

    from_cli = cli_value is not None
    raw_sdk_path = cli_value if from_cli else settings.get("bruker_sdk_path")

    resolved = resolve_bruker_sdk_path(raw_sdk_path)

    if from_cli:
        # Resolves first, so a bad path cannot poison settings.json. Storing None
        # for an empty --bruker_sdk_path gives it the meaning "forget my SDK".
        settings["bruker_sdk_path"] = resolved
        save_settings(settings)

    return resolved
