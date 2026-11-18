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

import os
import pickle
import sqlite3
from time import perf_counter

import numpy as np
import numba as nb
import polars as pl

from src.utils.errors import JModError
from src.utils.io.load_files import (
    Spectrum, SpectrumFile, PEAK_MZ_DTYPE, PEAK_INT_DTYPE, PEAK_MOB_DTYPE,
)
from src.utils.io.tdf_bin import TdfBinReader, frame_peak_counts
from src.utils.io.band_store import BandSpectrum, band_keys
from src.logger import logger

# Centroiding noise thresholds. TODO: expose via settings/CLI alongside
# --bruker_sdk_path; constant for now.
MIN_ION_COUNT_MS1 = 0.5
MIN_ION_COUNT_MS2 = 2.0

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

    ``width = 2 * stride`` (50% overlap) is required by ``band_store.band_keys``,
    which only tests two candidate bins per peak.
    """
    starts = np.arange(im_lo, im_hi - width + stride * 0.5, stride)
    return np.column_stack([starts, starts + width])


@nb.njit
def collapse_tof(mz_vals, intensities, mobilities, ppm_tol=10.0):
    """Merge peaks within ppm tolerance by intensity-weighted m/z averaging.

    Processes peaks in descending intensity order. For each unmerged peak,
    gathers all unmerged neighbors within ppm_tol (computed per-pair) and
    merges them into a single peak with summed intensity, weighted m/z,
    and the ion mobility of the seed (most intense) peak.

    mz_vals: 1D array of calibrated m/z values (unsorted, from multiple IM scans)
    intensities: 1D array of corresponding intensities
    mobilities: 1D array of ion mobility values (1/K0)
    ppm_tol: merging tolerance in ppm
    Returns: (centroided_mz, summed_intensity, seed_mobility) sorted by m/z
    """
    n = len(mz_vals)
    if n == 0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

    # Sort by m/z for neighbor searching
    mz_order = np.argsort(mz_vals)
    mz_sorted = mz_vals[mz_order]
    int_sorted = intensities[mz_order]
    mob_sorted = mobilities[mz_order]

    # Process order: most intense first
    int_order = np.argsort(-int_sorted)

    merged = np.zeros(n, dtype=nb.boolean)
    out_mz = np.empty(n, dtype=np.float64)
    out_int = np.empty(n, dtype=np.float64)
    out_mob = np.empty(n, dtype=np.float64)
    n_out = 0

    for k in range(n):
        idx = int_order[k]
        if merged[idx]:
            continue

        anchor_mz = mz_sorted[idx]
        wsum_mz = anchor_mz * int_sorted[idx]
        wsum_int = int_sorted[idx]
        seed_mob = mob_sorted[idx]
        merged[idx] = True

        # Search left
        j = idx - 1
        while j >= 0 and not merged[j]:
            tol_da = mz_sorted[j + 1] * ppm_tol * 1e-6
            if (mz_sorted[j + 1] - mz_sorted[j]) > tol_da:
                break
            wsum_mz += mz_sorted[j] * int_sorted[j]
            wsum_int += int_sorted[j]
            merged[j] = True
            j -= 1

        # Search right
        j = idx + 1
        while j < n and not merged[j]:
            tol_da = mz_sorted[j - 1] * ppm_tol * 1e-6
            if (mz_sorted[j] - mz_sorted[j - 1]) > tol_da:
                break
            wsum_mz += mz_sorted[j] * int_sorted[j]
            wsum_int += int_sorted[j]
            merged[j] = True
            j += 1

        out_mz[n_out] = wsum_mz / wsum_int
        out_int[n_out] = wsum_int
        out_mob[n_out] = seed_mob
        n_out += 1

    # Sort output by m/z
    result_mz = out_mz[:n_out]
    result_int = out_int[:n_out]
    result_mob = out_mob[:n_out]
    final_order = np.argsort(result_mz)
    return result_mz[final_order], result_int[final_order], result_mob[final_order]


class FileReader:
    """Dispatch to format-specific parsers based on file extension."""

    _FORMATS = {".mzml", ".d", ".raw"}

    def __init__(self, filepath: str, bruker_sdk_path: str = None):
        self.filepath = filepath.replace("\\", "/")
        self.bruker_sdk_path = bruker_sdk_path
        self.format = self._detect_format()

    def _detect_format(self) -> str:
        if self.filepath.rstrip("/").endswith(".d"):
            return ".d"
        _, ext = os.path.splitext(self.filepath)
        ext = ext.lower()
        if ext not in self._FORMATS:
            raise ValueError(
                f"Unsupported file format '{ext}'. Supported: {sorted(self._FORMATS)}"
            )
        return ext

    def read(self) -> SpectrumFile:
        if self.format == ".mzml":
            return _load_mzml(self.filepath)
        elif self.format == ".d":
            return _load_d(self.filepath, self.bruker_sdk_path)
        elif self.format == ".raw":
            return _load_raw(self.filepath)


def _load_mzml(filepath: str) -> SpectrumFile:
    logger.info("Loading Spectra... from .mzML file")
    return SpectrumFile(mzml_file=filepath)


def _load_raw(filepath: str) -> SpectrumFile:
    logger.info("Loading Spectra... from .raw file")
    return SpectrumFile(raw_file=filepath)


def _centroid_d(d_path: str, peaks_path: str, sdk_path: str = None) -> None:
    """Centroid a .d in place, writing peaks.parquet, via peppy_sage.

    Replaces the manual step of running the timscentroid binary beforehand;
    the result is the same file, with the same columns, read the same way.

    ``sdk_path`` points at Bruker's timsdata library, or the directory holding
    it, and comes from ``--bruker_sdk_path`` or the stored setting.  Without it
    m/z and 1/K0 are approximated from analysis.tdf and will not match an
    SDK-centroided run.
    """
    try:
        from peppy_sage import _peppy_sage
    except ImportError as e:
        raise FileNotFoundError(
            f"Peaks file not found: {peaks_path}, and peppy_sage is not "
            f"importable to create it ({e}). Either place a peaks.parquet "
            f"inside {d_path} or install peppy_sage."
        ) from e

    if not hasattr(_peppy_sage, "centroid_d"):
        raise FileNotFoundError(
            f"Peaks file not found: {peaks_path}, and the installed peppy_sage "
            f"has no centroid_d. Rebuild peppy_sage, or place a peaks.parquet "
            f"inside {d_path}."
        )

    sdk_path = sdk_path or None
    # run_jmod resolves this to a concrete library file before we ever get here.
    # Re-check, because the GUI writes bruker_sdk_path into settings directly and
    # a stale value would otherwise reach peppy_sage and hit its directory
    # fallback instead of failing with something the user can act on.
    if sdk_path and not os.path.isfile(sdk_path):
        _fail_bruker(f"Bruker timsdata library not found: {sdk_path}")

    # TODO: plumb these through the settings/CLI like bruker_sdk_path once we
    # know which values we actually want. Lowering them keeps signal the
    # defaults discard, which is worth revisiting.
    min_ms1 = MIN_ION_COUNT_MS1
    min_ms2 = MIN_ION_COUNT_MS2

    if sdk_path:
        logger.info(f"Centroiding with Bruker SDK at {sdk_path}")
    else:
        logger.warning(
            "Centroiding WITHOUT the Bruker SDK: m/z and 1/K0 are approximated "
            "from analysis.tdf and will not match an SDK-centroided run. Pass "
            "--bruker_sdk_path to use the real calibration."
        )
    logger.info(f"  min_ion_count_ms1={min_ms1}, min_ion_count_ms2={min_ms2}")

    # Write beside the target and rename, so an interrupted run cannot leave a
    # partial peaks.parquet that the next run would treat as complete.
    tmp_path = peaks_path + ".partial"
    t0 = perf_counter()

    # centroid_d reports (done, total) every 64 frames from its worker threads.
    # tqdm.total is not known until the first callback, so the bar is created
    # lazily and only closed if it was ever opened.
    from tqdm.auto import tqdm
    bar = {"obj": None}

    def _on_progress(done, total):
        if bar["obj"] is None:
            bar["obj"] = tqdm(total=total, desc="Centroiding", unit="frame")
        bar["obj"].update(done - bar["obj"].n)

    try:
        n = _peppy_sage.centroid_d(d_path, tmp_path, min_ms1, min_ms2, sdk_path,
                                   _on_progress)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
    finally:
        if bar["obj"] is not None:
            bar["obj"].close()
    os.replace(tmp_path, peaks_path)
    logger.info(f"Centroided {n:,} peaks -> {peaks_path} "
                f"({perf_counter() - t0:.0f}s)")


def _recalibrate_peaks_parquet(peaks_path: str, cal: dict) -> None:
    """Set a peaks.parquet's m/z and 1/K0 from the SDK tables in *cal* (by each
    peak's apex TOF index and scan), in place, if they differ."""
    import pyarrow as pa, pyarrow.parquet as pq
    src = pq.ParquetFile(peaks_path); schema = src.schema_arrow
    head = src.read_row_group(0, columns=["tof", "scan", "mz", "im"])
    mz_t, im_t = np.asarray(cal["mz_values"]), np.asarray(cal["mobility_values"])
    if (np.allclose(head.column("mz").to_numpy(), mz_t[head.column("tof").to_numpy()], rtol=0, atol=1e-9)
            and np.allclose(head.column("im").to_numpy(), im_t[head.column("scan").to_numpy()], rtol=0, atol=1e-9)):
        return
    tmp = peaks_path + ".sdk.partial"
    with pq.ParquetWriter(tmp, schema) as w:
        for k in range(src.metadata.num_row_groups):
            t = src.read_row_group(k); cols = {n: t.column(n) for n in schema.names}
            cols["mz"] = pa.array(mz_t[np.clip(t.column("tof").to_numpy(), 0, len(mz_t) - 1)])
            cols["im"] = pa.array(im_t[np.clip(t.column("scan").to_numpy(), 0, len(im_t) - 1)])
            w.write_table(pa.Table.from_arrays([cols[n] for n in schema.names], schema=schema))
    os.replace(tmp, peaks_path)
    logger.info(f"Set {peaks_path} m/z and 1/K0 from the Bruker SDK tables")


def use_centroided_ms1(sf: SpectrumFile) -> SpectrumFile:
    """Replace the MS1 peaks of a raw-read .d with its centroids, in place.

    Not called at present (raw ions quantified better on the 200pg 5plex).
    Identification runs on the raw ions, and once
    the IDs are scored the MS1 spectra are swapped for centroided peaks for the
    quantification.  Scan numbers, retention times, the MS2 spectra and the
    MS2 -> MS1 map are untouched, so the IDs' Ms1_spec_id still point at the
    same frames.

    The centroids are the .d's peaks.parquet, made by the centroider when
    missing.  It works in TOF-index and scan space and only labels each peak
    with m/z and 1/K0 at its apex, so those are taken here from the Bruker SDK
    tables the raw read already uses (bruker_calibration/): the same values an
    SDK-centroided peaks.parquet holds, on every platform.
    """
    d_path = getattr(sf, "_raw_d_path", None)
    if d_path is None:
        raise JModError("centroided MS1 needs .d input read raw")
    peaks_path = os.path.join(d_path, "peaks.parquet")
    if not os.path.exists(peaks_path):
        _centroid_d(d_path, peaks_path, None)
    tables = os.path.join(d_path, _SDK_TABLES_DIR)
    mz_table = np.load(os.path.join(tables, "mz.npy"))
    ook0_table = np.load(os.path.join(tables, "ook0.npy"))

    cal = sf._cal
    is_ms1 = np.asarray(cal["ms_level"]) == 1
    frame_rt = dict(zip(np.asarray(cal["frame_ids"])[is_ms1].tolist(),
                        np.asarray(cal["rt_values"])[is_ms1].tolist()))
    by_rt = {spec.RT: spec for spec in sf.ms1scans}

    import pyarrow.dataset as pads
    t = pads.dataset(peaks_path).to_table(
        filter=pads.field("frame").isin(list(frame_rt)),
        columns=["frame", "scan", "tof", "apex_intensity"])
    frame = t.column("frame").to_numpy()
    order = np.argsort(frame, kind="stable")
    frame = frame[order]
    scan = t.column("scan").to_numpy()[order]
    tof = t.column("tof").to_numpy()[order]
    intens = t.column("apex_intensity").to_numpy()[order]
    starts = np.flatnonzero(np.r_[True, frame[1:] != frame[:-1]]) if frame.size else np.array([], int)
    ends = np.r_[starts[1:], frame.size]

    empty = (np.empty(0, PEAK_MZ_DTYPE), np.empty(0, PEAK_INT_DTYPE), np.empty(0, PEAK_MOB_DTYPE))
    replaced = set()
    for a, b in zip(starts, ends):
        spec = by_rt.get(frame_rt[int(frame[a])])
        if spec is None:
            continue
        mz = mz_table[np.clip(tof[a:b], 0, len(mz_table) - 1)].astype(PEAK_MZ_DTYPE)
        o = np.argsort(mz, kind="stable")
        spec.mz = mz[o]
        spec.intens = intens[a:b][o].astype(PEAK_INT_DTYPE)
        spec.mobility = ook0_table[np.clip(scan[a:b], 0, len(ook0_table) - 1)][o].astype(PEAK_MOB_DTYPE)
        spec.TIC = float(spec.intens.sum(dtype=np.float64))
        replaced.add(id(spec))
    for spec in sf.ms1scans:          # frames without a centroid
        if id(spec) not in replaced:
            spec.mz, spec.intens, spec.mobility = empty
            spec.TIC = 0.0
    logger.info(f"MS1 quantified from centroids: {len(replaced)} of {len(sf.ms1scans)} "
                f"MS1 spectra, {frame.size} peaks ({peaks_path})")
    return sf


def use_centroided_ms2(sf: SpectrumFile) -> SpectrumFile:
    """Replace the MS2 peaks of a raw-read .d with its centroids, in place.

    Not called at present.  After identification and MS1 quant: each MS2
    spectrum (one isolation window of one frame, cut to one IM band) gets the
    frame's centroids whose apex scan lies in that window's scan range and
    whose 1/K0 lies in the band, by the same rules the raw ions were assigned
    with (scan_begin <= scan < scan_end; im_lo <= 1/K0 < im_hi).  Scan numbers
    and everything else are untouched.  m/z and 1/K0 come from the Bruker SDK
    tables, as in use_centroided_ms1.
    """
    d_path = getattr(sf, "_raw_d_path", None)
    if d_path is None:
        raise JModError("centroided MS2 needs .d input read raw")
    peaks_path = os.path.join(d_path, "peaks.parquet")
    if not os.path.exists(peaks_path):
        _centroid_d(d_path, peaks_path, None)
    tables = os.path.join(d_path, _SDK_TABLES_DIR)
    mz_table = np.load(os.path.join(tables, "mz.npy"))
    ook0_table = np.load(os.path.join(tables, "ook0.npy"))

    cal = sf._cal
    frame_ids = np.asarray(cal["frame_ids"])
    is_ms2 = np.asarray(cal["ms_level"]) == 2
    rt_frame = dict(zip(np.asarray(cal["rt_values"])[is_ms2].tolist(), frame_ids[is_ms2].tolist()))
    windows = {}                                   # (frame, prec_mz) -> (scan_begin, scan_end)
    for frame, scan_begin, scan_end, prec_mz, _w, _ce in sf._dia_lookup.iter_rows():
        windows[(int(frame), float(prec_mz))] = (int(scan_begin), int(scan_end))

    import pyarrow.dataset as pads
    t = pads.dataset(peaks_path).to_table(
        filter=pads.field("frame").isin(list(rt_frame.values())),
        columns=["frame", "scan", "tof", "apex_intensity"])
    frame = t.column("frame").to_numpy()
    order = np.argsort(frame, kind="stable")
    frame = frame[order]
    scan = t.column("scan").to_numpy()[order].astype(np.int64)
    tof = t.column("tof").to_numpy()[order]
    intens = t.column("apex_intensity").to_numpy()[order]
    del t
    mob = ook0_table[np.clip(scan, 0, len(ook0_table) - 1)].astype(PEAK_MOB_DTYPE)
    mz_all = mz_table[np.clip(tof, 0, len(mz_table) - 1)].astype(PEAK_MZ_DTYPE)
    starts = np.flatnonzero(np.r_[True, frame[1:] != frame[:-1]]) if frame.size else np.array([], int)
    span = {int(frame[a]): (a, b) for a, b in zip(starts, np.r_[starts[1:], frame.size])}

    n_peaks = 0
    for spec in sf.ms2scans:
        f = rt_frame.get(spec.RT)
        w = windows.get((f, float(spec.prec_mz))) if f is not None else None
        a, b = span.get(f, (0, 0)) if w is not None else (0, 0)
        m64 = mob[a:b].astype(np.float64)
        keep = ((scan[a:b] >= w[0]) & (scan[a:b] < w[1]) & (spec.im_lo <= m64) & (m64 < spec.im_hi)
                if b > a else np.zeros(0, bool))
        mz = mz_all[a:b][keep]
        o = np.argsort(mz, kind="stable")
        spec.mz = mz[o]
        spec.intens = intens[a:b][keep][o].astype(PEAK_INT_DTYPE)
        spec.mobility = mob[a:b][keep][o]
        spec.TIC = float(spec.intens.sum(dtype=np.float64))
        n_peaks += spec.mz.size
    logger.info(f"MS2 quantified from centroids: {len(sf.ms2scans)} MS2 spectra, "
                f"{n_peaks} peaks ({peaks_path})")
    return sf


def _load_d(filepath: str, bruker_sdk_path: str = None) -> SpectrumFile:
    """Load timsTOF .d data.

    With a peaks.parquet in the .d, peaks come from it (already centroided and
    calibrated).  Without one, the raw ions in analysis.tdf_bin are read
    uncentroided and calibrated with tables computed by Bruker's timsdata
    (see ``_sdk_calibration_tables``).
    """
    logger.info("Loading Spectra... from .d file")
    d_path = filepath.rstrip("/")

    peaks_path = os.path.join(d_path, "peaks.parquet")
    tdf_path = os.path.join(d_path, "analysis.tdf")
    bin_path = os.path.join(d_path, "analysis.tdf_bin")

    if not os.path.exists(tdf_path):
        raise FileNotFoundError(f"analysis.tdf not found in {d_path}")

    # Always read the raw, uncentroided ions with Bruker's calibration; a
    # peaks.parquet in the .d is ignored and the centroider is not used.
    # raw = not os.path.exists(peaks_path)
    # TODO: the centroided path below (the centroider's peaks.parquet, m/z and
    # 1/K0 set from the SDK tables by _recalibrate_peaks_parquet) is kept but
    # not reachable; on the 200pg 5plex raw ions quantified better.
    raw = True
    if not raw and not os.path.exists(peaks_path):
        _centroid_d(d_path, peaks_path, None)
    if raw and not os.path.exists(bin_path):
        raise FileNotFoundError(
            f"Neither {peaks_path} nor {bin_path} exists; there are no peaks "
            f"to read."
        )

    logger.info(f"Loading calibration from {d_path}")
    cal = _load_calibration(tdf_path)
    if raw:
        logger.info(f"No peaks.parquet in {d_path}; reading the raw, "
                    f"uncentroided ions with Bruker's calibration")
        _load_sdk_tables(_sdk_calibration_tables(d_path, bruker_sdk_path), cal)
    dia_lookup = _load_dia_windows(tdf_path, cal["frame_ids"], cal["ms_level"])

    im_bins = compute_im_bins(cal["im_range_lower"], cal["im_range_upper"])
    logger.info(f"Computed {len(im_bins)} overlapping IM bins "
                f"[{cal['im_range_lower']:.4f}, {cal['im_range_upper']:.4f}]")

    if raw:
        sf = _build_spectrum_file(filepath, None, cal, dia_lookup, im_bins,
                                  raw_d_path=d_path)
    else:
        _load_sdk_tables(_sdk_calibration_tables(d_path, bruker_sdk_path), cal)
        _recalibrate_peaks_parquet(peaks_path, cal)
        logger.info(f"Reading peaks from {peaks_path}")
        sf = _build_spectrum_file(filepath, peaks_path, cal, dia_lookup, im_bins)
        sf._raw_d_path = d_path

    logger.info(f"Loaded {len(sf.ms1scans)} MS1 spectra, {len(sf.ms2scans)} MS2 spectra")
    return sf


# Bruker's m/z and 1/K0 tables for a .d, cached inside it beside where
# peaks.parquet would go, as written by scripts/bruker_calibration_tables.py.
_SDK_TABLES_DIR = "bruker_calibration"
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_SDK_TABLES_SCRIPT = os.path.join(_REPO_ROOT, "scripts", "bruker_calibration_tables.py")

# macOS: Bruker ships no macOS library, so the Windows timsdata.dll from
# --bruker_sdk_path runs under Wine with a Windows Python (never bundled; Wine is
# LGPL).  Hardcoded to this machine's setup; see MONO_D_CALIBRATION_PLAN.md for
# how these were installed.  The helper script needs only the standard library.
_WINE = os.path.expanduser(
    "~/Applications/Wine Devel.app/Contents/Resources/wine/bin/wine")
_WINE_PREFIX = os.path.expanduser("~/.wine-jmod")
_WINE_PYTHON = os.path.expanduser("~/.wine-jmod-py/python.exe")


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
    import subprocess
    import sys

    tables_dir = os.path.join(d_path, _SDK_TABLES_DIR)
    if os.path.isfile(os.path.join(tables_dir, "calibration.json")):
        return tables_dir

    # Write beside the target and rename, so an interrupted run cannot leave
    # partial tables that the next run would treat as complete.
    tmp_dir = tables_dir + ".partial"
    if not bruker_sdk_path:
        _fail_bruker(f"No peaks.parquet in {d_path}, and reading it raw needs "
                     f"Bruker's calibration")
    if sys.platform == "darwin":
        for p in (_WINE, _WINE_PYTHON):
            if not os.path.exists(p):
                raise FileNotFoundError(
                    f"Cannot compute Bruker calibration for {d_path}: {p} is missing")
        cmd = [_WINE, _WINE_PYTHON, _wine_path(_SDK_TABLES_SCRIPT),
               _wine_path(bruker_sdk_path), _wine_path(d_path), _wine_path(tmp_dir)]
        env = dict(os.environ, WINEPREFIX=_WINE_PREFIX, WINEDEBUG="-all")
    else:
        cmd = [sys.executable, _SDK_TABLES_SCRIPT, bruker_sdk_path, d_path, tmp_dir]
        env = None

    logger.info(f"Computing Bruker calibration tables -> {tables_dir}")
    t0 = perf_counter()
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if result.returncode != 0 or not os.path.isfile(
            os.path.join(tmp_dir, "calibration.json")):
        raise RuntimeError(
            f"Computing Bruker calibration for {d_path} failed "
            f"(exit {result.returncode}):\n{result.stdout}\n{result.stderr}")
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
    follow it): ``_drain_sorted`` of a dict, or ``_iter_raw_ms2_groups``, which
    streams them so only one RT's un-banded peaks exist at a time.  Appends a
    Spectrum per band to ``sf.ms2scans`` and updates ``sf.ms2_by_id`` /
    ``sf.scan_pos``; returns the next free scan number and the number of windows.

    A peak lies in up to 2 overlapping fixed IM bins but is stored once: the
    bands are ``BandSpectrum`` views of their window's buffer
    (src/utils/io/band_store.py).
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
        # (src/utils/io/band_store.py).
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


def _read_peak_groups(peaks_path: str, cal: dict, dia_lookup: pl.DataFrame,
                      include_ms1: bool = True):
    """Read peaks.parquet and group peaks by acquisition, without IM banding.

    Returns ``(ms1_groups, ms2_groups)``:

    - ``ms1_groups``: ``rt -> (mz_list, intens_list, mob_list)``, one frame per RT,
      left as per-row-group lists for the caller to concatenate.  Empty when
      ``include_ms1`` is False.
    - ``ms2_groups``: ``(rt, prec_mz, iso_width, ce) -> (mz, intens, mob)``, already
      concatenated -- the form ``_band_ms2_groups`` consumes.

    Processes the parquet in row-group batches to avoid loading 6+ GB at once.
    ``include_ms1=False`` skips MS1 entirely, for callers that only need to redraw
    the MS2 bands (see ``reband_ms2``).
    """
    import pyarrow.parquet as pq

    rt_by_frame, ms_level_by_frame = _frame_tables(cal)
    dia_dict = _windows_by_frame(dia_lookup)

    # Accumulators: group peaks by rt for MS1 and (rt, prec_mz, iso_width, ce) for MS2
    ms1_groups = {}  # rt -> (mz_list, intens_list, mob_list); one frame per rt
    # (rt, prec_mz, iso_width, ce) -> (mz_list, intens_list, mob_list); raw peaks
    # accumulated across row groups, then IM-band segmented in a post-pass.
    ms2_groups = {}

    # Batched read with column pruning and prefetch. peaks.parquet files from
    # older centroid_d builds carry tens of thousands of tiny row groups (one
    # per frame); reading them one group at a time costs one network round
    # trip each (~10 min over SMB for a 7 GB file). iter_batches with
    # pre_buffer coalesces the small column chunks into large sequential
    # reads, and the 4 unused columns (tof/rt/isolation_window_*: RT is
    # rebuilt from analysis.tdf) are ~44% of the bytes.
    columns = ["frame", "scan", "mz", "im", "apex_intensity"]
    pf = pq.ParquetFile(peaks_path, pre_buffer=True)
    n_rows = pf.metadata.num_rows
    batch_size = 2_000_000  # ~70 MB working set over the 5 columns
    logger.info(f"Processing {n_rows:,} peaks from {peaks_path}")

    import pyarrow as pa
    for batch in pf.iter_batches(batch_size=batch_size, columns=columns,
                                 use_threads=True):
        # zero-copy wrap: Table.column(...).to_numpy() handles nullable
        # columns, RecordBatch.column(...).to_numpy() does not
        table = pa.Table.from_batches([batch])
        frames = table.column("frame").to_numpy()
        scans = table.column("scan").to_numpy()
        mz_arr = table.column("mz").to_numpy().astype(PEAK_MZ_DTYPE, copy=False)
        # Intensity and mobility are stored as float32 on Spectrum (see PEAK_INT_DTYPE
        # / PEAK_MOB_DTYPE): 1/K0 spans ~0.7-1.3, where float32 resolution (~1e-7) is
        # orders of magnitude below any IM tolerance, and intensities only need
        # relative precision. Cast at ingest so the per-row-group accumulators in
        # ms1_groups / ms2_groups are narrow too, not just the final spectra.
        mob_arr = table.column("im").to_numpy().astype(PEAK_MOB_DTYPE)
        intensities = table.column("apex_intensity").to_numpy().astype(PEAK_INT_DTYPE)

        # RT and MS level still come from analysis.tdf (indexed by frame)
        rt_arr = rt_by_frame[frames]
        level_arr = ms_level_by_frame[frames]

        # MS1 peaks — accumulate raw peaks per frame (one spectrum per RT),
        # keeping per-peak mobility (1/K0). No IM binning.
        # One stable argsort + boundary split per batch instead of an
        # equality mask per unique RT: a batch holds ~200 frames, so repeated
        # full-batch masks are O(rows x frames) and dominated the local-disk
        # runtime. Stable sort keeps within-frame row order, so output is
        # unchanged.
        ms1_mask = level_arr == 1 if include_ms1 else None
        if include_ms1 and ms1_mask.any():
            ms1_mz = mz_arr[ms1_mask]
            ms1_intens = intensities[ms1_mask]
            ms1_mob = mob_arr[ms1_mask]
            ms1_rt = rt_arr[ms1_mask]

            order = np.argsort(ms1_rt, kind="stable")
            sorted_rt = ms1_rt[order]
            bounds = np.flatnonzero(np.diff(sorted_rt)) + 1
            for sel in np.split(order, bounds):
                key = float(ms1_rt[sel[0]])
                if key not in ms1_groups:
                    ms1_groups[key] = ([], [], [])
                ms1_groups[key][0].append(ms1_mz[sel])
                ms1_groups[key][1].append(ms1_intens[sel])
                ms1_groups[key][2].append(ms1_mob[sel])

        # MS2 peaks: assign to DIA windows (same sort-and-split pattern)
        ms2_mask = level_arr == 2
        if ms2_mask.any():
            ms2_frames = frames[ms2_mask]
            ms2_scans = scans[ms2_mask]
            ms2_mz = mz_arr[ms2_mask]
            ms2_intens = intensities[ms2_mask]
            ms2_mob = mob_arr[ms2_mask]
            ms2_rt = rt_arr[ms2_mask]

            order = np.argsort(ms2_frames, kind="stable")
            sorted_frames = ms2_frames[order]
            bounds = np.flatnonzero(np.diff(sorted_frames)) + 1
            for sel in np.split(order, bounds):
                frame_id = ms2_frames[sel[0]]
                windows = dia_dict.get(int(frame_id))
                if windows is None:
                    continue
                # Accumulated per (frame, window) across row groups; IM-band
                # segmentation runs in a post-pass (it needs the whole window).
                _add_window_peaks(ms2_groups, float(ms2_rt[sel[0]]), windows, ms2_scans[sel],
                                  ms2_mz[sel], ms2_intens[sel], ms2_mob[sel])

    # Concatenate MS2 in place, dropping the per-row-group fragments as we go so
    # the list-of-chunks form never coexists in full with the concatenated form.
    for key in list(ms2_groups):
        v = ms2_groups.pop(key)
        ms2_groups[key] = (np.concatenate(v[0]), np.concatenate(v[1]),
                           np.concatenate(v[2]))

    return ms1_groups, ms2_groups


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
    and so do *not* match a peaks.parquet produced with the Bruker SDK.
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
    """The raw MS1 frames of ``analysis.tdf_bin``, in the shape ``_read_peak_groups``
    returns MS1: ``rt -> ([mz], [intens], [mob])``, one frame per RT.

    Reads every stored ion instead of the centroider's peak list; downstream code
    cannot tell the two apart.  MS2 is streamed separately
    (``_iter_raw_ms2_groups``).
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


def _reread_ms2_groups(sf: SpectrumFile):
    """The un-banded MS2 groups again, in sorted key order, from whichever source
    built ``sf``.

    ``reband_ms2`` deliberately does not keep the un-banded peaks in memory, so it
    goes back to the data: raw .d input is streamed again, one RT at a time; a
    peaks.parquet is re-read whole and drained window by window.
    """
    if sf._peaks_path is not None:
        _, groups = _read_peak_groups(sf._peaks_path, sf._cal, sf._dia_lookup, include_ms1=False)
        return _drain_sorted(groups)
    return _iter_raw_ms2_groups(sf._raw_d_path, sf._cal, sf._dia_lookup, sf._min_intensity)


def _build_spectrum_file(filepath: str, peaks_path: str, cal: dict, dia_lookup: pl.DataFrame,
                         im_bins: np.ndarray, raw_d_path: str = None,
                         min_intensity: int = 0) -> SpectrumFile:
    """Assign peaks to DIA windows and IM bins, and build a SpectrumFile.

    Peaks come from one of two sources, and everything after the read is
    identical for both:

    - ``peaks_path`` -- a centroider peaks.parquet, already calibrated, read in
      row-group batches to avoid loading 6+ GB at once.  Expects columns
      ``frame, scan, mz, im, apex_intensity``.
    - ``raw_d_path`` -- a stock .d, decoded frame by frame from analysis.tdf_bin
      and calibrated here (see ``_read_raw_frame``).

    Exactly one of the two must be given.  Each non-empty IM bin produces a
    separate Spectrum object.
    """
    if (peaks_path is None) == (raw_d_path is None):
        raise ValueError("Pass exactly one of peaks_path or raw_d_path")

    if raw_d_path is not None:
        # MS1 now; MS2 is streamed into banding below, one RT at a time, so the
        # whole run's un-banded MS2 peaks are never held at once.
        ms1_groups = _read_raw_ms1_groups(raw_d_path, cal, min_intensity)
        ms2_groups = _iter_raw_ms2_groups(raw_d_path, cal, dia_lookup, min_intensity)
    else:
        ms1_groups, ms2_dict = _read_peak_groups(peaks_path, cal, dia_lookup)
        ms2_groups = _drain_sorted(ms2_dict)

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
    # in memory for that (~9 GB on a large .d, untouched across the whole initial
    # search), record what it takes to re-read them from the parquet, which is
    # still on disk.  Re-reading costs ~25 s; holding them costs 9 GB for minutes.
    sf._peaks_path = peaks_path
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
    ``width / 2``.  That relationship is required by ``band_store.band_keys``,
    which only ever tests two candidate bins per peak.

    The un-banded peaks are not kept in memory for this -- they are re-read from
    whichever source built ``sf`` (peaks parquet or raw .d), which is still on
    disk.  A parquet re-read costs ~25 s; retaining the peaks would cost ~9 GB
    on a large .d, held untouched across the whole initial search.  Re-reading
    raw costs whatever the initial raw read cost.

    Returns True if the bands were rebuilt, False if there was nothing to do
    (non-IM data, i.e. anything but the .d path).
    """
    im_range = getattr(sf, "_im_range", None)
    has_source = (getattr(sf, "_peaks_path", None) is not None
                  or getattr(sf, "_raw_d_path", None) is not None)
    if not has_source or im_range is None:
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
    _band_ms2_groups(sf, _reread_ms2_groups(sf), new_bins, scan_counter)
    sf.build_ms2_to_ms1_map()

    logger.info(
        f"Re-banded MS2 at width {width:.5f} (stride {width / 2.0:.5f}): "
        f"{len(new_bins)} bins over [{im_lo:.4f}, {im_hi:.4f}], "
        f"{n_before} -> {len(sf.ms2scans)} MS2 spectra"
    )
    return True


def loadSpectra(input_file: str, bruker_sdk_path: str = None) -> SpectrumFile:
    """Drop-in replacement for load_files.loadSpectra with format dispatch."""
    logger.info("Loading Spectra...")
    # python_spec_file = input_file + "_pythonspec"
    # if not os.path.exists(python_spec_file):
    reader = FileReader(input_file, bruker_sdk_path)
    spectra = reader.read()
    #     with open(python_spec_file, "wb") as f:
    #         pickle.dump(spectra, f)
    # else:
    #     with open(python_spec_file, "rb") as f:
    #         logger.info("Loading Spectra... from pickle")
    #         spectra = pickle.load(f)

    logger.info(f"Loaded {len(spectra.ms1scans)} MS1 spectra")
    logger.info(f"Loaded {len(spectra.ms2scans)} MS2 spectra")
    logger.info("finished")
    return spectra

def load_rawfilereader(sdk_path):
    import sys
    from pathlib import Path

    try:
        import clr
    except Exception as e:
        if sys.platform == "darwin":
            raise JModError("Failure on import clr. Please use 'brew install mono' to use Thermo RawFileReader .dlls or convert to mzML")
        else:
            raise e
        
    try:
        sdk_path = Path(sdk_path)
    except:
        raise JModError(f"Thermo RawFileReader path does not exist: {sdk_path}\n          If using command line please add with --rawfilereader_path 'path' or manually update in JMod/Data/Settings.json.")

    if not sdk_path.exists():
        raise JModError(f"Thermo RawFileReader path could not be found: {sdk_path}")

    if str(sdk_path) not in sys.path:
        sys.path.append(str(sdk_path))

    logger.info("Attempting to load Thermo RawFileReader .dlls")

    try:
        clr.AddReference("ThermoFisher.CommonCore.Data")
        clr.AddReference("ThermoFisher.CommonCore.RawFileReader")
        from ThermoFisher.CommonCore.RawFileReader import RawFileReaderAdapter # type: ignore
        logger.info("Thermo RawFileReader .dlls loaded successfully")
    except Exception as e:
        error_message = ("Failed to load the Thermo RawFileReader SDK.\n\n"
                        "Please verify that:\n"
                        "  1. The RawFileReader SDK path is correct.\n"
                        "  2. The folder contains the ThermoFisher.CommonCore*.dll files.\n"
                        "  3. 'ThermoFisher.CommonCore.Data.dll' and 'ThermoFisher.CommonCore.RawFileReader.dll' have been unblocked after downloading "
                        "(Right-click the ZIP → Properties → Unblock.\n"
                        "  4. The required .NET runtime is installed.\n\n"
                        f"Configured SDK path: {sdk_path}\n\n"
                        f"Original error:\n{e}")
        raise JModError(error_message) from e


_BRUKER_REMEDIATION = (
    "Point --bruker_sdk_path at the unzipped Bruker timsdata SDK folder (the one "
    "containing win64/ and linux64/), at the platform subfolder itself, or directly "
    "at the library file. You can also set \"bruker_sdk_path\" in JMod/data/settings.json. "
    "Omit --bruker_sdk_path entirely to centroid without the SDK, with m/z and 1/K0 "
    "approximated from analysis.tdf (results will not match an SDK-centroided run)."
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

    Returns None if and only if nothing was specified (None or blank), in which
    case the caller may fall back to centroiding without the SDK. Otherwise
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


