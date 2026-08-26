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

from src.utils.io.load_files import (
    Spectrum, SpectrumFile, PEAK_INT_DTYPE, PEAK_MOB_DTYPE,
)
from src.utils.io import im_watershed
from src.utils.io.tdf_bin import TdfBinReader, frame_peak_counts
from src.logger import logger

# Centroiding noise thresholds. TODO: expose via settings/CLI alongside
# --bruker_sdk_path; constant for now.
MIN_ION_COUNT_MS1 = 0.5
MIN_ION_COUNT_MS2 = 2.0

# Bruker's timsdata SDK (5.0.4) unpacks to a folder holding exactly two platform
# directories -- win64/ and linux64/ -- alongside include/, examples/ and
# thirdparty/. Bruker publishes no macOS build, so there is deliberately no
# .dylib name and no 32-bit directory here; both would be dead entries.
_BRUKER_LIB_NAME = {"win32": "timsdata.dll"}
_BRUKER_LIB_DEFAULT = "libtimsdata.so"
_BRUKER_PLATFORM_DIR = {"win32": "win64"}
_BRUKER_PLATFORM_DIR_DEFAULT = "linux64"
# Deep enough to find <sdk_root>/linux64/lib when pointed one level above the
# root; shallow enough that pointing at a large directory stays cheap.
_BRUKER_SDK_MAX_DEPTH = 2


def compute_im_bins(im_lo, im_hi, width=0.05, stride=0.025):
    """Return (N, 2) array of [bin_lo, bin_hi] for overlapping IM bins.

    ``width = 2 * stride`` (50% overlap) is required by ``_assign_peaks_to_im_bins``,
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


def _load_d(filepath: str, bruker_sdk_path: str = None) -> SpectrumFile:
    """Load timsTOF .d data from peaks.parquet + analysis.tdf.

    If peaks.parquet is absent it is generated first, by centroiding the raw
    signal in analysis.tdf_bin (see ``_centroid_d``), and then read normally.
    Delete peaks.parquet to force a re-centroid, e.g. after changing the
    thresholds.
    """
    logger.info("Loading Spectra... from .d file")
    d_path = filepath.rstrip("/")

    peaks_path = os.path.join(d_path, "peaks.parquet")
    tdf_path = os.path.join(d_path, "analysis.tdf")
    bin_path = os.path.join(d_path, "analysis.tdf_bin")

    if not os.path.exists(tdf_path):
        raise FileNotFoundError(f"analysis.tdf not found in {d_path}")

    if not os.path.exists(peaks_path):
        if not os.path.exists(bin_path):
            raise FileNotFoundError(
                f"Peaks file not found: {peaks_path}, and {bin_path} is absent "
                f"too, so it cannot be created. Place a peaks.parquet inside "
                f"{d_path}."
            )
        logger.info(f"No peaks.parquet in {d_path}; centroiding the raw data")
        _centroid_d(d_path, peaks_path, bruker_sdk_path)

    logger.info(f"Loading calibration from {d_path}")
    cal = _load_calibration(tdf_path)
    dia_lookup = _load_dia_windows(tdf_path, cal["frame_ids"], cal["ms_level"])

    im_bins = compute_im_bins(cal["im_range_lower"], cal["im_range_upper"])
    logger.info(f"Computed {len(im_bins)} overlapping IM bins "
                f"[{cal['im_range_lower']:.4f}, {cal['im_range_upper']:.4f}]")

    logger.info(f"Reading peaks from {peaks_path}")
    sf = _build_spectrum_file(filepath, peaks_path, cal, dia_lookup, im_bins)

    logger.info(f"Loaded {len(sf.ms1scans)} MS1 spectra, {len(sf.ms2scans)} MS2 spectra")
    return sf


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


def _assign_peaks_to_im_bins(mob_arr, im_bins):
    """Assign each peak to its overlapping IM bins.

    Returns arrays (peak_indices, bin_indices) where each peak may appear
    in up to 2 bins due to 50% overlap.  Vectorized, O(N_peaks).
    """
    mob_arr = np.asarray(mob_arr, dtype=np.float64)
    bin_lo0 = im_bins[0, 0]
    stride = im_bins[1, 0] - im_bins[0, 0]  # 0.0125
    n_bins = len(im_bins)

    # Primary bin index for each peak; the two overlapping candidates are
    # raw_idx and raw_idx-1 (50% overlap => a peak lands in at most 2 bins).
    raw_idx = ((mob_arr - bin_lo0) / stride).astype(np.int64)
    all_peaks = np.arange(mob_arr.shape[0], dtype=np.int64)

    peak_chunks = []
    bin_chunks = []
    for cand in (raw_idx, raw_idx - 1):
        valid = (cand >= 0) & (cand < n_bins)
        cv = cand[valid]
        pv = all_peaks[valid]
        mv = mob_arr[valid]
        inb = (im_bins[cv, 0] <= mv) & (mv < im_bins[cv, 1])
        peak_chunks.append(pv[inb])
        bin_chunks.append(cv[inb])

    return (np.concatenate(peak_chunks) if peak_chunks else np.empty(0, np.int64),
            np.concatenate(bin_chunks) if bin_chunks else np.empty(0, np.int64))


def _band_ms2_groups(sf, ms2_groups, im_bins, scan_counter):
    """Segment each (frame, window) group into IM bands, one Spectrum per band.

    ``ms2_groups`` maps ``(rt, prec_mz, iso_width, ce)`` to concatenated
    ``(mz, intens, mob)`` arrays of that window's un-banded peaks.  Appends a
    Spectrum per band to ``sf.ms2scans`` and updates ``sf.ms2_by_id`` /
    ``sf.scan_pos``; returns the next free scan number.

    Consumes ``ms2_groups``: each window's un-banded arrays are released as soon
    as its bands are built.  The bands are ~2x the size of the un-banded peaks
    they come from (the 50% overlap duplicates every boundary peak), so holding
    both in full at once costs an extra ~9 GB on a large timsTOF .d.  Callers
    that need the un-banded peaks afterwards pass a shallow copy of the dict.

    Overlapping fixed-bin mode (the pre-watershed "overlapping windows"): each
    peak lands in up to 2 overlapping fixed IM bins (denormalized).  Swap the
    ``bands = ...`` line to re-enable the data-driven watershed.
    """
    im_watershed.reset_timings()
    for key in sorted(ms2_groups.keys()):
        rt_val, prec_mz_val, iso_width, ce = key
        mz_concat, intens_concat, mob_concat = ms2_groups.pop(key)
        if len(mz_concat) == 0:
            continue

        # bands = im_watershed.segment_window(mob_concat, intens_concat)  # watershed mode
        _pidxs, _bidxs = _assign_peaks_to_im_bins(mob_concat, im_bins)
        bands = []
        if _bidxs.shape[0] > 0:
            # sort by bin, then split into contiguous per-bin chunks (no repeated scans)
            _order = np.argsort(_bidxs, kind="stable")
            _sb = _bidxs[_order]
            _sp = _pidxs[_order]
            _cuts = np.flatnonzero(np.diff(_sb)) + 1
            _starts = np.concatenate(([0], _cuts))
            for _s, _chunk in zip(_starts, np.split(_sp, _cuts)):
                b = int(_sb[_s])
                bands.append((float(im_bins[b, 0]), float(im_bins[b, 1]), _chunk))

        half_width = iso_width / 2.0
        t0 = perf_counter()
        for im_lo, im_hi, peak_idx in bands:
            band_mz = mz_concat[peak_idx]
            band_intens = intens_concat[peak_idx]
            band_mob = mob_concat[peak_idx]

            # Sort by m/z (mobility kept parallel so spec.mobility[i] matches mz[i])
            order = np.argsort(band_mz)
            mz_sorted = band_mz[order]
            intens_sorted = band_intens[order]
            mob_sorted = band_mob[order]

            spec = Spectrum()
            spec.id = f"scan={scan_counter}"
            spec.scan_num = scan_counter
            spec.level = 2
            spec.RT = rt_val
            spec.mz = mz_sorted
            spec.intens = intens_sorted
            spec.mobility = mob_sorted
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
            spec.im_lo = im_lo
            spec.im_hi = im_hi
            spec.scanwindow = [float(mz_sorted[0]), float(mz_sorted[-1])]

            idx = len(sf.ms2scans)
            sf.ms2scans.append(spec)
            sf.ms2_by_id[scan_counter] = idx
            sf.scan_pos[scan_counter] = [2, idx]
            scan_counter += 1
        im_watershed.TIMINGS["spectrum_construct"] += perf_counter() - t0
    return scan_counter


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

    # Build numpy lookup arrays indexed by frame ID for fast vectorized access
    max_frame = int(cal["frame_ids"].max())
    rt_by_frame = np.zeros(max_frame + 1, dtype=np.float64)
    ms_level_by_frame = np.zeros(max_frame + 1, dtype=np.uint8)
    for i, fid in enumerate(cal["frame_ids"]):
        rt_by_frame[int(fid)] = cal["rt_values"][i]
        ms_level_by_frame[int(fid)] = cal["ms_level"][i]

    # Build DIA lookup as a dict: frame -> list of (scan_begin, scan_end, prec_mz, iso_width, ce)
    dia_dict = {}
    for row in dia_lookup.iter_rows():
        frame, scan_begin, scan_end, prec_mz, iso_width, ce = row
        dia_dict.setdefault(int(frame), []).append(
            (int(scan_begin), int(scan_end), float(prec_mz), float(iso_width), float(ce))
        )

    # Accumulators: group peaks by rt for MS1 and (rt, prec_mz, iso_width, ce) for MS2
    ms1_groups = {}  # rt -> (mz_list, intens_list, mob_list); one frame per rt
    # (rt, prec_mz, iso_width, ce) -> (mz_list, intens_list, mob_list); raw peaks
    # accumulated across row groups, then IM-band segmented in a post-pass.
    ms2_groups = {}

    pf = pq.ParquetFile(peaks_path)
    n_row_groups = pf.metadata.num_row_groups
    logger.info(f"Processing {n_row_groups} row groups from {peaks_path}")

    for rg_idx in range(n_row_groups):
        table = pf.read_row_group(rg_idx)
        frames = table.column("frame").to_numpy()
        scans = table.column("scan").to_numpy()
        mz_arr = table.column("mz").to_numpy()
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
        ms1_mask = level_arr == 1 if include_ms1 else None
        if include_ms1 and ms1_mask.any():
            ms1_mz = mz_arr[ms1_mask]
            ms1_intens = intensities[ms1_mask]
            ms1_mob = mob_arr[ms1_mask]
            ms1_rt = rt_arr[ms1_mask]

            for rt in np.unique(ms1_rt):
                sel = ms1_rt == rt
                key = float(rt)
                if key not in ms1_groups:
                    ms1_groups[key] = ([], [], [])
                ms1_groups[key][0].append(ms1_mz[sel])
                ms1_groups[key][1].append(ms1_intens[sel])
                ms1_groups[key][2].append(ms1_mob[sel])

        # MS2 peaks: assign to DIA windows
        ms2_mask = level_arr == 2
        if ms2_mask.any():
            ms2_frames = frames[ms2_mask]
            ms2_scans = scans[ms2_mask]
            ms2_mz = mz_arr[ms2_mask]
            ms2_intens = intensities[ms2_mask]
            ms2_mob = mob_arr[ms2_mask]
            ms2_rt = rt_arr[ms2_mask]

            unique_frames = np.unique(ms2_frames)
            for frame_id in unique_frames:
                windows = dia_dict.get(int(frame_id))
                if windows is None:
                    continue
                frame_mask = ms2_frames == frame_id
                f_scans = ms2_scans[frame_mask]
                f_mz = ms2_mz[frame_mask]
                f_intens = ms2_intens[frame_mask]
                f_mob = ms2_mob[frame_mask]
                f_rt = float(ms2_rt[frame_mask][0])

                for scan_begin, scan_end, prec_mz, iso_width, ce in windows:
                    win_mask = (f_scans >= scan_begin) & (f_scans < scan_end)
                    if not win_mask.any():
                        continue
                    w_mz = f_mz[win_mask]
                    w_intens = f_intens[win_mask]
                    w_mob = f_mob[win_mask]

                    # Accumulate raw peaks per (frame, window) across row groups;
                    # IM-band segmentation runs in a post-pass (it needs the whole
                    # window at once).
                    key = (f_rt, prec_mz, iso_width, ce)
                    if key not in ms2_groups:
                        ms2_groups[key] = ([], [], [])
                    ms2_groups[key][0].append(w_mz)
                    ms2_groups[key][1].append(w_intens)
                    ms2_groups[key][2].append(w_mob)

    # Concatenate MS2 in place, dropping the per-row-group fragments as we go so
    # the list-of-chunks form never coexists in full with the concatenated form.
    for key in list(ms2_groups):
        v = ms2_groups.pop(key)
        ms2_groups[key] = (np.concatenate(v[0]), np.concatenate(v[1]),
                           np.concatenate(v[2]))

    return ms1_groups, ms2_groups


def _read_raw_peak_groups(d_path: str, cal: dict, dia_lookup: pl.DataFrame,
                          include_ms1: bool = True, min_intensity: int = 0):
    """Group raw ``analysis.tdf_bin`` peaks by acquisition, without IM banding.

    Same contract as :func:`_read_peak_groups` -- returns
    ``(ms1_groups, ms2_groups)`` in the shapes ``_build_spectrum_file`` and
    ``_band_ms2_groups`` expect -- but reads every stored ion from the binary
    instead of the centroider's peak list.  Downstream code cannot tell the two
    apart.

    Calibration is estimation mode: the ``mz_values`` / ``mobility_values``
    lookup arrays that ``_load_calibration`` already builds from
    ``GlobalMetadata``.  These approximate the instrument response rather than
    reproducing it, so m/z here is *not* equivalent to a peaks.parquet produced
    with the Bruker SDK.  The real calibration coefficients sit unused in the
    ``MzCalibration`` / ``TimsCalibration`` tables of the same database; when a
    converter for them exists, it drops in at the two lookups below and nothing
    else in this function changes.

    ``min_intensity`` drops peaks at or below a raw detector count before
    grouping.  It defaults to 0 (keep everything), since recovering the signal
    the centroider discards is the point of this path; raise it if the
    accumulators do not fit in memory.
    """
    mz_values = cal["mz_values"]
    mobility_values = cal["mobility_values"]

    max_frame = int(cal["frame_ids"].max())
    rt_by_frame = np.zeros(max_frame + 1, dtype=np.float64)
    ms_level_by_frame = np.zeros(max_frame + 1, dtype=np.uint8)
    for i, fid in enumerate(cal["frame_ids"]):
        rt_by_frame[int(fid)] = cal["rt_values"][i]
        ms_level_by_frame[int(fid)] = cal["ms_level"][i]

    dia_dict = {}
    for row in dia_lookup.iter_rows():
        frame, scan_begin, scan_end, prec_mz, iso_width, ce = row
        dia_dict.setdefault(int(frame), []).append(
            (int(scan_begin), int(scan_end), float(prec_mz), float(iso_width), float(ce))
        )

    ms1_groups = {}
    ms2_groups = {}

    total_peaks = frame_peak_counts(d_path)
    logger.info(f"Reading raw frames from {d_path}/analysis.tdf_bin "
                f"({total_peaks} stored peaks)")

    kept = 0
    with TdfBinReader(d_path) as reader:
        for frame_id in reader.frame_ids:
            fid = int(frame_id)
            level = ms_level_by_frame[fid]
            if level == 0:
                continue  # frame absent from the calibration table
            if level == 1 and not include_ms1:
                continue
            if level == 2 and fid not in dia_dict:
                continue

            scans, tof_indices, intensities = reader.read_frame(fid)
            if scans.shape[0] == 0:
                continue

            if min_intensity > 0:
                keep = intensities > min_intensity
                if not keep.any():
                    continue
                scans = scans[keep]
                tof_indices = tof_indices[keep]
                intensities = intensities[keep]

            # Estimation-mode calibration by table lookup. Indices come from the
            # instrument and should always be in range; clipping keeps a corrupt
            # frame from taking the whole run down.
            mz = mz_values[np.clip(tof_indices, 0, len(mz_values) - 1)]
            mob = mobility_values[
                np.clip(scans, 0, len(mobility_values) - 1)
            ].astype(PEAK_MOB_DTYPE)
            intens = intensities.astype(PEAK_INT_DTYPE)
            rt = float(rt_by_frame[fid])
            kept += scans.shape[0]

            if level == 1:
                # One frame per RT, so this key is written exactly once.
                ms1_groups.setdefault(rt, ([], [], []))
                ms1_groups[rt][0].append(mz)
                ms1_groups[rt][1].append(intens)
                ms1_groups[rt][2].append(mob)
                continue

            for scan_begin, scan_end, prec_mz, iso_width, ce in dia_dict[fid]:
                win_mask = (scans >= scan_begin) & (scans < scan_end)
                if not win_mask.any():
                    continue
                key = (rt, prec_mz, iso_width, ce)
                if key not in ms2_groups:
                    ms2_groups[key] = ([], [], [])
                ms2_groups[key][0].append(mz[win_mask])
                ms2_groups[key][1].append(intens[win_mask])
                ms2_groups[key][2].append(mob[win_mask])

    # Match _read_peak_groups: MS2 arrives concatenated, MS1 as chunk lists.
    for key in list(ms2_groups):
        v = ms2_groups.pop(key)
        ms2_groups[key] = (np.concatenate(v[0]), np.concatenate(v[1]),
                           np.concatenate(v[2]))

    logger.info(f"Read {kept} raw peaks into {len(ms1_groups)} MS1 frames "
                f"and {len(ms2_groups)} MS2 windows")
    return ms1_groups, ms2_groups


def _reread_peak_groups(sf: SpectrumFile, include_ms1: bool = False):
    """Re-read un-banded peaks from whichever source built ``sf``.

    ``reband_ms2`` deliberately does not keep the un-banded peaks in memory, so
    it needs to go back to the data. Both ingest paths leave the .d on disk;
    this picks the right reader.
    """
    raw_path = getattr(sf, "_raw_d_path", None)
    if raw_path is not None:
        return _read_raw_peak_groups(raw_path, sf._cal, sf._dia_lookup,
                                     include_ms1=include_ms1,
                                     min_intensity=getattr(sf, "_min_intensity", 0))
    return _read_peak_groups(sf._peaks_path, sf._cal, sf._dia_lookup,
                             include_ms1=include_ms1)


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
      and calibrated here in estimation mode (see ``_read_raw_peak_groups``).

    Exactly one of the two must be given.  Each non-empty IM bin produces a
    separate Spectrum object.
    """
    if (peaks_path is None) == (raw_d_path is None):
        raise ValueError("Pass exactly one of peaks_path or raw_d_path")

    if raw_d_path is not None:
        ms1_groups, ms2_groups = _read_raw_peak_groups(
            raw_d_path, cal, dia_lookup, min_intensity=min_intensity
        )
    else:
        ms1_groups, ms2_groups = _read_peak_groups(peaks_path, cal, dia_lookup)

    # Build SpectrumFile from accumulated groups
    sf = SpectrumFile()
    sf.filename = filepath
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
    n_windows = len(ms2_groups)

    logger.info("Building MS2 spectra with overlapping fixed IM bins")
    scan_counter = _band_ms2_groups(sf, ms2_groups, im_bins, scan_counter)
    del ms2_groups
    logger.info(im_watershed.format_timings())
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
    ``width / 2``.  That relationship is required by ``_assign_peaks_to_im_bins``,
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
    _, groups = _reread_peak_groups(sf, include_ms1=False)

    # _band_ms2_groups drains the dict as it goes, so the un-banded peaks are
    # released window by window instead of sitting in memory alongside the ~2x
    # larger banded set being allocated.
    _band_ms2_groups(sf, groups, new_bins, scan_counter)
    del groups
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
        from src.utils.gui_utils import send_raise_to_TK
        if sys.platform == "darwin":
            send_raise_to_TK("Failure on import clr. Please use 'brew install mono' to use Thermo RawFileReader .dlls or convert to mzML")
            raise RuntimeError("Failure on import clr. Please use 'brew install mono' to use Thermo RawFileReader .dlls or convert to mzML")
        else:
            raise e
        
    try:
        sdk_path = Path(sdk_path)
    except:
        from src.utils.gui_utils import send_raise_to_TK
        send_raise_to_TK(f"Thermo RawFileReader path does not exist: {sdk_path}\n          If using command line please add with --rawfilereader_path 'path' or manually update in JMod/Data/Settings.json.")
        raise FileNotFoundError(f"Thermo RawFileReader path does not exist: {sdk_path}\n          If using command line please add with --rawfilereader_path 'path' or manually update in JMod/Data/Settings.json.")

    if not sdk_path.exists():
        from src.utils.gui_utils import send_raise_to_TK
        send_raise_to_TK(f"Thermo RawFileReader path could not be found: {sdk_path}")
        raise FileNotFoundError(f"Thermo RawFileReader path could not be found: {sdk_path}")

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
        from src.utils.gui_utils import send_raise_to_TK
        send_raise_to_TK(error_message)
        raise ValueError(error_message) from e


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
    """Report a bad --bruker_sdk_path the way the rest of the codebase does.

    send_raise_to_TK sets config.error_already_handled, which lets @log_exceptions
    exit cleanly instead of dumping a traceback, so it must come before the raise.
    """
    from src.utils.gui_utils import send_raise_to_TK
    full = f"{message}\n\n{_BRUKER_REMEDIATION}"
    send_raise_to_TK(full)
    raise FileNotFoundError(full)


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


def resolve_bruker_sdk_path(sdk_path, *, platform=None):
    """Resolve ``--bruker_sdk_path`` to a concrete timsdata library file.

    Returns None if and only if nothing was specified (None or blank), in which
    case the caller may fall back to centroiding without the SDK. Otherwise
    returns an absolute path to an existing library file -- never a directory.
    Raises FileNotFoundError (after send_raise_to_TK) when a path was given but
    no library could be found: a specified-but-unresolvable SDK is an error, not
    a reason to silently approximate.

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

    if sdk_path is None:
        return None
    # Strip quotes left over from shell copy-paste of a path with spaces.
    raw = str(sdk_path).strip().strip("\"'").strip()
    if not raw:
        return None

    if platform == "darwin":
        _fail_bruker(
            f"--bruker_sdk_path was given ({raw}), but Bruker does not publish a "
            f"macOS build of the timsdata SDK -- there is no library to load. "
            f"Real-calibration centroiding of .d data requires Linux or Windows "
            f"(for example the JMod Docker container)."
        )

    lib_name = _bruker_lib_name(platform)
    p = Path(raw).expanduser()

    # An explicit file is taken at the user's word, whatever it is named -- this
    # covers versioned or renamed copies such as libtimsdata.so.2.21.
    if p.is_file():
        return str(p.resolve())

    if not p.is_dir():
        _fail_bruker(
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

    _fail_bruker(
        f"No Bruker timsdata library ({lib_name}) found in {raw} or in its "
        f"subdirectories (searched {_BRUKER_SDK_MAX_DEPTH} levels deep).\n"
        f"The Bruker SDK zip unpacks to a folder containing win64/timsdata.dll "
        f"and linux64/libtimsdata.so -- point at that folder, or at {lib_name} "
        f"directly."
    )

