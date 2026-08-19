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

from src.utils.io.load_files import Spectrum, SpectrumFile
from src.utils.io import im_watershed
from src.logger import logger


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

    def __init__(self, filepath: str):
        self.filepath = filepath.replace("\\", "/")
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
            return _load_d(self.filepath)
        elif self.format == ".raw":
            return _load_raw(self.filepath)


def _load_mzml(filepath: str) -> SpectrumFile:
    logger.info("Loading Spectra... from .mzML file")
    return SpectrumFile(mzml_file=filepath)


def _load_raw(filepath: str) -> SpectrumFile:
    logger.info("Loading Spectra... from .raw file")
    return SpectrumFile(raw_file=filepath)


def _load_d(filepath: str) -> SpectrumFile:
    """Load timsTOF .d data from peaks.parquet + analysis.tdf."""
    logger.info("Loading Spectra... from .d file")
    d_path = filepath.rstrip("/")

    peaks_path = os.path.join(d_path, "peaks.parquet")
    tdf_path = os.path.join(d_path, "analysis.tdf")

    if not os.path.exists(tdf_path):
        raise FileNotFoundError(f"analysis.tdf not found in {d_path}")
    if not os.path.exists(peaks_path):
        raise FileNotFoundError(
            f"Peaks file not found: {peaks_path}. "
            f"Place peaks.parquet inside {d_path}."
        )

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


def _build_spectrum_file(filepath: str, peaks_path: str, cal: dict, dia_lookup: pl.DataFrame, im_bins: np.ndarray) -> SpectrumFile:
    """Read pre-calibrated peaks, assign DIA windows and IM bins, and build SpectrumFile.

    Processes the parquet in row-group batches to avoid loading 6+ GB at once.
    Expects columns: frame, scan, mz, inv_mobility, apex_intensity
    (mz and inv_mobility are pre-calibrated).
    Each non-empty IM bin produces a separate Spectrum object.
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

    # Accumulators: group peaks by (rt, bin_idx) for MS1 and (rt, prec_mz, iso_width, ce, bin_idx) for MS2
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
        mob_arr = table.column("im").to_numpy()
        intensities = table.column("apex_intensity").to_numpy()

        # RT and MS level still come from analysis.tdf (indexed by frame)
        rt_arr = rt_by_frame[frames]
        level_arr = ms_level_by_frame[frames]

        # MS1 peaks — accumulate raw peaks per frame (one spectrum per RT),
        # keeping per-peak mobility (1/K0). No IM binning.
        ms1_mask = level_arr == 1
        if ms1_mask.any():
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
                    # watershed IM-band segmentation runs in a post-pass below
                    # (it needs the whole window at once).
                    key = (f_rt, prec_mz, iso_width, ce)
                    if key not in ms2_groups:
                        ms2_groups[key] = ([], [], [])
                    ms2_groups[key][0].append(w_mz)
                    ms2_groups[key][1].append(w_intens)
                    ms2_groups[key][2].append(w_mob)

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
        intens_concat = np.concatenate(intens_list).astype(np.float64)
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
        spec.TIC = float(intens_sorted.sum())
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
    logger.info("Building MS2 spectra with overlapping fixed IM bins")
    im_watershed.reset_timings()
    for key in sorted(ms2_groups.keys()):
        rt_val, prec_mz_val, iso_width, ce = key
        mz_list, intens_list, mob_list = ms2_groups[key]
        mz_concat = np.concatenate(mz_list)
        intens_concat = np.concatenate(intens_list).astype(np.float64)
        mob_concat = np.concatenate(mob_list)
        if len(mz_concat) == 0:
            continue

        # Overlapping fixed-bin mode: group peaks by their overlapping IM bins.
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
            spec.TIC = float(intens_sorted.sum())
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

    logger.info(im_watershed.format_timings())
    logger.info(
        f"MS2 scans after IM binning: {len(sf.ms2scans)} "
        f"(original windows: {len(ms2_groups)})"
    )

    sf.build_ms2_to_ms1_map()
    return sf


def loadSpectra(input_file: str) -> SpectrumFile:
    """Drop-in replacement for load_files.loadSpectra with format dispatch."""
    logger.info("Loading Spectra...")
    # python_spec_file = input_file + "_pythonspec"
    # if not os.path.exists(python_spec_file):
    reader = FileReader(input_file)
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

