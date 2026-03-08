"""
This Source Code Form is subject to the terms of the Oxford Nanopore
Technologies, Ltd. Public License, v. 1.0.  Full licence can be found
at https://github.com/ParallelSquared/JMod/blob/main/LICENSE.txt
"""

import os
import pickle
import sqlite3

import numpy as np
import numba as nb
import polars as pl

from src.utils.io.load_files import Spectrum, SpectrumFile
from src.logger import logger

# Ion mobility filter: log(IM) = slope * log(mz) + intercept
# Derived from identified peptides; peaks beyond n_sd * residual_sd are removed
_IM_SLOPE = 0.4724
_IM_INTERCEPT = -3.1007
_IM_RESIDUAL_SD = 0.0366
_IM_N_SD = 4.0


def _im_filter(mz, mobility):
    """Return boolean mask keeping peaks within _IM_N_SD of the expected m/z-mobility line."""
    expected_log_mob = _IM_SLOPE * np.log(mz) + _IM_INTERCEPT
    residual = np.abs(np.log(mobility) - expected_log_mob)
    return residual <= _IM_N_SD * _IM_RESIDUAL_SD


@nb.njit
def collapse_tof(mz_vals, intensities, mobilities, ppm_tol=10.0):
    """Merge peaks within ppm tolerance by intensity-weighted m/z and mobility averaging.

    Processes peaks in descending intensity order. For each unmerged peak,
    gathers all unmerged neighbors within ppm_tol (computed per-pair) and
    merges them into a single peak with summed intensity, weighted m/z,
    and weighted ion mobility (1/K0).

    mz_vals: 1D array of calibrated m/z values (unsorted, from multiple IM scans)
    intensities: 1D array of corresponding intensities
    mobilities: 1D array of ion mobility values (1/K0)
    ppm_tol: merging tolerance in ppm
    Returns: (centroided_mz, summed_intensity, weighted_mobility) sorted by m/z
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
        wsum_mob = mob_sorted[idx] * int_sorted[idx]
        merged[idx] = True

        # Search left
        j = idx - 1
        while j >= 0 and not merged[j]:
            tol_da = mz_sorted[j + 1] * ppm_tol * 1e-6
            if (mz_sorted[j + 1] - mz_sorted[j]) > tol_da:
                break
            wsum_mz += mz_sorted[j] * int_sorted[j]
            wsum_int += int_sorted[j]
            wsum_mob += mob_sorted[j] * int_sorted[j]
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
            wsum_mob += mob_sorted[j] * int_sorted[j]
            merged[j] = True
            j += 1

        out_mz[n_out] = wsum_mz / wsum_int
        out_int[n_out] = wsum_int
        out_mob[n_out] = wsum_mob / wsum_int
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
    return SpectrumFile(filepath)


def _load_raw(filepath: str) -> SpectrumFile:
    raise NotImplementedError(f".raw file support is not yet implemented ({filepath})")


def _load_d(filepath: str) -> SpectrumFile:
    """Load timsTOF .d data from .peaks.parquet + analysis.tdf."""
    d_path = filepath.rstrip("/")
    parent_dir = os.path.dirname(d_path)
    base_name = os.path.basename(d_path)
    if base_name.endswith(".d"):
        base_name = base_name[:-2]

    peaks_path = os.path.join(parent_dir, base_name + ".peaks.parquet")
    tdf_path = os.path.join(d_path, "analysis.tdf")

    if not os.path.exists(tdf_path):
        raise FileNotFoundError(f"analysis.tdf not found in {d_path}")
    if not os.path.exists(peaks_path):
        raise FileNotFoundError(
            f"Peaks file not found: {peaks_path}. "
            f"Run timscentroid on {d_path} first."
        )

    logger.info(f"IM filter: {_IM_N_SD} SD cutoff (slope={_IM_SLOPE}, intercept={_IM_INTERCEPT}, sd={_IM_RESIDUAL_SD})")
    logger.info(f"Loading calibration from {d_path}")
    cal = _load_calibration(tdf_path)
    dia_lookup = _load_dia_windows(tdf_path, cal["frame_ids"], cal["ms_level"])

    logger.info(f"Reading and calibrating peaks from {peaks_path}")
    sf = _build_spectrum_file(filepath, peaks_path, cal, dia_lookup)

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


def _build_spectrum_file(filepath: str, peaks_path: str, cal: dict, dia_lookup: pl.DataFrame) -> SpectrumFile:
    """Read raw peaks, calibrate, assign DIA windows, and build SpectrumFile.

    Processes the parquet in row-group batches to avoid loading 6+ GB at once.
    Uses numpy indexing for calibration (no polars joins needed for mz/mobility/rt).
    """
    import pyarrow.parquet as pq

    # Build numpy lookup arrays indexed by frame ID for fast vectorized access
    frame_to_idx = {}
    for i, fid in enumerate(cal["frame_ids"]):
        frame_to_idx[int(fid)] = i
    max_frame = int(cal["frame_ids"].max())
    rt_by_frame = np.zeros(max_frame + 1, dtype=np.float64)
    ms_level_by_frame = np.zeros(max_frame + 1, dtype=np.uint8)
    for i, fid in enumerate(cal["frame_ids"]):
        rt_by_frame[int(fid)] = cal["rt_values"][i]
        ms_level_by_frame[int(fid)] = cal["ms_level"][i]

    mz_values = cal["mz_values"]
    mobility_values = cal["mobility_values"]

    # Build DIA lookup as a dict: frame -> list of (scan_begin, scan_end, prec_mz, iso_width, ce)
    dia_dict = {}
    for row in dia_lookup.iter_rows():
        frame, scan_begin, scan_end, prec_mz, iso_width, ce = row
        dia_dict.setdefault(int(frame), []).append(
            (int(scan_begin), int(scan_end), float(prec_mz), float(iso_width), float(ce))
        )

    # Warm up numba JIT
    _dummy_mz = np.array([100.0, 100.001], dtype=np.float64)
    _dummy_int = np.array([1.0, 1.0], dtype=np.float64)
    _dummy_mob = np.array([1.0, 1.0], dtype=np.float64)
    collapse_tof(_dummy_mz, _dummy_int, _dummy_mob, 10.0)

    # Accumulators: group peaks by (rt,) for MS1 and (rt, prec_mz, iso_width, ce) for MS2
    ms1_groups = {}  # rt -> (mz_list, intens_list, mob_list)
    ms2_groups = {}  # (rt, prec_mz, iso_width, ce) -> (mz_arr, intens_arr, mob_arr)

    pf = pq.ParquetFile(peaks_path)
    n_row_groups = pf.metadata.num_row_groups
    logger.info(f"Processing {n_row_groups} row groups from {peaks_path}")

    for rg_idx in range(n_row_groups):
        table = pf.read_row_group(rg_idx)
        frames = table.column("frame").to_numpy()
        scans = table.column("scan").to_numpy()
        tofs = table.column("tof").to_numpy()
        intensities = table.column("apex_intensity").to_numpy()

        # Vectorized calibration via numpy indexing
        mz_arr = mz_values[tofs]
        mob_arr = mobility_values[scans]
        rt_arr = rt_by_frame[frames]
        level_arr = ms_level_by_frame[frames]

        # MS1 peaks
        ms1_mask = level_arr == 1
        if ms1_mask.any():
            ms1_mz = mz_arr[ms1_mask]
            ms1_intens = intensities[ms1_mask]
            ms1_mob = mob_arr[ms1_mask]
            ms1_rt = rt_arr[ms1_mask]

            unique_rts = np.unique(ms1_rt)
            for rt in unique_rts:
                rt_mask = ms1_rt == rt
                key = float(rt)
                if key not in ms1_groups:
                    ms1_groups[key] = ([], [], [])
                ms1_groups[key][0].append(ms1_mz[rt_mask])
                ms1_groups[key][1].append(ms1_intens[rt_mask])
                ms1_groups[key][2].append(ms1_mob[rt_mask])

        # MS2 peaks: assign to DIA windows (vectorized per frame)
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
                    key = (f_rt, prec_mz, iso_width, ce)
                    if key not in ms2_groups:
                        ms2_groups[key] = (f_mz[win_mask], f_intens[win_mask], f_mob[win_mask])
                    else:
                        ms2_groups[key] = (
                            np.concatenate([ms2_groups[key][0], f_mz[win_mask]]),
                            np.concatenate([ms2_groups[key][1], f_intens[win_mask]]),
                            np.concatenate([ms2_groups[key][2], f_mob[win_mask]]),
                        )

    # Build SpectrumFile from accumulated groups
    sf = SpectrumFile()
    sf.filename = filepath
    sf.scan_pos = {}
    sf.ms1scans = []
    sf.ms2scans = []
    sf.ms1_by_id = {}
    sf.ms2_by_id = {}

    scan_counter = 1

    # MS1 spectra (sorted by RT), collapse mobility duplicates
    logger.info("Building MS1 spectra with mobility collapse")
    for rt_val in sorted(ms1_groups.keys()):
        mz_list, intens_list, mob_list = ms1_groups[rt_val]
        mz_concat = np.concatenate(mz_list)
        intens_concat = np.concatenate(intens_list).astype(np.float64)
        mob_concat = np.concatenate(mob_list).astype(np.float64)
        keep = _im_filter(mz_concat, mob_concat)
        if not keep.any():
            continue
        mz_collapsed, intens_collapsed, mob_collapsed = collapse_tof(
            mz_concat[keep], intens_concat[keep], mob_concat[keep], 10.0
        )

        spec = Spectrum()
        spec.id = f"scan={scan_counter}"
        spec.scan_num = scan_counter
        spec.level = 1
        spec.RT = rt_val
        spec.mz = mz_collapsed
        spec.intens = intens_collapsed
        spec.mobility = mob_collapsed
        spec.TIC = float(intens_collapsed.sum())
        spec.injection_time = 1.0
        spec.collision_energy = None
        spec.isolation_window = None
        spec.scanwindow = [float(spec.mz[0]), float(spec.mz[-1])]

        idx = len(sf.ms1scans)
        sf.ms1scans.append(spec)
        sf.ms1_by_id[scan_counter] = idx
        sf.scan_pos[scan_counter] = [1, idx]
        scan_counter += 1

    # MS2 spectra (sorted by RT, then prec_mz), collapse mobility duplicates
    logger.info("Building MS2 spectra with mobility collapse")
    for key in sorted(ms2_groups.keys()):
        rt_val, prec_mz_val, iso_width, ce = key
        mz_raw, intens_raw, mob_raw = ms2_groups[key]
        mz_f64 = mz_raw.astype(np.float64)
        intens_f64 = intens_raw.astype(np.float64)
        mob_f64 = mob_raw.astype(np.float64)
        keep = _im_filter(mz_f64, mob_f64)
        if not keep.any():
            continue
        mz_collapsed, intens_collapsed, mob_collapsed = collapse_tof(
            mz_f64[keep], intens_f64[keep], mob_f64[keep], 10.0
        )
        half_width = iso_width / 2.0

        spec = Spectrum()
        spec.id = f"scan={scan_counter}"
        spec.scan_num = scan_counter
        spec.level = 2
        spec.RT = rt_val
        spec.mz = mz_collapsed
        spec.intens = intens_collapsed
        spec.mobility = mob_collapsed
        spec.TIC = float(intens_collapsed.sum())
        spec.injection_time = 1.0
        spec.collision_energy = ce
        spec.prec_mz = prec_mz_val
        spec.isolation_window = {
            "isolation window target m/z": prec_mz_val,
            "isolation window lower offset": half_width,
            "isolation window upper offset": half_width,
        }
        spec.ms1window = prec_mz_val + np.array([-1, 1]) * half_width
        spec.scanwindow = [float(spec.mz[0]), float(spec.mz[-1])]

        idx = len(sf.ms2scans)
        sf.ms2scans.append(spec)
        sf.ms2_by_id[scan_counter] = idx
        sf.scan_pos[scan_counter] = [2, idx]
        scan_counter += 1

    sf.build_ms2_to_ms1_map()
    return sf


def loadSpectra(input_file: str) -> SpectrumFile:
    """Drop-in replacement for load_files.loadSpectra with format dispatch."""
    logger.info("Loading Spectra...")
    python_spec_file = input_file + "_pythonspec"
    if not os.path.exists(python_spec_file):
        logger.info("Loading Spectra... from file")
        reader = FileReader(input_file)
        spectra = reader.read()
        with open(python_spec_file, "wb") as f:
            pickle.dump(spectra, f)
    else:
        with open(python_spec_file, "rb") as f:
            logger.info("Loading Spectra... from pickle")
            spectra = pickle.load(f)

    logger.info(f"Loaded {len(spectra.ms1scans)} MS1 spectra")
    logger.info(f"Loaded {len(spectra.ms2scans)} MS2 spectra")
    logger.info("finished")
    return spectra
