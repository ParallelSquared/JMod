"""
This Source Code Form is subject to the terms of the Oxford Nanopore
Technologies, Ltd. Public License, v. 1.0.  Full licence can be found
at https://github.com/ParallelSquared/JMod/blob/main/LICENSE.txt
"""


import subprocess
import numpy as np
from pyteomics import mzml
import os
import matplotlib.pyplot as plt
import re
import pickle
from src.logger import logger
import peppy_sage as ps

# NB this may not work for all mzml files!!!
class Spectrum:

    def __init__(self,scan=None,raw_scan=None):
        self.id = None
        self.level=None
        self.RT=None
        self.mz=None
        self.intens=None
        self.mobility=None
        self.collision_energy = None
        self.TIC=None
        self.isolation_window = None
        self.im_lo = None
        self.im_hi = None

        if scan:
            self.get_vals(scan)
        if raw_scan:
            self.get_vals_raw(raw_scan)

    def get_vals(self,scan):
        # extract values from mzml spectrum
        self.id = scan["id"]
        self.scan_num = int(re.search("scan=(\d+)",self.id)[1])
        self.level=scan["ms level"]
        self.RT = scan['scanList']['scan'][0]["scan start time"]
        self.injection_time = scan["scanList"]["scan"][0]["ion injection time"]/1000 # assume milliseconds
        self.mz = scan["m/z array"]
        self.intens = scan["intensity array"]#/self.injection_time # Normalize by injection time
        self.scanwindow = [scan["scanList"]["scan"][0]["scanWindowList"]["scanWindow"][0][i] for i in ["scan window lower limit","scan window upper limit"]]
        if self.level==2:
            self.collision_energy = scan["precursorList"]["precursor"][0]["activation"]["collision energy"]
            self.isolation_window = scan["precursorList"]["precursor"][0]["isolationWindow"]
            self.prec_mz = self.isolation_window["isolation window target m/z"]
            self.ms1window = self.isolation_window["isolation window target m/z"]+np.array([-1,1])*[self.isolation_window['isolation window lower offset'],self.isolation_window['isolation window upper offset']]
        self.TIC = scan["total ion current"]

    def get_vals_raw(self, raw_scan):
        """Extract values from a Thermo RawFileReader scan (parallel to get_vals).
        Mirrors ProteoWizard's RawFile.cpp: reads MSOrder/MassAnalyzer/precursor info
        from GetFilterForScanNumber(), not from GetScanEventForScanNumber().GetReaction().
        """
        from ThermoFisher.CommonCore.Data.Business import Scan  # type: ignore
        from ThermoFisher.CommonCore.Data.FilterEnums import MSOrderType, MassAnalyzerType  # type: ignore

        raw_file = raw_scan["raw_file"]
        scan_number = raw_scan["scan_number"]
        scan_stats = raw_scan["scan_stats"]
        scan_filter = raw_scan["scan_filter"]
        ms_order = raw_scan["ms_order"]

        self.id = f"scan={scan_number}"
        self.scan_num = scan_number
        self.level = 1 if ms_order == MSOrderType.Ms else 2
        self.RT = raw_file.RetentionTimeFromScanNumber(scan_number)

        trailer = raw_file.GetTrailerExtraInformation(scan_number)
        self.injection_time = 1.0
        for label, value in zip(trailer.Labels, trailer.Values):
            if "Ion Injection Time" in label:
                self.injection_time = float(value) / 1000  # assume milliseconds
                break

        # Centroiding: FTMS uses vendor centroid stream; everything else (including
        # Astral) falls through to Scan.ToCentroid, exactly mirroring pwiz's getMassList.
        is_ftms = scan_filter.MassAnalyzer == MassAnalyzerType.MassAnalyzerFTMS
        centroid_stream = raw_file.GetCentroidStream(scan_number, True) if is_ftms else None

        if centroid_stream is not None and centroid_stream.Length > 0:
            self.mz = np.array(list(centroid_stream.Masses), dtype=np.float64)
            self.intens = np.array(list(centroid_stream.Intensities), dtype=np.float64)
        else:
            scan = Scan.FromFile(raw_file, scan_number)
            # pwiz guard: degenerate scans get an empty spectrum, not an exception
            if scan.SegmentedScanAccess.Positions.Length == 0 or scan.ScanStatistics.BasePeakIntensity == 0:
                self.mz = np.array([], dtype=np.float64)
                self.intens = np.array([], dtype=np.float64)
            else:
                centroided = Scan.ToCentroid(scan)
                self.mz = np.array(list(centroided.SegmentedScanAccess.Positions), dtype=np.float64)
                self.intens = np.array(list(centroided.SegmentedScanAccess.Intensities), dtype=np.float64)

        self.scanwindow = [scan_stats.LowMass, scan_stats.HighMass]

        if self.level == 2:
            # index into the filter's precursor list the same way pwiz does: i = msOrder - 2
            i = int(ms_order) - 2
            precursor_mz = scan_filter.GetMass(i)
            isolation_width = scan_filter.GetIsolationWidth(i)

            self.collision_energy = scan_filter.GetEnergy(i)
            self.isolation_window = {
                "isolation window target m/z": precursor_mz,
                "isolation window lower offset": isolation_width / 2.0,
                "isolation window upper offset": isolation_width / 2.0,
            }
            self.prec_mz = precursor_mz
            self.ms1window = precursor_mz + np.array([-1, 1]) * [
                isolation_width / 2.0, isolation_width / 2.0,
            ]

        self.TIC = scan_stats.TIC

    def closest_peak(self, target_mz):
        """
        Find the index and m/z of the peak closest to the target m/z.
        Assumes self.mz is sorted.

        Parameters
        ----------
        target_mz : float
            The m/z value to match.

        Returns
        -------
        closest_idx : int
            Index of the closest peak in self.mz
        closest_mz : float
            m/z value of the closest peak
        intensity : float
            Intensity of the closest peak
        """
        mz_array = self.mz
        idx = np.searchsorted(mz_array, target_mz)

        if idx == 0:
            closest_idx = 0
        elif idx >= len(mz_array):
            closest_idx = len(mz_array) - 1
        else:
            before = mz_array[idx - 1]
            after = mz_array[idx]
            if abs(target_mz - before) <= abs(target_mz - after):
                closest_idx = idx - 1
            else:
                closest_idx = idx

        return closest_idx, mz_array[closest_idx], self.intens[closest_idx]

    def peak_list(self):
        return(np.array([self.mz,self.intens]))

    def to_rust_spectrum(self):
        #proton_mass = 1.0072764
        rs = ps.core.Spectrum(id=self.id,
                              file_id=0, # placeholder
                              scan_start_time=self.RT,
                              mz_array=self.mz,#[mz - proton_mass for mz in self.mz],
                              intensity_array=self.intens,
                              precursors=[
                                  ps.core.Precursor(self.prec_mz,
                                                    1, # will sweep whole window in WWA mode
                                                    (-1*self.isolation_window['isolation window lower offset'],
                                                    self.isolation_window['isolation window upper offset'])
                                                    )
                                  ],
                              total_ion_current=self.TIC
                              )
        return rs

    @staticmethod
    def extract_scannum(scanname):
        return int(re.search("scan=(\d+)", scanname)[1])

    
class SpectrumFile:

    def __init__(self, mzml_file=None, raw_file=None):
        self.filename = None
        self.ms2_to_ms1_map = None

        if mzml_file:
            self.load_spectra(mzml_file)
        elif raw_file:
            self.load_spectra_raw(raw_file)

    def load_spectra(self,mzml_file):
        self.filename = mzml_file
        
        # this may need to be optimised better and probably should be in the init block
        self.scan_pos = {}
        self.ms1scans = []
        self.ms2scans = []
        self.ms1_by_id = {}
        self.ms2_by_id = {}

        with mzml.MzML(mzml_file) as reader:
            for scan in reader:
                if scan["ms level"] == 1:
                    spec = Spectrum(scan)
                    idx = len(self.ms1scans)
                    self.ms1scans.append(spec)
                    self.ms1_by_id[spec.scan_num] = idx
                    self.scan_pos[spec.scan_num] = [scan["ms level"],len(self.ms1scans)-1]
                if scan["ms level"] == 2:
                    spec = Spectrum(scan)
                    idx = len(self.ms2scans)
                    self.ms2scans.append(spec)
                    self.ms2_by_id[spec.scan_num] = idx
                    self.scan_pos[spec.scan_num] = [scan["ms level"],len(self.ms2scans)-1]

        self.build_ms2_to_ms1_map()

    def load_spectra_raw(self, raw_file):
        """Load Thermo .raw data via the RawFileReader .NET SDK (parallel to load_spectra)."""
        from ThermoFisher.CommonCore.RawFileReader import RawFileReaderAdapter  # type: ignore
        from ThermoFisher.CommonCore.Data.Business import Device  # type: ignore
        from ThermoFisher.CommonCore.Data.FilterEnums import MSOrderType  # type: ignore

        self.filename = raw_file

        self.scan_pos = {}
        self.ms1scans = []
        self.ms2scans = []
        self.ms1_by_id = {}
        self.ms2_by_id = {}

        reader = RawFileReaderAdapter.FileFactory(raw_file)
        reader.SelectInstrument(Device.MS, 1)
        if reader.IsError or not reader.IsOpen:
            raise IOError(f"Could not open Thermo .raw file: {raw_file}")

        try:
            first_scan = reader.RunHeaderEx.FirstSpectrum
            last_scan = reader.RunHeaderEx.LastSpectrum

            for scan_number in range(first_scan, last_scan + 1):
                scan_filter = reader.GetFilterForScanNumber(scan_number)
                ms_order = scan_filter.MSOrder
                if ms_order not in (MSOrderType.Ms, MSOrderType.Ms2):
                    continue

                raw_scan = {
                    "raw_file": reader,
                    "scan_number": scan_number,
                    "scan_stats": reader.GetScanStatsForScanNumber(scan_number),
                    "scan_filter": scan_filter,
                    "ms_order": ms_order,
                }

                if ms_order == MSOrderType.Ms:
                    spec = Spectrum(raw_scan=raw_scan)
                    idx = len(self.ms1scans)
                    self.ms1scans.append(spec)
                    self.ms1_by_id[spec.scan_num] = idx
                    self.scan_pos[spec.scan_num] = [1, idx]
                elif ms_order == MSOrderType.Ms2:
                    spec = Spectrum(raw_scan=raw_scan)
                    idx = len(self.ms2scans)
                    self.ms2scans.append(spec)
                    self.ms2_by_id[spec.scan_num] = idx
                    self.scan_pos[spec.scan_num] = [2, idx]
        finally:
            reader.Dispose()

        self.build_ms2_to_ms1_map()

    
    def get_by_idx(self,idx):
        level, level_idx = self.scan_pos[idx]
        if level==1:
            return self.ms1scans[level_idx]
        elif level==2:
            return self.ms2scans[level_idx]

    def build_ms2_to_ms1_map(self):
        """Precompute nearest MS1 scan index for each MS2 scan.

        When IM data is present (im_lo is not None on MS1 scans), matching
        is done per IM bin: each MS2 scan is paired with the nearest-RT MS1
        scan that shares the same (im_lo, im_hi) bin.
        """
        has_im = len(self.ms1scans) > 0 and self.ms1scans[0].im_lo is not None

        if has_im:
            # Build per-IM-bin lookup: (im_lo, im_hi) -> (sorted_rt_array, ms1_index_array)
            from collections import defaultdict
            bin_to_ms1 = defaultdict(lambda: ([], []))
            for i, s in enumerate(self.ms1scans):
                key = (s.im_lo, s.im_hi)
                bin_to_ms1[key][0].append(s.RT)
                bin_to_ms1[key][1].append(i)
            # Convert to sorted numpy arrays
            self._im_bin_to_ms1 = {}
            for key, (rts, idxs) in bin_to_ms1.items():
                rt_arr = np.array(rts)
                idx_arr = np.array(idxs, dtype=int)
                order = np.argsort(rt_arr)
                self._im_bin_to_ms1[key] = (rt_arr[order], idx_arr[order])

            ms1_nums = np.array([s.scan_num for s in self.ms1scans])
            ms2_to_ms1 = np.zeros(len(self.ms2scans), dtype=int)
            for i, s2 in enumerate(self.ms2scans):
                im_key = (s2.im_lo, s2.im_hi)
                if im_key in self._im_bin_to_ms1:
                    rt_arr, idx_arr = self._im_bin_to_ms1[im_key]
                    pos = np.searchsorted(rt_arr, s2.RT)
                    if pos == 0:
                        ms2_to_ms1[i] = idx_arr[0]
                    elif pos == len(rt_arr):
                        ms2_to_ms1[i] = idx_arr[-1]
                    else:
                        before, after = rt_arr[pos - 1], rt_arr[pos]
                        ms2_to_ms1[i] = idx_arr[pos - 1] if abs(s2.RT - before) < abs(s2.RT - after) else idx_arr[pos]
                else:
                    # Fallback: nearest RT across all MS1
                    all_ms1_rts = np.array([s.RT for s in self.ms1scans])
                    pos = np.searchsorted(all_ms1_rts, s2.RT)
                    if pos == 0:
                        ms2_to_ms1[i] = 0
                    elif pos == len(all_ms1_rts):
                        ms2_to_ms1[i] = len(all_ms1_rts) - 1
                    else:
                        before, after = all_ms1_rts[pos - 1], all_ms1_rts[pos]
                        ms2_to_ms1[i] = pos - 1 if abs(s2.RT - before) < abs(s2.RT - after) else pos
        else:
            self._im_bin_to_ms1 = None
            ms1_rts = np.array([s.RT for s in self.ms1scans])
            ms1_nums = np.array([s.scan_num for s in self.ms1scans])
            ms2_to_ms1 = np.zeros(len(self.ms2scans), dtype=int)
            for i, rt in enumerate([s.RT for s in self.ms2scans]):
                pos = np.searchsorted(ms1_rts, rt)
                if pos == 0:
                    closest_idx = 0
                elif pos == len(ms1_rts):
                    closest_idx = len(ms1_rts) - 1
                else:
                    before, after = ms1_rts[pos - 1], ms1_rts[pos]
                    closest_idx = pos - 1 if abs(rt - before) < abs(rt - after) else pos
                ms2_to_ms1[i] = closest_idx
            ms1_nums = np.array([s.scan_num for s in self.ms1scans])

        self.ms2_to_ms1_map = ms2_to_ms1
        self.ms2_to_ms1_scan_num = ms1_nums[ms2_to_ms1]  # parallel array for convenience

    def get_nearest_ms1_for_scan(self, scan_id_or_num):
        """Return the closest MS1 Spectrum to the given MS2 scan (by ID or number)."""
        # Normalize to scan number
        if isinstance(scan_id_or_num, str):
            match = re.search(r"scan=(\d+)", scan_id_or_num)
            if not match:
                raise ValueError(f"Could not parse scan number from ID '{scan_id_or_num}'")
            scan_num = int(match.group(1))
        else:
            scan_num = scan_id_or_num

        if scan_num not in self.ms2_by_id:
            raise KeyError(f"MS2 scan {scan_num} not found")

        ms2_idx = self.ms2_by_id[scan_num]
        ms1_idx = self.ms2_to_ms1_map[ms2_idx]
        return self.ms1scans[ms1_idx]



