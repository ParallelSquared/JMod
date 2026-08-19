
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

import subprocess
import numpy as np
from pyteomics import mzml
import os
import matplotlib.pyplot as plt
import re
import pickle
from src.logger import logger
import peppy_sage as ps

# Peak-array dtypes for Spectrum. m/z stays float64 — ppm-level matching needs the
# mantissa. Intensity and mobility are float32: a timsTOF .d run can hold >1e9 peak
# entries after IM-band denormalization, where each float64 array costs ~9 GB, and
# neither quantity needs more than float32's ~1e-7 relative precision (1/K0 spans
# ~0.7-1.3 against IM tolerances of ~1e-2; intensities are used relatively).
# Sum intensities with ``dtype=np.float64`` to keep float32 accumulation error out
# of TIC-style reductions.
PEAK_MZ_DTYPE = np.float64
PEAK_INT_DTYPE = np.float32
PEAK_MOB_DTYPE = np.float32


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
        self._ms2_flat = None

        if mzml_file:
            self.load_spectra(mzml_file)
        elif raw_file:
            self.load_spectra_raw(raw_file)

    def flatten_ms2_peaks(self):
        """Return the MS2 peaks as CSR-style flat arrays, built at most once.

        Returns ``(peak_mz, peak_int, peak_mob, offsets)``, where scan ``i``'s peaks
        are ``peak_mz[offsets[i]:offsets[i+1]]``. ``peak_mob`` is zero-length when the
        data has no ion mobility (e.g. mzML); callers must gate on that before using
        it. Consumers that need float64 should upcast the slice they actually touch.

        This does not merely copy: as each scan is written into the flat buffer, that
        scan's ``mz`` / ``intens`` / ``mobility`` are repointed at the corresponding
        *views*, dropping the last reference to the standalone per-scan arrays. Peak
        extra memory is therefore one scan, not a second copy of the run — which for a
        timsTOF .d holding ~1e9 MS2 peak entries is the difference between ~19 GB and
        an OOM kill. Values, dtypes and shapes are unchanged, so the repointing is
        invisible to every other consumer of ``ms2scans``.

        The result is cached on the instance; the first caller pays for it. Not
        safe to call concurrently from multiple threads on the same instance.
        """
        if self._ms2_flat is not None:
            return self._ms2_flat

        scans = self.ms2scans
        n = len(scans)
        lengths = np.fromiter((len(s.mz) for s in scans), dtype=np.int64, count=n)
        offsets = np.zeros(n + 1, dtype=np.int64)
        if n:
            np.cumsum(lengths, out=offsets[1:])
        total = int(offsets[-1])

        peak_mz = np.empty(total, dtype=PEAK_MZ_DTYPE)
        peak_int = np.empty(total, dtype=PEAK_INT_DTYPE)
        has_mob = n > 0 and getattr(scans[0], "mobility", None) is not None
        peak_mob = np.empty(total if has_mob else 0, dtype=PEAK_MOB_DTYPE)

        for i, s in enumerate(scans):
            off = int(offsets[i])
            ln = int(lengths[i])
            end = off + ln
            if ln:
                peak_mz[off:end] = s.mz
                peak_int[off:end] = s.intens
                if has_mob and getattr(s, "mobility", None) is not None:
                    peak_mob[off:end] = s.mobility
            # Adopt views; the per-scan arrays are released here.
            s.mz = peak_mz[off:end]
            s.intens = peak_int[off:end]
            if has_mob and getattr(s, "mobility", None) is not None:
                s.mobility = peak_mob[off:end]

        self._ms2_flat = (peak_mz, peak_int, peak_mob, offsets)
        return self._ms2_flat

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
        """Precompute the nearest-RT MS1 scan index for each MS2 scan.

        MS1 is one spectrum per frame (peaks carry per-peak mobility), so the
        pairing is purely nearest-RT; any IM-window restriction is applied later
        at query time using the MS2 band's (im_lo, im_hi). Vectorized over all
        MS2 scans at once (the old per-scan key-miss fallback was O(n_ms2*n_ms1)).
        """
        self._im_bin_to_ms1 = None
        n_ms1 = len(self.ms1scans)
        n_ms2 = len(self.ms2scans)
        ms1_nums = np.array([s.scan_num for s in self.ms1scans])

        if n_ms1 == 0 or n_ms2 == 0:
            self.ms2_to_ms1_map = np.zeros(n_ms2, dtype=int)
            self.ms2_to_ms1_scan_num = (
                ms1_nums[self.ms2_to_ms1_map] if n_ms1 > 0
                else np.zeros(n_ms2, dtype=int))
            return

        ms1_rts = np.array([s.RT for s in self.ms1scans])  # sorted ascending
        ms2_rts = np.array([s.RT for s in self.ms2scans])
        if n_ms1 == 1:
            nearest = np.zeros(n_ms2, dtype=int)
        else:
            pos = np.clip(np.searchsorted(ms1_rts, ms2_rts), 1, n_ms1 - 1)
            left_closer = np.abs(ms2_rts - ms1_rts[pos - 1]) <= np.abs(ms2_rts - ms1_rts[pos])
            nearest = np.where(left_closer, pos - 1, pos).astype(int)

        self.ms2_to_ms1_map = nearest
        self.ms2_to_ms1_scan_num = ms1_nums[nearest]  # parallel array for convenience

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

    def reband_ms1_to_ms2_bands(self, im_tol):
        """Re-band the full-range MS1 frames to mirror the MS2 watershed bands.

        For each distinct MS2 band ``(im_lo, im_hi)`` (taken from ``ms2scans``),
        draw peaks from that band's nearest-RT full-range MS1 frame that fall
        within ``[im_lo - im_tol, im_hi + im_tol]`` and emit a paired MS1 band
        spectrum carrying the *same* ``(im_lo, im_hi)`` key. Peaks are
        denormalized: a peak within ``im_tol`` of two overlapping bands appears
        in each band's MS1 spectrum. Downstream, ``im_bin_ms1`` keyed by
        ``(im_lo, im_hi)`` then resolves each MS2 band to its matching MS1 band.

        Must run after the preliminary search (so ``im_tol`` = ``config.opt_im_precision``
        is known) and before the main search. No-op if MS1 lacks per-peak mobility.
        """
        if len(self.ms1scans) == 0 or len(self.ms2scans) == 0:
            return
        if self.ms1scans[0].mobility is None:
            return  # non-IM data; nothing to band

        # nearest full-range MS1 frame per MS2 scan (RT-based)
        self.build_ms2_to_ms1_map()
        source_ms1 = self.ms1scans

        next_scan = max(max(self.ms1_by_id, default=0),
                        max(self.ms2_by_id, default=0)) + 1

        new_ms1 = []
        new_ms1_by_id = {}
        seen = {}  # (source_idx, im_lo, im_hi) -> None, dedup identical bands
        for ms2_idx, m2 in enumerate(self.ms2scans):
            im_lo = getattr(m2, "im_lo", None)
            if im_lo is None:
                continue
            im_hi = m2.im_hi
            src_idx = int(self.ms2_to_ms1_map[ms2_idx])
            ckey = (src_idx, im_lo, im_hi)
            if ckey in seen:
                continue
            seen[ckey] = None

            base = source_ms1[src_idx]
            if base.mobility is None or base.mz.size == 0:
                continue
            mask = (base.mobility >= im_lo - im_tol) & (base.mobility <= im_hi + im_tol)
            if not mask.any():
                continue  # no MS1 signal in this band; skip (rare)

            b = Spectrum()
            b.scan_num = next_scan
            b.id = f"scan={next_scan}"
            next_scan += 1
            b.level = 1
            b.RT = base.RT
            b.mz = base.mz[mask]            # base.mz is m/z-sorted; mask preserves order
            b.intens = base.intens[mask]
            b.mobility = base.mobility[mask]
            b.TIC = float(b.intens.sum(dtype=np.float64))
            b.injection_time = getattr(base, "injection_time", 1.0)
            b.collision_energy = None
            b.isolation_window = None
            b.im_lo = im_lo
            b.im_hi = im_hi
            b.scanwindow = [float(b.mz[0]), float(b.mz[-1])]

            new_ms1_by_id[b.scan_num] = len(new_ms1)
            new_ms1.append(b)

        if not new_ms1:
            return

        # replace MS1 frames with the banded spectra; refresh maps
        self.scan_pos = {k: v for k, v in self.scan_pos.items() if v[0] != 1}
        self.ms1scans = new_ms1
        self.ms1_by_id = new_ms1_by_id
        for scan_num, idx in new_ms1_by_id.items():
            self.scan_pos[scan_num] = [1, idx]
        self.build_ms2_to_ms1_map()
        logger.info(
            f"Re-banded MS1: {len(new_ms1)} band spectra "
            f"(im_tol={im_tol:.5f}) paired to MS2 watershed bands"
        )



