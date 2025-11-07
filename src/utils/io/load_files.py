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

    def __init__(self,scan=None):
        self.id = None
        self.level=None
        self.RT=None
        self.mz=None
        self.intens=None
        self.collision_energy = None
        self.TIC=None
        self.isolation_window = None

        if scan:
            self.get_vals(scan)

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
        proton_mass = 1.0072764
        rs = ps.core.Spectrum(id=self.id,
                              file_id=0, # placeholder
                              scan_start_time=self.RT,
                              mz_array=[mz - proton_mass for mz in self.mz],
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

    def __init__(self,mzml_file=None):

        self.filename = None
        self.ms2_to_ms1_map = None

        if mzml_file:
            self.load_spectra(mzml_file)

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
                
    
    def get_by_idx(self,idx):
        level, level_idx = self.scan_pos[idx]
        if level==1:
            return self.ms1scans[level_idx]
        elif level==2:
            return self.ms2scans[level_idx]

    def build_ms2_to_ms1_map(self):
        """Precompute nearest MS1 scan index for each MS2 scan."""
        ms1_rts = np.array([s.RT for s in self.ms1scans])
        ms1_nums = np.array([s.scan_num for s in self.ms1scans])
        ms2_rts = np.array([s.RT for s in self.ms2scans])

        ms2_to_ms1 = np.zeros(len(ms2_rts), dtype=int)
        for i, rt in enumerate(ms2_rts):
            pos = np.searchsorted(ms1_rts, rt)
            if pos == 0:
                closest_idx = 0
            elif pos == len(ms1_rts):
                closest_idx = len(ms1_rts) - 1
            else:
                before, after = ms1_rts[pos - 1], ms1_rts[pos]
                closest_idx = pos - 1 if abs(rt - before) < abs(rt - after) else pos
            ms2_to_ms1[i] = closest_idx

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


def loadSpectra(mzml_file):
    logger.info("Loading Spectra...")
    python_spec_file = mzml_file+"_pythonspec"
    if not os.path.exists(python_spec_file):
        logger.info("Loading Spectra... from file")
        spectra = SpectrumFile(mzml_file)
        with open(python_spec_file,"wb") as write_file:
            pickle.dump(spectra, write_file)
    else:
        with open(python_spec_file,"rb") as read_file:
            logger.info("Loading Spectra... from pickle")
            spectra = pickle.load(read_file)
            
    logger.info(f"Loaded {len(spectra.ms1scans)} MS1 spectra")
    logger.info(f"Loaded {len(spectra.ms2scans)} MS2 spectra")
    logger.info("finished")
    
    return spectra

