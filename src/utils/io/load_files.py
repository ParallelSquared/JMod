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

    def peak_list(self):
        return(np.array([self.mz,self.intens]))

    def to_rust_spectrum(self):
        rs = ps.core.Spectrum(id=self.id,
                              file_id=0, # placeholder
                              scan_start_time=self.RT,
                              mz_array=self.mz,
                              intensity_array=self.intens,
                              precursors=[
                                  ps.core.Precursor(self.prec_mz,
                                                    0, # will sweep whole window in WWA mode
                                                    (-1*self.isolation_window['isolation window lower offset'],
                                                    self.isolation_window['isolation window upper offset']) # will
                                                    )
                                  ],
                              total_ion_current=self.TIC
                              )
        return rs

    
class SpectrumFile:

    def __init__(self,mzml_file=None):

        self.filename = None
        
        if mzml_file:
            self.load_spectra(mzml_file)

    def load_spectra(self,mzml_file):
        self.filename = mzml_file
        
        # this may need to be optimised better
        self.scan_pos = {}
        self.ms1scans = []
        self.ms2scans = []
        with mzml.MzML(mzml_file) as reader:
            for scan in reader:
                if scan["ms level"] == 1:
                    spec = Spectrum(scan)
                    self.ms1scans.append(spec)
                    self.scan_pos[spec.scan_num] = [scan["ms level"],len(self.ms1scans)-1]
                if scan["ms level"] == 2:
                    spec = Spectrum(scan)
                    self.ms2scans.append(spec)
                    self.scan_pos[spec.scan_num] = [scan["ms level"],len(self.ms2scans)-1]
                
    
    def get_by_idx(self,idx):
        level, level_idx = self.scan_pos[idx]
        if level==1:
            return self.ms1scans[level_idx]
        elif level==2:
            return self.ms2scans[level_idx]
            
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

