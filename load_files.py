import subprocess
import numpy as np
from pyteomics import mzml
import os
import matplotlib.pyplot as plt
import re
import pickle

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

        if scan:
            self.get_vals(scan)

    def get_vals(self,scan):
        # extract values from mzml spectrum
        self.id = scan["id"]
        self.scan_num = int(re.search("scan=(\d+)",self.id)[1])
        self.level=scan["ms level"]
        self.RT = scan['scanList']['scan'][0]["scan start time"]
        self.mz = scan["m/z array"]
        self.intens = scan["intensity array"]
        if self.level==2:
            self.collision_energy = scan["precursorList"]["precursor"][0]["activation"]["collision energy"]
            self.scanwindow = [scan["scanList"]["scan"][0]["scanWindowList"]["scanWindow"][0][i] for i in ["scan window lower limit","scan window upper limit"]]
            isolationWindow = scan["precursorList"]["precursor"][0]["isolationWindow"]
            self.prec_mz = isolationWindow["isolation window target m/z"]
            self.ms1window = isolationWindow["isolation window target m/z"]+np.array([-1,1])*[isolationWindow['isolation window lower offset'],isolationWindow['isolation window upper offset']]
        self.TIC = scan["total ion current"]

    def peak_list(self):
        return(np.array([self.mz,self.intens]))
    
    
class SpectrumFile:

    def __init__(self,mzml_file=None):

        self.filename = None
        
        if mzml_file:
            self.load_spectra(mzml_file)

    def load_spectra(self,mzml_file):
        self.filename = mzml_file
        
        # this may need to be optimised better
        self.ms1scans = []
        self.ms2scans = []
        with mzml.MzML(mzml_file) as reader:
            for scan in reader:
                if scan["ms level"] == 1:
                    self.ms1scans.append(Spectrum(scan))
                if scan["ms level"] == 2:
                    self.ms2scans.append(Spectrum(scan))
           
def loadSpectra(mzml_file):
    print("Loading Spectra",end=" ")
    python_spec_file = mzml_file+"_pythonspec"
    if not os.path.exists(python_spec_file):
        print("... from file")
        spectra = SpectrumFile(mzml_file)
        with open(python_spec_file,"wb") as write_file:
            pickle.dump(spectra, write_file)
    else:
        with open(python_spec_file,"rb") as read_file:
            print("... from pickle")
            spectra = pickle.load(read_file)
            
    print(f"Loaded {len(spectra.ms1scans)} MS1 spectra")
    print(f"Loaded {len(spectra.ms2scans)} MS2 spectra")
    print("finished")
    
    return spectra



