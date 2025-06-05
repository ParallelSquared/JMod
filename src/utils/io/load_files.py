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
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

# Abstract base classes
class BaseSpectrum(ABC):
    """Abstract base class for all spectrum types - maintains current interface"""
    
    def __init__(self):
        # Keep original attribute names for compatibility
        self.id: Optional[str] = None
        self.level: Optional[int] = None
        self.RT: Optional[float] = None
        self.mz: Optional[np.ndarray] = None
        self.intens: Optional[np.ndarray] = None  # Keep original naming
        self.TIC: Optional[float] = None
        
    @abstractmethod
    def get_vals(self, scan_data: Any) -> None:
        """Extract values from scan data - each format implements differently"""
        pass
        
    def peak_list(self) -> np.ndarray:
        """Return peak list as in current implementation"""
        if self.mz is not None and self.intens is not None:
            return np.array([self.mz, self.intens])
        return np.array([]).reshape(2, 0)  # Empty array with correct shape

class BaseSpectrumFile(ABC):
    """Abstract base class for spectrum file readers - maintains current interface"""
    
    def __init__(self, filename: Optional[str] = None):
        # Keep original attribute names for compatibility
        self.filename: Optional[str] = filename
        self.scan_pos: Dict[int, List] = {}
        self.ms1scans: List[BaseSpectrum] = []
        self.ms2scans: List[BaseSpectrum] = []
        
    @abstractmethod
    def load_spectra(self, filename: str) -> None:
        """Load spectra from file"""
        pass
        
    def get_by_idx(self, idx: int) -> Optional[BaseSpectrum]:
        """Keep the existing interface"""
        if idx not in self.scan_pos:
            return None
        level, level_idx = self.scan_pos[idx]
        if level == 1:
            return self.ms1scans[level_idx]
        elif level == 2:
            return self.ms2scans[level_idx]
        return None

# Concrete implementations
class MzMLSpectrum(BaseSpectrum):
    """mzML-specific spectrum - enhanced version of current Spectrum class"""
    
    def __init__(self, scan=None):
        super().__init__()
        # Add mzML-specific attributes
        self.scan_num: Optional[int] = None
        self.collision_energy: Optional[float] = None
        self.injection_time: Optional[float] = None
        self.scanwindow: Optional[List[float]] = None
        self.prec_mz: Optional[float] = None
        self.ms1window: Optional[np.ndarray] = None
        
        if scan:
            self.get_vals(scan)
    
    def get_vals(self, scan: Dict[str, Any]) -> None:
        """Enhanced version of current get_vals with better error handling"""
        try:
            # Core data (same as current implementation)
            self.id = scan["id"]
            self.scan_num = int(re.search(r"scan=(\d+)", self.id)[1])
            self.level = scan["ms level"]
            self.RT = scan['scanList']['scan'][0]["scan start time"]
            self.injection_time = scan["scanList"]["scan"][0]["ion injection time"] / 1000
            self.mz = scan["m/z array"]
            self.intens = scan["intensity array"]
            self.TIC = scan["total ion current"]
            
            # Scan window
            self.scanwindow = [
                scan["scanList"]["scan"][0]["scanWindowList"]["scanWindow"][0][i] 
                for i in ["scan window lower limit", "scan window upper limit"]
            ]
            
            # MS2-specific data
            if self.level == 2:
                self.collision_energy = scan["precursorList"]["precursor"][0]["activation"]["collision energy"]
                isolationWindow = scan["precursorList"]["precursor"][0]["isolationWindow"]
                self.prec_mz = isolationWindow["isolation window target m/z"]
                self.ms1window = self.prec_mz + np.array([-1, 1]) * [
                    isolationWindow['isolation window lower offset'],
                    isolationWindow['isolation window upper offset']
                ]
                
        except KeyError as e:
            raise ValueError(f"Missing required field in mzML scan: {e}")
        except Exception as e:
            raise ValueError(f"Error parsing mzML scan: {e}")

class MzMLSpectrumFile(BaseSpectrumFile):
    """Enhanced version of current SpectrumFile class"""
    
    def __init__(self, mzml_file: Optional[str] = None):
        super().__init__(mzml_file)
        if mzml_file:
            self.load_spectra(mzml_file)
    
    def load_spectra(self, mzml_file: str) -> None:
        """Enhanced version of current load_spectra with error handling"""
        self.filename = mzml_file
        self.scan_pos.clear()
        self.ms1scans.clear()
        self.ms2scans.clear()
        
        try:
            with mzml.MzML(mzml_file) as reader:
                for scan in reader:
                    try:
                        if scan["ms level"] == 1:
                            spec = MzMLSpectrum(scan)
                            self.ms1scans.append(spec)
                            self.scan_pos[spec.scan_num] = [1, len(self.ms1scans) - 1]
                        elif scan["ms level"] == 2:
                            spec = MzMLSpectrum(scan)
                            self.ms2scans.append(spec)
                            self.scan_pos[spec.scan_num] = [2, len(self.ms2scans) - 1]
                    except Exception as e:
                        print(f"Warning: Skipped scan {scan.get('id', 'unknown')}: {e}")
                        continue
        except Exception as e:
            raise ValueError(f"Failed to load mzML file: {e}")

            
def loadSpectra(mzml_file):
    """Load spectra using the new MzMLSpectrumFile class"""
    print("Loading Spectra",end=" ")
    python_spec_file = mzml_file+"_pythonspec"
    if not os.path.exists(python_spec_file):
        print("... from file")
        spectra = MzMLSpectrumFile(mzml_file)
        with open(python_spec_file,"wb") as write_file:
            pickle.dump(spectra, write_file)
    else:
        try:
            with open(python_spec_file,"rb") as read_file:
                print("... from pickle")
                spectra = pickle.load(read_file)
        except (AttributeError, ModuleNotFoundError, pickle.UnpicklingError) as e:
            # Handle legacy pickle files that reference old class names or are corrupted
            if any(name in str(e) for name in ["SpectrumFile", "Spectrum", "load_files"]) or "UnpicklingError" in str(type(e)):
                print("... pickle file from older version, recreating from source")
                os.remove(python_spec_file)
                spectra = MzMLSpectrumFile(mzml_file)
                with open(python_spec_file,"wb") as write_file:
                    pickle.dump(spectra, write_file)
            else:
                raise e
            
    print(f"Loaded {len(spectra.ms1scans)} MS1 spectra")
    print(f"Loaded {len(spectra.ms2scans)} MS2 spectra")
    print("finished")
    
    return spectra

