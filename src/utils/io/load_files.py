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
import pyarrow as pa

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


# Apache Arrow-based spectrum classes
class ArrowSpectrum(BaseSpectrum):
    """Arrow-native spectrum - works directly with Arrow table data"""
    
    def __init__(self, arrow_table, table_index):
        # Don't call super().__init__() to avoid property conflicts
        # Initialize parent attributes that aren't properties
        # (mz and intens are properties in this class)
        
        self.arrow_table = arrow_table
        self.table_index = table_index
        
        # Cache frequently accessed metadata
        self._metadata_cached = False
        self._scan_num = None
        self._level = None
        self._rt = None
        self._tic = None
        self._id = None
        
        # Cache arrays after first conversion to avoid repeated conversion
        self._mz_cached = None
        self._intens_cached = None
        self._arrays_cached = False
        
    def _cache_metadata(self):
        """Cache frequently accessed metadata on first access"""
        if not self._metadata_cached:
            self._scan_num = self.arrow_table['scanNumber'][self.table_index].as_py()
            self._level = self.arrow_table['msOrder'][self.table_index].as_py()
            self._rt = self.arrow_table['retentionTime'][self.table_index].as_py()
            self._tic = self.arrow_table['TIC'][self.table_index].as_py()
            self._id = self.arrow_table['scanHeader'][self.table_index].as_py() or ''
            self._metadata_cached = True
    
    @property
    def scan_num(self):
        self._cache_metadata()
        return self._scan_num
    
    @property
    def level(self):
        self._cache_metadata()
        return self._level
    
    @property
    def RT(self):
        self._cache_metadata()
        return self._rt
    
    @property
    def TIC(self):
        self._cache_metadata()
        return self._tic
    
    @property
    def id(self):
        self._cache_metadata()
        return self._id
    
    def _cache_arrays(self):
        """Cache array data after first conversion to avoid repeated conversion"""
        if not self._arrays_cached:
            # Convert Arrow arrays to numpy once and cache
            mz_arrow = self.arrow_table['mz_array'][self.table_index]
            if mz_arrow.is_valid:
                self._mz_cached = np.array(mz_arrow.as_py(), dtype=np.float64)
            else:
                self._mz_cached = np.array([], dtype=np.float64)
                
            intens_arrow = self.arrow_table['intensity_array'][self.table_index]
            if intens_arrow.is_valid:
                self._intens_cached = np.array(intens_arrow.as_py(), dtype=np.float64)
            else:
                self._intens_cached = np.array([], dtype=np.float64)
                
            self._arrays_cached = True
    
    @property
    def mz(self) -> np.ndarray:
        """Get m/z array - cached after first access"""
        self._cache_arrays()
        return self._mz_cached
    
    @property
    def intens(self) -> np.ndarray:
        """Get intensity array - cached after first access"""
        self._cache_arrays()
        return self._intens_cached
    
    @property
    def scanwindow(self) -> Optional[List[float]]:
        """Get scan window"""
        low_mz = self.arrow_table['lowMz'][self.table_index]
        high_mz = self.arrow_table['highMz'][self.table_index]
        if low_mz.is_valid and high_mz.is_valid:
            return [low_mz.as_py(), high_mz.as_py()]
        return None
    
    @property
    def prec_mz(self) -> Optional[float]:
        """Get precursor m/z for MS2 spectra"""
        if self.level == 2:
            center_mz = self.arrow_table['centerMz'][self.table_index]
            if center_mz.is_valid:
                return center_mz.as_py()
        return None
    
    @property
    def collision_energy(self) -> Optional[float]:
        """Get collision energy for MS2 spectra"""
        if self.level == 2:
            ce = self.arrow_table['collisionEnergyEvField'][self.table_index]
            if ce.is_valid:
                return ce.as_py()
        return None
    
    @property
    def ms1window(self) -> Optional[List[float]]:
        """Get MS1 isolation window for MS2 spectra"""
        if self.level == 2 and self.prec_mz is not None:
            isolation_width = self.arrow_table['isolationWidthMz'][self.table_index]
            if isolation_width.is_valid:
                half_width = isolation_width.as_py() / 2.0
                return [self.prec_mz - half_width, self.prec_mz + half_width]
        return None
    
    def get_vals(self, scan_data: Any) -> None:
        """Not needed for Arrow implementation - data accessed directly"""
        pass


class ArrowSpectrumFile(BaseSpectrumFile):
    """Arrow-native spectrum file reader - no conversion to individual objects"""
    
    def __init__(self, filename: Optional[str] = None):
        # Initialize parent attributes manually to avoid property conflicts
        self.filename = filename
        self.scan_pos = {}
        
        # Arrow-specific attributes
        self.arrow_table = None
        self.ms1_indices = []
        self.ms2_indices = []
        
        if filename is not None:
            self.load_spectra(filename)
    
    def load_spectra(self, filename: str) -> None:
        """Load spectra from Apache Arrow IPC file - create spectrum objects efficiently"""
        try:
            print(f"Loading Arrow file: {filename}")
            
            # Read Arrow IPC file
            with pa.ipc.open_file(filename) as reader:
                self.arrow_table = reader.read_all()
            
            print(f"Loaded {len(self.arrow_table)} spectra from Arrow file")
            
            # Build indices and create spectrum objects efficiently
            ms_levels = self.arrow_table['msOrder']
            scan_numbers = self.arrow_table['scanNumber']
            
            # Pre-allocate lists
            self._ms1scans = []
            self._ms2scans = []
            
            # Create spectrum objects in bulk - this is faster than on-demand creation
            for i in range(len(self.arrow_table)):
                scan_num = scan_numbers[i].as_py()
                level = ms_levels[i].as_py()
                
                # Create spectrum object once
                spec = ArrowSpectrum(self.arrow_table, i)
                
                if level == 1:
                    self._ms1scans.append(spec)
                    self.scan_pos[scan_num] = [1, len(self._ms1scans) - 1]
                elif level == 2:
                    self._ms2scans.append(spec)
                    self.scan_pos[scan_num] = [2, len(self._ms2scans) - 1]
            
            print(f"Created {len(self._ms1scans)} MS1 and {len(self._ms2scans)} MS2 spectrum objects")
                    
        except Exception as e:
            raise ValueError(f"Failed to load Arrow file: {e}")
    
    @property
    def ms1scans(self):
        """Return actual list of ArrowSpectrum objects"""
        return self._ms1scans
    
    @property 
    def ms2scans(self):
        """Return actual list of ArrowSpectrum objects"""
        return self._ms2scans
    
    def get_by_idx(self, scan_num: int) -> Optional['ArrowSpectrum']:
        """Get spectrum by scan number"""
        if scan_num in self.scan_pos:
            level, idx = self.scan_pos[scan_num]
            if level == 1 and idx < len(self._ms1scans):
                return self._ms1scans[idx]
            elif level == 2 and idx < len(self._ms2scans):
                return self._ms2scans[idx]
        return None


def loadArrowSpectra(arrow_file):
    """Load spectra from Apache Arrow IPC file with caching"""
    print("Loading Spectra", end=" ")
    python_spec_file = arrow_file + "_pythonspec"
    
    if not os.path.exists(python_spec_file):
        print("... from Arrow file")
        spectra = ArrowSpectrumFile(arrow_file)
        with open(python_spec_file, "wb") as write_file:
            pickle.dump(spectra, write_file)
    else:
        try:
            with open(python_spec_file, "rb") as read_file:
                print("... from pickle")
                spectra = pickle.load(read_file)
                
                # Check if this is an old pickle file missing Arrow attributes
                if not hasattr(spectra, 'arrow_table') or spectra.arrow_table is None:
                    print("... pickle file from older version, recreating from source")
                    os.remove(python_spec_file)
                    spectra = ArrowSpectrumFile(arrow_file)
                    with open(python_spec_file, "wb") as write_file:
                        pickle.dump(spectra, write_file)
                        
        except (AttributeError, ModuleNotFoundError, pickle.UnpicklingError):
            # Handle corrupted pickle files
            print("... pickle file corrupted, recreating from source")
            os.remove(python_spec_file)
            spectra = ArrowSpectrumFile(arrow_file)
            with open(python_spec_file, "wb") as write_file:
                pickle.dump(spectra, write_file)
    
    print(f"Loaded {len(spectra.ms1scans)} MS1 spectra")
    print(f"Loaded {len(spectra.ms2scans)} MS2 spectra")
    print("finished")
    
    return spectra


def loadSpectra(file_path):
    """Load spectra from mzML or Arrow files automatically based on file extension"""
    print("Loading Spectra", end=" ")
    
    # Determine file type based on extension
    if file_path.lower().endswith('.arrow'):
        return loadArrowSpectra(file_path)
    elif file_path.lower().endswith(('.mzml', '.mzML')):
        return loadMzMLSpectra(file_path)
    else:
        # Default to mzML for backwards compatibility
        print("... assuming mzML format")
        return loadMzMLSpectra(file_path)


def loadMzMLSpectra(mzml_file):
    """Load spectra using the MzMLSpectrumFile class"""
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

