# Spectrum Classes Refactoring Recommendations

## Overview

This document outlines recommendations for refactoring the `Spectrum` and `SpectrumFile` classes in `load_files.py` to improve maintainability, extensibility, and robustness. The current implementation has several limitations that prevent it from scaling to support multiple instrument types and file formats.

## Current State Analysis

### Spectrum Class Summary
The `Spectrum` class represents a single mass spectrum from an mzML file with these key attributes:
- **Basic metadata**: `id`, `level` (MS1/MS2), `RT` (retention time), `TIC` (total ion current)
- **Spectral data**: `mz` and `intens` arrays containing m/z and intensity values
- **MS2-specific**: `collision_energy`, `prec_mz` (precursor m/z), `ms1window` (isolation window)
- **Scan details**: `scan_num`, `injection_time`, `scanwindow`

### SpectrumFile Class Summary
The `SpectrumFile` class manages collections of spectra from mzML files:
- Loads and organizes MS1 and MS2 spectra into separate lists
- Maintains a `scan_pos` dictionary for quick lookup by scan number
- Provides `get_by_idx()` method for retrieval

## Critical Issues Identified

### 1. Lack of Abstraction and Extensibility
**Problem**: The current design is tightly coupled to mzML format and pyteomics library, making it difficult to support other instrument types (Bruker timsTOF, Thermo Orbitrap, etc.).

### 2. Poor Error Handling and Validation
**Problems**:
- No validation of input data
- Hard-coded assumptions about mzML structure
- Silent failures when expected fields are missing

### 3. Inconsistent Naming and API Design
**Problems**:
- Mixed naming conventions (`intens` vs `intensity`, `RT` vs `rt`)
- Inconsistent return types
- Poor encapsulation

### 4. Performance and Memory Issues
**Problems**:
- Loads all spectra into memory at once
- No lazy loading or indexing
- Inefficient scan lookup

### 5. Lack of Metadata and Standardization
**Problems**:
- No instrument metadata
- No standardized units
- Missing acquisition parameters

### 6. Missing Utility Methods
**Problems**:
- Limited spectrum manipulation capabilities
- No filtering or processing methods
- No format conversion support

## Recommended Refactoring

### 1. Abstract Base Class Architecture

Create a hierarchy of abstract base classes that can be extended for different instrument types:

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
from datetime import datetime
import numpy as np

class BaseSpectrum(ABC):
    """Abstract base class for all spectrum types"""
    
    def __init__(self):
        self.id: Optional[str] = None
        self.level: Optional[int] = None
        self._rt: Optional[float] = None
        self.mz: Optional[np.ndarray] = None
        self.intensity: Optional[np.ndarray] = None
        
    @property
    def retention_time(self) -> Optional[float]:
        """Retention time in minutes"""
        return self._rt
    
    @retention_time.setter 
    def retention_time(self, value: float) -> None:
        if value is not None and value < 0:
            raise ValueError("Retention time cannot be negative")
        self._rt = value
    
    @property
    def peaks(self) -> Optional[np.ndarray]:
        """Return peaks as Nx2 array [mz, intensity]"""
        if self.mz is None or self.intensity is None:
            return None
        return np.column_stack([self.mz, self.intensity])
        
    @abstractmethod
    def load_from_scan(self, scan_data: Any) -> None:
        """Load spectrum from vendor-specific scan data"""
        pass
        
    @abstractmethod
    def peak_list(self) -> np.ndarray:
        """Return standardized peak list"""
        pass
    
    def filter_by_mz_range(self, min_mz: float, max_mz: float) -> 'BaseSpectrum':
        """Filter peaks by m/z range"""
        if self.mz is None or self.intensity is None:
            return self
        
        mask = (self.mz >= min_mz) & (self.mz <= max_mz)
        filtered_spectrum = type(self)()
        filtered_spectrum.mz = self.mz[mask]
        filtered_spectrum.intensity = self.intensity[mask]
        # Copy other attributes
        for attr in ['id', 'level', '_rt']:
            if hasattr(self, attr):
                setattr(filtered_spectrum, attr, getattr(self, attr))
        return filtered_spectrum
    
    def normalize(self, method: str = "max") -> 'BaseSpectrum':
        """Normalize peak intensities"""
        if self.intensity is None:
            return self
        
        if method == "max":
            max_intensity = np.max(self.intensity)
            if max_intensity > 0:
                self.intensity = self.intensity / max_intensity
        elif method == "tic":
            total_intensity = np.sum(self.intensity)
            if total_intensity > 0:
                self.intensity = self.intensity / total_intensity
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        return self
    
    def to_dict(self) -> Dict[str, Any]:
        """Export spectrum as dictionary"""
        return {
            'id': self.id,
            'level': self.level,
            'retention_time': self.retention_time,
            'mz': self.mz.tolist() if self.mz is not None else None,
            'intensity': self.intensity.tolist() if self.intensity is not None else None
        }

@dataclass
class InstrumentMetadata:
    """Metadata about the mass spectrometer"""
    manufacturer: str
    model: str
    software_version: str
    acquisition_date: Optional[datetime] = None

@dataclass 
class AcquisitionParameters:
    """Parameters used during data acquisition"""
    ms1_resolution: Optional[int] = None
    ms2_resolution: Optional[int] = None
    mass_range: Optional[Tuple[float, float]] = None
    ion_source: Optional[str] = None
    fragmentation_method: Optional[str] = None

class BaseSpectrumFile(ABC):
    """Abstract base class for spectrum file readers"""
    
    def __init__(self, filename: Optional[str] = None):
        self.filename = filename
        self.ms1_spectra: List[BaseSpectrum] = []
        self.ms2_spectra: List[BaseSpectrum] = []
        self.metadata = InstrumentMetadata("Unknown", "Unknown", "Unknown")
        self.acquisition_params = AcquisitionParameters()
        
    @abstractmethod
    def load_spectra(self, filename: str) -> None:
        """Load spectra from file"""
        pass
        
    @abstractmethod
    def get_spectrum_by_id(self, spectrum_id: str) -> Optional[BaseSpectrum]:
        """Retrieve spectrum by identifier"""
        pass
    
    def get_spectra_by_level(self, level: int) -> List[BaseSpectrum]:
        """Get all spectra of a specific MS level"""
        if level == 1:
            return self.ms1_spectra
        elif level == 2:
            return self.ms2_spectra
        else:
            return [spec for spec in self.ms1_spectra + self.ms2_spectra 
                   if spec.level == level]
    
    def get_spectra_in_rt_range(self, min_rt: float, max_rt: float) -> List[BaseSpectrum]:
        """Get spectra within retention time range"""
        result = []
        for spec in self.ms1_spectra + self.ms2_spectra:
            if (spec.retention_time is not None and 
                min_rt <= spec.retention_time <= max_rt):
                result.append(spec)
        return result
```

### 2. Concrete Implementations for Different Instruments

```python
class MzMLSpectrum(BaseSpectrum):
    """mzML-specific spectrum implementation"""
    
    def __init__(self, scan=None):
        super().__init__()
        self.collision_energy: Optional[float] = None
        self.tic: Optional[float] = None
        self.injection_time: Optional[float] = None
        self.scan_num: Optional[int] = None
        self.scanwindow: Optional[List[float]] = None
        self.prec_mz: Optional[float] = None
        self.ms1window: Optional[np.ndarray] = None
        
        if scan:
            self.load_from_scan(scan)
    
    def load_from_scan(self, scan: Dict[str, Any]) -> None:
        """Load spectrum from mzML scan data with proper error handling"""
        try:
            self.id = scan["id"]
            self.scan_num = self._extract_scan_number(self.id)
            self.level = scan["ms level"]
            
            # Validate required fields
            if self.level not in [1, 2]:
                raise ValueError(f"Unsupported MS level: {self.level}")
            
            # Safe access with defaults
            scan_list = scan.get('scanList', {}).get('scan', [{}])
            if scan_list:
                scan_info = scan_list[0]
                self.retention_time = scan_info.get("scan start time")
                self.injection_time = scan_info.get("ion injection time", 0)
                
                # Scan window
                scan_window_list = scan_info.get("scanWindowList", {}).get("scanWindow", [])
                if scan_window_list:
                    window = scan_window_list[0]
                    self.scanwindow = [
                        window.get("scan window lower limit", 0),
                        window.get("scan window upper limit", 2000)
                    ]
            
            # Spectral data
            self.mz = scan.get("m/z array", np.array([]))
            self.intensity = scan.get("intensity array", np.array([]))
            self.tic = scan.get("total ion current")
            
            # MS2-specific data
            if self.level == 2:
                precursor_list = scan.get("precursorList", {}).get("precursor", [])
                if precursor_list:
                    precursor = precursor_list[0]
                    activation = precursor.get("activation", {})
                    self.collision_energy = activation.get("collision energy")
                    
                    isolation_window = precursor.get("isolationWindow", {})
                    self.prec_mz = isolation_window.get("isolation window target m/z")
                    
                    if self.prec_mz:
                        lower_offset = isolation_window.get('isolation window lower offset', 0)
                        upper_offset = isolation_window.get('isolation window upper offset', 0)
                        self.ms1window = self.prec_mz + np.array([-lower_offset, upper_offset])
                        
        except KeyError as e:
            raise ValueError(f"Missing required field in scan data: {e}")
        except Exception as e:
            raise ValueError(f"Error parsing scan data: {e}")
    
    def _extract_scan_number(self, scan_id: str) -> int:
        """Safely extract scan number from ID"""
        import re
        match = re.search(r"scan=(\d+)", scan_id)
        if not match:
            raise ValueError(f"Cannot extract scan number from ID: {scan_id}")
        return int(match.group(1))
    
    def peak_list(self) -> np.ndarray:
        """Return peaks as Nx2 array [mz, intensity]"""
        if self.peaks is not None:
            return self.peaks
        return np.array([]).reshape(0, 2)

class TimsTOFSpectrum(BaseSpectrum):
    """Bruker timsTOF-specific spectrum with ion mobility"""
    
    def __init__(self, scan=None):
        super().__init__()
        self.ion_mobility: Optional[float] = None
        self.frame_id: Optional[int] = None
        self.mobility_range: Optional[Tuple[float, float]] = None
        
        if scan:
            self.load_from_scan(scan)
    
    def load_from_scan(self, scan_data: Any) -> None:
        """Load from Bruker-specific scan data"""
        # Implementation specific to Bruker data format
        pass
    
    def peak_list(self) -> np.ndarray:
        """Return peaks as Nx3 array [mz, intensity, mobility]"""
        if self.mz is None or self.intensity is None:
            return np.array([]).reshape(0, 3)
        
        mobility_col = np.full(len(self.mz), self.ion_mobility or 0)
        return np.column_stack([self.mz, self.intensity, mobility_col])

class OrbitrapSpectrum(BaseSpectrum):
    """Thermo Orbitrap-specific spectrum"""
    
    def __init__(self, scan=None):
        super().__init__()
        self.resolution: Optional[int] = None
        self.agc_target: Optional[float] = None
        self.max_injection_time: Optional[float] = None
        self.activation_type: Optional[str] = None
        
        if scan:
            self.load_from_scan(scan)
    
    def load_from_scan(self, scan_data: Any) -> None:
        """Load from Thermo-specific scan data"""
        # Implementation specific to Thermo data format
        pass
    
    def peak_list(self) -> np.ndarray:
        """Return standardized peak list"""
        return super().peaks if super().peaks is not None else np.array([]).reshape(0, 2)
```

### 3. Improved SpectrumFile Implementations

```python
class MzMLSpectrumFile(BaseSpectrumFile):
    """Improved mzML file reader with better error handling"""
    
    def __init__(self, filename: Optional[str] = None):
        super().__init__(filename)
        self.scan_pos: Dict[int, Tuple[int, int]] = {}  # scan_num -> (level, index)
        
        if filename:
            self.load_spectra(filename)
    
    def load_spectra(self, filename: str) -> None:
        """Load spectra with comprehensive error handling"""
        try:
            from pyteomics import mzml
        except ImportError:
            raise ImportError("pyteomics is required for mzML file support")
        
        self.filename = filename
        self.ms1_spectra.clear()
        self.ms2_spectra.clear()
        self.scan_pos.clear()
        
        try:
            with mzml.MzML(filename) as reader:
                for scan in reader:
                    try:
                        spectrum = MzMLSpectrum(scan)
                        
                        if spectrum.level == 1:
                            self.ms1_spectra.append(spectrum)
                            self.scan_pos[spectrum.scan_num] = (1, len(self.ms1_spectra) - 1)
                        elif spectrum.level == 2:
                            self.ms2_spectra.append(spectrum)
                            self.scan_pos[spectrum.scan_num] = (2, len(self.ms2_spectra) - 1)
                        else:
                            print(f"Warning: Unsupported MS level {spectrum.level} for scan {spectrum.scan_num}")
                            
                    except Exception as e:
                        print(f"Warning: Failed to parse scan {scan.get('id', 'unknown')}: {e}")
                        continue
                        
        except Exception as e:
            raise ValueError(f"Failed to load mzML file {filename}: {e}")
    
    def get_spectrum_by_id(self, spectrum_id: str) -> Optional[BaseSpectrum]:
        """Retrieve spectrum by scan number or ID"""
        # Try to extract scan number from ID
        try:
            import re
            match = re.search(r"scan=(\d+)", spectrum_id)
            if match:
                scan_num = int(match.group(1))
                return self.get_by_scan_num(scan_num)
        except:
            pass
        
        # Search by ID in all spectra
        for spec in self.ms1_spectra + self.ms2_spectra:
            if spec.id == spectrum_id:
                return spec
        
        return None
    
    def get_by_scan_num(self, scan_num: int) -> Optional[BaseSpectrum]:
        """Get spectrum by scan number"""
        if scan_num not in self.scan_pos:
            return None
        
        level, index = self.scan_pos[scan_num]
        if level == 1:
            return self.ms1_spectra[index]
        elif level == 2:
            return self.ms2_spectra[index]
        
        return None

class LazySpectrumFile(BaseSpectrumFile):
    """Memory-efficient spectrum file with lazy loading"""
    
    def __init__(self, filename: str):
        super().__init__(filename)
        self._scan_index: Dict[int, int] = {}  # scan_num -> file_position
        self._file_handle = None
        self._build_index()
    
    def _build_index(self) -> None:
        """Build index of scan positions without loading spectra"""
        # Implementation depends on file format
        # This would scan the file to build an index of spectrum locations
        # without actually loading the spectral data into memory
        pass
    
    def get_spectrum(self, scan_num: int) -> Optional[BaseSpectrum]:
        """Load spectrum on-demand"""
        if scan_num not in self._scan_index:
            return None
        
        # Load and return spectrum from file position
        # Implementation would seek to the position and load just that spectrum
        pass
    
    def __del__(self):
        """Clean up file handle"""
        if self._file_handle:
            self._file_handle.close()
```

### 4. Utility Functions

```python
def load_spectrum_file(filename: str, lazy: bool = False) -> BaseSpectrumFile:
    """Factory function to load appropriate spectrum file type"""
    import os
    
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")
    
    ext = os.path.splitext(filename)[1].lower()
    
    if ext == '.mzml':
        if lazy:
            return LazySpectrumFile(filename)
        else:
            return MzMLSpectrumFile(filename)
    elif ext == '.d':  # Bruker
        # return TimsTOFSpectrumFile(filename)
        raise NotImplementedError("Bruker .d files not yet supported")
    elif ext == '.raw':  # Thermo
        # return ThermoRawFile(filename)
        raise NotImplementedError("Thermo .raw files not yet supported")
    else:
        raise ValueError(f"Unsupported file format: {ext}")

def convert_spectrum_file(input_file: str, output_file: str, output_format: str) -> None:
    """Convert between different spectrum file formats"""
    spectrum_file = load_spectrum_file(input_file)
    
    if output_format.lower() == 'mzml':
        # Export to mzML format
        pass
    elif output_format.lower() == 'mgf':
        # Export to MGF format
        pass
    else:
        raise ValueError(f"Unsupported output format: {output_format}")
```

## Recommendations for gen_isotopes_dict

The `gen_isotopes_dict` function in `iso_functions.py` appears well-designed for isotope pattern calculation. However, consider these improvements:

### 1. Better Integration
Make it a method of spectrum/fragment classes:

```python
class BaseSpectrum(ABC):
    def generate_isotope_pattern(self, sequence: str, fragments: Dict[str, List[float]], 
                               tag=None) -> Tuple[np.ndarray, np.ndarray]:
        """Generate isotope patterns for fragments"""
        from ...utils.iso_functions import gen_isotopes_dict
        return gen_isotopes_dict(sequence, fragments, tag)
```

### 2. Caching
Cache isotope patterns for common fragments to improve performance:

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def _cached_isotope_pattern(sequence: str, fragment_key: str, tag_name: str = None):
    """Cache isotope patterns for common fragments"""
    # Implementation here
    pass
```

### 3. Enhanced Validation
Add input validation for sequences and fragments:

```python
def gen_isotopes_dict(seq: str, frags: Dict[str, List[float]], tag=None) -> Tuple[np.ndarray, np.ndarray]:
    """Generate isotope patterns with proper validation"""
    
    # Validate inputs
    if not seq or not isinstance(seq, str):
        raise ValueError("Sequence must be a non-empty string")
    
    if not frags or not isinstance(frags, dict):
        raise ValueError("Fragments must be a non-empty dictionary")
    
    for frag_name, frag_data in frags.items():
        if not isinstance(frag_data, list) or len(frag_data) != 2:
            raise ValueError(f"Fragment {frag_name} must have [mz, intensity] format")
    
    # Existing implementation...
```

## Implementation Priority

1. **Phase 1**: Create abstract base classes and refactor existing mzML implementation
2. **Phase 2**: Add proper error handling and validation
3. **Phase 3**: Implement instrument-specific classes (TimsTOF, Orbitrap)
4. **Phase 4**: Add lazy loading and performance optimizations
5. **Phase 5**: Implement format conversion utilities

## Benefits of This Refactoring

1. **Extensibility**: Easy to add support for new instrument types
2. **Maintainability**: Clear separation of concerns and consistent APIs
3. **Robustness**: Proper error handling and validation
4. **Performance**: Lazy loading and memory optimization options
5. **Usability**: Consistent naming conventions and utility methods
6. **Testing**: Each component can be unit tested independently

This refactoring would significantly improve the codebase's ability to handle diverse mass spectrometry data formats while maintaining backwards compatibility with existing code.