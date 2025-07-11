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

Create a simple hierarchy of abstract base classes that maintains compatibility with the existing implementation:

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np
import re

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
        self.scan_pos: Dict[int, tuple] = {}
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
```

### 2. Concrete Implementations for Different Instruments

```python
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

class TimsTOFSpectrum(BaseSpectrum):
    """Bruker timsTOF-specific spectrum with ion mobility"""
    
    def __init__(self, scan=None):
        super().__init__()
        self.ion_mobility: Optional[float] = None
        self.frame_id: Optional[int] = None
        self.scan_num: Optional[int] = None
        
        if scan:
            self.get_vals(scan)
    
    def get_vals(self, scan_data: Any) -> None:
        """Load from Bruker-specific scan data"""
        # Implementation specific to Bruker data format
        # Would extract Bruker-specific fields like ion mobility
        pass

class OrbitrapSpectrum(BaseSpectrum):
    """Thermo Orbitrap-specific spectrum"""
    
    def __init__(self, scan=None):
        super().__init__()
        self.resolution: Optional[int] = None
        self.agc_target: Optional[float] = None
        self.max_injection_time: Optional[float] = None
        self.activation_type: Optional[str] = None
        self.scan_num: Optional[int] = None
        
        if scan:
            self.get_vals(scan)
    
    def get_vals(self, scan_data: Any) -> None:
        """Load from Thermo-specific scan data"""
        # Implementation specific to Thermo data format
        pass
```

### 3. Improved SpectrumFile Implementations

```python
class MzMLSpectrumFile(BaseSpectrumFile):
    """Enhanced version of current SpectrumFile class"""
    
    def __init__(self, mzml_file: Optional[str] = None):
        super().__init__(mzml_file)
        if mzml_file:
            self.load_spectra(mzml_file)
    
    def load_spectra(self, mzml_file: str) -> None:
        """Enhanced version of current load_spectra with error handling"""
        from pyteomics import mzml
        
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

class LazySpectrumFile(BaseSpectrumFile):
    """Memory-efficient spectrum file with lazy loading (future enhancement)"""
    
    def __init__(self, filename: str):
        super().__init__(filename)
        self._scan_index: Dict[int, int] = {}  # scan_num -> file_position
        # Implementation would build index without loading all spectra
    
    def load_spectra(self, filename: str) -> None:
        """Build index without loading all spectra into memory"""
        # Implementation would scan file and build index of positions
        pass
    
    def get_spectrum_on_demand(self, scan_num: int) -> Optional[BaseSpectrum]:
        """Load spectrum only when requested"""
        # Implementation would load spectrum from file position
        pass
```

### 4. Simple Factory Function

```python
def load_spectrum_file(filename: str) -> BaseSpectrumFile:
    """Factory function to load appropriate spectrum file type"""
    import os
    
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")
    
    ext = os.path.splitext(filename)[1].lower()
    
    if ext == '.mzml':
        return MzMLSpectrumFile(filename)
    elif ext == '.d':  # Bruker
        # return TimsTOFSpectrumFile(filename)  # Future implementation
        raise NotImplementedError("Bruker .d files not yet supported")
    elif ext == '.raw':  # Thermo
        # return ThermoRawFile(filename)  # Future implementation
        raise NotImplementedError("Thermo .raw files not yet supported")
    else:
        raise ValueError(f"Unsupported file format: {ext}")

# Backwards compatibility function
def loadSpectra(mzml_file: str) -> MzMLSpectrumFile:
    """Backwards compatible version of current loadSpectra function"""
    return MzMLSpectrumFile(mzml_file)
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

1. **Backwards Compatibility**: Maintains existing interface (RT, intens, get_vals, etc.)
2. **Extensibility**: Easy to add support for new instrument types 
3. **Robustness**: Better error handling without changing core functionality
4. **Maintainability**: Clear inheritance structure while keeping it simple
5. **Minimal Disruption**: Current code continues to work with minimal changes
6. **Future-Ready**: Foundation for adding Bruker, Thermo support later

## Implementation Strategy

This refactoring approach:
- **Keeps it simple** - Only adds essential abstractions
- **Maintains compatibility** - Existing code continues to work
- **Enables growth** - Easy to extend for new instruments
- **Improves robustness** - Better error handling and validation
- **Stays focused** - Doesn't add unnecessary complexity

The key is to enhance what exists rather than completely rewrite it.