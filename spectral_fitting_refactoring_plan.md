# Detailed Refactoring Plan for `spectral_fitting.py`

## Current Issues Identified

### 1. **Redundant Data Structures** ✅
- `ref_spec_values_split`, `sparse_lib_matrix`, and `ref_spec_values` contain overlapping information
- `ref_spec_values_split` + `ref_spec_row_indices_split` + `ref_spec_col_indices_split` essentially duplicate what's in `sparse_lib_matrix`
- Same issue with decoy variants (`decoy_spec_values_split`, etc.)

### 2. **Separate Target/Decoy Handling** ⚠️ 
- `fit_to_lib()` and `fit_to_lib_decoy()` are nearly identical (900+ lines of duplication)
- Risk of inconsistent treatment between targets and decoys
- Debug print statement in decoy function: `print("AAAAAAAAA")`

### 3. **Monolithic Functions** ⚠️
- `fit_to_lib()`: ~380 lines
- `fit_to_lib_decoy()`: ~400 lines  
- `get_features()`: ~160 lines
- Violates single responsibility principle

### 4. **Missing Type Hints & Documentation** ⚠️
- No type annotations on any functions
- Minimal docstrings
- Hard to understand data flow

### 5. **Code Quality Issues** ⚠️
- Magic numbers and unclear variable names
- Nested conditionals
- Repeated calculations

---

## Proposed Refactoring Plan

### Phase 1: Data Structure Consolidation

#### 1.1 Create Unified Spectrum Matrix Class
```python
@dataclass
class SpectrumMatrix:
    """Unified container for spectral library data"""
    values: np.ndarray
    row_indices: np.ndarray  
    col_indices: np.ndarray
    peptide_candidates: List[int]
    is_decoy: np.ndarray  # Boolean mask for decoy status
    
    def to_sparse_matrix(self) -> sparse.coo_matrix:
        """Convert to scipy sparse matrix for fitting"""
        
    def get_peptide_data(self, peptide_idx: int) -> 'PeptideSpectralData':
        """Extract data for specific peptide"""
```

#### 1.2 Eliminate Redundant Structures
- **Remove**: `ref_spec_values_split`, `ref_spec_col_indices_split`, `ref_spec_row_indices_split`
- **Remove**: `decoy_spec_values_split`, `decoy_spec_col_indices_split`, `decoy_spec_row_indices_split`  
- **Keep**: Single `SpectrumMatrix` with decoy flags
- **Keep**: `sparse_lib_matrix` for fitting operations

### Phase 2: Function Decomposition

#### 2.1 Core Fitting Functions
```python
def prepare_spectrum_data(
    dia_spec: BaseSpectrum,
    library: Dict,
    rt_mz: np.ndarray,
    window_idxs: np.ndarray,
    include_decoys: bool = True
) -> SpectrumMatrix:
    """Prepare unified spectrum matrix for target and decoy peptides"""

def fit_spectrum_coefficients(
    spectrum_matrix: SpectrumMatrix,
    observed_intensities: np.ndarray
) -> np.ndarray:
    """Perform sparse NNLS fitting"""

def calculate_spectral_features(
    spectrum_matrix: SpectrumMatrix,
    coefficients: np.ndarray,
    observed_spectrum: np.ndarray,
    precursor_info: PrecursorInfo
) -> SpectralFeatures:
    """Calculate all spectral similarity features"""
```

#### 2.2 Feature Calculation Modules
```python
def calculate_basic_features(
    spectrum_matrix: SpectrumMatrix,
    coefficients: np.ndarray
) -> BasicFeatures:
    """Number of peaks matched, fraction intensities, etc."""

def calculate_similarity_metrics(
    spectrum_matrix: SpectrumMatrix, 
    observed_intensities: np.ndarray,
    predicted_intensities: np.ndarray
) -> SimilarityMetrics:
    """SCRIBE, Manhattan distance, correlations, etc."""

def calculate_statistical_features(
    residuals: np.ndarray,
    gof_stats: np.ndarray
) -> StatisticalFeatures:
    """Goodness of fit, residual analysis"""
```

#### 2.3 Unified Main Function
```python
def fit_spectrum_to_library(
    dia_spec: BaseSpectrum,
    library: Dict,
    rt_mz: np.ndarray,
    all_keys: List,
    include_decoys: bool = True,
    **fitting_params
) -> SpectralFitResult:
    """
    Unified function replacing both fit_to_lib and fit_to_lib_decoy
    
    Args:
        dia_spec: DIA spectrum to fit
        library: Spectral library
        rt_mz: RT-m/z matrix for candidates
        all_keys: Library keys
        include_decoys: Whether to include decoy peptides
        **fitting_params: RT tolerance, m/z tolerance, etc.
        
    Returns:
        SpectralFitResult containing features for all peptides
    """
```

### Phase 3: Type System Enhancement

#### 3.1 Core Data Types
```python
from typing import NamedTuple, List, Dict, Optional, Tuple, Union
from dataclasses import dataclass

@dataclass
class PrecursorInfo:
    mz: float
    rt: float
    scan_num: int
    window_width: float

@dataclass  
class FittingParameters:
    rt_tol: float = 0.5
    mz_tol: float = 1e-5  
    ms1_tol: float = 1e-7
    top_n: int = 10
    atleast_m: int = 3

class SpectralFeatures(NamedTuple):
    basic: BasicFeatures
    similarity: SimilarityMetrics  
    statistical: StatisticalFeatures
    fragment_info: FragmentInfo

class SpectralFitResult(NamedTuple):
    target_features: List[SpectralFeatures]
    decoy_features: List[SpectralFeatures]
    coefficients: np.ndarray
    peptide_ids: List[int]
```

#### 3.2 Function Signatures
All functions will have complete type hints:
```python
def fit_spectrum_to_library(
    dia_spec: BaseSpectrum,
    library: Dict[int, Dict[str, Any]],
    rt_mz: np.ndarray,
    all_keys: List[int],
    dino_features: Optional[Any] = None,
    rt_filter: bool = False,
    ms1_mz: Optional[float] = None,
    ms1_spectra: Optional[List[BaseSpectrum]] = None,
    fitting_params: Optional[FittingParameters] = None
) -> SpectralFitResult:
```

### Phase 4: Documentation Enhancement

#### 4.1 Module-Level Documentation
```python
"""
Spectral Fitting Module

This module provides functionality for fitting DIA spectra against spectral libraries
using sparse non-negative least squares. Key features:

- Unified handling of target and decoy peptides
- Comprehensive spectral similarity metrics (SCRIBE, Manhattan distance, etc.)
- Statistical analysis of fit quality
- Fragment-level feature extraction

Main entry point: fit_spectrum_to_library()
"""
```

#### 4.2 Function Documentation
Each function will have comprehensive docstrings with:
- Purpose and algorithm description
- Parameter descriptions with types and constraints  
- Return value descriptions
- Usage examples
- Performance notes where relevant

### Phase 5: Performance Optimizations

#### 5.1 Memory Efficiency
- Single sparse matrix instead of multiple split arrays
- Lazy evaluation of expensive features
- Efficient boolean indexing for decoy separation

#### 5.2 Computational Efficiency  
- Vectorized operations where possible
- Avoid redundant calculations
- Cache expensive computations

---

## Implementation Strategy

### Step 1: Create New Types (Low Risk)
- Define data classes and type hints
- No changes to existing logic

### Step 2: Extract Utility Functions (Low Risk)  
- Move feature calculations to separate functions
- Maintain existing interfaces initially

### Step 3: Consolidate Data Structures (Medium Risk)
- Replace split arrays with unified `SpectrumMatrix`
- Update data preparation logic
- Requires careful testing

### Step 4: Unify Target/Decoy Handling (High Risk)
- Merge `fit_to_lib` and `fit_to_lib_decoy`
- Use boolean flags instead of separate code paths
- Critical to maintain exact same behavior

### Step 5: Integration & Testing (High Risk)
- Update calling code to use new interfaces
- Comprehensive regression testing
- Performance validation

---

## Identified Mistakes in Current Code

1. **Data Redundancy**: Storing same spectral data in 3+ different formats
2. **Debug Code**: `print("AAAAAAAAA")` in production function  
3. **Code Duplication**: 900+ lines duplicated between target/decoy functions
4. **Inconsistent Error Handling**: Different warning suppression strategies
5. **Magic Numbers**: Hardcoded values without named constants
6. **Missing Null Checks**: Potential for array index errors
7. **Inefficient Memory Usage**: Multiple copies of large arrays
8. **Poor Separation of Concerns**: Feature calculation mixed with data preparation

This refactoring will result in ~50% reduction in code size while improving maintainability, type safety, and reducing the risk of target/decoy inconsistencies.

## File Structure Impact

### Current Structure:
```
spectral_fitting.py (1588 lines)
├── hyperscore2()
├── get_features() (160 lines)
├── unmatched_peaks()
├── create_entries()
├── fit_to_lib2()
├── fit_to_lib() (380 lines)
└── fit_to_lib_decoy() (400 lines)
```

### Proposed Structure:
```
spectral_fitting/
├── __init__.py
├── types.py (Data classes and type definitions)
├── matrix_operations.py (SpectrumMatrix class)
├── feature_calculation.py (All feature calculation functions)
├── fitting_core.py (Core fitting algorithms)
└── main.py (fit_spectrum_to_library() entry point)
```

### Benefits:
- **Modularity**: Each file has a single responsibility
- **Testability**: Smaller functions are easier to unit test
- **Maintainability**: Changes to one feature type don't affect others
- **Type Safety**: Comprehensive type hints prevent runtime errors
- **Documentation**: Clear interfaces and usage patterns
- **Performance**: Elimination of redundant data structures and calculations