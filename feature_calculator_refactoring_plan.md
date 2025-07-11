# Feature Calculator Refactoring Plan: Abstract Class Hierarchy

## Overview

This document outlines a plan to refactor the feature calculation system in JMod to use an abstract class hierarchy that separates concerns between RT alignment (`fit_to_lib`) and full spectral matching (`fit_to_lib2`).

## Current State Analysis

### fit_to_lib (RT Alignment)
- **Purpose**: Retention time alignment and calibration
- **Scope**: Targets only (filters out decoys)
- **Features**: Same 26 features but optimized for RT context
- **Implementation**: Direct calculation via `calculate_rt_alignment_features()`
- **Input**: Simpler data structure
- **Processing**: Streamlined pipeline focused on speed

### fit_to_lib2 (Main Spectral Fitting)
- **Purpose**: Primary peptide identification with FDR analysis
- **Scope**: Unified targets and decoys processing
- **Features**: Same 26 features but for comprehensive analysis
- **Implementation**: Modular approach via `FeatureCalculator` class
- **Input**: Complex `FeatureCalculatorInputs` dataclass
- **Processing**: Full unified architecture with validation

## Proposed Abstract Class Hierarchy

### 1. Abstract Base Classes

```python
from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

@runtime_checkable
class FeatureCalculatorInputsProtocol(Protocol):
    """Protocol defining the interface for feature calculator inputs."""
    sparse_matrix_csc: sparse.csc_matrix
    dia_spectrum: np.ndarray
    lib_coefficients: np.ndarray
    candidates: List[Any]
    peaks_in_dia: List[int]

class AbstractFeatureCalculatorInputs(ABC):
    """Abstract base class for feature calculator inputs."""
    
    @abstractmethod
    def validate(self) -> None:
        """Validate input data consistency."""
        pass
    
    @abstractmethod
    def get_n_candidates(self) -> int:
        """Get the number of candidates."""
        pass

class AbstractFeatureCalculator(ABC):
    """Abstract base class for feature calculators."""
    
    @abstractmethod
    def calculate_features(self, inputs: AbstractFeatureCalculatorInputs) -> np.ndarray:
        """Calculate all 26 features for spectral matching.
        
        Returns:
            Feature matrix of shape (n_candidates, 26)
        """
        pass
    
    @property
    @abstractmethod
    def feature_names(self) -> List[str]:
        """Get the names of calculated features."""
        pass
```

### 2. RT Alignment Implementation

```python
@dataclass
class RTAlignmentInputs(AbstractFeatureCalculatorInputs):
    """Simplified inputs for RT alignment feature calculation."""
    # Core required fields
    sparse_matrix_csc: sparse.csc_matrix
    dia_spectrum: np.ndarray
    lib_coefficients: np.ndarray
    candidates: List[Any]
    peaks_in_dia: List[int]
    
    # RT-specific fields
    prec_rt: float
    rt_mz: np.ndarray
    window_idxs: np.ndarray
    
    # Optional fields
    residuals: Optional[np.ndarray] = None
    y_pred: Optional[np.ndarray] = None
    
    def validate(self) -> None:
        """Validate RT alignment specific requirements."""
        assert len(self.candidates) == len(self.peaks_in_dia)
        assert self.sparse_matrix_csc.shape[1] == len(self.candidates)
    
    def get_n_candidates(self) -> int:
        return len(self.peaks_in_dia)

class RTAlignmentFeatureCalculator(AbstractFeatureCalculator):
    """Feature calculator optimized for RT alignment (targets only)."""
    
    def __init__(self):
        self._feature_names = FEATURE_NAMES
    
    @property
    def feature_names(self) -> List[str]:
        return self._feature_names
    
    def calculate_features(self, inputs: RTAlignmentInputs) -> np.ndarray:
        """Calculate features optimized for RT alignment."""
        # Simplified implementation focused on speed
        # Uses CSC operations throughout
        # No decoy handling needed
        pass
```

### 3. Full Spectral Matching Implementation

```python
# Keep existing FeatureCalculatorInputs but inherit from abstract base
@dataclass
class UnifiedFeatureCalculatorInputs(AbstractFeatureCalculatorInputs):
    """Complex inputs for full spectral matching (current implementation)."""
    # All existing fields from current FeatureCalculatorInputs
    # Plus unified target/decoy handling
    
    def validate(self) -> None:
        """Validate unified spectral matching requirements."""
        # Existing validation logic
        pass
    
    def get_n_candidates(self) -> int:
        return len(self.peaks_in_dia)

class UnifiedFeatureCalculator(AbstractFeatureCalculator):
    """Feature calculator for full spectral matching (targets + decoys)."""
    
    def __init__(self):
        self._feature_names = FEATURE_NAMES
    
    @property
    def feature_names(self) -> List[str]:
        return self._feature_names
    
    def calculate_features(self, inputs: UnifiedFeatureCalculatorInputs) -> np.ndarray:
        """Calculate features for unified spectral matching."""
        # Current implementation from FeatureCalculator.calculate_all_features
        # Handles unified target/decoy processing
        # Full modular feature calculation
        pass
```

## Implementation Plan

### Phase 1: Create Abstract Base Classes
1. **Create abstract base classes** in new file `src/features/abstract_calculator.py`
2. **Define protocols and interfaces** for type safety
3. **Establish common feature names** and validation patterns

### Phase 2: Implement RT Alignment Calculator
1. **Create `RTAlignmentInputs`** with simplified field set
2. **Implement `RTAlignmentFeatureCalculator`** 
3. **Extract and optimize** RT alignment specific logic from `calculate_rt_alignment_features`
4. **Focus on performance** - use CSC operations throughout

### Phase 3: Refactor Unified Calculator  
1. **Update existing `FeatureCalculatorInputs`** to inherit from abstract base
2. **Update existing `FeatureCalculator`** to inherit from abstract base
3. **Maintain backward compatibility** with current `fit_to_lib2` usage

### Phase 4: Integrate with Main Pipeline
1. **Update `fit_to_lib`** to use `RTAlignmentFeatureCalculator`
2. **Update `fit_to_lib2`** to use `UnifiedFeatureCalculator` 
3. **Add factory methods** for easy instantiation
4. **Update imports** throughout codebase

### Phase 5: Testing and Optimization
1. **Create comprehensive tests** for both implementations
2. **Performance benchmarking** to ensure RT alignment speed
3. **Regression testing** to ensure identical results
4. **Documentation updates**

## Benefits

### 1. Separation of Concerns
- **RT alignment**: Optimized for speed, targets only
- **Spectral matching**: Full feature set, unified processing

### 2. Type Safety
- **Protocol-based interfaces** for flexible typing
- **Abstract base classes** enforce implementation contracts
- **Clear separation** between input types

### 3. Performance Optimization
- **RT alignment**: Can be further optimized without affecting main pipeline
- **Specialized implementations** for different use cases
- **Reduced complexity** in RT alignment path

### 4. Maintainability
- **Clear inheritance hierarchy** shows relationships
- **Modular design** allows independent testing
- **Single responsibility principle** for each calculator

### 5. Future Extensibility
- **Easy to add new calculator types** (e.g., TimsTOF-specific)
- **Plugin architecture** for custom feature sets
- **Backward compatibility** maintained

## File Structure

```
src/features/
├── abstract_calculator.py          # Abstract base classes and protocols
├── rt_alignment_calculator.py      # RT alignment implementation
├── unified_calculator.py           # Full spectral matching (current logic)
├── feature_calculator.py           # Legacy interface (backward compatibility)
└── __init__.py                     # Public interface exports
```

## Migration Strategy

### Backward Compatibility
- **Keep existing interfaces** during transition period
- **Gradual migration** of calling code
- **Deprecation warnings** for old interfaces

### Testing Strategy
- **A/B testing** to ensure identical results
- **Performance benchmarks** for both implementations
- **Integration tests** with full pipeline

### Rollout Plan
1. **Phase 1-3**: Internal refactoring, no API changes
2. **Phase 4**: Update main pipeline integration
3. **Phase 5**: Testing and optimization
4. **Phase 6**: Remove legacy interfaces (future)

## Expected Outcomes

1. **Cleaner Architecture**: Clear separation between RT alignment and spectral matching
2. **Better Performance**: RT alignment optimized independently
3. **Improved Maintainability**: Modular design with clear responsibilities
4. **Type Safety**: Protocol-based interfaces prevent runtime errors
5. **Future Flexibility**: Easy to extend for new feature calculation needs

This refactoring aligns with software engineering best practices while maintaining the robust feature calculation system that JMod depends on.