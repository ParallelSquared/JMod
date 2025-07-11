# Refactoring Plan: Unifying fit_to_lib and fit_to_lib2

## Overview

This document outlines a careful, phased approach to refactor `fit_to_lib` and `fit_to_lib2` into a unified implementation that maintains exact backward compatibility while improving code maintainability.

## Current State Analysis

### fit_to_lib (RT Alignment)
- **Purpose**: Used exclusively for retention time alignment during initial processing
- **Key Behavior**: Filters out decoy peptides before processing
- **Output**: Simplified output with fewer features
- **Critical Code**: Line 1501 filters decoys: `[key for key in mass_window_candidates_all if not library[key].get('is_decoy', False)]`

### fit_to_lib2 (Main Spectral Fitting)
- **Purpose**: Primary spectral matching for FDR analysis
- **Key Behavior**: Processes targets and decoys together
- **Output**: Full feature set for scoring
- **Uses**: Unified data structures (UnifiedCandidates, UnifiedFeatures, etc.)

### Already Extracted Functions
1. `preprocess_dia_spectrum` - Merges peaks within tolerance ✓
2. `filter_candidates_by_window` - Filters by mass/RT window ✓
3. `separate_library_candidates` - Separates targets/decoys
4. `create_empty_output_row` - Creates default output row ✓
5. `extract_non_zero_coefficients` - Extracts non-zero NNLS results ✓
6. `format_fragment_information` - Formats fragment data
7. `get_protein_info` - Extracts protein information
8. `format_spectral_fitting_output` - Formats final output

## Refactoring Phases

### Phase 1: Extract MS1 Peak Filtering Logic ✅ COMPLETE
**Goal**: Extract the MS1 peak checking logic from fit_to_lib

**Current Code** (lines 1541-1542):
```python
ms1_peak = ~np.isnan([closest_peak_diff(mz,ms1_spec.mz) for mz in rt_mz[window_idxs,1]])
```

**New Function**:
```python
def check_ms1_peaks(
    rt_mz: np.ndarray,
    window_idxs: np.ndarray,
    ms1_spec: Any
) -> np.ndarray:
    """Check which candidates have matching MS1 peaks."""
```

**Validation**: 
- Compare `ms1_peak` arrays before/after
- Run RT alignment test cases

### Phase 2: Extract Peak Matching Logic ✅ COMPLETE
**Goal**: Extract the complex peak matching and filtering logic

**Current Code** (lines 1538-1550):
```python
ref_coords = [np.searchsorted(centroid_breaks,M[:,0]) for M in candidate_peaks]
top_ten = [np.searchsorted(centroid_breaks,M[np.argsort(-M[:,1])[0:min(top_n,M.shape[0])],0]) for M in candidate_peaks]
# ... filtering logic ...
```

**New Function**:
```python
def filter_candidates_by_peak_matching(
    candidate_peaks: List[np.ndarray],
    centroid_breaks: np.ndarray,
    ms1_peak: np.ndarray,
    top_n: int,
    atleast_m: int,
    frac_matched: float
) -> Tuple[List[int], List[np.ndarray], List[np.ndarray]]:
    """Filter candidates based on peak matching criteria."""
```

**Validation**:
- Compare filtered candidate lists
- Ensure same number of matches

### Phase 3: Extract Matrix Building Logic ✅ COMPLETE
**Goal**: Extract sparse matrix construction (similar between both functions)

**Current Code** (lines 1580-1621):
```python
# Matrix construction logic
```

**New Function**:
```python
def build_sparse_matrix_simple(
    ref_spec_row_indices_split: List[np.ndarray],
    ref_spec_col_indices_split: List[np.ndarray],
    ref_spec_values_split: List[np.ndarray],
    norm_intensities: List[np.ndarray],
    ref_pep_cand_loc: List[np.ndarray],
    unique_row_idxs: List[int]
) -> Tuple[sparse.coo_matrix, np.ndarray]:
    """Build sparse matrix for NNLS - simplified version for RT alignment."""
```

**Validation**:
- Compare matrix dimensions and values
- Verify NNLS results are identical

### Phase 4: Extract Feature Calculation ✅ COMPLETE
**Goal**: Extract the feature calculation logic specific to fit_to_lib

**New Function**:
```python
def calculate_rt_alignment_features(
    # parameters...
) -> np.ndarray:
    """Calculate features for RT alignment (simplified feature set)."""
```

**Validation**:
- Compare feature arrays element by element
- Run RT alignment pipeline end-to-end

### Phase 5: Create Unified Function
**Goal**: Create a single function that handles both use cases

**Approach**:
```python
def fit_to_lib_unified(
    dia_spec,
    library,
    rt_mz,
    all_keys,
    # ... other params ...
    mode: str = "full"  # "full" or "rt_alignment"
):
    """Unified spectral fitting function."""
    if mode == "rt_alignment":
        # RT alignment specific behavior (filter decoys, simplified features)
    else:
        # Full spectral fitting behavior
```

**Validation**:
- A/B test against original functions
- Full regression test suite

## Testing Strategy

### Unit Tests for Each Extracted Function
1. Test with minimal synthetic data
2. Test edge cases (empty arrays, single element, etc.)
3. Test with real data samples

### Integration Tests
1. **RT Alignment Test**: 
   - Run RT alignment with original `fit_to_lib`
   - Run with refactored version
   - Compare outputs element by element

2. **Spectral Fitting Test**:
   - Run full pipeline with original `fit_to_lib2`
   - Run with refactored version
   - Compare FDR results

### Regression Tests
1. Save outputs from current implementation
2. After each phase, verify outputs match exactly
3. Use `np.allclose` for floating point comparisons

## Validation Approach

### Phase Validation
After each phase:
1. Run all existing tests
2. Compare function outputs with saved baselines
3. Run integration test on real data
4. Commit only after validation passes

### Final Validation
1. Run complete JMod pipeline on test dataset
2. Compare:
   - Number of identifications
   - FDR calculations
   - RT alignment quality
   - Final output files

## Risk Mitigation

1. **Small Incremental Changes**: Each phase extracts one logical unit
2. **Comprehensive Testing**: Test each extraction before proceeding
3. **Easy Rollback**: Git commits after each successful phase
4. **Parallel Implementation**: Keep original functions until fully validated
5. **Output Comparison**: Save outputs from original for regression testing

## Success Criteria

1. All existing tests pass
2. RT alignment produces identical results
3. Spectral fitting produces identical results
4. Code coverage maintained or improved
5. Performance not degraded (timing tests)

## Timeline Estimate

- Phase 1: 2 hours (MS1 peak filtering)
- Phase 2: 3 hours (Peak matching logic)
- Phase 3: 3 hours (Matrix building)
- Phase 4: 2 hours (Feature calculation)
- Phase 5: 4 hours (Unified function)
- Validation: 2 hours
- **Total**: ~16 hours of careful implementation

## Next Steps

1. Create baseline test data from current implementation
2. Start with Phase 1 (MS1 peak filtering)
3. Validate each step thoroughly
4. Document any behavioral discoveries