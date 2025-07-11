# Spectral Fitting Refactoring Summary

## Overview
This document captures all the work done to refactor the monolithic `spectral_fitting.py` file (1588 lines) into a modular structure. The refactoring was ultimately reverted due to too many integration issues, but this document preserves the learnings and changes made.

## Timeline of Changes

### Initial Refactoring (Commits 4b903eb through 0b0e38e)

#### Step 1: Create modular structure (4b903eb)
- Created new directory structure: `src/spectral_fitting/`
- Added files:
  - `__init__.py` - Public API exports
  - `types.py` - Type definitions and data structures
  - `spectrum_processing.py` - DIA spectrum processing  
  - `candidate_filtering.py` - Library candidate selection
  - `matrix_operations.py` - Sparse matrix and NNLS
  - `feature_calculation.py` - Feature extraction
  - `fitting_core.py` - Main orchestration
  - `adapter.py` - Backward compatibility layer

#### Step 2: Extract spectrum processing (0926935)
- Moved peak merging and binning logic
- ~100 lines extracted to focused functions

#### Step 3: Extract candidate filtering (a7e9c2f)
- Moved mass window and RT filtering
- ~50 lines into dedicated module

#### Step 4: Extract matrix operations (7ed1100)
- Sparse matrix construction
- NNLS solving
- ~150 lines modularized

#### Step 5: Extract feature calculation (0b0e38e)
- 156 lines → 6 focused functions
- Similarity, intensity, error, hyperscore features

### Phase 6-7: Integration and Bug Fixes (Commits 3c1cd31 through 56fb262)

#### Major Issues Encountered and Fixed:

1. **Function Signature Mismatch** (25b6d58)
   - `handle_unmatched_peaks()` had wrong parameters
   - Fixed by matching actual 15-parameter signature

2. **NNLS Return Type Error** (5e6a045)
   - Function returned dict instead of array
   - Fixed: Return `fit_results['x']` directly

3. **Feature Calculation Order** (1021a1c)
   - Wrong y_pred passed to similarity features
   - Fixed: Calculate error features first

4. **Missing m/z Feature** (3a8607f)
   - Feature count mismatch (25 vs 26)
   - Fixed: Added `rt_mz[:,1]` as 26th feature

5. **IndexError in get_scribe** (910ab3e)
   - Passed filtered peaks instead of full spectrum
   - Fixed: Use `spectrum_info.dia_spectrum[:, 1]`

6. **Dimension Mismatch** (ba3f1d5)
   - Features calculated for different candidate sets
   - Fixed: Use `candidates.window_idxs[library_entries.ref_peaks_in_dia]`

7. **Missing Fragment Columns** (3c751db)
   - FDR analysis expected 7 fragment columns
   - Fixed: Added all fragment data with semicolon formatting

8. **Bin Centers Shape Error** (ea46709)
   - 2D array passed where 1D expected
   - Fixed: Proper 1D bin_centers calculation

9. **Decoy Feature Calculation** (1626ed8, 5761800, 379a18e)
   - Missing decoy features when decoy=True
   - KeyError from decoy library mismatch
   - Fixed: Separate code path for decoy processing
   - Use original create_entries for decoys

10. **Large Coefficient IndexError** (4a571c6)
    - Index out of bounds in coefficient analysis
    - Fixed: Pass concatenated target+decoy data

11. **Empty Decoy Matches** (66b35b7)
    - Dimension mismatch when no decoys match
    - Fixed: Check for empty arrays before processing

## FDR Analysis Fix (56fb262)

### The Problem
- Only 1 decoy found out of thousands of samples
- Root cause: Decoys have low `frac_lib_int` values (< 0.1)
- Default `score_lib_frac` threshold is 0.5
- All decoys filtered out before FDR scoring

### The Solution
```python
# Ensure minimum decoys for FDR
MIN_DECOYS_FOR_FDR = 20
if fdx_toscore['decoy'].sum() < MIN_DECOYS_FOR_FDR and fdc['decoy'].sum() >= MIN_DECOYS_FOR_FDR:
    # Include all decoys regardless of frac_lib_int
    all_decoys = fdc[fdc['decoy']]
    filtered_targets = fdx_toscore[~fdx_toscore['decoy']]
    fdx_toscore = pd.concat([filtered_targets, all_decoys], ignore_index=True)
```

## Key Lessons Learned

### 1. Data Flow Preservation
- Even logical optimizations can break hidden assumptions
- Functions often use indices referencing full arrays
- Filtering at different stages must be carefully tracked

### 2. Feature Dimensions
- All 26 features must align to same candidate set
- Careful indexing required: `window_idxs[ref_peaks_in_dia]`
- Separate processing paths for targets and decoys

### 3. Integration Testing Critical
- Unit tests passed but real data revealed issues
- Dimension mismatches only visible with varying data
- Edge cases (empty matches) must be handled

### 4. Original Implementation Quirks
- Decoy processing uses different code path
- Some arrays pre-filtered, others not
- Hidden dependencies between functions

### 5. Backward Compatibility Challenges
- Output format must match exactly
- Feature order critical for downstream
- Fragment data formatting requirements

## Code Structure Created

### Type Definitions (types.py)
```python
@dataclass
class SpectrumInfo:
    spec_idx: int
    dia_spectrum: np.ndarray
    prec_mz: float
    prec_rt: float
    window_width: float
    ms1_spec: Optional[Any]

@dataclass
class CandidateSelection:
    mass_window_candidates: List[Tuple[str, int]]
    candidate_peaks: List[np.ndarray]
    window_idxs: np.ndarray
    ref_peaks_in_dia: List[int]

# ... more types defined
```

### Modular Functions Created
- `process_dia_spectrum()` - Peak merging and binning
- `filter_mass_window_candidates()` - RT/mass filtering
- `create_library_entries_wrapper()` - Peak matching
- `handle_unmatched_peaks()` - Unmatched peak processing
- `build_sparse_matrix()` - Matrix construction
- `solve_nnls_system()` - NNLS optimization
- `calculate_similarity_features()` - SCRIBE, Manhattan, cosine
- `calculate_intensity_features()` - Peak counts, fractions
- `calculate_error_features()` - Residuals, RT/mass errors
- `calculate_hyperscore_features()` - Fragment scoring
- `fit_to_lib2_orchestrator()` - Main orchestration

## Why Refactoring Was Reverted

1. **Too Many Integration Issues**: Each fix revealed new problems
2. **Hidden Dependencies**: Original code had many implicit assumptions
3. **Risk vs Reward**: Working system being destabilized
4. **Time Investment**: Debugging taking longer than anticipated
5. **Data Flow Complexity**: Tracking indices across modules proved error-prone

## Recommendations for Future Attempts

1. **Incremental Refactoring**: One function at a time with full testing
2. **Preserve Data Flow**: Don't optimize index mappings
3. **Integration Tests First**: Need comprehensive test suite
4. **Document Assumptions**: Original code's implicit rules
5. **Keep Fallback**: Maintain ability to use original code
6. **Profile First**: Identify actual bottlenecks before optimizing

## Valuable Discoveries

### Decoy Processing
- Decoys created with "Decoy_" prefix
- Separate library and processing path
- Lower quality metrics expected

### Feature Calculation
- Exact 26-feature order required
- Features must align to filtered candidates
- Some features need full spectrum data

### FDR Analysis
- Decoys filtered by quality metrics
- Need minimum count for statistics
- Separate handling preserves FDR validity

## Files Changed
- Modified: `src/spectral_fitting.py` (import adapter)
- Created: `src/spectral_fitting/` directory with 8 new files
- Modified: `src/CLAUDE.md` (added documentation)
- Modified: `src/spectral_fitting/CLAUDE.md` (detailed docs)
- Modified: `src/fdr_analysis.py` (FDR fix)
- Total: ~2000 lines refactored into modular structure

## Conclusion

While the refactoring was ultimately reverted, it provided valuable insights into the codebase structure and hidden dependencies. The modular design created could serve as a blueprint for future refactoring efforts, but any such effort should be done more incrementally with comprehensive integration testing at each step.

The FDR analysis fix identified during this work is valuable and could be reapplied independently of the refactoring.