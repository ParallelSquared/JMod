# Phase 4 Migration Map

## Overview
This document maps out exactly where each component from the unified modules will be moved during Phase 4 consolidation.

## Current State
- `fit_to_lib2_original` (lines 318-802 in spectral_fitting.py) - TO BE REMOVED
- `fit_to_lib2` (lines 1600+ in spectral_fitting.py) - Already uses unified approach
- Unified modules imported only in spectral_fitting.py

## Migration Plan

### 1. From spectral_fitting_unified.py → spectral_fitting.py

#### Classes to Move:
- `UnifiedCandidates` (lines 14-72)
- `UnifiedMatrixData` (lines 75-111)
- `UnifiedFeatures` (lines 114-147)

#### Functions to Move:
- `create_unified_candidates` (lines 150-202)
- `create_entries_unified` (lines 205+)

**Action**: Copy these classes and functions to the top of spectral_fitting.py after imports

### 2. From spectral_fitting_unified_features.py → spectral_fitting.py

#### Functions to Merge:
- `calculate_features_unified` → Rename to `calculate_features`
- `get_residuals_unified` → Rename to `get_residuals` (replace existing)
- `get_manhattan_distance_unified` → Rename to `get_manhattan_distance` (replace existing)
- `get_scribe_unified` → Merge into existing `get_scribe` in spectral_similarity_metrics.py
- Other unified feature functions → Remove "_unified" suffix

**Action**: Update existing feature calculation functions to work with unified data structures

### 3. From spectral_fitting_unified_matrix.py → spectral_fitting.py

#### Functions to Move:
- `process_matrix_unified` → Rename to `process_matrix`
- `build_sparse_matrix_unified` → Rename to `build_sparse_matrix`
- Helper functions for matrix operations

**Action**: Integrate matrix operations into main spectral fitting flow

### 4. Files to Delete Completely:
- `spectral_fitting_unified.py`
- `spectral_fitting_unified_features.py`
- `spectral_fitting_unified_matrix.py`
- `spectral_fitting_unified_adapter.py` (adapter pattern no longer needed)
- `spectral_fitting_unified_integration.py` (demo file)

### 5. Function Renaming:
- Remove all "_unified" suffixes from function names
- Update all function calls throughout spectral_fitting.py
- Ensure consistent naming conventions

### 6. Import Updates:
- Remove all imports of unified modules from spectral_fitting.py
- Add any necessary imports for numpy, typing, etc. that were in unified modules

### 7. Code to Remove from spectral_fitting.py:
- `fit_to_lib2_original` function (lines 318-802)
- `fit_to_lib` function (lines 806-1183) if not used
- `fit_to_lib_decoy` function (lines 1186-1597) if not used
- `get_features` function (lines 49-174) if replaced by unified version
- Original `create_entries` function if it exists

### 8. Testing Points:
- Ensure `fit_to_lib2` still works after consolidation
- Verify output format matches exactly
- Test with both targets and decoys
- Test with different configurations (mTRAQ, isotopes, etc.)

## Expected Results:
- ~40% reduction in code size
- Single implementation for target/decoy processing
- Improved maintainability
- Same functionality and output