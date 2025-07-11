# Unused Function Analysis Report

This report identifies potentially unused functions in `spectral_fitting.py` and `spectral_similarity_metrics.py`.

## Methodology

Functions were analyzed by searching for:
1. Direct function calls `function_name(`
2. Import statements containing the function name
3. Usage across the entire codebase (excluding the definition itself)

## Analysis Results

### spectral_fitting.py

Based on manual inspection and code analysis, the following functions appear to be potentially unused or have limited usage:

#### Potentially Unused Functions:

1. **`compute_residuals`** - Old residual calculation function, likely replaced by CSC versions
2. **`compute_manhattan_distance`** - Old Manhattan distance function, likely replaced by CSC versions
3. **`unmatched_peaks`** - Appears to be an older helper function
4. **`calculate_frac_dia_intensity_sparse`** - Older sparse version, replaced by CSC version
5. **`calculate_r2_lib_spec_sparse`** - Older sparse version, replaced by CSC version
6. **`calculate_unique_peak_features_sparse`** - Older sparse version, replaced by CSC version
7. **`calculate_frac_dia_intensity_pred`** - Standalone function that may be inlined elsewhere
8. **`calculate_b_y_ion_counts`** - Fragment counting function that may be replaced
9. **`calculate_hyperscores`** - Hyperscore calculation that may be moved elsewhere
10. **`calculate_longest_y_ions`** - Y-ion series calculation that may be replaced

#### Functions That Are Actively Used:

- `fit_to_lib` - Main RT alignment function (used in rt_alignment.py)
- `fit_to_lib2` - Main spectral fitting function (used in run_jmod.py)
- `build_sparse_matrix_direct` - Used in fit_to_lib for matrix construction
- `solve_nnls_simple` - Used for NNLS optimization
- `format_rt_alignment_output` - Used to format fit_to_lib output
- `calculate_frac_lib_intensity_sparse` - Used in RT alignment feature calculation
- `calculate_frac_dia_intensity_csc` - Used in feature calculators
- `preprocess_dia_spectrum` - Used for spectrum preprocessing
- `filter_candidates_by_window` - Used for candidate filtering
- `check_ms1_peaks` - Used for MS1 validation
- Many other helper functions used internally

### spectral_similarity_metrics.py

All functions in this file appear to be actively used:

1. **`get_closest_ms1`** - Used for MS1 spectrum matching
2. **`get_scribe_csc`** - Used for SCRIBE score calculation
3. **`get_residuals_csc`** - Used for residual calculation
4. **`get_manhattan_distance_csc`** - Used for Manhattan distance calculation
5. **`gof_stat_csc`** - Used for goodness-of-fit statistics

These are all CSC sparse matrix implementations used by the feature calculators.

## Recommendations

### Functions to Consider Removing:

1. **Old sparse matrix functions** that have CSC replacements:
   - `calculate_frac_dia_intensity_sparse`
   - `calculate_r2_lib_spec_sparse`
   - `calculate_unique_peak_features_sparse`

2. **Legacy computation functions**:
   - `compute_residuals`
   - `compute_manhattan_distance`
   - `unmatched_peaks`

3. **Standalone fragment functions** (if confirmed unused):
   - `calculate_b_y_ion_counts`
   - `calculate_hyperscores`
   - `calculate_longest_y_ions`
   - `calculate_frac_dia_intensity_pred`

### Functions to Keep:

All other functions appear to be part of the active pipeline and should be retained.

## Notes

- Some functions marked as "unused" may be called dynamically or used in ways not detected by simple grep searches
- Before removing any function, verify it's not used in:
  - Test files
  - Dynamic imports
  - String-based function calls
  - External scripts or notebooks
- The CSC (Compressed Sparse Column) versions have generally replaced older sparse matrix implementations
- Many helper functions are used internally within fit_to_lib and fit_to_lib2

## Conclusion

The codebase shows signs of evolution from older sparse matrix implementations to newer CSC-based ones. The potentially unused functions are mostly:
1. Old implementations that have been replaced
2. Standalone feature calculation functions that may have been integrated elsewhere
3. Legacy helper functions from earlier versions

A careful review and testing should be done before removing any functions to ensure they are truly unused.