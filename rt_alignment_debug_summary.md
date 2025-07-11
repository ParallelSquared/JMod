# RT Alignment Debug Summary

## Key Findings

### Full Dataset (`default_config.json`) - FAILS
When running with the full dataset, the RT alignment fails because:

1. **manhattan_distances = -inf for ALL candidates (12,349)**
   - All values are negative infinity
   - Percentile calculation produces `nan` threshold
   - 0% of candidates pass this filter

2. **gof_stats = 0.0 for ALL candidates**
   - All values are exactly zero
   - Filter requires `< 0.0` which is impossible
   - 0% of candidates pass this filter

3. **max_matched_residuals = 0.0 for ALL candidates**
   - All values are exactly zero
   - Filter requires `< 0.0` which is impossible
   - 0% of candidates pass this filter

4. **Other features look normal:**
   - hyperscore: Valid distribution (min: -0.5095, max: 147.2370)
   - frag_cosines_p: Valid distribution (min: -0.9976, max: 1.0000)
   - scribe_scores: Valid distribution (min: 0.0006, max: 0.5913)
   - med_frag_error: All zeros (but filter allows this)

### Test Dataset (`test_config_mzml.json`) - WORKS
When running with the test dataset:
- Uses hyperscore filtering instead of the complex multi-feature filtering
- 744 out of 3,721 points pass the hyperscore filter (20%)
- RT alignment completes successfully
- **No debug output from the multi-feature filtering section**

## Critical Observation

**The debug statements for feature statistics don't print with the test dataset!**

This indicates that the code path is different:
- Test dataset uses the `else` branch starting around line 1007 (hyperscore-only filtering)
- Full dataset uses the complex multi-feature filtering (lines 852-1006)
- The `dino_features` parameter or data characteristics determine which path is taken

## Root Cause Analysis

### Why Different Code Paths?

Looking at the code structure:
```python
if dino_features is not None and len(output_hyper) > 0:
    # Complex multi-feature filtering (full dataset path)
    # This is where the debug statements are
else:
    # Simple hyperscore filtering (test dataset path)
    # No debug statements here
```

### Why Are Features Zero/Infinity?

1. **Residuals Not Calculated**: 
   - Both `gof_stats` and `max_matched_residuals` depend on residuals
   - If residuals are all zero, both features will be zero
   - This suggests `y_pred` (predicted spectrum) is not being calculated

2. **Manhattan Distance = -inf**:
   - The formula is likely: `manhattan = -log10(distance)`
   - If distance = 0 (perfect match), then log(0) = -inf
   - This suggests all spectra are matching perfectly (unlikely) or calculation error

3. **Connection to Refactoring**:
   - The unified feature calculator may not be passing `y_pred` correctly
   - CSC sparse matrix operations might be returning different results
   - The residual calculation chain is broken

## Recommended Investigation Steps

1. **Check why test dataset takes different path**:
   - Print `dino_features is not None` and `len(output_hyper)`
   - Understand why test dataset doesn't use multi-feature filtering

2. **Debug feature calculation in fit_to_lib2**:
   - Check if `y_pred` is being calculated
   - Verify residuals are computed
   - Trace the data flow to scoring functions

3. **Add debug output to feature calculator**:
   - Print shape and sample values of `y_pred`
   - Print residuals before passing to scoring functions
   - Check sparse matrix dimensions

## Hypothesis

The refactoring likely broke the residual/prediction calculation in the full spectral fitting pipeline:
- The test dataset works because it uses a simpler code path
- The full dataset fails because features that depend on spectral prediction are broken
- The issue is in how `fit_to_lib2` calculates or passes the predicted spectrum

## Next Steps

1. Add debugging to understand why code paths differ
2. Fix the residual/prediction calculation in the unified feature calculator
3. Ensure Manhattan distance handles edge cases properly
4. Verify all features are calculated correctly before filtering