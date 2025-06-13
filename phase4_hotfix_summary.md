# Phase 4 Hotfix: Name Collision Resolution

## Issue Discovered
After Phase 4 Part 3, the code broke with a TypeError:
```
TypeError: get_residuals() takes 6 positional arguments but 10 were given
```

## Root Cause
When removing the `_v2` suffix from functions, we created name collisions:

1. **Imported functions** from `utils.spectral_similarity_metrics`:
   - `get_residuals` (expects 10 parameters for ref/decoy separation)
   - `get_manhattan_distance` (original implementation)

2. **Local functions** defined in `spectral_fitting.py`:
   - `get_residuals` (expects 6 parameters for unified approach)
   - `get_manhattan_distance` (unified implementation)

When `fit_to_lib` called `get_residuals`, Python resolved to the imported version but passed arguments for the local version.

## Solution Implemented
Renamed local functions to avoid collisions:
- `get_residuals` → `compute_residuals`
- `get_manhattan_distance` → `compute_manhattan_distance`

This maintains:
- Backward compatibility with `fit_to_lib` (uses imported versions)
- Clear separation between old and new implementations
- All functionality preserved

## Verification
- All spectral fitting tests pass
- Both `fit_to_lib` and `fit_to_lib2` work correctly
- No other name collisions identified

## Lessons Learned
When refactoring and consolidating code:
1. Check for name collisions with imported functions
2. Consider function signatures when renaming
3. Test both old and new code paths
4. Use distinct names for different implementations

## Status
✅ Issue resolved and tested
✅ Code functioning correctly
✅ Ready for continued use

## Update: Second Fix Required
After the initial fix, discovered that `fit_to_lib` was incorrectly changed to use `compute_residuals`. 
- `fit_to_lib` must use the imported `get_residuals` and `get_manhattan_distance` (old signatures)
- `fit_to_lib2` uses the new `compute_residuals` and `compute_manhattan_distance` (unified approach)
- This properly separates the two implementations