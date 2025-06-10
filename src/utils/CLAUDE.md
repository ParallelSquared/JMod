# CLAUDE.md - Utils Module

This directory contains utility functions and helper modules used throughout JMod.

## Navigation
- [← Back to Main Documentation](../../CLAUDE.md)
- [→ Spectral Fitting Module](../spectral_fitting/CLAUDE.md)
- [→ I/O Subdirectory](io/CLAUDE.md)
- [→ Models Module](../models/CLAUDE.md)

## Overview

The utils module provides low-level functionality for:
- File I/O operations (see [I/O subdirectory](io/CLAUDE.md))
- Mathematical calculations
- Spectral processing utilities
- Peptide parsing and modification handling

## Module Structure

### Core Utility Files

1. **iso_functions.py**: Isotope pattern calculations
   - `isotope_pattern()`: Generate theoretical isotope distributions
   - `iso_pattern_combos()`: Handle multiple charge states
   - Critical for accurate mass deconvolution

2. **misc_functions.py**: General utility functions
   - `fragment_cor()`: Calculate fragment correlations between observed and theoretical
   - `string_floats()`/`unstring_floats()`: Convert arrays to/from semicolon-delimited strings
   - `hyperscore()` family: Calculate hyperscores for peptide matches
   - Various mathematical utilities (cosine similarity, percentile calculations)

3. **parse_peptides.py**: Peptide sequence parsing
   - Parse peptide sequences with modifications
   - Calculate molecular formulas
   - Handle various modification formats

4. **spectral_similarity_metrics.py**: Spectral comparison metrics
   - SCRIBE scores
   - Manhattan distances
   - Goodness-of-fit statistics
   - R² calculations

5. **sparse_nnls.py**: Sparse Non-Negative Least Squares solver
   - Core mathematical solver for deconvolution
   - Optimized for sparse matrices

### I/O Subdirectory (io/)

See [I/O Module Documentation](io/CLAUDE.md) for detailed information about:
- File loading utilities (mzML, Arrow formats)
- FASTA file handling
- Output file management

## Key Functions and Usage

### Fragment Correlation (misc_functions.py)
```python
def fragment_cor(df: pd.DataFrame, didx: int, fn: str = "cos") -> float:
    """Calculate correlation between observed and theoretical fragments"""
```
**Important**: Expects columns 'frag_names', 'obs_int', 'frag_int' as semicolon-delimited strings

### String/Float Conversion (misc_functions.py)
```python
# Convert array to semicolon-delimited string
string_floats([1.0, 2.0, 3.0]) → "1.0;2.0;3.0"

# Parse semicolon-delimited string to array
unstring_floats("1.0;2.0;3.0") → array([1.0, 2.0, 3.0])
```
**Note**: Cannot handle strings with brackets like "[1.0, 2.0]"

### Isotope Patterns (iso_functions.py)
```python
# Generate isotope pattern for peptide
pattern = isotope_pattern(peptide_dict, charge_state)
```

## Common Issues and Solutions

### Fragment Correlation Returns 0
**Causes**:
1. Missing columns in dataframe
2. Malformed string data (e.g., contains brackets)
3. No shared fragments between observed and theoretical
4. All zero intensities

**Solution**: Ensure proper string formatting without brackets

### Division by Zero in Statistical Functions
**Issue**: Empty arrays or single-element arrays in statistical calculations

**Solution**: Add checks for array length before operations:
```python
if len(data) > 0:
    result = sum(data) / len(data)
else:
    result = 0.0
```

### Arrow File Compatibility
**Issue**: Arrow files require specific structure and may have different spectrum access patterns

**Solution**: The ArrowSpectrumFile class provides property-based access to maintain compatibility

## Performance Considerations

- String parsing operations (`unstring_floats`) are slow for large datasets
- Consider caching parsed results when possible
- Isotope pattern calculations can be pre-computed for common peptides
- Sparse matrix operations are memory-efficient but require careful indexing

## Dependencies

- NumPy: Numerical operations
- Pandas: DataFrame operations
- SciPy: Scientific computing utilities
- PyArrow: Arrow file format support
- Pyteomics: mzML file parsing

## Future Improvements

1. **Optimize string operations**: Replace string-based fragment storage with structured arrays
2. **Add caching**: Cache frequently calculated values (isotope patterns, etc.)
3. **Improve error handling**: More informative error messages
4. **Type hints**: Add comprehensive type annotations
5. **Vectorize operations**: Replace loops with vectorized NumPy operations where possible

## Used By

- [Spectral Fitting Module](../spectral_fitting/CLAUDE.md): Uses fragment correlation, NNLS solver, similarity metrics
- [RT Alignment](../rt_alignment.py): Uses statistical functions, lowess fitting
- [Post Processing](../post_process.py): Uses output utilities, misc functions

## Related Documentation
- [Main JMod Documentation](../../CLAUDE.md)
- [Spectral Fitting Module](../spectral_fitting/CLAUDE.md)
- [I/O Subdirectory](io/CLAUDE.md)