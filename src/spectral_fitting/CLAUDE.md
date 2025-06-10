# CLAUDE.md - Spectral Fitting Module

This directory contains the refactored spectral fitting implementation for JMod.

## Navigation
- [← Back to Main Documentation](../../CLAUDE.md)
- [→ Utils Module](../utils/CLAUDE.md)
- [→ I/O Module](../utils/io/CLAUDE.md)

## Overview

The spectral fitting module performs spectrum-to-library matching with isotope deconvolution. It has been refactored from the original monolithic implementation to a modular architecture with clear separation of concerns.

## Module Structure

### Core Components

1. **types.py**: Type definitions and data structures
   - `SpectrumMatrix`: Unified spectrum representation
   - `SpectralFeatures`: Container for all calculated features
   - `SpectralFitResult`: Result of fitting operation
   - Various feature containers (BasicFeatures, SimilarityFeatures, etc.)

2. **matrix_operations.py**: Core numerical operations
   - Sparse matrix construction
   - NNLS (Non-Negative Least Squares) solving
   - Matrix transformations

3. **feature_calculation.py**: Feature extraction
   - Basic features (num peaks matched, intensities)
   - Similarity metrics (cosine, Manhattan distance, R²)
   - Statistical features (goodness of fit, residuals)
   - Fragment-level features (hyperscores, b/y counts)

4. **fitting_core.py**: Main fitting logic
   - `fit_spectrum_to_library()`: Unified fitting function
   - Handles both regular and decoy searches
   - RT and MS1 filtering

5. **adapter.py**: Backward compatibility layer
   - `fit_to_lib()`, `fit_to_lib_decoy()`, `fit_to_lib2()` wrappers
   - Converts new result format to legacy format
   - **Critical**: Fragment info indexing uses position in non-zero results

## Key Design Decisions

### Unified Spectrum Matrix
Instead of separate handling for library and decoy spectra, everything is combined into a single `SpectrumMatrix`. This simplifies the fitting logic and ensures consistent handling.

### Feature Calculation Pipeline
Features are calculated in a specific order:
1. Basic matching features
2. Similarity scores
3. Statistical metrics
4. Fragment-level information

### Backward Compatibility
The adapter layer ensures existing code continues to work while allowing gradual migration to the new API.

## Common Issues and Solutions

### Fragment Information Indexing
**Issue**: Fragment information lists are indexed by position in the non-zero coefficients array, not by the original peptide index.

**Solution**: In the adapter, use loop index `i` instead of `coeff_idx` when accessing fragment information:
```python
# Correct
if len(features.fragment_info.frag_int) > i:
    frag_data = features.fragment_info.frag_int[i]

# Incorrect (causes all correlations to be 0)
if len(features.fragment_info.frag_int) > coeff_idx:
    frag_data = features.fragment_info.frag_int[coeff_idx]
```

### Fragment Intensity Formatting
**Issue**: The legacy code expects semicolon-delimited strings, but lists might contain nested arrays.

**Solution**: The adapter handles both flat and nested lists:
```python
if isinstance(frag_data[0], (int, float, np.number)):
    frag_int_str = ";".join([str(float(x)) for x in frag_data])
else:
    # Handle nested lists
    flat_data = [float(x) for item in frag_data for x in (item if isinstance(item, (list, np.ndarray)) else [item])]
    frag_int_str = ";".join([str(x) for x in flat_data])
```

## Future Improvements

1. **Remove adapter layer**: Once all calling code is updated, the adapter can be removed
2. **Optimize memory usage**: The unified spectrum matrix can be large
3. **Parallel processing**: Feature calculation could be parallelized
4. **Type hints**: Add comprehensive type hints throughout
5. **Unit tests**: Add tests for each component separately

## Performance Considerations

- The unified spectrum matrix approach may use more memory but simplifies the code
- NNLS solving is the computational bottleneck
- Feature calculation is generally fast but could be optimized for large datasets

## Dependencies

- NumPy for numerical operations
- SciPy for sparse matrices and NNLS solving
- Internal modules:
  - [utils.misc_functions](../utils/CLAUDE.md#misс_functionspy): Fragment correlation, string conversions
  - [utils.sparse_nnls](../utils/CLAUDE.md#sparse_nnlspy): NNLS solver
  - [utils.spectral_similarity_metrics](../utils/CLAUDE.md#spectral_similarity_metricspy): Similarity calculations
  - [utils.io.read_output](../utils/io/CLAUDE.md#read_outputpy): Column definitions

## Related Documentation
- [Main JMod Documentation](../../CLAUDE.md)
- [Utils Module Documentation](../utils/CLAUDE.md)
- [I/O Module Documentation](../utils/io/CLAUDE.md)