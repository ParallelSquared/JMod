# Data Structures in `calculate_rt_alignment_features`

## Overview

This document explains the differences between `ref_spec_values_split`, `ref_spec_row_indices_split`, `ref_spec_col_indices_split`, and `sparse_lib_matrix` in the `calculate_rt_alignment_features` function within `spectral_fitting.py`.

These data structures represent different views and transformations of the same underlying spectral matching data, serving distinct purposes in the feature calculation pipeline.

## 1. `ref_spec_*_split` Arrays - Per-Candidate Decomposition

These are **lists of numpy arrays**, where each list element corresponds to one peptide candidate:

### `ref_spec_values_split` (Line 1615)
- **Purpose**: Normalized intensity values for matched peaks, per candidate
- **Structure**: `List[np.ndarray]` - one array per candidate
- **Content**: Intensity values from the spectral library for peaks that matched the DIA spectrum
- **Example**: `[np.array([0.3, 0.5, 0.2])]` - one candidate with 3 matched peaks having intensities 0.3, 0.5, 0.2
- **Creation**: `[ints[i] for ints, i in zip(norm_intensities, lib_peaks_matched)]`
- **Used for**: Per-candidate intensity correlation calculations (line 1769, 1793-1794)

### `ref_spec_row_indices_split` (Line 1612)
- **Purpose**: DIA spectrum row indices where each candidate's peaks matched
- **Structure**: `List[np.ndarray]` - one array per candidate  
- **Content**: Indices into the DIA spectrum array for matched peaks
- **Example**: `[np.array([0, 1, 2])]` - candidate's peaks matched at DIA spectrum rows 0, 1, 2
- **Creation**: `[np.int32(((i[j] + 1) / 2) - 1) for i, j in zip(ref_pep_cand_loc, lib_peaks_matched)]`
- **Used for**: Extracting DIA intensities for per-candidate correlations (line 1794)

### `ref_spec_col_indices_split` (Line 1614)
- **Purpose**: Column indices for sparse matrix construction, per candidate
- **Structure**: `List[np.ndarray]` - one array per candidate
- **Content**: All elements are the candidate index (0, 1, 2, etc.)
- **Example**: `[np.array([0, 0, 0])]` - all peaks from candidate 0 point to column 0
- **Creation**: `[np.array([idx] * i) for idx, i in zip(range(len(ref_pep_cand)), num_lib_peaks_matched)]`
- **Used for**: Sparse matrix construction and SCRIBE score calculations (line 1855-1857)

## 2. `sparse_lib_matrix` - Unified Matrix Representation

This is a **scipy sparse matrix** representing the complete system:

### `sparse_lib_matrix` (Lines 1677, 1680)
- **Purpose**: Complete linear system for NNLS optimization
- **Structure**: `scipy.sparse.coo_matrix` with shape `(n_dia_peaks + penalty_rows, n_candidates)`
- **Content**: Each column represents one candidate, each row represents one DIA spectrum peak
- **Example**: `sparse.coo_matrix(([0.3, 0.5, 0.2, 0.0], ([0, 1, 2, 3], [0, 0, 0, 0])), shape=(4, 1))`
  - 4 rows (3 DIA peaks + 1 penalty), 1 column (1 candidate)
  - Non-zero values: 0.3 at (0,0), 0.5 at (1,0), 0.2 at (2,0), 0.0 at (3,0)
- **Creation**: Constructed by concatenating all `ref_spec_*_split` arrays
- **Used for**: NNLS solving (`sparse_lib_matrix * lib_coefficients`) and spectrum prediction (line 1789)

## 3. Key Relationships and Differences

### Data Flow Transformation
```
Library Peaks → Per-candidate splits → Concatenation → Sparse Matrix
```

1. **Raw matching**: Library fragments matched to DIA spectrum peaks
2. **Per-candidate organization**: Split into separate arrays for each peptide candidate
3. **Matrix construction**: Concatenate splits and create sparse matrix for NNLS

### Different Use Cases

**Split Arrays (Per-Candidate Analysis)**:
- Individual peptide scoring and correlation
- Fragment-specific calculations 
- Per-candidate intensity ratios
- Used in lines 1769, 1771, 1793-1794, 1799-1800

**Sparse Matrix (System-Level Analysis)**:
- NNLS optimization solving
- Full spectrum prediction
- System-wide residual calculations
- Used in lines 1789, 1836-1837

### Memory and Computation Efficiency

**Split Arrays**:
- Allow vectorized per-candidate operations
- Enable efficient indexing into DIA spectrum
- Preserve candidate-specific information needed for detailed scoring

**Sparse Matrix**:
- Efficient storage for mostly-zero matrix
- Optimized for linear algebra operations (matrix multiplication)
- Single representation for NNLS solver

## 4. Example Data Consistency

From the test case in `tests/test_spectral_fitting.py`:

```python
# Split arrays (per-candidate view)
ref_spec_values_split = [np.array([0.3, 0.5, 0.2])]      # 1 candidate, 3 peaks
ref_spec_row_indices_split = [np.array([0, 1, 2])]       # matches DIA rows 0,1,2  
ref_spec_col_indices_split = [np.array([0, 0, 0])]       # all belong to candidate 0

# Sparse matrix (unified view)  
sparse_lib_matrix = sparse.coo_matrix(
    ([0.3, 0.5, 0.2, 0.0], ([0, 1, 2, 3], [0, 0, 0, 0])), # same values + penalty
    shape=(4, 1)  # 4 rows (3 peaks + penalty), 1 candidate
)
```

## 5. Summary

The sparse matrix contains the same data as the splits but in a unified linear algebra format suitable for optimization, while the splits maintain the per-candidate structure needed for detailed feature calculations.

- **Split arrays**: Maintain per-candidate organization for detailed scoring and feature extraction
- **Sparse matrix**: Provide unified format optimized for linear algebra operations and NNLS solving
- **Relationship**: The sparse matrix is constructed from the split arrays but serves different computational purposes
- **Efficiency**: Each format is optimized for its specific use case in the spectral fitting pipeline