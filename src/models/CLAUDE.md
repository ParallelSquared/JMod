# CLAUDE.md - Models Module

This directory contains model-related code for JMod, including spectral library handling.

## Navigation
- [← Back to Main Documentation](../../CLAUDE.md)
- [→ Spectral Fitting Module](../spectral_fitting/CLAUDE.md)
- [→ Utils Module](../utils/CLAUDE.md)
- [→ RT Models Directory](../../rt_models/CLAUDE.md)

## Overview

The models module provides:
- Spectral library management and indexing
- Library loading and caching mechanisms
- Efficient peptide lookup structures

## Module Structure

### spec_lib/spec_lib.py
Core spectral library functionality.

#### Key Classes and Functions

1. **SpecLibrary Class** (if present)
   - Manages spectral library data structures
   - Provides efficient peptide lookup
   - Handles library indexing

2. **Library Loading Functions**
   ```python
   # Load spectral library from TSV/CSV
   library = load_spectral_library(library_path)
   
   # Create indices for fast lookup
   library_index = create_library_index(library)
   ```

3. **Library Format**
   The spectral library typically contains:
   - Peptide sequences
   - Precursor m/z values
   - Charge states
   - Retention time (iRT)
   - Fragment ion information
   - Protein associations

#### Library Structure
```python
# Typical library entry structure
library_entry = {
    "peptide": "PEPTIDER",
    "prec_mz": 500.123,
    "z": 2,
    "iRT": 45.6,
    "fragments": {...},
    "protein": "P12345"
}
```

## Integration with RT Models

While this module handles spectral libraries, retention time prediction uses separate models stored in the [rt_models directory](../../rt_models/CLAUDE.md):

- **RT Prediction Models**: Pre-trained CNN models for different tags
  - Label-free (LF)
  - mTRAQ
  - DiEthyl
  - Tag6 (Transfer Learning)

- **Model Selection**: Based on the `--tag` parameter
  - Each tag type has 5 models (ensemble)
  - Models are in TensorFlow SavedModel format

## Library Management

### Loading and Caching
1. **Initial Load**: Libraries are loaded from TSV/CSV files
2. **Pickle Caching**: Loaded libraries are cached as pickle files
3. **Index Creation**: Peptide-to-index mappings for fast lookup

### Key Operations
- **Peptide Lookup**: O(1) lookup by (sequence, charge) tuple
- **m/z Filtering**: Efficient filtering by precursor m/z range
- **RT Filtering**: Filter candidates by retention time window

## Used By

- [Main Pipeline](../../run_jmod.py): Loads spectral library at startup
- [Spectral Fitting](../spectral_fitting/CLAUDE.md): Matches spectra against library
- [RT Alignment](../rt_alignment.py): Uses iRT values for alignment
- [FDR Analysis](../fdr_analysis.py): Accesses decoy information

## Performance Considerations

1. **Memory Usage**
   - Large libraries (>200k peptides) can use significant memory
   - Consider filtering by m/z range for targeted analysis

2. **Lookup Performance**
   - Dictionary-based lookup is O(1)
   - Pre-filtering by m/z range reduces search space

3. **Caching Strategy**
   - Pickle files speed up repeated runs
   - Cache includes pre-computed indices

## Library Format Requirements

### Required Columns
- `peptide` or `sequence`: Peptide sequence
- `prec_mz` or `precursorMz`: Precursor m/z
- `z` or `charge`: Charge state
- `iRT` or `RT`: Retention time

### Optional Columns
- `protein`: Protein accession
- `fragments`: Fragment ion information
- `modifications`: PTM information

## Common Issues

1. **Library Loading Errors**
   - Check column names match expected format
   - Ensure no missing required values
   - Verify peptide sequence format

2. **Memory Issues**
   - Large libraries may require memory management
   - Consider splitting libraries for parallel processing

3. **RT Model Mismatches**
   - Ensure correct tag parameter for RT models
   - Models expect specific modifications

## Future Improvements

1. **Library Format Standardization**
   - Support more library formats (MSP, BLIB)
   - Automatic format detection

2. **Optimization**
   - Implement lazy loading for large libraries
   - Add database backend option

3. **Validation**
   - Library integrity checks
   - Automatic decoy generation

## Related Documentation
- [Main JMod Documentation](../../CLAUDE.md)
- [RT Models Documentation](../../rt_models/CLAUDE.md)
- [Spectral Fitting Module](../spectral_fitting/CLAUDE.md)
- [Utils Module](../utils/CLAUDE.md)