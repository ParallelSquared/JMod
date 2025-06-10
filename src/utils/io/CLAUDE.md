# CLAUDE.md - I/O Module

This subdirectory handles all input/output operations for JMod.

## Navigation
- [← Back to Main Documentation](../../../CLAUDE.md)
- [← Back to Utils Module](../CLAUDE.md)
- [→ Spectral Fitting Module](../../spectral_fitting/CLAUDE.md)

## Overview

The I/O module provides unified interfaces for:
- Loading mass spectrometry data files (mzML, Arrow)
- Reading spectral libraries
- Loading protein sequences (FASTA)
- Managing output file formats

## Module Structure

### load_files.py
Handles loading of mass spectrometry data files.

#### Key Classes

1. **BaseSpectrum** (Abstract)
   - Common interface for all spectrum types
   - Properties: `id`, `level`, `RT`, `mz`, `intens`, `TIC`
   - Method: `peak_list()` returns array of [mz, intensity]

2. **MzMLSpectrum**
   - Concrete implementation for mzML spectra
   - Additional properties: `scan_num`, `collision_energy`, `prec_mz`
   - Handles MS1 and MS2 specific attributes

3. **ArrowSpectrum**
   - Efficient implementation for Apache Arrow format
   - Uses property-based access for lazy loading
   - Caches frequently accessed data

4. **BaseSpectrumFile** (Abstract)
   - Common interface for spectrum file readers
   - Properties: `ms1scans`, `ms2scans`
   - Method: `get_by_idx(scan_num)` retrieves spectrum by scan number

5. **MzMLSpectrumFile**
   - Loads mzML format files using pyteomics
   - Builds scan position index for fast access

6. **ArrowSpectrumFile**
   - Loads Apache Arrow IPC files
   - Memory-efficient with property-based access
   - Pre-creates spectrum objects for performance

#### Key Functions

```python
# Auto-detect file type and load
spectra = loadSpectra(file_path)  # .mzml or .arrow

# Load with caching (creates .mzml_pythonspec pickle)
spectra = loadMzMLSpectra("data.mzml")

# Load Arrow file with caching
spectra = loadArrowSpectra("data.arrow")
```

### read_output.py
Manages output file formats and column definitions.

#### Key Variables

1. **names**: List of column names for output CSV files
   - Defines the standard output format for JMod results
   - Used by [spectral fitting adapter](../../spectral_fitting/CLAUDE.md) for legacy format conversion
   - Critical for maintaining backward compatibility

2. **names_timeplex**: Extended column names for timeplex mode

#### Column Categories
- Basic info: `coeff`, `spec_id`, `seq`, `z`
- Scores: `hyperscore`, `cosine`, `manhattan_distances`
- Fragment info: `frag_names`, `frag_int`, `obs_int`
- Statistics: `gof_stats`, `max_matched_residuals`

### load_fasta.py
Handles protein sequence files.

#### Key Functions

```python
# Load FASTA file
proteins = load_fasta(fasta_file)

# Parse headers and extract metadata
protein_info = parse_fasta_header(header_line)
```

## Common Patterns

### File Loading with Caching
All file loaders support pickle caching for faster subsequent loads:
```python
# First load: reads from file, creates cache
spectra = loadMzMLSpectra("data.mzml")  # Creates data.mzml_pythonspec

# Subsequent loads: reads from cache
spectra = loadMzMLSpectra("data.mzml")  # Loads from pickle
```

### Error Handling
The module handles various error conditions:
- Corrupted pickle files are automatically recreated
- Missing required fields in mzML files raise informative errors
- Arrow files validate structure on load

## Integration Points

### Used By
- [Main pipeline](../../../run_jmod.py): Loads input files
- [Spectral Fitting](../../spectral_fitting/CLAUDE.md): Accesses spectrum data
- [RT Alignment](../../rt_alignment.py): Uses RT and m/z information
- [Post Processing](../../post_process.py): Writes results using column definitions

### Depends On
- PyArrow: For Arrow file support
- Pyteomics: For mzML parsing
- NumPy: For array operations
- Pickle: For caching

## Performance Considerations

1. **Arrow vs mzML**
   - Arrow files load ~10x faster than mzML
   - Arrow uses less memory with lazy loading
   - mzML provides better compatibility

2. **Caching Strategy**
   - Pickle files speed up repeated runs
   - Cache invalidation on file modification not implemented
   - Manual cache deletion required after file updates

3. **Memory Usage**
   - ArrowSpectrum uses property-based access to reduce memory
   - MzMLSpectrum loads all data upfront
   - Large files may require memory management

## Known Issues

1. **Arrow File Compatibility**
   - Requires specific column structure
   - No validation of Arrow schema on load
   - See [test recommendations](spectrum_classes_refactor_recommendations.md)

2. **Pickle Compatibility**
   - Pickle files may break between Python versions
   - Class name changes invalidate cached files
   - Automatic recreation handles most cases

## Future Improvements

1. **Unified Spectrum Interface**
   - Further abstract common operations
   - Add validation methods
   - Implement lazy loading for mzML

2. **Better Caching**
   - Check file modification times
   - Implement cache versioning
   - Add cache management utilities

3. **Format Support**
   - Add mzMLb (binary mzML) support
   - Support Parquet for better compression
   - Implement streaming for large files

## Related Documentation
- [Main JMod Documentation](../../../CLAUDE.md)
- [Utils Module](../CLAUDE.md)
- [Spectral Fitting Module](../../spectral_fitting/CLAUDE.md)