# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

JMod (Joint Modeling) is an open-source proteomics software that increases mass spectrometry throughput by supporting multiplexing in mass and time domains. It performs joint modeling of mass spectra to deconvolve overlapping isotopic envelopes in both MS1 and MS2 space.

## Common Development Commands

### Running JMod Analysis
```bash
# Basic run with mzML and spectral library
python run_jmod.py -i path/to/file.mzml -l path/to/library.tsv

# Run with JSON configuration
python run_jmod.py --config_json path/to/config.json

# Run with retention time filtering
python run_jmod.py -i file.mzml -l library.tsv -r --rt_tol 0.5

# Run with specific tag (e.g., mTRAQ, diethyl)
python run_jmod.py -i file.mzml -l library.tsv --tag mTRAQ
```

### Testing
```bash
# Run all tests with coverage
python run_tests.py -c

# Run tests with verbose output
python run_tests.py -v

# Run specific test suites
python run_tests.py --dummy      # Basic setup tests
python run_tests.py --misc       # Utility function tests

# Run specific test file
python run_tests.py tests/test_spectral_fitting.py

# Using pytest directly
pytest tests/ -v --cov=src --cov-report=html
pytest tests/utils/test_iso_functions.py -v  # Single test file
```

### Integration Testing
```bash
# Quick integration test (< 5 minutes)
python run_jmod.py --config_json ./data/test_config_mzml.json
```

### Linting and Type Checking
The project currently does not have explicit linting or type checking commands configured. Consider using:
```bash
# Python linting (if installed)
flake8 src/ tests/
pylint src/

# Type checking (if installed)
mypy src/
```

## Architecture Overview

### Core Pipeline Flow
1. **Input Processing** (`run_jmod.py`): Entry point that handles mzML files and spectral libraries
2. **RT Alignment** (`rt_alignment.py`): CNN-based retention time prediction and alignment using pre-trained models
3. **Spectral Fitting** (`spectral_fitting.py`): Core matching algorithm with isotope deconvolution
4. **FDR Analysis** (`fdr_analysis.py`): Statistical validation using target-decoy approach
5. **Post Processing** (`post_process.py`): Final result compilation and output generation

### Key Module Relationships
- `src/utils/`: Low-level utilities used throughout the pipeline
  - `iso_functions.py`: Isotope pattern calculations critical for deconvolution
  - `parse_peptides.py`: Peptide sequence parsing with modification handling
  - `spectral_similarity_metrics.py`: Spectral similarity scoring functions (SCRIBE, Manhattan distance, goodness-of-fit)
- `src/mass_tags.py`: Tag library definitions for different labeling methods (mTRAQ, diethyl, etc.)
- `src/models/spec_lib/`: Spectral library handling and indexing
- `rt_models/`: Pre-trained TensorFlow models for RT prediction

### Configuration System
JMod uses a flexible parameter system that accepts:
- Command-line arguments (see `src/config.py` for all options)
- JSON configuration files (example in `data/default_config.json`)
- Precedence: CLI args override JSON config

Key parameters to understand:
- `ppm`: MS2 mass tolerance
- `atleast_m`: Minimum number of fragments to match
- `use_rt`: Enable retention time filtering
- `tag`: Labeling method (affects mass calculations)
- `iso`: Use isotope patterns in matching
- `timeplex`/`plexDIA`: Multiplexing modes

### Testing Approach
- Uses pytest with fixtures in `tests/conftest.py`
- Mock configurations to avoid external dependencies
- Test data fixtures in `tests/fixtures/test_data.py`
- Coverage reporting integrated (currently ~27% coverage)
- Focus on unit tests for utility functions

## Unified Spectral Fitting Architecture

### Overview
JMod now uses a unified spectral fitting approach that processes target and decoy peptides together in single data structures. This architecture was implemented through a careful refactoring process (Phases 1-4) that consolidated duplicate code while maintaining identical functionality.

### Key Components

#### Data Structures
- **UnifiedCandidates**: Combines target and decoy candidates with boolean tracking
  - `candidates`: List of (sequence, charge) tuples
  - `is_decoy`: Boolean array indicating decoy status
  - `peaks`: Peak arrays for each candidate
  - Methods for filtering targets/decoys
  
- **UnifiedMatrixData**: Sparse matrix representation for NNLS
  - Row/column indices and values
  - Split arrays for per-candidate access
  - Target/decoy tracking integrated
  
- **UnifiedFeatures**: Feature matrix for all candidates
  - 26 features per candidate
  - Boolean tracking for decoy status
  - Feature names for interpretability

#### Core Functions
1. **create_entries**: Processes candidates and matches to DIA spectrum
   - Handles MS1 filtering
   - Peak matching with tolerance windows
   - Returns unified data structures

2. **Matrix Construction**:
   - `unmatched_peaks`: Handles peaks not in DIA spectrum
   - `build_sparse_matrix`: Creates sparse matrix for NNLS
   - `process_matrix`: Complete pipeline from candidates to coefficients

3. **Feature Calculation**:
   - `compute_residuals`: Calculate prediction residuals
   - `compute_manhattan_distance`: Spectral similarity metrics
   - `calculate_features`: Comprehensive feature extraction

4. **Main Entry Points**:
   - `fit_to_lib`: Used by RT alignment (no decoys)
   - `fit_to_lib2`: Primary function for spectral matching (with decoys)

### Benefits of Unified Approach
- **Code Reduction**: ~41% fewer lines (from ~2,700 to ~1,587)
- **Single Code Path**: No duplicate logic for targets/decoys
- **Better Performance**: Single matrix construction and NNLS solve
- **Maintainability**: Changes apply to all peptide types automatically
- **Type Safety**: Dataclasses provide structure and validation
- **Optimized Operations**: Vectorized calculations where possible

### Implementation Details
- Targets and decoys are processed together from the start
- Boolean arrays track peptide types throughout pipeline
- Column indices automatically handle target/decoy offsets
- Output format remains identical to preserve compatibility

### Performance Characteristics
- Memory usage similar to original implementation
- Computation time reduced due to single NNLS solve
- Vectorized numpy operations for efficiency
- No performance penalty from unified approach

### Migration Notes
- The unified implementation is now the default in `spectral_fitting.py`
- Original dual-path code has been removed
- All tests pass with identical results
- Backward compatibility maintained for RT alignment