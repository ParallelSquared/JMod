# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

JMod (Joint Modeling) is an open-source proteomics software that increases mass spectrometry throughput by supporting multiplexing in mass and time domains. It performs joint modeling of mass spectra to deconvolve overlapping isotopic envelopes in both MS1 and MS2 space.

## Documentation Structure

- **Main Documentation** (this file): Overview, commands, architecture
- **[Spectral Fitting Module](src/spectral_fitting/CLAUDE.md)**: Core matching algorithm implementation
- **[Utils Module](src/utils/CLAUDE.md)**: Utility functions and helpers
- **[I/O Module](src/utils/io/CLAUDE.md)**: File loading and data handling
- **[Models Module](src/models/CLAUDE.md)**: Spectral library management
- **[RT Models](rt_models/CLAUDE.md)**: Pre-trained retention time prediction models

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
- **[src/spectral_fitting/](src/spectral_fitting/CLAUDE.md)**: Refactored spectral matching implementation
  - Core algorithm for spectrum-to-library matching
  - Feature calculation and scoring
  - Backward compatibility adapter
- **[src/utils/](src/utils/CLAUDE.md)**: Low-level utilities used throughout the pipeline
  - `iso_functions.py`: Isotope pattern calculations critical for deconvolution
  - `parse_peptides.py`: Peptide sequence parsing with modification handling
  - `sparse_nnls.py`: Mathematical solver for spectral deconvolution
  - `spectral_similarity_metrics.py`: Spectral similarity scoring functions
- **[src/utils/io/](src/utils/io/CLAUDE.md)**: Input/output operations
  - `load_files.py`: mzML and Arrow file loading
  - `read_output.py`: Result file handling
  - `load_fasta.py`: Protein sequence loading
- **[src/models/](src/models/CLAUDE.md)**: Model-related functionality
  - `spec_lib/spec_lib.py`: Spectral library handling and indexing
- **[rt_models/](rt_models/CLAUDE.md)**: Pre-trained TensorFlow models for RT prediction
  - CNN models for different labeling methods
  - Ensemble predictions for robustness
- `src/mass_tags.py`: Tag library definitions for different labeling methods (mTRAQ, diethyl, etc.)

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

## Recent Issues and Fixes

### Fragment Information Indexing (Fixed in commits cc32aa5, 6ba76c2)
After the spectral_fitting refactoring, fragment correlation calculations were returning 0 for all entries due to incorrect indexing:
- **Issue**: The adapter was using `coeff_idx` to index fragment information lists
- **Fix**: Changed to use `i` (position in non-zero coefficients list) instead
- **Files affected**: `src/spectral_fitting/adapter.py`

### Division by Zero Errors (Fixed)
Multiple division by zero errors in RT alignment:
- **bad_IDs calculation**: Added check for empty arrays before division
- **AUC calculation**: Added handling for cases with < 2 data points
- **Files affected**: `src/rt_alignment.py`

### Known Issues
- Test mode with Arrow files may select 0 MS2 scans with restrictive m/z and RT ranges
- Spline fitting may fail with too few data points
- Some RT alignment models expect specific tag types (mTRAQ, diethyl, etc.)