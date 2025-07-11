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
- Coverage reporting integrated (currently ~310 tests passing, 11 failing)
- Focus on unit tests for utility functions with comprehensive integration testing

## Unified Library and Spectral Fitting Architecture

### Overview
JMod now uses a unified approach that combines target and decoy peptides from the point of library construction through spectral fitting. This architecture eliminates the need for separate decoy libraries and processes all peptides together in single data structures.

### Library Unification Details

#### Critical Implementation Order
1. Load library WITHOUT decoys initially (`create_decoys=False`)
2. Generate isotopes for TARGET library only
3. Add decoys AFTER isotope generation using `add_decoys_to_library()`
4. Update rt_mz and all_keys arrays to include decoys

This order prevents parsing errors during isotope generation and ensures proper decoy inclusion in spectral matching.

#### Library Entry Structure
Each entry now includes:
- `is_decoy`: Boolean flag indicating decoy status
- `parent_key`: Link from decoys to their parent targets
- All original fields (spectrum, fragments, RT, etc.)

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

### Project Status
- **Unified Library Architecture**: ✅ Complete
- **Test Coverage**: 310 tests passing, 11 failing (mainly legacy unified tests)
- **Performance**: Stable performance with feature extraction optimizations
- **Library Unification**: Targets and decoys processed together from initial load
- **Integration Testing**: Working with 25,892 IDs in recent test runs

## Spectral Fitting Refactoring

### Completed Refactoring Phases
- **Phase 1-4 ✅**: Successfully extracted helper functions for modular processing
- **Phase 5**: Unification attempted but reverted due to MS1 peak checking issues
- **Post-Phase 4 Cleanup ✅**: Codebase cleaned up and stabilized

### Current State (Post-Cleanup)
- `fit_to_lib`: RT alignment function with modular helper functions
- `fit_to_lib2`: Main spectral fitting with integrated target/decoy processing
- **Feature Extraction System**: Fully implemented modular system in `src/features/`
- **Stable Integration**: 25,892 IDs processed successfully in recent tests
- **Code Quality**: Improved organization with maintained backward compatibility

### Refactored fit_to_lib Structure
The `fit_to_lib` function has been refactored to use a collection of helper functions:

#### Helper Functions Used:
1. **get_closest_ms1**: Find the closest MS1 spectrum (shared with fit_to_lib2)
2. **filter_candidates_by_window**: Filter candidates by mass window
3. **preprocess_dia_spectrum**: Preprocess DIA spectrum peaks
4. **check_ms1_peaks**: Check for MS1 peaks in candidates
5. **filter_candidates_by_peak_matching**: Filter candidates by peak matching criteria
6. **extract_fragment_information**: Extract fragment-level information (NEW)
7. **build_sparse_matrix_simple**: Build sparse matrix for NNLS
8. **solve_nnls_simple**: Solve NNLS optimization (NEW)
9. **calculate_rt_alignment_features**: Calculate features for RT alignment
10. **format_rt_alignment_output**: Format output for RT alignment (NEW)

#### Benefits of Refactoring:
- Reduced function size from ~245 lines to ~90 lines
- Improved code organization and readability
- Increased code reuse between fit_to_lib and fit_to_lib2
- Easier testing of individual components
- Better separation of concerns

### Key Differences to Preserve
1. **Decoy Handling**: 
   - `fit_to_lib` filters decoys at line 1501: `[key for key in mass_window_candidates_all if not library[key].get('is_decoy', False)]`
   - `fit_to_lib2` processes all candidates together

2. **MS1 Peak Matching**:
   - `fit_to_lib` uses `rt_mz[window_idxs,1]` for MS1 matching
   - Critical for RT alignment accuracy

3. **Feature Calculation**:
   - `fit_to_lib` calculates simplified feature set for RT alignment
   - `fit_to_lib2` calculates full 26-feature set for FDR analysis

### Refactoring Guidelines
1. **Preserve Exact Behavior**: Any extracted function must produce identical outputs
2. **Test Each Phase**: Validate outputs after each extraction
3. **Use A/B Testing**: Compare original vs refactored outputs element-by-element
4. **Maintain Performance**: No degradation in speed or memory usage

### Testing Requirements
- Unit tests for each extracted function
- Integration tests comparing full pipeline outputs
- Regression tests with saved baseline outputs
- RT alignment validation with real data

See `refactoring_plan.md` for detailed phase-by-phase approach.

## Feature Extraction Architecture

### Overview
The feature extraction system has been refactored into a modular architecture located in `src/features/`. This improves code organization, testability, and maintainability.

### Feature Modules
1. **intensity_features.py**: Intensity-based features (0-6, 22-25)
   - Number of peaks matched
   - Fraction of library/DIA intensity
   - Intensity predictions

2. **error_features.py**: Error and residual features (3-4, 17-18)
   - MS1 and RT errors
   - Residual calculations

3. **correlation_features.py**: Correlation and R² features (7-11)
   - R² calculations between observed/predicted
   - Unique peak analysis

4. **fragment_features.py**: Fragment ion features (13-16)
   - B/Y ion counts
   - Longest Y-ion series
   - Hyperscore calculations

5. **scoring_features.py**: Scoring metrics (12, 16-21)
   - SCRIBE scores
   - Manhattan distance
   - Goodness-of-fit statistics

6. **feature_calculator.py**: Unified interface
   - Coordinates all feature modules
   - Provides clean API for spectral_fitting.py
   - Handles data flow between modules

### Benefits
- **Modularity**: Each feature type in its own module
- **Testability**: Individual feature functions can be tested in isolation
- **Maintainability**: Clear separation of concerns
- **Performance**: No overhead from modularization
- **Documentation**: Each module has clear purpose and interface

### Usage
The main entry point is through `FeatureCalculator` class:
```python
from src.features.feature_calculator import FeatureCalculator, FeatureCalculatorInputs

calculator = FeatureCalculator()
features = calculator.calculate_all_features(inputs)
```

The modular feature architecture is fully functional with comprehensive testing.

### Important Implementation Notes
- Features that use lib_coefficients must use loops instead of vectorized operations to avoid numpy broadcasting errors
- Features 6 and 11 specifically require element-wise operations with proper bounds checking
- The original code used `lib_coefficients[i]` as scalars in loops, which must be preserved
- Recent fixes address large coefficient feature calculations and broadcasting issues

## Current Development Status

### Recent Achievements
- **Library Unification**: Successfully unified target/decoy processing from library load
- **Feature Extraction**: Modular system with 12+ specialized feature modules
- **Performance Optimization**: Resolved broadcasting errors in coefficient calculations
- **Integration Testing**: Successfully processing 25,892 IDs with unified architecture
- **Code Cleanup**: Post-phase4 cleanup completed with stable codebase

### Known Issues
- 11 failing tests (mainly legacy unified spectral fitting tests)
- Some parse_peptides test failures related to modification handling
- Fragment processing tests need updates for new decoy handling

### Next Priority Areas
1. **Test Stabilization**: Fix remaining 11 failing tests
2. **Performance Monitoring**: Ensure consistent performance with large datasets
3. **Documentation Updates**: Sync inline documentation with recent changes
4. **Edge Case Handling**: Improve robustness for unusual input scenarios