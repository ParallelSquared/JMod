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
  - `sparse_nnls.py`: Mathematical solver for spectral deconvolution
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

## Phase 4: Unified Spectral Fitting Cleanup Implementation Plan

### Overview
Phase 4 completes the unified spectral fitting refactoring by removing the original implementation and consolidating all code to use the unified approach. This phase should be implemented carefully to ensure no functionality is lost.

### Prerequisites
- Ensure Phase 3 is fully tested and working
- Create a backup branch before starting Phase 4
- Run full test suite and save results for comparison

### Step-by-Step Implementation

#### Step 1: Create Backup and Branch
```bash
# Create backup tag
git tag pre-phase4-backup

# Create new branch for Phase 4
git checkout -b phase4-cleanup
```

#### Step 2: Remove Original Functions (spectral_fitting.py)
1. Delete `fit_to_lib2_original` function
2. Delete the original `create_entries` function
3. Remove any helper functions only used by the original implementation
4. Keep only the unified versions

#### Step 3: Consolidate Unified Modules
1. **Merge spectral_fitting_unified.py into spectral_fitting.py**
   - Move `UnifiedCandidates`, `UnifiedMatrixData`, `UnifiedFeatures` classes
   - Move `create_unified_candidates` function
   - Update imports throughout codebase

2. **Merge spectral_fitting_unified_features.py**
   - Move unified feature calculation functions into main feature module
   - Remove "_unified" suffix from function names
   - Update all callers

3. **Merge spectral_fitting_unified_matrix.py**
   - Integrate matrix operations into main spectral fitting module
   - Consolidate sparse matrix construction code

4. **Remove spectral_fitting_unified_integration.py**
   - This was a demo file and is no longer needed

#### Step 4: Rename and Simplify Functions
1. Remove "_unified" suffix from all function names
2. Update all function calls throughout the codebase
3. Ensure consistent naming conventions

#### Step 5: Update Imports and Dependencies
1. Search and replace all imports of unified modules
2. Remove imports of deleted modules
3. Update __init__.py files if needed

#### Step 6: Clean Up Configuration
1. Remove any config options related to choosing between original/unified
2. Remove unused configuration parameters
3. Update config documentation

#### Step 7: Update Documentation
1. Update all docstrings to reflect unified approach
2. Remove references to "ref" and "decoy" separation
3. Update this CLAUDE.md file with new architecture
4. Update README if it references the old approach

#### Step 8: Test Suite Updates
1. Update tests that explicitly test ref/decoy separation
2. Add tests for unified data structure integrity
3. Ensure all existing tests pass
4. Add performance benchmarks

#### Step 9: Code Optimization
1. Profile the unified implementation
2. Identify any performance bottlenecks
3. Optimize memory usage where possible
4. Consider parallelization opportunities

#### Step 10: Final Validation
1. Run full test suite
2. Compare results with pre-Phase 4 backup
3. Test with multiple datasets (mTRAQ, LFQ, etc.)
4. Benchmark performance improvements
5. Check memory usage

### Specific Files to Modify

#### Files to Delete:
- `src/spectral_fitting_unified.py`
- `src/spectral_fitting_unified_features.py`
- `src/spectral_fitting_unified_matrix.py`
- `src/spectral_fitting_unified_integration.py`

#### Files to Update:
- `src/spectral_fitting.py` - Main consolidation target
- `src/rt_alignment.py` - Update function calls
- `src/run_jmod.py` - Update imports
- `src/config.py` - Remove obsolete options
- Tests that reference old functions

### Testing Checklist
- [ ] All unit tests pass
- [ ] Integration tests pass
- [ ] Performance is equal or better
- [ ] Memory usage is reasonable
- [ ] Results match Phase 3 output
- [ ] Works with all tag types (mTRAQ, diethyl, etc.)
- [ ] Works with isotope patterns
- [ ] FDR analysis produces same results

### Rollback Plan
If issues are encountered:
1. `git checkout main`
2. `git checkout pre-phase4-backup`
3. Document specific issues encountered
4. Create targeted fixes before retry

### Success Metrics
- Code reduction of ~40%
- Improved maintainability
- Equal or better performance
- All tests passing
- Consistent results with previous implementation