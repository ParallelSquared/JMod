feat: complete CSC migration for penalty rows and residual features
fix: initialize unique_row_idxs default value to prevent NameError
fix: resolve test failures for RT alignment features and fragment info functions
feat: implement build_sparse_matrix_direct to eliminate split array dependencies
feat: implement calculate_frac_dia_intensity_csc for CSC sparse matrix operations
feat: implement gof_stat_csc for CSC sparse matrix operations
feat: implement CSC sparse matrix version of get_manhattan_distance for RT alignment
feat: implement CSC sparse matrix version of get_residuals for RT alignment
fix: resolve cvxopt matrix broadcasting issues in feature calculation
feat: add CSC sparse matrix version of get_scribe for RT alignment
chore: mark stable checkpoint before feature removal
refactor: split fragment scoring into separate functions
feat: vectorize frac_dia_intensity_pred calculation
feat: add sparse matrix-based unique peak analysis
perf: pre-allocate array in calculate_r2_lib_spec_sparse
fix: return NaN for constant values in calculate_r2_lib_spec_sparse
feat: replace split arrays with sparse matrix operations in calculate_rt_alignment_features
Integration test working. 25892 ID's
fix: simplify large coefficient features to match original implementation
docs: add important notes about feature calculation implementation
fix: use loop for feature 11 calculation to avoid broadcasting error
fix: use loop in calculate_fraction_intensity_predicted to avoid broadcasting error
docs: update CLAUDE.md with feature extraction architecture details
feat: add modular feature extraction system
refactor: improve spectral_fitting.py architecture and organization
chore: clean up unused code in spectral_fitting.py
refactor: improve fit_to_lib structure with modular helper functions
docs: update CLAUDE.md to note Phase 5 was attempted and reverted
refactor: complete Phase 4 - extract feature calculation logic
fix: add sparse indices to build_sparse_matrix_simple return values
fix: restore missing variable definitions in fit_to_lib
refactor: complete Phase 3 - extract matrix building logic for fit_to_lib
refactor(phase2): extract filter_candidates_by_peak_matching function
refactor(phase1): extract check_ms1_peaks function from fit_to_lib
docs: add comprehensive refactoring plan for fit_to_lib unification
feat: phase 5 - add integration tests and improve fit_to_lib2 documentation
feat: phase 4 - extract output formatting functions from fit_to_lib2
refactor: extract separate_library_candidates function from fit_to_lib2
refactor: extract filter_candidates_by_window function from fit_to_lib2
refactor: extract preprocess_dia_spectrum function from fit_to_lib2
docs: update CLAUDE.md with unified library architecture details
feat: complete unified library architecture implementation
fix: update rt_mz and all_keys after adding decoys to include them in spectral matching
fix: properly order isotope generation before decoy creation
fix: resolve 'Not a valid modX sequence: Deco' error by generating isotopes before adding decoys
docs: add Phase 4 next steps and deployment roadmap
docs: add Phase 4 Part 6 and final project summaries
feat(tests): add comprehensive error handling and robustness tests
feat(tests): add RT alignment compatibility and performance tests
feat(tests): add comprehensive tests for calculate_features and fragment processing
Update hotfix summary with second fix
Fix fit_to_lib to use imported functions
Document Phase 4 hotfix for name collision issue
Fix name collision issues from Phase 4 Part 3
Phase 4 Part 3 Step 8: Final documentation
Phase 4 Part 3 Step 6: Update CLAUDE.md
Phase 4 Part 3 Step 5: Performance optimizations
Phase 4 Part 3 Step 4: Optimize code structure
Phase 4 Part 3 Step 3: Clean up documentation
Phase 4 Part 3 Step 2: Remove _v2 suffix from functions
Phase 4 Part 3 Step 1: Remove remaining old functions
Add Phase 4 Part 2 completion summary
Phase 4 Part 2 Step 7: Delete unified module files
Phase 4 Part 2 Step 6: Clean up and rename functions
Phase 4 Part 2 Step 5: Remove dead code
Phase 4 Part 2 Steps 2-3: Consolidate feature and matrix functions
Phase 4 Part 2 Step 1: Consolidate core data structures
Phase 4 Part 1: Create backup, migration map, and validation script
Add Phase 4 implementation plan to CLAUDE.md
Complete Phase 3 of unified spectral fitting implementation
Comment out temporary debug statements
Handle empty match results in RT alignment
Fix RT alignment handling of empty output from unified spectral fitting
Fix missing fragment data in unified spectral fitting output
Add debugging to identify CSV writing issue
Add debugging and fix potential file append issue
Fix output format mismatch causing FDR analysis error
Fix TypeError in sparse matrix construction - ensure integer indices
Remove decoy-specific logic from feature calculation
Create unified spectral similarity functions to avoid ref/decoy separation
Fix spectral similarity metrics integration in unified features
Fix calculate_features_unified parameter error - remove decoy param
saving progress...
remove sparse_nnls.py
test files...
correct test library in default config
tests for mzML io
new spectrum file class structure
Fixed bug in get max matched residuals and spectral contrast. Working on type system for Spectrum/SpectrumFile
added docstrings for spectral simmilarity metric functions in spectral_fitting.py. Moved these into utils/spectral_simmilarity_metrics.py
First commit. Branch to document methods in 'src/utils'