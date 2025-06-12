"""
Unified feature calculation for spectral fitting.

This module provides unified feature calculation that processes all candidates
together, eliminating the need to call get_features twice.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from .spectral_fitting_unified import UnifiedCandidates, UnifiedMatrixData, UnifiedFeatures
from .utils.spectral_similarity_metrics import (
    get_scribe, get_manhattan_distance, 
    gof_stat, get_residuals, max_matched_residual
)
from .utils.misc_functions import cosim, np_pearson_cor


def calculate_features_unified(
    unified_candidates: UnifiedCandidates,
    matrix_data: UnifiedMatrixData,
    additional_outputs: Dict,
    dia_spectrum: np.ndarray,
    prec_rt: float,
    lib_coefficients: np.ndarray,
    sparse_matrix,
    peak_idx_convertor: Dict[int, int],
    unique_row_idxs: np.ndarray,
    rt_mz: np.ndarray,
    window_idxs: np.ndarray,
    library: Dict,
    decoy_mz_offset: float = 20.0
) -> UnifiedFeatures:
    """
    Calculate all features for unified candidates in a single pass.
    
    This replaces calling get_features twice (once for targets, once for decoys).
    
    Args:
        unified_candidates: Unified candidates with type tracking
        matrix_data: Matrix data from unified processing
        additional_outputs: Additional data from create_entries_unified
        dia_spectrum: DIA spectrum
        prec_rt: Precursor retention time
        lib_coefficients: NNLS coefficients
        sparse_matrix: Sparse matrix from NNLS
        peak_idx_convertor: Peak index mapping
        unique_row_idxs: Unique row indices
        rt_mz: RT and m/z array for all library entries
        window_idxs: Window indices for candidates
        library: Spectral library
        decoy_mz_offset: m/z offset for decoys
        
    Returns:
        UnifiedFeatures object with all calculated features
    """
    # Extract needed data
    peaks_in_dia = unified_candidates.peaks_in_dia
    is_decoy_matched = unified_candidates.is_decoy[peaks_in_dia]
    n_candidates = len(peaks_in_dia)
    
    if n_candidates == 0:
        # Return empty features
        return UnifiedFeatures(
            features=np.zeros((0, 26)),
            is_decoy=np.array([], dtype=bool)
        )
    
    # Get data for matched candidates
    pep_cand = additional_outputs['pep_cand']
    norm_intensities = additional_outputs['norm_intensities']
    lib_peaks_matched = additional_outputs['lib_peaks_matched']
    pep_cand_list = additional_outputs['pep_cand_list']
    ms1_error = additional_outputs['ms1_error_matched']
    
    # Initialize feature arrays
    features = np.zeros((n_candidates, 26))
    
    # Get split arrays from matrix data
    spec_values_split = matrix_data.values_split
    spec_row_indices_split = matrix_data.row_indices_split
    spec_col_indices_split = matrix_data.col_indices_split
    
    # First, calculate residuals and y_pred for all candidates
    # This is needed for manhattan distance and residual features
    residuals = None
    y_pred = None
    if n_candidates > 0:
        # Need to get offsets from matrix data
        target_mask = ~is_decoy_matched
        n_targets = np.sum(target_mask)
        ref_spec_offset = 0
        decoy_spec_offset = n_targets
        
        # Prepare data for get_residuals
        all_row_indices = np.concatenate([arr for arr in spec_row_indices_split if len(arr) > 0])
        all_col_indices = np.concatenate([arr for arr in spec_col_indices_split if len(arr) > 0])
        all_values = np.concatenate([arr for arr in spec_values_split if len(arr) > 0])
        
        if len(all_row_indices) > 0:
            # Split by type
            ref_mask = all_col_indices < n_targets
            ref_sparse_row = all_row_indices[ref_mask]
            ref_sparse_col = all_col_indices[ref_mask]
            ref_sparse_val = all_values[ref_mask]
            
            decoy_sparse_row = all_row_indices[~ref_mask]
            decoy_sparse_col = all_col_indices[~ref_mask] - n_targets
            decoy_sparse_val = all_values[~ref_mask]
            
            # Calculate residuals and predictions
            residuals, y_pred = get_residuals(
                ref_sparse_val,
                ref_sparse_row,
                ref_sparse_col,
                decoy_sparse_val,
                decoy_sparse_row,
                decoy_sparse_col,
                dia_spectrum[:, 1],
                lib_coefficients,
                ref_spec_offset,
                decoy_spec_offset
            )
    
    # Calculate features for each candidate
    for i in range(n_candidates):
        candidate_idx = peaks_in_dia[i]
        is_decoy = is_decoy_matched[i]
        
        # Feature 1: Number of library peaks matched
        features[i, 0] = np.sum(lib_peaks_matched[i])
        
        # Feature 2: Fraction of library intensity matched
        features[i, 1] = np.sum(spec_values_split[i])
        
        # Feature 3: Fraction of DIA intensity matched
        if len(spec_row_indices_split[i]) > 0:
            features[i, 2] = np.sum(dia_spectrum[spec_row_indices_split[i], 1]) / np.sum(dia_spectrum[:, 1])
        
        # Feature 4: MS1 relative error
        features[i, 3] = ms1_error[i]
        
        # Feature 5: RT error
        # Different calculation for decoys
        if is_decoy:
            # Decoys use offset m/z for RT lookup
            decoy_mz = rt_mz[window_idxs[candidate_idx], 1] - decoy_mz_offset
            # For now, simplified - would need proper RT prediction for decoys
            features[i, 4] = 0  # Placeholder
        else:
            candidate_rt = rt_mz[window_idxs[candidate_idx], 0]
            features[i, 4] = prec_rt - candidate_rt
        
        # Feature 6: Fraction intensity matched
        if len(spec_values_split[i]) > 0:
            features[i, 5] = np.sum(spec_values_split[i] * lib_coefficients[i])
        
        # Feature 7: Fraction intensity predicted
        features[i, 6] = features[i, 5] * lib_coefficients[i] if i < len(lib_coefficients) else 0
        
        # Features 8-10: Correlation features (placeholder)
        features[i, 7:10] = 0  # r2all, r2_lib_spec, r2_unique
        
        # Feature 11: Fraction unique predicted
        # Requires single_matched_rows calculation
        features[i, 10] = 0  # Placeholder
        
        # Feature 12: Fraction DIA intensity predicted
        features[i, 11] = features[i, 1] * lib_coefficients[i] / features[i, 2] if features[i, 2] > 0 else 0
        
        # Feature 13-16: Hyperscore features
        # Count b and y ions if fragment names available
        if 'frag_names' in additional_outputs and i < len(additional_outputs['frag_names']):
            frag_names = additional_outputs['frag_names'][i]
            b_count = sum(1 for f in frag_names if f.startswith('b'))
            y_count = sum(1 for f in frag_names if f.startswith('y'))
            features[i, 13] = b_count  # b_counts
            features[i, 14] = y_count  # y_counts
            # Hyperscore calculation would go here
            features[i, 12] = 0  # hyperscores placeholder
            features[i, 15] = 0  # longest_y_ions placeholder
        
        # Feature 17: SCRIBE score
        if len(spec_row_indices_split[i]) > 0 and len(spec_values_split[i]) > 0:
            try:
                features[i, 16] = get_scribe(
                    spec_values_split[i],
                    dia_spectrum[:, 1],  # Full spectrum intensities
                    spec_row_indices_split[i]
                )
            except:
                features[i, 16] = 0
        
        # Features 18-19: Residuals
        if residuals is not None and len(spec_row_indices_split[i]) > 0:
            # Get residuals for this candidate's peaks
            candidate_residuals = residuals[spec_row_indices_split[i]]
            if len(candidate_residuals) > 0:
                features[i, 17] = np.max(np.abs(candidate_residuals))  # max_unmatched_residuals
                features[i, 18] = np.max(candidate_residuals)  # max_matched_residuals
        
        # Feature 20: Goodness of fit
        # Skip for now as gof_stat requires different structure
        features[i, 19] = 0
    
    # Calculate manhattan distance and spectral contrast for all candidates at once
    if residuals is not None and y_pred is not None and n_candidates > 0:
        manhattan_distances, fitted_spectral_contrasts = get_manhattan_distance(
            spec_row_indices_split,
            spec_col_indices_split,
            spec_values_split,
            dia_spectrum[:, 1],
            y_pred
        )
        features[:, 20] = manhattan_distances
        features[:, 21] = fitted_spectral_contrasts
    
    # Continue with remaining features
    for i in range(n_candidates):
        candidate_idx = peaks_in_dia[i]
        is_decoy = is_decoy_matched[i]
        
        # Features 23-24: More intensity features
        features[i, 22] = features[i, 5]  # frac_int_matched_pred
        features[i, 23] = features[i, 5] if lib_coefficients[i] > 0.1 else 0  # with significance cutoff
        
        # Feature 25: Large coefficient cosine similarity
        features[i, 24] = 0  # Placeholder
        
        # Feature 26: m/z value
        if is_decoy:
            features[i, 25] = rt_mz[window_idxs[candidate_idx], 1] - decoy_mz_offset
        else:
            features[i, 25] = rt_mz[window_idxs[candidate_idx], 1]
    
    # Define feature names
    feature_names = [
        "num_lib_peaks_matched", "frac_lib_intensity", "frac_dia_intensity",
        "rel_error", "rt_error", "frac_int_matched", "frac_int_pred",
        "r2all", "r2_lib_spec", "r2_unique", "frac_unique_pred",
        "frac_dia_intensity_pred", "hyperscores", "b_counts", "y_counts",
        "longest_y_ions", "scribe_scores", "max_unmatched_residuals",
        "max_matched_residuals", "gof_stats", "manhattan_distances",
        "fitted_spectral_contrasts", "frac_int_matched_pred",
        "frac_int_matched_pred_sigcoeff", "large_coeff_cosine", "rt_mz"
    ]
    
    return UnifiedFeatures(
        features=features,
        is_decoy=is_decoy_matched,
        feature_names=feature_names
    )


def demonstrate_unified_features():
    """Show how unified feature calculation simplifies the code."""
    print("=== Unified Feature Calculation ===\n")
    
    print("Original approach:")
    print("  1. Call get_features for targets")
    print("  2. Call get_features for decoys with offset")
    print("  3. Pass combined data for some calculations")
    print("  4. Concatenate results")
    print()
    
    print("Unified approach:")
    print("  1. Call calculate_features_unified once")
    print("  2. Type-specific logic handled internally")
    print("  3. Results already combined")
    print()
    
    print("Benefits:")
    print("  - Single pass through data")
    print("  - No offset calculations")
    print("  - Cleaner feature logic")
    print("  - Easier to add new features")
    

if __name__ == "__main__":
    demonstrate_unified_features()