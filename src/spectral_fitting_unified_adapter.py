"""
Adapter functions to integrate unified spectral fitting with existing code.

This module provides functions to bridge between the existing separate target/decoy
processing and the new unified approach.
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from .spectral_fitting_unified import (
    UnifiedCandidates, UnifiedMatrixData, UnifiedFeatures,
    create_unified_candidates, create_entries_unified
)


def adapt_create_entries_to_unified(
    centroid_breaks: np.ndarray,
    candidate_peaks: List[np.ndarray],
    mass_window_candidates: List[Tuple],
    top_n: int = 10,
    atleast_m: int = 3,
    prec_mzs: Optional[np.ndarray] = None,
    ms1_spec: Optional[Any] = None,
    ms1_tol: float = 25.,
    spec_frags: Optional[List] = None,
    top_n_idxs: Optional[List[np.ndarray]] = None,
    decoy: bool = False,
    decoy_candidates: Optional[List[Tuple]] = None,
    decoy_peaks: Optional[List[np.ndarray]] = None,
    decoy_mzs: Optional[np.ndarray] = None,
    decoy_top_n_idxs: Optional[List[np.ndarray]] = None
) -> Dict[str, Any]:
    """
    Adapter that uses unified processing but returns results in original format.
    
    This function demonstrates how to use the unified approach while maintaining
    backward compatibility with existing code that expects separate target/decoy results.
    
    Args:
        All arguments from original create_entries plus decoy-specific ones
        
    Returns:
        Dictionary with separate target and decoy results matching original format
    """
    # Create unified candidates
    if decoy and decoy_candidates is not None:
        unified = create_unified_candidates(
            target_candidates=mass_window_candidates,
            target_peaks=candidate_peaks,
            decoy_candidates=decoy_candidates,
            decoy_peaks=decoy_peaks
        )
        
        # Combine precursor m/z values
        if decoy_mzs is not None:
            all_prec_mzs = np.concatenate([prec_mzs, decoy_mzs])
        else:
            all_prec_mzs = prec_mzs
            
        # Combine top_n indices if provided
        if top_n_idxs is not None and decoy_top_n_idxs is not None:
            all_top_n_idxs = top_n_idxs + decoy_top_n_idxs
        else:
            all_top_n_idxs = top_n_idxs
    else:
        # No decoys
        unified = create_unified_candidates(
            target_candidates=mass_window_candidates,
            target_peaks=candidate_peaks
        )
        all_prec_mzs = prec_mzs
        all_top_n_idxs = top_n_idxs
    
    # Process with unified function
    updated_unified, matrix_data, additional_outputs = create_entries_unified(
        centroid_breaks=centroid_breaks,
        unified_candidates=unified,
        top_n=top_n,
        atleast_m=atleast_m,
        prec_mzs=all_prec_mzs,
        ms1_spec=ms1_spec,
        ms1_tol=ms1_tol,
        spec_frags=spec_frags,
        top_n_idxs=all_top_n_idxs
    )
    
    # Split results back into targets and decoys
    results = {'target': {}, 'decoy': {}}
    
    # Get indices for targets and decoys in the matched set
    if len(updated_unified.peaks_in_dia) > 0:
        matched_is_decoy = updated_unified.is_decoy[updated_unified.peaks_in_dia]
        target_matched_idx = np.where(~matched_is_decoy)[0]
        decoy_matched_idx = np.where(matched_is_decoy)[0]
        
        # Split matrix data
        if len(target_matched_idx) > 0:
            results['target'] = {
                'peaks_in_dia': [updated_unified.peaks_in_dia[i] for i in target_matched_idx],
                'pep_cand': [additional_outputs['pep_cand'][i] for i in target_matched_idx],
                'pep_cand_loc': [additional_outputs['pep_cand_loc'][i] for i in target_matched_idx],
                'pep_cand_list': [additional_outputs['pep_cand_list'][i] for i in target_matched_idx],
                'spec_row_indices_split': [matrix_data.row_indices_split[i] for i in target_matched_idx],
                'spec_col_indices_split': [matrix_data.col_indices_split[i] for i in target_matched_idx],
                'spec_values_split': [matrix_data.values_split[i] for i in target_matched_idx],
                'norm_intensities': [additional_outputs['norm_intensities'][i] for i in target_matched_idx],
                'lib_peaks_matched': [additional_outputs['lib_peaks_matched'][i] for i in target_matched_idx],
                'ms1_error': additional_outputs['ms1_error_matched'][target_matched_idx]
            }
        
        if decoy and len(decoy_matched_idx) > 0:
            # Adjust column indices for decoys
            decoy_col_offset = len(target_matched_idx)
            results['decoy'] = {
                'peaks_in_dia': [updated_unified.peaks_in_dia[i] for i in decoy_matched_idx],
                'pep_cand': [additional_outputs['pep_cand'][i] for i in decoy_matched_idx],
                'pep_cand_loc': [additional_outputs['pep_cand_loc'][i] for i in decoy_matched_idx],
                'pep_cand_list': [additional_outputs['pep_cand_list'][i] for i in decoy_matched_idx],
                'spec_row_indices_split': [matrix_data.row_indices_split[i] for i in decoy_matched_idx],
                'spec_col_indices_split': [
                    col_idx - decoy_col_offset 
                    for i in decoy_matched_idx
                    for col_idx in [matrix_data.col_indices_split[i]]
                ],
                'spec_values_split': [matrix_data.values_split[i] for i in decoy_matched_idx],
                'norm_intensities': [additional_outputs['norm_intensities'][i] for i in decoy_matched_idx],
                'lib_peaks_matched': [additional_outputs['lib_peaks_matched'][i] for i in decoy_matched_idx],
                'ms1_error': additional_outputs['ms1_error_matched'][decoy_matched_idx]
            }
    
    return results


def demonstrate_unified_processing():
    """
    Example showing how unified processing simplifies the code.
    
    This demonstrates the benefits of unified processing:
    1. Single function call instead of two
    2. No duplicate code paths
    3. Automatic handling of mixed data
    4. Easier feature calculation
    """
    # Example: Original approach (pseudocode)
    print("Original approach:")
    print("  1. Call create_entries for targets")
    print("  2. Call create_entries for decoys")
    print("  3. Manage offsets manually")
    print("  4. Concatenate results")
    print("  5. Calculate features twice")
    print()
    
    # Example: Unified approach
    print("Unified approach:")
    print("  1. Create unified candidates")
    print("  2. Call create_entries_unified once")
    print("  3. Offsets handled automatically")
    print("  4. Results already combined")
    print("  5. Calculate features once")
    print()
    
    # Show data structure
    print("Unified data structure example:")
    print("  candidates = ['PEPTIDE1', 'PEPTIDE2', 'Decoy_PEPTIDE3', 'Decoy_PEPTIDE4']")
    print("  is_decoy = [False, False, True, True]")
    print("  - Automatically tracks type")
    print("  - No need for separate variables")
    print("  - Easy filtering when needed")


def convert_existing_to_unified(
    ref_pep_cand: List[Tuple],
    ref_peaks: List[np.ndarray],
    ref_ms1_error: np.ndarray,
    decoy_pep_cand: Optional[List[Tuple]] = None,
    decoy_peaks: Optional[List[np.ndarray]] = None,
    decoy_ms1_error: Optional[np.ndarray] = None
) -> UnifiedCandidates:
    """
    Convert existing separate target/decoy data to unified format.
    
    Args:
        ref_pep_cand: Target candidates
        ref_peaks: Target peak data
        ref_ms1_error: Target MS1 errors
        decoy_pep_cand: Optional decoy candidates
        decoy_peaks: Optional decoy peak data
        decoy_ms1_error: Optional decoy MS1 errors
        
    Returns:
        UnifiedCandidates object
    """
    return create_unified_candidates(
        target_candidates=ref_pep_cand,
        target_peaks=ref_peaks,
        decoy_candidates=decoy_pep_cand,
        decoy_peaks=decoy_peaks,
        target_ms1_error=ref_ms1_error,
        decoy_ms1_error=decoy_ms1_error
    )


def unified_get_features(
    unified_data: UnifiedCandidates,
    matrix_data: UnifiedMatrixData,
    dia_spectrum: np.ndarray,
    prec_rt: float,
    lib_coefficients: np.ndarray,
    # ... other feature calculation parameters
) -> UnifiedFeatures:
    """
    Calculate features for all candidates in a unified manner.
    
    This replaces calling get_features twice (once for targets, once for decoys).
    
    Args:
        unified_data: Unified candidates
        matrix_data: Unified matrix data
        dia_spectrum: DIA spectrum
        prec_rt: Precursor retention time
        lib_coefficients: NNLS coefficients
        
    Returns:
        UnifiedFeatures object with features for all candidates
    """
    # Placeholder - would implement actual feature calculation
    # Key point: Calculate features once for all candidates
    # Use is_decoy array to apply type-specific logic where needed
    
    n_candidates = len(unified_data.peaks_in_dia)
    n_features = 26  # Standard feature count
    
    features = np.zeros((n_candidates, n_features))
    
    # Example: RT error calculation differs for decoys
    for i, idx in enumerate(unified_data.peaks_in_dia):
        if unified_data.is_decoy[idx]:
            # Decoy-specific RT calculation
            features[i, 4] = 0  # Placeholder
        else:
            # Target RT calculation
            features[i, 4] = prec_rt - 0  # Placeholder
    
    return UnifiedFeatures(
        features=features,
        is_decoy=unified_data.is_decoy[unified_data.peaks_in_dia],
        feature_names=['feature_' + str(i) for i in range(n_features)]
    )


if __name__ == "__main__":
    # Demonstrate the concept
    demonstrate_unified_processing()