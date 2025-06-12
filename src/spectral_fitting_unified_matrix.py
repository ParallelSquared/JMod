"""
Unified matrix operations for spectral fitting.

This module provides unified matrix construction that handles targets and decoys
together, eliminating the need for separate processing and manual offset calculations.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from scipy import sparse, stats
from .spectral_fitting_unified import UnifiedCandidates, UnifiedMatrixData


def unmatched_peaks_unified(
    unified_candidates: UnifiedCandidates,
    norm_intensities: List[np.ndarray],
    pep_cand_loc: List[np.ndarray],
    last_row: int,
    fit_type: str = "a",
    lower_limit: float = 1e-10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate unmatched peaks for all candidates in a unified manner.
    
    This replaces calling unmatched_peaks twice (once for targets, once for decoys).
    
    Args:
        unified_candidates: Unified candidates with type tracking
        norm_intensities: Normalized intensities for matched candidates
        pep_cand_loc: Peak locations for matched candidates
        last_row: Last row index in the matrix
        fit_type: How to fit unmatched peaks ('a', 'b', or 'c')
        lower_limit: Minimum intensity threshold for type 'c'
        
    Returns:
        Tuple of (row_indices, col_indices, values, is_decoy) for unmatched peaks
    """
    assert fit_type in ["a", "b", "c"]
    
    n_candidates = len(pep_cand_loc)
    
    if fit_type == "a":
        # All unmatched peaks go to a single zero-intensity row
        not_dia_col_indices = np.arange(n_candidates)
        not_dia_row_indices = np.array([last_row] * n_candidates, dtype=int)
        not_dia_values = np.array([
            np.sum([norm_intensities[idx][peak_idx] 
                    for peak_idx in range(len(norm_intensities[idx])) 
                    if pep_cand_loc[idx][peak_idx] % 2 == 0])
            for idx in range(n_candidates)
        ])
        
    elif fit_type == "b":
        # Each candidate gets its own zero-intensity row
        not_dia_col_indices = np.arange(n_candidates)
        not_dia_row_indices = np.array([last_row + 1 + idx for idx in range(n_candidates)], dtype=int)
        not_dia_values = np.array([
            np.sum([norm_intensities[idx][peak_idx] 
                    for peak_idx in range(len(norm_intensities[idx])) 
                    if pep_cand_loc[idx][peak_idx] % 2 == 0])
            for idx in range(n_candidates)
        ])
        
    elif fit_type == "c":
        # Each unmatched peak gets its own row
        all_unmatched_peaks = [
            [norm_intensities[idx][peak_idx] 
             for peak_idx in range(len(norm_intensities[idx])) 
             if pep_cand_loc[idx][peak_idx] % 2 == 0 and 
                norm_intensities[idx][peak_idx] > lower_limit]
            for idx in range(n_candidates)
        ]
        num_unmatched_to_fit = [len(i) for i in all_unmatched_peaks]
        not_dia_col_indices = np.concatenate([[idx] * n for idx, n in enumerate(num_unmatched_to_fit)])
        not_dia_row_indices = np.arange(np.sum(num_unmatched_to_fit)) + last_row + 1
        not_dia_values = np.concatenate(all_unmatched_peaks)
    
    # Track which entries are decoys based on matched candidates
    is_decoy_matched = unified_candidates.is_decoy[unified_candidates.peaks_in_dia]
    
    # Create is_decoy array for unmatched peaks
    if fit_type in ["a", "b"]:
        not_dia_is_decoy = is_decoy_matched
    else:  # type "c"
        not_dia_is_decoy = np.concatenate([
            [is_decoy_matched[idx]] * n 
            for idx, n in enumerate(num_unmatched_to_fit)
        ])
    
    return not_dia_row_indices, not_dia_col_indices, not_dia_values, not_dia_is_decoy


def build_sparse_matrix_unified(
    matrix_data: UnifiedMatrixData,
    unmatched_row_indices: np.ndarray,
    unmatched_col_indices: np.ndarray,
    unmatched_values: np.ndarray,
    dia_spectrum: np.ndarray,
    unique_row_idxs: np.ndarray
) -> Tuple[sparse.coo_matrix, np.ndarray, Dict[int, int]]:
    """
    Build sparse matrix for NNLS optimization with unified data.
    
    Args:
        matrix_data: Unified matrix data with matched peaks
        unmatched_row_indices: Row indices for unmatched peaks
        unmatched_col_indices: Column indices for unmatched peaks
        unmatched_values: Values for unmatched peaks
        dia_spectrum: DIA spectrum
        unique_row_idxs: Unique row indices from matched peaks
        
    Returns:
        Tuple of (sparse_matrix, target_vector, peak_idx_convertor)
    """
    # Combine matched and unmatched peaks
    all_row_indices = np.concatenate([matrix_data.row_indices, unmatched_row_indices])
    all_col_indices = np.concatenate([matrix_data.col_indices, unmatched_col_indices])
    all_values = np.concatenate([matrix_data.values, unmatched_values])
    
    # Rank rows to remove gaps
    new_row_indices = stats.rankdata(all_row_indices, method="dense").astype(int) - 1
    peak_idx_convertor = {old: new for old, new in zip(all_row_indices, new_row_indices)}
    
    # Create sparse matrix
    sparse_lib_matrix = sparse.coo_matrix(
        (all_values, (new_row_indices, all_col_indices))
    )
    
    # Create target vector
    dia_spec_int = dia_spectrum[unique_row_idxs, 1]
    # Pad with zeros for unmatched peak rows
    n_extra_rows = sparse_lib_matrix.shape[0] - len(dia_spec_int)
    dia_spec_int = np.append(dia_spec_int, [0] * n_extra_rows)
    
    return sparse_lib_matrix, dia_spec_int, peak_idx_convertor


def process_matrix_unified(
    unified_candidates: UnifiedCandidates,
    matrix_data: UnifiedMatrixData,
    additional_outputs: Dict,
    dia_spectrum: np.ndarray,
    unmatched_fit_type: str = "a"
) -> Dict[str, any]:
    """
    Complete matrix processing pipeline with unified data.
    
    This replaces the entire matrix construction section of fit_to_lib2.
    
    Args:
        unified_candidates: Unified candidates
        matrix_data: Initial matrix data from create_entries_unified
        additional_outputs: Additional data from create_entries_unified
        dia_spectrum: DIA spectrum
        unmatched_fit_type: How to handle unmatched peaks
        
    Returns:
        Dictionary with:
        - sparse_matrix: Sparse matrix for NNLS
        - target_vector: Target intensity vector
        - peak_idx_convertor: Mapping of peak indices
        - lib_coefficients: NNLS solution
        - unique_row_idxs: Unique row indices
    """
    # Early exit if no matches
    if len(matrix_data.row_indices) == 0:
        return {
            'sparse_matrix': sparse.coo_matrix((0, 0)),
            'target_vector': np.array([]),
            'peak_idx_convertor': {},
            'lib_coefficients': np.array([]),
            'unique_row_idxs': np.array([])
        }
    
    # Get unique row indices
    unique_row_idxs = np.unique(matrix_data.row_indices)
    unique_row_idxs = np.sort(unique_row_idxs).astype(int)
    
    # Calculate unmatched peaks for all candidates
    last_row = max(unique_row_idxs)
    unmatched_row_idx, unmatched_col_idx, unmatched_vals, _ = unmatched_peaks_unified(
        unified_candidates=unified_candidates,
        norm_intensities=additional_outputs['norm_intensities'],
        pep_cand_loc=additional_outputs['pep_cand_loc'],
        last_row=last_row,
        fit_type=unmatched_fit_type
    )
    
    # Build sparse matrix
    sparse_matrix, target_vector, peak_idx_convertor = build_sparse_matrix_unified(
        matrix_data=matrix_data,
        unmatched_row_indices=unmatched_row_idx,
        unmatched_col_indices=unmatched_col_idx,
        unmatched_values=unmatched_vals,
        dia_spectrum=dia_spectrum,
        unique_row_idxs=unique_row_idxs
    )
    
    # Solve NNLS
    try:
        import ptinnls as sparse_nnls
    except ImportError:
        # Fallback to scipy if ptinnls not available
        from scipy.optimize import nnls
        lib_coefficients, _ = nnls(sparse_matrix.toarray(), target_vector)
    else:
        fit_results = sparse_nnls.lsqnonneg(
            sparse_matrix, 
            target_vector, 
            {"show_progress": False}
        )
        lib_coefficients = fit_results['x']
    
    return {
        'sparse_matrix': sparse_matrix,
        'target_vector': target_vector,
        'peak_idx_convertor': peak_idx_convertor,
        'lib_coefficients': lib_coefficients,
        'unique_row_idxs': unique_row_idxs
    }


def demonstrate_unified_matrix():
    """Show how unified matrix operations simplify the code."""
    print("=== Unified Matrix Operations ===\n")
    
    print("Original approach:")
    print("  1. Calculate unmatched peaks for targets")
    print("  2. Calculate unmatched peaks for decoys") 
    print("  3. Manually calculate decoy_col_offset")
    print("  4. Append indices with offset")
    print("  5. Concatenate everything")
    print("  6. Build matrix")
    print()
    
    print("Unified approach:")
    print("  1. Calculate unmatched peaks once for all")
    print("  2. Build matrix directly")
    print("  3. No offset calculations needed")
    print()
    
    print("Code reduction example:")
    print("  OLD: ~50 lines for matrix construction")
    print("  NEW: ~10 lines with unified approach")
    

if __name__ == "__main__":
    demonstrate_unified_matrix()