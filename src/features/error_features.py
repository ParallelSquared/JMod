"""
Error and accuracy features for spectral matching.

This module contains functions to calculate features related to
mass errors and retention time errors.
"""

import numpy as np
from typing import List, Optional, Any, Tuple
from scipy import sparse


def calculate_ms1_error(
    ms1_error_array: np.ndarray,
    peaks_in_dia: List[int]
) -> np.ndarray:
    """
    Extract MS1 relative error for matched candidates.
    
    Args:
        ms1_error_array: Array of MS1 errors for all candidates
        peaks_in_dia: Indices of candidates that passed filtering
        
    Returns:
        Array of MS1 errors for matched candidates
    """
    if ms1_error_array is None or len(peaks_in_dia) == 0:
        return np.zeros(len(peaks_in_dia))
    
    return ms1_error_array[peaks_in_dia]


def calculate_rt_error(
    prec_rt: float,
    rt_mz: np.ndarray,
    window_idxs: np.ndarray,
    peaks_in_dia: List[int]
) -> np.ndarray:
    """
    Calculate retention time error for matched candidates.
    
    Args:
        prec_rt: Precursor retention time
        rt_mz: Array with RT in column 0 and m/z in column 1
        window_idxs: Window indices for candidates
        peaks_in_dia: Indices of matched candidates
        
    Returns:
        Array of RT errors (observed - library)
    """
    rt_errors = []
    for candidate_idx in peaks_in_dia:
        if candidate_idx < len(window_idxs):
            library_rt = rt_mz[window_idxs[candidate_idx], 0]
            rt_errors.append(prec_rt - library_rt)
        else:
            rt_errors.append(0.0)  # Default for out of bounds
    
    return np.array(rt_errors)


def calculate_fragment_errors(
    bin_centers: np.ndarray,
    spec_row_indices_split: List[np.ndarray],
    lib_frag_mz: List[np.ndarray]
) -> List[np.ndarray]:
    """
    Calculate mass errors for matched fragments.
    
    Args:
        bin_centers: Center m/z values for DIA bins
        spec_row_indices_split: Row indices for each candidate
        lib_frag_mz: Library fragment m/z values
        
    Returns:
        List of error arrays for each candidate
    """
    frag_errors = []
    
    for i, (row_indices, lib_mz) in enumerate(zip(spec_row_indices_split, lib_frag_mz)):
        if len(row_indices) > 0 and len(lib_mz) > 0:
            matched_bin_centers = bin_centers[row_indices]
            errors = (matched_bin_centers - lib_mz) / matched_bin_centers
            frag_errors.append(errors)
        else:
            frag_errors.append(np.array([]))
    
    return frag_errors


def calculate_residual_features(
    residuals: np.ndarray,
    spec_row_indices_split: List[np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate residual-based features.
    
    Args:
        residuals: Array of residuals from NNLS fit
        spec_row_indices_split: Row indices for each candidate
        
    Returns:
        Tuple of (max_unmatched_residuals, max_matched_residuals)
    """
    n_candidates = len(spec_row_indices_split)
    max_unmatched = np.zeros(n_candidates)
    max_matched = np.zeros(n_candidates)
    
    if residuals is None:
        return max_unmatched, max_matched
    
    for i, row_indices in enumerate(spec_row_indices_split):
        if len(row_indices) > 0:
            candidate_residuals = residuals[row_indices]
            if len(candidate_residuals) > 0:
                max_unmatched[i] = np.max(np.abs(candidate_residuals))
                max_matched[i] = np.max(candidate_residuals)
    
    return max_unmatched, max_matched


def calculate_residual_features_csc(
    sparse_matrix_csc: sparse.csc_matrix,
    residuals: np.ndarray,
    dia_spectrum_intensities: np.ndarray,
    lib_coefficients: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate residual-based features using CSC sparse matrix operations.
    
    This function computes maximum matched and unmatched residuals for each candidate
    using the same logic as gof_stat_csc. It distinguishes between matched and unmatched
    peaks based on observed DIA intensities and applies normalization and log transformation.
    
    Args:
        sparse_matrix_csc: Sparse matrix in CSC format with shape (n_peaks, n_candidates)
        residuals: Array of residuals from NNLS fit
        dia_spectrum_intensities: DIA spectrum intensities (column 1 of dia_spectrum)
        lib_coefficients: NNLS coefficients for each candidate
        
    Returns:
        Tuple of (max_unmatched_residuals, max_matched_residuals)
    """
    # Ensure coefficients are in the correct format
    coeffs = np.asarray(lib_coefficients).ravel()
    
    # Handle empty matrix case
    if sparse_matrix_csc.shape[1] == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    
    n_candidates = sparse_matrix_csc.shape[1]
    
    # Initialize output arrays
    max_unmatched_residuals = np.zeros(n_candidates, dtype=np.float64)
    max_matched_residuals = np.zeros(n_candidates, dtype=np.float64)
    
    # Process each candidate (column) in the sparse matrix
    for col_idx in range(n_candidates):
        # Extract the column for this candidate
        col_data = sparse_matrix_csc[:, col_idx]
        
        # Get non-zero entries (matched peaks for this candidate)
        nonzero_rows, _ = col_data.nonzero()
        
        if len(nonzero_rows) == 0:
            # No matched peaks, set to default values
            max_unmatched_residuals[col_idx] = np.log2(1e-10)
            max_matched_residuals[col_idx] = np.log2(1e-10)
            continue
        
        # Get coefficient for this candidate
        if col_idx < len(coeffs):
            coeff = coeffs[col_idx]
        else:
            coeff = 0.0
        
        # Initialize accumulators
        sum_of_fitted_peaks = 0.0
        max_unmatched_residual = 0.0
        max_matched_residual = 0.0
        
        # Process each matched peak for this candidate
        for row_idx in nonzero_rows:
            # Get library intensity value for this peak
            lib_intensity = col_data[row_idx, 0]
            
            # Calculate absolute residual
            abs_residual = abs(residuals[row_idx])
            
            # Calculate fitted peak intensity
            fitted_peak = abs(coeff * lib_intensity)
            sum_of_fitted_peaks += fitted_peak
            
            # Determine if this is a matched or unmatched peak based on observed intensity
            observed_intensity = dia_spectrum_intensities[row_idx]
            
            if observed_intensity > 1e-6:
                # Matched peak
                if abs_residual > max_matched_residual:
                    max_matched_residual = abs_residual
            else:
                # Unmatched peak
                if abs_residual > max_unmatched_residual:
                    max_unmatched_residual = abs_residual
        
        # Handle edge cases
        if sum_of_fitted_peaks == 0:
            sum_of_fitted_peaks = 1e-6
        
        # Calculate normalized and log-transformed maximum residuals
        max_matched_residuals[col_idx] = np.log2(max_matched_residual / (sum_of_fitted_peaks + 1e-10) + 1e-10)
        max_unmatched_residuals[col_idx] = np.log2(max_unmatched_residual / (sum_of_fitted_peaks + 1e-10) + 1e-10)
    
    return max_unmatched_residuals, max_matched_residuals