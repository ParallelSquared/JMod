"""
Intensity-based features for spectral matching.

This module contains functions to calculate features related to
intensity matching between library and observed spectra.
"""

import numpy as np
from typing import List, Tuple, Optional
from scipy import sparse


def calculate_num_peaks_matched(lib_peaks_matched: List[np.ndarray]) -> np.ndarray:
    """
    Calculate the number of library peaks matched for each candidate.
    
    Args:
        lib_peaks_matched: List of boolean arrays indicating which peaks matched
        
    Returns:
        Array of counts for each candidate
    """
    return np.array([np.sum(matched) for matched in lib_peaks_matched])


def calculate_fraction_library_intensity(spec_values_split: List[np.ndarray]) -> np.ndarray:
    """
    Calculate fraction of library intensity matched for each candidate.
    
    Since library intensities are normalized to sum to 1, this is just
    the sum of matched peak intensities.
    
    Args:
        spec_values_split: List of intensity arrays for matched peaks
        
    Returns:
        Array of fractions for each candidate
    """
    return np.array([np.sum(values) for values in spec_values_split])


def calculate_fraction_dia_intensity(
    spec_row_indices_split: List[np.ndarray],
    dia_spectrum: np.ndarray
) -> np.ndarray:
    """
    Calculate fraction of DIA intensity matched for each candidate.
    
    Args:
        spec_row_indices_split: Row indices in DIA spectrum for each candidate
        dia_spectrum: Full DIA spectrum [m/z, intensity]
        
    Returns:
        Array of fractions relative to total DIA intensity
    """
    total_intensity = np.sum(dia_spectrum[:, 1])
    if total_intensity == 0:
        return np.zeros(len(spec_row_indices_split))
    
    fractions = []
    for row_indices in spec_row_indices_split:
        if len(row_indices) > 0:
            matched_intensity = np.sum(dia_spectrum[row_indices, 1])
            fractions.append(matched_intensity / total_intensity)
        else:
            fractions.append(0.0)
    
    return np.array(fractions)


def calculate_fraction_intensity_matched(
    spec_values_split: List[np.ndarray],
    lib_coefficients: np.ndarray
) -> np.ndarray:
    """
    Calculate fraction of intensity matched weighted by coefficients.
    
    Args:
        spec_values_split: Intensity values for matched peaks
        lib_coefficients: NNLS coefficients for each candidate
        
    Returns:
        Array of weighted intensity fractions
    """
    fractions = []
    for i, values in enumerate(spec_values_split):
        if i < len(lib_coefficients) and len(values) > 0:
            fractions.append(np.sum(values) * lib_coefficients[i])
        else:
            fractions.append(0.0)
    
    return np.array(fractions)


def calculate_fraction_intensity_matched_csc(
    sparse_matrix_csc: sparse.csc_matrix,
    lib_coefficients: np.ndarray
) -> np.ndarray:
    """
    Calculate fraction of intensity matched weighted by coefficients using CSC sparse matrix.
    
    This is the CSC version of calculate_fraction_intensity_matched that replaces
    split array operations with efficient sparse matrix operations.
    
    Args:
        sparse_matrix_csc: Sparse matrix in CSC format with shape (n_peaks, n_candidates)
                          where non-zero values are library intensities
        lib_coefficients: NNLS coefficients for each candidate
        
    Returns:
        Array of weighted intensity fractions
    """
    # Handle empty matrix case
    if sparse_matrix_csc.shape[1] == 0:
        return np.array([], dtype=np.float64)
    
    # Get intensity sums per candidate using robust sparse matrix operations
    intensity_sums = np.asarray(sparse_matrix_csc.sum(axis=0)).flatten()
    
    # Ensure lib_coefficients is a 1D array
    lib_coeffs_1d = np.asarray(lib_coefficients).flatten()
    
    # Handle size mismatch between intensity_sums and lib_coefficients
    n_candidates = sparse_matrix_csc.shape[1]
    if len(lib_coeffs_1d) != n_candidates:
        # Resize lib_coefficients to match number of candidates
        if len(lib_coeffs_1d) < n_candidates:
            # Pad with zeros if lib_coefficients is too short
            padded_coeffs = np.zeros(n_candidates, dtype=np.float64)
            padded_coeffs[:len(lib_coeffs_1d)] = lib_coeffs_1d
            lib_coeffs_1d = padded_coeffs
        else:
            # Truncate if lib_coefficients is too long
            lib_coeffs_1d = lib_coeffs_1d[:n_candidates]
    
    # Multiply intensity sums by coefficients element-wise
    return intensity_sums * lib_coeffs_1d



def calculate_fraction_intensity_predicted(
    frac_int_matched: np.ndarray,
    lib_coefficients: np.ndarray,
    n_candidates: int
) -> np.ndarray:
    """
    Calculate fraction of intensity predicted by the model.
    
    Args:
        frac_int_matched: Fraction of intensity matched for each candidate
        lib_coefficients: NNLS coefficients
        n_candidates: Number of candidates
        
    Returns:
        Array of predicted intensity fractions
    """
    # Use loop to avoid broadcasting issues
    result = np.zeros(n_candidates)
    for i in range(n_candidates):
        if i < len(lib_coefficients) and i < len(frac_int_matched):
            result[i] = frac_int_matched[i] * lib_coefficients[i]
        else:
            result[i] = 0.0
    
    return result


def calculate_large_coefficient_features(
    spec_row_indices_split: List[np.ndarray],
    spec_values_split: List[np.ndarray],
    lib_coefficients: np.ndarray,
    dia_spectrum: np.ndarray,
    sparse_matrix: 'sparse.coo_matrix',
    sparse_row_indices: Optional[np.ndarray] = None,
    sparse_col_indices: Optional[np.ndarray] = None,
    threshold: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate features related to candidates with large coefficients.
    
    Args:
        spec_row_indices_split: Row indices for each candidate
        spec_values_split: Intensity values for each candidate
        lib_coefficients: NNLS coefficients
        dia_spectrum: Full DIA spectrum
        sparse_matrix: Sparse matrix representation
        sparse_row_indices: Row indices from sparse matrix (optional)
        sparse_col_indices: Column indices from sparse matrix (optional)
        threshold: Coefficient threshold for "large" (default 1.0)
        
    Returns:
        Tuple of (frac_int_matched_pred_sigcoeff, large_coeff_cosine)
    """
    n_candidates = len(spec_row_indices_split)
    large_coeff_indices = np.where(np.array(lib_coefficients) > threshold)[0]
    
    if len(large_coeff_indices) == 0:
        return np.zeros(n_candidates), np.zeros(n_candidates)
    
    # Calculate intensity from large coefficient candidates
    large_coeff_matched_peaks = []
    for idx in large_coeff_indices:
        if idx < len(spec_row_indices_split):
            large_coeff_matched_peaks.extend(spec_row_indices_split[idx])
    
    if len(large_coeff_matched_peaks) > 0:
        unique_peaks = np.unique(large_coeff_matched_peaks)
        large_coeff_int_matched = np.sum(dia_spectrum[unique_peaks, 1])
        
        # Calculate predicted intensity from large coefficients
        large_coeff_int_pred = 0
        for idx in large_coeff_indices:
            if idx < len(spec_values_split) and idx < len(lib_coefficients):
                large_coeff_int_pred += np.sum(spec_values_split[idx]) * lib_coefficients[idx]
        
        if large_coeff_int_matched > 0:
            frac_int_matched_pred_sigcoeff = np.full(n_candidates, 
                                                     large_coeff_int_pred / large_coeff_int_matched)
        else:
            frac_int_matched_pred_sigcoeff = np.zeros(n_candidates)
    else:
        frac_int_matched_pred_sigcoeff = np.zeros(n_candidates)
    
    # Calculate cosine similarity for large coefficient subset
    # This requires the sparse matrix - simplified version here
    large_coeff_cosine = np.zeros(n_candidates)  # Placeholder
    
    return frac_int_matched_pred_sigcoeff, large_coeff_cosine