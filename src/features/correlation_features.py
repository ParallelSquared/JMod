"""
Correlation and similarity features for spectral matching.

This module contains functions to calculate features related to
correlations between library and observed spectra.
"""

import numpy as np
from typing import List, Tuple, Optional, Any
from scipy import stats, sparse
import warnings


def calculate_r2_all(
    dia_spec_int: np.ndarray,
    predicted_spec: np.ndarray,
    n_candidates: int
) -> np.ndarray:
    """
    Calculate R² between observed and predicted spectrum for all peaks.
    
    Args:
        dia_spec_int: Observed DIA spectrum intensities
        predicted_spec: Predicted spectrum from NNLS
        n_candidates: Number of candidates
        
    Returns:
        Array of R² values (same for all candidates)
    """
    if len(dia_spec_int) > 1 and len(predicted_spec) > 1:
        # Remove penalty term if present
        dia_int = dia_spec_int[:-1] if len(dia_spec_int) > len(predicted_spec) else dia_spec_int
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r2 = stats.pearsonr(dia_int, predicted_spec)[0]
            if np.isnan(r2):
                r2 = 0.0
    else:
        r2 = 0.0
    
    return np.full(n_candidates, r2)


def calculate_r2_lib_spec(
    spec_values_split: List[np.ndarray],
    spec_row_indices_split: List[np.ndarray],
    dia_spectrum: np.ndarray
) -> np.ndarray:
    """
    Calculate R² between library and observed peaks for each candidate.
    
    Args:
        spec_values_split: Library intensity values for matched peaks
        spec_row_indices_split: Row indices in DIA spectrum
        dia_spectrum: Full DIA spectrum
        
    Returns:
        Array of R² values for each candidate
    """
    r2_values = []
    
    for lib_values, row_indices in zip(spec_values_split, spec_row_indices_split):
        if len(lib_values) > 1 and len(row_indices) > 1:
            obs_values = dia_spectrum[row_indices, 1]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r2 = stats.pearsonr(lib_values, obs_values)[0]
                if np.isnan(r2):
                    r2 = 0.0
            r2_values.append(r2)
        else:
            r2_values.append(0.0)
    
    return np.array(r2_values)


def calculate_r2_lib_spec_csc(
    sparse_matrix_csc: sparse.csc_matrix,
    dia_spectrum: np.ndarray
) -> np.ndarray:
    """
    Calculate R² between library and observed peaks for each candidate using CSC sparse matrix.
    
    This is the CSC version of calculate_r2_lib_spec that replaces split array operations
    with efficient sparse matrix operations.
    
    Args:
        sparse_matrix_csc: Sparse matrix in CSC format with shape (n_peaks, n_candidates)
                          where non-zero values are library intensities
        dia_spectrum: Full DIA spectrum with shape (n_peaks, 2) where column 1 contains intensities
        
    Returns:
        Array of R² values for each candidate
    """
    n_candidates = sparse_matrix_csc.shape[1]
    if n_candidates == 0:
        return np.array([], dtype=np.float64)
    
    r2_values = np.zeros(n_candidates)
    
    # Process each candidate (column) in the sparse matrix
    for candidate_idx in range(n_candidates):
        # Get column data for this candidate
        col_start = sparse_matrix_csc.indptr[candidate_idx]
        col_end = sparse_matrix_csc.indptr[candidate_idx + 1]
        
        if col_end > col_start:  # Candidate has matched peaks
            # Get row indices and values for this candidate
            row_indices = sparse_matrix_csc.indices[col_start:col_end]
            lib_values = sparse_matrix_csc.data[col_start:col_end]
            
            if len(lib_values) > 1:  # Need at least 2 points for correlation
                # Get corresponding observed values from DIA spectrum
                obs_values = dia_spectrum[row_indices, 1]
                
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    r2 = stats.pearsonr(lib_values, obs_values)[0]
                    if np.isnan(r2):
                        r2 = 0.0
                r2_values[candidate_idx] = r2
            else:
                r2_values[candidate_idx] = 0.0
        else:
            r2_values[candidate_idx] = 0.0
    
    return r2_values


def calculate_cosine_similarity(
    vec1: np.ndarray,
    vec2: np.ndarray
) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Cosine similarity value
    """
    if len(vec1) == 0 or len(vec2) == 0:
        return 0.0
    
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return np.dot(vec1, vec2) / (norm1 * norm2)


def calculate_spectral_contrast(
    pred_vals: np.ndarray,
    obs_vals: np.ndarray
) -> float:
    """
    Calculate spectral contrast angle between predicted and observed.
    
    Args:
        pred_vals: Predicted intensity values
        obs_vals: Observed intensity values
        
    Returns:
        Spectral contrast value (1 - normalized angle)
    """
    if np.sum(pred_vals) == 0 or np.sum(obs_vals) == 0:
        return 0.0
    
    # Normalize vectors
    pred_norm = pred_vals / np.sqrt(np.sum(pred_vals**2))
    obs_norm = obs_vals / np.sqrt(np.sum(obs_vals**2))
    
    # Calculate dot product (clipped to valid range)
    dot_product = np.clip(np.sum(pred_norm * obs_norm), -1, 1)
    
    # Calculate spectral contrast
    angle = np.arccos(dot_product)
    return 1 - (2 * angle / np.pi)


def calculate_unique_peak_features(
    spec_row_indices_split: List[np.ndarray],
    spec_values_split: List[np.ndarray],
    lib_coefficients: np.ndarray,
    sparse_matrix: Optional[Any] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate features for peaks unique to each candidate.
    
    Args:
        spec_row_indices_split: Row indices for each candidate
        spec_values_split: Intensity values for each candidate
        lib_coefficients: NNLS coefficients
        sparse_matrix: Sparse matrix (optional for unique peak detection)
        
    Returns:
        Tuple of (r2_unique, frac_unique_pred)
    """
    n_candidates = len(spec_row_indices_split)
    r2_unique = np.zeros(n_candidates)
    frac_unique_pred = np.zeros(n_candidates)
    
    # Simplified version without sparse matrix analysis
    # In full implementation, would identify truly unique peaks
    for i in range(n_candidates):
        if i < len(lib_coefficients) and len(spec_values_split[i]) > 0:
            # Placeholder: treat all peaks as potentially unique
            frac_unique_pred[i] = np.sum(spec_values_split[i]) * lib_coefficients[i]
    
    return r2_unique, frac_unique_pred


def calculate_unique_peak_features_csc(
    sparse_matrix_csc: sparse.csc_matrix,
    lib_coefficients: np.ndarray,
    dia_spectrum: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate unique peak analysis features using CSC sparse matrix operations.
    
    This function identifies peaks that are matched to only one candidate (unique peaks),
    then calculates correlations and fraction predictions for these unique peaks.
    This is the CSC version for the feature calculator that replaces split array operations.
    
    Args:
        sparse_matrix_csc: Sparse matrix in CSC format with shape (n_dia_peaks, n_candidates)
                          where non-zero values are library intensities at matched peaks
        lib_coefficients: Array of coefficients from NNLS optimization, one per candidate
        dia_spectrum: DIA spectrum array with shape (n_peaks, 2) where column 1 contains intensities
    
    Returns:
        Tuple containing:
        - r2_unique: Array of Pearson correlations for unique peaks, one per candidate
        - frac_unique_pred: Array of fraction unique predicted intensities, one per candidate
    """
    # Handle empty matrix case
    if sparse_matrix_csc.shape[1] == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    
    n_candidates = sparse_matrix_csc.shape[1]
    
    # Find rows that have exactly one non-zero entry (unique peaks)
    # Count non-zeros per row using sparse matrix operations
    row_counts = np.array((sparse_matrix_csc > 0).sum(axis=1)).flatten()
    single_matched_rows = np.where(row_counts == 1)[0]
    
    # Pre-allocate result arrays
    r2_unique = np.zeros(n_candidates, dtype=np.float64)
    frac_unique_pred = np.zeros(n_candidates, dtype=np.float64)
    
    # Process each candidate
    for col_idx in range(n_candidates):
        # Get the column for this candidate
        col_data = sparse_matrix_csc[:, col_idx]
        nonzero_rows, _ = col_data.nonzero()
        
        # Find which of this candidate's matches are unique peaks
        unique_rows = np.intersect1d(nonzero_rows, single_matched_rows)
        
        if len(unique_rows) > 0:
            # Filter out penalty rows (beyond DIA spectrum bounds)
            valid_unique_rows = unique_rows[unique_rows < len(dia_spectrum)]
            
            if len(valid_unique_rows) > 1:
                # Calculate correlation for unique peaks
                n = len(valid_unique_rows)
                sum_lib = 0.0
                sum_dia = 0.0
                
                # Calculate means
                for row_idx in valid_unique_rows:
                    lib_val = col_data[row_idx, 0]
                    dia_val = dia_spectrum[row_idx, 1]
                    sum_lib += lib_val
                    sum_dia += dia_val
                
                mean_lib = sum_lib / n
                mean_dia = sum_dia / n
                
                # Calculate correlation components
                sum_lib_dev_sq = 0.0
                sum_dia_dev_sq = 0.0
                sum_cross_dev = 0.0
                
                for row_idx in valid_unique_rows:
                    lib_val = col_data[row_idx, 0]
                    dia_val = dia_spectrum[row_idx, 1]
                    
                    lib_dev = lib_val - mean_lib
                    dia_dev = dia_val - mean_dia
                    
                    sum_lib_dev_sq += lib_dev * lib_dev
                    sum_dia_dev_sq += dia_dev * dia_dev
                    sum_cross_dev += lib_dev * dia_dev
                
                # Calculate correlation coefficient
                denominator = (sum_lib_dev_sq * sum_dia_dev_sq) ** 0.5
                if denominator > 0:
                    r2_unique[col_idx] = sum_cross_dev / denominator
                else:
                    r2_unique[col_idx] = np.nan  # Correlation is undefined when one variable is constant
            else:
                r2_unique[col_idx] = 0.0
            
            # Calculate fraction unique predicted
            # Sum of library intensities at unique peaks for this candidate
            total_lib_intensity = 0.0
            total_dia_intensity = 0.0
            
            for row_idx in valid_unique_rows:
                lib_val = col_data[row_idx, 0]
                dia_val = dia_spectrum[row_idx, 1]
                total_lib_intensity += lib_val
                total_dia_intensity += dia_val
            
            # Calculate fraction: (dia_sum / lib_sum) * coefficient
            if total_lib_intensity > 0 and col_idx < len(lib_coefficients):
                frac_unique_pred[col_idx] = (total_dia_intensity / total_lib_intensity) * lib_coefficients[col_idx]
            else:
                frac_unique_pred[col_idx] = 0.0
        else:
            # No unique peaks for this candidate
            r2_unique[col_idx] = 0.0
            frac_unique_pred[col_idx] = 0.0
    
    return r2_unique, frac_unique_pred