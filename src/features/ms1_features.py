"""
MS1-related feature calculations for spectral fitting.

This module contains functions for calculating features related to MS1 spectra,
including correlation features and unique peak analysis.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class MS1FeatureInputs:
    """Input data required for MS1 feature calculations."""
    n_candidates: int
    sparse_matrix: Any  # scipy.sparse matrix
    lib_coefficients: np.ndarray
    dia_spectrum: np.ndarray
    unique_row_idxs: np.ndarray
    spec_row_indices_split: List[np.ndarray]
    spec_col_indices_split: List[np.ndarray]
    spec_values_split: List[np.ndarray]
    peak_idx_convertor: Dict[int, int]


def calculate_ms1_features(inputs: MS1FeatureInputs) -> np.ndarray:
    """
    Calculate MS1-related features for all candidates.
    
    Features calculated:
    - Feature 7: R² for all peaks
    - Feature 8: R² for library spectrum
    - Feature 9: R² for unique peaks
    - Feature 10: Fraction of unique peaks predicted
    - Feature 11: Fraction of DIA intensity predicted
    
    Args:
        inputs: MS1FeatureInputs containing all required data
        
    Returns:
        Array of shape (n_candidates, 5) containing MS1 features
    """
    features = np.zeros((inputs.n_candidates, 5))
    
    # Calculate predicted spectrum once
    if inputs.sparse_matrix is not None and len(inputs.lib_coefficients) > 0:
        y_predicted = inputs.sparse_matrix.dot(inputs.lib_coefficients)
    else:
        y_predicted = np.zeros(len(inputs.unique_row_idxs))
    
    # Calculate R² for all peaks (Feature 7)
    features[:, 0] = calculate_r2_all(
        inputs.dia_spectrum[inputs.unique_row_idxs, 1],
        y_predicted
    )
    
    # Calculate per-candidate features
    for i in range(inputs.n_candidates):
        # Feature 8: R² for library spectrum
        features[i, 1] = calculate_r2_lib_spec(
            inputs.spec_row_indices_split[i],
            inputs.spec_values_split[i],
            inputs.lib_coefficients[i] if i < len(inputs.lib_coefficients) else 0,
            inputs.dia_spectrum,
            inputs.peak_idx_convertor
        )
        
        # Feature 9: R² for unique peaks
        features[i, 2] = calculate_r2_unique(
            i,
            inputs.spec_row_indices_split,
            inputs.spec_col_indices_split,
            inputs.dia_spectrum,
            y_predicted,
            inputs.peak_idx_convertor
        )
        
        # Feature 10: Fraction of unique peaks predicted
        features[i, 3] = calculate_frac_unique_predicted(
            i,
            inputs.spec_row_indices_split,
            inputs.spec_col_indices_split,
            y_predicted,
            inputs.peak_idx_convertor
        )
        
        # Feature 11: Fraction of DIA intensity predicted
        features[i, 4] = calculate_frac_dia_intensity_predicted(
            inputs.spec_row_indices_split[i],
            y_predicted,
            inputs.dia_spectrum,
            inputs.peak_idx_convertor
        )
    
    return features


def calculate_r2_all(observed: np.ndarray, predicted: np.ndarray) -> float:
    """Calculate R² for all peaks."""
    if len(observed) > 0 and len(predicted) > 0:
        return calculate_r_squared(observed, predicted)
    return 0.0


def calculate_r2_lib_spec(
    row_indices: np.ndarray,
    spec_values: np.ndarray,
    lib_coefficient: float,
    dia_spectrum: np.ndarray,
    peak_idx_convertor: Dict[int, int]
) -> float:
    """Calculate R² for library spectrum."""
    if len(row_indices) > 0 and lib_coefficient > 0:
        observed = dia_spectrum[row_indices, 1]
        predicted = spec_values * lib_coefficient
        return calculate_r_squared(observed, predicted)
    return 0.0


def calculate_r2_unique(
    candidate_idx: int,
    spec_row_indices_split: List[np.ndarray],
    spec_col_indices_split: List[np.ndarray],
    dia_spectrum: np.ndarray,
    y_predicted: np.ndarray,
    peak_idx_convertor: Dict[int, int]
) -> float:
    """Calculate R² for unique peaks of this candidate."""
    unique_peaks = get_unique_peaks(
        candidate_idx,
        spec_row_indices_split,
        spec_col_indices_split
    )
    
    if len(unique_peaks) > 0:
        observed = []
        predicted = []
        
        for peak in unique_peaks:
            if peak in peak_idx_convertor:
                idx = peak_idx_convertor[peak]
                observed.append(dia_spectrum[peak, 1])
                predicted.append(y_predicted[idx])
        
        if len(observed) > 0:
            return calculate_r_squared(np.array(observed), np.array(predicted))
    
    return 0.0


def calculate_frac_unique_predicted(
    candidate_idx: int,
    spec_row_indices_split: List[np.ndarray],
    spec_col_indices_split: List[np.ndarray],
    y_predicted: np.ndarray,
    peak_idx_convertor: Dict[int, int]
) -> float:
    """Calculate fraction of unique peaks predicted."""
    unique_peaks = get_unique_peaks(
        candidate_idx,
        spec_row_indices_split,
        spec_col_indices_split
    )
    
    if len(unique_peaks) > 0:
        total_predicted = 0.0
        for peak in unique_peaks:
            if peak in peak_idx_convertor:
                idx = peak_idx_convertor[peak]
                total_predicted += y_predicted[idx]
        
        if np.sum(y_predicted) > 0:
            return total_predicted / np.sum(y_predicted)
    
    return 0.0


def calculate_frac_dia_intensity_predicted(
    row_indices: np.ndarray,
    y_predicted: np.ndarray,
    dia_spectrum: np.ndarray,
    peak_idx_convertor: Dict[int, int]
) -> float:
    """Calculate fraction of DIA intensity predicted."""
    if len(row_indices) > 0:
        total_predicted = 0.0
        for idx in row_indices:
            if idx in peak_idx_convertor:
                total_predicted += y_predicted[peak_idx_convertor[idx]]
        
        total_observed = np.sum(dia_spectrum[:, 1])
        if total_observed > 0:
            return total_predicted / total_observed
    
    return 0.0


def get_unique_peaks(
    candidate_idx: int,
    spec_row_indices_split: List[np.ndarray],
    spec_col_indices_split: List[np.ndarray]
) -> List[int]:
    """Get peaks unique to this candidate."""
    # Get all peaks from this candidate
    candidate_peaks = set(spec_row_indices_split[candidate_idx])
    
    # Remove peaks shared with other candidates
    for i, row_indices in enumerate(spec_row_indices_split):
        if i != candidate_idx:
            candidate_peaks -= set(row_indices)
    
    return list(candidate_peaks)


def calculate_r_squared(observed: np.ndarray, predicted: np.ndarray) -> float:
    """Calculate R² between observed and predicted values."""
    if len(observed) == 0 or len(predicted) == 0:
        return 0.0
    
    # Ensure arrays have same length
    if len(observed) != len(predicted):
        return 0.0
    
    # Calculate mean of observed
    mean_observed = np.mean(observed)
    
    # Calculate total sum of squares
    ss_tot = np.sum((observed - mean_observed) ** 2)
    
    # Calculate residual sum of squares
    ss_res = np.sum((observed - predicted) ** 2)
    
    # Calculate R²
    if ss_tot > 0:
        r2 = 1 - (ss_res / ss_tot)
        # Ensure R² is between 0 and 1
        return max(0, min(1, r2))
    
    return 0.0