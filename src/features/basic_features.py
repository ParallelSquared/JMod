"""
Basic feature calculations for spectral fitting.

This module contains functions for calculating fundamental features like
peak counts, intensity fractions, and coefficients.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class BasicFeatureInputs:
    """Input data required for basic feature calculations."""
    n_candidates: int
    peaks_in_dia: List[int]
    lib_peaks_matched: List[np.ndarray]
    spec_values_split: List[np.ndarray]
    spec_row_indices_split: List[np.ndarray]
    dia_spectrum: np.ndarray
    dia_total_intensity: float
    lib_coefficients: np.ndarray
    ms1_error: np.ndarray
    rt_mz: np.ndarray
    window_idxs: np.ndarray
    prec_rt: float


def calculate_basic_features(inputs: BasicFeatureInputs) -> np.ndarray:
    """
    Calculate basic features for all candidates.
    
    Features calculated:
    - Feature 0: Number of library peaks matched
    - Feature 1: Fraction of library intensity matched
    - Feature 2: Fraction of DIA intensity matched
    - Feature 3: MS1 relative error
    - Feature 4: RT error
    - Feature 5: Fraction intensity matched
    - Feature 6: Fraction intensity predicted
    
    Args:
        inputs: BasicFeatureInputs containing all required data
        
    Returns:
        Array of shape (n_candidates, 7) containing basic features
    """
    features = np.zeros((inputs.n_candidates, 7))
    
    for i in range(inputs.n_candidates):
        candidate_idx = inputs.peaks_in_dia[i]
        
        # Feature 0: Number of library peaks matched
        features[i, 0] = calculate_num_peaks_matched(inputs.lib_peaks_matched[i])
        
        # Feature 1: Fraction of library intensity matched
        features[i, 1] = calculate_frac_lib_intensity(inputs.spec_values_split[i])
        
        # Feature 2: Fraction of DIA intensity matched
        features[i, 2] = calculate_frac_dia_intensity(
            inputs.spec_row_indices_split[i],
            inputs.dia_spectrum,
            inputs.dia_total_intensity
        )
        
        # Feature 3: MS1 relative error
        features[i, 3] = inputs.ms1_error[i]
        
        # Feature 4: RT error
        features[i, 4] = calculate_rt_error(
            candidate_idx,
            inputs.window_idxs,
            inputs.rt_mz,
            inputs.prec_rt
        )
        
        # Feature 5: Fraction intensity matched
        features[i, 5] = calculate_frac_intensity_matched(
            inputs.spec_values_split[i],
            inputs.lib_coefficients[i] if i < len(inputs.lib_coefficients) else 0
        )
        
        # Feature 6: Fraction intensity predicted
        features[i, 6] = calculate_frac_intensity_predicted(
            features[i, 5],
            inputs.lib_coefficients[i] if i < len(inputs.lib_coefficients) else 0
        )
    
    return features


def calculate_num_peaks_matched(lib_peaks_matched: np.ndarray) -> float:
    """Calculate the number of library peaks matched."""
    return np.sum(lib_peaks_matched)


def calculate_frac_lib_intensity(spec_values: np.ndarray) -> float:
    """Calculate the fraction of library intensity matched."""
    return np.sum(spec_values)


def calculate_frac_dia_intensity(
    row_indices: np.ndarray,
    dia_spectrum: np.ndarray,
    dia_total_intensity: float
) -> float:
    """Calculate the fraction of DIA intensity matched."""
    if len(row_indices) > 0 and dia_total_intensity > 0:
        return np.sum(dia_spectrum[row_indices, 1]) / dia_total_intensity
    return 0.0


def calculate_rt_error(
    candidate_idx: int,
    window_idxs: np.ndarray,
    rt_mz: np.ndarray,
    prec_rt: float
) -> float:
    """Calculate retention time error."""
    if candidate_idx < len(window_idxs):
        candidate_rt = rt_mz[window_idxs[candidate_idx], 0]
        return prec_rt - candidate_rt
    return 0.0  # Default value for out of bounds


def calculate_frac_intensity_matched(
    spec_values: np.ndarray,
    lib_coefficient: float
) -> float:
    """Calculate fraction intensity matched."""
    if len(spec_values) > 0:
        return np.sum(spec_values * lib_coefficient)
    return 0.0


def calculate_frac_intensity_predicted(
    frac_intensity_matched: float,
    lib_coefficient: float
) -> float:
    """Calculate fraction intensity predicted."""
    return frac_intensity_matched * lib_coefficient