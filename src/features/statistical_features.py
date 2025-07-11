"""
Statistical feature calculations for spectral fitting.

This module contains functions for calculating statistical features
including significance tests and feature combinations.
"""

import numpy as np
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class StatisticalFeatureInputs:
    """Input data required for statistical feature calculations."""
    n_candidates: int
    peaks_in_dia: List[int]
    basic_features: np.ndarray  # Features from basic_features module
    lib_coefficients: np.ndarray
    rt_mz: np.ndarray
    window_idxs: np.ndarray


def calculate_statistical_features(inputs: StatisticalFeatureInputs) -> np.ndarray:
    """
    Calculate statistical features for all candidates.
    
    Features calculated:
    - Feature 22: Fraction intensity matched predicted (copy of feature 5)
    - Feature 23: Fraction intensity matched predicted with significance cutoff
    - Feature 25: RT/MZ value
    
    Args:
        inputs: StatisticalFeatureInputs containing all required data
        
    Returns:
        Array of shape (n_candidates, 3) containing statistical features
    """
    features = np.zeros((inputs.n_candidates, 3))
    
    for i in range(inputs.n_candidates):
        candidate_idx = inputs.peaks_in_dia[i]
        
        # Feature 22: Copy of fraction intensity matched (feature 5)
        if inputs.basic_features is not None and inputs.basic_features.shape[1] > 5:
            features[i, 0] = inputs.basic_features[i, 5]
        
        # Feature 23: Same as feature 22 but with coefficient significance cutoff
        features[i, 1] = calculate_frac_intensity_with_cutoff(
            features[i, 0],
            inputs.lib_coefficients[i] if i < len(inputs.lib_coefficients) else 0,
            cutoff=0.1
        )
        
        # Feature 25: m/z value
        features[i, 2] = calculate_mz_value(
            candidate_idx,
            inputs.window_idxs,
            inputs.rt_mz
        )
    
    return features


def calculate_frac_intensity_with_cutoff(
    frac_intensity_matched: float,
    lib_coefficient: float,
    cutoff: float = 0.1
) -> float:
    """
    Calculate fraction intensity matched with significance cutoff.
    
    Only returns the value if the library coefficient exceeds the cutoff.
    
    Args:
        frac_intensity_matched: Fraction intensity matched value
        lib_coefficient: Library coefficient for this candidate
        cutoff: Minimum coefficient value to be considered significant
        
    Returns:
        Fraction intensity matched if coefficient > cutoff, else 0
    """
    if lib_coefficient > cutoff:
        return frac_intensity_matched
    return 0.0


def calculate_mz_value(
    candidate_idx: int,
    window_idxs: np.ndarray,
    rt_mz: np.ndarray
) -> float:
    """
    Calculate m/z value for the candidate.
    
    Args:
        candidate_idx: Index of the candidate
        window_idxs: Window indices array
        rt_mz: RT and m/z values array
        
    Returns:
        m/z value for the candidate
    """
    if candidate_idx < len(window_idxs):
        return rt_mz[window_idxs[candidate_idx], 1]
    return 0.0  # Default value for out of bounds


def calculate_p_values(
    features: np.ndarray,
    null_distribution: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Calculate p-values for features based on null distribution.
    
    This is a placeholder for more sophisticated statistical testing.
    
    Args:
        features: Feature matrix
        null_distribution: Optional null distribution for comparison
        
    Returns:
        Array of p-values
    """
    n_candidates = features.shape[0]
    p_values = np.ones(n_candidates)
    
    if null_distribution is not None:
        # Implement statistical test against null distribution
        # This would typically use empirical p-value calculation
        pass
    
    return p_values


def calculate_fdr(p_values: np.ndarray, method: str = 'benjamini-hochberg') -> np.ndarray:
    """
    Calculate false discovery rate from p-values.
    
    Args:
        p_values: Array of p-values
        method: FDR correction method
        
    Returns:
        Array of FDR-corrected values
    """
    n = len(p_values)
    if n == 0:
        return np.array([])
    
    # Sort p-values
    sorted_indices = np.argsort(p_values)
    sorted_p_values = p_values[sorted_indices]
    
    if method == 'benjamini-hochberg':
        # Benjamini-Hochberg procedure
        fdr_values = np.zeros(n)
        for i in range(n):
            fdr_values[i] = sorted_p_values[i] * n / (i + 1)
        
        # Ensure monotonicity
        for i in range(n - 2, -1, -1):
            fdr_values[i] = min(fdr_values[i], fdr_values[i + 1])
        
        # Cap at 1
        fdr_values = np.minimum(fdr_values, 1.0)
        
        # Unsort
        unsorted_fdr = np.zeros(n)
        unsorted_fdr[sorted_indices] = fdr_values
        return unsorted_fdr
    
    else:
        raise ValueError(f"Unknown FDR method: {method}")


def calculate_combined_scores(
    features: np.ndarray,
    weights: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Calculate combined scores from multiple features.
    
    Args:
        features: Feature matrix (n_candidates x n_features)
        weights: Optional weights for each feature
        
    Returns:
        Array of combined scores
    """
    if weights is None:
        # Use equal weights
        weights = np.ones(features.shape[1]) / features.shape[1]
    
    # Normalize features to [0, 1] range
    normalized_features = normalize_features(features)
    
    # Calculate weighted sum
    scores = np.dot(normalized_features, weights)
    
    return scores


def normalize_features(features: np.ndarray) -> np.ndarray:
    """
    Normalize features to [0, 1] range.
    
    Args:
        features: Feature matrix
        
    Returns:
        Normalized feature matrix
    """
    normalized = np.zeros_like(features)
    
    for i in range(features.shape[1]):
        col = features[:, i]
        min_val = np.min(col)
        max_val = np.max(col)
        
        if max_val > min_val:
            normalized[:, i] = (col - min_val) / (max_val - min_val)
        else:
            normalized[:, i] = 0.5  # All values are the same
    
    return normalized