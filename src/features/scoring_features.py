"""
Advanced scoring features for spectral matching.

This module contains functions to calculate complex scoring metrics
like SCRIBE score, Manhattan distance, and goodness-of-fit.
"""

import numpy as np
from typing import List, Tuple, Optional
from scipy import sparse


def calculate_scribe_score(
    spec_values: np.ndarray,
    dia_intensities: np.ndarray,
    row_indices: np.ndarray
) -> float:
    """
    Calculate SCRIBE score for a single candidate.
    
    SCRIBE (Spectral Contrast angle with Intensity-Based REweighting)
    measures spectral similarity with intensity weighting.
    
    Args:
        spec_values: Library intensity values for matched peaks
        dia_intensities: Full DIA spectrum intensities
        row_indices: Indices of matched peaks in DIA spectrum
        
    Returns:
        SCRIBE score value
    """
    if len(spec_values) == 0 or len(row_indices) == 0:
        return 0.0
    
    # Get matched DIA intensities
    matched_dia = dia_intensities[row_indices]
    
    # Normalize both vectors
    lib_norm = spec_values / (np.sum(spec_values) + 1e-10)
    dia_norm = matched_dia / (np.sum(matched_dia) + 1e-10)
    
    # Calculate weighted dot product
    dot_product = np.sum(lib_norm * dia_norm)
    
    # SCRIBE score incorporates intensity weighting
    intensity_weight = np.sqrt(np.sum(matched_dia) / (np.sum(dia_intensities) + 1e-10))
    
    return dot_product * intensity_weight


def calculate_manhattan_distance(
    predicted: np.ndarray,
    observed: np.ndarray
) -> float:
    """
    Calculate log-transformed Manhattan distance.
    
    Args:
        predicted: Predicted intensity values
        observed: Observed intensity values
        
    Returns:
        Log10 of normalized Manhattan distance
    """
    if len(predicted) == 0 or np.sum(observed) == 0:
        return -np.inf
    
    manhattan = np.sum(np.abs(predicted - observed))
    normalized = manhattan / np.sum(observed)
    
    # Avoid log(0)
    if normalized == 0:
        return -np.inf
    
    return np.log10(normalized)


def calculate_goodness_of_fit(
    residuals: np.ndarray,
    n_peaks: int
) -> float:
    """
    Calculate goodness-of-fit statistic.
    
    Args:
        residuals: Array of residuals from NNLS fit
        n_peaks: Number of peaks used in fit
        
    Returns:
        Goodness-of-fit value
    """
    if len(residuals) == 0 or n_peaks == 0:
        return 0.0
    
    # Chi-square-like statistic
    chi2 = np.sum(residuals**2) / n_peaks
    
    # Convert to probability-like score
    return np.exp(-chi2)


def calculate_all_scoring_features(
    spec_row_indices_split: List[np.ndarray],
    spec_values_split: List[np.ndarray],
    dia_spectrum: np.ndarray,
    residuals: Optional[np.ndarray],
    y_pred: Optional[np.ndarray]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate all advanced scoring features.
    
    Args:
        spec_row_indices_split: Row indices for each candidate
        spec_values_split: Intensity values for each candidate
        dia_spectrum: Full DIA spectrum
        residuals: Residuals from NNLS fit (optional)
        y_pred: Predicted spectrum from NNLS (optional)
        
    Returns:
        Tuple of (scribe_scores, manhattan_distances, spectral_contrasts, gof_stats)
    """
    n_candidates = len(spec_row_indices_split)
    scribe_scores = np.zeros(n_candidates)
    manhattan_distances = np.full(n_candidates, -np.inf)
    spectral_contrasts = np.zeros(n_candidates)
    gof_stats = np.zeros(n_candidates)
    
    for i in range(n_candidates):
        if len(spec_row_indices_split[i]) == 0:
            continue
        
        # SCRIBE score
        scribe_scores[i] = calculate_scribe_score(
            spec_values_split[i],
            dia_spectrum[:, 1],
            spec_row_indices_split[i]
        )
        
        # Manhattan distance and spectral contrast
        if y_pred is not None:
            row_indices = spec_row_indices_split[i]
            obs_vals = dia_spectrum[row_indices, 1]
            pred_vals = y_pred[row_indices]
            
            manhattan_distances[i] = calculate_manhattan_distance(pred_vals, obs_vals)
            
            # Spectral contrast (similar to cosine but with angle normalization)
            if np.sum(pred_vals) > 0 and np.sum(obs_vals) > 0:
                pred_norm = pred_vals / np.sqrt(np.sum(pred_vals**2))
                obs_norm = obs_vals / np.sqrt(np.sum(obs_vals**2))
                dot_product = np.clip(np.sum(pred_norm * obs_norm), -1, 1)
                spectral_contrasts[i] = 1 - (2 * np.arccos(dot_product) / np.pi)
        
        # Goodness of fit
        if residuals is not None:
            candidate_residuals = residuals[spec_row_indices_split[i]]
            gof_stats[i] = calculate_goodness_of_fit(
                candidate_residuals,
                len(spec_row_indices_split[i])
            )
    
    return scribe_scores, manhattan_distances, spectral_contrasts, gof_stats


def calculate_all_scoring_features_csc(
    sparse_matrix_csc: sparse.csc_matrix,
    dia_spectrum: np.ndarray,
    residuals: Optional[np.ndarray],
    y_pred: Optional[np.ndarray],
    lib_coefficients: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate all advanced scoring features using CSC sparse matrix operations.
    
    This is the CSC version that replaces split array operations with efficient
    sparse matrix operations. Uses proven CSC implementations from spectral_similarity_metrics.
    
    Args:
        sparse_matrix_csc: Sparse matrix in CSC format with shape (n_peaks, n_candidates)
        dia_spectrum: Full DIA spectrum with shape (n_peaks, 2)
        residuals: Residuals from NNLS fit (optional)
        y_pred: Predicted spectrum from NNLS (optional)
        lib_coefficients: NNLS coefficients for each candidate
        
    Returns:
        Tuple of (scribe_scores, manhattan_distances, spectral_contrasts, gof_stats, max_unmatched_residuals, max_matched_residuals)
    """
    from ..utils.spectral_similarity_metrics import (
        get_scribe_csc, get_manhattan_distance_csc, gof_stat_csc
    )
    
    n_candidates = sparse_matrix_csc.shape[1]
    if n_candidates == 0:
        empty_array = np.array([], dtype=np.float64)
        return empty_array, empty_array, empty_array, empty_array, empty_array, empty_array
    
    # Initialize result arrays
    scribe_scores = np.zeros(n_candidates, dtype=np.float64)
    manhattan_distances = np.full(n_candidates, -np.inf, dtype=np.float64)
    spectral_contrasts = np.zeros(n_candidates, dtype=np.float64)
    gof_stats = np.zeros(n_candidates, dtype=np.float64)
    max_unmatched_residuals = np.zeros(n_candidates, dtype=np.float64)
    max_matched_residuals = np.zeros(n_candidates, dtype=np.float64)
    
    # Calculate SCRIBE scores using CSC operations
    scribe_scores = get_scribe_csc(sparse_matrix_csc, dia_spectrum[:, 1])
    
    # Calculate Manhattan distances and spectral contrasts
    if y_pred is not None:
        manhattan_distances, spectral_contrasts = get_manhattan_distance_csc(
            sparse_matrix_csc, dia_spectrum[:, 1], y_pred
        )
    
    # Calculate goodness-of-fit statistics and residual features
    if residuals is not None:
        gof_results = gof_stat_csc(
            sparse_matrix_csc, residuals, dia_spectrum[:, 1], lib_coefficients
        )
        # gof_stat_csc returns (gof_stats, max_unmatched_residuals, max_matched_residuals)
        gof_stats = gof_results[0]
        max_unmatched_residuals = gof_results[1]
        max_matched_residuals = gof_results[2]
    
    return scribe_scores, manhattan_distances, spectral_contrasts, gof_stats, max_unmatched_residuals, max_matched_residuals