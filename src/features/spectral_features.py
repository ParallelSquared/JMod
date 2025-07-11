"""
Spectral similarity and quality feature calculations.

This module contains functions for calculating spectral similarity metrics,
hyperscores, and other spectral quality features.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class SpectralFeatureInputs:
    """Input data required for spectral feature calculations."""
    n_candidates: int
    candidates: List[Tuple]  # (sequence, charge) tuples
    frag_names: Optional[List[List[str]]]
    library: Dict[Tuple, Dict]
    spec_row_indices_split: List[np.ndarray]
    spec_col_indices_split: List[np.ndarray]
    spec_values_split: List[np.ndarray]
    dia_spectrum: np.ndarray
    residuals: Optional[np.ndarray]
    y_pred: Optional[np.ndarray]
    lib_coefficients: np.ndarray


def calculate_spectral_features(inputs: SpectralFeatureInputs) -> np.ndarray:
    """
    Calculate spectral similarity and quality features.
    
    Features calculated:
    - Features 12-15: Hyperscore components (hyperscore, b_count, y_count, longest_y)
    - Feature 16: SCRIBE score
    - Features 17-18: Residual features (max unmatched, max matched)
    - Feature 19: Goodness of fit
    - Features 20-21: Manhattan distance and spectral contrast
    - Feature 24: Large coefficient cosine similarity
    
    Args:
        inputs: SpectralFeatureInputs containing all required data
        
    Returns:
        Array of shape (n_candidates, 10) containing spectral features
    """
    features = np.zeros((inputs.n_candidates, 10))
    
    # Calculate hyperscore features (12-15)
    hyperscore_features = calculate_hyperscore_features(
        inputs.n_candidates,
        inputs.candidates,
        inputs.frag_names,
        inputs.library
    )
    features[:, 0:4] = hyperscore_features
    
    # Calculate SCRIBE scores (Feature 16)
    scribe_scores = calculate_scribe_scores(
        inputs.n_candidates,
        inputs.candidates,
        inputs.library
    )
    features[:, 4] = scribe_scores
    
    # Calculate residual features (17-18)
    residual_features = calculate_residual_features(
        inputs.n_candidates,
        inputs.spec_row_indices_split,
        inputs.residuals,
        inputs.y_pred
    )
    features[:, 5:7] = residual_features
    
    # Goodness of fit (Feature 19) - placeholder for now
    features[:, 7] = 0
    
    # Calculate manhattan distance and spectral contrast (20-21)
    if inputs.residuals is not None and inputs.y_pred is not None:
        manhattan_features = calculate_manhattan_features(
            inputs.spec_row_indices_split,
            inputs.spec_col_indices_split,
            inputs.spec_values_split,
            inputs.dia_spectrum[:, 1],
            inputs.y_pred
        )
        features[:, 8:10] = manhattan_features.T
    
    return features


def calculate_hyperscore_features(
    n_candidates: int,
    candidates: List[Tuple],
    frag_names: Optional[List[List[str]]],
    library: Dict[Tuple, Dict]
) -> np.ndarray:
    """
    Calculate hyperscore-related features.
    
    Returns array of shape (n_candidates, 4) with:
    - hyperscore
    - b_count
    - y_count
    - longest_y_ion_series
    """
    features = np.zeros((n_candidates, 4))
    
    if frag_names is None:
        return features
    
    for i in range(n_candidates):
        # Get fragment ion types
        if i < len(frag_names) and len(frag_names[i]) > 0:
            b_ions, y_ions = parse_fragment_ions(frag_names[i])
            
            # Calculate hyperscore components
            b_count = len(b_ions)
            y_count = len(y_ions)
            hyperscore = calculate_hyperscore(b_count, y_count)
            longest_y = calculate_longest_y_series(y_ions)
            
            features[i, 0] = hyperscore
            features[i, 1] = b_count
            features[i, 2] = y_count
            features[i, 3] = longest_y
    
    return features


def calculate_scribe_scores(
    n_candidates: int,
    candidates: List[Tuple],
    library: Dict[Tuple, Dict]
) -> np.ndarray:
    """Calculate SCRIBE scores for all candidates."""
    scores = np.zeros(n_candidates)
    
    for i in range(n_candidates):
        if i < len(candidates):
            candidate = candidates[i]
            if candidate in library and 'scribe' in library[candidate]:
                scores[i] = library[candidate]['scribe']
    
    return scores


def calculate_residual_features(
    n_candidates: int,
    spec_row_indices_split: List[np.ndarray],
    residuals: Optional[np.ndarray],
    y_pred: Optional[np.ndarray]
) -> np.ndarray:
    """
    Calculate residual-based features.
    
    Returns array of shape (n_candidates, 2) with:
    - max_unmatched_residual
    - max_matched_residual
    """
    features = np.zeros((n_candidates, 2))
    
    if residuals is None or y_pred is None:
        return features
    
    for i in range(n_candidates):
        row_indices = spec_row_indices_split[i]
        if len(row_indices) > 0:
            # Max unmatched residual
            candidate_residuals = residuals[row_indices]
            if len(candidate_residuals) > 0:
                features[i, 0] = np.max(np.abs(candidate_residuals))
            
            # Max matched residual (where prediction > 0)
            matched_mask = y_pred[row_indices] > 0
            if np.any(matched_mask):
                matched_residuals = candidate_residuals[matched_mask]
                features[i, 1] = np.max(np.abs(matched_residuals))
    
    return features


def calculate_manhattan_features(
    spec_row_indices_split: List[np.ndarray],
    spec_col_indices_split: List[np.ndarray],
    spec_values_split: List[np.ndarray],
    dia_intensities: np.ndarray,
    y_pred: np.ndarray
) -> np.ndarray:
    """
    Calculate Manhattan distance and spectral contrast.
    
    Returns array of shape (2, n_candidates) with:
    - manhattan distances
    - spectral contrasts
    """
    n_candidates = len(spec_row_indices_split)
    manhattan_distances = np.zeros(n_candidates)
    spectral_contrasts = np.zeros(n_candidates)
    
    for i in range(n_candidates):
        if len(spec_row_indices_split[i]) == 0:
            manhattan_distances[i] = -np.inf
            spectral_contrasts[i] = 0
            continue
        
        # Get values for this candidate
        rows = spec_row_indices_split[i]
        obs_vals = dia_intensities[rows]
        pred_vals = y_pred[rows]
        
        # Manhattan distance
        if np.sum(obs_vals) > 0:
            manhattan_distances[i] = np.log10(
                np.sum(np.abs(pred_vals - obs_vals)) / np.sum(obs_vals)
            )
        else:
            manhattan_distances[i] = -np.inf
        
        # Spectral contrast
        if np.sum(pred_vals) > 0 and np.sum(obs_vals) > 0:
            # Normalize and compute angle
            pred_norm = pred_vals / np.sqrt(np.sum(pred_vals**2))
            obs_norm = obs_vals / np.sqrt(np.sum(obs_vals**2))
            dot_product = np.clip(np.sum(pred_norm * obs_norm), -1, 1)
            spectral_contrasts[i] = 1 - (2 * np.arccos(dot_product) / np.pi)
        else:
            spectral_contrasts[i] = 0
    
    return np.array([manhattan_distances, spectral_contrasts])


def parse_fragment_ions(frag_names: List[str]) -> Tuple[List[str], List[str]]:
    """Parse fragment names into b and y ions."""
    b_ions = []
    y_ions = []
    
    for frag in frag_names:
        if frag.startswith('b'):
            b_ions.append(frag)
        elif frag.startswith('y'):
            y_ions.append(frag)
    
    return b_ions, y_ions


def calculate_hyperscore(b_count: int, y_count: int) -> float:
    """Calculate hyperscore from b and y ion counts."""
    if b_count == 0 or y_count == 0:
        return 0.0
    
    # Simplified hyperscore calculation
    # In practice, this would include intensity information
    return np.log(b_count * y_count + 1)


def calculate_longest_y_series(y_ions: List[str]) -> int:
    """Calculate the longest consecutive y-ion series."""
    if not y_ions:
        return 0
    
    # Extract y-ion numbers
    y_numbers = []
    for ion in y_ions:
        try:
            # Extract number from y-ion (e.g., 'y5' -> 5)
            num = int(''.join(filter(str.isdigit, ion)))
            y_numbers.append(num)
        except ValueError:
            continue
    
    if not y_numbers:
        return 0
    
    # Sort and find longest consecutive sequence
    y_numbers.sort()
    max_length = 1
    current_length = 1
    
    for i in range(1, len(y_numbers)):
        if y_numbers[i] == y_numbers[i-1] + 1:
            current_length += 1
            max_length = max(max_length, current_length)
        else:
            current_length = 1
    
    return max_length