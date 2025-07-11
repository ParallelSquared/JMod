"""
Unified feature calculator for spectral matching.

This module provides a clean interface to calculate all features
using the modular feature functions.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from scipy import sparse

from .intensity_features import (
    calculate_num_peaks_matched,
    calculate_fraction_library_intensity,
    calculate_fraction_intensity_matched_csc,
    calculate_fraction_intensity_predicted
)
from .error_features import (
    calculate_ms1_error,
    calculate_rt_error
)
from .correlation_features import (
    calculate_r2_all,
    calculate_r2_lib_spec_csc,
    calculate_unique_peak_features_csc
)
from .fragment_features import calculate_fragment_features
from .scoring_features import calculate_all_scoring_features_csc

# Note: calculate_frac_dia_intensity_csc imported dynamically to avoid circular imports


FEATURE_NAMES = [
    "num_lib_peaks_matched", "frac_lib_intensity", "frac_dia_intensity",
    "rel_error", "rt_error", "frac_int_matched", "frac_int_pred",
    "r2all", "r2_lib_spec", "r2_unique", "frac_unique_pred",
    "frac_dia_intensity_pred", "hyperscores", "b_counts", "y_counts",
    "longest_y_ions", "scribe_scores", "max_unmatched_residuals",
    "max_matched_residuals", "gof_stats", "manhattan_distances",
    "fitted_spectral_contrasts", "frac_int_matched_pred",
    "frac_int_matched_pred_sigcoeff", "large_coeff_cosine", "rt_mz"
]


@dataclass
class FeatureCalculatorInputs:
    """Container for all inputs needed for feature calculation."""
    # Candidate information
    candidates: List[Any]
    peaks_in_dia: List[int]
    is_decoy_matched: np.ndarray
    
    # Matrix data
    spec_values_split: List[np.ndarray]
    spec_row_indices_split: List[np.ndarray]
    spec_col_indices_split: List[np.ndarray]
    
    # Additional outputs
    lib_peaks_matched: List[np.ndarray]
    ms1_error_array: Optional[np.ndarray]
    frag_names: Optional[List[np.ndarray]]
    
    # Spectrum data
    dia_spectrum: np.ndarray
    prec_rt: float
    lib_coefficients: np.ndarray
    
    # Reference data
    rt_mz: np.ndarray
    window_idxs: np.ndarray
    library: Dict
    
    # Sparse matrix data (required)
    sparse_matrix_csc: sparse.csc_matrix
    
    # Optional data
    residuals: Optional[np.ndarray] = None
    y_pred: Optional[np.ndarray] = None


class FeatureCalculator:
    """Unified feature calculator with modular design."""
    
    def __init__(self):
        """Initialize the feature calculator."""
        self.feature_names = FEATURE_NAMES
    
    def calculate_all_features(self, inputs: FeatureCalculatorInputs) -> np.ndarray:
        """
        Calculate all 26 features for spectral matching.
        
        Args:
            inputs: FeatureCalculatorInputs object with all required data
            
        Returns:
            Feature matrix of shape (n_candidates, 26)
        """
        n_candidates = len(inputs.peaks_in_dia)
        if n_candidates == 0:
            return np.zeros((0, 26))
        
        # Initialize feature matrix
        features = np.zeros((n_candidates, 26))
        
        # Create padded DIA spectrum to handle penalty rows
        # Penalty rows (indices >= len(dia_spectrum)) should have intensity 0
        sparse_matrix_rows = inputs.sparse_matrix_csc.shape[0]
        original_dia_rows = len(inputs.dia_spectrum)
        
        if sparse_matrix_rows > original_dia_rows:
            # Pad DIA spectrum with zeros for penalty rows
            padding_rows = sparse_matrix_rows - original_dia_rows
            padding = np.zeros((padding_rows, inputs.dia_spectrum.shape[1]))
            dia_spectrum_padded = np.vstack([inputs.dia_spectrum, padding])
            
            # Also pad y_pred if it exists (predicted spectrum from NNLS)
            if inputs.y_pred is not None:
                y_pred_padding = np.zeros(padding_rows)
                y_pred_padded = np.concatenate([inputs.y_pred, y_pred_padding])
            else:
                y_pred_padded = None
            
            # Also pad residuals if they exist (NNLS residuals)
            if inputs.residuals is not None:
                residuals_padding = np.zeros(padding_rows)
                residuals_padded = np.concatenate([inputs.residuals, residuals_padding])
            else:
                residuals_padded = None
        else:
            # No padding needed
            dia_spectrum_padded = inputs.dia_spectrum
            y_pred_padded = inputs.y_pred
            residuals_padded = inputs.residuals
        
        # Feature 0: Number of library peaks matched
        features[:, 0] = calculate_num_peaks_matched(inputs.lib_peaks_matched)
        
        # Feature 1: Fraction of library intensity
        features[:, 1] = calculate_fraction_library_intensity(inputs.spec_values_split)
        
        # Feature 2: Fraction of DIA intensity
        # Dynamic import to avoid circular import with spectral_fitting
        from ..spectral_fitting import calculate_frac_dia_intensity_csc
        tic = np.sum(dia_spectrum_padded[:, 1])
        features[:, 2] = calculate_frac_dia_intensity_csc(
            inputs.sparse_matrix_csc,
            dia_spectrum_padded,
            tic
        )
        
        # Feature 3: MS1 relative error
        features[:, 3] = calculate_ms1_error(
            inputs.ms1_error_array,
            inputs.peaks_in_dia
        )
        
        # Feature 4: RT error
        features[:, 4] = calculate_rt_error(
            inputs.prec_rt,
            inputs.rt_mz,
            inputs.window_idxs,
            inputs.peaks_in_dia
        )
        
        # Feature 5: Fraction intensity matched
        features[:, 5] = calculate_fraction_intensity_matched_csc(
            inputs.sparse_matrix_csc,
            inputs.lib_coefficients
        )
        
        # Feature 6: Fraction intensity predicted
        features[:, 6] = calculate_fraction_intensity_predicted(
            features[:, 5],
            inputs.lib_coefficients,
            n_candidates
        )
        
        # Features 7-9: Correlation features
        # Calculate predicted spectrum if not provided
        if y_pred_padded is not None:
            predicted_spec = y_pred_padded
        else:
            # Simple prediction without full matrix
            predicted_spec = np.zeros(len(dia_spectrum_padded))
        
        features[:, 7] = calculate_r2_all(
            dia_spectrum_padded[:, 1],
            predicted_spec,
            n_candidates
        )
        
        features[:, 8] = calculate_r2_lib_spec_csc(
            inputs.sparse_matrix_csc,
            dia_spectrum_padded
        )
        
        # Features 9-10: Unique peak features
        r2_unique, frac_unique_pred = calculate_unique_peak_features_csc(
            inputs.sparse_matrix_csc,
            inputs.lib_coefficients,
            dia_spectrum_padded
        )
        features[:, 9] = r2_unique
        features[:, 10] = frac_unique_pred
        
        # Feature 11: Fraction DIA intensity predicted
        for i in range(n_candidates):
            if i < len(inputs.lib_coefficients) and features[i, 2] > 0:
                features[i, 11] = features[i, 1] * inputs.lib_coefficients[i] / features[i, 2]
            else:
                features[i, 11] = 0.0
        
        # Features 12-15: Fragment features
        if inputs.frag_names is not None:
            hyperscores, b_counts, y_counts, longest_y = calculate_fragment_features(
                inputs.frag_names,
                inputs.lib_peaks_matched,
                inputs.library,
                inputs.candidates
            )
            features[:, 12] = hyperscores
            features[:, 13] = b_counts
            features[:, 14] = y_counts
            features[:, 15] = longest_y
        
        # Features 16-19 and 17-18: Scoring and residual features (combined for efficiency)
        scribe, manhattan, contrast, gof, max_unmatched, max_matched = calculate_all_scoring_features_csc(
            inputs.sparse_matrix_csc,
            dia_spectrum_padded,
            residuals_padded,
            y_pred_padded,
            inputs.lib_coefficients
        )
        features[:, 16] = scribe
        features[:, 20] = manhattan
        features[:, 21] = contrast
        features[:, 19] = gof
        features[:, 17] = max_unmatched
        features[:, 18] = max_matched
        
        # Feature 22: Same as feature 5
        features[:, 22] = features[:, 5]
        
        # Feature 23: Feature 5 with significance cutoff
        for i in range(n_candidates):
            if i < len(inputs.lib_coefficients) and inputs.lib_coefficients[i] > 0.1:
                features[i, 23] = features[i, 5]
            else:
                features[i, 23] = 0.0
        
        # Feature 24: Placeholder (always 0)
        features[:, 24] = 0
        
        # Feature 25: m/z value
        for i, candidate_idx in enumerate(inputs.peaks_in_dia):
            if candidate_idx < len(inputs.window_idxs):
                features[i, 25] = inputs.rt_mz[inputs.window_idxs[candidate_idx], 1]
            else:
                features[i, 25] = 0.0
        
        return features
    
    def calculate_rt_alignment_features(self, inputs: FeatureCalculatorInputs) -> np.ndarray:
        """
        Calculate features specifically for RT alignment.
        
        This is a specialized version that matches the original
        calculate_rt_alignment_features function behavior.
        
        Args:
            inputs: FeatureCalculatorInputs object
            
        Returns:
            Feature matrix for RT alignment
        """
        # RT alignment uses the same features but with some differences
        # in how they're calculated (e.g., no decoys)
        return self.calculate_all_features(inputs)