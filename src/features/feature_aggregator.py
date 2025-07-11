"""
Feature aggregator for combining all feature types.

This module provides a unified interface for calculating all features
needed for spectral fitting scoring.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

from .basic_features import BasicFeatureInputs, calculate_basic_features
from .ms1_features import MS1FeatureInputs, calculate_ms1_features
from .spectral_features import SpectralFeatureInputs, calculate_spectral_features
from .statistical_features import StatisticalFeatureInputs, calculate_statistical_features


@dataclass
class FeatureCalculatorInputs:
    """Complete inputs for all feature calculations."""
    # Common inputs
    candidates: List[Tuple]
    peaks_in_dia: List[int]
    is_decoy_matched: np.ndarray
    spec_values_split: List[np.ndarray]
    spec_row_indices_split: List[np.ndarray]
    spec_col_indices_split: List[np.ndarray]
    lib_peaks_matched: List[np.ndarray]
    
    # Spectrum data
    dia_spectrum: np.ndarray
    prec_rt: float
    lib_coefficients: np.ndarray
    
    # Reference data
    rt_mz: np.ndarray
    window_idxs: np.ndarray
    library: Dict[Tuple, Dict]
    
    # Additional data
    ms1_error_array: Optional[np.ndarray]
    frag_names: Optional[List[List[str]]]
    sparse_matrix: Optional[Any]
    residuals: Optional[np.ndarray]
    y_pred: Optional[np.ndarray]
    
    # Optional parameters
    unique_row_idxs: Optional[np.ndarray] = None
    peak_idx_convertor: Optional[Dict[int, int]] = None


class FeatureCalculator:
    """
    Unified feature calculator for spectral fitting.
    
    This class coordinates the calculation of all features using
    specialized modules for different feature types.
    """
    
    def __init__(self):
        """Initialize the feature calculator."""
        self.feature_names = self._get_feature_names()
    
    def calculate_all_features(self, inputs: FeatureCalculatorInputs) -> np.ndarray:
        """
        Calculate all features for the given candidates.
        
        Args:
            inputs: Complete inputs for feature calculation
            
        Returns:
            Feature matrix of shape (n_candidates, 26)
        """
        n_candidates = len(inputs.peaks_in_dia)
        
        if n_candidates == 0:
            return np.zeros((0, 26))
        
        # Initialize feature matrix
        features = np.zeros((n_candidates, 26))
        
        # Calculate DIA total intensity
        dia_total_intensity = np.sum(inputs.dia_spectrum[:, 1])
        
        # Prepare MS1 errors
        ms1_error = self._prepare_ms1_errors(
            inputs.ms1_error_array,
            inputs.peaks_in_dia,
            n_candidates
        )
        
        # 1. Calculate basic features (0-6)
        basic_inputs = BasicFeatureInputs(
            n_candidates=n_candidates,
            peaks_in_dia=inputs.peaks_in_dia,
            lib_peaks_matched=inputs.lib_peaks_matched,
            spec_values_split=inputs.spec_values_split,
            spec_row_indices_split=inputs.spec_row_indices_split,
            dia_spectrum=inputs.dia_spectrum,
            dia_total_intensity=dia_total_intensity,
            lib_coefficients=inputs.lib_coefficients,
            ms1_error=ms1_error,
            rt_mz=inputs.rt_mz,
            window_idxs=inputs.window_idxs,
            prec_rt=inputs.prec_rt
        )
        basic_features = calculate_basic_features(basic_inputs)
        features[:, 0:7] = basic_features
        
        # 2. Calculate MS1 features (7-11)
        if inputs.sparse_matrix is not None and inputs.unique_row_idxs is not None:
            ms1_inputs = MS1FeatureInputs(
                n_candidates=n_candidates,
                sparse_matrix=inputs.sparse_matrix,
                lib_coefficients=inputs.lib_coefficients,
                dia_spectrum=inputs.dia_spectrum,
                unique_row_idxs=inputs.unique_row_idxs,
                spec_row_indices_split=inputs.spec_row_indices_split,
                spec_col_indices_split=inputs.spec_col_indices_split,
                spec_values_split=inputs.spec_values_split,
                peak_idx_convertor=inputs.peak_idx_convertor or {}
            )
            ms1_features = calculate_ms1_features(ms1_inputs)
            features[:, 7:12] = ms1_features
        
        # 3. Calculate spectral features (12-21)
        spectral_inputs = SpectralFeatureInputs(
            n_candidates=n_candidates,
            candidates=inputs.candidates,
            frag_names=inputs.frag_names,
            library=inputs.library,
            spec_row_indices_split=inputs.spec_row_indices_split,
            spec_col_indices_split=inputs.spec_col_indices_split,
            spec_values_split=inputs.spec_values_split,
            dia_spectrum=inputs.dia_spectrum,
            residuals=inputs.residuals,
            y_pred=inputs.y_pred,
            lib_coefficients=inputs.lib_coefficients
        )
        spectral_features = calculate_spectral_features(spectral_inputs)
        features[:, 12:22] = spectral_features
        
        # 4. Calculate statistical features (22-25)
        statistical_inputs = StatisticalFeatureInputs(
            n_candidates=n_candidates,
            peaks_in_dia=inputs.peaks_in_dia,
            basic_features=basic_features,
            lib_coefficients=inputs.lib_coefficients,
            rt_mz=inputs.rt_mz,
            window_idxs=inputs.window_idxs
        )
        statistical_features = calculate_statistical_features(statistical_inputs)
        features[:, 22:25] = statistical_features[:, :3]
        
        # Feature 24 (large coefficient cosine) is placeholder
        features[:, 24] = 0
        
        # Feature 25 is from statistical features
        features[:, 25] = statistical_features[:, 2]
        
        return features
    
    def _prepare_ms1_errors(
        self,
        ms1_error_array: Optional[np.ndarray],
        peaks_in_dia: List[int],
        n_candidates: int
    ) -> np.ndarray:
        """Prepare MS1 errors for the matched candidates."""
        ms1_error = np.zeros(n_candidates)
        
        if ms1_error_array is not None:
            for i, candidate_idx in enumerate(peaks_in_dia):
                if candidate_idx < len(ms1_error_array):
                    ms1_error[i] = ms1_error_array[candidate_idx]
        
        return ms1_error
    
    def _get_feature_names(self) -> List[str]:
        """Get the names of all features."""
        return [
            "num_lib_peaks_matched",         # 0
            "frac_lib_intensity",            # 1
            "frac_dia_intensity",            # 2
            "rel_error",                     # 3
            "rt_error",                      # 4
            "frac_int_matched",              # 5
            "frac_int_pred",                 # 6
            "r2all",                         # 7
            "r2_lib_spec",                   # 8
            "r2_unique",                     # 9
            "frac_unique_pred",              # 10
            "frac_dia_intensity_pred",       # 11
            "hyperscores",                   # 12
            "b_counts",                      # 13
            "y_counts",                      # 14
            "longest_y_ions",                # 15
            "scribe_scores",                 # 16
            "max_unmatched_residuals",       # 17
            "max_matched_residuals",         # 18
            "gof_stats",                     # 19
            "manhattan_distances",           # 20
            "fitted_spectral_contrasts",     # 21
            "frac_int_matched_pred",         # 22
            "frac_int_matched_pred_sigcoeff",# 23
            "large_coeff_cosine",            # 24
            "rt_mz"                          # 25
        ]


def aggregate_features(
    basic: np.ndarray,
    ms1: np.ndarray,
    spectral: np.ndarray,
    statistical: np.ndarray
) -> np.ndarray:
    """
    Aggregate features from different modules into a single matrix.
    
    Args:
        basic: Basic features (n_candidates x 7)
        ms1: MS1 features (n_candidates x 5)
        spectral: Spectral features (n_candidates x 10)
        statistical: Statistical features (n_candidates x 3)
        
    Returns:
        Complete feature matrix (n_candidates x 26)
    """
    n_candidates = basic.shape[0]
    features = np.zeros((n_candidates, 26))
    
    # Copy features in order
    features[:, 0:7] = basic
    features[:, 7:12] = ms1
    features[:, 12:22] = spectral
    features[:, 22:25] = statistical
    
    # Feature 24 is placeholder
    features[:, 24] = 0
    
    # Feature 25 is in statistical
    features[:, 25] = statistical[:, 2]
    
    return features