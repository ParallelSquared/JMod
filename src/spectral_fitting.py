"""
Spectral fitting module for JMod proteomics software.

This module implements the core spectral matching algorithm that processes
both target and decoy peptides together in a unified approach. It performs
joint modeling of mass spectra to deconvolve overlapping isotopic envelopes
in DIA (Data-Independent Acquisition) MS/MS data.

Key Components:
- UnifiedCandidates: Data structure combining targets and decoys
- create_entries: Processes candidates and matches to DIA spectrum
- Matrix construction: Builds sparse matrices for NNLS optimization
- Feature calculation: Computes scoring features for FDR analysis
- fit_to_lib/fit_to_lib2: Main entry points for spectral matching

This Source Code Form is subject to the terms of the Oxford Nanopore
Technologies, Ltd. Public License, v. 1.0.  Full licence can be found
at https://github.com/ParallelSquared/JMod/blob/main/LICENSE.txt
"""


import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any, Union

import warnings
import ptinnls as sparse_nnls

from scipy import stats
from scipy import sparse
import re
from .utils.io.read_output import names
import src.config as config

from .utils.misc_functions import createTolWindows, window_width, feature_list_mz, feature_list_rt, \
hyperscore_b_y, longest_y, closest_ms1spec, closest_peak_diff, cosim, np_pearson_cor, ms1_error
from .utils.spectral_similarity_metrics import (
    get_closest_ms1, get_scribe, get_scribe_csc, get_residuals, get_residuals_csc, max_matched_residual, 
    gof_stat, gof_stat_csc, get_manhattan_distance, get_manhattan_distance_csc
)
from .features.feature_calculator import FeatureCalculator, FeatureCalculatorInputs


# ===== DATA STRUCTURES =====

@dataclass
class UnifiedCandidates:
    """
    Unified structure for both target and decoy candidates.
    
    Attributes:
        candidates: List of candidate tuples (seq, charge, ...) for both targets and decoys
        is_decoy: Boolean array indicating which candidates are decoys
        peaks: List of peak arrays for each candidate
        ms1_error: Array of MS1 errors for each candidate
        peaks_in_dia: Indices of candidates that have peaks in DIA spectrum
    """
    candidates: List[Tuple]
    is_decoy: np.ndarray
    peaks: List[np.ndarray]
    ms1_error: Optional[np.ndarray] = None
    peaks_in_dia: Optional[List[int]] = None
    
    def __post_init__(self):
        """Validate that arrays have consistent lengths."""
        n = len(self.candidates)
        assert len(self.is_decoy) == n, "is_decoy array must match candidates length"
        assert len(self.peaks) == n, "peaks list must match candidates length"
        if self.ms1_error is not None:
            assert len(self.ms1_error) == n, "ms1_error must match candidates length"
    
    @property
    def n_targets(self) -> int:
        """Number of target candidates."""
        return np.sum(~self.is_decoy)
    
    @property
    def n_decoys(self) -> int:
        """Number of decoy candidates."""
        return np.sum(self.is_decoy)
    
    def get_targets(self) -> 'UnifiedCandidates':
        """Return a new UnifiedCandidates with only targets."""
        target_mask = ~self.is_decoy
        target_indices = np.where(target_mask)[0]
        return UnifiedCandidates(
            candidates=[self.candidates[i] for i in target_indices],
            is_decoy=self.is_decoy[target_mask],
            peaks=[self.peaks[i] for i in target_indices],
            ms1_error=self.ms1_error[target_mask] if self.ms1_error is not None else None,
            peaks_in_dia=[i for i in self.peaks_in_dia if target_mask[i]] if self.peaks_in_dia else None
        )
    
    def get_decoys(self) -> 'UnifiedCandidates':
        """Return a new UnifiedCandidates with only decoys."""
        decoy_mask = self.is_decoy
        decoy_indices = np.where(decoy_mask)[0]
        return UnifiedCandidates(
            candidates=[self.candidates[i] for i in decoy_indices],
            is_decoy=self.is_decoy[decoy_mask],
            peaks=[self.peaks[i] for i in decoy_indices],
            ms1_error=self.ms1_error[decoy_mask] if self.ms1_error is not None else None,
            peaks_in_dia=[i for i in self.peaks_in_dia if decoy_mask[i]] if self.peaks_in_dia else None
        )


@dataclass
class UnifiedMatrixData:
    """
    Unified sparse matrix data for NNLS optimization.
    
    Attributes:
        row_indices: Row indices for sparse matrix (DIA spectrum peaks)
        col_indices: Column indices for sparse matrix (library candidates)
        values: Intensity values for sparse matrix
        is_decoy: Boolean array tracking which columns are decoys
        row_indices_split: List of row indices per candidate
        col_indices_split: List of column indices per candidate
        values_split: List of values per candidate
    """
    row_indices: np.ndarray
    col_indices: np.ndarray
    values: np.ndarray
    is_decoy: np.ndarray
    row_indices_split: Optional[List[np.ndarray]] = None
    col_indices_split: Optional[List[np.ndarray]] = None
    values_split: Optional[List[np.ndarray]] = None
    
    def __post_init__(self):
        """Validate array consistency."""
        n = len(self.row_indices)
        assert len(self.col_indices) == n, "col_indices must match row_indices length"
        assert len(self.values) == n, "values must match indices length"
        
    @property
    def n_cols(self) -> int:
        """Number of columns (candidates) in matrix."""
        return len(np.unique(self.col_indices))
    
    def get_target_indices(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return row/col/values for target candidates only."""
        target_cols = np.where(~self.is_decoy)[0]
        mask = np.isin(self.col_indices, target_cols)
        return self.row_indices[mask], self.col_indices[mask], self.values[mask]
    
    def get_decoy_indices(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return row/col/values for decoy candidates only."""
        decoy_cols = np.where(self.is_decoy)[0]
        mask = np.isin(self.col_indices, decoy_cols)
        return self.row_indices[mask], self.col_indices[mask], self.values[mask]


@dataclass
class UnifiedFeatures:
    """
    Unified feature matrix for all candidates.
    
    Attributes:
        features: Feature matrix (n_candidates x n_features)
        is_decoy: Boolean array indicating which rows are decoys
        feature_names: Optional list of feature names
    """
    features: np.ndarray
    is_decoy: np.ndarray
    feature_names: Optional[List[str]] = None
    
    def __post_init__(self):
        """Validate dimensions."""
        assert len(self.features) == len(self.is_decoy), "features and is_decoy must have same length"
        if self.feature_names is not None:
            assert len(self.feature_names) == self.features.shape[1], "feature_names must match number of features"
    
    def get_target_features(self) -> np.ndarray:
        """Return features for targets only."""
        return self.features[~self.is_decoy]
    
    def get_decoy_features(self) -> np.ndarray:
        """Return features for decoys only."""
        return self.features[self.is_decoy]


def create_unified_candidates(
    target_candidates: List[Tuple],
    target_peaks: List[np.ndarray],
    decoy_candidates: Optional[List[Tuple]] = None,
    decoy_peaks: Optional[List[np.ndarray]] = None,
    target_ms1_error: Optional[np.ndarray] = None,
    decoy_ms1_error: Optional[np.ndarray] = None
) -> UnifiedCandidates:
    """
    Create a unified candidates structure from separate target and decoy data.
    
    Args:
        target_candidates: List of target candidate tuples
        target_peaks: List of peak arrays for targets
        decoy_candidates: Optional list of decoy candidate tuples
        decoy_peaks: Optional list of peak arrays for decoys
        target_ms1_error: Optional MS1 errors for targets
        decoy_ms1_error: Optional MS1 errors for decoys
        
    Returns:
        UnifiedCandidates object combining all data
    """
    if decoy_candidates is None:
        # No decoys, just wrap targets
        return UnifiedCandidates(
            candidates=target_candidates,
            is_decoy=np.zeros(len(target_candidates), dtype=bool),
            peaks=target_peaks,
            ms1_error=target_ms1_error
        )
    
    # Combine targets and decoys
    all_candidates = target_candidates + decoy_candidates
    all_peaks = target_peaks + decoy_peaks
    
    # Create boolean array
    is_decoy = np.array([False] * len(target_candidates) + [True] * len(decoy_candidates))
    
    # Combine MS1 errors if provided
    ms1_error = None
    if target_ms1_error is not None:
        if decoy_ms1_error is not None:
            ms1_error = np.concatenate([target_ms1_error, decoy_ms1_error])
        else:
            # Only targets have MS1 error
            ms1_error = np.concatenate([target_ms1_error, np.zeros(len(decoy_candidates))])
    
    return UnifiedCandidates(
        candidates=all_candidates,
        is_decoy=is_decoy,
        peaks=all_peaks,
        ms1_error=ms1_error
    )


def create_entries(
    centroid_breaks: np.ndarray,
    unified_candidates: UnifiedCandidates,
    top_n: int = 10,
    atleast_m: int = 3,
    prec_mzs: Optional[np.ndarray] = None,
    ms1_spec: Optional[Any] = None,
    ms1_tol: float = 25.,
    spec_frags: Optional[List] = None,
    top_n_idxs: Optional[List[np.ndarray]] = None,
    frac_matched: float = 0.25,
    library: Optional[Dict] = None,
    bin_centers: Optional[np.ndarray] = None,
    dia_spectrum: Optional[np.ndarray] = None
) -> Tuple[UnifiedCandidates, UnifiedMatrixData, Dict[str, Any]]:
    """
    Create entries for spectral fitting, processing targets and decoys together.
    
    Args:
        centroid_breaks: Sorted array of m/z bin boundaries
        unified_candidates: UnifiedCandidates object with all candidates
        top_n: Number of top intensity peaks to consider
        atleast_m: Minimum number of matched peaks required
        prec_mzs: Precursor m/z values for candidates
        ms1_spec: MS1 spectrum for precursor matching
        ms1_tol: MS1 mass tolerance
        spec_frags: Optional fragment specifications
        top_n_idxs: Pre-computed top N peak indices
        frac_matched: Minimum fraction of intensity matched
        library: Spectral library for fragment information
        decoy_library: Decoy spectral library
        bin_centers: Center m/z values for bins
        dia_spectrum: DIA spectrum data
        
    Returns:
        Tuple of:
        - Updated UnifiedCandidates with peaks_in_dia filled
        - UnifiedMatrixData with sparse matrix data
        - Dictionary with additional outputs (lib_peaks_matched, norm_intensities, etc.)
    """
    # Extract data from unified structure
    candidate_peaks = unified_candidates.peaks
    candidates = unified_candidates.candidates
    is_decoy = unified_candidates.is_decoy
    n_candidates = len(candidates)
    
    # Calculate coordinates and top peaks for all candidates
    ref_coords = [np.searchsorted(centroid_breaks, M[:, 0]) for M in candidate_peaks]
    
    if top_n_idxs is None:
        top_ten = [np.searchsorted(centroid_breaks, 
                                   M[np.argsort(-M[:, 1])[0:min(top_n, M.shape[0])], 0]) 
                   for M in candidate_peaks]
    else:
        top_ten = [np.searchsorted(centroid_breaks, M[top_n_idxs[i], 0]) 
                   for i, M in enumerate(candidate_peaks)]
    
    # MS1 filtering - vectorized
    ms1_peak = np.ones(n_candidates, dtype=bool)  # Default to True
    if ms1_spec is not None and prec_mzs is not None:
        # Vectorized MS1 difference calculation
        ms1_diffs = np.array([closest_peak_diff(mz, ms1_spec.mz) for mz in prec_mzs])
        ms1_peak = ~np.isnan(ms1_diffs)
        
        # Vectorized error calculation
        ms1_error = np.where(
            ~np.isnan(ms1_diffs),
            ms1_diffs / prec_mzs * 1e6,
            np.nan
        )
    else:
        ms1_error = np.zeros(n_candidates)
    
    # Normalize intensities
    all_norm_intensities = [M[:, 1] / np.sum(M[:, 1]) for M in candidate_peaks]
    
    # Find candidates with peaks in DIA - vectorized approach
    peaks_in_dia = []
    for i in range(n_candidates):
        if not ms1_peak[i] or len(top_ten[i]) == 0:
            continue
        
        # Vectorized checks
        ref_coords_i = ref_coords[i]
        top_ten_i = top_ten[i]
        in_dia_mask = (ref_coords_i % 2) == 1
        
        # Check all conditions
        if (np.sum(all_norm_intensities[i][in_dia_mask]) > frac_matched and
            np.sum(top_ten_i % 2) > atleast_m and
            top_ten_i[0] % 2 == 1 and
            np.sum(top_ten_i[:min(3, len(top_ten_i))] % 2 == 1) >= 2):
            peaks_in_dia.append(i)
    
    # Update unified candidates with peaks_in_dia
    unified_candidates.peaks_in_dia = peaks_in_dia
    unified_candidates.ms1_error = ms1_error
    
    # Filter to candidates with peaks in DIA
    if len(peaks_in_dia) == 0:
        # No matches, return empty results
        empty_matrix = UnifiedMatrixData(
            row_indices=np.array([], dtype=np.int32),
            col_indices=np.array([], dtype=np.int32),
            values=np.array([], dtype=np.float32),
            is_decoy=np.array([], dtype=bool),
            row_indices_split=[],
            col_indices_split=[],
            values_split=[]
        )
        return unified_candidates, empty_matrix, {
            'lib_peaks_matched': [],
            'norm_intensities': [],
            'pep_cand_loc': [],
            'pep_cand_list': []
        }
    
    # Extract data for matched candidates
    pep_cand_loc = [ref_coords[i] for i in peaks_in_dia]
    pep_cand_list = [candidate_peaks[i] for i in peaks_in_dia]
    pep_cand = [candidates[i] for i in peaks_in_dia]
    is_decoy_matched = is_decoy[peaks_in_dia]
    norm_intensities = [M[:, 1] / np.sum(M[:, 1]) for M in pep_cand_list]
    
    # Find which library peaks match DIA peaks
    lib_peaks_matched = [j % 2 == 1 for j in pep_cand_loc]
    
    # Build sparse matrix data
    spec_row_indices_split = [
        np.int32(((i[j] + 1) / 2) - 1) 
        for i, j in zip(pep_cand_loc, lib_peaks_matched)
    ]
    num_peaks_matched = np.array([np.sum(i) for i in lib_peaks_matched])
    spec_col_indices_split = [
        np.array([idx] * n, dtype=np.int32) 
        for idx, n in enumerate(num_peaks_matched)
    ]
    spec_values_split = [
        ints[matched] 
        for ints, matched in zip(norm_intensities, lib_peaks_matched)
    ]
    
    # Create unified matrix data
    if len(spec_row_indices_split) > 0:
        all_row_indices = np.concatenate(spec_row_indices_split)
        all_col_indices = np.concatenate(spec_col_indices_split)
        all_values = np.concatenate(spec_values_split)
    else:
        all_row_indices = np.array([], dtype=np.int32)
        all_col_indices = np.array([], dtype=np.int32)
        all_values = np.array([], dtype=np.float32)
    
    matrix_data = UnifiedMatrixData(
        row_indices=all_row_indices,
        col_indices=all_col_indices,
        values=all_values,
        is_decoy=is_decoy_matched,
        row_indices_split=spec_row_indices_split,
        col_indices_split=spec_col_indices_split,
        values_split=spec_values_split
    )
    
    # Calculate fragment data if needed
    frag_names = []
    frag_errors = []
    lib_frag_mz = []
    lib_frag_int = []
    obs_frag_int = []
    
    if library is not None and bin_centers is not None and dia_spectrum is not None:
        for i, cand in enumerate(pep_cand):
            if len(spec_row_indices_split[i]) > 0:
                # Get library entry from unified library
                lib_entry = library.get(cand, {})
                
                # Get fragment names
                if "ordered_frags" in lib_entry:
                    all_frag_names = np.array(lib_entry["ordered_frags"])
                    matched_frag_names = all_frag_names[lib_peaks_matched[i]]
                else:
                    matched_frag_names = np.array(["" for _ in range(np.sum(lib_peaks_matched[i]))])
                
                # Calculate fragment errors
                matched_bin_centers = bin_centers[spec_row_indices_split[i]]
                matched_lib_mz = pep_cand_list[i][:, 0][lib_peaks_matched[i]]
                matched_errors = (matched_bin_centers - matched_lib_mz) / matched_bin_centers
                
                # Get intensities
                matched_lib_int = pep_cand_list[i][:, 1][lib_peaks_matched[i]]
                matched_obs_int = dia_spectrum[spec_row_indices_split[i], 1]
                
                frag_names.append(matched_frag_names)
                frag_errors.append(matched_errors)
                lib_frag_mz.append(matched_lib_mz)
                lib_frag_int.append(matched_lib_int)
                obs_frag_int.append(matched_obs_int)
            else:
                # Empty data for candidates with no matches
                frag_names.append(np.array([]))
                frag_errors.append(np.array([]))
                lib_frag_mz.append(np.array([]))
                lib_frag_int.append(np.array([]))
                obs_frag_int.append(np.array([]))
    
    # Additional outputs for compatibility
    additional_outputs = {
        'lib_peaks_matched': lib_peaks_matched,
        'norm_intensities': norm_intensities,
        'pep_cand_loc': pep_cand_loc,
        'pep_cand_list': pep_cand_list,
        'pep_cand': pep_cand,
        'ms1_error_matched': ms1_error[peaks_in_dia],
        'frag_names': frag_names,
        'frag_errors': frag_errors,
        'lib_frag_mz': lib_frag_mz,
        'lib_frag_int': lib_frag_int,
        'obs_frag_int': obs_frag_int
    }
    
    return unified_candidates, matrix_data, additional_outputs


# ===== FEATURE CALCULATION FUNCTIONS =====

def compute_residuals(
    row_indices_split: List[np.ndarray],
    col_indices_split: List[np.ndarray],
    values_split: List[np.ndarray],
    is_decoy: np.ndarray,
    val_obs: np.ndarray,
    coeffs: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate residuals for all candidates.
    
    This function computes residuals without separating ref/decoy data.
    It uses the is_decoy mask to apply correct offsets internally.
    """
    coeffs = np.asarray(coeffs).ravel()
    N = len(val_obs)  # Number of rows in the sparse matrix
    
    # Initialize prediction array
    y_pred = np.zeros(N)
    
    # Compute predictions for all candidates
    #n_targets = np.sum(~is_decoy)
    
    for i, (row_idx, col_idx, vals) in enumerate(zip(row_indices_split, col_indices_split, values_split)):
        if len(row_idx) == 0:
            continue
            
        # Apply offset for decoys
        #if is_decoy[i]:
        #    adjusted_col_idx = col_idx - n_targets + n_targets  # This simplifies to just col_idx
        #    offset = n_targets
        #else:
        #    adjusted_col_idx = col_idx
        #    offset = 0
            
        # Compute predictions
        for r, c, v in zip(row_idx, col_idx, vals):
            #if c + offset < len(coeffs):
            y_pred[r] += v * coeffs[c]
    
    # Compute residuals
    residuals = val_obs - y_pred
    
    return residuals, y_pred


def compute_manhattan_distance(
    row_indices_split: List[np.ndarray],
    col_indices_split: List[np.ndarray],
    values_split: List[np.ndarray],
    val_obs: np.ndarray,
    y_pred: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate manhattan distance and spectral contrast for all candidates.
    
    Works with both targets and decoys in a single pass.
    """
    n_precursors = len(row_indices_split)
    manhattan_distances = np.zeros(n_precursors)
    spectral_contrasts = np.zeros(n_precursors)
    
    for i in range(n_precursors):
        if len(row_indices_split[i]) == 0:
            manhattan_distances[i] = -np.inf
            spectral_contrasts[i] = 0
            continue
            
        # Get values for this precursor
        rows = row_indices_split[i]
        obs_vals = val_obs[rows]
        pred_vals = y_pred[rows]
        
        # Manhattan distance
        if np.sum(obs_vals) > 0:
            manhattan_distances[i] = np.log10(np.sum(np.abs(pred_vals - obs_vals)) / np.sum(obs_vals))
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
    
    return manhattan_distances, spectral_contrasts


def calculate_features_original(
    unified_candidates: UnifiedCandidates,
    matrix_data: UnifiedMatrixData,
    additional_outputs: Dict,
    dia_spectrum: np.ndarray,
    prec_rt: float,
    lib_coefficients: np.ndarray,
    sparse_matrix,
    peak_idx_convertor: Dict[int, int],
    unique_row_idxs: np.ndarray,
    rt_mz: np.ndarray,
    window_idxs: np.ndarray,
    library: Dict
) -> UnifiedFeatures:
    """
    Calculate all features for all candidates in a single pass.
    
    This replaces calling get_features twice (once for targets, once for decoys).
    
    Args:
        unified_candidates: Candidates with target/decoy tracking
        matrix_data: Matrix data from create_entries
        additional_outputs: Additional data from create_entries
        dia_spectrum: DIA spectrum
        prec_rt: Precursor retention time
        lib_coefficients: NNLS coefficients
        sparse_matrix: Sparse matrix from NNLS
        peak_idx_convertor: Peak index mapping
        unique_row_idxs: Unique row indices
        rt_mz: RT and m/z array for all library entries
        window_idxs: Window indices for candidates
        library: Spectral library
        
    Returns:
        UnifiedFeatures object with all calculated features
    """
    # Extract needed data
    peaks_in_dia = unified_candidates.peaks_in_dia
    is_decoy_matched = unified_candidates.is_decoy[peaks_in_dia]
    n_candidates = len(peaks_in_dia)
    
    if n_candidates == 0:
        # Return empty features
        return UnifiedFeatures(
            features=np.zeros((0, 26)),
            is_decoy=np.array([], dtype=bool)
        )
    
    # Get data for matched candidates
    pep_cand = additional_outputs['pep_cand']
    norm_intensities = additional_outputs['norm_intensities']
    lib_peaks_matched = additional_outputs['lib_peaks_matched']
    pep_cand_list = additional_outputs['pep_cand_list']
    ms1_error = additional_outputs['ms1_error_matched']
    
    # Initialize feature arrays
    features = np.zeros((n_candidates, 26))
    
    # Get split arrays from matrix data
    spec_values_split = matrix_data.values_split
    spec_row_indices_split = matrix_data.row_indices_split
    spec_col_indices_split = matrix_data.col_indices_split
    
    # Pre-compute commonly used values
    dia_total_intensity = np.sum(dia_spectrum[:, 1])
    lib_coeffs_array = np.asarray(lib_coefficients)
    
    # First, calculate residuals and y_pred for all candidates
    # This is needed for manhattan distance and residual features
    residuals = None
    y_pred = None
    
    if n_candidates > 0:
        # Calculate residuals and predictions
        residuals, y_pred = compute_residuals(
            spec_row_indices_split,
            spec_col_indices_split,
            spec_values_split,
            is_decoy_matched,
            dia_spectrum[:, 1],
            lib_coefficients
        )
    
    # Calculate features for each candidate
    for i in range(n_candidates):
        candidate_idx = peaks_in_dia[i]
        
        # Feature 1: Number of library peaks matched
        features[i, 0] = np.sum(lib_peaks_matched[i])
        
        # Feature 2: Fraction of library intensity matched
        features[i, 1] = np.sum(spec_values_split[i])
        
        # Feature 3: Fraction of DIA intensity matched
        if len(spec_row_indices_split[i]) > 0:
            features[i, 2] = np.sum(dia_spectrum[spec_row_indices_split[i], 1]) / dia_total_intensity
        
        # Feature 4: MS1 relative error
        features[i, 3] = ms1_error[i]
        
        # Feature 5: RT error
        # Use same calculation for all candidates (unified approach)
        if candidate_idx < len(window_idxs):
            candidate_rt = rt_mz[window_idxs[candidate_idx], 0]
            features[i, 4] = prec_rt - candidate_rt
        else:
            # Handle case where index is out of bounds
            features[i, 4] = 0  # Default value
        
        # Feature 6: Fraction intensity matched
        if len(spec_values_split[i]) > 0:
            features[i, 5] = np.sum(spec_values_split[i] * lib_coefficients[i])
        
        # Feature 7: Fraction intensity predicted
        features[i, 6] = features[i, 5] * lib_coefficients[i] if i < len(lib_coefficients) else 0
        
        # Features 8-10: Correlation features (placeholder)
        features[i, 7:10] = 0  # r2all, r2_lib_spec, r2_unique
        
        # Feature 11: Fraction unique predicted
        # Requires single_matched_rows calculation
        features[i, 10] = 0  # Placeholder
        
        # Feature 12: Fraction DIA intensity predicted
        features[i, 11] = features[i, 1] * lib_coefficients[i] / features[i, 2] if features[i, 2] > 0 else 0
        
        # Feature 13-16: Hyperscore features
        # Count b and y ions if fragment names available
        if 'frag_names' in additional_outputs and i < len(additional_outputs['frag_names']):
            frag_names = additional_outputs['frag_names'][i]
            b_count = sum(1 for f in frag_names if f.startswith('b'))
            y_count = sum(1 for f in frag_names if f.startswith('y'))
            features[i, 13] = b_count  # b_counts
            features[i, 14] = y_count  # y_counts
            # Hyperscore calculation would go here
            features[i, 12] = 0  # hyperscores placeholder
            features[i, 15] = 0  # longest_y_ions placeholder
        
        # Feature 17: SCRIBE score
        if len(spec_row_indices_split[i]) > 0 and len(spec_values_split[i]) > 0:
            try:
                features[i, 16] = get_scribe(
                    spec_values_split[i],
                    dia_spectrum[:, 1],  # Full spectrum intensities
                    spec_row_indices_split[i]
                )
            except:
                features[i, 16] = 0
        
        # Features 18-19: Residuals
        if residuals is not None and len(spec_row_indices_split[i]) > 0:
            # Get residuals for this candidate's peaks
            candidate_residuals = residuals[spec_row_indices_split[i]]
            if len(candidate_residuals) > 0:
                features[i, 17] = np.max(np.abs(candidate_residuals))  # max_unmatched_residuals
                features[i, 18] = np.max(candidate_residuals)  # max_matched_residuals
        
        # Feature 20: Goodness of fit
        # Skip for now as gof_stat requires different structure
        features[i, 19] = 0
    
    # Calculate manhattan distance and spectral contrast for all candidates at once
    if residuals is not None and y_pred is not None and n_candidates > 0:
        # Calculate manhattan distance and spectral contrast
        manhattan_distances, fitted_spectral_contrasts = compute_manhattan_distance(
            spec_row_indices_split,
            spec_col_indices_split,
            spec_values_split,
            dia_spectrum[:, 1],
            y_pred
        )
        
        # Results are already in the correct order
        features[:, 20] = manhattan_distances
        features[:, 21] = fitted_spectral_contrasts
    
    # Continue with remaining features
    for i in range(n_candidates):
        candidate_idx = peaks_in_dia[i]
        
        # Features 23-24: More intensity features
        features[i, 22] = features[i, 5]  # frac_int_matched_pred
        features[i, 23] = features[i, 5] if lib_coefficients[i] > 0.1 else 0  # with significance cutoff
        
        # Feature 25: Large coefficient cosine similarity
        features[i, 24] = 0  # Placeholder
        
        # Feature 26: m/z value
        # Use same logic for all candidates (unified approach)
        if candidate_idx < len(window_idxs):
            features[i, 25] = rt_mz[window_idxs[candidate_idx], 1]
        else:
            # Handle case where index is out of bounds
            features[i, 25] = 0  # Default value
    
    # Define feature names
    feature_names = [
        "num_lib_peaks_matched", "frac_lib_intensity", "frac_dia_intensity",
        "rel_error", "rt_error", "frac_int_matched", "frac_int_pred",
        "r2all", "r2_lib_spec", "r2_unique", "frac_unique_pred",
        "frac_dia_intensity_pred", "hyperscores", "b_counts", "y_counts",
        "longest_y_ions", "scribe_scores", "max_unmatched_residuals",
        "max_matched_residuals", "gof_stats", "manhattan_distances",
        "fitted_spectral_contrasts", "frac_int_matched_pred",
        "frac_int_matched_pred_sigcoeff", "large_coeff_cosine", "rt_mz"
    ]
    
    return UnifiedFeatures(
        features=features,
        is_decoy=is_decoy_matched,
        feature_names=feature_names
    )


def calculate_features(
    unified_candidates: UnifiedCandidates,
    matrix_data: UnifiedMatrixData,
    additional_outputs: Dict,
    dia_spectrum: np.ndarray,
    prec_rt: float,
    lib_coefficients: np.ndarray,
    sparse_matrix,
    peak_idx_convertor: Dict[int, int],
    unique_row_idxs: np.ndarray,
    rt_mz: np.ndarray,
    window_idxs: np.ndarray,
    library: Dict
) -> UnifiedFeatures:
    """
    Calculate all features using the new modular approach.
    
    This is a refactored version that uses the modular feature calculator
    for better maintainability and testability.
    """
    # Extract needed data
    peaks_in_dia = unified_candidates.peaks_in_dia
    is_decoy_matched = unified_candidates.is_decoy[peaks_in_dia]
    n_candidates = len(peaks_in_dia)
    
    if n_candidates == 0:
        return UnifiedFeatures(
            features=np.zeros((0, 26)),
            is_decoy=np.array([], dtype=bool),
            feature_names=FeatureCalculator().feature_names
        )
    
    # Calculate residuals and predictions using efficient CSC matrix operations
    sparse_matrix_csc = sparse_matrix.tocsc()
    residuals, y_pred = get_residuals_csc(
        sparse_matrix_csc,
        dia_spectrum[:, 1],
        lib_coefficients
    )
    
    # Create inputs for feature calculator
    sparse_matrix_csc = sparse_matrix.tocsc()
    calc_inputs = FeatureCalculatorInputs(
        candidates=[additional_outputs['pep_cand'][i] for i in range(len(peaks_in_dia))],
        peaks_in_dia=peaks_in_dia,
        is_decoy_matched=is_decoy_matched,
        spec_values_split=matrix_data.values_split,
        spec_row_indices_split=matrix_data.row_indices_split,
        spec_col_indices_split=matrix_data.col_indices_split,
        lib_peaks_matched=additional_outputs['lib_peaks_matched'],
        ms1_error_array=unified_candidates.ms1_error,
        frag_names=additional_outputs.get('frag_names'),
        dia_spectrum=dia_spectrum,
        prec_rt=prec_rt,
        lib_coefficients=lib_coefficients,
        rt_mz=rt_mz,
        window_idxs=window_idxs,
        library=library,
        sparse_matrix_csc=sparse_matrix_csc,
        residuals=residuals,
        y_pred=y_pred
    )
    
    # Calculate features using the modular calculator
    calculator = FeatureCalculator()
    features = calculator.calculate_all_features(calc_inputs)
    
    return UnifiedFeatures(
        features=features,
        is_decoy=is_decoy_matched,
        feature_names=calculator.feature_names
    )


# ===== MATRIX CONSTRUCTION FUNCTIONS =====

def unmatched_peaks(
    unified_candidates: UnifiedCandidates,
    norm_intensities: List[np.ndarray],
    pep_cand_loc: List[np.ndarray],
    last_row: int,
    fit_type: str = "a",
    lower_limit: float = 1e-10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate unmatched peaks for all candidates.
    
    Calculate unmatched peaks for both targets and decoys in a single pass.
    
    Args:
        unified_candidates: Candidates with target/decoy tracking
        norm_intensities: Normalized intensities for matched candidates
        pep_cand_loc: Peak locations for matched candidates
        last_row: Last row index in the matrix
        fit_type: How to fit unmatched peaks ('a', 'b', or 'c')
        lower_limit: Minimum intensity threshold for type 'c'
        
    Returns:
        Tuple of (row_indices, col_indices, values, is_decoy) for unmatched peaks
    """
    assert fit_type in ["a", "b", "c"]
    
    n_candidates = len(pep_cand_loc)
    
    if fit_type == "a":
        # All unmatched peaks go to a single zero-intensity row
        not_dia_col_indices = np.arange(n_candidates)
        not_dia_row_indices = np.array([last_row] * n_candidates, dtype=int)
        # Vectorized calculation of unmatched peak sums
        not_dia_values = np.zeros(n_candidates)
        for idx in range(n_candidates):
            mask = pep_cand_loc[idx] % 2 == 0
            not_dia_values[idx] = np.sum(norm_intensities[idx][mask])
        
    elif fit_type == "b":
        # Each candidate gets its own zero-intensity row
        not_dia_col_indices = np.arange(n_candidates)
        not_dia_row_indices = np.array([last_row + 1 + idx for idx in range(n_candidates)], dtype=int)
        not_dia_values = np.array([
            np.sum([norm_intensities[idx][peak_idx] 
                    for peak_idx in range(len(norm_intensities[idx])) 
                    if pep_cand_loc[idx][peak_idx] % 2 == 0])
            for idx in range(n_candidates)
        ])
        
    elif fit_type == "c":
        # Each unmatched peak gets its own row - vectorized approach
        all_unmatched_peaks = []
        for idx in range(n_candidates):
            mask = (pep_cand_loc[idx] % 2 == 0) & (norm_intensities[idx] > lower_limit)
            all_unmatched_peaks.append(norm_intensities[idx][mask])
        num_unmatched_to_fit = [len(i) for i in all_unmatched_peaks]
        not_dia_col_indices = np.concatenate([[idx] * n for idx, n in enumerate(num_unmatched_to_fit)])
        not_dia_row_indices = np.arange(np.sum(num_unmatched_to_fit)) + last_row + 1
        not_dia_values = np.concatenate(all_unmatched_peaks)
    
    # Track which entries are decoys based on matched candidates
    is_decoy_matched = unified_candidates.is_decoy[unified_candidates.peaks_in_dia]
    
    # Create is_decoy array for unmatched peaks
    if fit_type in ["a", "b"]:
        not_dia_is_decoy = is_decoy_matched
    else:  # type "c"
        not_dia_is_decoy = np.concatenate([
            [is_decoy_matched[idx]] * n 
            for idx, n in enumerate(num_unmatched_to_fit)
        ])
    
    return not_dia_row_indices, not_dia_col_indices, not_dia_values, not_dia_is_decoy


def build_sparse_matrix(
    matrix_data: UnifiedMatrixData,
    unmatched_row_indices: np.ndarray,
    unmatched_col_indices: np.ndarray,
    unmatched_values: np.ndarray,
    dia_spectrum: np.ndarray,
    unique_row_idxs: np.ndarray
) -> Tuple[sparse.coo_matrix, np.ndarray, Dict[int, int]]:
    """
    Build sparse matrix for NNLS optimization.
    
    Args:
        matrix_data: Unified matrix data with matched peaks
        unmatched_row_indices: Row indices for unmatched peaks
        unmatched_col_indices: Column indices for unmatched peaks
        unmatched_values: Values for unmatched peaks
        dia_spectrum: DIA spectrum
        unique_row_idxs: Unique row indices from matched peaks
        
    Returns:
        Tuple of (sparse_matrix, target_vector, peak_idx_convertor)
    """
    # Combine matched and unmatched peaks (ensure integer indices)
    all_row_indices = np.concatenate([matrix_data.row_indices, unmatched_row_indices]).astype(int)
    all_col_indices = np.concatenate([matrix_data.col_indices, unmatched_col_indices]).astype(int)
    all_values = np.concatenate([matrix_data.values, unmatched_values])
    
    # Rank rows to remove gaps
    new_row_indices = stats.rankdata(all_row_indices, method="dense").astype(int) - 1
    peak_idx_convertor = {old: new for old, new in zip(all_row_indices, new_row_indices)}
    
    # Create sparse matrix
    sparse_lib_matrix = sparse.coo_matrix(
        (all_values, (new_row_indices, all_col_indices))
    )
    
    # Create target vector
    dia_spec_int = dia_spectrum[unique_row_idxs, 1]
    # Pad with zeros for unmatched peak rows
    n_extra_rows = sparse_lib_matrix.shape[0] - len(dia_spec_int)
    dia_spec_int = np.append(dia_spec_int, [0] * n_extra_rows)
    
    return sparse_lib_matrix, dia_spec_int, peak_idx_convertor


def process_matrix(
    unified_candidates: UnifiedCandidates,
    matrix_data: UnifiedMatrixData,
    additional_outputs: Dict,
    dia_spectrum: np.ndarray,
    unmatched_fit_type: str = "a"
) -> Dict[str, any]:
    """
    Complete matrix processing pipeline.
    
    This replaces the entire matrix construction section of fit_to_lib2.
    
    Args:
        unified_candidates: Unified candidates
        matrix_data: Initial matrix data from create_entries_unified
        additional_outputs: Additional data from create_entries
        dia_spectrum: DIA spectrum
        unmatched_fit_type: How to handle unmatched peaks
        
    Returns:
        Dictionary with:
        - sparse_matrix: Sparse matrix for NNLS
        - target_vector: Target intensity vector
        - peak_idx_convertor: Mapping of peak indices
        - lib_coefficients: NNLS solution
        - unique_row_idxs: Unique row indices
    """
    # Early exit if no matches
    if len(matrix_data.row_indices) == 0:
        return {
            'sparse_matrix': sparse.coo_matrix((0, 0)),
            'target_vector': np.array([]),
            'peak_idx_convertor': {},
            'lib_coefficients': np.array([]),
            'unique_row_idxs': np.array([])
        }
    
    # Get unique row indices
    unique_row_idxs = np.unique(matrix_data.row_indices)
    unique_row_idxs = np.sort(unique_row_idxs).astype(int)
    
    # Calculate unmatched peaks for all candidates
    last_row = max(unique_row_idxs)
    unmatched_row_idx, unmatched_col_idx, unmatched_vals, _ = unmatched_peaks(
        unified_candidates=unified_candidates,
        norm_intensities=additional_outputs['norm_intensities'],
        pep_cand_loc=additional_outputs['pep_cand_loc'],
        last_row=last_row,
        fit_type=unmatched_fit_type
    )
    
    # Build sparse matrix
    sparse_matrix, target_vector, peak_idx_convertor = build_sparse_matrix(
        matrix_data=matrix_data,
        unmatched_row_indices=unmatched_row_idx,
        unmatched_col_indices=unmatched_col_idx,
        unmatched_values=unmatched_vals,
        dia_spectrum=dia_spectrum,
        unique_row_idxs=unique_row_idxs
    )
    
    # Solve NNLS
    try:
        fit_results = sparse_nnls.lsqnonneg(
            sparse_matrix, 
            target_vector, 
            {"show_progress": False}
        )
        lib_coefficients = fit_results['x']
    except Exception as e:
        # Fallback to scipy if ptinnls fails
        from scipy.optimize import nnls
        lib_coefficients, _ = nnls(sparse_matrix.toarray(), target_vector)
    
    return {
        'sparse_matrix': sparse_matrix,
        'target_vector': target_vector,
        'peak_idx_convertor': peak_idx_convertor,
        'lib_coefficients': lib_coefficients,
        'unique_row_idxs': unique_row_idxs
    }

# ===== UTILITY FUNCTIONS =====

def filter_candidates_by_window(
    rt_mz: np.ndarray,
    all_keys: List,
    prec_mz: float,
    prec_rt: float,
    windowWidth: float,
    ms1_mz: Optional[float] = None,
    rt_filter: bool = False,
    ms1_tol: Optional[float] = None,
    rt_tol: Optional[float] = None,
    dino_features: Optional[Any] = None
) -> Tuple[np.ndarray, List]:
    """
    Filter spectral library candidates based on mass and retention time windows.
    
    This function filters library entries to find candidates that match the precursor
    within specified mass and optionally retention time tolerances. It also supports
    filtering based on MS1 features (dino features) if provided.
    
    Args:
        rt_mz: 2D array with RT in column 0 and m/z in column 1 for all library entries
        all_keys: List of all library keys corresponding to rt_mz rows
        prec_mz: Precursor m/z value
        prec_rt: Precursor retention time
        windowWidth: Mass window width for filtering
        ms1_mz: Optional MS1 m/z for more precise filtering
        rt_filter: Whether to apply retention time filtering
        ms1_tol: MS1 mass tolerance (required if ms1_mz is provided)
        rt_tol: Retention time tolerance (required if rt_filter is True)
        dino_features: Optional MS1 features for additional filtering
        
    Returns:
        Tuple of:
        - window_idxs: Array of indices into all_keys that pass filtering
        - mass_window_candidates: List of library keys that pass filtering
        
    Example:
        >>> rt_mz = np.array([[10.0, 500.0], [10.1, 500.1], [20.0, 600.0]])
        >>> all_keys = ['pep1', 'pep2', 'pep3']
        >>> idxs, candidates = filter_candidates_by_window(
        ...     rt_mz, all_keys, 500.05, 10.0, 1.0
        ... )
    """
    # Apply mass window filtering
    if ms1_mz:
        # Use MS1 m/z for filtering if provided
        if ms1_tol is None:
            raise ValueError("ms1_tol must be provided when ms1_mz is specified")
        _bool = (np.abs(rt_mz[:, 1] - ms1_mz) / ms1_mz) < ms1_tol
    else:
        # Standard mass window filtering
        if rt_filter:
            if rt_tol is None:
                raise ValueError("rt_tol must be provided when rt_filter is True")
            _bool = np.logical_and(
                np.abs(rt_mz[:, 1] - prec_mz) < (windowWidth / 2),
                np.abs(rt_mz[:, 0] - prec_rt) < rt_tol
            )
        else:
            _bool = np.abs(rt_mz[:, 1] - prec_mz) < (windowWidth / 2)
    
    window_idxs = np.where(_bool)[0]
    
    # Apply dino feature filtering if provided
    if dino_features is not None:
        if rt_tol is None:
            raise ValueError("rt_tol must be provided when dino_features are specified")
        if ms1_tol is None:
            raise ValueError("ms1_tol must be provided when dino_features are specified")
            
        # Filter dino features by RT and m/z
        filtered_dino = feature_list_mz(
            feature_list_rt(dino_features, prec_rt, rt_tol=rt_tol),
            prec_mz, 
            windowWidth
        )
        
        # Create tolerance windows and filter candidates
        window_edges = createTolWindows(filtered_dino.mz, tolerance=ms1_tol)
        window_idxs = window_idxs[
            np.where((np.searchsorted(window_edges, rt_mz[window_idxs, 1]) % 2) == 1)[0]
        ]
    
    # Get candidate keys
    mass_window_candidates = [all_keys[i] for i in window_idxs]
    
    return window_idxs, mass_window_candidates


def separate_library_candidates(
    mass_window_candidates: List,
    library: Dict,
    include_decoys: bool = True
) -> UnifiedCandidates:
    """
    Separate library candidates into targets and decoys and create a unified structure.
    
    This function takes filtered library candidates and separates them based on their
    decoy status, then creates a UnifiedCandidates structure that can be processed
    together while maintaining the distinction between targets and decoys.
    
    Args:
        mass_window_candidates: List of library keys that passed window filtering
        library: Spectral library dictionary
        include_decoys: Whether to include decoys in the output (False for RT alignment)
        
    Returns:
        UnifiedCandidates object containing all candidates with their peaks and metadata
        
    Example:
        >>> candidates = ['PEPTIDE/2', 'Decoy_PEPTIDE/2', 'PEPTIDEK/3']
        >>> library = {
        ...     'PEPTIDE/2': {'spectrum': np.array([[500.1, 100]]), 'is_decoy': False},
        ...     'Decoy_PEPTIDE/2': {'spectrum': np.array([[500.1, 100]]), 'is_decoy': True},
        ...     'PEPTIDEK/3': {'spectrum': np.array([[600.1, 200]]), 'is_decoy': False}
        ... }
        >>> unified = separate_library_candidates(candidates, library)
        >>> print(f"Targets: {unified.n_targets}, Decoys: {unified.n_decoys}")
        Targets: 2, Decoys: 1
    """
    # Separate targets and decoys
    target_candidates = [k for k in mass_window_candidates if not library[k].get("is_decoy", False)]
    target_peaks = [library[k]["spectrum"] for k in target_candidates]
    
    if include_decoys:
        decoy_candidates = [k for k in mass_window_candidates if library[k].get("is_decoy", False)]
        decoy_peaks = [library[k]["spectrum"] for k in decoy_candidates]
        
        if len(decoy_candidates) > 0:
            # Create unified structure with both targets and decoys
            unified = create_unified_candidates(
                target_candidates=target_candidates,
                target_peaks=target_peaks,
                decoy_candidates=decoy_candidates,
                decoy_peaks=decoy_peaks
            )
        else:
            # No decoys found, just use targets
            unified = create_unified_candidates(
                target_candidates=target_candidates,
                target_peaks=target_peaks
            )
    else:
        # RT alignment mode - only use targets
        unified = create_unified_candidates(
            target_candidates=target_candidates,
            target_peaks=target_peaks
        )
    
    return unified


def create_empty_output_row(
    spec_idx: int,
    ms1_spec_id: int,
    prec_mz: float,
    prec_rt: float,
    num_columns: int
) -> List:
    """
    Create an empty output row for cases where no matches are found.
    
    This function creates a standardized empty row with zeros for all feature
    and fragment columns, used when no library candidates match the spectrum.
    
    Args:
        spec_idx: Spectrum index/scan number
        ms1_spec_id: MS1 spectrum ID (0 if not available)
        prec_mz: Precursor m/z value
        prec_rt: Precursor retention time
        num_columns: Total number of columns in output (len(names))
        
    Returns:
        List representing one output row with appropriate default values
        
    Example:
        >>> row = create_empty_output_row(100, 99, 500.5, 10.5, 49)
        >>> print(f"Coeff: {row[0]}, Spec: {row[1]}, m/z: {row[5]}")
        Coeff: 0, Spec: 100, m/z: 500.5
    """
    # First 7 columns have specific values
    # [coeff, spec_id, Ms1_spec_id, seq, z, window_mz, rt]
    row = [0, spec_idx, ms1_spec_id, 0, 0, prec_mz, prec_rt]
    
    # Remaining columns are zeros
    row.extend(np.zeros(num_columns - 7))
    
    return row


def extract_non_zero_coefficients(
    lib_coefficients: np.ndarray
) -> Tuple[List[float], List[int]]:
    """
    Extract non-zero coefficients and their indices from NNLS results.
    
    Args:
        lib_coefficients: Array of coefficients from NNLS optimization
        
    Returns:
        Tuple of:
        - List of non-zero coefficient values
        - List of indices where coefficients are non-zero
        
    Example:
        >>> coeffs = np.array([0.0, 0.5, 0.0, 0.3, 0.0])
        >>> values, indices = extract_non_zero_coefficients(coeffs)
        >>> print(f"Values: {values}, Indices: {indices}")
        Values: [0.5, 0.3], Indices: [1, 3]
    """
    non_zero_coeffs = []
    non_zero_coeffs_idxs = []
    
    for i, c in enumerate(lib_coefficients):
        if c != 0:
            non_zero_coeffs.append(c)
            non_zero_coeffs_idxs.append(i)
    
    return non_zero_coeffs, non_zero_coeffs_idxs


def format_fragment_information(
    additional_outputs: Dict,
    candidate_idx: int
) -> List[str]:
    """
    Format fragment information into semicolon-delimited strings.
    
    Args:
        additional_outputs: Dictionary containing fragment data arrays
        candidate_idx: Index of the candidate to format
        
    Returns:
        List of 7 semicolon-delimited strings for fragment information:
        [frag_names, frag_errors, lib_frag_mz, lib_frag_int, obs_frag_int,
         unique_frags, unique_frags_int]
         
    Example:
        >>> outputs = {
        ...     'frag_names': [np.array(['b2', 'y3'])],
        ...     'frag_errors': [np.array([0.001, 0.002])]
        ... }
        >>> frags = format_fragment_information(outputs, 0)
        >>> print(frags[0])  # Fragment names
        b2;y3
    """
    # Get fragment data arrays
    frag_names_list = additional_outputs.get('frag_names', [])
    frag_errors_list = additional_outputs.get('frag_errors', [])
    lib_frag_mz_list = additional_outputs.get('lib_frag_mz', [])
    lib_frag_int_list = additional_outputs.get('lib_frag_int', [])
    obs_frag_int_list = additional_outputs.get('obs_frag_int', [])
    
    # Get data for this candidate if available (check each list independently)
    frag_names = frag_names_list[candidate_idx] if candidate_idx < len(frag_names_list) else np.array([])
    frag_errors = frag_errors_list[candidate_idx] if candidate_idx < len(frag_errors_list) else np.array([])
    lib_frag_mz = lib_frag_mz_list[candidate_idx] if candidate_idx < len(lib_frag_mz_list) else np.array([])
    lib_frag_int = lib_frag_int_list[candidate_idx] if candidate_idx < len(lib_frag_int_list) else np.array([])
    obs_frag_int = obs_frag_int_list[candidate_idx] if candidate_idx < len(obs_frag_int_list) else np.array([])
    
    # TODO: Calculate unique fragments after matrix construction
    # For now, use empty arrays for unique fragments
    unique_frags = np.array([])
    unique_frags_int = np.array([])
    
    # Format as semicolon-delimited strings
    ms2_frags = [
        ";".join(map(str, frag_names)) if len(frag_names) > 0 else "",
        ";".join(map(str, frag_errors)) if len(frag_errors) > 0 else "",
        ";".join(map(str, lib_frag_mz)) if len(lib_frag_mz) > 0 else "",
        ";".join(map(str, lib_frag_int)) if len(lib_frag_int) > 0 else "",
        ";".join(map(str, obs_frag_int)) if len(obs_frag_int) > 0 else "",
        ";".join(map(str, unique_frags)) if len(unique_frags) > 0 else "",
        ";".join(map(str, unique_frags_int)) if len(unique_frags_int) > 0 else ""
    ]
    
    return ms2_frags


def get_protein_info(
    candidate: Tuple,
    library: Dict,
    protein_column: Optional[str] = None
) -> str:
    """
    Extract protein information for a candidate from the library.
    
    Args:
        candidate: Tuple of (sequence, charge) representing the candidate
        library: Spectral library dictionary
        protein_column: Name of the protein column in library (None to skip)
        
    Returns:
        Protein identifier string or "NA" if not found
        
    Example:
        >>> lib = {('PEPTIDE', 2): {'protein': 'PROT1'}}
        >>> protein = get_protein_info(('PEPTIDE', 2), lib, 'protein')
        >>> print(protein)
        PROT1
    """
    if not protein_column or not library:
        return "NA"
    
    try:
        # Remove decoy prefix if present
        clean_seq = candidate[0].replace("Decoy_", "")
        clean_key = (clean_seq, candidate[1])
        
        # Look up protein info
        return library.get(clean_key, {}).get(protein_column, "NA")
    except:
        return "NA"


def format_spectral_fitting_output(
    lib_coefficients: np.ndarray,
    unified_candidates: UnifiedCandidates,
    unified_features: UnifiedFeatures,
    additional_outputs: Dict,
    spec_idx: int,
    ms1_spec: Optional[Any],
    prec_mz: float,
    prec_rt: float,
    library: Dict,
    config: Any
) -> List[List]:
    """
    Format spectral fitting results into output rows.
    
    This function takes the results from spectral fitting and formats them into
    the standardized output format expected by downstream processing.
    
    Args:
        lib_coefficients: NNLS coefficients
        unified_candidates: Candidates that were processed
        unified_features: Calculated features for candidates
        additional_outputs: Fragment and other additional data
        spec_idx: Spectrum index
        ms1_spec: MS1 spectrum object (optional)
        prec_mz: Precursor m/z
        prec_rt: Precursor retention time
        library: Spectral library
        config: Configuration object
        
    Returns:
        List of output rows, each row is a list of values
    """
    # Extract non-zero coefficients
    non_zero_coeffs, non_zero_coeffs_idxs = extract_non_zero_coefficients(lib_coefficients)
    
    # Default output if no matches
    if len(non_zero_coeffs) == 0:
        return [create_empty_output_row(
            spec_idx, 
            ms1_spec.scan_num if ms1_spec else 0,
            prec_mz,
            prec_rt,
            len(names)
        )]
    
    # Get matched candidates
    matched_candidates = [unified_candidates.candidates[i] for i in unified_candidates.peaks_in_dia]
    
    # Build output rows
    output = []
    for i, j in zip(range(len(non_zero_coeffs)), non_zero_coeffs_idxs):
        if j < len(matched_candidates):
            candidate = matched_candidates[j]
            features = unified_features.features[j]
            
            # Format fragment information
            ms2_frags = format_fragment_information(additional_outputs, j)
            
            # Get protein info
            protein = get_protein_info(
                candidate, 
                library, 
                config.protein_column if hasattr(config, 'protein_column') else None
            )
            
            # Get file name
            file_name = config.args.mzml if hasattr(config, 'args') and hasattr(config.args, 'mzml') else ""
            
            # Build complete row
            row = [
                non_zero_coeffs[i],                          # coeff
                spec_idx,                                    # spec_id
                ms1_spec.scan_num if ms1_spec else 0,        # Ms1_spec_id
                candidate[0],                                # seq
                candidate[1],                                # z
                prec_mz,                                     # window_mz
                prec_rt,                                     # rt
                *features,                                   # 26 feature values
                *ms2_frags,                                  # 7 fragment strings
                file_name,                                   # file_name
                protein                                      # protein
            ]
            
            output.append(row)
    
    return output


def check_ms1_peaks(
    rt_mz: np.ndarray,
    window_idxs: np.ndarray,
    ms1_spec: Any
) -> np.ndarray:
    """
    Check which candidates have matching MS1 peaks.
    
    This function determines which library candidates have a corresponding peak
    in the MS1 spectrum, which is used for filtering during spectral matching.
    
    Args:
        rt_mz: 2D array with RT in column 0 and m/z in column 1 for all library entries
        window_idxs: Array of indices into rt_mz for candidates that passed window filtering
        ms1_spec: MS1 spectrum object with .mz attribute containing m/z values
        
    Returns:
        Boolean array indicating which candidates have MS1 peaks (True) or not (False)
        
    Example:
        >>> rt_mz = np.array([[10.0, 500.0], [10.1, 500.1], [10.2, 600.0]])
        >>> window_idxs = np.array([0, 2])
        >>> ms1_spec = Mock(mz=np.array([499.9, 600.1]))
        >>> ms1_peak = check_ms1_peaks(rt_mz, window_idxs, ms1_spec)
        >>> print(ms1_peak)  # [True, True] - both candidates have MS1 peaks
    """
    # Extract m/z values for the filtered candidates
    candidate_mz_values = rt_mz[window_idxs, 1]
    
    # Handle edge case of empty MS1 spectrum
    if len(ms1_spec.mz) == 0:
        # No MS1 peaks means no candidates can match
        return np.zeros(len(window_idxs), dtype=bool)
    
    # Check for closest peak difference for each candidate
    ms1_peak = ~np.isnan([closest_peak_diff(mz, ms1_spec.mz) for mz in candidate_mz_values])
    
    return ms1_peak


def filter_candidates_by_peak_matching(
    candidate_peaks: List[np.ndarray],
    centroid_breaks: np.ndarray,
    ms1_peak: np.ndarray,
    top_n: int,
    atleast_m: int,
    frac_matched: float
) -> Tuple[List[int], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Filter candidates based on peak matching criteria.
    
    This function applies multiple filtering criteria to identify candidates that have
    sufficient matching peaks in the DIA spectrum. It checks for:
    - Minimum number of top peaks in DIA
    - Sufficient fraction of intensity matched
    - MS1 peak presence
    - Top peak in DIA
    - At least 2 of top 3 peaks in DIA
    
    Args:
        candidate_peaks: List of peak arrays for each candidate [m/z, intensity]
        centroid_breaks: Array of centroid break points from DIA spectrum preprocessing
        ms1_peak: Boolean array indicating which candidates have MS1 peaks
        top_n: Number of top intensity peaks to consider
        atleast_m: Minimum number of matched peaks required
        frac_matched: Minimum fraction of intensity that must be matched
        
    Returns:
        Tuple containing:
        - ref_peaks_in_dia: List of indices for candidates that pass all filters
        - ref_coords: List of coordinate arrays for all candidates
        - top_ten: List of top peak coordinate arrays for all candidates
        - all_norm_intensities: List of normalized intensity arrays
        
    Example:
        >>> peaks = [np.array([[100.0, 50.0], [200.0, 100.0]])]
        >>> breaks = np.array([99.0, 101.0, 199.0, 201.0])
        >>> ms1 = np.array([True])
        >>> indices, coords, top, norm = filter_candidates_by_peak_matching(
        ...     peaks, breaks, ms1, 10, 2, 0.5
        ... )
    """
    # Calculate reference coordinates for all candidates
    ref_coords = [np.searchsorted(centroid_breaks, M[:, 0]) for M in candidate_peaks]
    
    # Get top N peaks for each candidate
    top_ten = [np.searchsorted(centroid_breaks, 
                               M[np.argsort(-M[:, 1])[0:min(top_n, M.shape[0])], 0]) 
               for M in candidate_peaks]
    
    # Normalize intensities
    all_norm_intensities = [M[:, 1] / np.sum(M[:, 1]) for M in candidate_peaks]
    
    # Apply complex filtering criteria
    ref_peaks_in_dia = []
    for i in range(len(candidate_peaks)):
        # Check all conditions:
        # 1. Fraction of intensity matched > frac_matched
        intensity_matched = np.sum(all_norm_intensities[i][(ref_coords[i] % 2) == 1]) > frac_matched
        
        # 2. At least atleast_m peaks in top N are in DIA
        top_peaks_in_dia = np.sum(top_ten[i] % 2) > atleast_m
        
        # 3. Has MS1 peak
        has_ms1 = ms1_peak[i]
        
        # 4. First of top peaks is in DIA
        first_top_in_dia = len(top_ten[i]) > 0 and top_ten[i][0] % 2 == 1
        
        # 5. At least 2 of top 3 peaks are in DIA
        top_3_peaks = top_ten[i][:min(3, len(top_ten[i]))]
        at_least_2_of_top_3 = np.sum(top_3_peaks % 2 == 1) >= 2
        
        if intensity_matched and top_peaks_in_dia and has_ms1 and first_top_in_dia and at_least_2_of_top_3:
            ref_peaks_in_dia.append(i)
    
    return ref_peaks_in_dia, ref_coords, top_ten, all_norm_intensities


def build_sparse_matrix_simple(
    ref_pep_cand_loc: List[np.ndarray],
    norm_intensities: List[np.ndarray],
    ref_pep_cand: List,
    unique_row_idxs: List[int],
    dia_spec_int: np.ndarray
) -> Tuple[sparse.coo_matrix, np.ndarray, Dict[str, List]]:
    """
    Build sparse matrix for NNLS - simplified version for RT alignment.
    
    This function constructs a sparse matrix representation of the spectral matching
    problem, including handling of unmatched peaks. It's optimized for the simpler
    requirements of RT alignment compared to full spectral fitting.
    
    Args:
        ref_pep_cand_loc: List of coordinate arrays for filtered candidates
        norm_intensities: List of normalized intensity arrays
        ref_pep_cand: List of candidate identifiers
        unique_row_idxs: Sorted list of unique row indices
        dia_spec_int: DIA spectrum intensities for matched peaks
        
    Returns:
        Tuple containing:
        - sparse_lib_matrix: Sparse matrix for NNLS optimization
        - dia_spec_int_padded: DIA intensities with padding for unmatched peaks
        - split_data: Dictionary containing split arrays for later use
        
    Example:
        >>> loc = [np.array([1, 3, 5])]  # Odd indices are "in DIA"
        >>> norm = [np.array([0.2, 0.5, 0.3])]
        >>> cand = [('PEPTIDE', 2)]
        >>> unique = [0, 1, 2]
        >>> dia_int = np.array([100.0, 200.0, 150.0])
        >>> matrix, dia_padded, splits = build_sparse_matrix_simple(
        ...     loc, norm, cand, unique, dia_int
        ... )
    """
    # Calculate which library peaks match DIA peaks
    lib_peaks_matched = [j % 2 == 1 for j in ref_pep_cand_loc]
    
    # Build split arrays for sparse matrix construction
    ref_spec_row_indices_split = [np.int32(((i[j] + 1) / 2) - 1) for i, j in zip(ref_pep_cand_loc, lib_peaks_matched)]
    num_lib_peaks_matched = np.array([np.sum(i) for i in lib_peaks_matched])
    ref_spec_col_indices_split = [np.array([idx] * i) for idx, i in zip(range(len(ref_pep_cand)), num_lib_peaks_matched)]
    ref_spec_values_split = [ints[i] for ints, i in zip(norm_intensities, lib_peaks_matched)]
    
    # Handle empty case
    if len(ref_pep_cand) == 0:
        # Return empty matrix with 1 row for penalty
        sparse_lib_matrix = sparse.coo_matrix((1, 0))
        dia_spec_int_padded = np.array([0])
        return sparse_lib_matrix, dia_spec_int_padded, {
            'ref_spec_row_indices_split': [],
            'ref_spec_col_indices_split': [],
            'ref_spec_values_split': [],
            'lib_peaks_matched': [],
            'num_lib_peaks_matched': np.array([]),
            'sparse_row_indices': np.array([]),
            'sparse_col_indices': np.array([])
        }
    
    # Concatenate matrix values if we have any matched peaks
    if any(len(split) > 0 for split in ref_spec_row_indices_split):
        ref_spec_row_indices = np.concatenate([s for s in ref_spec_row_indices_split if len(s) > 0])
        ref_spec_col_indices = np.concatenate([s for s in ref_spec_col_indices_split if len(s) > 0])
        ref_spec_values = np.concatenate([s for s in ref_spec_values_split if len(s) > 0])
    else:
        ref_spec_row_indices = np.array([])
        ref_spec_col_indices = np.array([])
        ref_spec_values = np.array([])
    
    # Add padding for unmatched peaks penalty
    dia_spec_int_padded = np.append(dia_spec_int, [0])
    
    # Handle unmatched library peaks
    not_dia_col_indices = np.arange(len(ref_pep_cand))
    if len(unique_row_idxs) > 0:
        num_rows = max(unique_row_idxs)
    else:
        num_rows = -1  # Will become 0 after +1
    not_dia_row_indices = [num_rows + 1] * len(not_dia_col_indices)
    
    # Calculate intensities of unmatched peaks
    not_dia_values = np.array([
        np.sum([norm_intensities[idx][peak_idx] 
                for peak_idx in range(len(norm_intensities[idx])) 
                if ref_pep_cand_loc[idx][peak_idx] % 2 == 0])
        for idx in range(len(norm_intensities))
    ])
    
    # Combine matched and unmatched data
    if len(ref_spec_row_indices) > 0:
        sparse_row_indices = np.append(ref_spec_row_indices, not_dia_row_indices)
        sparse_col_indices = np.append(ref_spec_col_indices, not_dia_col_indices)
        sparse_values = np.append(ref_spec_values, not_dia_values)
    else:
        sparse_row_indices = np.array(not_dia_row_indices)
        sparse_col_indices = not_dia_col_indices
        sparse_values = not_dia_values
    
    # Rank rows to handle missing indices
    if len(sparse_row_indices) > 0:
        sparse_row_indices = stats.rankdata(sparse_row_indices, method="dense").astype(int) - 1
    
    # Generate sparse matrix
    if len(sparse_values) > 0:
        sparse_lib_matrix = sparse.coo_matrix((sparse_values, (sparse_row_indices, sparse_col_indices)))
    else:
        # Create empty matrix with appropriate shape
        sparse_lib_matrix = sparse.coo_matrix((1, len(ref_pep_cand)))
    
    # Return split data for feature calculation
    split_data = {
        'lib_peaks_matched': lib_peaks_matched,
        'ref_spec_row_indices_split': ref_spec_row_indices_split,
        'ref_spec_col_indices_split': ref_spec_col_indices_split,
        'ref_spec_values_split': ref_spec_values_split,
        'num_lib_peaks_matched': num_lib_peaks_matched,
        'sparse_row_indices': sparse_row_indices,
        'sparse_col_indices': sparse_col_indices
    }
    
    return sparse_lib_matrix, dia_spec_int_padded, split_data


def build_sparse_matrix_direct(
    ref_pep_cand_loc: List[np.ndarray],
    norm_intensities: List[np.ndarray],
    ref_pep_cand: List,
    dia_spectrum: np.ndarray
) -> Tuple[sparse.coo_matrix, np.ndarray, List[int], Dict[str, Any]]:
    """
    Build sparse matrix for NNLS directly without split array intermediates.
    
    This function constructs a sparse matrix representation by directly building
    the coordinate arrays instead of creating intermediate split arrays. This 
    eliminates the need for split array generation and concatenation.
    
    Args:
        ref_pep_cand_loc: List of coordinate arrays for filtered candidates
        norm_intensities: List of normalized intensity arrays  
        ref_pep_cand: List of candidate identifiers
        dia_spectrum: Full DIA spectrum array with shape (n_peaks, 2)
        
    Returns:
        Tuple containing:
        - sparse_lib_matrix: Sparse matrix for NNLS optimization
        - dia_spec_int_padded: DIA intensities with padding for unmatched peaks
        - unique_row_idxs: Sorted list of unique row indices
        - fragment_info: Dictionary containing fragment information for later use
        
    Examples:
        >>> loc = [np.array([1, 3, 5])]  # Odd indices are "in DIA"
        >>> norm = [np.array([0.2, 0.5, 0.3])]
        >>> cand = [('PEPTIDE', 2)]
        >>> dia_spec = np.array([[500.0, 100.0], [501.0, 200.0], [502.0, 150.0]])
        >>> matrix, dia_padded, unique_idxs, info = build_sparse_matrix_direct(
        ...     loc, norm, cand, dia_spec
        ... )
        >>> matrix.shape
        (4, 1)  # 3 matched peaks + 1 penalty row, 1 candidate
        >>> len(unique_idxs)
        3
    """
    # Handle empty case
    if len(ref_pep_cand) == 0:
        sparse_lib_matrix = sparse.coo_matrix((1, 0))
        dia_spec_int_padded = np.array([0])
        return sparse_lib_matrix, dia_spec_int_padded, [], {
            'lib_peaks_matched': [],
            'num_lib_peaks_matched': np.array([]),
        }
    
    # Calculate which library peaks match DIA peaks
    lib_peaks_matched = [j % 2 == 1 for j in ref_pep_cand_loc]
    
    # Build coordinate arrays directly without split intermediates
    all_row_indices = []
    all_col_indices = []
    all_values = []
    
    for candidate_idx, (locations, intensities, matched) in enumerate(zip(ref_pep_cand_loc, norm_intensities, lib_peaks_matched)):
        if np.any(matched):
            # Get matched peaks
            matched_locations = locations[matched]
            matched_intensities = intensities[matched]
            
            # Convert to DIA spectrum indices: ((loc + 1) / 2) - 1
            row_indices = np.int32(((matched_locations + 1) / 2) - 1)
            col_indices = np.full(len(matched_locations), candidate_idx, dtype=np.int32)
            
            all_row_indices.extend(row_indices)
            all_col_indices.extend(col_indices) 
            all_values.extend(matched_intensities)
    
    # Convert to numpy arrays
    if len(all_row_indices) > 0:
        all_row_indices = np.array(all_row_indices, dtype=np.int32)
        all_col_indices = np.array(all_col_indices, dtype=np.int32)
        all_values = np.array(all_values)
        
        # Get unique row indices and create mapping
        unique_row_idxs = sorted(set(all_row_indices))
        row_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_row_idxs)}
        
        # Remap row indices to compressed space
        mapped_row_indices = np.array([row_mapping[idx] for idx in all_row_indices])
        
        # Extract DIA intensities for matched peaks
        dia_spec_int = dia_spectrum[unique_row_idxs, 1]
    else:
        # No matched peaks
        mapped_row_indices = np.array([], dtype=np.int32)
        all_col_indices = np.array([], dtype=np.int32) 
        all_values = np.array([])
        unique_row_idxs = []
        dia_spec_int = np.array([])
    
    # Add padding for unmatched peaks penalty
    dia_spec_int_padded = np.append(dia_spec_int, [0])
    
    # Add penalty row for unmatched library peaks
    penalty_row_idx = len(unique_row_idxs)
    penalty_col_indices = np.arange(len(ref_pep_cand), dtype=np.int32)
    
    # Calculate intensities of unmatched peaks (even indices) for each candidate
    penalty_values = np.array([
        np.sum([norm_intensities[idx][peak_idx] 
                for peak_idx in range(len(norm_intensities[idx])) 
                if ref_pep_cand_loc[idx][peak_idx] % 2 == 0])
        for idx in range(len(norm_intensities))
    ])
    
    # Combine matched peaks and penalty row
    final_row_indices = np.concatenate([mapped_row_indices, np.full(len(ref_pep_cand), penalty_row_idx)])
    final_col_indices = np.concatenate([all_col_indices, penalty_col_indices])
    final_values = np.concatenate([all_values, penalty_values])
    
    # Create sparse matrix
    matrix_shape = (len(unique_row_idxs) + 1, len(ref_pep_cand))
    sparse_lib_matrix = sparse.coo_matrix((final_values, (final_row_indices, final_col_indices)), shape=matrix_shape)
    
    # Calculate derived information
    num_lib_peaks_matched = np.array([np.sum(matched) for matched in lib_peaks_matched])
    
    fragment_info = {
        'lib_peaks_matched': lib_peaks_matched,
        'num_lib_peaks_matched': num_lib_peaks_matched,
    }
    
    return sparse_lib_matrix, dia_spec_int_padded, unique_row_idxs, fragment_info


def calculate_frac_lib_intensity_sparse(sparse_lib_matrix_csc: sparse.csc_matrix) -> np.ndarray:
    """
    Calculate fractional library intensity for each candidate using sparse matrix.
    
    This function replaces the split array approach with sparse matrix column sums.
    Each column in the sparse matrix represents one candidate, and we sum each column
    to get the total intensity for that candidate.
    
    Args:
        sparse_lib_matrix_csc: Sparse matrix in CSC format with shape (n_dia_peaks, n_candidates)
                              where each column represents one candidate's matched peaks
    
    Returns:
        np.ndarray: Array of intensity sums, one per candidate
    """
    # Handle empty matrix case
    if sparse_lib_matrix_csc.shape[1] == 0:
        return np.array([], dtype=np.float64)
    
    # Sum each column (candidate) to get total intensity per candidate
    # Convert to array and flatten for consistent numpy array output
    column_sums = np.array(sparse_lib_matrix_csc.sum(axis=0)).flatten()
    return column_sums


def calculate_tic(dia_spectrum: np.ndarray) -> float:
    """
    Calculate total ion current (TIC) from DIA spectrum.
    
    Args:
        dia_spectrum: DIA spectrum array with shape (n_peaks, 2) where column 1 contains intensities
    
    Returns:
        float: Total ion current (sum of all intensities)
    """
    return np.sum(dia_spectrum[:, 1])


def calculate_frac_dia_intensity_sparse(sparse_lib_matrix_csc: sparse.csc_matrix, ref_spec_row_indices_split: List[np.ndarray], dia_spectrum: np.ndarray, tic: float) -> np.ndarray:
    """
    Calculate fractional DIA intensity for each candidate using sparse matrix.
    
    This function replaces the split array approach but still needs the row indices
    to correctly map sparse matrix rows to DIA spectrum rows.
    
    Args:
        sparse_lib_matrix_csc: Sparse matrix in CSC format with shape (n_dia_peaks, n_candidates)
        ref_spec_row_indices_split: Row indices in DIA spectrum per candidate (for mapping)
        dia_spectrum: DIA spectrum array with shape (n_peaks, 2) where column 1 contains intensities
        tic: Total ion current (pre-calculated for efficiency)
    
    Returns:
        np.ndarray: Array of fractional DIA intensities, one per candidate
    """
    # Handle empty matrix case
    if sparse_lib_matrix_csc.shape[1] == 0:
        return np.array([], dtype=np.float64)
    
    # For now, fall back to the original approach using row indices
    # This maintains correctness while we work on the full sparse conversion
    frac_dia_intensities = []
    for row_indices in ref_spec_row_indices_split:
        if len(row_indices) > 0:
            dia_intensity_sum = np.sum(dia_spectrum[row_indices, 1])
            frac_dia_intensities.append(dia_intensity_sum / tic)
        else:
            frac_dia_intensities.append(0.0)
    
    return np.array(frac_dia_intensities)


def calculate_frac_dia_intensity_csc(
    sparse_lib_matrix_csc: sparse.csc_matrix,
    dia_spectrum: np.ndarray,
    tic: float
) -> np.ndarray:
    """
    Calculate fractional DIA intensity for each candidate using CSC sparse matrix operations.
    
    This function computes the fraction of total DIA intensity represented by the 
    matched peaks for each candidate. It uses the sparse matrix structure to efficiently 
    identify which peaks belong to each candidate without requiring split arrays.
    
    Parameters
    ----------
    sparse_lib_matrix_csc : sparse.csc_matrix
        Sparse matrix in CSC format with shape (n_peaks, n_candidates) where
        non-zero values indicate which peaks are matched for each candidate.
    dia_spectrum : np.ndarray
        DIA spectrum array with shape (n_peaks, 2) where column 1 contains intensities.
    tic : float
        Total ion current (pre-calculated for efficiency).
        
    Returns
    -------
    np.ndarray
        Array of fractional DIA intensities, one per candidate. Each value represents
        the fraction of total DIA intensity accounted for by that candidate's matched peaks.
        
    Examples
    --------
    >>> from scipy import sparse
    >>> import numpy as np
    >>> # Create a simple sparse matrix for testing
    >>> row_indices = [0, 1, 2, 3]
    >>> col_indices = [0, 0, 1, 1]
    >>> values = [1.0, 1.0, 1.0, 1.0]  # Values don't matter, just structure
    >>> matrix = sparse.csc_matrix((values, (row_indices, col_indices)), shape=(4, 2))
    >>> dia_spectrum = np.array([[500.0, 100.0], [501.0, 200.0], [502.0, 150.0], [503.0, 250.0]])
    >>> tic = 1000.0  # Total intensity
    >>> frac_intensities = calculate_frac_dia_intensity_csc(matrix, dia_spectrum, tic)
    >>> frac_intensities.shape
    (2,)
    >>> # Candidate 0 has peaks at indices 0,1 -> intensities 100+200=300 -> 300/1000=0.3
    >>> # Candidate 1 has peaks at indices 2,3 -> intensities 150+250=400 -> 400/1000=0.4
    >>> np.isclose(frac_intensities[0], 0.3)
    True
    >>> np.isclose(frac_intensities[1], 0.4)
    True
    
    Notes
    -----
    This function is designed for use in RT alignment where only reference data
    is processed. It replaces the split array approach with direct sparse matrix
    operations for better performance and code consistency.
    
    The function calculates: sum(dia_intensities_at_matched_peaks) / tic for each candidate.
    """
    # Handle empty matrix case
    if sparse_lib_matrix_csc.shape[1] == 0:
        return np.array([], dtype=np.float64)
    
    n_candidates = sparse_lib_matrix_csc.shape[1]
    frac_dia_intensities = np.zeros(n_candidates, dtype=np.float64)
    
    # Process each candidate (column) in the sparse matrix
    for col_idx in range(n_candidates):
        # Extract the column for this candidate
        col_data = sparse_lib_matrix_csc[:, col_idx]
        
        # Get non-zero entries (matched peaks for this candidate)
        nonzero_rows, _ = col_data.nonzero()
        
        if len(nonzero_rows) == 0:
            # No matched peaks, fractional intensity is 0
            frac_dia_intensities[col_idx] = 0.0
            continue
        
        # Sum DIA intensities at matched peak positions
        dia_intensity_sum = np.sum(dia_spectrum[nonzero_rows, 1])
        
        # Calculate fractional intensity (avoid division by zero)
        if tic > 0:
            frac_dia_intensities[col_idx] = dia_intensity_sum / tic
        else:
            frac_dia_intensities[col_idx] = 0.0
    
    return frac_dia_intensities


def calculate_r2_lib_spec_sparse(sparse_lib_matrix_csc: sparse.csc_matrix, dia_spectrum: np.ndarray) -> np.ndarray:
    """
    Calculate per-candidate Pearson correlations between library and DIA intensities using sparse matrix.
    
    This function computes the correlation between each candidate's library intensity values
    and the corresponding DIA spectrum intensity values at the matched peak positions,
    using only the sparse matrix structure. Optimized to avoid numpy array allocations
    by calculating correlations directly in scalar loops.
    
    Args:
        sparse_lib_matrix_csc: Sparse matrix in CSC format with shape (n_dia_peaks, n_candidates)
                              where non-zero values are library intensities at matched peaks
        dia_spectrum: DIA spectrum array with shape (n_peaks, 2) where column 1 contains intensities
    
    Returns:
        np.ndarray: Array of Pearson correlation coefficients, one per candidate
    """
    # Handle empty matrix case
    if sparse_lib_matrix_csc.shape[1] == 0:
        return np.array([], dtype=np.float64)
    
    # Pre-allocate array for better performance
    r2_values = np.zeros(sparse_lib_matrix_csc.shape[1], dtype=np.float64)
    
    # Process each candidate (column) in the sparse matrix
    for col_idx in range(sparse_lib_matrix_csc.shape[1]):
        # Extract the column for this candidate
        col_data = sparse_lib_matrix_csc[:, col_idx]
        
        # Get non-zero entries (matched peaks)
        nonzero_rows, _ = col_data.nonzero()
        
        if len(nonzero_rows) > 1:  # Need at least 2 points for correlation
            # Filter out rows that exceed DIA spectrum bounds (penalty rows)
            valid_rows = nonzero_rows[nonzero_rows < len(dia_spectrum)]
            
            if len(valid_rows) > 1:
                # Calculate Pearson correlation directly without array allocations
                # Extract values using direct indexing to avoid creating intermediate arrays
                
                # Calculate means
                n = len(valid_rows)
                sum_lib = 0.0
                sum_dia = 0.0
                
                for row_idx in valid_rows:
                    lib_val = col_data[row_idx, 0]  # Get library intensity at this row
                    dia_val = dia_spectrum[row_idx, 1]  # Get DIA intensity at this row
                    sum_lib += lib_val
                    sum_dia += dia_val
                
                mean_lib = sum_lib / n
                mean_dia = sum_dia / n
                
                # Calculate correlation components
                sum_lib_dev_sq = 0.0
                sum_dia_dev_sq = 0.0
                sum_cross_dev = 0.0
                
                for row_idx in valid_rows:
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
                    r2_values[col_idx] = sum_cross_dev / denominator
                else:
                    r2_values[col_idx] = np.nan  # Correlation is undefined when one variable is constant
            else:
                r2_values[col_idx] = 0.0
        else:
            # Not enough points for correlation
            r2_values[col_idx] = 0.0
    
    return r2_values


def calculate_unique_peak_features_sparse(
    sparse_lib_matrix_csc: sparse.csc_matrix,
    dia_spectrum: np.ndarray,
    lib_coefficients: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate unique peak analysis features using sparse matrix operations.
    
    This function identifies peaks that are matched to only one candidate (unique peaks),
    then calculates correlations and fraction predictions for these unique peaks.
    Replaces the split-array based unique peak analysis with pure sparse matrix operations.
    
    Args:
        sparse_lib_matrix_csc: Sparse matrix in CSC format with shape (n_dia_peaks, n_candidates)
                              where non-zero values are library intensities at matched peaks
        dia_spectrum: DIA spectrum array with shape (n_peaks, 2) where column 1 contains intensities
        lib_coefficients: Array of coefficients from NNLS optimization, one per candidate
    
    Returns:
        Tuple containing:
        - r2_unique: Array of Pearson correlations for unique peaks, one per candidate
        - frac_unique_pred: Array of fraction unique predicted intensities, one per candidate
    """
    # Handle empty matrix case
    if sparse_lib_matrix_csc.shape[1] == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    
    n_candidates = sparse_lib_matrix_csc.shape[1]
    
    # Find rows that have exactly one non-zero entry (unique peaks)
    # Count non-zeros per row using sparse matrix operations
    row_counts = np.array((sparse_lib_matrix_csc > 0).sum(axis=1)).flatten()
    single_matched_rows = np.where(row_counts == 1)[0]
    
    # Pre-allocate result arrays
    r2_unique = np.zeros(n_candidates, dtype=np.float64)
    frac_unique_pred = np.zeros(n_candidates, dtype=np.float64)
    
    # Process each candidate
    for col_idx in range(n_candidates):
        # Get the column for this candidate
        col_data = sparse_lib_matrix_csc[:, col_idx]
        nonzero_rows, _ = col_data.nonzero()
        
        # Find which of this candidate's matches are unique peaks
        unique_rows = np.intersect1d(nonzero_rows, single_matched_rows)
        
        if len(unique_rows) > 0:
            # Filter out penalty rows (beyond DIA spectrum bounds)
            valid_unique_rows = unique_rows[unique_rows < len(dia_spectrum)]
            
            if len(valid_unique_rows) > 1:
                # Calculate correlation for unique peaks using the same approach as r2_lib_spec
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
            if total_lib_intensity > 0:
                frac_unique_pred[col_idx] = (total_dia_intensity / total_lib_intensity) * lib_coefficients[col_idx]
            else:
                frac_unique_pred[col_idx] = 0.0
        else:
            # No unique peaks for this candidate
            r2_unique[col_idx] = 0.0
            frac_unique_pred[col_idx] = 0.0
    
    return r2_unique, frac_unique_pred


def calculate_frac_dia_intensity_pred(
    frac_lib_intensity: np.ndarray,
    frac_dia_intensity: np.ndarray,
    lib_coefficients: np.ndarray
) -> np.ndarray:
    """
    Calculate predicted DIA intensity fraction for each candidate.
    
    This function computes the predicted fraction of DIA intensity by scaling
    the library intensity fraction by the coefficient and normalizing by the
    observed DIA intensity fraction: (frac_lib * coeff) / frac_dia
    
    Args:
        frac_lib_intensity: Array of library intensity fractions, one per candidate
        frac_dia_intensity: Array of DIA intensity fractions, one per candidate  
        lib_coefficients: Array of NNLS coefficients, one per candidate
        
    Returns:
        np.ndarray: Array of predicted DIA intensity fractions, one per candidate
    """
    # Handle empty case
    if len(frac_lib_intensity) == 0:
        return np.array([], dtype=np.float64)
    
    # Pre-allocate result array
    result = np.zeros(len(frac_lib_intensity), dtype=np.float64)
    
    # Vectorized calculation with safe division
    # Where frac_dia_intensity is 0, result will be 0 (or inf, handled below)
    with np.errstate(divide='ignore', invalid='ignore'):
        result = (frac_lib_intensity * lib_coefficients) / frac_dia_intensity
        
    # Handle division by zero cases - set to 0 when DIA intensity fraction is 0
    result = np.where(frac_dia_intensity == 0, 0.0, result)
    
    return result


def calculate_b_y_ion_counts(
    library: dict,
    ref_pep_cand: List,
    lib_peaks_matched: List
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate b and y ion counts for each candidate.
    
    This function counts the number of matched b-ions and y-ions for each peptide candidate
    by examining the fragment names and their match status.
    
    Args:
        library: Spectral library dictionary
        ref_pep_cand: List of candidate keys for the library
        lib_peaks_matched: List of boolean arrays indicating which fragments matched
        
    Returns:
        Tuple containing:
        - b_counts: Array of matched b-ion counts, one per candidate
        - y_counts: Array of matched y-ion counts, one per candidate
    """
    if len(ref_pep_cand) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    
    b_counts = np.zeros(len(ref_pep_cand), dtype=int)
    y_counts = np.zeros(len(ref_pep_cand), dtype=int)
    
    for idx, (cand_key, matches) in enumerate(zip(ref_pep_cand, lib_peaks_matched)):
        frag_list = library[cand_key]["frags"]
        
        # Count b and y ions for matched fragments
        for frag_name, is_matched in zip(frag_list.keys(), matches):
            if is_matched:
                if "b" in frag_name:
                    b_counts[idx] += 1
                elif "y" in frag_name:
                    y_counts[idx] += 1
    
    return b_counts, y_counts


def calculate_hyperscores(
    library: dict,
    ref_pep_cand: List,
    lib_peaks_matched: List,
    b_counts: np.ndarray,
    y_counts: np.ndarray
) -> np.ndarray:
    """
    Calculate hyperscores using b/y counts and fragment intensities.
    
    Hyperscore formula: max(0, log(dp * factorial(b_count) * factorial(y_count)))
    where dp is the sum of intensities of matched fragments.
    
    Args:
        library: Spectral library dictionary
        ref_pep_cand: List of candidate keys for the library
        lib_peaks_matched: List of boolean arrays indicating which fragments matched
        b_counts: Array of matched b-ion counts, one per candidate
        y_counts: Array of matched y-ion counts, one per candidate
        
    Returns:
        np.ndarray: Array of hyperscores, one per candidate
    """
    if len(ref_pep_cand) == 0:
        return np.array([], dtype=np.float64)
    
    hyperscores = np.zeros(len(ref_pep_cand), dtype=np.float64)
    
    for idx, (cand_key, matches) in enumerate(zip(ref_pep_cand, lib_peaks_matched)):
        frag_list = library[cand_key]["frags"]
        
        # Calculate sum of intensities for matched fragments (dp)
        # Using the same approach as the original frag_to_peak function
        from .utils.misc_functions import frag_to_peak
        frag_array = frag_to_peak(frag_list)
        dp = np.sum(frag_array[:, 1][matches])
        
        # Calculate hyperscore using the standard formula
        import math
        num_b = b_counts[idx]
        num_y = y_counts[idx]
        
        if num_b > 0 or num_y > 0:
            hyperscore = max(0, np.log(dp * math.factorial(num_b) * math.factorial(num_y)))
        else:
            hyperscore = 0.0
            
        hyperscores[idx] = hyperscore
    
    return hyperscores


def calculate_longest_y_ions(
    library: dict,
    ref_pep_cand: List,
    lib_peaks_matched: List
) -> np.ndarray:
    """
    Calculate longest y-ion series for each candidate.
    
    This function finds the highest numbered y-ion that was matched for each candidate.
    
    Args:
        library: Spectral library dictionary
        ref_pep_cand: List of candidate keys for the library
        lib_peaks_matched: List of boolean arrays indicating which fragments matched
        
    Returns:
        np.ndarray: Array of longest y-ion indices, one per candidate
    """
    if len(ref_pep_cand) == 0:
        return np.array([], dtype=int)
    
    # Use the existing longest_y function for each candidate
    from .utils.misc_functions import longest_y
    
    longest_y_ions = np.zeros(len(ref_pep_cand), dtype=int)
    for idx, (cand_key, matches) in enumerate(zip(ref_pep_cand, lib_peaks_matched)):
        frag_list = library[cand_key]["frags"]
        longest_y_ions[idx] = longest_y(frag_list, matches)
    
    return longest_y_ions


def calculate_rt_alignment_features(
    num_lib_peaks_matched: np.ndarray,
    lib_peaks_matched: List[np.ndarray],
    
    # Spectrum data
    dia_spectrum: np.ndarray,
    dia_spec_int: np.ndarray,
    sparse_lib_matrix: sparse.coo_matrix,
    lib_coefficients: np.ndarray,
    
    # Candidate info
    ref_pep_cand: List,
    ref_peaks_in_dia: List[int],
    window_idxs: np.ndarray,
    
    # Reference data
    library: Dict,
    rt_mz: np.ndarray,
    
    # RT/MS1 parameters
    prec_rt: float,
    prec_mz: float,
    windowWidth: float,
    rt_tol: float,
    ms1_tol: float,
    
    # Optional MS1 features
    dino_features: Optional[Any] = None
) -> np.ndarray:
    """
    Calculate features for RT alignment spectral fitting.
    
    This function computes a comprehensive set of 26 features used for scoring
    spectral matches during retention time alignment. These features capture
    various aspects of the match quality including intensity correlations,
    fragment matching, and spectral similarity metrics.
    
    Args:
        ref_spec_values_split: Intensity values for matched peaks per candidate
        ref_spec_row_indices_split: Row indices in DIA spectrum per candidate
        ref_spec_col_indices_split: Column indices for sparse matrix per candidate
        num_lib_peaks_matched: Number of matched peaks per candidate
        lib_peaks_matched: Boolean arrays indicating which peaks matched
        dia_spectrum: Full DIA spectrum (mz, intensity)
        dia_spec_int: DIA intensities for matched peaks with penalty term
        sparse_lib_matrix: Sparse matrix representation of library matches
        lib_coefficients: NNLS coefficients from spectral fitting
        ref_pep_cand: List of candidate peptide identifiers
        ref_peaks_in_dia: Indices of candidates with peaks in DIA
        window_idxs: Window indices for candidates
        library: Spectral library dictionary
        rt_mz: RT and m/z array for all library entries
        prec_rt: Precursor retention time
        prec_mz: Precursor m/z
        windowWidth: m/z window width
        rt_tol: RT tolerance
        ms1_tol: MS1 m/z tolerance
        sparse_row_indices: Row indices from sparse matrix construction
        sparse_col_indices: Column indices from sparse matrix construction
        dino_features: Optional MS1 features for m/z error calculation
        
    Returns:
        np.ndarray: Stacked feature array with shape (n_candidates, 26)
    """
    # Handle empty input case
    if len(ref_pep_cand) == 0 or len(lib_coefficients) == 0:
        return np.empty((0, 26))
    
    # Convert sparse matrix to CSC format for efficient column operations
    sparse_lib_matrix_csc = sparse_lib_matrix.tocsc()
    
    # Basic intensity calculations
    # frac_lib_intensity = [np.sum(i) for i in ref_spec_values_split]  # all ints sum to 1
    frac_lib_intensity = calculate_frac_lib_intensity_sparse(sparse_lib_matrix_csc)
    tic = calculate_tic(dia_spectrum)
    # frac_dia_intensity = [np.sum(dia_spectrum[i, 1])/tic for i in ref_spec_row_indices_split]
    frac_dia_intensity = calculate_frac_dia_intensity_csc(sparse_lib_matrix_csc, dia_spectrum, tic)
    
    # MS1 error calculations
    if dino_features is not None:
        # Recalculate filtered_dino for feature calculation
        filtered_dino = feature_list_mz(feature_list_rt(dino_features, prec_rt, rt_tol=rt_tol),
                                      prec_mz, windowWidth)
        rel_error = ms1_error(np.array(filtered_dino.mz), rt_mz[window_idxs[ref_peaks_in_dia], 1], tol=ms1_tol)
    else:
        rel_error = np.zeros(len(ref_peaks_in_dia))
    
    if len(ref_peaks_in_dia) > 0:
        rt_error = prec_rt - rt_mz[window_idxs[ref_peaks_in_dia], 0]
    else:
        rt_error = np.array([])
    
    # Intensity matching metrics
    frac_int_matched = np.sum(dia_spec_int) / np.sum(dia_spectrum[:, 1])
    predicted_spec = np.squeeze(sparse_lib_matrix * lib_coefficients)[:-1]
    r2all = np_pearson_cor(dia_spec_int[:-1], predicted_spec).statistic
    
    # Per-candidate correlations
    # r2_lib_spec = [np_pearson_cor(i, dia_spectrum[j, 1]).statistic 
    #                for i, j in zip(ref_spec_values_split, ref_spec_row_indices_split)]
    r2_lib_spec = calculate_r2_lib_spec_sparse(sparse_lib_matrix_csc, dia_spectrum)
    
    # Unique peak analysis using sparse matrix operations
    r2_unique, frac_unique_pred = calculate_unique_peak_features_sparse(
        sparse_lib_matrix_csc, dia_spectrum, lib_coefficients
    )
    
    # Ensure lib_coefficients is a numpy array (cvxopt matrix causes broadcasting issues)
    if not isinstance(lib_coefficients, np.ndarray):
        lib_coefficients = np.array(lib_coefficients).flatten()
    
    frac_dia_intensity_pred = calculate_frac_dia_intensity_pred(
        frac_lib_intensity, frac_dia_intensity, lib_coefficients
    )
    
    # Stack spectrum-level features
    r2all = np.ones_like(num_lib_peaks_matched) * r2all
    frac_int_matched = np.ones_like(num_lib_peaks_matched) * frac_int_matched
    frac_int_pred = (np.ones_like(num_lib_peaks_matched) * np.sum(predicted_spec)) / tic
    frac_int_matched_pred = (np.ones_like(num_lib_peaks_matched) * np.sum(predicted_spec)) / np.sum(dia_spec_int)
    frac_int_matched_pred_sigcoeff = np.zeros_like(num_lib_peaks_matched)
    large_coeff_cosine = np.zeros_like(num_lib_peaks_matched)
    
    # Fragment-based scores
    b_counts, y_counts = calculate_b_y_ion_counts(library, ref_pep_cand, lib_peaks_matched)
    hyperscores = calculate_hyperscores(library, ref_pep_cand, lib_peaks_matched, b_counts, y_counts)
    longest_y_ions = calculate_longest_y_ions(library, ref_pep_cand, lib_peaks_matched)
    
    # Advanced scoring metrics
    scribe_scores = get_scribe_csc(
        sparse_lib_matrix_csc,
        dia_spectrum[:, 1]
    )
    
    residuals, y_pred = get_residuals_csc(
        sparse_lib_matrix_csc,
        dia_spectrum[:, 1],
        lib_coefficients
    )
    
    manhattan_distances, fitted_spectral_contrasts = get_manhattan_distance_csc(
        sparse_lib_matrix_csc,
        dia_spectrum[:, 1],
        y_pred
    )
    
    gof_stats, max_unmatched_residuals, max_matched_residuals = gof_stat_csc(
        sparse_lib_matrix_csc,
        residuals,
        dia_spectrum[:, 1],
        lib_coefficients
    )
    
    # Stack all features
    if len(ref_peaks_in_dia) > 0:
        # Debug: print lengths of all arrays going into stack
        arrays_to_stack = [
            num_lib_peaks_matched,
            frac_lib_intensity,
            frac_dia_intensity,
            rel_error,
            rt_error,
            frac_int_matched,
            frac_int_pred,
            r2all,
            r2_lib_spec,
            r2_unique,
            frac_unique_pred,
            frac_dia_intensity_pred,
            hyperscores,
            b_counts,
            y_counts,
            longest_y_ions,
            scribe_scores,
            max_unmatched_residuals,
            max_matched_residuals,
            gof_stats,
            manhattan_distances,
            fitted_spectral_contrasts,
            frac_int_matched_pred,
            frac_int_matched_pred_sigcoeff,
            large_coeff_cosine,
            rt_mz[:, 1][window_idxs[ref_peaks_in_dia]]
        ]
        
        array_names = [
            "num_lib_peaks_matched", "frac_lib_intensity", "frac_dia_intensity",
            "rel_error", "rt_error", "frac_int_matched", "frac_int_pred",
            "r2all", "r2_lib_spec", "r2_unique", "frac_unique_pred",
            "frac_dia_intensity_pred", "hyperscores", "b_counts", "y_counts",
            "longest_y_ions", "scribe_scores", "max_unmatched_residuals",
            "max_matched_residuals", "gof_stats", "manhattan_distances",
            "fitted_spectral_contrasts", "frac_int_matched_pred",
            "frac_int_matched_pred_sigcoeff", "large_coeff_cosine", "rt_mz"
        ]
        
        #print("DEBUG: Array shapes going into np.stack:")
        #for name, arr in zip(array_names, arrays_to_stack):
        #    print(f"  {name}: shape={getattr(arr, 'shape', 'scalar')}, type={type(arr)}")
        
        features = np.stack(arrays_to_stack, -1)
    else:
        # Return empty array with correct shape
        features = np.zeros((0, 26))
    
    return features


def preprocess_dia_spectrum(dia_spectrum: np.ndarray, mz_tol: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Preprocess DIA spectrum by merging peaks within tolerance and calculating centroid breaks.
    
    This function groups peaks that fall within the mass tolerance of each other,
    sums their intensities, and creates tolerance windows for peak matching.
    
    Args:
        dia_spectrum: 2D array with m/z values in column 0 and intensities in column 1
        mz_tol: Mass tolerance as a fraction (e.g., 10 ppm = 10e-6)
        
    Returns:
        Tuple containing:
        - merged_spectrum: 2D array of merged peaks (m/z, intensity)
        - centroid_breaks: Array of tolerance window boundaries
        - bin_centers: Array of bin center m/z values
        
    Example:
        >>> spectrum = np.array([[100.0, 50.0], [100.001, 30.0], [200.0, 100.0]])
        >>> merged, breaks, centers = preprocess_dia_spectrum(spectrum, 10e-6)
    """
    # Handle empty spectrum
    if len(dia_spectrum) == 0:
        return np.array([]).reshape(0, 2), np.array([]), np.array([])
    
    # Find groups of peaks within tolerance
    merged_coords_idxs = np.searchsorted(
        dia_spectrum[:, 0] + mz_tol * dia_spectrum[:, 0], 
        dia_spectrum[:, 0]
    )
    
    # Get unique peak group indices
    unique_idxs = np.unique(merged_coords_idxs)
    
    # Sum intensities for each peak group
    merged_intensities = np.zeros(len(unique_idxs))
    for i, unique_idx in enumerate(unique_idxs):
        # Sum all intensities that belong to this group
        group_mask = merged_coords_idxs == unique_idx
        merged_intensities[i] = np.sum(dia_spectrum[group_mask, 1])
    
    # Filter out zero-intensity peaks
    non_zero_mask = merged_intensities != 0
    merged_coords = dia_spectrum[unique_idxs[non_zero_mask], 0]
    merged_intensities = merged_intensities[non_zero_mask]
    
    # Create merged spectrum
    merged_spectrum = np.array((merged_coords, merged_intensities)).transpose()
    
    # Calculate tolerance windows for each merged peak
    centroid_breaks = np.concatenate((
        merged_spectrum[:, 0] - mz_tol * merged_spectrum[:, 0],
        merged_spectrum[:, 0] + mz_tol * merged_spectrum[:, 0]
    ))
    centroid_breaks = np.sort(centroid_breaks)
    
    # Calculate bin centers
    bin_centers = np.mean(np.stack((centroid_breaks[::2], centroid_breaks[1::2]), 1), 1)
    
    return merged_spectrum, centroid_breaks, bin_centers


def format_rt_alignment_output(
    lib_coefficients: np.ndarray,
    ref_pep_cand: List[Tuple],
    features: np.ndarray,
    spec_idx: int,
    ms1_spec: Any,
    prec_mz: float,
    prec_rt: float,
    library: Dict[Tuple, Dict],
    return_frags: bool,
    # Additional parameters for detailed fragment info calculation
    ref_pep_cand_loc: List[np.ndarray] = None,
    norm_intensities: List[np.ndarray] = None,
    ref_pep_cand_list: List[np.ndarray] = None,
    bin_centers: np.ndarray = None,
    dia_spectrum: np.ndarray = None,
    unique_row_idxs: List[int] = None
) -> Union[List[List], Tuple[List[List], List]]:
    """
    Format output for RT alignment results.
    
    Args:
        lib_coefficients: NNLS coefficients for each candidate
        ref_pep_cand: List of (sequence, charge) tuples for candidates
        features: Feature matrix from calculate_rt_alignment_features
        spec_idx: Spectrum index
        ms1_spec: MS1 spectrum object (can be None)
        prec_mz: Precursor m/z
        prec_rt: Precursor retention time
        library: Spectral library
        return_frags: Whether to return fragment information
        ref_pep_cand_loc: Peak location arrays (needed for detailed fragment info)
        norm_intensities: Normalized intensities (needed for detailed fragment info)
        ref_pep_cand_list: Peak arrays (needed for detailed fragment info)
        bin_centers: m/z bin centers (needed for detailed fragment info)
        dia_spectrum: DIA spectrum (needed for detailed fragment info)
        unique_row_idxs: Unique row indices from sparse matrix (needed for detailed fragment info)
        
    Returns:
        If return_frags=False: List of output rows
        If return_frags=True: Tuple of (output rows, [frag_errors, frag_mz])
    """
    # Select non-zero coefficients
    non_zero_coeffs = [c for c in lib_coefficients if c != 0]
    non_zero_coeffs_idxs = [i for i, c in enumerate(lib_coefficients) if c != 0]
    
    # Default output if no matches
    output = [[0, spec_idx, 0, 0, prec_mz, prec_rt, *np.zeros(len(names)-6)]]
    
    if len(non_zero_coeffs) > 0:
        lib_spec_ids = [ref_pep_cand[i] for i in range(len(ref_pep_cand)) if lib_coefficients[i] != 0]
        all_spec_ids = lib_spec_ids
        all_features = features
        
        # Calculate detailed fragment information if needed
        if (ref_pep_cand_loc is not None and norm_intensities is not None and 
            ref_pep_cand_list is not None and bin_centers is not None and 
            dia_spectrum is not None and unique_row_idxs is not None):
            
            frag_info = extract_detailed_fragment_info(
                ref_pep_cand_loc=ref_pep_cand_loc,
                norm_intensities=norm_intensities,
                ref_pep_cand=ref_pep_cand,
                ref_pep_cand_list=ref_pep_cand_list,
                bin_centers=bin_centers,
                dia_spectrum=dia_spectrum,
                library=library,
                unique_row_idxs=unique_row_idxs
            )
            
            # Format fragment information
            all_ms2_frags = [[";".join(map(str, j)) for j in i] for i in zip(
                frag_info['frag_names'],
                frag_info['frag_errors'],
                frag_info['lib_frag_mz'],
                frag_info['lib_frag_int'],
                frag_info['obs_frag_int']
            )]
        else:
            # No detailed fragment info available, use empty values
            all_ms2_frags = [[""] * 5 for _ in range(len(non_zero_coeffs))]
            frag_info = {'frag_errors': [], 'frag_mz': []}
        
        return_prot = config.protein_column in library[next(iter(library))]
        
        output = [[
            non_zero_coeffs[i],
            spec_idx,
            ms1_spec.scan_num if ms1_spec else -1,
            all_spec_ids[i][0],
            all_spec_ids[i][1],
            prec_mz,
            prec_rt,
            *all_features[j],
            *all_ms2_frags[j],
            config.args.mzml,
            library[(re.sub("Decoy_", "", all_spec_ids[i][0]), all_spec_ids[i][1])][config.protein_column] if return_prot else "NA"
        ] for i, j in zip(range(len(non_zero_coeffs)), non_zero_coeffs_idxs)]
    
    if return_frags:
        return output, [frag_info['frag_errors'], frag_info['frag_mz']]
    else:
        return output


def solve_nnls_simple(sparse_lib_matrix: sparse.coo_matrix, dia_spec_int: np.ndarray) -> np.ndarray:
    """
    Solve non-negative least squares optimization.
    
    Simple wrapper around the NNLS solver for cleaner code organization.
    
    Args:
        sparse_lib_matrix: Sparse matrix of library spectra
        dia_spec_int: DIA spectrum intensities
        
    Returns:
        Array of coefficients from NNLS solution
    """
    fit_results = sparse_nnls.lsqnonneg(sparse_lib_matrix, dia_spec_int, {"show_progress": False})
    return fit_results['x']


def extract_basic_fragment_info(ref_pep_cand_loc: List[np.ndarray]) -> Dict[str, Any]:
    """
    Extract basic fragment information needed for sparse matrix construction.
    
    This lightweight function extracts only the essential information needed
    for RT alignment sparse matrix building, without dependencies on split arrays.
    
    Args:
        ref_pep_cand_loc: List of peak location arrays for each candidate
        
    Returns:
        Dictionary containing:
        - lib_peaks_matched: Boolean arrays indicating which peaks matched
        - num_lib_peaks_matched: Number of matched peaks per candidate
    """
    # Extract which library peaks are matched in DIA (odd indices)
    lib_peaks_matched = [j % 2 == 1 for j in ref_pep_cand_loc]
    num_lib_peaks_matched = np.array([np.sum(i) for i in lib_peaks_matched])
    
    return {
        'lib_peaks_matched': lib_peaks_matched,
        'num_lib_peaks_matched': num_lib_peaks_matched
    }


def extract_detailed_fragment_info(
    ref_pep_cand_loc: List[np.ndarray],
    norm_intensities: List[np.ndarray],
    ref_pep_cand: List[Tuple],
    ref_pep_cand_list: List[np.ndarray],
    bin_centers: np.ndarray,
    dia_spectrum: np.ndarray,
    library: Dict[Tuple, Dict],
    unique_row_idxs: List[int]
) -> Dict[str, Any]:
    """
    Extract detailed fragment information for output formatting.
    
    This function calculates fragment-level details using sparse matrix results
    instead of split arrays, for use in output formatting functions.
    
    Args:
        ref_pep_cand_loc: List of peak location arrays for each candidate
        norm_intensities: List of normalized intensity arrays for each candidate
        ref_pep_cand: List of (sequence, charge) tuples for matched candidates
        ref_pep_cand_list: List of peak arrays for matched candidates
        bin_centers: Centers of m/z bins from DIA spectrum preprocessing
        dia_spectrum: Preprocessed DIA spectrum
        library: Spectral library dictionary
        unique_row_idxs: Unique DIA spectrum row indices from sparse matrix construction
        
    Returns:
        Dictionary containing:
        - frag_errors: Fragment mass errors (per candidate)
        - lib_frag_mz: Library fragment m/z values (per candidate)
        - lib_frag_int: Library fragment intensities (per candidate)
        - obs_frag_int: Observed fragment intensities (per candidate)
        - frag_names: Fragment ion names (per candidate)
        - frag_mz: Fragment m/z values for output
    """
    # Extract which library peaks are matched in DIA
    lib_peaks_matched = [j % 2 == 1 for j in ref_pep_cand_loc]
    
    # Calculate DIA row indices for matched peaks per candidate
    dia_row_indices_per_candidate = []
    for locations, matched in zip(ref_pep_cand_loc, lib_peaks_matched):
        if np.any(matched):
            matched_locations = locations[matched]
            # Convert to DIA spectrum indices: ((loc + 1) / 2) - 1
            dia_indices = np.int32(((matched_locations + 1) / 2) - 1)
            dia_row_indices_per_candidate.append(dia_indices)
        else:
            dia_row_indices_per_candidate.append(np.array([], dtype=np.int32))
    
    # Calculate fragment-level information
    frag_errors = []
    lib_frag_mz = []
    lib_frag_int = []
    obs_frag_int = []
    frag_names = []
    frag_mz = []
    
    for i, (locations, matched, dia_indices) in enumerate(zip(ref_pep_cand_loc, lib_peaks_matched, dia_row_indices_per_candidate)):
        if len(dia_indices) > 0:
            # Fragment errors: (observed_mz - library_mz) / observed_mz
            obs_mz = bin_centers[dia_indices]
            lib_mz = ref_pep_cand_list[i][:, 0][matched]
            frag_errors.append((obs_mz - lib_mz) / obs_mz)
            
            # Fragment m/z and intensities
            lib_frag_mz.append(ref_pep_cand_list[i][:, 0][matched])
            lib_frag_int.append(ref_pep_cand_list[i][:, 1][matched])
            obs_frag_int.append(dia_spectrum[dia_indices, 1])
            
            # Fragment names
            frag_names.append(library[ref_pep_cand[i]]["ordered_frags"][matched])
            frag_mz.append(ref_pep_cand_list[i][:, 0][matched])
        else:
            # No matched peaks for this candidate
            frag_errors.append(np.array([]))
            lib_frag_mz.append(np.array([]))
            lib_frag_int.append(np.array([]))
            obs_frag_int.append(np.array([]))
            frag_names.append(np.array([]))
            frag_mz.append(np.array([]))
    
    return {
        'frag_errors': frag_errors,
        'lib_frag_mz': lib_frag_mz,
        'lib_frag_int': lib_frag_int,
        'obs_frag_int': obs_frag_int,
        'frag_names': frag_names,
        'frag_mz': frag_mz
    }


# ===== MAIN FITTING FUNCTIONS =====

def fit_to_lib(dia_spec,library,rt_mz,all_keys,dino_features=None,rt_filter=False,ms1_mz=None,
               ms1_spectra = None,
               rt_tol = config.rt_tol,
               ms1_tol = config.ms1_tol,
               mz_tol = config.mz_tol,
               return_frags = False,
               frac_matched = 0.5):
    """
    Perform retention time alignment spectral fitting.
    
    This function is specifically designed for RT alignment and differs from fit_to_lib2
    in that it filters out decoy peptides before processing. It matches a DIA spectrum
    against a spectral library to identify peptides for retention time calibration.
    
    Pipeline Overview:
    1. Extract spectrum information and find closest MS1 spectrum
    2. Filter candidates by mass window (filter_candidates_by_window)
    3. Remove decoy peptides (RT alignment only uses targets)
    4. Preprocess DIA spectrum (preprocess_dia_spectrum)
    5. Check MS1 peaks (check_ms1_peaks)
    6. Filter by peak matching (filter_candidates_by_peak_matching)
    7. Extract fragment information for matched peaks
    8. Build sparse matrix (build_sparse_matrix_simple)
    9. Solve NNLS optimization
    10. Calculate RT alignment features (calculate_rt_alignment_features)
    11. Format output for RT alignment
    
    Args:
        dia_spec: DIA spectrum object with scan_num, prec_mz, RT, peak_list()
        library: Spectral library dict with peptide info and spectra
        rt_mz: Array of retention times and m/z values for all library entries
        all_keys: List of (sequence, charge) tuples for all library entries
        dino_features: Optional DINO features for filtering
        rt_filter: Whether to apply retention time filtering
        ms1_mz: Optional MS1 m/z values for filtering
        ms1_spectra: Optional list of MS1 spectra for MS1 matching
        rt_tol: Retention time tolerance (default from config)
        ms1_tol: MS1 mass tolerance (default from config)
        mz_tol: MS2 mass tolerance (default from config)
        return_frags: Whether to return fragment information
        frac_matched: Minimum fraction of peaks that must match (default 0.5)
        
    Returns:
        If return_frags=False:
            List of output rows for RT alignment
        If return_frags=True:
            Tuple of (output_rows, [frag_errors, frag_mz])
    """
    # Extract spectrum information
    spec_idx = dia_spec.scan_num
    top_n = config.top_n
    atleast_m = config.atleast_m
    dia_spectrum = np.stack(dia_spec.peak_list(), 1)
    prec_mz = dia_spec.prec_mz
    prec_rt = dia_spec.RT
    windowWidth = window_width(dia_spec)
    
    # Get closest MS1 spectrum using the shared function
    ms1_spec = None
    if ms1_spectra is not None:
        ms1_spec = get_closest_ms1(prec_rt, ms1_spectra)
   
    # Use extracted function for filtering
    window_idxs, mass_window_candidates_all = filter_candidates_by_window(
        rt_mz=rt_mz,
        all_keys=all_keys,
        prec_mz=prec_mz,
        prec_rt=prec_rt,
        windowWidth=windowWidth,
        ms1_mz=ms1_mz,
        rt_filter=rt_filter,
        ms1_tol=ms1_tol,
        rt_tol=rt_tol,
        dino_features=dino_features
    )
    
    # Filter out decoys for RT alignment (fit_to_lib is only used for RT alignment)
    mass_window_candidates = [key for key in mass_window_candidates_all if not library[key].get('is_decoy', False)] 
    candidate_peaks = [library[i]['spectrum'] for i in mass_window_candidates]
    
    # Process dia spectrum using extracted function
    dia_spectrum, centroid_breaks, bin_centers = preprocess_dia_spectrum(dia_spectrum, mz_tol)
    
    # Filter precursors based on resp. MS1 peak
    ms1_peak = check_ms1_peaks(rt_mz, window_idxs, ms1_spec)
    
    # Use extracted function for peak matching and filtering
    ref_peaks_in_dia, ref_coords, top_ten, all_norm_intensities = filter_candidates_by_peak_matching(
        candidate_peaks=candidate_peaks,
        centroid_breaks=centroid_breaks,
        ms1_peak=ms1_peak,
        top_n=top_n,
        atleast_m=atleast_m,
        frac_matched=frac_matched
    )
    
    # filter database further to those that match the required num peaks
    ref_pep_cand_loc = [ref_coords[i] for i in ref_peaks_in_dia]
    ref_pep_cand_list = [candidate_peaks[i] for i in ref_peaks_in_dia]
    ref_pep_cand = [mass_window_candidates[i] for i in ref_peaks_in_dia]
    
    norm_intensities = [all_norm_intensities[i] for i in ref_peaks_in_dia]
    
    # Extract fragment information using the new helper function
    frag_info = extract_basic_fragment_info(ref_pep_cand_loc)
    
    # Default values for when no candidates match
    lib_coefficients = np.array([])
    features = np.zeros((len(ref_pep_cand) if ref_pep_cand else 0, 26))
    unique_row_idxs = []  # Initialize default for when no candidates match
    
    #if (len(frag_info['ref_spec_row_indices_split']) > 0 and 
    #    len(frag_info['ref_spec_col_indices_split']) > 0 and 
    #    len(frag_info['ref_spec_values_split']) > 0):
    if len(frag_info['lib_peaks_matched']) > 0:  
        # Use the new build_sparse_matrix_direct function
        sparse_lib_matrix, dia_spec_int, unique_row_idxs, fragment_info = build_sparse_matrix_direct(
            ref_pep_cand_loc,
            norm_intensities,
            ref_pep_cand,
            dia_spectrum
        )
        
        # Solve NNLS using the new wrapper function
        lib_coefficients = solve_nnls_simple(sparse_lib_matrix, dia_spec_int)
        
        # Calculate features
        features = calculate_rt_alignment_features(
            num_lib_peaks_matched=frag_info['num_lib_peaks_matched'],
            lib_peaks_matched=frag_info['lib_peaks_matched'],
            
            # Spectrum data
            dia_spectrum=dia_spectrum,
            dia_spec_int=dia_spec_int,
            sparse_lib_matrix=sparse_lib_matrix,
            lib_coefficients=lib_coefficients,
            
            # Candidate info
            ref_pep_cand=ref_pep_cand,
            ref_peaks_in_dia=ref_peaks_in_dia,
            window_idxs=window_idxs,
            
            # Reference data
            library=library,
            rt_mz=rt_mz,
            
            # RT/MS1 parameters
            prec_rt=prec_rt,
            prec_mz=prec_mz,
            windowWidth=windowWidth,
            rt_tol=rt_tol,
            ms1_tol=ms1_tol,
            
            # Optional MS1 features
            dino_features=dino_features
        )
        
    
    # Format and return output using the new function
    return format_rt_alignment_output(
        lib_coefficients=lib_coefficients,
        ref_pep_cand=ref_pep_cand,
        features=features,
        spec_idx=spec_idx,
        ms1_spec=ms1_spec,
        prec_mz=prec_mz,
        prec_rt=prec_rt,
        library=library,
        return_frags=return_frags,
        ref_pep_cand_loc=ref_pep_cand_loc,
        norm_intensities=norm_intensities,
        ref_pep_cand_list=ref_pep_cand_list,
        bin_centers=bin_centers,
        dia_spectrum=dia_spectrum,
        unique_row_idxs=unique_row_idxs
    )


def fit_to_lib2(dia_spec,
                library,
                rt_mz,
                all_keys,
                dino_features=None,rt_filter=False,ms1_mz=None,
               ms1_spectra = None,
               rt_tol = config.rt_tol,
               ms1_tol = config.ms1_tol,
               mz_tol = config.mz_tol,
               return_frags = False,
               decoy=True):
    """
    Perform spectral fitting of DIA spectrum against spectral library.
    
    This is the main entry point for spectral matching in JMod. It processes a DIA
    (Data-Independent Acquisition) spectrum and matches it against a spectral library
    containing both target and decoy peptides.
    
    Pipeline Overview:
    1. Extract spectrum information and handle empty spectra
    2. Find candidates within mass window (filter_candidates_by_window)
    3. Preprocess DIA spectrum by merging peaks (preprocess_dia_spectrum)
    4. Separate targets/decoys and create unified structure (separate_library_candidates)
    5. Process candidates and create sparse matrix (create_entries)
    6. Solve NNLS optimization (process_matrix)
    7. Calculate scoring features (calculate_features)
    8. Format output rows (format_spectral_fitting_output)
    
    Args:
        dia_spec: DIA spectrum object with scan_num, prec_mz, RT, peak_list()
        library: Spectral library dict with peptide info and spectra
        rt_mz: Array of retention times and m/z values for all library entries
        all_keys: List of (sequence, charge) tuples for all library entries
        dino_features: Optional DINO features for filtering
        rt_filter: Whether to apply retention time filtering
        ms1_mz: Optional MS1 m/z values for filtering
        ms1_spectra: Optional list of MS1 spectra for MS1 matching
        rt_tol: Retention time tolerance (default from config)
        ms1_tol: MS1 mass tolerance (default from config)
        mz_tol: MS2 mass tolerance (default from config)
        return_frags: Whether to return fragment information
        decoy: Whether to include decoy peptides in matching
        
    Returns:
        If return_frags=False:
            List of output rows, each containing coefficient, IDs, features, etc.
        If return_frags=True:
            Tuple of (output_rows, [frag_errors, lib_frag_mz])
            
    Example:
        >>> result = fit_to_lib2(
        ...     dia_spec=spectrum,
        ...     library=spec_lib,
        ...     rt_mz=rt_mz_array,
        ...     all_keys=peptide_keys
        ... )
        >>> for row in result:
        ...     if row[0] > 0:  # Non-zero coefficient
        ...         print(f"Matched {row[3]} with score {row[0]}")
    """
    # 1. Extract spectrum information (same as original)
    spec_idx = dia_spec.scan_num
    peak_list = dia_spec.peak_list()
    # Handle empty spectrum
    if len(peak_list) == 0:
        dia_spectrum = np.array([]).reshape(0, 2)  # Empty 0x2 array
    else:
        dia_spectrum = np.stack(peak_list, 1)
    prec_mz = dia_spec.prec_mz
    prec_rt = dia_spec.RT
    windowWidth = window_width(dia_spec)
    
    ms1_spec = None
    if ms1_spectra is not None:
        ms1_spec = get_closest_ms1(prec_rt, ms1_spectra)
    
    # 2. Filter candidates by mass window using extracted function
    window_idxs, mass_window_candidates = filter_candidates_by_window(
        rt_mz=rt_mz,
        all_keys=all_keys,
        prec_mz=prec_mz,
        prec_rt=prec_rt,
        windowWidth=windowWidth,
        ms1_mz=ms1_mz,
        rt_filter=rt_filter,
        ms1_tol=ms1_tol,
        rt_tol=rt_tol,
        dino_features=dino_features
    )
    
    # Early exit if no candidates
    if len(mass_window_candidates) == 0:
        empty_result = [create_empty_output_row(
            spec_idx, 
            ms1_spec.scan_num if ms1_spec else 0,
            prec_mz, 
            prec_rt, 
            len(names)
        )]
        if return_frags:
            return empty_result, [[], []]
        else:
            return empty_result
    
    # 3. Process DIA spectrum using extracted function
    dia_spectrum, centroid_breaks, bin_centers = preprocess_dia_spectrum(dia_spectrum, mz_tol)
    
    # ===== PROCESSING STARTS HERE =====
    
    # 4. Create unified candidates structure using extracted function
    unified = separate_library_candidates(
        mass_window_candidates=mass_window_candidates,
        library=library,
        include_decoys=decoy
    )
    
    # 5. Process all candidates in ONE call
    updated_unified, matrix_data, additional_outputs = create_entries(
        centroid_breaks=centroid_breaks,
        unified_candidates=unified,
        top_n=config.top_n,
        atleast_m=config.atleast_m,
        prec_mzs=np.array([library[k]["prec_mz"] for k in unified.candidates]),
        ms1_spec=ms1_spec,
        ms1_tol=ms1_tol,
        library=library,
        bin_centers=bin_centers,
        dia_spectrum=dia_spectrum
    )
    
    # Check if any candidates passed filtering
    if len(updated_unified.peaks_in_dia) == 0:
        empty_result = [create_empty_output_row(
            spec_idx,
            ms1_spec.scan_num if ms1_spec else 0,
            prec_mz,
            prec_rt,
            len(names)
        )]
        if return_frags:
            return empty_result, [[], []]
        else:
            return empty_result
    
    # 6. Build matrix and solve NNLS
    matrix_results = process_matrix(
        unified_candidates=updated_unified,
        matrix_data=matrix_data,
        additional_outputs=additional_outputs,
        dia_spectrum=dia_spectrum,
        unmatched_fit_type=config.unmatched_fit_type
    )
    
    # 7. Calculate features
    unified_features = calculate_features(
        unified_candidates=updated_unified,
        matrix_data=matrix_data,
        additional_outputs=additional_outputs,
        dia_spectrum=dia_spectrum,
        prec_rt=prec_rt,
        lib_coefficients=matrix_results['lib_coefficients'],
        sparse_matrix=matrix_results['sparse_matrix'],
        peak_idx_convertor=matrix_results['peak_idx_convertor'],
        unique_row_idxs=matrix_results['unique_row_idxs'],
        rt_mz=rt_mz,
        window_idxs=window_idxs,
        library=library
    )
    
    # 8. Format output using extracted function
    output = format_spectral_fitting_output(
        lib_coefficients=matrix_results['lib_coefficients'],
        unified_candidates=updated_unified,
        unified_features=unified_features,
        additional_outputs=additional_outputs,
        spec_idx=spec_idx,
        ms1_spec=ms1_spec,
        prec_mz=prec_mz,
        prec_rt=prec_rt,
        library=library,
        config=config
    )
    
    if return_frags:
        frag_errors = matrix_results.get('frag_errors', [])
        lib_frag_mz = matrix_results.get('lib_frag_mz', [])
        return output, [frag_errors, lib_frag_mz]
    else:
        return output


