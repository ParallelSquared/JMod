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
from typing import List, Tuple, Optional, Dict, Any

import warnings
import ptinnls as sparse_nnls

from scipy import stats
from scipy import sparse
from pyteomics import mass
import re
from .utils.io.read_output import names
import src.config as config

from .utils.misc_functions import createTolWindows, window_width, feature_list_mz, feature_list_rt, \
hyperscore_b_y, longest_y, closest_ms1spec, closest_peak_diff, cosim, np_pearson_cor, ms1_error
from .utils.parse_peptides import change_seq, convert_frags
from .models.spec_lib.spec_lib import frag_to_peak
from .utils.spectral_similarity_metrics import (
    get_closest_ms1, get_scribe, get_residuals, max_matched_residual, 
    gof_stat, get_manhattan_distance
)


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
    n_targets = np.sum(~is_decoy)
    
    for i, (row_idx, col_idx, vals) in enumerate(zip(row_indices_split, col_indices_split, values_split)):
        if len(row_idx) == 0:
            continue
            
        # Apply offset for decoys
        if is_decoy[i]:
            adjusted_col_idx = col_idx - n_targets + n_targets  # This simplifies to just col_idx
            offset = n_targets
        else:
            adjusted_col_idx = col_idx
            offset = 0
            
        # Compute predictions
        for r, c, v in zip(row_idx, adjusted_col_idx, vals):
            if c + offset < len(coeffs):
                y_pred[r] += v * coeffs[c + offset]
    
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


def hyperscore2(frags,frag_names_matched):
    
    num_b = sum(["b" in i for i in frag_names_matched if "iso" not in i])
    num_y = sum(["y" in i for i in frag_names_matched if "iso" not in i])
    dp = np.sum([frags[i] for i in frag_names_matched if "iso" not in i])
    return max(0,np.log(dp*np.math.factorial(num_b)*np.math.factorial(num_y))), num_b, num_y



# ===== MAIN FITTING FUNCTIONS =====

#@profile
def fit_to_lib(dia_spec,library,rt_mz,all_keys,dino_features=None,rt_filter=False,ms1_mz=None,
               ms1_spectra = None,
               rt_tol = config.rt_tol,
               ms1_tol = config.ms1_tol,
               mz_tol = config.mz_tol,
               return_frags = False,
               frac_matched = 0.5):
    # spec_idx,dia_spec,library = inputs
    
    spec_idx=dia_spec.scan_num
    
    # mz_tol = config.mz_tol
    # rt_tol = min(config.rt_tol,config.opt_rt_tol)
    # ms1_tol = min(config.ms1_tol,config.opt_ms1_tol)
    top_n=config.top_n
    atleast_m=config.atleast_m
    spec = dia_spec#spectra.ms2scans[spec_idx]
    dia_spectrum = np.stack(spec.peak_list(),1)
    prec_mz = spec.prec_mz
    prec_rt = spec.RT
    # spec_idx = spec.id
    
    windowWidth = window_width(dia_spec)
    
    
    if ms1_spectra is not None:
        ms1_rt = np.array([i.RT for i in ms1_spectra])
        closest_ms1_scan_idx = closest_ms1spec(prec_rt, ms1_rt)
        ms1_spec = ms1_spectra[closest_ms1_scan_idx]
    
    
    lib_coefficients = []
   
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
    
    # # filter possible lib entries for windows.. NB: DONT LIKE HOW I DO SAME LOOP TWICE
    # candidate_lib = [spectrum for key,spectrum in library.items() if spectrum["prec_mz"]>spec.ms1window[0] and spectrum["prec_mz"]<spec.ms1window[1]]
    # mass_window_candidates = [key for key,spectrum in library.items() if spectrum["prec_mz"]>spec.ms1window[0] and spectrum["prec_mz"]<spec.ms1window[1]]
    # # list of peaks from each candiate pep
    # # candidate_peaks = [SpecLib.frag_to_peak(i["frags"]) for i in candidate_lib]
    # candidate_peaks = [i["spectrum"] for i in candidate_lib]
    
    
    
    ###### Process dia spectrum using extracted function
    dia_spectrum, centroid_breaks, bin_centers = preprocess_dia_spectrum(dia_spectrum, mz_tol)
    
    # if "spec_frags" in library[all_keys[0]].keys():
    #     spec_peaks = [library[i]['spec_frags'] for i in mass_window_candidates]
    #     ref_coords = [np.searchsorted(centroid_breaks,M[:,0]) for M in candidate_peaks]
    #     spec_ref_coords = [np.searchsorted(centroid_breaks,M[:,0]) for M in spec_peaks]
    #     top_ten = [np.searchsorted(centroid_breaks,M[np.argsort(-M[:,1])[0:min(top_n,M.shape[0])],0]) for M in spec_peaks]
       
    #     ## Filter precursors based on resp. MS1 peak
    #     ms1_peak = ~np.isnan([closest_peak_diff(mz,ms1_spec.mz) for mz in rt_mz[window_idxs,1]])
        
    #     # does the top ten peaks fall between centroid breaks? i.e. odd numbers (%2==1), 
    #     ref_peaks_in_dia = [i for i in range(len(spec_peaks)) if len([a for a in top_ten[i] if a%2 ==1])>atleast_m]
        
    #     all_norm_intensities = [M[:,1]/sum(M[:,1]) for M in spec_peaks]
    #     # ref_peaks_in_dia = [i for i in range(len(candidate_peaks)) if np.sum(all_norm_intensities[i][(spec_ref_coords[i]%2)==1])>0.5 and np.sum(top_ten[i]%2)>atleast_m and ms1_peak[i]]
    #     ref_peaks_in_dia = [i for i in range(len(candidate_peaks)) if np.sum(top_ten[i]%2)>atleast_m]
    #     # ref_peaks_in_dia = [i for i in range(len(spec_peaks)) if np.sum(all_norm_intensities[i][(spec_ref_coords[i]%2)==1])>0.5 and np.sum(top_ten[i]%2)>atleast_m and ms1_peak[i] and top_ten[i][0]%2==1 and np.sum(top_ten[i][:3]%2==1)>=2]
    #     # ref_peaks_in_dia = [i for i in range(len(candidate_peaks)) if  np.sum(top_ten[i]%2)>atleast_m and ms1_peak[i]]
    
    
    # else:
    # lib_idx=0
    # M = candidate_peaks[lib_idx]
    ref_coords = [np.searchsorted(centroid_breaks,M[:,0]) for M in candidate_peaks]
    top_ten = [np.searchsorted(centroid_breaks,M[np.argsort(-M[:,1])[0:min(top_n,M.shape[0])],0]) for M in candidate_peaks]
    
    ## Filter precursors based on resp. MS1 peak
    ms1_peak = ~np.isnan([closest_peak_diff(mz,ms1_spec.mz) for mz in rt_mz[window_idxs,1]])
    
    # does the top ten peaks fall between centroid breaks? i.e. odd numbers (%2==1), 
    ref_peaks_in_dia = [i for i in range(len(candidate_peaks)) if len([a for a in top_ten[i] if a%2 ==1])>atleast_m]
    prop_ref_peaks_in_dia = [len([a for a in top_ten[i] if a%2 ==1])/candidate_peaks[i].shape[0] for i in range(len(candidate_peaks))]
    
    all_norm_intensities = [M[:,1]/sum(M[:,1]) for M in candidate_peaks]
    # ref_peaks_in_dia = [i for i in range(len(candidate_peaks)) if np.sum(all_norm_intensities[i][(ref_coords[i]%2)==1])>0.5 and np.sum(top_ten[i]%2)>atleast_m and ms1_peak[i]]
    ref_peaks_in_dia = [i for i in range(len(candidate_peaks)) if np.sum(all_norm_intensities[i][(ref_coords[i]%2)==1])>frac_matched and np.sum(top_ten[i]%2)>atleast_m and ms1_peak[i] and top_ten[i][0]%2==1 and np.sum(top_ten[i][:3]%2==1)>=2]
    # ref_peaks_in_dia = [i for i in range(len(candidate_peaks)) if  np.sum(top_ten[i]%2)>atleast_m and ms1_peak[i]]
    
    # print(len(ref_peaks_in_dia))
    
    # filter database further to those that match the required num peaks
    ref_pep_cand_loc = [ref_coords[i] for i in ref_peaks_in_dia]
    ref_pep_cand_list = [candidate_peaks[i] for i in ref_peaks_in_dia]
    # ref_pep_cand = [candidate_lib[i]["seq"] for i in ref_peaks_in_dia] # Nb this is modified seq!!
    ref_pep_cand = [mass_window_candidates[i] for i in ref_peaks_in_dia] # Nb this is modified seq!!
    
    norm_intensities = [M[:,1]/sum(M[:,1]) for M in ref_pep_cand_list]


    ########## Update
    # lib peaks that match
    lib_peaks_matched = [j%2==1 for j in ref_pep_cand_loc]
    
    # name these something different so can be accessed later
    ref_spec_row_indices_split = [np.int32(((i[j]+1)/2)-1) for i,j in zip(ref_pep_cand_loc,lib_peaks_matched)] # NB these are floats
    num_lib_peaks_matched = np.array([np.sum(i) for i in lib_peaks_matched]) #f1
    ref_spec_col_indices_split = [np.array([idx]*i) for idx,i in zip(range(len(ref_pep_cand)),num_lib_peaks_matched)] 
    ref_spec_values_split = [ints[i] for ints,i in zip(norm_intensities,lib_peaks_matched)]
    
    
    
    # ref_spec_row_indices = ((np.array([j for i in ref_pep_cand_loc for j in i if j%2==1])+1)/2)-1 # NB these are floats
    # ref_spec_col_indices = np.array([i for idx in range(len(ref_pep_cand)) for i in [idx]*len([loc for loc in ref_pep_cand_loc[idx] if loc%2==1])])
    # ref_spec_values = np.array([norm_intensities[idx][peak_idx] for idx in range(len(ref_pep_cand)) for peak_idx in range(len(norm_intensities[idx])) if ref_pep_cand_loc[idx][peak_idx]%2==1])
    
    frag_errors = []
    frag_mz = []
    
    
    if len(ref_spec_row_indices_split)>0 and len(ref_spec_col_indices_split)>0 and len(ref_spec_values_split)>0:
        
        #### concatenate the matrix values
        ref_spec_row_indices = np.concatenate(ref_spec_row_indices_split)
        ref_spec_col_indices = np.concatenate(ref_spec_col_indices_split)
        ref_spec_values = np.concatenate(ref_spec_values_split)
        
        frag_errors = [np.array(bin_centers[ref_spec_row_indices_split[i]]-ref_pep_cand_list[i][:,0][lib_peaks_matched[i]])/bin_centers[ref_spec_row_indices_split[i]] for i in range(len(lib_peaks_matched))]
        lib_frag_mz = [ref_pep_cand_list[i][:,0][lib_peaks_matched[i]] for i in range(len(lib_peaks_matched))]
        lib_frag_int = [ref_pep_cand_list[i][:,1][lib_peaks_matched[i]] for i in range(len(lib_peaks_matched))]
        obs_frag_int = [dia_spectrum[ref_spec_row_indices_split[i],1] for i in range(len(ref_spec_row_indices_split))]
        frag_names = [library[i]["ordered_frags"][j] for i,j in zip(ref_pep_cand,lib_peaks_matched)]
        
        frag_mz = [ref_pep_cand_list[i][:,0][lib_peaks_matched[i]] for i in range(len(lib_peaks_matched))]
        
        # what peaks from the spectrum are matched by library peps
        unique_row_idxs = [int(i) for i in set(ref_spec_row_indices)]
        unique_row_idxs.sort()
        
        dia_spec_int = dia_spectrum[unique_row_idxs,1]
        
        # add another term to penalise additional lib peaks
        dia_spec_int = np.append(dia_spec_int,[0]) 
        # find peaks that are bot matched in dia spectrum
        ref_peaks_not_in_dia = np.array([idx for loc_list in ref_pep_cand_loc for idx in range(len(loc_list)) if loc_list[idx]%2==0])
        # get col indices (will just be one for each)
        not_dia_col_indices = np.arange(len(ref_pep_cand))
        num_rows = max(unique_row_idxs)
        # row indices always the last row (num peaks+1)
        not_dia_row_indices = [num_rows+1]*len(not_dia_col_indices)
        # sum peak intensities not in dia spectrum
        not_dia_values = np.array([np.sum([norm_intensities[idx][peak_idx] for peak_idx in range(len(norm_intensities[idx])) if ref_pep_cand_loc[idx][peak_idx]%2==0])
                                  for idx in range(len(norm_intensities))])
    
        sparse_row_indices = np.append(ref_spec_row_indices,not_dia_row_indices)
        sparse_col_indices = np.append(ref_spec_col_indices,not_dia_col_indices)
        sparse_values = np.append(ref_spec_values,not_dia_values)
        
        # some dia peaks are not matched and are therefore ignored
        # below ranks the rows by number therefore removing missing rows
        sparse_row_indices = stats.rankdata(sparse_row_indices,method="dense").astype(int)-1
        
        # Generate sparse matrix from data
        sparse_lib_matrix = sparse.coo_matrix((sparse_values,(sparse_row_indices,sparse_col_indices)))
        # print("Starting Fit")
        # Fit lib spectra to observed spectra
        fit_results = sparse_nnls.lsqnonneg(sparse_lib_matrix,dia_spec_int,{"show_progress":False})
        lib_coefficients = fit_results['x']
        
        
        ####################################
        ### features 
        frac_lib_intensity = [np.sum(i) for i in ref_spec_values_split] # all ints sum to 1 so these give frac
        tic = np.sum(dia_spectrum[:,1])
        frac_dia_intensity = [np.sum(dia_spectrum[i,1])/tic for i in ref_spec_row_indices_split]
        # mz tol
        if dino_features is not None:
            # Recalculate filtered_dino for feature calculation
            filtered_dino = feature_list_mz(feature_list_rt(dino_features, prec_rt, rt_tol=rt_tol),
                                          prec_mz, windowWidth)
            rel_error = ms1_error(np.array(filtered_dino.mz), rt_mz[window_idxs[ref_peaks_in_dia],1], tol=ms1_tol)
        else:
            rel_error = np.zeros(len(ref_peaks_in_dia))
        rt_error = prec_rt-rt_mz[window_idxs[ref_peaks_in_dia],0]
        
        frac_int_matched = np.sum(dia_spec_int)/np.sum(dia_spectrum[:,1])
        predicted_spec = np.squeeze(sparse_lib_matrix*lib_coefficients)[:-1]
        # print(len(dia_spec_int),len(predicted_spec))
        r2all = np_pearson_cor(dia_spec_int[:-1],predicted_spec).statistic
        
        r2_lib_spec = [np_pearson_cor(i,dia_spectrum[j,1]).statistic for i,j in zip(ref_spec_values_split,ref_spec_row_indices_split)]
        
        single_matched_rows = np.where(np.sum(sparse_lib_matrix>0,1)==1)[0]
        peaks_not_shared = [
            np.array([[dia_spectrum[i,1],j] for i,j in zip(dia,lib) if i in single_matched_rows]) for dia,lib in zip(ref_spec_row_indices_split,ref_spec_values_split)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r2_unique = [np_pearson_cor(*i.T).statistic if i.shape[0]>1 else 0 for i in peaks_not_shared ]
            
        frac_unique_pred = [np.divide(*np.sum(i,axis=0)[::-1])*c if i.shape[0]>0 else 0 for i,c in zip(peaks_not_shared,lib_coefficients)] #frac of int matched by unique peaks pred by unique peaks
        
        frac_dia_intensity_pred = [(i*c)/j for i,j,c in zip(frac_lib_intensity,frac_dia_intensity,lib_coefficients)]
        
        #### stack spectrum features
        r2all = np.ones_like(num_lib_peaks_matched)*r2all
        frac_int_matched = np.ones_like(num_lib_peaks_matched)*frac_int_matched
        frac_int_pred = (np.ones_like(num_lib_peaks_matched)*np.sum(predicted_spec))/tic
        frac_int_matched_pred = (np.ones_like(num_lib_peaks_matched)*np.sum(predicted_spec))/np.sum(dia_spec_int)
        large_coeff_indices = np.where(np.array(lib_coefficients)>1)[0] # identify large coeffs
        large_coeff_matched_peaks = np.unique(np.concatenate(([ref_spec_row_indices_split[i] for i in large_coeff_indices]))) # select the peaks matched to these
        large_coeff_int_pred = np.sum([np.sum(ref_spec_values_split[i])*list(lib_coefficients)[i] for i in large_coeff_indices]) # sum the intensity predicted
        large_coeff_int_matched = np.sum(dia_spectrum[large_coeff_matched_peaks,1]) # sum the intensity matched
        ## Note: some predictions over-shoot the matched peak so we overestimate this value
        ## Q: Should we report different values for coeffs < 1??
        frac_int_matched_pred_sigcoeff = (np.ones_like(num_lib_peaks_matched)*large_coeff_int_pred)/large_coeff_int_matched # create vals for all peaks
        
                
        subset_row_indices = np.unique(sparse_row_indices[np.where(np.isin(sparse_col_indices,large_coeff_indices))])
        subset_row_indices = np.delete(subset_row_indices,np.where(subset_row_indices==max(subset_row_indices))[0][0])
        large_coeffs = np.squeeze(lib_coefficients) # get the coeffs
        large_coeffs[large_coeffs<1] = 0 # set those <1 to 0
        scaled_matrix = np.multiply(sparse_lib_matrix.toarray(),large_coeffs)#scale the matrix
        subset_pred_spec = np.sum(scaled_matrix,1)
        subset_cosine = cosim(dia_spec_int[subset_row_indices],subset_pred_spec[subset_row_indices])
        large_coeff_cosine = np.ones_like(num_lib_peaks_matched)*subset_cosine
        hyperscores, b_counts, y_counts = map(list, zip(*[hyperscore_b_y(library[i]["frags"],j) for i,j in zip(ref_pep_cand,lib_peaks_matched)]))
        longest_y_ions = [longest_y(library[i]["frags"],j) for i,j in zip(ref_pep_cand,lib_peaks_matched)]

        scribe_scores = get_scribe(
            ref_spec_row_indices_split,
            ref_spec_col_indices_split,
            ref_spec_values_split,
            dia_spectrum[:,1]
        )
    
        residuals, y_pred = get_residuals(
            ref_spec_values_split,
            ref_spec_row_indices_split,
            ref_spec_col_indices_split,
            [],
            [],
            [],
            dia_spectrum[:,1],
            lib_coefficients,
            0,
            0
        )
        # Then use y_pred for the manhattan distance
        manhattan_distances, fitted_spectral_contrasts = get_manhattan_distance(
            ref_spec_row_indices_split,
            ref_spec_col_indices_split,
            ref_spec_values_split,
            dia_spectrum[:,1],
            y_pred
        )

        gof_stats, max_unmatched_residuals, max_matched_residuals = gof_stat(
            ref_spec_row_indices_split,
            ref_spec_col_indices_split,
            ref_spec_values_split,
            residuals,
            dia_spectrum[:,1],
            lib_coefficients,
            0
        )

        features = np.stack([num_lib_peaks_matched,
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
                            rt_mz[:,1][window_idxs[ref_peaks_in_dia]]
                                ],-1)
        
        
        ####################################
            
    #Select non-zero coeffs
    # Note: many coeffs are non-zero but essentially zero!! Perhaps set less than 1e-7??
    non_zero_coeffs = [c for c in lib_coefficients if c!=0]
    non_zero_coeffs_idxs = [i for i,c in enumerate(lib_coefficients) if c!=0]
    
    output = [[0,spec_idx,0,0,prec_mz,prec_rt,*np.zeros(len(names)-6)]]
    
    if len(non_zero_coeffs)>0:
        lib_spec_ids = [ref_pep_cand[i] for i in range(len(ref_pep_cand)) if lib_coefficients[i] != 0]
        all_spec_ids = lib_spec_ids
        all_features = features
        all_ms2_frags = [[";".join(map(str,j)) for j in i] for i in zip(frag_names,
                                                                        frag_errors,
                                                                        lib_frag_mz,
                                                                        lib_frag_int,
                                                                        obs_frag_int)]
        
        return_prot = config.protein_column in library[next(iter(library))]
        output = [[non_zero_coeffs[i],
                   spec_idx,
                   ms1_spec.scan_num,
                   all_spec_ids[i][0],
                   all_spec_ids[i][1],
                   prec_mz,
                   prec_rt,
                   *all_features[j],
                   *all_ms2_frags[j],
                   config.args.mzml,
                   library[(re.sub("Decoy_","",all_spec_ids[i][0]),all_spec_ids[i][1])][config.protein_column] if return_prot else "NA" ]
                   for i,j in zip(range(len(non_zero_coeffs)),non_zero_coeffs_idxs)]
        
        # output = [[non_zero_coeffs[i],
        #            spec_idx,
        #            lib_spec_ids[i][0],
        #            lib_spec_ids[i][1],
        #            prec_mz,
        #            prec_rt,
        #            *features[j]] for i,j in zip(range(len(non_zero_coeffs)),non_zero_coeffs_idxs)]
    
    if return_frags:
        return output, [frag_errors,frag_mz]
    else:
        return output


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
    Modern implementation of fit_to_lib2 using unified library structure.
    
    This version uses a single unified library containing both targets and decoys,
    identified by the 'is_decoy' flag in each library entry.
    """
    # 1. Extract spectrum information (same as original)
    spec_idx = dia_spec.scan_num
    dia_spectrum = np.stack(dia_spec.peak_list(), 1)
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
    
    candidate_peaks = [library[i]['spectrum'] for i in mass_window_candidates]
    
    # Early exit if no candidates
    if len(mass_window_candidates) == 0:
        return [create_empty_output_row(
            spec_idx, 
            ms1_spec.scan_num if ms1_spec else 0,
            prec_mz, 
            prec_rt, 
            len(names)
        )]
    
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
        return [create_empty_output_row(
            spec_idx,
            ms1_spec.scan_num if ms1_spec else 0,
            prec_mz,
            prec_rt,
            len(names)
        )]
    
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


