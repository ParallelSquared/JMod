"""
Unified data structures for spectral fitting that combine target and decoy processing.

This module provides data structures and utilities to handle targets and decoys
in a unified manner, eliminating code duplication and improving maintainability.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import numpy as np


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


def split_by_type(unified_data: UnifiedCandidates) -> Tuple[List, List]:
    """
    Split unified candidates back into targets and decoys.
    
    For backward compatibility with existing code.
    
    Args:
        unified_data: UnifiedCandidates object
        
    Returns:
        Tuple of (target_candidates, decoy_candidates)
    """
    target_mask = ~unified_data.is_decoy
    target_indices = np.where(target_mask)[0]
    decoy_indices = np.where(~target_mask)[0]
    
    target_candidates = [unified_data.candidates[i] for i in target_indices]
    decoy_candidates = [unified_data.candidates[i] for i in decoy_indices]
    
    return target_candidates, decoy_candidates


def create_entries_unified(
    centroid_breaks: np.ndarray,
    unified_candidates: UnifiedCandidates,
    top_n: int = 10,
    atleast_m: int = 3,
    prec_mzs: Optional[np.ndarray] = None,
    ms1_spec: Optional[Any] = None,
    ms1_tol: float = 25.,
    spec_frags: Optional[List] = None,
    top_n_idxs: Optional[List[np.ndarray]] = None,
    frac_matched: float = 0.25
) -> Tuple[UnifiedCandidates, UnifiedMatrixData, Dict[str, Any]]:
    """
    Unified version of create_entries that processes targets and decoys together.
    
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
        
    Returns:
        Tuple of:
        - Updated UnifiedCandidates with peaks_in_dia filled
        - UnifiedMatrixData with sparse matrix data
        - Dictionary with additional outputs (lib_peaks_matched, norm_intensities, etc.)
    """
    # Imports needed for processing
    import sys
    import os
    # Add parent directory to path for imports
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from src.utils.misc_functions import closest_peak_diff
    
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
    
    # MS1 filtering
    ms1_peak = np.ones(n_candidates, dtype=bool)  # Default to True
    if ms1_spec is not None and prec_mzs is not None:
        ms1_diffs = [closest_peak_diff(mz, ms1_spec.mz) for mz in prec_mzs]
        ms1_peak = ~np.isnan(ms1_diffs)
        ms1_error = np.array([diff / prec_mzs[i] * 1e6 if not np.isnan(diff) else np.nan 
                              for i, diff in enumerate(ms1_diffs)])
    else:
        ms1_error = np.zeros(n_candidates)
    
    # Normalize intensities
    all_norm_intensities = [M[:, 1] / np.sum(M[:, 1]) for M in candidate_peaks]
    
    # Find candidates with peaks in DIA
    peaks_in_dia = [
        i for i in range(n_candidates) 
        if (np.sum(all_norm_intensities[i][(ref_coords[i] % 2) == 1]) > frac_matched and
            np.sum(top_ten[i] % 2) > atleast_m and
            ms1_peak[i] and
            len(top_ten[i]) > 0 and top_ten[i][0] % 2 == 1 and
            np.sum(top_ten[i][:3] % 2 == 1) >= 2)
    ]
    
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
    
    # Additional outputs for compatibility
    additional_outputs = {
        'lib_peaks_matched': lib_peaks_matched,
        'norm_intensities': norm_intensities,
        'pep_cand_loc': pep_cand_loc,
        'pep_cand_list': pep_cand_list,
        'pep_cand': pep_cand,
        'ms1_error_matched': ms1_error[peaks_in_dia]
    }
    
    return unified_candidates, matrix_data, additional_outputs