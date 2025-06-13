"""
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
hyperscore_b_y, longest_y, closest_ms1spec, closest_peak_diff, cosim,np_pearson_cor
from .utils.parse_peptides import change_seq, convert_frags
from .models.spec_lib.spec_lib import frag_to_peak
from .utils.spectral_similarity_metrics import (
    get_closest_ms1, get_scribe, get_residuals, max_matched_residual, 
    gof_stat, get_manhattan_distance
)


# ===== UNIFIED DATA STRUCTURES (consolidated from spectral_fitting_unified.py) =====

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
    frac_matched: float = 0.25,
    library: Optional[Dict] = None,
    decoy_library: Optional[Dict] = None,
    bin_centers: Optional[np.ndarray] = None,
    dia_spectrum: Optional[np.ndarray] = None
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
    
    # Calculate fragment data if needed
    frag_names = []
    frag_errors = []
    lib_frag_mz = []
    lib_frag_int = []
    obs_frag_int = []
    
    if library is not None and bin_centers is not None and dia_spectrum is not None:
        for i, cand in enumerate(pep_cand):
            if len(spec_row_indices_split[i]) > 0:
                # Get library entry - handle both target and decoy
                if cand[0].startswith("Decoy_") and decoy_library is not None:
                    # For decoys, look up in decoy library with original sequence
                    lib_entry = decoy_library.get(cand, {})
                else:
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


# ===== UNIFIED FEATURE AND MATRIX FUNCTIONS (to be consolidated next) =====
# Temporarily keeping these imports until we consolidate the functions
from .spectral_fitting_unified_matrix import (
    process_matrix_unified, build_sparse_matrix_unified
)
from .spectral_fitting_unified_features import calculate_features_unified

def hyperscore2(frags,frag_names_matched):
    
    num_b = sum(["b" in i for i in frag_names_matched if "iso" not in i])
    num_y = sum(["y" in i for i in frag_names_matched if "iso" not in i])
    dp = np.sum([frags[i] for i in frag_names_matched if "iso" not in i])
    return max(0,np.log(dp*np.math.factorial(num_b)*np.math.factorial(num_y))), num_b, num_y
    
#@profile
def get_features(
    rt_mz,
    ref_spec_values_split,
    ref_spec_row_indices_split,
    ref_spec_col_indices_split,
    decoy_spec_values_split,
    decoy_spec_row_indices_split,
    decoy_spec_col_indices_split,
    ref_peaks_in_dia,
    dia_spectrum,
    prec_rt,
    window_idxs,
    dia_spec_int,
    lib_coefficients,
    sparse_lib_matrix,
    sparse_row_indices,
    sparse_col_indices,
    lib_peaks_matched,
    ref_pep_cand,
    all_row_indices,
    all_values,
    prec_frags,
    ms1_error,
    ref_spec_offset,
    decoy_spec_offset,
    ordered_frags=None):
    
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
        decoy_spec_values_split,
        decoy_spec_row_indices_split,
        decoy_spec_col_indices_split,
        dia_spectrum[:,1],
        lib_coefficients,
        ref_spec_offset,
        decoy_spec_offset
    )
    gof_stats, max_unmatched_residuals, max_matched_residuals = gof_stat(
        ref_spec_row_indices_split,
        ref_spec_col_indices_split,
        ref_spec_values_split,
        residuals,
        dia_spectrum[:,1],
        lib_coefficients,
        ref_spec_offset
    )
    # Add our new function call
    manhattan_distances, fitted_spectral_contrasts = get_manhattan_distance(
        ref_spec_row_indices_split,
        ref_spec_col_indices_split,
        ref_spec_values_split,
        dia_spectrum[:,1],
        y_pred
    )
    ### features 
    num_lib_peaks_matched = np.array([np.sum(i) for i in lib_peaks_matched])
    frac_lib_intensity = [np.sum(i) for i in ref_spec_values_split] # all ints sum to 1 so these give frac
    tic = np.sum(dia_spectrum[:,1])
    frac_dia_intensity = [np.sum(dia_spectrum[i,1])/tic for i in ref_spec_row_indices_split]
    # mz tol
    rel_error = ms1_error#np.zeros(len(ref_peaks_in_dia))
    rt_error = prec_rt-rt_mz[:,0]
    
    frac_int_matched = np.sum(dia_spec_int)/np.sum(dia_spectrum[:,1])
    predicted_spec = np.squeeze(sparse_lib_matrix*lib_coefficients)[:-1]
    
    # r2all = np_pearson_cor(dia_spec_int[:-1],predicted_spec).statistic
    # r2_lib_spec = [np_pearson_cor(i,dia_spectrum[j,1]).statistic for i,j in zip(ref_spec_values_split,ref_spec_row_indices_split)]
    r2all = np.zeros_like(rt_error)
    r2_lib_spec = np.zeros_like(rt_error)
    
    single_matched_rows = np.where(np.sum(sparse_lib_matrix>0,1)==1)[0]
    peaks_not_shared = [np.array([[dia_spectrum[i,1],j] for i,j in zip(dia,lib) if i in single_matched_rows]) for dia,lib in zip(ref_spec_row_indices_split,ref_spec_values_split)]
    # with warnings.catch_warnings():
    #     warnings.simplefilter("ignore")
    #     r2_unique = [np_pearson_cor(*i.T).statistic if i.shape[0]>1 else 0 for i in peaks_not_shared ]
    r2_unique = np.zeros_like(rt_error)
        
    frac_unique_pred = [np.divide(*np.sum(i,axis=0)[::-1])*c if i.shape[0]>0 else 0 for i,c in zip(peaks_not_shared,lib_coefficients[ref_spec_offset:])] #frac of int matched by unique peaks pred by unique peaks

    frac_dia_intensity_pred = [(i*c)/j for i,j,c in zip(frac_lib_intensity,frac_dia_intensity,lib_coefficients[ref_spec_offset:])]
    
    #### stack spectrum features
    # r2all = np.ones_like(num_lib_peaks_matched)*r2all
    frac_int_matched = np.ones_like(num_lib_peaks_matched)*frac_int_matched
    frac_int_pred = (np.ones_like(num_lib_peaks_matched)*np.sum(predicted_spec))/tic
    frac_int_matched_pred = (np.ones_like(num_lib_peaks_matched)*np.sum(predicted_spec))/np.sum(dia_spec_int)
    large_coeff_indices = np.where(np.array(lib_coefficients)>1)[0] # identify large coeffs
    large_coeff_matched_peaks = np.unique(np.concatenate(([all_row_indices[i] for i in large_coeff_indices]))) # select the peaks matched to these
    large_coeff_int_pred = np.sum([np.sum(all_values[i])*list(lib_coefficients)[i] for i in large_coeff_indices]) # sum the intensity predicted
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
    
    if len(prec_frags)>0 and len(list(prec_frags)[0])==len(lib_peaks_matched[0]):
        hyperscores, b_counts, y_counts = map(list, zip(*[hyperscore_b_y(frags,j) for frags,j in zip(prec_frags,lib_peaks_matched)]))
        longest_y_ions = [longest_y(frags,j) for frags,j in zip(prec_frags,lib_peaks_matched)]
    elif len(prec_frags)>0 and ordered_frags is not None:
        hyperscores, b_counts, y_counts  = map(list, zip(*[hyperscore2(frags,frag_names) for frags,frag_names in zip(prec_frags,ordered_frags)]))
        longest_y_ions = [max([int(re.match("[by](\d+)",i)[1]) for i in frag_names])  for frag_names in ordered_frags]
    else:
        hyperscores, b_counts, y_counts = np.zeros_like(num_lib_peaks_matched), np.zeros_like(num_lib_peaks_matched), np.zeros_like(num_lib_peaks_matched)
        longest_y_ions = np.zeros_like(num_lib_peaks_matched)
    
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
                          rt_mz[:,1],
                          # peaks
                            ],-1)
    return features


#@profile
def unmatched_peaks(norm_intensities,
                    pep_cand_loc,
                    last_row,
                    fit_type="a",
                    lower_limit = 1e-10):
    """
    3 fit_types:
        a: All summed unmatched intensities are fit to a single zero intensity "obs peak"
        b: Summed unmatched intensities of each precursor are fit to their own zero intensity "obs peak"
        c: Each unmatched peak is fit to its own zero intensity "obs peak"
    
    lower_limit:
        if normalized fragment intensity is below this threshold, exclude from fit (default essentially includes all peaks)
        Only applicable to type c
        
    """
    assert fit_type in ["a","b","c"]
    
    if fit_type=="a":
        # get col indices (will just be one for each)
        not_dia_col_indices = np.arange(len(pep_cand_loc))
        # row indices always the last row (num peaks+1)
        not_dia_row_indices = np.array([last_row]*len(not_dia_col_indices),dtype=int)
        # sum peak intensities not in dia spectrum
        not_dia_values = np.array([np.sum([norm_intensities[idx][peak_idx] for peak_idx in range(len(norm_intensities[idx])) if pep_cand_loc[idx][peak_idx]%2==0])
                                  for idx in range(len(norm_intensities))])
        
    elif fit_type=="b":
        # get col indices (will just be one for each)
        not_dia_col_indices = np.arange(len(pep_cand_loc))
        # row indices always one for each precursor after last_row
        not_dia_row_indices = [last_row+1]*len(not_dia_col_indices)+not_dia_col_indices
        not_dia_values = np.array([np.sum([norm_intensities[idx][peak_idx] for peak_idx in range(len(norm_intensities[idx])) if pep_cand_loc[idx][peak_idx]%2==0])
                                  for idx in range(len(norm_intensities))])
        
    elif fit_type=="c":
        ## each value is kept separate
        all_unmatched_peaks = [[norm_intensities[idx][peak_idx] for peak_idx in range(len(norm_intensities[idx])) if pep_cand_loc[idx][peak_idx]%2==0 and norm_intensities[idx][peak_idx]>lower_limit]
                                  for idx in range(len(norm_intensities))]
        num_unmatched_to_fit = [len(i) for i in all_unmatched_peaks]
        not_dia_col_indices = np.array(np.concatenate([[idx]*i for idx,i in enumerate(num_unmatched_to_fit)]),dtype=int)
        not_dia_row_indices = np.array(np.arange(np.sum(num_unmatched_to_fit))+last_row+1,dtype=int)
        not_dia_values = np.concatenate(all_unmatched_peaks)
    
    return not_dia_row_indices, not_dia_col_indices, not_dia_values



#@profile
def create_entries(centroid_breaks,
                   candidate_peaks,
                   mass_window_candidates,
                   top_n,atleast_m,
                   prec_mzs,
                   ms1_spec,
                   ms1_tol,
                   spec_frags=None,
                   top_n_idxs=None
                   ):
    
    coords = [np.searchsorted(centroid_breaks,M[:,0]) for M in candidate_peaks]
    
    # if spec_frags:
        
        
    #     spec_ref_coords = [np.searchsorted(centroid_breaks,M[:,0]) for M in spec_frags]
    #     top_ten = [np.searchsorted(centroid_breaks,M[np.argsort(-M[:,1])[0:min(top_n,M.shape[0])],0]) for M in spec_frags]
       
    #     ms1_error = np.array([closest_peak_diff(mz,ms1_spec.mz,max_diff=ms1_tol) for mz in prec_mzs])
    #     ms1_peak = ~np.isnan(ms1_error)
        
    #     all_norm_intensities = [M[:,1]/sum(M[:,1]) for M in spec_frags]
    #     peaks_in_dia = [i for i in range(len(spec_frags)) if np.sum(all_norm_intensities[i][(spec_ref_coords[i]%2)==1])>0.5 and np.sum(top_ten[i]%2)>atleast_m and ms1_peak[i] and top_ten[i][0]%2==1 and np.sum(top_ten[i][:3]%2==1)>=2]
        
    
    # else:
    
    # top_ten = [np.searchsorted(centroid_breaks,M[np.argsort(-M[:,1])[0:min(top_n,M.shape[0])],0]) for M in candidate_peaks]
    top_ten = [c[idxs] for c,idxs in zip(coords,top_n_idxs)]
    # peaks_in_dia = [i for i in range(len(candidate_peaks)) if len([a for a in top_ten_decoy[i] if a%2 ==1])>atleast_m]
    all_norm_intensities = [M[:,1]/(M[:,1]).sum() for M in candidate_peaks]
    # all_norm_intensities = [M[:,1]/sum(M[:,1]) for M in candidate_peaks]
    ms1_error = np.array([closest_peak_diff(mz,ms1_spec.mz,max_diff=ms1_tol) for mz in prec_mzs])
    ms1_peak = ~np.isnan(ms1_error)
    
    # peaks_in_dia = [i for i in range(len(candidate_peaks)) if np.sum(all_norm_intensities[i][(coords[i]%2)==1])>0.5 and np.sum(top_ten[i]%2)>atleast_m and ms1_peak[i] and top_ten[i][0]%2==1 and np.sum(top_ten[i][:3]%2==1)>=2]
    # peaks_in_dia = [i for i in range(len(candidate_peaks)) if np.sum(top_ten[i]%2)>atleast_m and ms1_peak[i]]
    # peaks_in_dia = [i for i in range(len(candidate_peaks)) if np.sum(all_norm_intensities[i][(coords[i]%2)==1])>0.5 and np.sum(top_ten[i]%2)>atleast_m and ms1_peak[i]]
    if config.match_ms1:
        peaks_in_dia = [i for i in range(len(candidate_peaks)) if (all_norm_intensities[i][(coords[i]%2)==1]).sum()>config.frac_lib_matched and (top_ten[i]%2).sum()>atleast_m and ms1_peak[i]]
    else:
       peaks_in_dia = [i for i in range(len(candidate_peaks)) if (all_norm_intensities[i][(coords[i]%2)==1]).sum()>config.frac_lib_matched and (top_ten[i]%2).sum()>atleast_m]
    
    pep_cand_loc = [coords[i] for i in peaks_in_dia]
    pep_cand_list = [candidate_peaks[i] for i in peaks_in_dia]
    pep_cand = [mass_window_candidates[i] for i in peaks_in_dia] # Nb this is modified seq!!
    
    norm_intensities = [M[:,1]/sum(M[:,1]) for M in pep_cand_list]
    lib_peaks_matched = [j%2==1 for j in pep_cand_loc]
    
    row_indices_split = [np.int32(((i[j]+1)/2)-1) for i,j in zip(pep_cand_loc,lib_peaks_matched)] # NB these are floats
    num_peaks_matched = np.array([np.sum(i) for i in lib_peaks_matched]) #f1
    col_indices_split = [np.array([idx]*i,dtype=int) for idx,i in zip(range(len(pep_cand)),num_peaks_matched)] 
    values_split = [ints[i] for ints,i in zip(norm_intensities,lib_peaks_matched)]
    
    return (peaks_in_dia,
            pep_cand,
            pep_cand_loc,
            pep_cand_list,
            row_indices_split,col_indices_split,values_split, norm_intensities, lib_peaks_matched, ms1_error[peaks_in_dia])


#@profile
def fit_to_lib2_original(dia_spec,
                library,
                rt_mz,
                all_keys,
                dino_features=None,rt_filter=False,ms1_mz=None,
               ms1_spectra = None,
               rt_tol = config.rt_tol,
               ms1_tol = config.ms1_tol,
               mz_tol = config.mz_tol,
               return_frags = False,
               decoy=False,
               decoy_library=None):
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
        ms1_spec = get_closest_ms1(prec_rt,ms1_spectra)
    
    
    lib_coefficients = []
   
    if ms1_mz:
        _bool = (np.abs(rt_mz[:,1]-ms1_mz)/ms1_mz)<ms1_tol
        
    else:
        if rt_filter:
            _bool = np.logical_and(np.abs(rt_mz[:,1]-prec_mz)<(windowWidth/2),np.abs(rt_mz[:,0]-prec_rt)<rt_tol)
        else:
            _bool = np.abs(rt_mz[:,1]-prec_mz)<(windowWidth/2)
            
    window_idxs = np.where(_bool)[0]        
        
    
    mass_window_candidates = [all_keys[i] for i in window_idxs] 
    candidate_peaks = [library[i]['spectrum'] for i in mass_window_candidates]
    
    
    ###### Process dia spectrum 
    
    # what are the first indices of peaks grouped by tolerance
    merged_coords_idxs = np.searchsorted(dia_spectrum[:,0]+mz_tol*dia_spectrum[:,0],dia_spectrum[:,0])
    
    # what are the first mz of these peak groups
    merged_coords = dia_spectrum[np.unique(merged_coords_idxs),0]
    # print(merged_coords)
    
    
    # NB - should we not sum the intensities?????
    # merged_intensities = [np.mean(dia_spectrum[np.where(merged_coords_idxs==i)[0],1]) for i in np.unique(merged_coords_idxs)]
    merged_intensities = np.zeros(len((merged_coords_idxs)))
    for j,val in zip(merged_coords_idxs,dia_spectrum[:,1]):
        merged_intensities[j]+=val
    #merged_intensities = [np.mean(dia_spectrum[merged_coords_idxs==i,1]) for i in np.unique(merged_coords_idxs)]
    merged_intensities = merged_intensities[merged_intensities!=0]
    
    #update spectrum to new values (note mz remains first in group as this will eventually be rounded)
    dia_spectrum = np.array((merged_coords,merged_intensities)).transpose()
    # print(dia_spectrum)
    
    #get window edge positions each side of peaks in observed spectra (NB the tolerance is now about the first peak in the group not the middile)
    centroid_breaks = np.concatenate((dia_spectrum[:,0]-mz_tol*dia_spectrum[:,0],dia_spectrum[:,0]+mz_tol*dia_spectrum[:,0]))
    centroid_breaks = np.sort(centroid_breaks)
    bin_centers = np.mean(np.stack((centroid_breaks[::2],centroid_breaks[1::2]),1),1)
    
    # lib_idx=0
    # M = candidate_peaks[lib_idx]
    # ref_coords = [np.searchsorted(centroid_breaks,M[:,0]) for M in candidate_peaks]
    # top_ten = [np.searchsorted(centroid_breaks,M[np.argsort(-M[:,1])[0:min(top_n,M.shape[0])],0]) for M in candidate_peaks]
    
    ## Filter precursors based on resp. MS1 peak
    # ms1_peak = ~np.isnan([closest_peak_diff(mz,ms1_spec.mz) for mz in rt_mz[window_idxs,1]])
    
    # does the top ten peaks fall between centroid breaks? i.e. odd numbers (%2==1), 
    # ref_peaks_in_dia = [i for i in range(len(candidate_peaks)) if len([a for a in top_ten[i] if a%2 ==1])>atleast_m]
    # prop_ref_peaks_in_dia = [len([a for a in top_ten[i] if a%2 ==1])/candidate_peaks[i].shape[0] for i in range(len(candidate_peaks))]
    
    # all_norm_intensities = [M[:,1]/sum(M[:,1]) for M in candidate_peaks]
    # ref_peaks_in_dia = [i for i in range(len(candidate_peaks)) if np.sum(all_norm_intensities[i][(ref_coords[i]%2)==1])>0.5 and np.sum(top_ten[i]%2)>atleast_m and ms1_peak[i]]
    # ref_peaks_in_dia = [i for i in range(len(candidate_peaks)) if  np.sum(top_ten[i]%2)>atleast_m and ms1_peak[i]]
    
    # print(len(ref_peaks_in_dia))
    
    # filter database further to those that match the required num peaks
    # ref_pep_cand_loc = [ref_coords[i] for i in ref_peaks_in_dia]
    # ref_pep_cand_list = [candidate_peaks[i] for i in ref_peaks_in_dia]
    # # ref_pep_cand = [candidate_lib[i]["seq"] for i in ref_peaks_in_dia] # Nb this is modified seq!!
    # ref_pep_cand = [mass_window_candidates[i] for i in ref_peaks_in_dia] # Nb this is modified seq!!
    
    # norm_intensities = [M[:,1]/sum(M[:,1]) for M in ref_pep_cand_list]


    ########## Update
    # lib peaks that match
    # lib_peaks_matched = [j%2==1 for j in ref_pep_cand_loc]
    
    # # name these something different so can be accessed later
    # ref_spec_row_indices_split = [np.int32(((i[j]+1)/2)-1) for i,j in zip(ref_pep_cand_loc,lib_peaks_matched)] # NB these are floats
    # num_lib_peaks_matched = np.array([np.sum(i) for i in lib_peaks_matched]) #f1
    # ref_spec_col_indices_split = [np.array([idx]*i) for idx,i in zip(range(len(ref_pep_cand)),num_lib_peaks_matched)] 
    # ref_spec_values_split = [ints[i] for ints,i in zip(norm_intensities,lib_peaks_matched)]
    
    
    top_n_idxs = [library[i]['top_n'] for i in mass_window_candidates]
    
    
    spec_frags = None
    if "spec_frags" in library[all_keys[0]].keys():
        spec_frags = [library[i]['spec_frags'] for i in mass_window_candidates]
        
    ref_peaks_in_dia,\
    ref_pep_cand,\
    ref_pep_cand_loc,\
    ref_pep_cand_list,\
    ref_spec_row_indices_split,\
        ref_spec_col_indices_split,\
        ref_spec_values_split, \
        norm_intensities, \
        lib_peaks_matched, \
        ref_ms1_error = create_entries(centroid_breaks=centroid_breaks, 
                                        candidate_peaks=candidate_peaks, 
                                        mass_window_candidates=mass_window_candidates, 
                                        top_n=top_n, 
                                        atleast_m=atleast_m, 
                                        prec_mzs=rt_mz[:,1][window_idxs], 
                                        ms1_spec=ms1_spec,
                                        ms1_tol=ms1_tol,
                                        top_n_idxs=top_n_idxs)

    
    ### Generate eqivalent Decoy spectra
    if decoy:
        mass_window_decoy_candidates = [("Decoy_"+i[0],*i[1:]) for i in mass_window_candidates] 
        # print("old")
        # converted_seqs = [change_seq(i[0],config.args.decoy) for i in mass_window_candidates]
        # decoy_mz = np.array([convert_prec_mz(i, z=j[1]) for i,j in zip(converted_seqs, mass_window_candidates)])
        # if config.args.decoy=="rev": ## this will have the same mz as many correct mathces and therefore a really good ms1 isotope corr
        #     decoy_mz -= config.decoy_mz_offset
        # ## NB: Below needs to change to ibcorporate iso frags!!
        # converted_frags = [convert_frags(i[0], library[i]["frags"],config.args.decoy) for i in mass_window_candidates]
        # decoy_sorted_frags = [sorted(converted_frags[i],key = lambda x: converted_frags[i][x][0]) for i in range(len(converted_frags))]
        # if config.args.iso:
        #     candidate_decoy_peaks = [gen_isotopes(i,j) for i,j in zip(converted_seqs,converted_frags)]
        # else:
        #     candidate_decoy_peaks = [frag_to_peak(i) for i in converted_frags]
        
        # ## if using decoy_library
        # print("new")
        converted_frags = [decoy_library[i]["frags"] for i in mass_window_candidates]
        decoy_sorted_frags = [decoy_library[i]["ordered_frags"] for i in mass_window_candidates]
        candidate_decoy_peaks = [decoy_library[i]["spectrum"] for i in mass_window_candidates]
        # decoy_mz = np.array([decoy_library[i]["prec_mz"] for i in mass_window_candidates])
        decoy_mz = rt_mz[:,1][window_idxs] - config.decoy_mz_offset
    
        decoy_top_n_idxs = [decoy_library[i]['top_n'] for i in mass_window_candidates]
        
        decoy_spec_frags = None
        # if "spec_frags" in library[all_keys[0]].keys():
        #     decoy_spec_frags = [specific_frags(i) for i in converted_frags]
        
        # ## Decoy equiv
        # decoy_coords = [np.searchsorted(centroid_breaks,M[:,0]) for M in candidate_decoy_peaks]
        # top_ten_decoy = [np.searchsorted(centroid_breaks,M[np.argsort(-M[:,1])[0:min(top_n,M.shape[0])],0]) for M in candidate_decoy_peaks]
        # # decoy_peaks_in_dia = [i for i in range(len(candidate_decoy_peaks)) if len([a for a in top_ten_decoy[i] if a%2 ==1])>atleast_m]
        # all_norm_decoy_intensities = [M[:,1]/sum(M[:,1]) for M in candidate_decoy_peaks]
        # decoy_ms1_peak = ~np.isnan([closest_peak_diff(mz,ms1_spec.mz) for mz in decoy_mz])
        # # decoy_peaks_in_dia = [i for i in range(len(candidate_peaks)) if np.sum(all_norm_decoy_intensities[i][(decoy_coords[i]%2)==1])>0.5 and np.sum(top_ten_decoy[i]%2)>atleast_m and decoy_ms1_peak[i]]
        # decoy_peaks_in_dia = [i for i in range(len(candidate_peaks)) if np.sum(top_ten_decoy[i]%2)>atleast_m and decoy_ms1_peak[i]]
        
        # decoy_pep_cand_loc = [decoy_coords[i] for i in decoy_peaks_in_dia]
        # decoy_pep_cand_list = [candidate_decoy_peaks[i] for i in decoy_peaks_in_dia]
        # decoy_pep_cand = [mass_window_decoy_candidates[i] for i in decoy_peaks_in_dia] # Nb this is modified seq!!
        
        # norm_decoy_intensities = [M[:,1]/sum(M[:,1]) for M in decoy_pep_cand_list]
        
        # decoy_lib_peaks_matched = [j%2==1 for j in decoy_pep_cand_loc]
        
        # decoy_spec_row_indices_split = [np.int32(((i[j]+1)/2)-1) for i,j in zip(decoy_pep_cand_loc,decoy_lib_peaks_matched)] # NB these are floats
        # num_decoy_peaks_matched = np.array([np.sum(i) for i in decoy_lib_peaks_matched]) #f1
        # decoy_spec_col_indices_split = [np.array([idx]*i,dtype=int) for idx,i in zip(range(len(decoy_pep_cand)),num_decoy_peaks_matched)] 
        # decoy_spec_values_split = [ints[i] for ints,i in zip(norm_decoy_intensities,decoy_lib_peaks_matched)]
        
        
    
        decoy_peaks_in_dia,\
        decoy_pep_cand,\
        decoy_pep_cand_loc,\
        decoy_pep_cand_list,\
        decoy_spec_row_indices_split,\
            decoy_spec_col_indices_split,\
                decoy_spec_values_split, \
                    norm_decoy_intensities, \
                        decoy_lib_peaks_matched, \
                            decoy_ms1_error = create_entries(centroid_breaks=centroid_breaks, 
                                                                candidate_peaks=candidate_decoy_peaks, 
                                                                mass_window_candidates=mass_window_decoy_candidates, 
                                                                top_n=top_n, 
                                                                atleast_m=atleast_m, 
                                                                prec_mzs=decoy_mz, 
                                                                ms1_spec=ms1_spec,
                                                                ms1_tol=ms1_tol,
                                                                spec_frags=decoy_spec_frags,
                                                                top_n_idxs=decoy_top_n_idxs)
       
    frag_errors = []
    lib_frag_mz = []
    decoy_col_offset = 0
    
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
        
        decoy_col_offset = np.max(ref_spec_col_indices)+1 
        
    else:
        ref_spec_row_indices=np.array([],dtype=int)
        ref_spec_col_indices=np.array([],dtype=int)
        ref_spec_values=np.array([],dtype=int)
        frag_errors = []#np.array([],dtype=float)
        lib_frag_mz = []#np.array([],dtype=float)
        lib_frag_int = []
        obs_frag_int = []
        frag_names = []
        
        
    if decoy and len(decoy_spec_row_indices_split)>0:
        decoy_spec_row_indices = np.concatenate(decoy_spec_row_indices_split)
        decoy_spec_col_indices = np.concatenate(decoy_spec_col_indices_split)+decoy_col_offset
        decoy_spec_values = np.concatenate(decoy_spec_values_split)
        decoy_frag_errors = [np.array(bin_centers[decoy_spec_row_indices_split[i]]-decoy_pep_cand_list[i][:,0][decoy_lib_peaks_matched[i]])/bin_centers[decoy_spec_row_indices_split[i]] for i in range(len(decoy_lib_peaks_matched))]
        decoy_lib_frag_mz = [decoy_pep_cand_list[i][:,0][decoy_lib_peaks_matched[i]] for i in range(len(decoy_lib_peaks_matched))]
        decoy_lib_frag_int = [decoy_pep_cand_list[i][:,1][decoy_lib_peaks_matched[i]] for i in range(len(decoy_lib_peaks_matched))]
        decoy_obs_frag_int = [dia_spectrum[decoy_spec_row_indices_split[i],1] for i in range(len(decoy_spec_row_indices_split))]
        decoy_frag_names =  [decoy_sorted_frags[i][decoy_lib_peaks_matched[idx]] for idx,i in enumerate(decoy_peaks_in_dia)]
    else:
        decoy_spec_row_indices_split=[] ## needs to be improved
        decoy_spec_values_split=[] ## needs to be improved
        decoy_spec_row_indices=np.array([],dtype=int)
        decoy_spec_col_indices=np.array([],dtype=int)
        decoy_spec_values=np.array([],dtype=int)
        decoy_frag_errors = []#np.array([],dtype=float)
        decoy_lib_frag_mz = []#np.array([],dtype=float)
        decoy_lib_frag_int = []
        decoy_obs_frag_int = []
        decoy_frag_names = []
        
    if len(decoy_spec_row_indices_split)>0 or len(ref_spec_row_indices_split)>0:
        # what peaks from the spectrum are matched by library peps
        unique_row_idxs = np.unique(np.concatenate((ref_spec_row_indices,decoy_spec_row_indices)))
        unique_row_idxs = np.array(np.sort(unique_row_idxs),dtype=int)
        
        
        # # find peaks that are bot matched in dia spectrum
        # ref_peaks_not_in_dia = np.array([idx for loc_list in ref_pep_cand_loc for idx in range(len(loc_list)) if loc_list[idx]%2==0])
        # # get col indices (will just be one for each)
        # not_dia_col_indices = np.arange(len(ref_pep_cand))
        # num_rows = max(unique_row_idxs)
        # # row indices always the last row (num peaks+1)
        # not_dia_row_indices = [num_rows+1]*len(not_dia_col_indices)
        # # sum peak intensities not in dia spectrum
        # not_dia_values = np.array([np.sum([norm_intensities[idx][peak_idx] for peak_idx in range(len(norm_intensities[idx])) if ref_pep_cand_loc[idx][peak_idx]%2==0])
        #                           for idx in range(len(norm_intensities))])
       
        if len(ref_spec_row_indices_split)>0:
            not_dia_row_indices,not_dia_col_indices,not_dia_values = unmatched_peaks(norm_intensities=norm_intensities,
                                                                                     pep_cand_loc=ref_pep_cand_loc,
                                                                                     last_row=max(unique_row_idxs)+1,
                                                                                     fit_type=config.unmatched_fit_type)
        else:
            not_dia_row_indices=np.array([],dtype=np.int32)
            not_dia_col_indices=np.array([],dtype=np.int32)
            not_dia_values=np.array([],dtype=np.int32)
            
        if decoy and len(decoy_spec_row_indices_split)>0:
            decoy_not_dia_row_indices,decoy_not_dia_col_indices,decoy_not_dia_values = unmatched_peaks(norm_intensities=norm_decoy_intensities,
                                                                                                         pep_cand_loc=decoy_pep_cand_loc,
                                                                                                         last_row=max(not_dia_row_indices,default=max(unique_row_idxs)+1), # if all ref are mathched the initial list is empty
                                                                                                         fit_type=config.unmatched_fit_type)
        else:
            decoy_not_dia_row_indices=np.array([],dtype=np.int32)
            decoy_not_dia_col_indices=np.array([],dtype=np.int32)
            decoy_not_dia_values=np.array([],dtype=np.int32)
            
        ref_sparse_row_indices = np.append(ref_spec_row_indices,not_dia_row_indices)
        ref_sparse_col_indices = np.append(ref_spec_col_indices,not_dia_col_indices)
        ref_sparse_values = np.append(ref_spec_values,not_dia_values)
        
        decoy_sparse_row_indices = np.append(decoy_spec_row_indices,decoy_not_dia_row_indices)
        decoy_sparse_col_indices = np.append(decoy_spec_col_indices,decoy_not_dia_col_indices+decoy_col_offset)
        decoy_sparse_values = np.append(decoy_spec_values,decoy_not_dia_values)
        
        
        sparse_row_indices = np.concatenate((ref_sparse_row_indices,decoy_sparse_row_indices))
        sparse_col_indices = np.concatenate((ref_sparse_col_indices,decoy_sparse_col_indices))
        sparse_values = np.concatenate((ref_sparse_values,decoy_sparse_values))
        
        # some dia peaks are not matched and are therefore ignored
        # below ranks the rows by number therefore removing missing rows
        new_row_indices = stats.rankdata(sparse_row_indices,method="dense").astype(int)-1
        peak_idx_convertor = {i:j for i,j in zip(sparse_row_indices,new_row_indices)}
        sparse_row_indices =new_row_indices
        
        # Generate sparse matrix from data
        sparse_lib_matrix = sparse.coo_matrix((sparse_values,(sparse_row_indices,sparse_col_indices)))
        
        
        dia_spec_int = dia_spectrum[unique_row_idxs,1]
        
        # add another term to penalise additional lib peaks
        dia_spec_int = np.append(dia_spec_int,[0]*(sparse_lib_matrix.shape[0]-dia_spec_int.shape[0])) 
        
        # Fit lib spectra to observed spectra
        fit_results = sparse_nnls.lsqnonneg(sparse_lib_matrix,dia_spec_int,{"show_progress":False})
        lib_coefficients = fit_results['x']
        
        
        ####################################
        features = get_features(rt_mz[window_idxs[ref_peaks_in_dia]],
                                ref_spec_values_split,
                                ref_spec_row_indices_split,
                                ref_spec_col_indices_split,
                                decoy_spec_values_split,
                                decoy_spec_row_indices_split,
                                decoy_spec_col_indices_split,
                                ref_peaks_in_dia,
                                dia_spectrum,
                                prec_rt,
                                window_idxs,
                                dia_spec_int,
                                lib_coefficients,
                                sparse_lib_matrix,
                                sparse_row_indices,
                                sparse_col_indices,
                                lib_peaks_matched,
                                ref_pep_cand,
                                (ref_spec_row_indices_split+decoy_spec_row_indices_split),
                                (ref_spec_values_split+decoy_spec_values_split),
                                [library[i]["frags"] for i in ref_pep_cand],
                                ref_ms1_error,
                                0,
                                decoy_col_offset,
                                frag_names)
        
        single_matched_rows = np.where(np.sum(sparse_lib_matrix>0,1)==1)[0]
        
        # new_row_indices_split = [[peak_idx_convertor[j] for j in i] for i in ref_spec_row_indices_split]
        unique_row_indices_split = [[peak_idx_convertor[j] in single_matched_rows for j in i] for i in ref_spec_row_indices_split]
        unique_frags = [i[j] for i,j in zip(lib_frag_mz,unique_row_indices_split)]
        unique_frags_int = [i[j] for i,j in zip(obs_frag_int,unique_row_indices_split)]
        
        ####################################
        if decoy:
            decoy_features = get_features(np.stack([rt_mz[window_idxs[decoy_peaks_in_dia],0],decoy_mz[decoy_peaks_in_dia]],1),
                                          decoy_spec_values_split,
                                            decoy_spec_row_indices_split,
                                            decoy_spec_col_indices_split,
                                            ref_spec_values_split,
                                            ref_spec_row_indices_split,
                                            ref_spec_col_indices_split,
                                            decoy_peaks_in_dia,
                                            dia_spectrum,
                                            prec_rt,
                                            window_idxs,
                                            dia_spec_int,
                                            lib_coefficients,
                                            sparse_lib_matrix,
                                            sparse_row_indices,
                                            sparse_col_indices,
                                            decoy_lib_peaks_matched,
                                            decoy_pep_cand,
                                            (ref_spec_row_indices_split+decoy_spec_row_indices_split),
                                            (ref_spec_values_split+decoy_spec_values_split),
                                            [converted_frags[i] for i in decoy_peaks_in_dia],
                                            decoy_ms1_error,
                                            decoy_col_offset,
                                            0,
                                            decoy_frag_names)
        
            # new_row_indices_split = [[peak_idx_convertor[j] for j in i] for i in decoy_spec_row_indices_split]
            unique_row_indices_split_decoy = [[peak_idx_convertor[j] in single_matched_rows for j in i] for i in decoy_spec_row_indices_split]
            unique_frags_decoy = [i[j] for i,j in zip(decoy_lib_frag_mz,unique_row_indices_split_decoy)]
            unique_frags_int_decoy = [i[j] for i,j in zip(decoy_obs_frag_int,unique_row_indices_split_decoy)]
                
        ####################################
            
    #Select non-zero coeffs
    # Note: many coeffs are non-zero but essentially zero!! Perhaps set less than 1e-7??
    non_zero_coeffs = [c for c in lib_coefficients if c!=0]
    non_zero_coeffs_idxs = [i for i,c in enumerate(lib_coefficients) if c!=0]
    # print(f"N: {len(lib_coefficients)}, {len(non_zero_coeffs)}")
    if config.args.timeplex:
        output = [[0,spec_idx,ms1_spec.scan_num,0,0,-1,prec_mz,prec_rt,*np.zeros(len(names)-7)]]
    else:
        output = [[0,spec_idx,ms1_spec.scan_num,0,0,prec_mz,prec_rt,*np.zeros(len(names)-7)]]
    
    if len(non_zero_coeffs)>0:
        lib_spec_ids = [ref_pep_cand[i] for i in range(len(ref_pep_cand)) if lib_coefficients[i] != 0]
        if decoy:
            updated_decoy_offset = int(max(ref_sparse_col_indices))+1 if len(ref_sparse_col_indices)>0 else 0
            decoy_spec_ids = [decoy_pep_cand[i] for i in range(len(decoy_pep_cand)) if lib_coefficients[updated_decoy_offset+i] != 0]
        
            all_spec_ids = lib_spec_ids+decoy_spec_ids
            all_features = np.concatenate((features,decoy_features))
            all_ms2_frags = [[";".join(map(str,j)) for j in i] for i in zip(frag_names+decoy_frag_names,
                                                                            frag_errors+decoy_frag_errors,
                                                                            lib_frag_mz+decoy_lib_frag_mz,
                                                                            lib_frag_int+decoy_lib_frag_int,
                                                                            obs_frag_int+decoy_obs_frag_int,
                                                                            unique_frags+unique_frags_decoy,
                                                                            unique_frags_int+unique_frags_int_decoy)]
            
            
        else:
            all_spec_ids = lib_spec_ids
            all_features = features
            all_ms2_frags = [[";".join(map(str,j)) for j in i] for i in zip(frag_names,
                                                                            frag_errors,
                                                                            lib_frag_mz,
                                                                            lib_frag_int,
                                                                            obs_frag_int,
                                                                            unique_frags,
                                                                            unique_frags_int)]
            
        return_prot = config.protein_column in library[next(iter(library))]
        
        if config.args.timeplex:
            output = [[non_zero_coeffs[i],
                       spec_idx,
                       ms1_spec.scan_num,
                       all_spec_ids[i][0],
                       all_spec_ids[i][1],
                       all_spec_ids[i][2],
                       prec_mz,
                       prec_rt,
                       *all_features[j],
                       *all_ms2_frags[j],
                       config.args.mzml,
                       library[(re.sub("Decoy_","",all_spec_ids[i][0]),all_spec_ids[i][1],all_spec_ids[i][2])][config.protein_column] if return_prot else "NA" ]
                       for i,j in zip(range(len(non_zero_coeffs)),non_zero_coeffs_idxs)]
        
        else:
            
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
            
        # lib_spec_ids = [ref_pep_cand[i] for i in range(len(ref_pep_cand)) if lib_coefficients[i] != 0]
        # output = [[non_zero_coeffs[i],spec_idx,lib_spec_ids[i][0],lib_spec_ids[i][1],prec_mz,prec_rt,*features[j]] for i,j in zip(range(len(non_zero_coeffs)),non_zero_coeffs_idxs)]
    
    if return_frags:
        return output, [frag_errors,lib_frag_mz]
    else:
        return output


# #@profile
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
   
    if ms1_mz:
        _bool = (np.abs(rt_mz[:,1]-ms1_mz)/ms1_mz)<ms1_tol
        
    else:
        if rt_filter:
            _bool = np.logical_and(np.abs(rt_mz[:,1]-prec_mz)<(windowWidth/2),np.abs(rt_mz[:,0]-prec_rt)<rt_tol)
        else:
            _bool = np.abs(rt_mz[:,1]-prec_mz)<(windowWidth/2)
            
    window_idxs = np.where(_bool)[0]        
        
        
        
    ### match lib spec to features
    if dino_features is not None:
        filtered_dino = feature_list_mz(feature_list_rt(dino_features,prec_rt,rt_tol=rt_tol),
                                        prec_mz,windowWidth)
        window_edges = createTolWindows(filtered_dino.mz, tolerance=ms1_tol)
        window_idxs = window_idxs[np.where((np.searchsorted(window_edges,rt_mz[window_idxs,1])%2)==1)[0]]
        
    
    mass_window_candidates = [all_keys[i] for i in window_idxs] 
    candidate_peaks = [library[i]['spectrum'] for i in mass_window_candidates]
    
    # # filter possible lib entries for windows.. NB: DONT LIKE HOW I DO SAME LOOP TWICE
    # candidate_lib = [spectrum for key,spectrum in library.items() if spectrum["prec_mz"]>spec.ms1window[0] and spectrum["prec_mz"]<spec.ms1window[1]]
    # mass_window_candidates = [key for key,spectrum in library.items() if spectrum["prec_mz"]>spec.ms1window[0] and spectrum["prec_mz"]<spec.ms1window[1]]
    # # list of peaks from each candiate pep
    # # candidate_peaks = [SpecLib.frag_to_peak(i["frags"]) for i in candidate_lib]
    # candidate_peaks = [i["spectrum"] for i in candidate_lib]
    
    
    
    ###### Process dia spectrum 
    
    # what are the first indices of peaks grouped by tolerance
    merged_coords_idxs = np.searchsorted(dia_spectrum[:,0]+mz_tol*dia_spectrum[:,0],dia_spectrum[:,0])
    
    # what are the first mz of these peak groups
    merged_coords = dia_spectrum[np.unique(merged_coords_idxs),0]
    # print(merged_coords)
    
    
    # NB - should we not sum the intensities?????
    # merged_intensities = [np.mean(dia_spectrum[np.where(merged_coords_idxs==i)[0],1]) for i in np.unique(merged_coords_idxs)]
    merged_intensities = np.zeros(len((merged_coords_idxs)))
    for j,val in zip(merged_coords_idxs,dia_spectrum[:,1]):
        merged_intensities[j]+=val
    #merged_intensities = [np.mean(dia_spectrum[merged_coords_idxs==i,1]) for i in np.unique(merged_coords_idxs)]
    merged_intensities = merged_intensities[merged_intensities!=0]
    
    #update spectrum to new values (note mz remains first in group as this will eventually be rounded)
    dia_spectrum = np.array((merged_coords,merged_intensities)).transpose()
    # print(dia_spectrum)
    
    #get window edge positions each side of peaks in observed spectra (NB the tolerance is now about the first peak in the group not the middile)
    centroid_breaks = np.concatenate((dia_spectrum[:,0]-mz_tol*dia_spectrum[:,0],dia_spectrum[:,0]+mz_tol*dia_spectrum[:,0]))
    centroid_breaks = np.sort(centroid_breaks)
    bin_centers = np.mean(np.stack((centroid_breaks[::2],centroid_breaks[1::2]),1),1)
    
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


def fit_to_lib_decoy(dia_spec,library,rt_mz,all_keys,dino_features=None,rt_filter=False,ms1_mz=None,mz_func = np.array, # mz_func is calibration function - default is just keeping values the same,
               ms1_spectra = None,
               rt_tol = config.rt_tol,
               ms1_tol = config.ms1_tol,
               mz_tol = config.mz_tol):
    print("AAAAAAAAA")
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
   
    if ms1_mz:
        _bool = np.abs(rt_mz[:,1]-ms1_mz)<ms1_tol
        
    else:
        if rt_filter:
            _bool = np.logical_and(np.abs(rt_mz[:,1]-prec_mz)<(windowWidth/2),np.abs(rt_mz[:,0]-prec_rt)<rt_tol)
        else:
            _bool = np.abs(rt_mz[:,1]-prec_mz)<(windowWidth/2)
            
    window_idxs = np.where(_bool)[0]        
        
        
        
    ### match lib spec to features
    if dino_features is not None:
        filtered_dino = feature_list_mz(feature_list_rt(dino_features,prec_rt,rt_tol=rt_tol),
                                        prec_mz,windowWidth)
        window_edges = createTolWindows(filtered_dino.mz, tolerance=ms1_tol)
        window_idxs = window_idxs[np.where((np.searchsorted(window_edges,rt_mz[window_idxs,1])%2)==1)[0]]
        
    
    mass_window_candidates = [all_keys[i] for i in window_idxs]
    candidate_peaks = [library[i]['spectrum'] for i in mass_window_candidates]
    
    
    ###### Process dia spectrum 
    
    # what are the first indices of peaks grouped by tolerance
    merged_coords_idxs = np.searchsorted(dia_spectrum[:,0]+mz_tol*dia_spectrum[:,0],dia_spectrum[:,0])
    
    # what are the first mz of these peak groups
    merged_coords = dia_spectrum[np.unique(merged_coords_idxs),0]
    # print(merged_coords)
    
    
    # NB - should we not sum the intensities?????
    # merged_intensities = [np.mean(dia_spectrum[np.where(merged_coords_idxs==i)[0],1]) for i in np.unique(merged_coords_idxs)]
    merged_intensities = np.zeros(len((merged_coords_idxs)))
    for j,val in zip(merged_coords_idxs,dia_spectrum[:,1]):
        merged_intensities[j]+=val
    #merged_intensities = [np.mean(dia_spectrum[merged_coords_idxs==i,1]) for i in np.unique(merged_coords_idxs)]
    merged_intensities = merged_intensities[merged_intensities!=0]
    
    #update spectrum to new values (note mz remains first in group as this will eventually be rounded)
    dia_spectrum = np.array((merged_coords,merged_intensities)).transpose()
    # print(dia_spectrum)
    
    #get window edge positions each side of peaks in observed spectra (NB the tolerance is now about the first peak in the group not the middile)
    centroid_breaks = np.concatenate((dia_spectrum[:,0]-mz_tol*dia_spectrum[:,0],dia_spectrum[:,0]+mz_tol*dia_spectrum[:,0]))
    centroid_breaks = np.sort(centroid_breaks)
    bin_centers = np.mean(np.stack((centroid_breaks[::2],centroid_breaks[1::2]),1),1)
    
    # lib_idx=0
    # M = candidate_peaks[lib_idx]
    ref_coords = [np.searchsorted(centroid_breaks,M[:,0]) for M in candidate_peaks]
    top_ten = [np.searchsorted(centroid_breaks,M[np.argsort(-M[:,1])[0:min(top_n,M.shape[0])],0]) for M in candidate_peaks]
    
    ## Filter precursors based on resp. MS1 peak
    ms1_peak = ~np.isnan([closest_peak_diff(mz,ms1_spec.mz) for mz in rt_mz[window_idxs,1]])
     
    
    # does the top ten peaks fall between centroid breaks? i.e. odd numbers (%2==1), 
    # ref_peaks_in_dia = [i for i in range(len(candidate_peaks)) if len([a for a in top_ten[i] if a%2 ==1])>atleast_m]
    # print(ref_peaks_in_dia)
    all_norm_intensities = [M[:,1]/sum(M[:,1]) for M in candidate_peaks]
    # ref_peaks_in_dia = [i for i in range(len(candidate_peaks)) if np.sum(all_norm_intensities[i][(ref_coords[i]%2)==1])>0.5 and np.sum(top_ten[i]%2)>atleast_m]
    ref_peaks_in_dia = [i for i in range(len(candidate_peaks)) if np.sum(all_norm_intensities[i][(ref_coords[i]%2)==1])>0.5 and np.sum(top_ten[i]%2)>atleast_m and ms1_peak[i]]
    # print(ref_peaks_in_dia)
    
    prop_ref_peaks_in_dia = [len([a for a in top_ten[i] if a%2 ==1])/candidate_peaks[i].shape[0] for i in range(len(candidate_peaks))]
    
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
    
    
    
    
    ### Generate eqivalent Decoy spectra
    
    mass_window_decoy_candidates = [("Decoy_"+i[0],i[1]) for i in mass_window_candidates] 
    converted_seqs = [change_seq(i[0]) for i in mass_window_candidates]
    decoy_mz = np.array([mass.fast_mass(i, charge=j[1]) for i,j in zip(converted_seqs, mass_window_candidates)])
    converted_frags = [convert_frags(i[0], library[i]["frags"]) for i in mass_window_candidates]
    candidate_decoy_peaks = [frag_to_peak(i) for i in converted_frags]
    
    ## Decoy equiv
    decoy_coords = [np.searchsorted(centroid_breaks,M[:,0]) for M in candidate_decoy_peaks]
    top_ten_decoy = [np.searchsorted(centroid_breaks,M[np.argsort(-M[:,1])[0:min(top_n,M.shape[0])],0]) for M in candidate_decoy_peaks]
    # decoy_peaks_in_dia = [i for i in range(len(candidate_decoy_peaks)) if len([a for a in top_ten_decoy[i] if a%2 ==1])>atleast_m]
    all_norm_decoy_intensities = [M[:,1]/sum(M[:,1]) for M in candidate_decoy_peaks]
    decoy_ms1_peak = ~np.isnan([closest_peak_diff(mz,ms1_spec.mz) for mz in decoy_mz])
    decoy_peaks_in_dia = [i for i in range(len(candidate_peaks)) if np.sum(all_norm_decoy_intensities[i][(decoy_coords[i]%2)==1])>0.5 and np.sum(top_ten_decoy[i]%2)>atleast_m and decoy_ms1_peak[i]]
    
    decoy_pep_cand_loc = [decoy_coords[i] for i in decoy_peaks_in_dia]
    decoy_pep_cand_list = [candidate_decoy_peaks[i] for i in decoy_peaks_in_dia]
    decoy_pep_cand = [mass_window_decoy_candidates[i] for i in decoy_peaks_in_dia] # Nb this is modified seq!!
    
    norm_decoy_intensities = [M[:,1]/sum(M[:,1]) for M in decoy_pep_cand_list]
    
    decoy_lib_peaks_matched = [j%2==1 for j in decoy_pep_cand_loc]
    
    decoy_spec_row_indices_split = [np.int32(((i[j]+1)/2)-1) for i,j in zip(decoy_pep_cand_loc,decoy_lib_peaks_matched)] # NB these are floats
    num_decoy_peaks_matched = np.array([np.sum(i) for i in decoy_lib_peaks_matched]) #f1
    decoy_spec_col_indices_split = [np.array([idx]*i,dtype=int) for idx,i in zip(range(len(decoy_pep_cand)),num_decoy_peaks_matched)] 
    decoy_spec_values_split = [ints[i] for ints,i in zip(norm_decoy_intensities,decoy_lib_peaks_matched)]
    
    frag_errors = []
    frag_mz = []
    decoy_frag_errors = []
    decoy_frag_mz = []
    
    if len(ref_spec_row_indices_split)>0 and len(ref_spec_col_indices_split)>0 and len(ref_spec_values_split)>0:
        
        #### concatenate the matrix values
        ref_spec_row_indices = np.concatenate(ref_spec_row_indices_split)
        ref_spec_col_indices = np.concatenate(ref_spec_col_indices_split)
        ref_spec_values = np.concatenate(ref_spec_values_split)
        
        frag_errors = [np.array(bin_centers[ref_spec_row_indices_split[i]]-ref_pep_cand_list[i][:,0][lib_peaks_matched[i]])/bin_centers[ref_spec_row_indices_split[i]] for i in range(len(lib_peaks_matched))]
        frag_mz = [ref_pep_cand_list[i][:,0][lib_peaks_matched[i]] for i in range(len(lib_peaks_matched))]
        
        
        if len(decoy_spec_row_indices_split)>0:
            decoy_spec_row_indices = np.concatenate(decoy_spec_row_indices_split)
            decoy_spec_col_indices = np.concatenate(decoy_spec_col_indices_split)+max(ref_spec_col_indices)+1
            decoy_spec_values = np.concatenate(decoy_spec_values_split)
            decoy_frag_errors = [np.array(bin_centers[decoy_spec_row_indices_split[i]]-decoy_pep_cand_list[i][:,0][decoy_lib_peaks_matched[i]])/bin_centers[decoy_spec_row_indices_split[i]] for i in range(len(decoy_lib_peaks_matched))]
            decoy_frag_mz = [decoy_pep_cand_list[i][:,0][decoy_lib_peaks_matched[i]] for i in range(len(decoy_lib_peaks_matched))]
            
        else:
            decoy_spec_row_indices=np.array([],dtype=int)
            decoy_spec_col_indices=np.array([],dtype=int)
            decoy_spec_values=np.array([],dtype=int)
        
        # what peaks from the spectrum are matched by library peps
        # unique_row_idxs = [int(i) for i in set(np.concatenate([ref_spec_row_indices,decoy_spec_row_indices]))]
        # unique_row_idxs.sort()
        unique_row_idxs = np.unique(np.concatenate((ref_spec_row_indices,decoy_spec_row_indices)))
        unique_row_idxs = np.array(np.sort(unique_row_idxs),dtype=int)
        
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
    
        ref_sparse_row_indices = np.append(ref_spec_row_indices,not_dia_row_indices)
        ref_sparse_col_indices = np.append(ref_spec_col_indices,not_dia_col_indices)
        ref_sparse_values = np.append(ref_spec_values,not_dia_values)
        
        ### Decoy
        decoy_peaks_not_in_dia = np.array([idx for loc_list in decoy_pep_cand_loc for idx in range(len(loc_list)) if loc_list[idx]%2==0])
        decoy_not_dia_col_indices = np.arange(len(decoy_pep_cand))
        num_rows = max(unique_row_idxs)
        decoy_not_dia_row_indices = [num_rows+1]*len(decoy_not_dia_col_indices)
        decoy_not_dia_values = np.array([np.sum([norm_decoy_intensities[idx][peak_idx] for peak_idx in range(len(norm_decoy_intensities[idx])) if decoy_pep_cand_loc[idx][peak_idx]%2==0])
                                  for idx in range(len(norm_decoy_intensities))])
    
        decoy_sparse_row_indices = np.append(decoy_spec_row_indices,decoy_not_dia_row_indices)
        decoy_sparse_col_indices = np.append(decoy_spec_col_indices,decoy_not_dia_col_indices+max(ref_spec_col_indices)+1)
        decoy_sparse_values = np.append(decoy_spec_values,decoy_not_dia_values)
        
        
        sparse_row_indices = np.concatenate((ref_sparse_row_indices,decoy_sparse_row_indices))
        sparse_col_indices = np.concatenate((ref_sparse_col_indices,decoy_sparse_col_indices))
        sparse_values = np.concatenate((ref_sparse_values,decoy_sparse_values))
        
        # some dia peaks are not matched and are therefore ignored
        # below ranks the rows by number therefore removing missing rows
        sparse_row_indices = stats.rankdata(sparse_row_indices,method="dense").astype(int)-1
        
        # Generate sparse matrix from data
        sparse_lib_matrix = sparse.coo_matrix((sparse_values,(sparse_row_indices,sparse_col_indices)))

        # Fit lib spectra to observed spectra
        fit_results = sparse_nnls.lsqnonneg(sparse_lib_matrix,dia_spec_int,{"show_progress":False})
        lib_coefficients = fit_results['x']
        
        
        
        ####################################
        ### features 
        frac_lib_intensity = [np.sum(i) for i in ref_spec_values_split] # all ints sum to 1 so these give frac
        tic = np.sum(dia_spectrum[:,1])
        frac_dia_intensity = [np.sum(dia_spectrum[i,1])/tic for i in ref_spec_row_indices_split]
        # mz tol
        if ms1_spectra is not None:
            rel_error = np.array([closest_peak_diff(mz,ms1_spec.mz) for mz in rt_mz[window_idxs[ref_peaks_in_dia],1]])
        elif dino_features is not None:
            rel_error = ms1_error(np.array(filtered_dino.mz), rt_mz[window_idxs[ref_peaks_in_dia],1], tol=ms1_tol)
        else:
            rel_error = np.zeros(len(ref_peaks_in_dia))
        rt_error = prec_rt-rt_mz[window_idxs[ref_peaks_in_dia],0]
        
        frac_int_matched = np.sum(dia_spec_int)/np.sum(dia_spectrum[:,1])
        predicted_spec = np.squeeze(sparse_lib_matrix*lib_coefficients)[:-1]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r2all = np_pearson_cor(dia_spec_int[:-1],predicted_spec).statistic
        
            r2_lib_spec = [np_pearson_cor(i,dia_spectrum[j,1]).statistic for i,j in zip(ref_spec_values_split,ref_spec_row_indices_split)]
        
        single_matched_rows = np.where(np.sum(sparse_lib_matrix>0,1)==1)[0]
        peaks_not_shared = [np.array([[dia_spectrum[i,1],j] for i,j in zip(dia,lib) if i in single_matched_rows]) for dia,lib in zip(ref_spec_row_indices_split,ref_spec_values_split)]
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
        large_coeff_matched_peaks = np.unique(np.concatenate(([(ref_spec_row_indices_split+decoy_spec_row_indices_split)[i] for i in large_coeff_indices]))) # select the peaks matched to these
        large_coeff_int_pred = np.sum([np.sum((ref_spec_values_split+decoy_spec_values_split)[i])*list(lib_coefficients)[i] for i in large_coeff_indices]) # sum the intensity predicted
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
                              frac_int_matched_pred,
                              frac_int_matched_pred_sigcoeff,
                              large_coeff_cosine
                                ],-1)
        
        
        ####################################
        ####################################
        ### DECOY features 
        
        frac_lib_intensity = [np.sum(i) for i in decoy_spec_values_split] # all ints sum to 1 so these give frac
        tic = np.sum(dia_spectrum[:,1])
        frac_dia_intensity = [np.sum(dia_spectrum[i,1])/tic for i in decoy_spec_row_indices_split]
        # mz tol
        if ms1_spectra is not None:
            rel_error = np.array([closest_peak_diff(mz,ms1_spec.mz) for mz in rt_mz[window_idxs[decoy_peaks_in_dia],1]])
        elif dino_features is not None:
            rel_error = ms1_error(np.array(filtered_dino.mz), mz_func(np.array(decoy_mz)[decoy_peaks_in_dia]), tol=ms1_tol)
        else:
            rel_error = np.zeros(len(decoy_peaks_in_dia))
        rt_error = prec_rt-rt_mz[window_idxs[decoy_peaks_in_dia],0] #NB this is not a true reflection of rt
        
        frac_int_matched = np.sum(dia_spec_int)/np.sum(dia_spectrum[:,1])
        predicted_spec = np.squeeze(sparse_lib_matrix*lib_coefficients)[:-1]
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            r2all = np_pearson_cor(dia_spec_int[:-1],predicted_spec).statistic
            r2_lib_spec = [np_pearson_cor(i,dia_spectrum[j,1]).statistic for i,j in zip(decoy_spec_values_split,decoy_spec_row_indices_split)]
        
        single_matched_rows = np.where(np.sum(sparse_lib_matrix>0,1)==1)[0]
        peaks_not_shared = [np.array([[dia_spectrum[i,1],j] for i,j in zip(dia,lib) if i in single_matched_rows]) for dia,lib in zip(decoy_spec_row_indices_split,decoy_spec_values_split)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r2_unique = [np_pearson_cor(*i.T).statistic if i.shape[0]>1 else 0 for i in peaks_not_shared ]
        frac_unique_pred = [np.divide(*np.sum(i,axis=0)[::-1])*c if i.shape[0]>0 else 0 for i,c in zip(peaks_not_shared,lib_coefficients)] #frac of int matched by unique peaks pred by unique peaks
        
        frac_dia_intensity_pred = [(i*c)/j for i,j,c in zip(frac_lib_intensity,frac_dia_intensity,lib_coefficients)]
        
        #### stack spectrum features
        r2all = np.ones_like(num_decoy_peaks_matched)*r2all
        frac_int_matched = np.ones_like(num_decoy_peaks_matched)*frac_int_matched
        frac_int_pred = (np.ones_like(num_decoy_peaks_matched)*np.sum(predicted_spec))/tic
        frac_int_matched_pred = (np.ones_like(num_decoy_peaks_matched)*np.sum(predicted_spec))/np.sum(dia_spec_int)
        large_coeff_indices = np.where(np.array(lib_coefficients)>1)[0] # identify large coeffs
        large_coeff_matched_peaks = np.unique(np.concatenate(([(ref_spec_row_indices_split+decoy_spec_row_indices_split)[i] for i in large_coeff_indices]))) # select the peaks matched to these
        large_coeff_int_pred = np.sum([np.sum((ref_spec_values_split+decoy_spec_values_split)[i])*list(lib_coefficients)[i] for i in large_coeff_indices]) # sum the intensity predicted
        large_coeff_int_matched = np.sum(dia_spectrum[large_coeff_matched_peaks,1]) # sum the intensity matched
        ## Note: some predictions over-shoot the matched peak so we overestimate this value
        frac_int_matched_pred_sigcoeff = (np.ones_like(num_decoy_peaks_matched)*large_coeff_int_pred)/large_coeff_int_matched # create vals for all peaks
        
        large_coeff_cosine = np.ones_like(num_decoy_peaks_matched)*subset_cosine
                              
        hyperscores, b_counts, y_counts = map(list, zip(*[hyperscore_b_y(i,j) for i,j in zip([converted_frags[k] for k in decoy_peaks_in_dia],decoy_lib_peaks_matched)]))
        longest_y_ions = [longest_y(i,j) for i,j in zip([longest_y(converted_frags[k]) for k in decoy_peaks_in_dia],decoy_lib_peaks_matched)]
        print("TEST")
        decoy_features = np.stack([num_decoy_peaks_matched,
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
                              frac_int_matched_pred,
                              frac_int_matched_pred_sigcoeff,
                              large_coeff_cosine
                                ],-1)
        
        # all_features.append(features)
        ####################################
    #Select non-zero coeffs
    # Note: many coeffs are non-zero but essentially zero!! Perhaps set less than 1e-7??
    non_zero_coeffs = [c for c in lib_coefficients if c!=0]
    non_zero_coeffs_idxs = [i for i,c in enumerate(lib_coefficients) if c!=0]
    
    output = [[0,spec_idx,ms1_spec.scan_num,0,0,prec_mz,prec_rt,*np.zeros(len(names)-7)]]
    
    if len(non_zero_coeffs)>0:
        lib_spec_ids = [ref_pep_cand[i] for i in range(len(ref_pep_cand)) if lib_coefficients[i] != 0]
        decoy_spec_ids = [decoy_pep_cand[i] for i in range(len(decoy_pep_cand)) if lib_coefficients[int(max(ref_sparse_col_indices))+1+i] != 0]
        
        all_spec_ids = lib_spec_ids+decoy_spec_ids
        output = [[non_zero_coeffs[i],
                   spec_idx,
                   all_spec_ids[i][0],
                   all_spec_ids[i][1],
                   prec_mz,
                   prec_rt,
                   *np.concatenate((features,decoy_features))[j]] for i,j in zip(range(len(non_zero_coeffs)),non_zero_coeffs_idxs)]
        # output = [[non_zero_coeffs[i],spec_idx,lib_spec_ids[i][0],lib_spec_ids[i][1],prec_mz,prec_rt,*features[j]] for i,j in zip(range(len(non_zero_coeffs)),non_zero_coeffs_idxs)]
    
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
               decoy=False,
               decoy_library=None):
    """
    Unified implementation of fit_to_lib2 using unified data structures.
    
    This version processes targets and decoys together in a single pass,
    eliminating code duplication while maintaining identical output.
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
    
    # 2. Filter candidates by mass window (same as original)
    if ms1_mz:
        _bool = (np.abs(rt_mz[:, 1] - ms1_mz) / ms1_mz) < ms1_tol
    else:
        if rt_filter:
            _bool = np.logical_and(
                np.abs(rt_mz[:, 1] - prec_mz) < (windowWidth / 2),
                np.abs(rt_mz[:, 0] - prec_rt) < rt_tol
            )
        else:
            _bool = np.abs(rt_mz[:, 1] - prec_mz) < (windowWidth / 2)
    
    window_idxs = np.where(_bool)[0]
    
    # Filter by dino features if provided
    if dino_features is not None:
        filtered_dino = feature_list_mz(feature_list_rt(dino_features, prec_rt, rt_tol=rt_tol),
                                      prec_mz, windowWidth)
        window_edges = createTolWindows(filtered_dino.mz, tolerance=ms1_tol)
        window_idxs = window_idxs[np.where((np.searchsorted(window_edges, rt_mz[window_idxs, 1]) % 2) == 1)[0]]
    
    mass_window_candidates = [all_keys[i] for i in window_idxs]
    candidate_peaks = [library[i]['spectrum'] for i in mass_window_candidates]
    
    # Early exit if no candidates
    if len(mass_window_candidates) == 0:
        return [[0, spec_idx, ms1_spec.scan_num if ms1_spec else 0, 0, 0, 
                prec_mz, prec_rt, *np.zeros(len(names) - 7)]]
    
    # 3. Process DIA spectrum
    # Merge peaks within tolerance
    merged_coords_idxs = np.searchsorted(dia_spectrum[:, 0] + mz_tol * dia_spectrum[:, 0], dia_spectrum[:, 0])
    merged_coords = dia_spectrum[np.unique(merged_coords_idxs), 0]
    
    # Sum intensities for merged peaks
    merged_intensities = np.zeros(len(merged_coords_idxs))
    for j, val in zip(merged_coords_idxs, dia_spectrum[:, 1]):
        merged_intensities[j] += val
    merged_intensities = merged_intensities[merged_intensities != 0]
    
    # Update spectrum
    dia_spectrum = np.array((merged_coords, merged_intensities)).transpose()
    
    # Calculate centroid breaks and bin centers
    centroid_breaks = np.concatenate((dia_spectrum[:, 0] - mz_tol * dia_spectrum[:, 0],
                                    dia_spectrum[:, 0] + mz_tol * dia_spectrum[:, 0]))
    centroid_breaks = np.sort(centroid_breaks)
    bin_centers = np.mean(np.stack((centroid_breaks[::2], centroid_breaks[1::2]), 1), 1)
    
    # ===== UNIFIED PROCESSING STARTS HERE =====
    
    # 4. Create unified candidates structure
    if decoy and decoy_library:
        # Generate decoy candidates
        decoy_candidates = [(f"Decoy_{k[0]}", k[1]) for k in mass_window_candidates]
        decoy_peaks = [decoy_library[k]["spectrum"] for k in mass_window_candidates]
        
        unified = create_unified_candidates(
            target_candidates=mass_window_candidates,
            target_peaks=candidate_peaks,
            decoy_candidates=decoy_candidates,
            decoy_peaks=decoy_peaks
        )
    else:
        unified = create_unified_candidates(
            target_candidates=mass_window_candidates,
            target_peaks=candidate_peaks
        )
    
    # 5. Process all candidates in ONE call
    updated_unified, matrix_data, additional_outputs = create_entries_unified(
        centroid_breaks=centroid_breaks,
        unified_candidates=unified,
        top_n=config.top_n,
        atleast_m=config.atleast_m,
        prec_mzs=rt_mz[:, 1][window_idxs] if not decoy else np.concatenate([
            rt_mz[:, 1][window_idxs],
            rt_mz[:, 1][window_idxs] - config.decoy_mz_offset
        ]),
        ms1_spec=ms1_spec,
        ms1_tol=ms1_tol,
        library=library,
        decoy_library=decoy_library,
        bin_centers=bin_centers,
        dia_spectrum=dia_spectrum
    )
    
    # Check if any candidates passed filtering
    if len(updated_unified.peaks_in_dia) == 0:
        return [[0, spec_idx, ms1_spec.scan_num if ms1_spec else 0, 0, 0,
                prec_mz, prec_rt, *np.zeros(len(names) - 7)]]
    
    # 6. Build matrix and solve NNLS
    matrix_results = process_matrix_unified(
        unified_candidates=updated_unified,
        matrix_data=matrix_data,
        additional_outputs=additional_outputs,
        dia_spectrum=dia_spectrum,
        unmatched_fit_type=config.unmatched_fit_type
    )
    
    # 7. Calculate features
    unified_features = calculate_features_unified(
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
    
    # 8. Format output for compatibility
    lib_coefficients = matrix_results['lib_coefficients']
    non_zero_coeffs = [c for c in lib_coefficients if c != 0]
    non_zero_coeffs_idxs = [i for i, c in enumerate(lib_coefficients) if c != 0]
    
    output = [[0, spec_idx, ms1_spec.scan_num if ms1_spec else 0, 0, 0,
              prec_mz, prec_rt, *np.zeros(len(names) - 7)]]
    
    if len(non_zero_coeffs) > 0:
        # Get matched candidates
        matched_candidates = [updated_unified.candidates[i] for i in updated_unified.peaks_in_dia]
        
        # Build output rows
        output = []
        for i, j in zip(range(len(non_zero_coeffs)), non_zero_coeffs_idxs):
            if j < len(matched_candidates):
                candidate = matched_candidates[j]
                features = unified_features.features[j]
                
                # Get fragment data and format it
                if j < len(additional_outputs.get('frag_names', [])):
                    # Get fragment data from additional_outputs
                    frag_names = additional_outputs['frag_names'][j]
                    frag_errors = additional_outputs['frag_errors'][j]
                    lib_frag_mz = additional_outputs['lib_frag_mz'][j]
                    lib_frag_int = additional_outputs['lib_frag_int'][j]
                    obs_frag_int = additional_outputs['obs_frag_int'][j]
                    
                    # Calculate unique fragments after matrix construction
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
                else:
                    ms2_frags = [""] * 7
                
                # Get protein info if available
                if config.protein_column and library:
                    try:
                        clean_key = (candidate[0].replace("Decoy_", ""), candidate[1])
                        protein = library.get(clean_key, {}).get(config.protein_column, "NA")
                    except:
                        protein = "NA"
                else:
                    protein = "NA"
                
                row = [
                    non_zero_coeffs[i],
                    spec_idx,
                    ms1_spec.scan_num if ms1_spec else 0,  # Ms1_spec_id
                    candidate[0],  # sequence
                    candidate[1],  # charge
                    prec_mz,
                    prec_rt,
                    *features,
                    *ms2_frags,
                    config.args.mzml if hasattr(config, 'args') else "",
                    protein
                ]
                # Debug: Print first row to check column alignment
                # if len(output) == 0:
                #     print(f"DEBUG: First output row columns:")
                #     print(f"  [0] coeff: {row[0]}")
                #     print(f"  [1] spec_id: {row[1]}")
                #     print(f"  [2] Ms1_spec_id: {row[2]}")
                #     print(f"  [3] seq: {row[3]}")
                #     print(f"  [4] z: {row[4]}")
                #     print(f"  Row length: {len(row)}")
                output.append(row)
    
    if return_frags:
        frag_errors = matrix_results.get('frag_errors', [])
        lib_frag_mz = matrix_results.get('lib_frag_mz', [])
        return output, [frag_errors, lib_frag_mz]
    else:
        return output


