"""
Core fitting algorithms for spectral fitting.

This module contains the unified fit_spectrum_to_library function that replaces
the duplicated fit_to_lib and fit_to_lib_decoy functions.
"""

from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import warnings
from scipy import sparse

from .types import (
    SpectralFitResult, FittingParameters, PrecursorInfo,
    SpectrumMatrix, SpectralFeatures
)
from .matrix_operations import (
    create_spectrum_matrix, add_unmatched_peaks_to_matrix,
    rank_and_create_sparse_matrix
)
from .feature_calculation import calculate_all_features
from ..utils.misc_functions import (
    createTolWindows, window_width, feature_list_mz, 
    feature_list_rt, closest_ms1spec, closest_peak_diff
)
from ..utils.spectral_similarity_metrics import get_closest_ms1
from ..utils.io.read_output import names
from ..spectral_fitting_legacy import create_entries
import src.config as config
import ptinnls as sparse_nnls


def prepare_dia_spectrum(
    dia_spectrum: np.ndarray,
    mz_tol: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Process DIA spectrum by merging peaks within tolerance.
    
    Args:
        dia_spectrum: Raw DIA spectrum [mz, intensity]
        mz_tol: m/z tolerance for merging peaks
        
    Returns:
        Tuple of (processed_spectrum, centroid_breaks, bin_centers)
    """
    # Find first indices of peaks grouped by tolerance
    merged_coords_idxs = np.searchsorted(
        dia_spectrum[:, 0] + mz_tol * dia_spectrum[:, 0],
        dia_spectrum[:, 0]
    )
    
    # Get first mz of each peak group
    merged_coords = dia_spectrum[np.unique(merged_coords_idxs), 0]
    
    # Sum intensities for each group
    merged_intensities = np.zeros(len(merged_coords_idxs))
    for j, val in zip(merged_coords_idxs, dia_spectrum[:, 1]):
        merged_intensities[j] += val
    merged_intensities = merged_intensities[merged_intensities != 0]
    
    # Update spectrum to merged values
    processed_spectrum = np.array((merged_coords, merged_intensities)).transpose()
    
    # Calculate tolerance windows
    centroid_breaks = np.concatenate((
        processed_spectrum[:, 0] - mz_tol * processed_spectrum[:, 0],
        processed_spectrum[:, 0] + mz_tol * processed_spectrum[:, 0]
    ))
    centroid_breaks = np.sort(centroid_breaks)
    bin_centers = np.mean(np.stack((centroid_breaks[::2], centroid_breaks[1::2]), 1), 1)
    
    return processed_spectrum, centroid_breaks, bin_centers


def filter_candidates_by_window(
    rt_mz: np.ndarray,
    prec_info: PrecursorInfo,
    params: FittingParameters,
    ms1_mz: Optional[float] = None,
    dino_features: Optional[Any] = None
) -> np.ndarray:
    """
    Filter peptide candidates based on m/z and RT windows.
    
    Args:
        rt_mz: Array of [RT, m/z] for all candidates
        prec_info: Precursor information
        params: Fitting parameters
        ms1_mz: Optional MS1 m/z for filtering
        dino_features: Optional DINO features for filtering
        
    Returns:
        Indices of candidates within window
    """
    if ms1_mz:
        _bool = np.abs(rt_mz[:, 1] - ms1_mz) < params.ms1_tol
    else:
        if params.use_rt:
            _bool = np.logical_and(
                np.abs(rt_mz[:, 1] - prec_info.mz) < (prec_info.window_width / 2),
                np.abs(rt_mz[:, 0] - prec_info.rt) < params.rt_tol
            )
        else:
            _bool = np.abs(rt_mz[:, 1] - prec_info.mz) < (prec_info.window_width / 2)
    
    window_idxs = np.where(_bool)[0]
    
    # Additional filtering with DINO features if provided
    if dino_features is not None:
        filtered_dino = feature_list_mz(
            feature_list_rt(dino_features, prec_info.rt, rt_tol=params.rt_tol),
            prec_info.mz, prec_info.window_width
        )
        window_edges = createTolWindows(filtered_dino.mz, tolerance=params.ms1_tol)
        window_idxs = window_idxs[
            np.where((np.searchsorted(window_edges, rt_mz[window_idxs, 1]) % 2) == 1)[0]
        ]
    
    return window_idxs


def create_entries(
    centroid_breaks: np.ndarray,
    candidate_peaks: List[np.ndarray],
    mass_window_candidates: List[Any],
    top_n: int,
    atleast_m: int,
    prec_mzs: np.ndarray,
    ms1_spec: Any,
    ms1_tol: float,
    frac_matched: float = 0.5,
    spec_frags: Optional[List[Any]] = None,
    top_n_idxs: Optional[List[np.ndarray]] = None
) -> Tuple[List[int], List[Any], List[np.ndarray], List[np.ndarray], 
           List[np.ndarray], List[np.ndarray], List[np.ndarray], 
           List[np.ndarray], List[np.ndarray], np.ndarray]:
    """
    Create entries for peptides that meet matching criteria.
    
    This function filters peptides based on peak matching requirements and
    prepares the data structures needed for fitting.
    """
    # Find coordinates of library peaks in DIA spectrum
    ref_coords = [np.searchsorted(centroid_breaks, M[:, 0]) for M in candidate_peaks]
    
    # Get top N peaks by intensity
    if top_n_idxs is not None and spec_frags is not None:
        top_ten = top_n_idxs
    else:
        top_ten = [
            np.searchsorted(centroid_breaks, M[np.argsort(-M[:, 1])[0:min(top_n, M.shape[0])], 0])
            for M in candidate_peaks
        ]
    
    # Filter by MS1 peak presence
    ms1_peak = ~np.isnan([closest_peak_diff(mz, ms1_spec.mz) for mz in prec_mzs])
    
    # Normalize intensities
    all_norm_intensities = [M[:, 1] / sum(M[:, 1]) for M in candidate_peaks]
    
    # Filter peptides meeting criteria
    ref_peaks_in_dia = [
        i for i in range(len(candidate_peaks))
        if (np.sum(all_norm_intensities[i][(ref_coords[i] % 2) == 1]) > frac_matched and
            np.sum(top_ten[i] % 2) > atleast_m and
            ms1_peak[i] and
            top_ten[i][0] % 2 == 1 and
            np.sum(top_ten[i][:3] % 2 == 1) >= 2)
    ]
    
    # Extract data for filtered peptides
    ref_pep_cand_loc = [ref_coords[i] for i in ref_peaks_in_dia]
    ref_pep_cand_list = [candidate_peaks[i] for i in ref_peaks_in_dia]
    ref_pep_cand = [mass_window_candidates[i] for i in ref_peaks_in_dia]
    
    norm_intensities = [M[:, 1] / sum(M[:, 1]) for M in ref_pep_cand_list]
    
    # Library peaks that match DIA peaks
    lib_peaks_matched = [j % 2 == 1 for j in ref_pep_cand_loc]
    
    # Create split arrays for matrix construction
    ref_spec_row_indices_split = [
        np.int32(((i[j] + 1) / 2) - 1) 
        for i, j in zip(ref_pep_cand_loc, lib_peaks_matched)
    ]
    num_lib_peaks_matched = np.array([np.sum(i) for i in lib_peaks_matched])
    ref_spec_col_indices_split = [
        np.array([idx] * i) 
        for idx, i in zip(range(len(ref_pep_cand)), num_lib_peaks_matched)
    ]
    ref_spec_values_split = [
        ints[i] for ints, i in zip(norm_intensities, lib_peaks_matched)
    ]
    
    # Calculate MS1 errors
    ref_ms1_error = np.zeros(len(ref_peaks_in_dia))  # Placeholder
    
    return (ref_peaks_in_dia, ref_pep_cand, ref_pep_cand_loc, ref_pep_cand_list,
            ref_spec_row_indices_split, ref_spec_col_indices_split,
            ref_spec_values_split, norm_intensities, lib_peaks_matched, ref_ms1_error)


def fit_spectrum_to_library(
    dia_spec: Any,
    library: Dict[Union[int, str], Dict[str, Any]],
    rt_mz: np.ndarray,
    all_keys: List[Union[int, str]],
    dino_features: Optional[Any] = None,
    rt_filter: bool = False,
    ms1_mz: Optional[float] = None,
    ms1_spectra: Optional[List[Any]] = None,
    mz_func: callable = np.array,
    rt_tol: float = None,
    ms1_tol: float = None,
    mz_tol: float = None,
    decoy_library: Optional[Dict[Union[int, str], Dict[str, Any]]] = None,
    include_decoys: bool = True
) -> SpectralFitResult:
    """
    Unified function for fitting a DIA spectrum to a spectral library.
    
    This function replaces both fit_to_lib and fit_to_lib_decoy, eliminating
    900+ lines of code duplication. It handles both target and decoy peptides
    in a single pass using the unified SpectrumMatrix data structure.
    
    Args:
        dia_spec: DIA spectrum to fit
        library: Spectral library
        rt_mz: RT-m/z matrix for candidates
        all_keys: Library keys
        dino_features: Optional DINO features for filtering
        rt_filter: Whether to apply RT filtering
        ms1_mz: Optional MS1 m/z for filtering
        ms1_spectra: Optional MS1 spectra
        mz_func: Calibration function (default: identity)
        rt_tol: RT tolerance (uses config default if None)
        ms1_tol: MS1 tolerance (uses config default if None)
        mz_tol: m/z tolerance (uses config default if None)
        decoy_library: Optional separate decoy library
        include_decoys: Whether to include decoy peptides
        
    Returns:
        SpectralFitResult containing features for all peptides
    """
    # Use config defaults if tolerances not provided
    if rt_tol is None:
        rt_tol = config.rt_tol
    if ms1_tol is None:
        ms1_tol = config.ms1_tol
    if mz_tol is None:
        mz_tol = config.mz_tol
    
    # Get parameters from config
    params = FittingParameters(
        rt_tol=rt_tol,
        mz_tol=mz_tol,
        ms1_tol=ms1_tol,
        top_n=config.top_n,
        atleast_m=config.atleast_m,
        frac_matched=0.5,
        use_rt=rt_filter,
        use_decoy=include_decoys
    )
    
    # Extract spectrum information
    spec = dia_spec
    dia_spectrum = np.stack(spec.peak_list(), 1)
    prec_info = PrecursorInfo(
        mz=spec.prec_mz,
        rt=spec.RT,
        scan_num=spec.scan_num,
        window_width=window_width(dia_spec)
    )
    
    # Get MS1 spectrum if available
    ms1_spec = None
    if ms1_spectra is not None:
        ms1_spec = get_closest_ms1(prec_info.rt, ms1_spectra)
    
    # Filter candidates by window
    window_idxs = filter_candidates_by_window(
        rt_mz, prec_info, params, ms1_mz, dino_features
    )
    
    if len(window_idxs) == 0:
        # No candidates found
        return SpectralFitResult(
            features=[],
            coefficients=np.array([]),
            peptide_ids=[],
            is_decoy=np.array([], dtype=bool),
            sparse_matrix=sparse.coo_matrix((0, 0)),
            matched_peak_indices=np.array([])
        )
    
    # Get candidate peptides and their spectra
    mass_window_candidates = [all_keys[i] for i in window_idxs]
    candidate_peaks = [library[i]['spectrum'] for i in mass_window_candidates]
    
    # Process DIA spectrum
    processed_spectrum, centroid_breaks, bin_centers = prepare_dia_spectrum(
        dia_spectrum, params.mz_tol
    )
    
    # Get top N indices if available
    top_n_idxs = [library[i].get('top_n') for i in mass_window_candidates]
    spec_frags = None
    if "spec_frags" in library[all_keys[0]]:
        spec_frags = [library[i]['spec_frags'] for i in mass_window_candidates]
    
    # Create entries for target peptides
    (ref_peaks_in_dia, ref_pep_cand, ref_pep_cand_loc, ref_pep_cand_list,
     ref_spec_row_indices_split, ref_spec_col_indices_split,
     ref_spec_values_split, norm_intensities, lib_peaks_matched, 
     ref_ms1_error) = create_entries(
        centroid_breaks, candidate_peaks, mass_window_candidates,
        params.top_n, params.atleast_m, rt_mz[:, 1][window_idxs],
        ms1_spec, params.ms1_tol, params.frac_matched,
        spec_frags, top_n_idxs
    )
    
    # Process decoy peptides if requested
    decoy_spec_row_indices_split = []
    decoy_spec_col_indices_split = []
    decoy_spec_values_split = []
    decoy_pep_cand = []
    decoy_ms1_error = np.array([])
    
    if include_decoys and decoy_library is not None:
        # Create decoy candidates
        mass_window_decoy_candidates = [
            ("Decoy_" + str(i[0]) if isinstance(i, tuple) else "Decoy_" + str(i), *i[1:]) 
            if isinstance(i, tuple) else ("Decoy_" + str(i),)
            for i in mass_window_candidates
        ]
        
        # Get decoy data from library
        converted_frags = [decoy_library[i]["frags"] for i in mass_window_candidates]
        decoy_sorted_frags = [decoy_library[i]["ordered_frags"] for i in mass_window_candidates]
        candidate_decoy_peaks = [decoy_library[i]["spectrum"] for i in mass_window_candidates]
        decoy_mz = rt_mz[:, 1][window_idxs] - config.decoy_mz_offset
        
        decoy_top_n_idxs = [decoy_library[i].get('top_n') for i in mass_window_candidates]
        
        # Create entries for decoy peptides
        (decoy_peaks_in_dia, decoy_pep_cand, decoy_pep_cand_loc, decoy_pep_cand_list,
         decoy_spec_row_indices_split, decoy_spec_col_indices_split,
         decoy_spec_values_split, norm_decoy_intensities, decoy_lib_peaks_matched,
         decoy_ms1_error) = create_entries(
            centroid_breaks, candidate_decoy_peaks, mass_window_decoy_candidates,
            params.top_n, params.atleast_m, decoy_mz, ms1_spec, params.ms1_tol,
            params.frac_matched, None, decoy_top_n_idxs
        )
    
    # Create unified spectrum matrix
    spectrum_matrix = create_spectrum_matrix(
        ref_spec_row_indices_split,
        ref_spec_col_indices_split,
        ref_spec_values_split,
        ref_pep_cand,
        decoy_spec_row_indices_split,
        decoy_spec_col_indices_split,
        decoy_spec_values_split,
        decoy_pep_cand
    )
    
    # Handle no matches case
    if len(spectrum_matrix.row_indices) == 0 or len(ref_pep_cand) == 0:
        return SpectralFitResult(
            features=[],
            coefficients=np.array([]),
            peptide_ids=spectrum_matrix.peptide_candidates if len(spectrum_matrix.peptide_candidates) > 0 else [],
            is_decoy=spectrum_matrix.is_decoy if len(spectrum_matrix.is_decoy) > 0 else np.array([], dtype=bool),
            sparse_matrix=sparse.coo_matrix((1, 1)),  # Avoid empty matrix
            matched_peak_indices=np.array([])
        )
    
    # Get unique matched peak indices
    unique_row_idxs = np.unique(spectrum_matrix.row_indices)
    # Filter to valid indices
    unique_row_idxs = unique_row_idxs[unique_row_idxs < len(processed_spectrum)]
    dia_spec_int = processed_spectrum[unique_row_idxs, 1]
    
    # Add unmatched peaks
    spectrum_matrix, _, _, _ = add_unmatched_peaks_to_matrix(
        spectrum_matrix, norm_intensities, ref_pep_cand_loc,
        fit_type=config.unmatched_fit_type
    )
    
    # Create sparse matrix and fit
    sparse_matrix, dia_spec_int = rank_and_create_sparse_matrix(
        spectrum_matrix, dia_spec_int
    )
    
    # Ensure dia_spec_int is the right length for the sparse matrix
    if sparse_matrix.shape[0] > len(dia_spec_int):
        dia_spec_int = np.append(dia_spec_int, [0] * (sparse_matrix.shape[0] - len(dia_spec_int)))
    elif sparse_matrix.shape[0] < len(dia_spec_int):
        dia_spec_int = dia_spec_int[:sparse_matrix.shape[0]]
    
    # Convert to dense array if needed for ptinnls
    if hasattr(sparse_matrix, 'toarray'):
        sparse_matrix_dense = sparse_matrix.toarray()
    else:
        sparse_matrix_dense = sparse_matrix
    
    # Ensure dia_spec_int is a column vector
    dia_spec_int = dia_spec_int.reshape(-1)
    
    # Perform sparse NNLS fitting
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Check if we have a valid matrix
        if sparse_matrix_dense.shape[0] == 0 or sparse_matrix_dense.shape[1] == 0:
            lib_coefficients = np.array([])
        else:
            fit_results = sparse_nnls.lsqnonneg(sparse_matrix_dense, dia_spec_int, {"show_progress": False})
            lib_coefficients = fit_results['x']
            # Convert to numpy array if it's a cvxopt matrix
            if hasattr(lib_coefficients, '__array__'):
                lib_coefficients = np.array(lib_coefficients).flatten()
            else:
                lib_coefficients = np.asarray(lib_coefficients).flatten()
    
    # Calculate all features
    ref_spec_offset = 0
    decoy_spec_offset = np.sum(~spectrum_matrix.is_decoy)
    
    # If no coefficients, return empty result
    if len(lib_coefficients) == 0:
        return SpectralFitResult(
            features=[],
            coefficients=lib_coefficients,
            peptide_ids=spectrum_matrix.peptide_candidates,
            is_decoy=spectrum_matrix.is_decoy,
            sparse_matrix=sparse_matrix,
            matched_peak_indices=unique_row_idxs
        )
    
    # Pass the original spectrum_matrix (with original indices) and the full processed_spectrum 
    # to feature calculations, not the ranked/filtered versions
    features = calculate_all_features(
        spectrum_matrix,  # This has the original row indices
        processed_spectrum,  # This is the full spectrum
        lib_coefficients,
        sparse_matrix,
        rt_mz,
        window_idxs,
        prec_info.rt,
        np.concatenate([ref_ms1_error, decoy_ms1_error]) if len(decoy_pep_cand) > 0 else ref_ms1_error,
        ref_spec_offset,
        decoy_spec_offset,
        lib_peaks_matched,
        None,  # prec_frags
        None   # ordered_frags
    )
    
    return SpectralFitResult(
        features=features,
        coefficients=lib_coefficients,
        peptide_ids=spectrum_matrix.peptide_candidates,
        is_decoy=spectrum_matrix.is_decoy,
        sparse_matrix=sparse_matrix,
        matched_peak_indices=unique_row_idxs
    )