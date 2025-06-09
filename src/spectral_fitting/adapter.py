"""
Adapter module to provide backward compatibility during the transition to the new spectral fitting module.

This module provides wrapper functions that maintain the existing API while using the new
refactored implementation underneath.
"""

from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import re

from .fitting_core import fit_spectrum_to_library
from .types import SpectralFitResult
from ..utils.io.read_output import names
import src.config as config


def _convert_result_to_legacy_format(
    result: SpectralFitResult,
    spec_idx: int,
    ms1_scan_num: int,
    prec_mz: float,
    prec_rt: float,
    library: Dict[Any, Dict[str, Any]],
    return_frags: bool = False
) -> Union[List[List[Any]], Tuple[List[List[Any]], List[Any]]]:
    """
    Convert the new SpectralFitResult to the legacy output format.
    
    Args:
        result: SpectralFitResult from new fitting function
        spec_idx: Spectrum scan number
        ms1_scan_num: MS1 scan number
        prec_mz: Precursor m/z
        prec_rt: Precursor RT
        library: Spectral library
        return_frags: Whether to return fragment information
        
    Returns:
        Legacy format output matching the original fit_to_lib/fit_to_lib2 functions
    """
    # Handle empty results
    if len(result.coefficients) == 0 or len(result.features) == 0:
        if config.args.timeplex:
            empty_output = [[0, spec_idx, ms1_scan_num, 0, 0, -1, prec_mz, prec_rt, *np.zeros(len(names)-7)]]
        else:
            empty_output = [[0, spec_idx, ms1_scan_num, 0, 0, prec_mz, prec_rt, *np.zeros(len(names)-7)]]
        
        if return_frags:
            return empty_output, [[], []]
        return empty_output
    
    # Get non-zero coefficients
    non_zero_mask = result.coefficients != 0
    non_zero_coeffs = result.coefficients[non_zero_mask]
    non_zero_idxs = np.where(non_zero_mask)[0]
    
    if len(non_zero_coeffs) == 0:
        if config.args.timeplex:
            empty_output = [[0, spec_idx, ms1_scan_num, 0, 0, -1, prec_mz, prec_rt, *np.zeros(len(names)-7)]]
        else:
            empty_output = [[0, spec_idx, ms1_scan_num, 0, 0, prec_mz, prec_rt, *np.zeros(len(names)-7)]]
        
        if return_frags:
            return empty_output, [[], []]
        return empty_output
    
    # Build output rows
    output = []
    frag_errors_all = []
    lib_frag_mz_all = []
    
    return_prot = config.protein_column in library[next(iter(library))]
    
    for i, coeff_idx in enumerate(non_zero_idxs):
        if coeff_idx >= len(result.peptide_ids):
            continue
            
        peptide_id = result.peptide_ids[coeff_idx]
        is_decoy = result.is_decoy[coeff_idx]
        features = result.features[coeff_idx]
        
        # Stack features in the expected order
        feature_array = np.concatenate([
            features.basic.num_peaks_matched,
            features.basic.frac_lib_intensity,
            features.basic.frac_dia_intensity,
            features.ms1_error,
            features.rt_error,
            features.basic.frac_int_matched,
            features.basic.frac_int_pred,
            features.similarity.r2_all,
            features.similarity.r2_lib_spec,
            features.similarity.r2_unique,
            features.statistical.frac_unique_pred,
            features.statistical.frac_dia_intensity_pred,
            features.fragment_info.hyperscores,
            features.fragment_info.b_counts,
            features.fragment_info.y_counts,
            features.fragment_info.longest_y_ions,
            features.similarity.scribe_scores,
            features.statistical.max_unmatched_residuals,
            features.statistical.max_matched_residuals,
            features.statistical.gof_stats,
            features.similarity.manhattan_distances,
            features.similarity.fitted_spectral_contrasts,
            features.basic.frac_int_pred,  # frac_int_matched_pred
            features.basic.frac_int_pred,  # frac_int_matched_pred_sigcoeff
            features.similarity.cosine_similarity,  # large_coeff_cosine
            features.precursor_mz
        ])
        
        # Format fragment information
        # The 7 fragment-related fields are:
        # frag_names, frag_errors, frag_mz, frag_int, obs_int, unique_frag_mz, unique_obs_int
        
        # For now, populate with proper default values that won't break downstream parsing
        # These should be semicolon-delimited strings of values, not empty strings
        frag_names = ""
        frag_errors = "0"  # Also needs to be a valid float string since it's parsed by unstring_floats
        frag_mz = "0"  # Same for these numeric fields
        frag_int = "0"
        obs_int = "0"  # This needs to be a valid float string, not empty
        unique_frag_mz = "0"
        unique_obs_int = "0"  # This also needs to be a valid float string
        
        # TODO: Properly extract these from features.fragment_info when available
        if hasattr(features.fragment_info, 'frag_names') and features.fragment_info.frag_names:
            if coeff_idx < len(features.fragment_info.frag_names):
                frag_names = ";".join(features.fragment_info.frag_names[coeff_idx])
        
        frag_info = [frag_names, frag_errors, frag_mz, frag_int, obs_int, unique_frag_mz, unique_obs_int]
        
        # Handle protein column
        if return_prot:
            # Remove "Decoy_" prefix if present for protein lookup
            lookup_id = peptide_id
            if isinstance(peptide_id, str) and peptide_id.startswith("Decoy_"):
                lookup_id = peptide_id[6:]  # Remove "Decoy_" prefix
            elif isinstance(peptide_id, tuple) and peptide_id[0].startswith("Decoy_"):
                lookup_id = (peptide_id[0][6:], *peptide_id[1:])
            
            try:
                protein = library[lookup_id].get(config.protein_column, "NA")
            except KeyError:
                protein = "NA"
        else:
            protein = "NA"
        
        if config.args.timeplex:
            if isinstance(peptide_id, tuple) and len(peptide_id) >= 3:
                row = [
                    non_zero_coeffs[i],
                    spec_idx,
                    ms1_scan_num,
                    peptide_id[0],
                    peptide_id[1],
                    peptide_id[2],
                    prec_mz,
                    prec_rt,
                    *feature_array,
                    *frag_info,
                    config.args.mzml,
                    protein
                ]
            else:
                # Handle non-tuple peptide IDs
                row = [
                    non_zero_coeffs[i],
                    spec_idx,
                    ms1_scan_num,
                    peptide_id if not isinstance(peptide_id, tuple) else peptide_id[0],
                    0,  # Default value
                    0,  # Default value
                    prec_mz,
                    prec_rt,
                    *feature_array,
                    *frag_info,
                    config.args.mzml,
                    protein
                ]
        else:
            if isinstance(peptide_id, tuple) and len(peptide_id) >= 2:
                row = [
                    non_zero_coeffs[i],
                    spec_idx,
                    ms1_scan_num,
                    peptide_id[0],
                    peptide_id[1],
                    prec_mz,
                    prec_rt,
                    *feature_array,
                    *frag_info,
                    config.args.mzml,
                    protein
                ]
            else:
                # Handle non-tuple peptide IDs
                row = [
                    non_zero_coeffs[i],
                    spec_idx,
                    ms1_scan_num,
                    peptide_id if not isinstance(peptide_id, tuple) else peptide_id[0],
                    0,  # Default value
                    prec_mz,
                    prec_rt,
                    *feature_array,
                    *frag_info,
                    config.args.mzml,
                    protein
                ]
        
        output.append(row)
        
        # Collect fragment information if requested
        if return_frags and coeff_idx < len(features.fragment_info.frag_errors):
            frag_errors_all.extend(features.fragment_info.frag_errors)
            lib_frag_mz_all.extend(features.fragment_info.frag_mz)
    
    if return_frags:
        return output, [frag_errors_all, lib_frag_mz_all]
    return output


def fit_to_lib(
    dia_spec,
    library,
    rt_mz,
    all_keys,
    dino_features=None,
    rt_filter=False,
    ms1_mz=None,
    ms1_spectra=None,
    rt_tol=None,
    ms1_tol=None,
    mz_tol=None,
    return_frags=False,
    frac_matched=0.5
):
    """
    Legacy wrapper for fit_to_lib function.
    
    This function maintains the original API while using the new unified implementation.
    """
    # Use the new unified function (without decoys for fit_to_lib)
    result = fit_spectrum_to_library(
        dia_spec=dia_spec,
        library=library,
        rt_mz=rt_mz,
        all_keys=all_keys,
        dino_features=dino_features,
        rt_filter=rt_filter,
        ms1_mz=ms1_mz,
        ms1_spectra=ms1_spectra,
        rt_tol=rt_tol,
        ms1_tol=ms1_tol,
        mz_tol=mz_tol,
        include_decoys=False,  # fit_to_lib doesn't include decoys
        decoy_library=None
    )
    
    # Get MS1 scan number
    ms1_scan_num = 0
    if ms1_spectra is not None:
        from ..utils.spectral_similarity_metrics import get_closest_ms1
        ms1_spec = get_closest_ms1(dia_spec.RT, ms1_spectra)
        ms1_scan_num = ms1_spec.scan_num
    
    # Convert to legacy format
    return _convert_result_to_legacy_format(
        result,
        spec_idx=dia_spec.scan_num,
        ms1_scan_num=ms1_scan_num,
        prec_mz=dia_spec.prec_mz,
        prec_rt=dia_spec.RT,
        library=library,
        return_frags=return_frags
    )


def fit_to_lib_decoy(
    dia_spec,
    library,
    rt_mz,
    all_keys,
    dino_features=None,
    rt_filter=False,
    ms1_mz=None,
    mz_func=np.array,
    ms1_spectra=None,
    rt_tol=None,
    ms1_tol=None,
    mz_tol=None
):
    """
    Legacy wrapper for fit_to_lib_decoy function.
    
    This function maintains the original API while using the new unified implementation.
    Note: This function previously had "print('AAAAAAAAA')" which has been removed.
    """
    # Use the new unified function (with decoys)
    result = fit_spectrum_to_library(
        dia_spec=dia_spec,
        library=library,
        rt_mz=rt_mz,
        all_keys=all_keys,
        dino_features=dino_features,
        rt_filter=rt_filter,
        ms1_mz=ms1_mz,
        ms1_spectra=ms1_spectra,
        mz_func=mz_func,
        rt_tol=rt_tol,
        ms1_tol=ms1_tol,
        mz_tol=mz_tol,
        include_decoys=True,  # fit_to_lib_decoy includes decoys
        decoy_library=library  # Assuming decoy library is the same as regular library
    )
    
    # Get MS1 scan number
    ms1_scan_num = 0
    if ms1_spectra is not None:
        from ..utils.misc_functions import closest_ms1spec
        ms1_rt = np.array([i.RT for i in ms1_spectra])
        closest_ms1_scan_idx = closest_ms1spec(dia_spec.RT, ms1_rt)
        ms1_scan_num = ms1_spectra[closest_ms1_scan_idx].scan_num
    
    # Convert to legacy format
    return _convert_result_to_legacy_format(
        result,
        spec_idx=dia_spec.scan_num,
        ms1_scan_num=ms1_scan_num,
        prec_mz=dia_spec.prec_mz,
        prec_rt=dia_spec.RT,
        library=library,
        return_frags=False
    )


def fit_to_lib2(
    dia_spec,
    library,
    rt_mz,
    all_keys,
    dino_features=None,
    rt_filter=False,
    ms1_mz=None,
    ms1_spectra=None,
    rt_tol=None,
    ms1_tol=None,
    mz_tol=None,
    return_frags=False,
    decoy=False,
    decoy_library=None
):
    """
    Legacy wrapper for fit_to_lib2 function.
    
    This function maintains the original API while using the new unified implementation.
    """
    # Use the new unified function
    result = fit_spectrum_to_library(
        dia_spec=dia_spec,
        library=library,
        rt_mz=rt_mz,
        all_keys=all_keys,
        dino_features=dino_features,
        rt_filter=rt_filter,
        ms1_mz=ms1_mz,
        ms1_spectra=ms1_spectra,
        rt_tol=rt_tol,
        ms1_tol=ms1_tol,
        mz_tol=mz_tol,
        include_decoys=decoy,
        decoy_library=decoy_library
    )
    
    # Get MS1 scan number
    ms1_scan_num = 0
    if ms1_spectra is not None:
        from ..utils.misc_functions import closest_ms1spec
        ms1_rt = np.array([i.RT for i in ms1_spectra])
        closest_ms1_scan_idx = closest_ms1spec(dia_spec.RT, ms1_rt)
        ms1_scan_num = ms1_spectra[closest_ms1_scan_idx].scan_num
    
    # Convert to legacy format
    return _convert_result_to_legacy_format(
        result,
        spec_idx=dia_spec.scan_num,
        ms1_scan_num=ms1_scan_num,
        prec_mz=dia_spec.prec_mz,
        prec_rt=dia_spec.RT,
        library=library,
        return_frags=return_frags
    )