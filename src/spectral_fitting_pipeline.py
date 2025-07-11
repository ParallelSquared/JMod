"""
Refactored spectral fitting pipeline with cleaner separation of concerns.

This module demonstrates how the main fitting functions could be
reorganized into smaller, more focused functions.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass

from .spectral_fitting import (
    UnifiedCandidates, UnifiedMatrixData, UnifiedFeatures,
    filter_candidates_by_window, preprocess_dia_spectrum,
    separate_library_candidates, create_entries, process_matrix,
    calculate_features, format_spectral_fitting_output,
    create_empty_output_row, get_closest_ms1, window_width
)
from .utils.io.read_output import names
from .config_wrapper import ConfigManager, SpectralFittingConfig


@dataclass
class SpectrumData:
    """Container for DIA spectrum data."""
    scan_num: int
    prec_mz: float
    prec_rt: float
    peak_list: List[Tuple[float, float]]
    window_width: float
    
    @property
    def spectrum_array(self) -> np.ndarray:
        """Get spectrum as numpy array."""
        if len(self.peak_list) == 0:
            return np.array([]).reshape(0, 2)
        return np.stack(self.peak_list, 1)


@dataclass 
class FittingContext:
    """Context object containing all data needed for fitting."""
    spectrum: SpectrumData
    library: Dict
    rt_mz: np.ndarray
    all_keys: List
    config: SpectralFittingConfig
    ms1_spectra: Optional[List] = None
    dino_features: Optional[Any] = None
    rt_filter: bool = False
    ms1_mz: Optional[float] = None


class SpectralFittingPipeline:
    """Refactored spectral fitting pipeline with modular design."""
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        """Initialize pipeline with configuration."""
        self.config_manager = config_manager or ConfigManager()
    
    def extract_spectrum_data(self, dia_spec: Any) -> SpectrumData:
        """Extract and validate spectrum data."""
        return SpectrumData(
            scan_num=dia_spec.scan_num,
            prec_mz=dia_spec.prec_mz,
            prec_rt=dia_spec.RT,
            peak_list=dia_spec.peak_list(),
            window_width=window_width(dia_spec)
        )
    
    def find_ms1_spectrum(
        self, 
        prec_rt: float, 
        ms1_spectra: Optional[List]
    ) -> Optional[Any]:
        """Find closest MS1 spectrum if available."""
        if ms1_spectra is None:
            return None
        return get_closest_ms1(prec_rt, ms1_spectra)
    
    def filter_library_candidates(
        self,
        context: FittingContext
    ) -> Tuple[np.ndarray, List]:
        """Filter library candidates by mass window."""
        return filter_candidates_by_window(
            rt_mz=context.rt_mz,
            all_keys=context.all_keys,
            prec_mz=context.spectrum.prec_mz,
            prec_rt=context.spectrum.prec_rt,
            windowWidth=context.spectrum.window_width,
            ms1_mz=context.ms1_mz,
            rt_filter=context.rt_filter,
            ms1_tol=context.config.ms1_tol,
            rt_tol=context.config.rt_tol,
            dino_features=context.dino_features
        )
    
    def process_candidates(
        self,
        candidates: List,
        library: Dict,
        dia_spectrum: np.ndarray,
        centroid_breaks: np.ndarray,
        bin_centers: np.ndarray,
        ms1_spec: Optional[Any],
        config: SpectralFittingConfig,
        include_decoys: bool = True
    ) -> Tuple[UnifiedCandidates, UnifiedMatrixData, Dict]:
        """Process candidates through the matching pipeline."""
        # Create unified structure
        unified = separate_library_candidates(
            mass_window_candidates=candidates,
            library=library,
            include_decoys=include_decoys
        )
        
        # Process all candidates
        updated_unified, matrix_data, additional_outputs = create_entries(
            centroid_breaks=centroid_breaks,
            unified_candidates=unified,
            top_n=config.top_n,
            atleast_m=config.atleast_m,
            prec_mzs=np.array([library[k]["prec_mz"] for k in unified.candidates]),
            ms1_spec=ms1_spec,
            ms1_tol=config.ms1_tol,
            library=library,
            bin_centers=bin_centers,
            dia_spectrum=dia_spectrum,
            frac_matched=config.frac_matched
        )
        
        return updated_unified, matrix_data, additional_outputs
    
    def solve_spectral_matching(
        self,
        unified_candidates: UnifiedCandidates,
        matrix_data: UnifiedMatrixData,
        additional_outputs: Dict,
        dia_spectrum: np.ndarray,
        config: SpectralFittingConfig
    ) -> Dict:
        """Solve the spectral matching problem using NNLS."""
        return process_matrix(
            unified_candidates=unified_candidates,
            matrix_data=matrix_data,
            additional_outputs=additional_outputs,
            dia_spectrum=dia_spectrum,
            unmatched_fit_type=config.unmatched_fit_type
        )
    
    def calculate_scoring_features(
        self,
        unified_candidates: UnifiedCandidates,
        matrix_data: UnifiedMatrixData,
        additional_outputs: Dict,
        matrix_results: Dict,
        context: FittingContext,
        window_idxs: np.ndarray
    ) -> UnifiedFeatures:
        """Calculate all scoring features."""
        return calculate_features(
            unified_candidates=unified_candidates,
            matrix_data=matrix_data,
            additional_outputs=additional_outputs,
            dia_spectrum=context.spectrum.spectrum_array,
            prec_rt=context.spectrum.prec_rt,
            lib_coefficients=matrix_results['lib_coefficients'],
            sparse_matrix=matrix_results['sparse_matrix'],
            peak_idx_convertor=matrix_results['peak_idx_convertor'],
            unique_row_idxs=matrix_results['unique_row_idxs'],
            rt_mz=context.rt_mz,
            window_idxs=window_idxs,
            library=context.library
        )
    
    def fit_spectrum(
        self,
        context: FittingContext,
        return_frags: bool = False,
        include_decoys: bool = True
    ) -> Union[List[List], Tuple[List[List], List]]:
        """
        Main entry point for spectral fitting with cleaner organization.
        
        This demonstrates how fit_to_lib2 could be refactored into
        smaller, more focused steps.
        """
        # Step 1: Extract spectrum data
        spectrum_data = context.spectrum
        dia_spectrum_array = spectrum_data.spectrum_array
        
        # Step 2: Find MS1 spectrum
        ms1_spec = self.find_ms1_spectrum(
            spectrum_data.prec_rt,
            context.ms1_spectra
        )
        
        # Step 3: Filter candidates
        window_idxs, candidates = self.filter_library_candidates(context)
        
        # Handle empty candidates
        if len(candidates) == 0:
            empty_result = [create_empty_output_row(
                spectrum_data.scan_num,
                ms1_spec.scan_num if ms1_spec else 0,
                spectrum_data.prec_mz,
                spectrum_data.prec_rt,
                len(names)
            )]
            return (empty_result, [[], []]) if return_frags else empty_result
        
        # Step 4: Preprocess DIA spectrum
        dia_spectrum, centroid_breaks, bin_centers = preprocess_dia_spectrum(
            dia_spectrum_array,
            context.config.mz_tol
        )
        
        # Step 5: Process candidates
        unified, matrix_data, additional_outputs = self.process_candidates(
            candidates=candidates,
            library=context.library,
            dia_spectrum=dia_spectrum,
            centroid_breaks=centroid_breaks,
            bin_centers=bin_centers,
            ms1_spec=ms1_spec,
            config=context.config,
            include_decoys=include_decoys
        )
        
        # Check if any candidates passed
        if len(unified.peaks_in_dia) == 0:
            empty_result = [create_empty_output_row(
                spectrum_data.scan_num,
                ms1_spec.scan_num if ms1_spec else 0,
                spectrum_data.prec_mz,
                spectrum_data.prec_rt,
                len(names)
            )]
            return (empty_result, [[], []]) if return_frags else empty_result
        
        # Step 6: Solve NNLS
        matrix_results = self.solve_spectral_matching(
            unified, matrix_data, additional_outputs,
            dia_spectrum, context.config
        )
        
        # Step 7: Calculate features
        features = self.calculate_scoring_features(
            unified, matrix_data, additional_outputs,
            matrix_results, context, window_idxs
        )
        
        # Step 8: Format output
        output = format_spectral_fitting_output(
            lib_coefficients=matrix_results['lib_coefficients'],
            unified_candidates=unified,
            unified_features=features,
            additional_outputs=additional_outputs,
            spec_idx=spectrum_data.scan_num,
            ms1_spec=ms1_spec,
            prec_mz=spectrum_data.prec_mz,
            prec_rt=spectrum_data.prec_rt,
            library=context.library,
            config=context.config
        )
        
        if return_frags:
            frag_errors = additional_outputs.get('frag_errors', [])
            lib_frag_mz = additional_outputs.get('lib_frag_mz', [])
            return output, [frag_errors, lib_frag_mz]
        else:
            return output