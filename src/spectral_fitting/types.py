"""
Type definitions and data classes for spectral fitting module.

This module provides structured data types to improve type safety and code clarity
throughout the spectral fitting pipeline.
"""

from typing import NamedTuple, List, Dict, Optional, Tuple, Union, Any
from dataclasses import dataclass
import numpy as np
from scipy import sparse


@dataclass
class PrecursorInfo:
    """Information about a precursor ion."""
    mz: float
    rt: float
    scan_num: int
    window_width: float
    

@dataclass
class FittingParameters:
    """Parameters controlling spectral fitting behavior."""
    rt_tol: float = 0.5
    mz_tol: float = 1e-5
    ms1_tol: float = 1e-7
    top_n: int = 10
    atleast_m: int = 3
    frac_matched: float = 0.5
    use_rt: bool = False
    use_decoy: bool = True
    

@dataclass
class PeptideSpectralData:
    """Data for a single peptide's spectrum."""
    peptide_id: Union[int, str]
    mz_values: np.ndarray
    intensity_values: np.ndarray
    is_decoy: bool
    fragments: Optional[Dict[str, Any]] = None
    

@dataclass
class SpectrumMatrix:
    """
    Unified container for spectral library data.
    
    This replaces the redundant split arrays (ref_spec_values_split, etc.)
    with a single consolidated structure.
    """
    values: np.ndarray
    row_indices: np.ndarray
    col_indices: np.ndarray
    peptide_candidates: List[Union[int, str]]
    is_decoy: np.ndarray  # Boolean mask for decoy status
    
    def to_sparse_matrix(self) -> sparse.coo_matrix:
        """Convert to scipy sparse matrix for fitting."""
        return sparse.coo_matrix(
            (self.values, (self.row_indices, self.col_indices))
        )
    
    def get_peptide_indices(self, peptide_idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract row indices, col indices, and values for a specific peptide."""
        mask = self.col_indices == peptide_idx
        return (
            self.row_indices[mask],
            self.col_indices[mask], 
            self.values[mask]
        )
    
    def get_target_data(self) -> 'SpectrumMatrix':
        """Extract only target (non-decoy) peptide data."""
        target_peptide_indices = np.where(~self.is_decoy)[0]
        mask = np.isin(self.col_indices, target_peptide_indices)
        
        # Re-index column indices to be contiguous
        new_col_indices = np.zeros_like(self.col_indices[mask])
        for new_idx, old_idx in enumerate(target_peptide_indices):
            new_col_indices[self.col_indices[mask] == old_idx] = new_idx
            
        return SpectrumMatrix(
            values=self.values[mask],
            row_indices=self.row_indices[mask],
            col_indices=new_col_indices,
            peptide_candidates=[self.peptide_candidates[i] for i in target_peptide_indices],
            is_decoy=self.is_decoy[target_peptide_indices]
        )
    
    def get_decoy_data(self) -> 'SpectrumMatrix':
        """Extract only decoy peptide data."""
        decoy_peptide_indices = np.where(self.is_decoy)[0]
        mask = np.isin(self.col_indices, decoy_peptide_indices)
        
        # Re-index column indices to be contiguous
        new_col_indices = np.zeros_like(self.col_indices[mask])
        for new_idx, old_idx in enumerate(decoy_peptide_indices):
            new_col_indices[self.col_indices[mask] == old_idx] = new_idx
            
        return SpectrumMatrix(
            values=self.values[mask],
            row_indices=self.row_indices[mask],
            col_indices=new_col_indices,
            peptide_candidates=[self.peptide_candidates[i] for i in decoy_peptide_indices],
            is_decoy=self.is_decoy[decoy_peptide_indices]
        )


class BasicFeatures(NamedTuple):
    """Basic spectral matching features."""
    num_peaks_matched: np.ndarray
    frac_lib_intensity: np.ndarray
    frac_dia_intensity: np.ndarray
    frac_int_matched: np.ndarray
    frac_int_pred: np.ndarray
    

class SimilarityMetrics(NamedTuple):
    """Spectral similarity metrics."""
    scribe_scores: np.ndarray
    manhattan_distances: np.ndarray
    fitted_spectral_contrasts: np.ndarray
    r2_all: np.ndarray
    r2_lib_spec: np.ndarray
    r2_unique: np.ndarray
    cosine_similarity: np.ndarray
    

class StatisticalFeatures(NamedTuple):
    """Statistical analysis features."""
    gof_stats: np.ndarray
    max_unmatched_residuals: np.ndarray
    max_matched_residuals: np.ndarray
    frac_unique_pred: np.ndarray
    frac_dia_intensity_pred: np.ndarray
    

class FragmentInfo(NamedTuple):
    """Fragment-level information."""
    hyperscores: np.ndarray
    b_counts: np.ndarray
    y_counts: np.ndarray
    longest_y_ions: np.ndarray
    frag_errors: List[np.ndarray]
    frag_mz: List[np.ndarray]
    frag_names: List[List[str]]
    

class SpectralFeatures(NamedTuple):
    """Complete set of spectral features for a peptide match."""
    basic: BasicFeatures
    similarity: SimilarityMetrics
    statistical: StatisticalFeatures
    fragment_info: FragmentInfo
    ms1_error: np.ndarray
    rt_error: np.ndarray
    precursor_mz: np.ndarray
    

class SpectralFitResult(NamedTuple):
    """Result of fitting a spectrum to the library."""
    features: List[SpectralFeatures]  # Features for all peptides (target and decoy)
    coefficients: np.ndarray
    peptide_ids: List[Union[int, str]]
    is_decoy: np.ndarray
    sparse_matrix: sparse.coo_matrix
    matched_peak_indices: np.ndarray