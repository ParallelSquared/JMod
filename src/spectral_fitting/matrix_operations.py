"""
Matrix operations for spectral fitting.

This module provides functions to create and manipulate the unified SpectrumMatrix
data structure, consolidating the previously redundant split arrays.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from scipy import sparse, stats

from .types import SpectrumMatrix, PeptideSpectralData


def create_spectrum_matrix(
    target_row_indices_split: List[np.ndarray],
    target_col_indices_split: List[np.ndarray], 
    target_values_split: List[np.ndarray],
    target_peptide_candidates: List[Any],
    decoy_row_indices_split: Optional[List[np.ndarray]] = None,
    decoy_col_indices_split: Optional[List[np.ndarray]] = None,
    decoy_values_split: Optional[List[np.ndarray]] = None,
    decoy_peptide_candidates: Optional[List[Any]] = None
) -> SpectrumMatrix:
    """
    Create a unified SpectrumMatrix from split target and decoy data.
    
    This consolidates the redundant split arrays into a single structure.
    
    Args:
        target_row_indices_split: Row indices for each target peptide
        target_col_indices_split: Column indices for each target peptide
        target_values_split: Intensity values for each target peptide
        target_peptide_candidates: List of target peptide identifiers
        decoy_row_indices_split: Row indices for each decoy peptide (optional)
        decoy_col_indices_split: Column indices for each decoy peptide (optional)
        decoy_values_split: Intensity values for each decoy peptide (optional)
        decoy_peptide_candidates: List of decoy peptide identifiers (optional)
        
    Returns:
        Unified SpectrumMatrix containing both target and decoy data
    """
    # Process target data
    if target_row_indices_split and target_col_indices_split and target_values_split:
        target_row_indices = np.concatenate(target_row_indices_split)
        target_col_indices = np.concatenate(target_col_indices_split)
        target_values = np.concatenate(target_values_split)
    else:
        target_row_indices = np.array([], dtype=int)
        target_col_indices = np.array([], dtype=int)
        target_values = np.array([], dtype=float)
    
    # Initialize with target data
    all_row_indices = target_row_indices
    all_col_indices = target_col_indices
    all_values = target_values
    all_peptide_candidates = list(target_peptide_candidates)
    
    # Create decoy mask
    num_targets = len(target_peptide_candidates)
    is_decoy = np.zeros(num_targets, dtype=bool)
    
    # Process decoy data if provided
    if (decoy_row_indices_split is not None and 
        decoy_col_indices_split is not None and 
        decoy_values_split is not None and
        decoy_peptide_candidates is not None):
        
        if decoy_row_indices_split and decoy_col_indices_split and decoy_values_split:
            decoy_row_indices = np.concatenate(decoy_row_indices_split)
            # Offset decoy column indices to avoid overlap with targets
            decoy_col_offset = num_targets
            decoy_col_indices = np.concatenate(decoy_col_indices_split) + decoy_col_offset
            decoy_values = np.concatenate(decoy_values_split)
            
            # Combine with target data
            all_row_indices = np.concatenate([all_row_indices, decoy_row_indices])
            all_col_indices = np.concatenate([all_col_indices, decoy_col_indices])
            all_values = np.concatenate([all_values, decoy_values])
            all_peptide_candidates.extend(decoy_peptide_candidates)
            
            # Update decoy mask
            num_decoys = len(decoy_peptide_candidates)
            is_decoy = np.concatenate([is_decoy, np.ones(num_decoys, dtype=bool)])
    
    return SpectrumMatrix(
        values=all_values,
        row_indices=all_row_indices,
        col_indices=all_col_indices,
        peptide_candidates=all_peptide_candidates,
        is_decoy=is_decoy
    )


def add_unmatched_peaks_to_matrix(
    spectrum_matrix: SpectrumMatrix,
    norm_intensities: List[np.ndarray],
    pep_cand_loc: List[np.ndarray],
    fit_type: str = "a",
    lower_limit: float = 1e-10
) -> Tuple[SpectrumMatrix, np.ndarray, np.ndarray, np.ndarray]:
    """
    Add unmatched library peaks to the spectrum matrix.
    
    This handles peaks in the library that don't match any peaks in the DIA spectrum.
    
    Args:
        spectrum_matrix: The current spectrum matrix
        norm_intensities: Normalized intensities for each peptide
        pep_cand_loc: Peak locations for each peptide candidate
        fit_type: Type of fitting ("a", "b", or "c")
        lower_limit: Lower limit for intensity values
        
    Returns:
        Updated spectrum matrix and arrays for unmatched peaks
    """
    # Get unique row indices
    unique_row_idxs = np.unique(spectrum_matrix.row_indices)
    if len(unique_row_idxs) == 0:
        return spectrum_matrix, np.array([]), np.array([]), np.array([])
    
    last_row = max(unique_row_idxs) + 1
    
    # Split by target/decoy
    target_matrix = spectrum_matrix.get_target_data()
    num_targets = np.sum(~spectrum_matrix.is_decoy)
    
    # Calculate unmatched values for each peptide
    not_dia_values = []
    not_dia_col_indices = []
    
    for idx in range(len(norm_intensities)):
        if idx < num_targets:  # Target peptide
            # Sum intensities of peaks not in DIA spectrum
            unmatched_intensity = np.sum([
                norm_intensities[idx][peak_idx] 
                for peak_idx in range(len(norm_intensities[idx])) 
                if pep_cand_loc[idx][peak_idx] % 2 == 0
            ])
            
            if fit_type == "a":
                not_dia_values.append(unmatched_intensity)
            elif fit_type == "b":
                not_dia_values.append(max(unmatched_intensity, lower_limit))
            elif fit_type == "c":
                not_dia_values.append(1.0)
            else:
                raise ValueError(f"Unknown fit_type: {fit_type}")
                
            not_dia_col_indices.append(idx)
    
    if not not_dia_values:
        return spectrum_matrix, np.array([]), np.array([]), np.array([])
    
    not_dia_values = np.array(not_dia_values)
    not_dia_col_indices = np.array(not_dia_col_indices)
    not_dia_row_indices = np.full(len(not_dia_col_indices), last_row)
    
    # Create new spectrum matrix with unmatched peaks added
    new_row_indices = np.concatenate([spectrum_matrix.row_indices, not_dia_row_indices])
    new_col_indices = np.concatenate([spectrum_matrix.col_indices, not_dia_col_indices])
    new_values = np.concatenate([spectrum_matrix.values, not_dia_values])
    
    updated_matrix = SpectrumMatrix(
        values=new_values,
        row_indices=new_row_indices,
        col_indices=new_col_indices,
        peptide_candidates=spectrum_matrix.peptide_candidates,
        is_decoy=spectrum_matrix.is_decoy
    )
    
    return updated_matrix, not_dia_row_indices, not_dia_col_indices, not_dia_values


def rank_and_create_sparse_matrix(
    spectrum_matrix: SpectrumMatrix,
    dia_spec_int: np.ndarray
) -> Tuple[sparse.coo_matrix, np.ndarray]:
    """
    Rank row indices and create sparse matrix for fitting.
    
    Args:
        spectrum_matrix: The spectrum matrix with all peaks
        dia_spec_int: DIA spectrum intensities
        
    Returns:
        Sparse matrix for fitting and updated DIA spectrum intensities
    """
    # Rank row indices to handle missing rows
    ranked_row_indices = stats.rankdata(spectrum_matrix.row_indices, method="dense").astype(int) - 1
    
    # Add zero intensity for unmatched library peaks
    dia_spec_int = np.append(dia_spec_int, [0])
    
    # Create sparse matrix
    sparse_matrix = sparse.coo_matrix(
        (spectrum_matrix.values, (ranked_row_indices, spectrum_matrix.col_indices))
    )
    
    return sparse_matrix, dia_spec_int