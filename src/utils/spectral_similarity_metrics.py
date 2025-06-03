"""
This Source Code Form is subject to the terms of the Oxford Nanopore
Technologies, Ltd. Public License, v. 1.0.  Full licence can be found
at https://github.com/ParallelSquared/JMod/blob/main/LICENSE.txt
"""

"""
Spectral similarity metrics for mass spectrometry data analysis.

This module contains functions for calculating various similarity metrics
between predicted and observed mass spectra, including SCRIBE scores,
residuals, goodness-of-fit statistics, and Manhattan distances.
"""

from typing import List, Tuple, Any, Union
import numpy as np
from .misc_functions import closest_ms1spec


def get_closest_ms1(prec_rt: float, ms1_spectra: List[Any]) -> Any:
    """
    Find the MS1 spectrum closest in retention time to a given precursor.
    
    This function identifies the MS1 spectrum from a collection of MS1 spectra 
    that has the retention time (RT) closest to the specified precursor RT.
    
    Parameters
    ----------
    prec_rt : float
        Retention time of the precursor ion.
    ms1_spectra : List[Any]
        List of MS1 spectrum objects, each having an RT attribute.
        
    Returns
    -------
    Any
        The MS1 spectrum object with the closest retention time to prec_rt.
        
    Examples
    --------
    >>> # Assuming MS1Spectrum objects with RT attribute
    >>> class MS1Spectrum:
    ...     def __init__(self, rt):
    ...         self.RT = rt
    >>> ms1_spectra = [MS1Spectrum(10.5), MS1Spectrum(15.2), MS1Spectrum(20.7)]
    >>> prec_rt = 16.0
    >>> closest_spectrum = get_closest_ms1(prec_rt, ms1_spectra)
    >>> closest_spectrum.RT
    15.2
    """
    ms1_rt = np.array([i.RT for i in ms1_spectra])
    closest_ms1_scan_idx = closest_ms1spec(prec_rt, ms1_rt)
    ms1_spec = ms1_spectra[closest_ms1_scan_idx]
    return ms1_spec


def get_scribe(
    row_idx_split: List[np.ndarray],
    col_idx_split: List[np.ndarray],
    prec_val_split: List[np.ndarray],
    val_obs: np.ndarray) -> np.ndarray:
    """
    Calculate Scribe scores for each precursor (Searle, Shannon, Wilburn, 2023, PMID: 36695531)
    
    This function computes the Scribe score, which measures spectral similarity by comparing
    the normalized distribution of fragment ion intensities between predicted and observed spectra.
    Lower scores indicate better matches.
    
    Parameters
    ----------
    row_idx_split : List[np.ndarray]
        List of arrays containing row indices for each precursor's fragments.
    col_idx_split : List[np.ndarray]
        List of arrays containing column indices for each precursor (unused in calculation).
    prec_val_split : List[np.ndarray]
        List of arrays containing predicted intensity values for each precursor's fragments.
    val_obs : np.ndarray
        Array of observed intensity values.
        
    Returns
    -------
    np.ndarray
        Array of SCRIBE scores for each precursor, one score per precursor.
        
    Examples
    --------
    >>> # Two precursors with their fragment indices and intensities
    >>> row_idx_split = [np.array([0, 1, 2]), np.array([3, 4])]
    >>> col_idx_split = [np.array([0, 0, 0]), np.array([1, 1])]  # Not used
    >>> prec_val_split = [np.array([100.0, 200.0, 150.0]), np.array([300.0, 250.0])]
    >>> val_obs = np.array([120.0, 180.0, 160.0, 290.0, 260.0])
    >>> scores = get_scribe(row_idx_split, col_idx_split, prec_val_split, val_obs)
    >>> scores.shape
    (2,)
    >>> # Perfect match would give score near 0
    >>> all(scores >= 0)
    True
    """
    n = len(row_idx_split)
    if n > 0:
        #Sum of sqrt of predicted fragment intensities for each precursor/column
        h_sqrt_sum = np.zeros(n)
        #Sum of sqrt of observed fragment intensities for each precursor/column
        x_sqrt_sum = np.zeros(n)
        scribe_scores = np.zeros(n)
        for j in range(n):
            for (i, val) in zip(row_idx_split[j], prec_val_split[j]):
                h_sqrt_sum[j] += np.sqrt(val)
                x_sqrt_sum[j] += np.sqrt(val_obs[i])
        
        for j in range(n):
            for (i, val) in zip(row_idx_split[j], prec_val_split[j]):
                scribe_scores[j] += (
                    (np.sqrt(val)/h_sqrt_sum[j]) - 
                    (np.sqrt(val_obs[i])/x_sqrt_sum[j])
                )**2

        return scribe_scores
    else:
        return np.zeros(0)


def get_residuals(
    ref_sparse_val: List[np.ndarray],
    ref_sparse_row: List[np.ndarray],
    ref_sparse_col: List[np.ndarray],
    decoy_sparse_val: List[np.ndarray],
    decoy_sparse_row: List[np.ndarray],
    decoy_sparse_col: List[np.ndarray],
    val_obs: np.ndarray,
    coeffs: np.ndarray,
    ref_spec_offset: int,
    decoy_spec_offset: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate residuals (Ax - b) and prediction values for both reference and decoy data.
    
    This function computes the predicted values by multiplying sparse matrix representations 
    of reference and decoy data by the coefficient vector, then calculates residuals 
    as the difference between observed and predicted values.
    
    Parameters
    ----------
    ref_sparse_val : List[np.ndarray]
        List of arrays with sparse values for reference data.
    ref_sparse_row : List[np.ndarray]
        List of arrays with sparse row indices for reference data.
    ref_sparse_col : List[np.ndarray]
        List of arrays with sparse column indices for reference data.
    decoy_sparse_val : List[np.ndarray]
        List of arrays with sparse values for decoy data.
    decoy_sparse_row : List[np.ndarray]
        List of arrays with sparse row indices for decoy data.
    decoy_sparse_col : List[np.ndarray]
        List of arrays with sparse column indices for decoy data.
    val_obs : np.ndarray
        Observed values (the 'b' in Ax = b).
    coeffs : np.ndarray
        Coefficients from the fit (the 'x' in Ax = b).
    ref_spec_offset : int
        Column offset for reference spectra in coefficient array.
    decoy_spec_offset : int
        Column offset for decoy spectra in coefficient array.
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        - residuals: Residuals between observed and predicted values (b - Ax).
        - y_pred: Predicted values calculated as A*x.
        
    Examples
    --------
    >>> # Sparse matrix representation: one reference precursor, no decoys
    >>> ref_sparse_val = [np.array([2.0, 3.0])]
    >>> ref_sparse_row = [np.array([0, 1])]
    >>> ref_sparse_col = [np.array([0, 0])]
    >>> decoy_sparse_val = []
    >>> decoy_sparse_row = []
    >>> decoy_sparse_col = []
    >>> val_obs = np.array([5.0, 7.0])
    >>> coeffs = np.array([2.0])  # Single coefficient
    >>> residuals, y_pred = get_residuals(
    ...     ref_sparse_val, ref_sparse_row, ref_sparse_col,
    ...     decoy_sparse_val, decoy_sparse_row, decoy_sparse_col,
    ...     val_obs, coeffs, 0, 1)
    >>> y_pred  # Should be [2.0*2.0, 3.0*2.0] = [4.0, 6.0]
    array([4., 6.])
    >>> residuals  # Should be [5.0-4.0, 7.0-6.0] = [1.0, 1.0]
    array([1., 1.])
    """
    
    def _compute_prediction(sparse_val, sparse_row, sparse_col, coeff_array, y_pred, offset):
        """Helper function to compute predictions for a set of sparse data"""
        for j in range(len(sparse_row)):
            for row, col, val in zip(sparse_row[j], sparse_col[j], sparse_val[j]):
                y_pred[row] += val * coeff_array[col+offset]
        return y_pred
    
    coeffs = np.asarray(coeffs).ravel()
    N = len(val_obs)  # Number of rows in the sparse matrix (A)
    
    # Initialize prediction array
    y_pred = np.zeros(N)
    
    # Compute predictions for reference data
    y_pred = _compute_prediction(ref_sparse_val, ref_sparse_row, ref_sparse_col, coeffs, y_pred, ref_spec_offset)
    
    # Add predictions for decoy data
    y_pred = _compute_prediction(decoy_sparse_val, decoy_sparse_row, decoy_sparse_col, coeffs, y_pred, decoy_spec_offset)
    
    # Compute residuals
    #r = np.zeros_like(y_pred)
    
    # Residuals for matched peaks (where we have observations)
    r = val_obs - y_pred

    return r, y_pred


def max_matched_residual(
    row_idx_split: List[np.ndarray],
    residuals: np.ndarray
) -> np.ndarray:
    """
    Find the maximum residual for each precursor's matched peaks.
    
    This function finds the largest residual value among the matched peaks
    for each precursor, which can indicate the worst-fit fragment.
    
    NOTE: This function has a bug - it zips row indices with the full residuals array
    instead of using indices to access specific residuals. This results in only
    checking the first N residuals where N is the length of each precursor's indices.
    
    Parameters
    ----------
    row_idx_split : List[np.ndarray]
        List of arrays containing row indices for each precursor's fragments.
    residuals : np.ndarray
        Array of residuals between observed and predicted values.
        
    Returns
    -------
    np.ndarray
        Array of maximum residual values for each precursor.
        
    Examples
    --------
    >>> # Two precursors with different fragment indices
    >>> row_idx_split = [np.array([0, 1, 2]), np.array([3, 4])]
    >>> residuals = np.array([0.1, 0.3, 0.2, 0.5, 0.4])
    >>> max_residuals = max_matched_residual(row_idx_split, residuals)
    >>> max_residuals.shape
    (2,)
    >>> # Due to the bug, it checks first 3 residuals for precursor 1
    >>> # and first 2 residuals for precursor 2
    >>> max_residuals[0]  # max of residuals[0:3] = max([0.1, 0.3, 0.2])
    0.3
    >>> max_residuals[1]  # max of residuals[0:2] = max([0.1, 0.3])  
    0.3
    """
    n = len(row_idx_split)
    if n > 0:
        max_matched_residuals = np.zeros(n)
        for j in range(n):
            for (i, val) in zip(row_idx_split[j], residuals):
                if val > max_matched_residuals[j]:
                    max_matched_residuals[j] = val
        return max_matched_residuals
    else:
        return np.zeros(0)


def gof_stat(
    row_idx_split: List[np.ndarray],
    col_idx_split: List[np.ndarray],
    val_split: List[np.ndarray],
    residuals: np.ndarray,
    val_obs: np.ndarray,
    coeffs: np.ndarray,
    offset: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate goodness-of-fit statistics and maximum residuals for each precursor.
    
    This function computes several metrics to assess fit quality:
    1. Overall goodness-of-fit statistic based on sum of residuals to sum of fitted peaks
    2. Maximum residual for matched peaks (peaks with observed intensity > 1e-6)
    3. Maximum residual for unmatched peaks (peaks with observed intensity <= 1e-6)
    
    All metrics are log-transformed and normalized by the sum of fitted peaks.
    
    Parameters
    ----------
    row_idx_split : List[np.ndarray]
        List of arrays containing row indices for each precursor's fragments.
    col_idx_split : List[np.ndarray]
        List of arrays containing column indices for each precursor.
    val_split : List[np.ndarray]
        List of arrays containing predicted intensity values for each precursor's fragments.
    residuals : np.ndarray
        Array of residuals between observed and predicted values.
    val_obs : np.ndarray
        Array of observed intensity values.
    coeffs : np.ndarray
        Coefficients from the fit.
    offset : int
        Column offset for accessing coefficients.
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        - result: Goodness-of-fit score for each precursor (log2 of residuals/fitted).
        - max_unmatched_residuals: Maximum residual for unmatched peaks, normalized and log-transformed.
        - max_matched_residuals: Maximum residual for matched peaks, normalized and log-transformed.
        
    Examples
    --------
    >>> # Single precursor with 3 fragments
    >>> row_idx_split = [np.array([0, 1, 2])]
    >>> col_idx_split = [np.array([0, 0, 0])]
    >>> val_split = [np.array([100.0, 200.0, 150.0])]
    >>> residuals = np.array([10.0, -20.0, 15.0])
    >>> val_obs = np.array([110.0, 180.0, 1e-7])  # Last peak is "unmatched"
    >>> coeffs = np.array([1.0])
    >>> result, max_unmatched, max_matched = gof_stat(
    ...     row_idx_split, col_idx_split, val_split,
    ...     residuals, val_obs, coeffs, 0)
    >>> result.shape
    (1,)
    >>> # result[0] = log2(sum(|residuals|) / sum(fitted_peaks))
    >>> # where sum(|residuals|) = 10 + 20 + 15 = 45
    >>> # and sum(fitted_peaks) = 1.0*100 + 1.0*200 + 1.0*150 = 450
    >>> np.isclose(result[0], np.log2(45.0 / 450.0))
    True
    """
    coeffs = np.asarray(coeffs).ravel()
    n = len(row_idx_split)
    if n > 0:
        sum_of_residuals = np.zeros(n)
        sum_of_fitted_peaks = np.zeros(n)
        result = np.zeros(n)
        max_unmatched_residuals = np.zeros(n)
        max_matched_residuals = np.zeros(n)
        for j in range(n):
            max_unmatched_residual = 0.0
            max_matched_residual = 0.0
            for (row_idx, col_idx, val) in zip(row_idx_split[j], col_idx_split[j], val_split[j]):
                r = abs(residuals[row_idx])
                sum_of_residuals[j] += r
                sum_of_fitted_peaks[j] += abs(coeffs[col_idx+offset]*val)
                if (val_obs[row_idx] > 1e-6):
                    if r > max_matched_residual:
                        max_matched_residual = r
                elif (val_obs[row_idx] < 1e-6):
                    if r > max_unmatched_residual:
                        max_unmatched_residual = r
            max_unmatched_residuals[j] = max_unmatched_residual
            max_matched_residuals[j] = max_matched_residual

        #Handle bad values         
        for j in range(n):
            if sum_of_fitted_peaks[j] == 0:
                sum_of_fitted_peaks[j] = 1e-6
            if sum_of_residuals[j] == 0:
                sum_of_residuals[j] = 1e-6  # Perfect agreement (no residuals, no signal)
            result[j] = np.log2(sum_of_residuals[j] / sum_of_fitted_peaks[j])
            max_matched_residuals[j] = np.log2(max_matched_residuals[j]/(sum_of_fitted_peaks[j] + 1e-10) + 1e-10)
            max_unmatched_residuals[j] = np.log2(max_unmatched_residuals[j]/(sum_of_fitted_peaks[j] + 1e-10) + 1e-10)
        return result, max_unmatched_residuals, max_matched_residuals 
    else:
        return np.zeros(0), np.zeros(0), np.zeros(0)


def get_manhattan_distance(
    row_idx_split: List[np.ndarray],
    col_idx_split: List[np.ndarray],
    prec_val_split: List[np.ndarray],
    val_obs: np.ndarray,
    y_pred: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate fit metrics between predicted and observed fragment intensity values.
    
    This function computes two metrics for each precursor:
    1. Modified Manhattan distance: Sum of absolute differences between predicted and observed 
       values, normalized by sum of observed values and log-transformed. Higher (less negative) 
       values indicate better fits.
    2. Spectral contrast angle: Spectral contrast between model (Ax) and observed (b) intensities for 
    the fragments matching each respective precursor
    
    Parameters
    ----------
    row_idx_split : List[np.ndarray]
        List of arrays containing row indices for each precursor's fragments.
    col_idx_split : List[np.ndarray]
        List of arrays containing column indices for each precursor (unused).
    prec_val_split : List[np.ndarray]
        List of arrays containing predicted intensity values for each precursor's fragments (unused).
    val_obs : np.ndarray
        Array of observed intensity values.
    y_pred : np.ndarray
        Array of predicted intensity values after applying model coefficients.
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        - manhattan_distances: Array of modified Manhattan distances for each precursor, 
          with higher values indicating better fits.
        - fitted_spectral_contrast: Array of spectral contrast values for each precursor (0-1).
    
    Notes
    -----
    - Edge cases are handled: when sum of observed values is zero (bad fit) or 
      Manhattan distance is zero (perfect fit).
    - The col_idx_split and prec_val_split parameters are not used in the current implementation.
      
    Examples
    --------
    >>> # Two precursors with their fragment indices
    >>> row_idx_split = [np.array([0, 1]), np.array([2, 3])]
    >>> col_idx_split = [np.array([0, 0]), np.array([1, 1])]  # Not used
    >>> prec_val_split = [np.array([100.0, 200.0]), np.array([150.0, 250.0])]  # Not used
    >>> val_obs = np.array([100.0, 200.0, 150.0, 250.0])
    >>> y_pred = np.array([95.0, 205.0, 145.0, 255.0])
    >>> distances, contrasts = get_manhattan_distance(
    ...     row_idx_split, col_idx_split, prec_val_split, val_obs, y_pred)
    >>> distances.shape
    (2,)
    >>> # Manhattan distance for precursor 1: |95-100| + |205-200| = 10
    >>> # Normalized by sum of observed (300), then -log2(10/300)
    >>> np.isclose(distances[0], -np.log2(10.0 / 300.0))
    True
    >>> # Spectral contrast is between 0 and 1
    >>> all(0 <= c <= 1 for c in contrasts)
    True
    """
    n = len(row_idx_split)
    N = len(val_obs)
    if (n > 0) & (N > 0):
        manhattan_distances = np.zeros(n)
        fitted_spectral_contrast = np.zeros(n)

        x_sums = np.zeros(n)
        
        for j in range(n):
            u2_sum, v2_sum, uv_sum = 0.0, 0.0, 0.0
            for i, row in enumerate(row_idx_split[j]):
                # Sum observed intensities for normalization
                x_sums[j] += val_obs[row]
                # Calculate Manhattan distance using predicted values
                manhattan_distances[j] += abs(y_pred[row] - val_obs[row])
                u2_sum += y_pred[row]**2 
                v2_sum += val_obs[row]**2
                uv_sum += y_pred[row] * val_obs[row]
            # Normalize and transform

            if x_sums[j] > 0 and manhattan_distances[j] > 0:
                manhattan_distances[j] = -np.log2(manhattan_distances[j] / x_sums[j])
                fitted_spectral_contrast[j] = np.sqrt(uv_sum)/(np.sqrt(u2_sum) * np.sqrt(v2_sum) + 1e-10)
            else:
                # Handle edge cases
                if x_sums[j] == 0:
                    manhattan_distances[j] = np.finfo(np.float32).max  # Bad fit
                    fitted_spectral_contrast[j] = 0.0
                else:  # manhattan_distances[j] == 0
                    manhattan_distances[j] = np.finfo(np.float32).min  # Perfect fit
                    fitted_spectral_contrast[j] = np.sqrt(uv_sum)/(np.sqrt(u2_sum) * np.sqrt(v2_sum) + 1e-10)
                
        return manhattan_distances, fitted_spectral_contrast
    else:
        return np.zeros(0), np.zeros(0)