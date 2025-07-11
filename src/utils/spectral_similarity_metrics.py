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
from scipy import sparse
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


def get_scribe_csc(
    sparse_lib_matrix_csc: sparse.csc_matrix,
    dia_spectrum_intensities: np.ndarray
) -> np.ndarray:
    """
    Calculate Scribe scores for each precursor using CSC sparse matrix operations.
    
    This function computes the Scribe score using efficient sparse matrix operations
    instead of split arrays. It processes each candidate (column) independently to
    calculate the normalized distribution differences between predicted and observed spectra.
    
    Args:
        sparse_lib_matrix_csc: Sparse matrix in CSC format with shape (n_dia_peaks, n_candidates)
                              where non-zero values are library intensities at matched peaks
        dia_spectrum_intensities: Array of observed intensity values from DIA spectrum
        
    Returns:
        np.ndarray: Array of SCRIBE scores for each candidate, one score per candidate.
                   Lower scores indicate better matches.
                   
    Examples:
        >>> # Create a simple sparse matrix for testing
        >>> from scipy import sparse
        >>> # Matrix with 2 candidates and 5 peaks
        >>> row_indices = [0, 1, 2, 3, 4]
        >>> col_indices = [0, 0, 0, 1, 1] 
        >>> values = [1.0, 2.0, 3.0, 4.0, 5.0]
        >>> matrix = sparse.csc_matrix((values, (row_indices, col_indices)), shape=(5, 2))
        >>> dia_intensities = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
        >>> scores = get_scribe_csc(matrix, dia_intensities)
        >>> scores.shape
        (2,)
        >>> all(scores >= 0)
        True
    """
    # Handle empty matrix case
    if sparse_lib_matrix_csc.shape[1] == 0:
        return np.array([], dtype=np.float64)
    
    n_candidates = sparse_lib_matrix_csc.shape[1]
    scribe_scores = np.zeros(n_candidates, dtype=np.float64)
    
    # Process each candidate (column) in the sparse matrix
    for col_idx in range(n_candidates):
        # Extract the column for this candidate
        col_data = sparse_lib_matrix_csc[:, col_idx]
        
        # Get non-zero entries (matched peaks)
        nonzero_rows, _ = col_data.nonzero()
        
        if len(nonzero_rows) == 0:
            # No matched peaks, SCRIBE score is 0
            scribe_scores[col_idx] = 0.0
            continue
            
        # Extract predicted intensities (library values) for this candidate
        predicted_intensities = np.array([col_data[row, 0] for row in nonzero_rows])
        
        # Extract corresponding observed intensities from DIA spectrum
        observed_intensities = dia_spectrum_intensities[nonzero_rows]
        
        # Calculate sqrt sums for normalization
        h_sqrt_sum = np.sum(np.sqrt(predicted_intensities))  # Sum of sqrt(predicted)
        x_sqrt_sum = np.sum(np.sqrt(observed_intensities))   # Sum of sqrt(observed)
        
        # Avoid division by zero
        if h_sqrt_sum == 0 or x_sqrt_sum == 0:
            scribe_scores[col_idx] = 0.0
            continue
            
        # Calculate SCRIBE score: sum of squared differences of normalized sqrt intensities
        normalized_predicted = np.sqrt(predicted_intensities) / h_sqrt_sum
        normalized_observed = np.sqrt(observed_intensities) / x_sqrt_sum
        
        scribe_scores[col_idx] = np.sum((normalized_predicted - normalized_observed) ** 2)
    
    return scribe_scores


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


def get_residuals_csc(
    sparse_lib_matrix_csc: sparse.csc_matrix,
    dia_spectrum_intensities: np.ndarray,
    lib_coefficients: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate residuals (Ax - b) and prediction values using CSC sparse matrix operations.
    
    This function computes the predicted values by multiplying a CSC sparse matrix 
    by the coefficient vector, then calculates residuals as the difference between 
    observed and predicted values. This is an optimized version of get_residuals
    that uses direct sparse matrix operations instead of split arrays.
    
    Parameters
    ----------
    sparse_lib_matrix_csc : sparse.csc_matrix
        Sparse matrix in CSC format with shape (n_peaks, n_candidates) where
        non-zero values are library intensities at matched peaks.
    dia_spectrum_intensities : np.ndarray
        Array of observed intensity values from DIA spectrum (the 'b' in Ax = b).
    lib_coefficients : np.ndarray
        Coefficients from the fit (the 'x' in Ax = b).
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        - residuals: Residuals between observed and predicted values (b - Ax).
        - y_pred: Predicted values calculated as A*x.
        
    Examples
    --------
    >>> from scipy import sparse
    >>> import numpy as np
    >>> # Create a simple sparse matrix for testing
    >>> row_indices = [0, 1, 2]
    >>> col_indices = [0, 0, 1] 
    >>> values = [2.0, 3.0, 1.5]
    >>> matrix = sparse.csc_matrix((values, (row_indices, col_indices)), shape=(3, 2))
    >>> dia_intensities = np.array([5.0, 7.5, 3.0])
    >>> coeffs = np.array([2.0, 1.5])
    >>> residuals, y_pred = get_residuals_csc(matrix, dia_intensities, coeffs)
    >>> y_pred  # Should be [2.0*2.0, 3.0*2.0, 1.5*1.5] = [4.0, 6.0, 2.25]
    array([4.  , 6.  , 2.25])
    >>> residuals  # Should be [5.0-4.0, 7.5-6.0, 3.0-2.25] = [1.0, 1.5, 0.75]
    array([1.  , 1.5 , 0.75])
    
    Notes
    -----
    This function is designed for use in RT alignment where only reference data
    is processed (no decoy data). For cases with both reference and decoy data,
    use the original get_residuals function.
    
    The CSC (Compressed Sparse Column) format is optimal for matrix-vector
    multiplication operations used in this function.
    """
    # Ensure coefficients are in the correct format
    coeffs = np.asarray(lib_coefficients).ravel()
    
    # Handle empty matrix case
    if sparse_lib_matrix_csc.shape[1] == 0 or len(coeffs) == 0:
        return np.zeros_like(dia_spectrum_intensities), np.zeros_like(dia_spectrum_intensities)
    
    # Ensure coefficient array matches matrix dimensions
    n_candidates = sparse_lib_matrix_csc.shape[1]
    if len(coeffs) < n_candidates:
        # Pad with zeros if needed
        coeffs_padded = np.zeros(n_candidates)
        coeffs_padded[:len(coeffs)] = coeffs
        coeffs = coeffs_padded
    elif len(coeffs) > n_candidates:
        # Truncate if too long
        coeffs = coeffs[:n_candidates]
    
    # Calculate predicted values using sparse matrix multiplication
    # y_pred = A * x where A is the sparse matrix and x is coefficients
    y_pred = sparse_lib_matrix_csc @ coeffs
    
    # Ensure y_pred has the same length as observed values
    if len(y_pred) != len(dia_spectrum_intensities):
        # This shouldn't happen with proper matrix construction, but handle gracefully
        if len(y_pred) < len(dia_spectrum_intensities):
            y_pred_padded = np.zeros(len(dia_spectrum_intensities))
            y_pred_padded[:len(y_pred)] = y_pred
            y_pred = y_pred_padded
        else:
            y_pred = y_pred[:len(dia_spectrum_intensities)]
    
    # Calculate residuals: observed - predicted
    residuals = dia_spectrum_intensities - y_pred
    
    return residuals, y_pred


def max_matched_residual(
    row_idx_split: List[np.ndarray],
    residuals: np.ndarray
) -> np.ndarray:
    """
    Find the maximum residual for each precursor's matched peaks.
    
    This function finds the largest residual value among the matched peaks
    for each precursor, which can indicate the worst-fit fragment.
    
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
    >>> # Precursor 1 uses indices [0, 1, 2] -> residuals [0.1, 0.3, 0.2]
    >>> max_residuals[0]  # max([0.1, 0.3, 0.2])
    0.3
    >>> # Precursor 2 uses indices [3, 4] -> residuals [0.5, 0.4]
    >>> max_residuals[1]  # max([0.5, 0.4])
    0.5
    """
    n = len(row_idx_split)
    if n > 0:
        max_matched_residuals = np.full(n, -np.inf)
        for j in range(n):
            for i in row_idx_split[j]:
                val = residuals[i]
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
                fitted_spectral_contrast[j] = uv_sum/(np.sqrt(u2_sum) * np.sqrt(v2_sum) + 1e-10)
            else:
                # Handle edge cases
                if x_sums[j] == 0:
                    manhattan_distances[j] = np.finfo(np.float32).max  # Bad fit
                    fitted_spectral_contrast[j] = 0.0
                else:  # manhattan_distances[j] == 0
                    manhattan_distances[j] = np.finfo(np.float32).min  # Perfect fit
                    fitted_spectral_contrast[j] = uv_sum/(np.sqrt(u2_sum) * np.sqrt(v2_sum) + 1e-10)
                
        return manhattan_distances, fitted_spectral_contrast
    else:
        return np.zeros(0), np.zeros(0)


def get_manhattan_distance_csc(
    sparse_lib_matrix_csc: sparse.csc_matrix,
    dia_spectrum_intensities: np.ndarray,
    y_pred: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate Manhattan distance and spectral contrast using CSC sparse matrix operations.
    
    This function computes Manhattan distance and spectral contrast metrics for each 
    candidate using efficient sparse matrix operations instead of split arrays. It 
    processes each candidate (column) independently to calculate the fit quality metrics.
    
    Parameters
    ----------
    sparse_lib_matrix_csc : sparse.csc_matrix
        Sparse matrix in CSC format with shape (n_peaks, n_candidates) where
        non-zero values indicate which peaks are matched for each candidate.
    dia_spectrum_intensities : np.ndarray
        Array of observed intensity values from DIA spectrum.
    y_pred : np.ndarray
        Array of predicted intensity values after applying model coefficients.
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        - manhattan_distances: Array of modified Manhattan distances for each candidate,
          with higher values indicating better fits.
        - fitted_spectral_contrast: Array of spectral contrast values for each candidate (0-1).
        
    Examples
    --------
    >>> from scipy import sparse
    >>> import numpy as np
    >>> # Create a simple sparse matrix for testing
    >>> row_indices = [0, 1, 2, 3]
    >>> col_indices = [0, 0, 1, 1]
    >>> values = [1.0, 1.0, 1.0, 1.0]  # Values don't matter, just structure
    >>> matrix = sparse.csc_matrix((values, (row_indices, col_indices)), shape=(4, 2))
    >>> dia_intensities = np.array([1.0, 2.0, 3.0, 4.0])
    >>> y_pred = np.array([1.1, 2.1, 2.9, 4.1])
    >>> distances, contrasts = get_manhattan_distance_csc(matrix, dia_intensities, y_pred)
    >>> distances.shape
    (2,)
    >>> contrasts.shape
    (2,)
    >>> # Manhattan distances should be finite and contrasts in [0,1]
    >>> all(np.isfinite(distances))
    True
    >>> all(0 <= c <= 1 for c in contrasts)
    True
    
    Notes
    -----
    This function is designed for use in RT alignment where only reference data
    is processed. It calculates:
    
    1. Manhattan distance: -log2(sum(|y_pred - observed|) / sum(observed))
       Higher values indicate better fits (less error relative to signal)
       
    2. Spectral contrast: Cosine similarity between predicted and observed intensities
       Values range from 0 (no correlation) to 1 (perfect correlation)
       
    Edge cases:
    - Zero observed intensities: Manhattan distance = max float (bad fit)
    - Perfect prediction: Manhattan distance = min float (perfect fit)
    - Zero norms: Spectral contrast = 0.0
    """
    # Handle empty matrix case
    if sparse_lib_matrix_csc.shape[1] == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    
    n_candidates = sparse_lib_matrix_csc.shape[1]
    manhattan_distances = np.zeros(n_candidates, dtype=np.float64)
    fitted_spectral_contrasts = np.zeros(n_candidates, dtype=np.float64)
    
    # Process each candidate (column) in the sparse matrix
    for col_idx in range(n_candidates):
        # Extract the column for this candidate
        col_data = sparse_lib_matrix_csc[:, col_idx]
        
        # Get non-zero entries (matched peaks for this candidate)
        nonzero_rows, _ = col_data.nonzero()
        
        if len(nonzero_rows) == 0:
            # No matched peaks, set to bad fit values
            manhattan_distances[col_idx] = np.finfo(np.float32).max
            fitted_spectral_contrasts[col_idx] = 0.0
            continue
        
        # Extract predicted and observed intensities for matched peaks
        pred_intensities = y_pred[nonzero_rows]
        obs_intensities = dia_spectrum_intensities[nonzero_rows]
        
        # Calculate sum of observed intensities for normalization
        obs_sum = np.sum(obs_intensities)
        
        # Calculate Manhattan distance
        manhattan_distance_raw = np.sum(np.abs(pred_intensities - obs_intensities))
        
        # Calculate spectral contrast (cosine similarity)
        pred_norm_sq = np.sum(pred_intensities ** 2)
        obs_norm_sq = np.sum(obs_intensities ** 2)
        dot_product = np.sum(pred_intensities * obs_intensities)
        
        # Handle edge cases for Manhattan distance
        if obs_sum > 0 and manhattan_distance_raw > 0:
            # Normal case: normalize and log-transform
            manhattan_distances[col_idx] = -np.log2(manhattan_distance_raw / obs_sum)
        elif obs_sum == 0:
            # Bad fit: no observed signal
            manhattan_distances[col_idx] = np.finfo(np.float32).max
        else:  # manhattan_distance_raw == 0
            # Perfect fit: prediction exactly matches observation
            manhattan_distances[col_idx] = np.finfo(np.float32).min
        
        # Handle spectral contrast calculation
        denominator = np.sqrt(pred_norm_sq * obs_norm_sq)
        if denominator > 1e-10:
            fitted_spectral_contrasts[col_idx] = dot_product / denominator
        else:
            # One or both vectors have zero norm
            fitted_spectral_contrasts[col_idx] = 0.0
        
        # Ensure spectral contrast is in valid range [0, 1]
        fitted_spectral_contrasts[col_idx] = max(0.0, min(1.0, fitted_spectral_contrasts[col_idx]))
    
    return manhattan_distances, fitted_spectral_contrasts


def gof_stat_csc(
    sparse_lib_matrix_csc: sparse.csc_matrix,
    residuals: np.ndarray,
    dia_spectrum_intensities: np.ndarray,
    lib_coefficients: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate goodness-of-fit statistics using CSC sparse matrix operations.
    
    This function computes goodness-of-fit metrics for each candidate using efficient 
    sparse matrix operations instead of split arrays. It calculates the same three 
    metrics as the original gof_stat function:
    1. Overall goodness-of-fit statistic based on sum of residuals to sum of fitted peaks
    2. Maximum residual for matched peaks (peaks with observed intensity > 1e-6)
    3. Maximum residual for unmatched peaks (peaks with observed intensity <= 1e-6)
    
    All metrics are log-transformed and normalized by the sum of fitted peaks.
    
    Parameters
    ----------
    sparse_lib_matrix_csc : sparse.csc_matrix
        Sparse matrix in CSC format with shape (n_peaks, n_candidates) where
        non-zero values are library intensities at matched peaks.
    residuals : np.ndarray
        Array of residuals between observed and predicted values.
    dia_spectrum_intensities : np.ndarray
        Array of observed intensity values from DIA spectrum.
    lib_coefficients : np.ndarray
        Coefficients from the fit.
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        - result: Goodness-of-fit score for each candidate (log2 of residuals/fitted).
        - max_unmatched_residuals: Maximum residual for unmatched peaks, normalized and log-transformed.
        - max_matched_residuals: Maximum residual for matched peaks, normalized and log-transformed.
        
    Examples
    --------
    >>> from scipy import sparse
    >>> import numpy as np
    >>> # Create a simple sparse matrix for testing
    >>> row_indices = [0, 1, 2]
    >>> col_indices = [0, 0, 1]
    >>> values = [100.0, 200.0, 150.0]
    >>> matrix = sparse.csc_matrix((values, (row_indices, col_indices)), shape=(3, 2))
    >>> residuals = np.array([10.0, -20.0, 15.0])
    >>> dia_intensities = np.array([110.0, 180.0, 1e-7])  # Last peak is "unmatched"
    >>> coeffs = np.array([1.0, 1.0])
    >>> result, max_unmatched, max_matched = gof_stat_csc(
    ...     matrix, residuals, dia_intensities, coeffs)
    >>> result.shape
    (2,)
    >>> max_unmatched.shape
    (2,)
    >>> max_matched.shape
    (2,)
    >>> # Results should be finite and log-transformed
    >>> all(np.isfinite(result))
    True
    
    Notes
    -----
    This function is designed for use in RT alignment where only reference data
    is processed. It uses the sparse matrix structure to efficiently identify
    which peaks belong to each candidate and calculate fit statistics.
    
    The threshold of 1e-6 is used to distinguish between matched and unmatched peaks
    based on their observed intensities, following the original implementation.
    """
    # Ensure coefficients are in the correct format
    coeffs = np.asarray(lib_coefficients).ravel()
    
    # Handle empty matrix case
    if sparse_lib_matrix_csc.shape[1] == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    
    n_candidates = sparse_lib_matrix_csc.shape[1]
    
    # Initialize output arrays
    result = np.zeros(n_candidates, dtype=np.float64)
    max_unmatched_residuals = np.zeros(n_candidates, dtype=np.float64)
    max_matched_residuals = np.zeros(n_candidates, dtype=np.float64)
    
    # Process each candidate (column) in the sparse matrix
    for col_idx in range(n_candidates):
        # Extract the column for this candidate
        col_data = sparse_lib_matrix_csc[:, col_idx]
        
        # Get non-zero entries (matched peaks for this candidate)
        nonzero_rows, _ = col_data.nonzero()
        
        if len(nonzero_rows) == 0:
            # No matched peaks, set to default values
            result[col_idx] = np.log2(1e-6)  # Perfect agreement (no residuals, no signal)
            max_unmatched_residuals[col_idx] = np.log2(1e-10)
            max_matched_residuals[col_idx] = np.log2(1e-10)
            continue
        
        # Get coefficient for this candidate
        if col_idx < len(coeffs):
            coeff = coeffs[col_idx]
        else:
            coeff = 0.0
        
        # Initialize accumulators
        sum_of_residuals = 0.0
        sum_of_fitted_peaks = 0.0
        max_unmatched_residual = 0.0
        max_matched_residual = 0.0
        
        # Process each matched peak for this candidate
        for row_idx in nonzero_rows:
            # Get library intensity value for this peak
            lib_intensity = col_data[row_idx, 0]
            
            # Calculate absolute residual
            abs_residual = abs(residuals[row_idx])
            sum_of_residuals += abs_residual
            
            # Calculate fitted peak intensity
            fitted_peak = abs(coeff * lib_intensity)
            sum_of_fitted_peaks += fitted_peak
            
            # Determine if this is a matched or unmatched peak based on observed intensity
            observed_intensity = dia_spectrum_intensities[row_idx]
            
            if observed_intensity > 1e-6:
                # Matched peak
                if abs_residual > max_matched_residual:
                    max_matched_residual = abs_residual
            else:
                # Unmatched peak
                if abs_residual > max_unmatched_residual:
                    max_unmatched_residual = abs_residual
        
        # Handle edge cases and calculate final metrics
        if sum_of_fitted_peaks == 0:
            sum_of_fitted_peaks = 1e-6
        if sum_of_residuals == 0:
            sum_of_residuals = 1e-6  # Perfect agreement (no residuals, no signal)
        
        # Calculate goodness-of-fit statistic
        result[col_idx] = np.log2(sum_of_residuals / sum_of_fitted_peaks)
        
        # Calculate normalized and log-transformed maximum residuals
        max_matched_residuals[col_idx] = np.log2(max_matched_residual / (sum_of_fitted_peaks + 1e-10) + 1e-10)
        max_unmatched_residuals[col_idx] = np.log2(max_unmatched_residual / (sum_of_fitted_peaks + 1e-10) + 1e-10)
    
    return result, max_unmatched_residuals, max_matched_residuals