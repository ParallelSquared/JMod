"""
Fragment ion features for spectral matching.

This module contains functions to calculate features related to
fragment ion matches, particularly b and y ions.
"""

import numpy as np
from typing import List, Tuple, Dict, Any


def count_b_y_ions(fragment_names: np.ndarray) -> Tuple[int, int]:
    """
    Count the number of b and y ions in matched fragments.
    
    Args:
        fragment_names: Array of fragment ion names
        
    Returns:
        Tuple of (b_count, y_count)
    """
    b_count = sum(1 for name in fragment_names if str(name).startswith('b'))
    y_count = sum(1 for name in fragment_names if str(name).startswith('y'))
    return b_count, y_count


def calculate_hyperscore(
    fragment_names: np.ndarray,
    fragment_intensities: np.ndarray
) -> float:
    """
    Calculate hyperscore based on b and y ion matches.
    
    Simplified version of the X!Tandem hyperscore.
    
    Args:
        fragment_names: Array of fragment ion names
        fragment_intensities: Array of fragment intensities
        
    Returns:
        Hyperscore value
    """
    b_count, y_count = count_b_y_ions(fragment_names)
    
    if b_count + y_count == 0:
        return 0.0
    
    # Simplified hyperscore: log(factorial(b) * factorial(y) * sum(intensities))
    # Using Stirling's approximation for large factorials
    b_score = b_count * np.log(b_count) - b_count if b_count > 0 else 0
    y_score = y_count * np.log(y_count) - y_count if y_count > 0 else 0
    intensity_score = np.log(np.sum(fragment_intensities) + 1)
    
    return b_score + y_score + intensity_score


def find_longest_y_series(fragment_names: np.ndarray) -> int:
    """
    Find the longest consecutive y-ion series.
    
    Args:
        fragment_names: Array of fragment ion names
        
    Returns:
        Length of longest y-ion series
    """
    # Extract y-ion numbers
    y_ions = []
    for name in fragment_names:
        if str(name).startswith('y'):
            try:
                # Extract number from y-ion name (e.g., 'y7' -> 7)
                num = int(''.join(filter(str.isdigit, str(name))))
                y_ions.append(num)
            except:
                pass
    
    if not y_ions:
        return 0
    
    # Sort and find longest consecutive sequence
    y_ions = sorted(set(y_ions))
    
    max_length = 1
    current_length = 1
    
    for i in range(1, len(y_ions)):
        if y_ions[i] == y_ions[i-1] + 1:
            current_length += 1
            max_length = max(max_length, current_length)
        else:
            current_length = 1
    
    return max_length


def calculate_fragment_features(
    frag_names: List[np.ndarray],
    lib_peaks_matched: List[np.ndarray],
    library: Dict[Any, Dict],
    candidates: List[Any]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate all fragment-based features.
    
    Args:
        frag_names: Fragment names for each candidate
        lib_peaks_matched: Boolean arrays of matched peaks
        library: Spectral library
        candidates: List of candidate identifiers
        
    Returns:
        Tuple of (hyperscores, b_counts, y_counts, longest_y_ions)
    """
    n_candidates = len(frag_names)
    hyperscores = np.zeros(n_candidates)
    b_counts = np.zeros(n_candidates)
    y_counts = np.zeros(n_candidates)
    longest_y_ions = np.zeros(n_candidates)
    
    for i in range(n_candidates):
        if i < len(frag_names) and len(frag_names[i]) > 0:
            # Count b and y ions
            b_count, y_count = count_b_y_ions(frag_names[i])
            b_counts[i] = b_count
            y_counts[i] = y_count
            
            # Calculate hyperscore if we have intensity information
            if i < len(candidates) and candidates[i] in library:
                lib_entry = library[candidates[i]]
                if 'spectrum' in lib_entry and i < len(lib_peaks_matched):
                    matched_intensities = lib_entry['spectrum'][:, 1][lib_peaks_matched[i]]
                    hyperscores[i] = calculate_hyperscore(frag_names[i], matched_intensities)
            
            # Find longest y-ion series
            longest_y_ions[i] = find_longest_y_series(frag_names[i])
    
    return hyperscores, b_counts, y_counts, longest_y_ions