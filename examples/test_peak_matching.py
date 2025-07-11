#!/usr/bin/env python
"""
Example script demonstrating the new peak matching functionality.

This shows how to use the efficient O(n+m) peak matching algorithm
inspired by Pioneer.jl's matchPeaks.jl.
"""

import numpy as np
import time
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.peak_matching import (
    MassErrorModel, match_peaks, prepare_fragments_for_matching,
    DETAILED_FRAG_DTYPE, FRAGMENT_MATCH_DTYPE
)


def demo_basic_matching():
    """Demonstrate basic peak matching."""
    print("=== Basic Peak Matching Demo ===\n")
    
    # Create a mass error model (10 ppm tolerance)
    mass_model = MassErrorModel(tolerance_ppm=10.0, use_ppm=True)
    
    # Create a mock library entry
    library_entry = {
        'spectrum': np.array([
            [147.1128, 100.0],   # y1
            [246.1812, 80.0],    # y2
            [359.2653, 120.0],   # y3
            [506.3337, 150.0],   # y4
            [175.1190, 50.0],    # b2
            [288.2030, 60.0],    # b3
        ]),
        'ordered_frags': ['y1', 'y2', 'y3', 'y4', 'b2', 'b3']
    }
    
    # Convert to fragment array
    fragments = prepare_fragments_for_matching(library_entry, prec_id=1001, prec_charge=2)
    print(f"Prepared {len(fragments)} fragments from library")
    
    # Create a mock DIA spectrum with some matching peaks
    dia_masses = np.array([
        100.0,      # noise
        147.1130,   # matches y1 (within 10 ppm)
        200.0,      # noise
        246.1815,   # matches y2 (within 10 ppm)
        288.2035,   # matches b3 (within 10 ppm)
        350.0,      # noise
        400.0,      # noise
        506.3340,   # matches y4 (within 10 ppm)
        600.0       # noise
    ])
    
    dia_intensities = np.array([
        10.0,   # noise
        95.0,   # y1
        15.0,   # noise
        75.0,   # y2
        55.0,   # b3
        20.0,   # noise
        25.0,   # noise
        140.0,  # y4
        30.0    # noise
    ])
    
    # Perform matching
    matched, unmatched = match_peaks(
        fragments, dia_masses, dia_intensities, mass_model,
        scan_idx=100, ms_file_idx=1
    )
    
    print(f"\nMatching results:")
    print(f"  Matched: {len(matched)} fragments")
    print(f"  Unmatched: {len(unmatched)} fragments")
    
    # Display matched fragments
    print("\nMatched fragments:")
    print(f"{'Fragment':<10} {'Theo m/z':<10} {'Obs m/z':<10} {'Error (ppm)':<12} {'Intensity':<10}")
    print("-" * 60)
    
    for match in matched:
        error_ppm = (match['match_mz'] - match['theoretical_mz']) / match['theoretical_mz'] * 1e6
        # Determine fragment type
        if match['ion_type'] == 2:
            frag_type = f"y{match['frag_index']}"
        elif match['ion_type'] == 1:
            frag_type = f"b{match['frag_index']}"
        else:
            frag_type = "unknown"
            
        print(f"{frag_type:<10} {match['theoretical_mz']:<10.4f} "
              f"{match['match_mz']:<10.4f} {error_ppm:<12.2f} {match['intensity']:<10.1f}")
    
    # Display unmatched fragments
    print("\nUnmatched fragments:")
    for unmatch in unmatched:
        if unmatch['ion_type'] == 2:
            frag_type = f"y{unmatch['frag_index']}"
        elif unmatch['ion_type'] == 1:
            frag_type = f"b{unmatch['frag_index']}"
        else:
            frag_type = "unknown"
        print(f"  {frag_type}: {unmatch['theoretical_mz']:.4f} m/z")


def demo_performance_comparison():
    """Compare performance with naive O(n*m) approach."""
    print("\n\n=== Performance Comparison Demo ===\n")
    
    # Create large test datasets
    n_fragments = 5000
    n_peaks = 5000
    
    print(f"Generating {n_fragments} fragments and {n_peaks} peaks...")
    
    # Random but sorted m/z values
    np.random.seed(42)
    frag_mz = np.sort(np.random.uniform(100, 2000, n_fragments))
    peak_mz = np.sort(np.random.uniform(100, 2000, n_peaks))
    peak_int = np.random.uniform(10, 1000, n_peaks)
    
    # Create fragment array
    fragments = np.zeros(n_fragments, dtype=DETAILED_FRAG_DTYPE)
    for i, mz in enumerate(frag_mz):
        fragments[i] = (
            1001,     # prec_id
            mz,       # mz
            100.0,    # intensity
            2,        # ion_type (y)
            True,     # is_y
            False,    # is_b
            False,    # is_p
            False,    # is_isotope
            1,        # frag_charge
            min(i + 1, 255),  # ion_position
            2,        # prec_charge
            min(i, 255),      # rank
            0         # sulfur_count
        )
    
    mass_model = MassErrorModel(tolerance_ppm=10.0)
    
    # Time the efficient algorithm
    print("\nTiming efficient O(n+m) algorithm...")
    start = time.time()
    matched, unmatched = match_peaks(fragments, peak_mz, peak_int, mass_model)
    efficient_time = time.time() - start
    
    print(f"  Time: {efficient_time:.3f}s")
    print(f"  Matched: {len(matched)}")
    print(f"  Unmatched: {len(unmatched)}")
    
    # Naive O(n*m) approach for comparison
    print("\nTiming naive O(n*m) approach...")
    start = time.time()
    naive_matches = 0
    
    for frag in fragments:
        low, high = mass_model.get_mz_bounds(frag['mz'])
        best_idx = -1
        best_error = float('inf')
        
        # Check all peaks (naive)
        for j, peak in enumerate(peak_mz):
            if low <= peak <= high:
                error = abs(peak - frag['mz'])
                if error < best_error:
                    best_error = error
                    best_idx = j
        
        if best_idx >= 0:
            naive_matches += 1
    
    naive_time = time.time() - start
    
    print(f"  Time: {naive_time:.3f}s")
    print(f"  Matched: {naive_matches}")
    
    print(f"\nSpeedup: {naive_time/efficient_time:.1f}x")


def demo_complex_matching():
    """Demonstrate handling of complex matching scenarios."""
    print("\n\n=== Complex Matching Scenarios Demo ===\n")
    
    mass_model = MassErrorModel(tolerance_ppm=20.0)  # Wider tolerance
    
    # Scenario 1: Multiple peaks within tolerance
    print("Scenario 1: Multiple peaks within tolerance of one fragment")
    fragments = np.zeros(1, dtype=DETAILED_FRAG_DTYPE)
    fragments[0] = (1001, 500.0, 100.0, 2, True, False, False, False, 1, 3, 2, 0, 0)
    
    # Three peaks all within 20 ppm of 500.0
    peaks_mz = np.array([499.99, 500.0, 500.01])
    peaks_int = np.array([80.0, 100.0, 90.0])
    
    matched, _ = match_peaks(fragments, peaks_mz, peaks_int, mass_model)
    
    print(f"  Fragment at {fragments[0]['mz']:.2f} m/z")
    print(f"  Peaks: {peaks_mz}")
    print(f"  Best match: {matched[0]['match_mz']:.2f} m/z (exact match selected)")
    
    # Scenario 2: Overlapping tolerance windows
    print("\nScenario 2: Fragments with overlapping tolerance windows")
    fragments = np.zeros(2, dtype=DETAILED_FRAG_DTYPE)
    fragments[0] = (1001, 500.0, 100.0, 2, True, False, False, False, 1, 3, 2, 0, 0)
    fragments[1] = (1001, 500.005, 100.0, 2, True, False, False, False, 1, 4, 2, 1, 0)
    
    # One peak between the two fragments
    peaks_mz = np.array([500.0025])
    peaks_int = np.array([100.0])
    
    matched, unmatched = match_peaks(fragments, peaks_mz, peaks_int, mass_model)
    
    print(f"  Fragment 1: {fragments[0]['mz']:.3f} m/z")
    print(f"  Fragment 2: {fragments[1]['mz']:.3f} m/z")
    print(f"  Peak: {peaks_mz[0]:.4f} m/z")
    print(f"  Matched to fragment at {matched[0]['theoretical_mz']:.3f} m/z")
    print(f"  Unmatched fragments: {len(unmatched)}")


if __name__ == '__main__':
    demo_basic_matching()
    demo_performance_comparison()
    demo_complex_matching()