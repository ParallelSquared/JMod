#!/usr/bin/env python3
"""Test script to verify renamed functions work correctly."""

import sys
import numpy as np
from src.spectral_fitting import (
    UnifiedCandidates, UnifiedMatrixData, UnifiedFeatures,
    create_unified_candidates,
    create_entries_v2,
    get_residuals_v2,
    get_manhattan_distance_v2,
    calculate_features_v2,
    unmatched_peaks_v2,
    build_sparse_matrix_v2,
    process_matrix_v2,
    fit_to_lib2
)

def test_data_structures():
    """Test that data structures are accessible."""
    print("Testing data structures...")
    
    # Create test data
    candidates = [("PEPTIDE", 2), ("SEQUENCE", 3)]
    peaks = [np.array([[100.0, 1000.0], [200.0, 2000.0]]),
             np.array([[150.0, 1500.0], [250.0, 2500.0]])]
    
    # Test UnifiedCandidates
    unified = create_unified_candidates(
        target_candidates=candidates,
        target_peaks=peaks
    )
    
    assert len(unified.candidates) == 2
    assert unified.n_targets == 2
    assert unified.n_decoys == 0
    print("✓ UnifiedCandidates works")
    
    # Test with decoys
    decoy_candidates = [("Decoy_PEPTIDE", 2)]
    decoy_peaks = [np.array([[105.0, 1050.0], [205.0, 2050.0]])]
    
    unified_with_decoys = create_unified_candidates(
        target_candidates=candidates,
        target_peaks=peaks,
        decoy_candidates=decoy_candidates,
        decoy_peaks=decoy_peaks
    )
    
    assert len(unified_with_decoys.candidates) == 3
    assert unified_with_decoys.n_targets == 2
    assert unified_with_decoys.n_decoys == 1
    print("✓ UnifiedCandidates with decoys works")

def test_renamed_functions():
    """Test that renamed functions are callable."""
    print("\nTesting renamed functions...")
    
    # Test function existence
    functions = [
        create_entries_v2,
        get_residuals_v2,
        get_manhattan_distance_v2,
        calculate_features_v2,
        unmatched_peaks_v2,
        build_sparse_matrix_v2,
        process_matrix_v2,
    ]
    
    for func in functions:
        assert callable(func), f"{func.__name__} is not callable"
        print(f"✓ {func.__name__} exists")

def test_fit_to_lib2():
    """Test that fit_to_lib2 still exists and is callable."""
    print("\nTesting fit_to_lib2...")
    assert callable(fit_to_lib2), "fit_to_lib2 is not callable"
    print("✓ fit_to_lib2 exists")

if __name__ == "__main__":
    try:
        test_data_structures()
        test_renamed_functions()
        test_fit_to_lib2()
        print("\n✅ All tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)