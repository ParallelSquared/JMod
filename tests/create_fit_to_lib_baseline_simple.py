#!/usr/bin/env python3
"""
Simplified baseline test for fit_to_lib function.
This version focuses on capturing the current behavior with minimal setup.
"""

import pickle
import numpy as np
import sys
import os
from unittest.mock import Mock

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import after path setup
from src.spectral_fitting import fit_to_lib
import src.config as config

# Configure settings
config.rt_tol = 10.0  # Large tolerance to ensure matches
config.ms1_tol = 50e-6  # 50 ppm
config.mz_tol = 20e-6   # 20 ppm
config.top_n = 10
config.atleast_m = 2  # Lower threshold
config.args = Mock(mzml="test.mzML")


def main():
    """Create minimal test case for fit_to_lib baseline."""
    print("Creating simplified baseline for fit_to_lib...")
    
    # Create simple DIA spectrum
    dia_spec = Mock()
    dia_spec.scan_num = 1000
    dia_spec.prec_mz = 500.0
    dia_spec.RT = 30.0
    dia_spec.ms1window = (495.0, 505.0)
    
    # Simple peaks that should match
    peaks = [
        (100.0, 1000.0),
        (200.0, 2000.0),
        (300.0, 3000.0),
    ]
    dia_spec.peak_list = Mock(return_value=peaks)
    
    # Create simple MS1 spectrum
    ms1_spec = Mock()
    ms1_spec.RT = 30.0
    ms1_spec.mz = np.array([499.0, 500.0, 501.0])
    ms1_spec.intensities = np.array([1000, 5000, 2000])
    
    # Create minimal library with one entry
    library = {
        'TEST_2': {
            'mod_seq': 'TEST',
            'prec_mz': 500.0,
            'prec_z': 2,
            'iRT': 30.0,
            'IonMob': 1.0,
            'protein': 'TEST_PROT',
            'is_decoy': False,
            'spectrum': np.array([
                [100.0, 1000.0],  # Exact match
                [200.0, 1000.0],  # Exact match
                [300.0, 1000.0],  # Exact match
            ]),
            'ordered_frags': ['b1_1', 'b2_1', 'y1_1']
        }
    }
    
    all_keys = list(library.keys())
    rt_mz = np.array([[30.0, 500.0]])  # Single entry
    
    print(f"Library: {all_keys}")
    print(f"DIA peaks: {peaks}")
    print(f"Library spectrum: {library['TEST_2']['spectrum'].tolist()}")
    
    # Call fit_to_lib without RT filter first
    print("\nCalling fit_to_lib without RT filter...")
    result = fit_to_lib(
        dia_spec=dia_spec,
        library=library,
        rt_mz=rt_mz,
        all_keys=all_keys,
        rt_filter=False,
        ms1_mz=None,
        ms1_spectra=[ms1_spec],
        rt_tol=config.rt_tol,
        ms1_tol=config.ms1_tol,
        mz_tol=config.mz_tol,
        return_frags=False,
        frac_matched=0.1  # Low threshold
    )
    
    # Analyze result
    print(f"\nResult type: {type(result)}")
    if isinstance(result, tuple):
        print(f"Tuple length: {len(result)}")
        if len(result) > 0:
            print(f"First element type: {type(result[0])}")
            if hasattr(result[0], '__len__'):
                print(f"First element length: {len(result[0])}")
                if len(result[0]) > 0:
                    print(f"First row: {result[0][0] if isinstance(result[0][0], list) else result[0]}")
    
    # Save baseline
    baseline_data = {
        'config': {
            'rt_tol': config.rt_tol,
            'ms1_tol': config.ms1_tol,
            'mz_tol': config.mz_tol,
            'top_n': config.top_n,
            'atleast_m': config.atleast_m,
        },
        'input': {
            'dia_spec': {
                'scan_num': dia_spec.scan_num,
                'prec_mz': dia_spec.prec_mz,
                'RT': dia_spec.RT,
                'ms1window': dia_spec.ms1window,
                'peaks': peaks,
            },
            'library_keys': all_keys,
            'rt_mz_shape': rt_mz.shape,
        },
        'output': {
            'raw_result': result,
            'result_type': str(type(result)),
        }
    }
    
    # Try to interpret result structure
    if isinstance(result, tuple) and len(result) == 2:
        output_list, frag_info = result
        baseline_data['output']['output_list'] = output_list
        baseline_data['output']['frag_info'] = frag_info
        
        if output_list and len(output_list) > 0:
            # Check if it's the expected empty result format
            if isinstance(output_list, list) and len(output_list[0]) == 48:
                print("\nGot standard 48-column output format")
                print(f"Coefficient: {output_list[0][0]}")
                print(f"Scan ID: {output_list[0][1]}")
                print(f"Sequence: {output_list[0][3]}")
    
    # Save pickle
    with open('fit_to_lib_baseline_simple.pkl', 'wb') as f:
        pickle.dump(baseline_data, f)
    
    print("\nBaseline saved to fit_to_lib_baseline_simple.pkl")
    
    # Also try with even simpler setup
    print("\n" + "="*50)
    print("Testing with minimal setup...")
    
    # Super simple case - single peak match
    dia_spec2 = Mock()
    dia_spec2.scan_num = 2000
    dia_spec2.prec_mz = 400.0
    dia_spec2.RT = 20.0
    dia_spec2.ms1window = (395.0, 405.0)
    dia_spec2.peak_list = Mock(return_value=[(150.0, 1000.0)])
    
    library2 = {
        'SIMPLE_2': {
            'mod_seq': 'SIMPLE',
            'prec_mz': 400.0,
            'prec_z': 2,
            'iRT': 20.0,
            'IonMob': 1.0,
            'protein': 'PROT1',
            'is_decoy': False,
            'spectrum': np.array([[150.0, 1000.0]]),
            'ordered_frags': ['b1_1']
        }
    }
    
    rt_mz2 = np.array([[20.0, 400.0]])
    
    result2 = fit_to_lib(
        dia_spec=dia_spec2,
        library=library2,
        rt_mz=rt_mz2,
        all_keys=['SIMPLE_2'],
        rt_filter=False,
        ms1_mz=None,
        ms1_spectra=[ms1_spec],  # Reuse MS1
        rt_tol=100.0,  # Very large tolerance
        ms1_tol=100e-6,
        mz_tol=100e-6,
        return_frags=False,
        frac_matched=0.0  # No threshold
    )
    
    print(f"Minimal test result: {type(result2)}")
    if isinstance(result2, tuple) and len(result2) > 0:
        print(f"Output: {result2[0]}")
    
    return baseline_data


if __name__ == '__main__':
    baseline_data = main()