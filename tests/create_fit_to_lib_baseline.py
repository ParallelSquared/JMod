#!/usr/bin/env python3
"""
Script to create baseline values for fit_to_lib function testing.

This script captures the current behavior of fit_to_lib with sample data,
saving key intermediate values and final output for later comparison.
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
from src.utils.io.read_output import names
import src.config as config

# Configure settings
config.rt_tol = 10.0  # Large tolerance to help with matching
config.ms1_tol = 50e-6  # 50 ppm
config.mz_tol = 20e-6   # 20 ppm
config.top_n = 10
config.atleast_m = 3
config.args = Mock(mzml="test.mzML")


def create_mock_dia_spectrum():
    """Create a mock DIA spectrum for testing."""
    dia_spec = Mock()
    dia_spec.scan_num = 1000
    dia_spec.prec_mz = 500.25
    dia_spec.RT = 30.5
    dia_spec.ms1window = (495.0, 505.0)
    
    # Create realistic peak list
    peaks = [
        (147.1128, 1000.0),
        (227.1026, 800.0),
        (276.1555, 900.0),
        (324.1554, 600.0),
        (377.2032, 700.0),
        (425.2031, 400.0),
        (490.2872, 500.0),
        (588.3400, 300.0),
    ]
    dia_spec.peak_list = Mock(return_value=peaks)
    
    return dia_spec


def create_mock_ms1_spectrum():
    """Create a mock MS1 spectrum for testing."""
    ms1_spec = Mock()
    ms1_spec.RT = 30.4
    ms1_spec.mz = np.array([499.5, 500.25, 501.0, 502.0])
    ms1_spec.intensities = np.array([1000, 5000, 2000, 500])
    return ms1_spec


def create_sample_library():
    """Create a sample spectral library for testing."""
    library = {}
    
    # Add target peptide 1 - should match
    library['PEPTIDE_2'] = {
        'mod_seq': 'PEPTIDE',
        'prec_mz': 500.25,  # Exact match to DIA precursor
        'prec_z': 2,
        'iRT': 30.5,  # Exact match to DIA RT
        'IonMob': 0.95,
        'protein': 'PROT1',
        'is_decoy': False,
        'spectrum': np.array([
            [227.1026, 1000.0],  # Matches DIA peak
            [276.1555, 900.0],   # Matches DIA peak
            [324.1554, 800.0],   # Matches DIA peak
            [377.2032, 700.0],   # Matches DIA peak
            [425.2031, 600.0],   # Matches DIA peak
            [490.2872, 500.0],   # Matches DIA peak
        ]),
        'ordered_frags': ['b2_1', 'y2_1', 'b3_1', 'y3_1', 'b4_1', 'y4_1']
    }
    
    # Add target peptide 2 (partial match)
    library['SAMPLE_2'] = {
        'mod_seq': 'SAMPLE',
        'prec_mz': 499.8,
        'prec_z': 2,
        'iRT': 29.8,
        'IonMob': 0.94,
        'protein': 'PROT2',
        'is_decoy': False,
        'spectrum': np.array([
            [147.1128, 800.0],  # Matches DIA peak
            [250.0, 600.0],     # No match
            [377.2032, 500.0],  # Matches DIA peak
            [450.0, 400.0],     # No match
        ]),
        'ordered_frags': ['b1_1', 'b2_1', 'y3_1', 'y4_1']
    }
    
    # Add decoy peptide (should be filtered out in fit_to_lib)
    library['DECOY1_2'] = {
        'mod_seq': 'YOCED',
        'prec_mz': 500.5,
        'prec_z': 2,
        'iRT': 30.8,
        'IonMob': 0.96,
        'protein': 'DECOY_PROT1',
        'is_decoy': True,
        'spectrum': np.array([
            [200.0, 1000.0],
            [300.0, 800.0],
            [400.0, 600.0],
        ]),
        'ordered_frags': ['b2_1', 'b3_1', 'y2_1']
    }
    
    return library


def create_rt_mz_array(library):
    """Create rt_mz array from library."""
    rt_mz_list = []
    for key, entry in library.items():
        rt_mz_list.append([entry['iRT'], entry['prec_mz']])
    return np.array(rt_mz_list)


def run_fit_to_lib_test(dia_spec, library, rt_mz, all_keys, ms1_spec, 
                        rt_filter=False, return_frags=False, test_name=""):
    """Run fit_to_lib with given parameters and return structured results."""
    
    print(f"\n{'='*50}")
    print(f"Running test: {test_name}")
    print(f"  RT filter: {rt_filter}")
    print(f"  Return frags: {return_frags}")
    
    ms1_spectra = [ms1_spec]
    result = fit_to_lib(
        dia_spec=dia_spec,
        library=library,
        rt_mz=rt_mz,
        all_keys=all_keys,
        rt_filter=rt_filter,
        ms1_mz=None,
        ms1_spectra=ms1_spectra,
        rt_tol=config.rt_tol,
        ms1_tol=config.ms1_tol,
        mz_tol=config.mz_tol,
        return_frags=return_frags,
        frac_matched=0.5
    )
    
    # Analyze result structure
    result_info = {
        'raw_result': result,
        'result_type': type(result).__name__,
        'is_tuple': isinstance(result, tuple),
        'is_list': isinstance(result, list),
    }
    
    if isinstance(result, tuple):
        result_info['tuple_length'] = len(result)
        if len(result) >= 1:
            result_info['first_element'] = result[0]
            result_info['first_element_type'] = type(result[0]).__name__
        if len(result) >= 2:
            result_info['second_element'] = result[1]
            result_info['second_element_type'] = type(result[1]).__name__
    elif isinstance(result, list):
        result_info['list_length'] = len(result)
        if len(result) > 0:
            result_info['first_row'] = result[0]
            if hasattr(result[0], '__len__'):
                result_info['row_length'] = len(result[0])
    
    # Extract key values if we have standard output format
    output_data = result[0] if isinstance(result, tuple) and len(result) > 0 else result
    if output_data and len(output_data) > 0:
        first_row = output_data[0] if isinstance(output_data[0], list) else output_data
        if hasattr(first_row, '__len__') and len(first_row) >= 48:
            result_info['parsed'] = {
                'coefficient': first_row[0],
                'spec_id': first_row[1],
                'ms1_spec_id': first_row[2],
                'sequence': first_row[3],
                'charge': first_row[4],
                'window_mz': first_row[5],
                'rt': first_row[6],
                'num_lib': first_row[7],
                'has_match': first_row[0] != 0,
            }
            print(f"  Result: {'MATCH' if first_row[0] != 0 else 'NO MATCH'}")
            print(f"  Coefficient: {first_row[0]}")
            if first_row[0] != 0:
                print(f"  Sequence: {first_row[3]}")
    
    return result_info


def main():
    """Main function to create and save baseline data."""
    print("Creating baseline data for fit_to_lib...")
    
    # Create test data
    dia_spec = create_mock_dia_spectrum()
    ms1_spec = create_mock_ms1_spectrum()
    library = create_sample_library()
    all_keys = list(library.keys())
    rt_mz = create_rt_mz_array(library)
    
    print(f"\nTest setup:")
    print(f"  Library entries: {len(library)}")
    print(f"  Library keys: {list(library.keys())}")
    print(f"  DIA spectrum: {dia_spec.scan_num}, m/z={dia_spec.prec_mz}, RT={dia_spec.RT}")
    print(f"  DIA peaks: {len(list(dia_spec.peak_list()))}")
    
    # Run tests with different configurations
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
                'peaks': list(dia_spec.peak_list()),
            },
            'library_keys': all_keys,
            'library_entries': {k: {
                'prec_mz': v['prec_mz'],
                'iRT': v['iRT'],
                'is_decoy': v.get('is_decoy', False),
                'num_peaks': len(v['spectrum'])
            } for k, v in library.items()},
            'rt_mz_shape': rt_mz.shape,
        },
        'tests': {}
    }
    
    # Test 1: No RT filter, no fragments
    baseline_data['tests']['no_rt_no_frags'] = run_fit_to_lib_test(
        dia_spec, library, rt_mz, all_keys, ms1_spec,
        rt_filter=False, return_frags=False,
        test_name="No RT filter, no fragments"
    )
    
    # Test 2: No RT filter, with fragments
    baseline_data['tests']['no_rt_with_frags'] = run_fit_to_lib_test(
        dia_spec, library, rt_mz, all_keys, ms1_spec,
        rt_filter=False, return_frags=True,
        test_name="No RT filter, with fragments"
    )
    
    # Test 3: With RT filter, no fragments
    baseline_data['tests']['with_rt_no_frags'] = run_fit_to_lib_test(
        dia_spec, library, rt_mz, all_keys, ms1_spec,
        rt_filter=True, return_frags=False,
        test_name="With RT filter, no fragments"
    )
    
    # Test 4: With RT filter, with fragments
    baseline_data['tests']['with_rt_with_frags'] = run_fit_to_lib_test(
        dia_spec, library, rt_mz, all_keys, ms1_spec,
        rt_filter=True, return_frags=True,
        test_name="With RT filter, with fragments"
    )
    
    # Save baseline data
    output_file = 'fit_to_lib_baseline.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(baseline_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"\n{'='*50}")
    print(f"Baseline data saved to {output_file}")
    
    # Create summary
    summary_file = 'fit_to_lib_baseline_summary.txt'
    with open(summary_file, 'w') as f:
        f.write("fit_to_lib Baseline Summary\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Configuration:\n")
        for key, value in baseline_data['config'].items():
            f.write(f"  {key}: {value}\n")
        
        f.write("\nInput Data:\n")
        f.write(f"  DIA Scan: {baseline_data['input']['dia_spec']['scan_num']}\n")
        f.write(f"  DIA m/z: {baseline_data['input']['dia_spec']['prec_mz']}\n")
        f.write(f"  DIA RT: {baseline_data['input']['dia_spec']['RT']}\n")
        f.write(f"  DIA peaks: {len(baseline_data['input']['dia_spec']['peaks'])}\n")
        f.write(f"  Library entries: {len(baseline_data['input']['library_keys'])}\n")
        
        f.write("\nTest Results:\n")
        for test_name, test_data in baseline_data['tests'].items():
            f.write(f"\n{test_name}:\n")
            f.write(f"  Result type: {test_data['result_type']}\n")
            if 'parsed' in test_data:
                f.write(f"  Match found: {test_data['parsed']['has_match']}\n")
                if test_data['parsed']['has_match']:
                    f.write(f"  Coefficient: {test_data['parsed']['coefficient']}\n")
                    f.write(f"  Sequence: {test_data['parsed']['sequence']}\n")
        
        f.write("\nColumn Names (from src.utils.io.read_output.names):\n")
        for i, name in enumerate(names[:10]):  # First 10 columns
            f.write(f"  {i}: {name}\n")
        f.write("  ...\n")
    
    print(f"Summary saved to {summary_file}")
    
    # Also print key findings
    print("\nKey findings:")
    for test_name, test_data in baseline_data['tests'].items():
        result_type = test_data['result_type']
        is_tuple = test_data.get('is_tuple', False)
        has_match = test_data.get('parsed', {}).get('has_match', False)
        print(f"  {test_name}: {result_type} (tuple={is_tuple}, match={has_match})")
    
    return baseline_data


if __name__ == '__main__':
    baseline_data = main()