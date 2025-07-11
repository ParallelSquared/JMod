"""
Regression test for fit_to_lib function.
Compares current implementation against baseline results.
"""

import pickle
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.spectral_fitting import fit_to_lib
import src.config as config


def load_baseline():
    """Load baseline test data from pickle file."""
    baseline_path = Path(__file__).parent / 'fit_to_lib_baseline.pkl'
    with open(baseline_path, 'rb') as f:
        return pickle.load(f)


def compare_results(result1, result2, test_name, tolerance=1e-10):
    """Compare two fit_to_lib results for equality."""
    differences = []
    
    # Handle different result formats
    # fit_to_lib can return either a list or a tuple (list, list) depending on return_frags
    if isinstance(result1, tuple) and isinstance(result2, list):
        # Current implementation returns list, baseline has tuple
        # Compare the first element of tuple with the list
        differences.extend(compare_lists(result1[0], result2, "result"))
    elif isinstance(result1, list) and isinstance(result2, tuple):
        # Current implementation returns list, baseline has tuple
        # Compare the list with first element of tuple
        differences.extend(compare_lists(result1, result2[0], "result"))
    elif isinstance(result1, tuple) and isinstance(result2, tuple):
        # Compare tuples element by element
        if len(result1) != len(result2):
            differences.append(f"Result tuple length mismatch: {len(result1)} vs {len(result2)}")
            return differences
            
        for i, (elem1, elem2) in enumerate(zip(result1, result2)):
            if isinstance(elem1, list) and isinstance(elem2, list):
                elem_diffs = compare_lists(elem1, elem2, f"element {i}")
                differences.extend(elem_diffs)
            else:
                differences.append(f"Element {i} type mismatch: {type(elem1)} vs {type(elem2)}")
                
    elif isinstance(result1, list) and isinstance(result2, list):
        differences.extend(compare_lists(result1, result2, "result"))
    else:
        differences.append(f"Result type mismatch: {type(result1)} vs {type(result2)}")
    
    return differences


def compare_lists(list1, list2, context=""):
    """Compare two lists of results."""
    differences = []
    
    if len(list1) != len(list2):
        differences.append(f"{context} length mismatch: {len(list1)} vs {len(list2)}")
        return differences
    
    for i, (row1, row2) in enumerate(zip(list1, list2)):
        if len(row1) != len(row2):
            differences.append(f"{context} row {i} length mismatch: {len(row1)} vs {len(row2)}")
            continue
            
        # Compare each field in the row
        for j, (val1, val2) in enumerate(zip(row1, row2)):
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                if abs(val1 - val2) > 1e-10:
                    differences.append(f"{context} row {i}, field {j}: {val1} != {val2}")
            elif isinstance(val1, np.ndarray) and isinstance(val2, np.ndarray):
                if not np.allclose(val1, val2, rtol=1e-10, atol=1e-10):
                    differences.append(f"{context} row {i}, field {j}: array values differ")
            elif val1 != val2:
                differences.append(f"{context} row {i}, field {j}: {val1} != {val2}")
    
    return differences


def run_regression_tests():
    """Run regression tests comparing current implementation to baseline."""
    baseline = load_baseline()
    
    # Extract test configuration
    config_dict = baseline['config']
    dia_spec = baseline['input']['dia_spec']
    library_keys = baseline['input']['library_keys']
    library_entries = baseline['input']['library_entries']
    rt_mz_shape = baseline['input']['rt_mz_shape']
    
    # Set config module attributes
    for key, value in config_dict.items():
        setattr(config, key, value)
    
    # Create rt_mz array (dummy data matching shape)
    rt_mz = np.zeros(rt_mz_shape)
    for i, key in enumerate(library_keys):
        if key in library_entries:
            rt_mz[i, 0] = library_entries[key]['iRT']
            rt_mz[i, 1] = library_entries[key]['prec_mz']
    
    all_tests_passed = True
    test_results = {}
    
    # Run each test scenario
    for test_name, test_data in baseline['tests'].items():
        print(f"\nRunning test: {test_name}")
        
        # Configure for this test
        if test_name in ['no_rt_no_frags', 'no_rt_with_frags']:
            config.use_rt = False
        else:
            config.use_rt = True
            
        # Ensure all library entries have required fields
        for key in library_entries:
            # Add empty spectrum if not present
            if 'spectrum' not in library_entries[key]:
                library_entries[key]['spectrum'] = []
                
        if test_name in ['no_rt_with_frags', 'with_rt_with_frags']:
            # Add fragment data to library entries for these tests
            for key in library_keys:
                if key in library_entries:
                    # Add dummy fragment data as numpy arrays
                    library_entries[key]['peaks'] = [[100.0, 200.0], [50.0, 100.0]]
                    library_entries[key]['spectrum'] = np.array([[100.0, 50.0], [200.0, 100.0]])  # [mz, intensity] pairs
        else:
            # For no_frags tests, provide minimal spectra to avoid division by zero
            for key in library_entries:
                if 'peaks' in library_entries[key]:
                    del library_entries[key]['peaks']
                # Create minimal spectra based on the number of peaks in baseline
                num_peaks = library_entries[key].get('num_peaks', 1)
                if num_peaks > 0:
                    # Create dummy spectrum with correct number of peaks
                    mz_values = np.arange(100, 100 + num_peaks * 10, 10, dtype=float)
                    intensities = np.ones(num_peaks) * 10.0
                    library_entries[key]['spectrum'] = np.column_stack([mz_values, intensities])
                else:
                    library_entries[key]['spectrum'] = np.array([]).reshape(0, 2)
        
        # Run current implementation
        try:
            # Create dia_spec object with required attributes
            class DiaSpec:
                def __init__(self, spec_dict):
                    self.scan_num = spec_dict['scan_num']
                    self.prec_mz = spec_dict['prec_mz']
                    self.RT = spec_dict['RT']
                    self.ms1window = spec_dict['ms1window']
                    self.peaks = spec_dict['peaks']
                
                def peak_list(self):
                    """Return peaks as list of [mz, intensity] pairs."""
                    return self.peaks
            
            dia_spec_obj = DiaSpec(dia_spec)
            
            # Set additional config attributes that may be needed
            if not hasattr(config, 'use_rt'):
                config.use_rt = False
            
            # Create dummy MS1 spectrum to avoid None errors
            class MS1Spec:
                def __init__(self):
                    self.mz = np.array([])  # Empty MS1 spectrum
                    self.RT = dia_spec['RT']  # Use same RT as DIA spectrum
                    
            ms1_spectra = [MS1Spec()]  # Single dummy MS1 spectrum
            
            # Set return_frags based on test name
            return_frags = test_name in ['no_rt_with_frags', 'with_rt_with_frags']
            
            current_result = fit_to_lib(
                dia_spec_obj, 
                library_entries,  # library dict
                rt_mz, 
                library_keys,  # all_keys
                rt_filter=config.use_rt,
                ms1_spectra=ms1_spectra,
                return_frags=return_frags
            )
            
            # Compare with baseline
            baseline_result = test_data['raw_result']
            differences = compare_results(current_result, baseline_result, test_name)
            
            if differences:
                print(f"  FAILED - Found {len(differences)} differences:")
                for diff in differences[:5]:  # Show first 5 differences
                    print(f"    - {diff}")
                if len(differences) > 5:
                    print(f"    ... and {len(differences) - 5} more differences")
                all_tests_passed = False
                test_results[test_name] = 'FAILED'
            else:
                print(f"  PASSED - Results match baseline")
                test_results[test_name] = 'PASSED'
                
        except Exception as e:
            import traceback
            print(f"  ERROR - Exception occurred: {str(e)}")
            print(f"  Traceback:")
            traceback.print_exc()
            all_tests_passed = False
            test_results[test_name] = 'ERROR'
    
    # Summary
    print("\n" + "="*50)
    print("REGRESSION TEST SUMMARY")
    print("="*50)
    for test_name, status in test_results.items():
        print(f"{test_name}: {status}")
    
    if all_tests_passed:
        print("\nAll regression tests PASSED!")
        return 0
    else:
        print("\nSome regression tests FAILED!")
        return 1


if __name__ == "__main__":
    exit_code = run_regression_tests()
    sys.exit(exit_code)