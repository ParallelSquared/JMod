#!/usr/bin/env python3
"""
Quick test to compare performance between legacy and refactored spectral fitting.
"""

import sys
import time
sys.path.insert(0, 'src')

def test_single_spectrum():
    """Test a single spectrum through both legacy and new fitting."""
    
    from src.spectral_fitting import fit_to_lib
    from src.spectral_fitting_legacy import fit_to_lib as legacy_fit_to_lib
    
    # This is just a conceptual test - would need actual data
    print("This would test individual spectrum performance")
    print("Need to run with actual JMod data to see timing differences")

def quick_timing_check():
    """Just run with timing enabled for a few iterations"""
    
    print("="*60)
    print("PERFORMANCE DIAGNOSIS PLAN")
    print("="*60)
    
    print("""
1. The initial search has become very slow (7 iterations/second vs expected ~100+)

2. Recent changes that could impact performance:
   - Spectral fitting module refactoring
   - Fragment correlation fixes
   - RT alignment fixes
   - Addition of comprehensive feature calculations

3. Most likely bottlenecks:
   a) NNLS solving taking too long
   b) Feature calculation overhead
   c) Matrix operations in spectral fitting
   d) Fragment correlation calculations
   e) Memory allocation/garbage collection

4. To diagnose:
   - Run with the added timing decorators
   - Look for functions taking >10ms per call
   - Check for functions called excessively
   - Monitor memory usage

5. Quick tests to try:
   - Run: python run_jmod.py [your normal args] --debug_log
   - Stop after ~100 iterations (Ctrl+C)
   - Check timing output in terminal and debug.log
   
   OR
   
   - Run: python profile_performance.py
   - This will run with Python's cProfile enabled
   
6. Expected performance:
   - Each spectrum should process in ~10-50ms
   - NNLS should be <5ms per spectrum
   - Feature calculation should be <10ms per spectrum
   
7. If you see any function taking >100ms consistently, that's the bottleneck.

Current hypothesis: The refactored spectral fitting is doing more work
per spectrum than the legacy version, possibly in feature calculation.
""")

if __name__ == "__main__":
    quick_timing_check()