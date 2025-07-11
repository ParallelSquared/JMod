#!/usr/bin/env python3
"""
Performance benchmarking for unified spectral fitting implementation.

This script compares the performance of the unified implementation against
the original dual-path implementation, measuring execution time, memory usage,
and result consistency.
"""

import time
import tracemalloc
import numpy as np
from typing import Dict, List, Tuple, Any
import psutil
import os
import json
from datetime import datetime

# Import the spectral fitting module
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.spectral_fitting import fit_to_lib2
import src.config as config


class PerformanceBenchmark:
    """Benchmark performance of spectral fitting implementations."""
    
    def __init__(self):
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'system_info': self._get_system_info(),
            'benchmarks': []
        }
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information for benchmark context."""
        return {
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'python_version': sys.version,
            'platform': sys.platform
        }
    
    def _measure_memory(self, func, *args, **kwargs) -> Tuple[Any, float, float]:
        """
        Measure memory usage of a function call.
        
        Returns:
            Tuple of (result, peak_memory_mb, execution_time_s)
        """
        # Start memory tracking
        tracemalloc.start()
        
        # Measure execution time
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        # Get peak memory usage
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        peak_mb = peak / (1024 * 1024)
        exec_time = end_time - start_time
        
        return result, peak_mb, exec_time
    
    def benchmark_fit_to_lib2(self, dia_spec, library, rt_mz, all_keys, 
                             decoy_library=None, name="fit_to_lib2"):
        """Benchmark a single fit_to_lib2 call."""
        print(f"\nBenchmarking {name}...")
        
        # Warm-up run (not measured)
        _ = fit_to_lib2(dia_spec, library, rt_mz, all_keys, 
                       decoy=bool(decoy_library), decoy_library=decoy_library)
        
        # Measured runs
        times = []
        memories = []
        n_runs = 5
        
        for i in range(n_runs):
            result, peak_mb, exec_time = self._measure_memory(
                fit_to_lib2, dia_spec, library, rt_mz, all_keys,
                decoy=bool(decoy_library), decoy_library=decoy_library
            )
            times.append(exec_time)
            memories.append(peak_mb)
            print(f"  Run {i+1}: {exec_time:.3f}s, {peak_mb:.1f}MB")
        
        # Calculate statistics
        benchmark_result = {
            'name': name,
            'n_candidates': len(all_keys),
            'has_decoys': bool(decoy_library),
            'execution_times': times,
            'memory_peaks': memories,
            'avg_time': np.mean(times),
            'std_time': np.std(times),
            'avg_memory': np.mean(memories),
            'std_memory': np.std(memories),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'result_length': len(result)
        }
        
        self.results['benchmarks'].append(benchmark_result)
        
        print(f"  Average: {benchmark_result['avg_time']:.3f}s ± {benchmark_result['std_time']:.3f}s")
        print(f"  Memory: {benchmark_result['avg_memory']:.1f}MB ± {benchmark_result['std_memory']:.1f}MB")
        
        return result
    
    def compare_results(self, result1: List, result2: List, name1="Result1", name2="Result2"):
        """Compare two results for consistency."""
        print(f"\nComparing {name1} vs {name2}:")
        
        if len(result1) != len(result2):
            print(f"  WARNING: Different result lengths: {len(result1)} vs {len(result2)}")
            return False
        
        # Compare numerical values
        differences = []
        for i, (row1, row2) in enumerate(zip(result1, result2)):
            if len(row1) != len(row2):
                print(f"  WARNING: Row {i} has different lengths")
                continue
            
            # Compare numerical columns (skip strings)
            for j, (val1, val2) in enumerate(zip(row1, row2)):
                if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                    if abs(val1 - val2) > 1e-6:
                        differences.append((i, j, val1, val2))
        
        if differences:
            print(f"  Found {len(differences)} numerical differences")
            # Show first few differences
            for i, j, val1, val2 in differences[:5]:
                print(f"    Row {i}, Col {j}: {val1} vs {val2} (diff: {abs(val1-val2)})")
        else:
            print("  Results are numerically identical!")
        
        return len(differences) == 0
    
    def save_results(self, filename="benchmark_results.json"):
        """Save benchmark results to JSON file."""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\nBenchmark results saved to {filename}")
    
    def print_summary(self):
        """Print summary of benchmark results."""
        print("\n" + "="*60)
        print("BENCHMARK SUMMARY")
        print("="*60)
        
        for bench in self.results['benchmarks']:
            print(f"\n{bench['name']}:")
            print(f"  Candidates: {bench['n_candidates']}")
            print(f"  Execution Time: {bench['avg_time']:.3f}s ± {bench['std_time']:.3f}s")
            print(f"  Memory Usage: {bench['avg_memory']:.1f}MB ± {bench['std_memory']:.1f}MB")
            print(f"  Time Range: [{bench['min_time']:.3f}s - {bench['max_time']:.3f}s]")
        
        # Calculate improvements if we have pairs to compare
        if len(self.results['benchmarks']) >= 2:
            print("\n" + "-"*60)
            print("PERFORMANCE COMPARISON")
            print("-"*60)
            
            for i in range(0, len(self.results['benchmarks'])-1, 2):
                if i+1 < len(self.results['benchmarks']):
                    bench1 = self.results['benchmarks'][i]
                    bench2 = self.results['benchmarks'][i+1]
                    
                    time_diff = (bench2['avg_time'] - bench1['avg_time']) / bench1['avg_time'] * 100
                    mem_diff = (bench2['avg_memory'] - bench1['avg_memory']) / bench1['avg_memory'] * 100
                    
                    print(f"\n{bench2['name']} vs {bench1['name']}:")
                    print(f"  Time: {time_diff:+.1f}% ({'faster' if time_diff < 0 else 'slower'})")
                    print(f"  Memory: {mem_diff:+.1f}% ({'less' if mem_diff < 0 else 'more'})")


def create_mock_data(n_candidates=100, n_peaks=50, has_decoys=True):
    """Create mock data for benchmarking."""
    # Create mock DIA spectrum
    class MockDIASpec:
        def __init__(self):
            self.scan_num = 1000
            self.prec_mz = 500.0
            self.RT = 30.0
            self.ms1window = (495.0, 505.0)
        
        def peak_list(self):
            # Random peaks - ensure at least 3 peaks for processing
            actual_n_peaks = max(n_peaks, 10)
            mz_values = np.sort(np.random.uniform(100, 1000, actual_n_peaks))
            intensities = np.random.exponential(1000, actual_n_peaks)
            return [(mz, intensity) for mz, intensity in zip(mz_values, intensities)]
    
    dia_spec = MockDIASpec()
    
    # Create mock library
    library = {}
    all_keys = []
    
    for i in range(n_candidates):
        seq = f"PEPTIDE{i}"
        charge = 2
        key = (seq, charge)
        all_keys.append(key)
        
        # Create mock spectrum
        n_frags = np.random.randint(10, 30)
        frag_mz = np.sort(np.random.uniform(100, 900, n_frags))
        frag_int = np.random.exponential(100, n_frags)
        
        library[key] = {
            'spectrum': np.column_stack([frag_mz, frag_int]),
            'prec_mz': 495 + i * 0.1,  # Spread across the window
            'RT': 30.0 + np.random.normal(0, 1),
            'ordered_frags': [f'y{j}' if j < n_frags//2 else f'b{j-n_frags//2}' 
                             for j in range(n_frags)],
            'frags': {f'frag{j}': (frag_mz[j], frag_int[j]) for j in range(n_frags)}
        }
    
    # Create RT/MZ array
    rt_mz = np.array([[library[key]['RT'], library[key]['prec_mz']] for key in all_keys])
    
    # Create decoy library if requested
    decoy_library = None
    if has_decoys:
        decoy_library = {}
        for key in all_keys:
            decoy_key = (f"Decoy_{key[0]}", key[1])
            # Shuffle the spectrum
            orig_spec = library[key]['spectrum'].copy()
            np.random.shuffle(orig_spec[:, 0])  # Shuffle m/z values
            decoy_library[key] = {
                'spectrum': orig_spec,
                'prec_mz': library[key]['prec_mz'] - config.decoy_mz_offset,
                'RT': library[key]['RT'],
                'ordered_frags': library[key]['ordered_frags'],
                'frags': library[key]['frags']
            }
    
    return dia_spec, library, rt_mz, all_keys, decoy_library


def main():
    """Run performance benchmarks."""
    print("JMod Spectral Fitting Performance Benchmark")
    print("=" * 60)
    
    # Create benchmark instance
    benchmark = PerformanceBenchmark()
    
    # Test different sizes
    test_sizes = [
        (50, "Small"),
        (200, "Medium"),
        (500, "Large")
    ]
    
    for n_candidates, size_name in test_sizes:
        print(f"\n\n{'='*60}")
        print(f"Testing with {size_name} dataset ({n_candidates} candidates)")
        print(f"{'='*60}")
        
        # Create test data
        dia_spec, library, rt_mz, all_keys, decoy_library = create_mock_data(
            n_candidates=n_candidates, has_decoys=True
        )
        
        # Benchmark without decoys
        result_no_decoy = benchmark.benchmark_fit_to_lib2(
            dia_spec, library, rt_mz, all_keys,
            name=f"{size_name} - No Decoys"
        )
        
        # Benchmark with decoys
        result_with_decoy = benchmark.benchmark_fit_to_lib2(
            dia_spec, library, rt_mz, all_keys, decoy_library,
            name=f"{size_name} - With Decoys"
        )
        
        # Compare results (should be different due to decoys)
        print(f"\nResult comparison for {size_name}:")
        print(f"  No decoys: {len(result_no_decoy)} results")
        print(f"  With decoys: {len(result_with_decoy)} results")
    
    # Print summary
    benchmark.print_summary()
    
    # Save results
    benchmark.save_results()
    
    print("\nBenchmark complete!")


if __name__ == "__main__":
    # Set some config values
    config.top_n = 10
    config.atleast_m = 3
    config.rt_tol = 2.0
    config.ms1_tol = 20e-6
    config.mz_tol = 20e-6
    config.decoy_mz_offset = 20.0
    config.unmatched_fit_type = 'a'
    config.protein_column = 'protein'
    
    # Mock args for config
    class MockArgs:
        mzml = "benchmark.mzML"
    config.args = MockArgs()
    
    main()