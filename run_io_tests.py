#!/usr/bin/env python3
"""
Test runner for load_files.py module tests
"""

import sys
import subprocess
import argparse

def run_tests(test_pattern=None, verbose=False, coverage=False):
    """Run the load_files tests with optional patterns and coverage"""
    
    cmd = ["python", "-m", "pytest"]
    
    if coverage:
        cmd.extend(["--cov=src.utils.io.load_files", "--cov-report=html", "--cov-report=term"])
    
    if verbose:
        cmd.append("-v")
    
    if test_pattern:
        cmd.append(f"tests/io/test_load_files.py::{test_pattern}")
    else:
        cmd.append("tests/io/test_load_files.py")
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode

def main():
    parser = argparse.ArgumentParser(description="Run load_files tests")
    parser.add_argument("--pattern", "-p", help="Test pattern to run (e.g., TestMzMLSpectrum)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--coverage", "-c", action="store_true", help="Run with coverage")
    parser.add_argument("--list-tests", "-l", action="store_true", help="List available test classes")
    
    args = parser.parse_args()
    
    if args.list_tests:
        print("Available test classes:")
        print("  TestBaseSpectrum")
        print("  TestBaseSpectrumFile") 
        print("  TestMzMLSpectrum")
        print("  TestMzMLSpectrumFile")
        print("  TestLoadSpectraFunction")
        print("  TestIntegration")
        print("  TestPerformance")
        print("\nExample usage:")
        print("  python run_io_tests.py --pattern TestMzMLSpectrum")
        print("  python run_io_tests.py --verbose --coverage")
        return 0
    
    return run_tests(args.pattern, args.verbose, args.coverage)

if __name__ == "__main__":
    sys.exit(main())