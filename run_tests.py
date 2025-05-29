#!/usr/bin/env python
"""
Test runner for JMod
Run this script to execute all tests in the test suite
"""
import sys
import os
import subprocess
import argparse


def run_tests(test_path=None, verbose=False, coverage=False):
    """
    Run the JMod test suite using pytest
    
    Args:
        test_path: Specific test file or directory to run (default: all tests)
        verbose: Enable verbose output
        coverage: Generate coverage report
    """
    # Base pytest command
    cmd = [sys.executable, "-m", "pytest"]
    
    # Add test directory or specific test
    if test_path:
        cmd.append(test_path)
    else:
        cmd.append("tests/")
    
    # Add verbosity
    if verbose:
        cmd.append("-v")
    else:
        cmd.append("-q")
    
    # Add coverage if requested
    if coverage:
        cmd.extend(["--cov=.", "--cov-report=html", "--cov-report=term"])
    
    # Add color output
    cmd.append("--color=yes")
    
    # Show print statements
    cmd.append("-s")
    
    print(f"Running tests with command: {' '.join(cmd)}")
    print("-" * 70)
    
    # Run the tests
    result = subprocess.run(cmd)
    
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Run JMod test suite")
    parser.add_argument(
        "test_path",
        nargs="?",
        help="Specific test file or directory to run (default: all tests)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "-c", "--coverage",
        action="store_true",
        help="Generate coverage report"
    )
    parser.add_argument(
        "--dummy",
        action="store_true",
        help="Run only dummy tests"
    )
    parser.add_argument(
        "--misc",
        action="store_true",
        help="Run only miscFunctions tests"
    )
    
    args = parser.parse_args()
    
    # Handle specific test selections
    test_path = args.test_path
    if args.dummy:
        test_path = "tests/test_dummy.py"
    elif args.misc:
        test_path = "tests/test_miscFunctions.py"
    
    # Check if pytest is installed
    try:
        import pytest
    except ImportError:
        print("Error: pytest is not installed!")
        print("Please install it with: pip install pytest")
        if args.coverage:
            print("For coverage support, also install: pip install pytest-cov")
        return 1
    
    # Run the tests
    exit_code = run_tests(test_path, args.verbose, args.coverage)
    
    if exit_code == 0:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())