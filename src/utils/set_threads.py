import os
import sys
import numpy as np
import threading
import logging

# --- 1. Set the Target Environment Variables ---
# Setting these environment variables forces the underlying C/Fortran libraries
# (like OpenBLAS, MKL, OpenMP, etc.) to use a single thread for matrix operations.
# This eliminates floating-point race conditions, restoring determinism.

# Set the maximum number of threads for the following libraries to 1
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'  # Also controls threads for numexpr (used by Pandas/NumPy)


# --- 2. Verification Function ---

def check_threading_settings():
    """Checks the environment variables and prints the detected thread counts."""
    # Use the logging module for clear, organized output
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    print("\n--- Threading Environment Check ---")

    # Check environment variables
    env_vars = ['OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'NUMEXPR_NUM_THREADS']
    for var in env_vars:
        value = os.environ.get(var, 'NOT SET')
        print(f"[{var}]: {value}")

    # Check actual runtime threading limit (if available)
    try:
        # NumPy/MKL check (may vary based on installation)
        if hasattr(np.__config__, 'blas_info') and 'mkl' in str(np.__config__.blas_info):
            print(f"[NumPy/MKL Threads]: {os.environ.get('MKL_NUM_THREADS', 'N/A')}")
        elif hasattr(np.__config__, 'blas_info') and 'openblas' in str(np.__config__.blas_info):
            print(f"[NumPy/OpenBLAS Threads]: {os.environ.get('OPENBLAS_NUM_THREADS', 'N/A')}")

    except Exception:
        # Catch errors if __config__ is unavailable or unusual
        print("[NumPy Runtime Check]: Failed to retrieve thread count.")

    print("---------------------------------")
    print("If all reported values are '1' (or 'NOT SET' for non-used backends), threading is controlled.")