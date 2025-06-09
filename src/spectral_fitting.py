"""
Spectral fitting module - refactored version.

This module now serves as a compatibility layer that imports from the new
refactored spectral_fitting package while maintaining the original API.

The original monolithic file has been split into:
- spectral_fitting/types.py: Data classes and type definitions
- spectral_fitting/matrix_operations.py: SpectrumMatrix operations
- spectral_fitting/feature_calculation.py: Feature extraction functions
- spectral_fitting/fitting_core.py: Core fitting algorithm
- spectral_fitting/adapter.py: Legacy API compatibility

The original code is preserved in spectral_fitting_legacy.py
"""

# Re-export everything from the legacy module for backward compatibility
from .spectral_fitting_legacy import (
    hyperscore2,
    get_features,
    unmatched_peaks,
    create_entries,
    # Keep any other functions that are still needed
)

# Import the refactored versions from the new module subpackage
from .spectral_fitting.adapter import (
    fit_to_lib,
    fit_to_lib_decoy,
    fit_to_lib2,
)

from .spectral_fitting.fitting_core import (
    fit_spectrum_to_library,  # New unified function
)

# Export all functions
__all__ = [
    # Legacy functions still in use
    'hyperscore2',
    'get_features',
    'unmatched_peaks', 
    'create_entries',
    # Refactored functions
    'fit_to_lib',
    'fit_to_lib_decoy',
    'fit_to_lib2',
    'fit_spectrum_to_library',
]