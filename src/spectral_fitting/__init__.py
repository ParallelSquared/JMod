"""
Spectral Fitting Module

This module provides functionality for fitting DIA spectra against spectral libraries
using sparse non-negative least squares. Key features:

- Unified handling of target and decoy peptides
- Comprehensive spectral similarity metrics (SCRIBE, Manhattan distance, etc.)
- Statistical analysis of fit quality
- Fragment-level feature extraction

Main entry point: fit_spectrum_to_library()
"""

from .types import (
    PrecursorInfo,
    FittingParameters,
    PeptideSpectralData,
    SpectrumMatrix,
    BasicFeatures,
    SimilarityMetrics,
    StatisticalFeatures,
    FragmentInfo,
    SpectralFeatures,
    SpectralFitResult
)

__all__ = [
    'PrecursorInfo',
    'FittingParameters', 
    'PeptideSpectralData',
    'SpectrumMatrix',
    'BasicFeatures',
    'SimilarityMetrics',
    'StatisticalFeatures',
    'FragmentInfo',
    'SpectralFeatures',
    'SpectralFitResult'
]