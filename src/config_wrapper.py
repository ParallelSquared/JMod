"""
Configuration wrapper for spectral fitting.

This module provides a clean interface to configuration values,
eliminating direct global config access throughout the codebase.
"""

from dataclasses import dataclass
from typing import Optional, Any
import src.config as global_config


@dataclass
class SpectralFittingConfig:
    """Configuration container for spectral fitting parameters."""
    # MS tolerances
    rt_tol: float
    ms1_tol: float
    mz_tol: float
    
    # Peak matching parameters
    top_n: int
    atleast_m: int
    frac_matched: float
    
    # Fitting parameters
    unmatched_fit_type: str
    
    # Optional parameters
    protein_column: Optional[str] = None
    mzml_file: Optional[str] = None
    
    @classmethod
    def from_global_config(cls) -> 'SpectralFittingConfig':
        """Create configuration from global config module."""
        return cls(
            rt_tol=getattr(global_config, 'rt_tol', 0.5),
            ms1_tol=getattr(global_config, 'ms1_tol', 25.0),
            mz_tol=getattr(global_config, 'mz_tol', 10e-6),
            top_n=getattr(global_config, 'top_n', 10),
            atleast_m=getattr(global_config, 'atleast_m', 3),
            frac_matched=getattr(global_config, 'frac_matched', 0.25),
            unmatched_fit_type=getattr(global_config, 'unmatched_fit_type', 'a'),
            protein_column=getattr(global_config, 'protein_column', None),
            mzml_file=get_mzml_file(global_config)
        )
    
    @classmethod
    def with_overrides(cls, **kwargs) -> 'SpectralFittingConfig':
        """Create configuration with specific overrides."""
        base_config = cls.from_global_config()
        for key, value in kwargs.items():
            if hasattr(base_config, key):
                setattr(base_config, key, value)
        return base_config


def get_mzml_file(config: Any) -> Optional[str]:
    """Safely extract mzML file path from config."""
    try:
        if hasattr(config, 'args') and hasattr(config.args, 'mzml'):
            return config.args.mzml
    except:
        pass
    return None


class ConfigManager:
    """Manager for configuration access throughout spectral fitting."""
    
    def __init__(self, config: Optional[SpectralFittingConfig] = None):
        """Initialize with optional config override."""
        self._config = config or SpectralFittingConfig.from_global_config()
    
    @property
    def config(self) -> SpectralFittingConfig:
        """Get current configuration."""
        return self._config
    
    def update(self, **kwargs):
        """Update configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)
    
    def get_tolerance_params(self) -> dict:
        """Get all tolerance parameters as a dict."""
        return {
            'rt_tol': self._config.rt_tol,
            'ms1_tol': self._config.ms1_tol,
            'mz_tol': self._config.mz_tol
        }
    
    def get_matching_params(self) -> dict:
        """Get peak matching parameters as a dict."""
        return {
            'top_n': self._config.top_n,
            'atleast_m': self._config.atleast_m,
            'frac_matched': self._config.frac_matched
        }