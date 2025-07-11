"""
Configuration management for spectral fitting.

This module provides a configuration wrapper to eliminate global dependencies
and improve testability.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class SpectralFittingConfig:
    """
    Configuration for spectral fitting operations.
    
    This replaces direct access to the global config module.
    """
    # Tolerances
    mz_tol: float = 1e-5
    rt_tol: float = 0.5
    ms1_tol: float = 20e-6
    
    # Peak matching parameters
    top_n: int = 10
    atleast_m: int = 3
    lib_frac: float = 0.5
    score_lib_frac: float = 0.5
    
    # Fitting parameters
    unmatched: str = 'c'
    lower_limit: float = 1e-10
    
    # RT alignment parameters
    use_emp_rt: bool = False
    initial_percentile: int = 50
    
    # Feature calculation parameters
    frac_matched: float = 0.5
    
    # Column names
    protein_column: str = 'protein'
    
    # File paths and names
    mzml: Optional[str] = None
    
    # Additional parameters
    additional_params: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_global_config(cls, global_config) -> 'SpectralFittingConfig':
        """
        Create configuration from global config module.
        
        Args:
            global_config: The global config module
            
        Returns:
            SpectralFittingConfig instance
        """
        return cls(
            mz_tol=getattr(global_config, 'mz_tol', cls.mz_tol),
            rt_tol=getattr(global_config, 'rt_tol', cls.rt_tol),
            ms1_tol=getattr(global_config, 'ms1_tol', cls.ms1_tol),
            top_n=getattr(global_config, 'top_n', cls.top_n),
            atleast_m=getattr(global_config, 'atleast_m', cls.atleast_m),
            lib_frac=getattr(global_config, 'lib_frac', cls.lib_frac),
            score_lib_frac=getattr(global_config, 'score_lib_frac', cls.score_lib_frac),
            unmatched=getattr(global_config, 'unmatched', cls.unmatched),
            lower_limit=getattr(global_config, 'lower_limit', cls.lower_limit),
            use_emp_rt=getattr(global_config, 'use_emp_rt', cls.use_emp_rt),
            initial_percentile=getattr(global_config, 'initial_percentile', cls.initial_percentile),
            protein_column=getattr(global_config, 'protein_column', cls.protein_column),
            mzml=getattr(global_config.args, 'mzml', None) if hasattr(global_config, 'args') else None
        )
    
    def update(self, **kwargs):
        """Update configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.additional_params[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with default."""
        if hasattr(self, key):
            return getattr(self, key)
        return self.additional_params.get(key, default)


@dataclass
class RTAlignmentConfig(SpectralFittingConfig):
    """Configuration specific to RT alignment."""
    frac_matched: float = 0.8  # Higher threshold for RT alignment
    filter_decoys: bool = True  # Always filter decoys for RT alignment


@dataclass
class FullFittingConfig(SpectralFittingConfig):
    """Configuration specific to full spectral fitting."""
    filter_decoys: bool = False  # Include decoys for FDR calculation
    calculate_fdr: bool = True