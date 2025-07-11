"""
Example of pipeline-based refactoring for spectral fitting functions.

This demonstrates how the large fit_to_lib and fit_to_lib2 functions
could be refactored into smaller, testable pipeline steps.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any, Union
import numpy as np

from .spectral_fitting_config import SpectralFittingConfig


@dataclass
class PipelineContext:
    """Context object passed through pipeline steps."""
    # Input data
    dia_spec: Any
    library: Dict[Tuple, Dict]
    rt_mz: np.ndarray
    all_keys: List[Tuple]
    
    # Configuration
    config: SpectralFittingConfig
    
    # Optional inputs
    dino_features: Optional[Any] = None
    rt_filter: bool = False
    ms1_mz: Optional[np.ndarray] = None
    ms1_spectra: Optional[List] = None
    
    # Processing state
    spec_idx: Optional[int] = None
    dia_spectrum: Optional[np.ndarray] = None
    prec_mz: Optional[float] = None
    prec_rt: Optional[float] = None
    window_width: Optional[float] = None
    ms1_spec: Optional[Any] = None
    
    # Candidates
    window_idxs: Optional[np.ndarray] = None
    mass_window_candidates: Optional[List] = None
    candidate_peaks: Optional[List] = None
    
    # Processing results
    centroid_breaks: Optional[np.ndarray] = None
    bin_centers: Optional[np.ndarray] = None
    ms1_peak: Optional[np.ndarray] = None
    ref_peaks_in_dia: Optional[List] = None
    
    # Final results
    output: Optional[List] = None
    frag_info: Optional[Dict] = None


class PipelineStep(ABC):
    """Abstract base class for pipeline steps."""
    
    @abstractmethod
    def execute(self, context: PipelineContext) -> PipelineContext:
        """Execute the pipeline step."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the pipeline step."""
        pass


class ExtractSpectrumInfoStep(PipelineStep):
    """Extract basic spectrum information."""
    
    @property
    def name(self) -> str:
        return "ExtractSpectrumInfo"
    
    def execute(self, context: PipelineContext) -> PipelineContext:
        """Extract spectrum information from DIA spec."""
        context.spec_idx = context.dia_spec.scan_num
        context.dia_spectrum = np.stack(context.dia_spec.peak_list(), 1)
        context.prec_mz = context.dia_spec.prec_mz
        context.prec_rt = context.dia_spec.RT
        # context.window_width = calculate_window_width(context.dia_spec)
        return context


class FindMS1SpectrumStep(PipelineStep):
    """Find closest MS1 spectrum."""
    
    @property
    def name(self) -> str:
        return "FindMS1Spectrum"
    
    def execute(self, context: PipelineContext) -> PipelineContext:
        """Find closest MS1 spectrum if available."""
        if context.ms1_spectra is not None:
            # context.ms1_spec = get_closest_ms1(context.prec_rt, context.ms1_spectra)
            pass
        return context


class FilterCandidatesByWindowStep(PipelineStep):
    """Filter candidates by mass window."""
    
    @property
    def name(self) -> str:
        return "FilterCandidatesByWindow"
    
    def execute(self, context: PipelineContext) -> PipelineContext:
        """Filter candidates within mass window."""
        # This would call filter_candidates_by_window
        # context.window_idxs, context.mass_window_candidates = filter_candidates_by_window(...)
        return context


class FilterDecoysStep(PipelineStep):
    """Filter out decoy peptides (RT alignment only)."""
    
    @property
    def name(self) -> str:
        return "FilterDecoys"
    
    def execute(self, context: PipelineContext) -> PipelineContext:
        """Filter out decoys for RT alignment."""
        if context.config.get('filter_decoys', False):
            context.mass_window_candidates = [
                key for key in context.mass_window_candidates
                if not context.library[key].get('is_decoy', False)
            ]
        return context


class PreprocessDIASpectrumStep(PipelineStep):
    """Preprocess DIA spectrum."""
    
    @property
    def name(self) -> str:
        return "PreprocessDIASpectrum"
    
    def execute(self, context: PipelineContext) -> PipelineContext:
        """Preprocess DIA spectrum by merging peaks."""
        # This would call preprocess_dia_spectrum
        # context.dia_spectrum, context.centroid_breaks, context.bin_centers = preprocess_dia_spectrum(...)
        return context


class CheckMS1PeaksStep(PipelineStep):
    """Check MS1 peaks for candidates."""
    
    @property
    def name(self) -> str:
        return "CheckMS1Peaks"
    
    def execute(self, context: PipelineContext) -> PipelineContext:
        """Check MS1 peaks if MS1 spectrum available."""
        if context.ms1_spec is not None:
            # context.ms1_peak = check_ms1_peaks(...)
            pass
        else:
            context.ms1_peak = np.ones(len(context.mass_window_candidates), dtype=bool)
        return context


class SpectralFittingPipeline:
    """
    Base pipeline for spectral fitting.
    
    This demonstrates how to organize pipeline steps for better
    modularity and testability.
    """
    
    def __init__(self, config: SpectralFittingConfig):
        """Initialize pipeline with configuration."""
        self.config = config
        self.steps: List[PipelineStep] = []
    
    def add_step(self, step: PipelineStep) -> 'SpectralFittingPipeline':
        """Add a step to the pipeline."""
        self.steps.append(step)
        return self
    
    def execute(
        self,
        dia_spec: Any,
        library: Dict[Tuple, Dict],
        rt_mz: np.ndarray,
        all_keys: List[Tuple],
        **kwargs
    ) -> Any:
        """Execute the pipeline."""
        # Create initial context
        context = PipelineContext(
            dia_spec=dia_spec,
            library=library,
            rt_mz=rt_mz,
            all_keys=all_keys,
            config=self.config,
            **kwargs
        )
        
        # Execute each step
        for step in self.steps:
            try:
                context = step.execute(context)
            except Exception as e:
                raise RuntimeError(f"Error in step {step.name}: {str(e)}")
        
        return context.output


class RTAlignmentPipeline(SpectralFittingPipeline):
    """Pipeline for RT alignment (fit_to_lib replacement)."""
    
    def __init__(self, config: SpectralFittingConfig):
        """Initialize RT alignment pipeline."""
        super().__init__(config)
        
        # Configure pipeline steps
        self.add_step(ExtractSpectrumInfoStep())
        self.add_step(FindMS1SpectrumStep())
        self.add_step(FilterCandidatesByWindowStep())
        self.add_step(FilterDecoysStep())  # RT alignment specific
        self.add_step(PreprocessDIASpectrumStep())
        self.add_step(CheckMS1PeaksStep())
        # Add more steps...


class FullSpectraFittingPipeline(SpectralFittingPipeline):
    """Pipeline for full spectral fitting (fit_to_lib2 replacement)."""
    
    def __init__(self, config: SpectralFittingConfig):
        """Initialize full fitting pipeline."""
        super().__init__(config)
        
        # Configure pipeline steps (no decoy filtering)
        self.add_step(ExtractSpectrumInfoStep())
        self.add_step(FindMS1SpectrumStep())
        self.add_step(FilterCandidatesByWindowStep())
        # No FilterDecoysStep - process all candidates
        self.add_step(PreprocessDIASpectrumStep())
        # Add more steps...


def create_pipeline(mode: str, config: SpectralFittingConfig) -> SpectralFittingPipeline:
    """
    Factory function to create appropriate pipeline.
    
    Args:
        mode: 'rt_alignment' or 'full'
        config: Configuration object
        
    Returns:
        Configured pipeline instance
    """
    if mode == 'rt_alignment':
        return RTAlignmentPipeline(config)
    elif mode == 'full':
        return FullSpectraFittingPipeline(config)
    else:
        raise ValueError(f"Unknown mode: {mode}")