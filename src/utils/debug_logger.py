"""
Debug logging utilities for JMod.

This module provides a centralized logging system that writes debug output
to a file instead of cluttering the terminal.
"""

import logging
import os
from typing import Optional
import random


def setup_debug_logger(results_folder: str, log_level: str = "DEBUG") -> logging.Logger:
    """
    Set up a debug logger that writes to a file in the results folder.
    
    Args:
        results_folder: Path to the results folder where debug.log will be created
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger('jmod_debug')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove any existing handlers to avoid duplicates
    logger.handlers = []
    
    # Create file handler
    log_file = os.path.join(results_folder, 'debug.log')
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(getattr(logging, log_level.upper()))
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(file_handler)
    
    # Log initialization
    logger.info(f"Debug logging initialized. Writing to: {log_file}")
    logger.info(f"Log level: {log_level}")
    
    return logger


def get_debug_logger(module_name: Optional[str] = None) -> logging.Logger:
    """
    Get a debug logger instance.
    
    Args:
        module_name: Optional module name for the logger
        
    Returns:
        Logger instance
    """
    if module_name:
        return logging.getLogger(f'jmod_debug.{module_name}')
    return logging.getLogger('jmod_debug')


# Global variable to store whether debug logging is enabled
_debug_enabled = True


def set_debug_enabled(enabled: bool):
    """Enable or disable debug logging globally."""
    global _debug_enabled
    _debug_enabled = enabled
    
    # Update all jmod_debug loggers
    logger = logging.getLogger('jmod_debug')
    if enabled:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.WARNING)


def is_debug_enabled() -> bool:
    """Check if debug logging is enabled."""
    return _debug_enabled


class SamplingLogger:
    """
    A logger wrapper that only logs a sample of messages to improve performance.
    """
    def __init__(self, logger: logging.Logger, sample_rate: float = 0.001):
        """
        Initialize the sampling logger.
        
        Args:
            logger: The underlying logger to use
            sample_rate: Fraction of messages to log (0.001 = 1 in 1000)
        """
        self.logger = logger
        self.sample_rate = sample_rate
        self._counter = 0
        self._logged_count = 0
        
    def should_log(self) -> bool:
        """Determine if this message should be logged based on sampling."""
        self._counter += 1
        if random.random() < self.sample_rate:
            self._logged_count += 1
            return True
        return False
        
    def debug(self, msg: str, *args, **kwargs):
        """Log a debug message with sampling."""
        if self.should_log():
            # Add sampling info to the message
            sampled_msg = f"[Sample {self._logged_count}/{self._counter}] {msg}"
            self.logger.debug(sampled_msg, *args, **kwargs)
            
    def info(self, msg: str, *args, **kwargs):
        """Log an info message (always logged)."""
        self.logger.info(msg, *args, **kwargs)
        
    def warning(self, msg: str, *args, **kwargs):
        """Log a warning message (always logged)."""
        self.logger.warning(msg, *args, **kwargs)
        
    def error(self, msg: str, *args, **kwargs):
        """Log an error message (always logged)."""
        self.logger.error(msg, *args, **kwargs)
        
    def get_summary(self) -> str:
        """Get a summary of sampling statistics."""
        return f"Logged {self._logged_count} out of {self._counter} debug messages ({self._logged_count/self._counter*100:.2f}%)"


def get_sampling_logger(module_name: Optional[str] = None, sample_rate: float = 0.001) -> SamplingLogger:
    """
    Get a sampling logger instance for high-frequency logging.
    
    Args:
        module_name: Optional module name for the logger
        sample_rate: Fraction of messages to log (default: 0.001 = 1 in 1000)
        
    Returns:
        SamplingLogger instance
    """
    base_logger = get_debug_logger(module_name)
    logger = SamplingLogger(base_logger, sample_rate)
    register_sampling_logger(logger)
    return logger


# Global sampling rate
_global_sample_rate = 0.001

# Track all sampling loggers
_sampling_loggers = []


def set_global_sample_rate(rate: float):
    """Set the global sampling rate for all sampling loggers."""
    global _global_sample_rate
    _global_sample_rate = rate
    

def get_global_sample_rate() -> float:
    """Get the current global sampling rate."""
    return _global_sample_rate


def register_sampling_logger(logger: SamplingLogger):
    """Register a sampling logger for tracking."""
    _sampling_loggers.append(logger)
    

def get_all_sampling_summaries() -> str:
    """Get summaries from all registered sampling loggers."""
    summaries = []
    for logger in _sampling_loggers:
        if logger._counter > 0:
            summaries.append(f"{logger.logger.name}: {logger.get_summary()}")
    return "\n".join(summaries)