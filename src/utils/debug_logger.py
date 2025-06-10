"""
Debug logging utilities for JMod.

This module provides a centralized logging system that writes debug output
to a file instead of cluttering the terminal.
"""

import logging
import os
from typing import Optional


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