"""
This Source Code Form is subject to the terms of the Oxford Nanopore
Technologies, Ltd. Public License, v. 1.0.  Full licence can be found
at https://github.com/ParallelSquared/JMod/blob/main/LICENSE.txt
"""

"""
Alternative conftest.py that avoids the config import issue
This approach modifies sys.argv before any imports
"""
import pytest
import sys
import os
from unittest.mock import Mock, patch
from src.logger import logger

# Save original argv
original_argv = sys.argv.copy()

# Set minimal argv to prevent argparse issues
sys.argv = ['pytest']

# Add the parent directory to the path so we can import JMod modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now we can safely import config without argparse errors
try:
    import src.config as config
    # Override any problematic attributes
    config.tag = None
    if not hasattr(config, 'diann_mods'):
        config.diann_mods = {}
except Exception as e:
    # If config still fails, create a complete mock
    logger.warning(f"Warning: Could not import config, using mock: {e}")
    config = Mock()
    config.tag = None
    config.diann_mods = {}
    config.args = Mock()
    config.args.tag = None
    sys.modules['config'] = config

# Restore original argv
sys.argv = original_argv

@pytest.fixture(autouse=True)
def reset_config():
    """Reset config for each test"""
    config.tag = None
    yield
    config.tag = None

@pytest.fixture
def diann_rules():
    """Fixture providing the DIANN rules dictionary"""
    return {
        'G': 'L',
        'A': 'L',
        'V': 'L',
        'L': 'V',
        'I': 'V',
        'F': 'L',
        'M': 'L',
        'P': 'L',
        'W': 'L',
        'S': 'T',
        'C': 'S',
        'T': 'S',
        'Y': 'S',
        'H': 'S',
        'K': 'L',
        'R': 'L',
        'Q': 'N',
        'E': 'D',
        'N': 'Q',
        'D': 'E'
    }

@pytest.fixture
def sample_sequences():
    """Fixture providing sample peptide sequences for testing"""
    return {
        'simple': 'PEPTIDE',
        'with_mod': 'PEP(+79.97)TIDE',
        'with_multiple_mods': 'P(+15.99)EP(+79.97)TID(+0.98)E',
        'with_tag': 'PEP(mTRAQ)TIDE',
        'complex': 'K(mTRAQ-0)PEPTIDE(+15.99)R'
    }