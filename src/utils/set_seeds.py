"""
This Source Code Form is subject to the terms of the Oxford Nanopore
Technologies, Ltd. Public License, v. 1.0.  Full licence can be found
at https://github.com/ParallelSquared/JMod/blob/main/LICENSE.txt
"""

import numpy as np

def set_seeds(seed):
    """
    Description: Sets all global seed values to config.seed. Must run at the beginning of each execution. When new
    libraries are added, set their seed values using this function so that they are reflected throughout the program.

    Parameters:
    ----------
    seed: int

    Returns:
    -------
    None

    Notes:
    ------

    """

    np.random.seed(seed)
