#!/usr/bin/env python3
"""
This Source Code Form is subject to the terms of the Oxford Nanopore
Technologies, Ltd. Public License, v. 1.0.  Full licence can be found
at https://github.com/ParallelSquared/JMod/blob/main/LICENSE.txt
"""

"""
Wrapper script to run JMod from the project root directory.
This allows you to keep the main script in src/ while still running from root.
"""

import sys
import os
import multiprocessing

# Add src directory to Python path
src_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, src_path)

if __name__ == "__main__":
    multiprocessing.freeze_support()  ###This must be the first line under if __name__ == "__main__" to not crash pyinstaller compiled JMod 
    from src import logger
    # Import and run the main modul
    from src.run_jmod_from_GUI import make_GUI
    make_GUI()