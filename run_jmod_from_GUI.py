#!/usr/bin/env python3
"""
Wrapper script to run JMod from the project root directory.
This allows you to keep the main script in src/ while still running from root.
"""

#  Copyright (c) 2026 Parallel Squared Technology Institute
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#          http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import sys
import os
import multiprocessing
import tqdm

#Give tqdm somewhere to write to if running without a console (Pyinstaller Compiled)
if sys.stdout is None:
    sys.stdout = open(os.devnull, 'w')
if sys.stderr is None:
    sys.stderr = open(os.devnull, 'w')

# Add src directory to Python path
src_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, src_path)

if __name__ == "__main__":
    multiprocessing.freeze_support()  ###This must be the first line under if __name__ == "__main__" to not crash pyinstaller compiled JMod 
    from src import logger
    # Import and run the main modul
    from src.run_jmod_from_GUI import make_GUI
    make_GUI()