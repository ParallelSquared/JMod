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
import logging
import warnings

# TODO: remove once we figure out what imports psims at startup on Windows (jmod never writes mzMLb)
warnings.filterwarnings("ignore", message="hdf5plugin is missing")

# TODO: revisit if TF C++ INFO logs are ever needed; hides the benign oneDNN round-off notice
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
# TODO: remove once the dependency stops calling deprecated tf.losses.sparse_softmax_cross_entropy;
# note this silences ALL TF Python-side warnings, not just that one
logging.getLogger("tensorflow").setLevel(logging.ERROR)


# Add src directory to Python path
src_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, src_path)


if __name__ == "__main__":
    from src import logger
    # Import and run the main module
    from src.run_jmod import main
    main()

