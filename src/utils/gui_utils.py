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

import logging
import sys
import importlib
import os
import shutil
from pathlib import Path
import json
import os

# Anchored to the repo root, not the CWD: run_jmod_from_GUI.py resolves this same
# data/ folder as dirname(dirname(abspath(__file__))) for the app icon, so a
# CWD-relative constant here meant the GUI and load_settings disagreed about where
# settings.json lives whenever JMod was launched from anywhere but the repo root.
SETTINGS_DIR = Path(__file__).resolve().parents[2] / "data"
SETTINGS_FILE = SETTINGS_DIR / "settings.json"

DEFAULT_SETTINGS = {
    "rawfilereader_path": None,
    "bruker_sdk_path": None,
}

def load_settings():
    if not SETTINGS_FILE.exists():
        return DEFAULT_SETTINGS.copy()

    with open(SETTINGS_FILE) as f:
        settings = json.load(f)

    return DEFAULT_SETTINGS | settings

def save_settings(settings):
    # SETTINGS_DIR may not exist yet on a fresh checkout or install.
    SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
    with open(SETTINGS_FILE, "w") as f:
        json.dump(settings, f, indent=2)




