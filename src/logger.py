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
import logging.config
import time
import os
import sys 
import importlib

class ElapsedFormatter(logging.Formatter):
    def __init__(self, fmt=None, datefmt=None, style="%"):
        super().__init__(fmt, datefmt, style)
        self.start_time = time.time()

    def formatTime(self, record, datefmt=None):
        elapsed = record.created - self.start_time
        hours, rem = divmod(int(elapsed), 3600)
        minutes, seconds = divmod(rem, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    def reset_start_time(self):
        self.start_time = time.time()
    
def set_log_filepath(logfile_path):
    with open (logfile_path, "w"):
        pass
    logger = logging.getLogger("appLogger")

    existing_formatters = [
        h.formatter for h in logger.handlers
        if isinstance(h.formatter, ElapsedFormatter)
    ]
    start_time = existing_formatters[0].start_time if existing_formatters else time.time()

    for h in logger.handlers[:]:
        if isinstance(h, logging.FileHandler):
            logger.removeHandler(h)
            h.close()
    file_handler = logging.FileHandler(logfile_path, mode="a")
    file_handler.setLevel(logging.DEBUG)
    fmt = ElapsedFormatter("%(asctime)s - %(levelname)s - %(message)s")
    fmt.start_time = start_time 
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)


# load config
config_file = os.path.join(os.path.dirname(__file__), "logging.conf")
logging.config.fileConfig(config_file, disable_existing_loggers=False)

# patch handlers for elapsed time
for handler in logging.getLogger("appLogger").handlers:
    handler.setFormatter(
        ElapsedFormatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    
logger = logging.getLogger("appLogger")


