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

from src.logger import logger


class JModError(Exception):
    """An error with a message meant for the user: a bad path, an unknown tag,
    an unsupported option.

    Raise it wherever the problem is found; it travels up to run_jmod.main,
    which reports it with report_error and either moves on to the next run or
    ends the experiment.
    """


def report_error(e, context):
    """Log an error where the user will see it.

    The console, the log file and the GUI's log panel all show the appLogger's
    records.  A JModError is the user's to fix and is logged as its message; any
    other exception is a bug and is logged with its traceback.  *context* says
    what the error stopped, e.g. "Run 2 of 5 failed (file.mzML)".
    """
    if isinstance(e, JModError):
        logger.error(f"{context}: {e}")
    else:
        logger.error(f"{context}: unexpected error", exc_info=e)
