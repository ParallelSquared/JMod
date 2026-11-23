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

import os

from src.utils.errors import JModError
from src.utils.io.load_files import SpectrumFile
from src.utils.io.bruker_d import _load_d
from src.logger import logger


class FileReader:
    """Dispatch to format-specific parsers based on file extension."""

    _FORMATS = {".mzml", ".d", ".raw"}

    def __init__(self, filepath: str, bruker_sdk_path: str = None):
        self.filepath = filepath.replace("\\", "/")
        self.bruker_sdk_path = bruker_sdk_path
        self.format = self._detect_format()

    def _detect_format(self) -> str:
        if self.filepath.rstrip("/").endswith(".d"):
            return ".d"
        _, ext = os.path.splitext(self.filepath)
        ext = ext.lower()
        if ext not in self._FORMATS:
            raise ValueError(
                f"Unsupported file format '{ext}'. Supported: {sorted(self._FORMATS)}"
            )
        return ext

    def read(self) -> SpectrumFile:
        if self.format == ".mzml":
            return _load_mzml(self.filepath)
        elif self.format == ".d":
            return _load_d(self.filepath, self.bruker_sdk_path)
        elif self.format == ".raw":
            return _load_raw(self.filepath)


def _load_mzml(filepath: str) -> SpectrumFile:
    logger.info("Loading Spectra... from .mzML file")
    return SpectrumFile(mzml_file=filepath)


def _load_raw(filepath: str) -> SpectrumFile:
    logger.info("Loading Spectra... from .raw file")
    return SpectrumFile(raw_file=filepath)


def loadSpectra(input_file: str, bruker_sdk_path: str = None) -> SpectrumFile:
    """Drop-in replacement for load_files.loadSpectra with format dispatch."""
    logger.info("Loading Spectra...")
    # python_spec_file = input_file + "_pythonspec"
    # if not os.path.exists(python_spec_file):
    reader = FileReader(input_file, bruker_sdk_path)
    spectra = reader.read()
    #     with open(python_spec_file, "wb") as f:
    #         pickle.dump(spectra, f)
    # else:
    #     with open(python_spec_file, "rb") as f:
    #         logger.info("Loading Spectra... from pickle")
    #         spectra = pickle.load(f)

    logger.info(f"Loaded {len(spectra.ms1scans)} MS1 spectra")
    logger.info(f"Loaded {len(spectra.ms2scans)} MS2 spectra")
    logger.info("finished")
    return spectra

def load_rawfilereader(sdk_path):
    import sys
    from pathlib import Path

    try:
        import clr
    except Exception as e:
        if sys.platform == "darwin":
            raise JModError("Failure on import clr. Please use 'brew install mono' to use Thermo RawFileReader .dlls or convert to mzML")
        else:
            raise e
        
    try:
        sdk_path = Path(sdk_path)
    except:
        raise JModError(f"Thermo RawFileReader path does not exist: {sdk_path}\n          If using command line please add with --rawfilereader_path 'path' or manually update in JMod/Data/Settings.json.")

    if not sdk_path.exists():
        raise JModError(f"Thermo RawFileReader path could not be found: {sdk_path}")

    if str(sdk_path) not in sys.path:
        sys.path.append(str(sdk_path))

    logger.info("Attempting to load Thermo RawFileReader .dlls")

    try:
        clr.AddReference("ThermoFisher.CommonCore.Data")
        clr.AddReference("ThermoFisher.CommonCore.RawFileReader")
        from ThermoFisher.CommonCore.RawFileReader import RawFileReaderAdapter # type: ignore
        logger.info("Thermo RawFileReader .dlls loaded successfully")
    except Exception as e:
        error_message = ("Failed to load the Thermo RawFileReader SDK.\n\n"
                        "Please verify that:\n"
                        "  1. The RawFileReader SDK path is correct.\n"
                        "  2. The folder contains the ThermoFisher.CommonCore*.dll files.\n"
                        "  3. 'ThermoFisher.CommonCore.Data.dll' and 'ThermoFisher.CommonCore.RawFileReader.dll' have been unblocked after downloading "
                        "(Right-click the ZIP → Properties → Unblock.\n"
                        "  4. The required .NET runtime is installed.\n\n"
                        f"Configured SDK path: {sdk_path}\n\n"
                        f"Original error:\n{e}")
        raise JModError(error_message) from e
