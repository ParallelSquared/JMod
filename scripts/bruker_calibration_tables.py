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

"""Write a .d's m/z and 1/K0 calibration tables, computed by Bruker's timsdata.

Calls the unmodified Bruker library through its documented C API (timsdata.h)
-- tims_open_v2, tims_index_to_mz, tims_scannum_to_oneoverk0, tims_close --
exactly as peppy_sage's tims_centroid does: the acquisition-time calibration
(use_recalibrated_state = 0), no pressure compensation, frame 1, every TOF
index 0..DigitizerNumSamples and every scan 0..max(Frames.NumScans).

Runs wherever the library does: natively on Windows or Linux, or on a Mac
under Wine with the Windows library and a Windows Python.  Standard library
only, so a bare (embeddable) Python is enough.

    python bruker_calibration_tables.py <timsdata.dll|libtimsdata.so> <run.d> <out_dir>

writes <out_dir>/mz.npy (float64, indexed by TOF index), <out_dir>/ook0.npy
(float64, indexed by scan number) and <out_dir>/calibration.json (what was
called, on what).  It only reads the .d.
"""

import array
import ctypes
import json
import os
import sqlite3
import sys

USE_RECALIBRATED_STATE = 0      # acquisition-time calibration
NO_PRESSURE_COMPENSATION = 0    # pressure_compensation_strategy, timsdata_common.h
FRAME_ID = 1                    # one calibration per run, as tims_centroid assumes


def _load(lib_path):
    lib = ctypes.CDLL(os.path.abspath(lib_path))
    lib.tims_open_v2.argtypes = [ctypes.c_char_p, ctypes.c_uint32, ctypes.c_int]
    lib.tims_open_v2.restype = ctypes.c_uint64
    lib.tims_close.argtypes = [ctypes.c_uint64]
    lib.tims_close.restype = None
    lib.tims_get_last_error_string.argtypes = [ctypes.c_char_p, ctypes.c_uint32]
    lib.tims_get_last_error_string.restype = ctypes.c_uint32
    for name in ("tims_index_to_mz", "tims_scannum_to_oneoverk0"):
        f = getattr(lib, name)
        f.argtypes = [ctypes.c_uint64, ctypes.c_int64, ctypes.POINTER(ctypes.c_double),
                      ctypes.POINTER(ctypes.c_double), ctypes.c_uint32]
        f.restype = ctypes.c_uint32
    return lib


def _last_error(lib):
    buf = ctypes.create_string_buffer(2048)
    lib.tims_get_last_error_string(buf, len(buf))
    return buf.value.decode("utf-8", "replace")


def _convert(lib, func, handle, max_index):
    """func over 0..max_index inclusive, in one call."""
    n = max_index + 1
    values = (ctypes.c_double * n)(*range(n))
    out = (ctypes.c_double * n)()
    if func(handle, FRAME_ID, values, out, n) == 0:
        raise RuntimeError(f"{func.__name__} failed: {_last_error(lib)}")
    return array.array("d", out)


def _write_npy(path, values):
    """A 1-D little-endian float64 .npy (format 1.0), without numpy."""
    header = "{'descr': '<f8', 'fortran_order': False, 'shape': (%d,), }" % len(values)
    header += " " * (63 - (10 + len(header)) % 64) + "\n"
    if sys.byteorder != "little":
        values = array.array("d", values)
        values.byteswap()
    with open(path, "wb") as f:
        f.write(b"\x93NUMPY\x01\x00" + len(header).to_bytes(2, "little") + header.encode("latin1"))
        f.write(values.tobytes())


def main(lib_path, d_path, out_dir):
    tdf = os.path.join(d_path, "analysis.tdf")
    conn = sqlite3.connect(f"file:{tdf}?mode=ro", uri=True)
    try:
        tof_max_index = int(conn.execute(
            "SELECT Value FROM GlobalMetadata WHERE Key = 'DigitizerNumSamples'").fetchone()[0])
        scan_max_index = int(conn.execute("SELECT MAX(NumScans) FROM Frames").fetchone()[0])
    finally:
        conn.close()

    lib = _load(lib_path)
    handle = lib.tims_open_v2(os.path.abspath(d_path).encode("utf-8"),
                              USE_RECALIBRATED_STATE, NO_PRESSURE_COMPENSATION)
    if handle == 0:
        raise RuntimeError(f"tims_open_v2 failed: {_last_error(lib)}")
    try:
        mz = _convert(lib, lib.tims_index_to_mz, handle, tof_max_index)
        ook0 = _convert(lib, lib.tims_scannum_to_oneoverk0, handle, scan_max_index)
    finally:
        lib.tims_close(handle)

    os.makedirs(out_dir, exist_ok=True)
    _write_npy(os.path.join(out_dir, "mz.npy"), mz)
    _write_npy(os.path.join(out_dir, "ook0.npy"), ook0)
    with open(os.path.join(out_dir, "calibration.json"), "w") as f:
        json.dump({
            "d_path": os.path.abspath(d_path),
            "library": os.path.basename(lib_path),
            "frame_id": FRAME_ID,
            "use_recalibrated_state": USE_RECALIBRATED_STATE,
            "pressure_compensation_strategy": NO_PRESSURE_COMPENSATION,
            "tof_max_index": tof_max_index,
            "scan_max_index": scan_max_index,
        }, f, indent=2)
    print(f"m/z {mz[0]:.6f}..{mz[-1]:.6f} over {len(mz)} TOF indices; "
          f"1/K0 {ook0[0]:.6f}..{ook0[-1]:.6f} over {len(ook0)} scans")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    main(*sys.argv[1:])
