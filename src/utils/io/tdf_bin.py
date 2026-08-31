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

"""Decoder for Bruker ``analysis.tdf_bin``, the raw signal half of a ``.d``.

A ``.d`` folder holds two files that matter here:

- ``analysis.tdf`` -- a plain SQLite database with all the metadata (frames,
  retention times, DIA windows, calibration coefficients).  Read elsewhere,
  with ``sqlite3``.
- ``analysis.tdf_bin`` -- nothing but compressed per-frame signal blobs,
  concatenated.  No header, no index, no metadata.  This module reads it.

The link between the two is ``Frames.TimsId``, the byte offset of that frame's
blob.  Each blob decodes to ``(scan, tof_index, intensity)`` integer triples;
every physical quantity (m/z, 1/K0, RT) is derived afterwards from calibration
tables, not stored here.

Blob layout, once decompressed and un-transposed into ``uint32``:

===========================  ====================================
index                        contents
===========================  ====================================
``0``                        ``scan_count``
``1 .. scan_count-1``        per-scan sizes (halved -> peak count)
``scan_count + 2*i``         tof index of peak ``i``
``scan_count + 1 + 2*i``     intensity of peak ``i``
===========================  ====================================

with two wrinkles: the ``uint32`` array is stored **byte-transposed** (all the
low bytes of every value first, then all the second bytes, and so on -- a
shuffle that makes the data compress better), and tof indices are
**delta-encoded within each scan**, needing a cumulative sum and a ``-1``.

Only ``TimsCompressionType = 2`` (zstd) is supported.  Type 1 is an older LZF
scheme used by pre-2018 acquisitions; it raises rather than guessing.
"""

import sqlite3
import struct

import numpy as np
import pyarrow as pa

from src.logger import logger

# Two uint32 words -- byte count, then scan count -- precede each blob's payload.
_HEADER_BYTES = 8

ZSTD_COMPRESSION = 2
LZF_COMPRESSION = 1


class TdfBinError(RuntimeError):
    """Raised when a .tdf_bin cannot be decoded."""


class TdfBinReader:
    """Random-access reader over the frame blobs in ``analysis.tdf_bin``.

    Frames are addressed by their ``Frames.Id`` from ``analysis.tdf``.  The
    frame table is read once at construction; the binary itself is read lazily,
    one frame at a time, so a 6 GB acquisition never has to be resident.

    Intended use is a single pass::

        with TdfBinReader(d_path) as reader:
            for frame_id in reader.frame_ids:
                scans, tofs, intensities = reader.read_frame(frame_id)
    """

    def __init__(self, d_path: str):
        self.d_path = d_path.rstrip("/")
        self.tdf_path = f"{self.d_path}/analysis.tdf"
        self.bin_path = f"{self.d_path}/analysis.tdf_bin"

        conn = sqlite3.connect(self.tdf_path)
        try:
            compression = conn.execute(
                "SELECT Value FROM GlobalMetadata WHERE Key = 'TimsCompressionType'"
            ).fetchone()
            frame_rows = conn.execute(
                "SELECT Id, TimsId, NumScans, NumPeaks FROM Frames ORDER BY Id"
            ).fetchall()
        finally:
            conn.close()

        if compression is None:
            raise TdfBinError(
                f"TimsCompressionType missing from {self.tdf_path}; cannot "
                f"determine how frame blobs are compressed."
            )
        self.compression_type = int(compression[0])
        if self.compression_type == LZF_COMPRESSION:
            raise TdfBinError(
                f"{self.d_path} uses TimsCompressionType 1 (LZF), which is not "
                f"supported. Only type 2 (zstd) is implemented."
            )
        if self.compression_type != ZSTD_COMPRESSION:
            raise TdfBinError(
                f"{self.d_path} uses unknown TimsCompressionType "
                f"{self.compression_type}; expected 2 (zstd)."
            )

        # TimsId is NULL for frames with no stored signal; drop them here so
        # callers never have to special-case an unreadable frame.
        self._frames = {
            int(fid): (int(tims_id), int(n_scans), int(n_peaks))
            for fid, tims_id, n_scans, n_peaks in frame_rows
            if tims_id is not None
        }
        self.frame_ids = np.array(sorted(self._frames), dtype=np.uint32)
        self._fh = None

    def __enter__(self):
        self._fh = open(self.bin_path, "rb")
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def close(self):
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def read_frame(self, frame_id: int):
        """Decode one frame into ``(scans, tof_indices, intensities)``.

        All three are parallel arrays with one entry per peak: ``scans`` is the
        mobility scan number the peak was found in, ``tof_indices`` the
        digitizer sample number (*not* an m/z), ``intensities`` the raw detector
        count.

        Intensities are returned uncalibrated.  Bruker's own
        ``Frames.SummedIntensities`` applies a factor of
        ``1000 / AccumulationTime``; the centroider this path replaces does not
        apply it either, so leaving it off keeps the two ingest paths on the
        same intensity scale.
        """
        if self._fh is None:
            raise TdfBinError("TdfBinReader used outside a `with` block")

        entry = self._frames.get(int(frame_id))
        if entry is None:
            return _empty_frame()
        offset, n_scans, n_peaks = entry
        if n_peaks == 0:
            return _empty_frame()

        # The frame table tells us the decompressed size exactly, which is what
        # lets pyarrow decompress this without a separate zstd dependency:
        # one word per scan, plus a (tof, intensity) pair per peak.
        n_words = n_scans + 2 * n_peaks

        self._fh.seek(offset)
        header = self._fh.read(_HEADER_BYTES)
        if len(header) < _HEADER_BYTES:
            raise TdfBinError(
                f"Frame {frame_id}: truncated header at byte {offset}"
            )
        byte_count, blob_scan_count = struct.unpack("<II", header)
        if blob_scan_count != n_scans:
            raise TdfBinError(
                f"Frame {frame_id}: blob declares {blob_scan_count} scans but "
                f"analysis.tdf says {n_scans}; .d is inconsistent or corrupt."
            )

        payload = self._fh.read(byte_count - _HEADER_BYTES)
        try:
            raw = pa.decompress(
                payload, codec="zstd", decompressed_size=4 * n_words
            )
        except Exception as exc:
            raise TdfBinError(
                f"Frame {frame_id}: zstd decompression failed at byte {offset}"
            ) from exc

        return _decode_blob(memoryview(raw), n_scans, n_peaks)


def _empty_frame():
    return (
        np.empty(0, dtype=np.uint32),
        np.empty(0, dtype=np.uint32),
        np.empty(0, dtype=np.uint32),
    )


def _decode_blob(raw: memoryview, n_scans: int, n_peaks: int):
    """Un-transpose, de-interleave and un-delta one decompressed frame blob."""
    n_words = n_scans + 2 * n_peaks

    # Un-transpose: the blob stores byte plane 0 of every word, then plane 1,
    # and so on, so reading it as (4, n_words) puts each plane on a row.
    planes = np.frombuffer(raw, dtype=np.uint8, count=4 * n_words).reshape(
        4, n_words
    )
    words = (
        planes[0].astype(np.uint32)
        | (planes[1].astype(np.uint32) << 8)
        | (planes[2].astype(np.uint32) << 16)
        | (planes[3].astype(np.uint32) << 24)
    )

    # Per-scan peak counts. Sizes are stored in half-words, and the final scan's
    # size is implied by the total rather than stored.
    sizes = (words[1:n_scans] // 2).astype(np.int64)
    offsets = np.empty(n_scans + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(sizes, out=offsets[1:n_scans])
    offsets[n_scans] = n_peaks

    counts = np.diff(offsets)
    if counts.min() < 0 or offsets[n_scans - 1] > n_peaks:
        raise TdfBinError("Corrupt frame: scan offsets are not monotonic")

    body = words[n_scans:]
    tof_deltas = body[0::2].astype(np.int64)
    intensities = body[1::2]

    # Tof indices are cumulative within each scan, offset by one. Subtracting
    # each scan's exclusive prefix converts the global cumsum to a per-scan one.
    running = np.cumsum(tof_deltas)
    starts = offsets[:-1]
    nonempty = counts > 0
    prefix = np.zeros(n_scans, dtype=np.int64)
    prefix[nonempty] = running[starts[nonempty]] - tof_deltas[starts[nonempty]]
    tof_indices = running - np.repeat(prefix, counts) - 1

    scans = np.repeat(np.arange(n_scans, dtype=np.uint32), counts)

    return scans, tof_indices.astype(np.uint32), intensities


def frame_peak_counts(d_path: str) -> int:
    """Total stored peaks across the acquisition, from metadata alone.

    Reads ``SUM(NumPeaks)`` without touching the binary, so callers can size
    buffers or report progress before decoding anything.
    """
    conn = sqlite3.connect(f"{d_path.rstrip('/')}/analysis.tdf")
    try:
        total = conn.execute("SELECT SUM(NumPeaks) FROM Frames").fetchone()[0]
    finally:
        conn.close()
    return int(total or 0)
