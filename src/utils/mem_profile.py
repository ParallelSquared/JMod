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

"""Memory profiling: process footprint, swap, and what the big objects hold.

On macOS the footprint is the kernel's ``phys_footprint`` -- what Activity
Monitor and ``footprint`` report, counting compressed and swapped-out pages,
which RSS does not.  Elsewhere it is RSS.
"""

import ctypes
import ctypes.util
import os
import sys
import threading

import numpy as np
import psutil

from src.logger import logger

GB = 1024 ** 3


class _RusageInfoV4(ctypes.Structure):
    # struct rusage_info_v4, <sys/resource.h>
    _fields_ = [("ri_uuid", ctypes.c_uint8 * 16)] + [(name, ctypes.c_uint64) for name in (
        "ri_user_time", "ri_system_time", "ri_pkg_idle_wkups", "ri_interrupt_wkups",
        "ri_pageins", "ri_wired_size", "ri_resident_size", "ri_phys_footprint",
        "ri_proc_start_abstime", "ri_proc_exit_abstime", "ri_child_user_time",
        "ri_child_system_time", "ri_child_pkg_idle_wkups", "ri_child_interrupt_wkups",
        "ri_child_pageins", "ri_child_elapsed_abstime", "ri_diskio_bytesread",
        "ri_diskio_byteswritten", "ri_cpu_time_qos_default", "ri_cpu_time_qos_maintenance",
        "ri_cpu_time_qos_background", "ri_cpu_time_qos_utility", "ri_cpu_time_qos_legacy",
        "ri_cpu_time_qos_user_initiated", "ri_cpu_time_qos_user_interactive",
        "ri_billed_system_time", "ri_serviced_system_time", "ri_logical_writes",
        "ri_lifetime_max_phys_footprint", "ri_instructions", "ri_cycles",
        "ri_billed_energy", "ri_serviced_energy", "ri_interval_max_phys_footprint",
        "ri_runnable_time")]


_RUSAGE_INFO_V4 = 4
_libproc = None
if sys.platform == "darwin":
    try:
        _libproc = ctypes.CDLL(ctypes.util.find_library("proc"), use_errno=True)
        _libproc.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int,
                                             ctypes.POINTER(_RusageInfoV4)]
        _libproc.proc_pid_rusage.restype = ctypes.c_int
    except (OSError, AttributeError):
        _libproc = None

_proc = psutil.Process(os.getpid())


def footprint():
    """(current, lifetime peak) process memory in bytes."""
    if _libproc is not None:
        info = _RusageInfoV4()
        if _libproc.proc_pid_rusage(os.getpid(), _RUSAGE_INFO_V4, ctypes.byref(info)) == 0:
            return info.ri_phys_footprint, info.ri_lifetime_max_phys_footprint
    mem = _proc.memory_info()
    peak = getattr(mem, "peak_wset", None)  # Windows
    if peak is None:
        try:
            import resource
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024  # Linux: KiB
        except ImportError:
            peak = mem.rss
    return mem.rss, max(peak, mem.rss)


def array_bytes(obj, max_depth=4):
    """Bytes of numpy data reachable from *obj*, each buffer counted once.

    Walks attributes (``__dict__`` and ``__slots__``), lists, tuples and dict
    values down to *max_depth*.  Views count as the array they view.  Python
    objects themselves (strings, dicts, lists) are not counted, and neither are
    memory-mapped arrays, which are file-backed.
    """
    seen_obj = set()
    seen_buf = set()
    total = 0

    def _root(a):
        while isinstance(a.base, np.ndarray):
            a = a.base
        return a

    def _is_leaf(v):
        return v is None or isinstance(v, (str, bytes, int, float, bool, np.generic))

    def _walk(o, depth):
        nonlocal total
        if isinstance(o, np.ndarray):
            if isinstance(o, np.memmap) or isinstance(_root(o), np.memmap):
                return
            r = _root(o)
            if id(r) not in seen_buf:
                seen_buf.add(id(r))
                total += r.nbytes
            return
        if depth >= max_depth or id(o) in seen_obj or isinstance(o, (str, bytes, int, float)):
            return
        seen_obj.add(id(o))
        if isinstance(o, (dict, list, tuple)):
            items = list(o.values()) if isinstance(o, dict) else o
            # Skip big containers of scalars/strings (e.g. 10M sequence
            # strings), judged by their first element.
            if items and _is_leaf(items[0]):
                return
        else:
            items = list(getattr(o, "__dict__", {}).values())
            for cls in type(o).__mro__:
                for name in getattr(cls, "__slots__", ()):
                    if hasattr(o, name):
                        items.append(getattr(o, name))
        for v in items:
            _walk(v, depth + 1)

    _walk(obj, 0)
    return total


class _MallocStatistics(ctypes.Structure):
    # malloc_statistics_t, <malloc/malloc.h>
    _fields_ = [("blocks_in_use", ctypes.c_uint), ("size_in_use", ctypes.c_size_t),
                ("max_size_in_use", ctypes.c_size_t), ("size_allocated", ctypes.c_size_t)]


class _Mallinfo2(ctypes.Structure):
    # struct mallinfo2, glibc >= 2.33
    _fields_ = [(n, ctypes.c_size_t) for n in (
        "arena", "ordblks", "smblks", "hblks", "hblkhd", "usmblks", "fsmblks",
        "uordblks", "fordblks", "keepcost")]


_malloc_stats_fn = None
try:
    if sys.platform == "darwin":
        _libc = ctypes.CDLL(None)
        _libc.malloc_zone_statistics.argtypes = [ctypes.c_void_p,
                                                 ctypes.POINTER(_MallocStatistics)]
        _libc.malloc_zone_statistics.restype = None

        def _malloc_stats_fn():
            st = _MallocStatistics()
            _libc.malloc_zone_statistics(None, ctypes.byref(st))
            return st.size_in_use, st.size_allocated
    elif sys.platform.startswith("linux"):
        _libc = ctypes.CDLL(ctypes.util.find_library("c"))
        _libc.mallinfo2.restype = _Mallinfo2

        def _malloc_stats_fn():
            mi = _libc.mallinfo2()
            return mi.uordblks + mi.hblkhd, mi.arena + mi.hblkhd
except (OSError, AttributeError):
    _malloc_stats_fn = None


def malloc_stats():
    """(in use, held) bytes of the C allocator, where numpy, numba and sklearn
    arrays live.  Held minus in use is memory freed but not returned to the
    OS.  (None, None) where unavailable (Windows)."""
    if _malloc_stats_fn is None:
        return None, None
    return _malloc_stats_fn()


def arrow_bytes():
    """Bytes held by pyarrow's memory pool (Parquet writing and reading)."""
    try:
        import pyarrow
        return pyarrow.total_allocated_bytes()
    except Exception:
        return 0


class SearchTracker:
    """Live counters the main search updates: fits running, and results held
    in memory (returned by a fit, not yet written to the batch parquet)."""

    def __init__(self):
        self._lock = threading.Lock()
        self.fits_running = 0
        self.results_held = 0        # rows
        self.results_held_bytes = 0

    def fit_started(self):
        with self._lock:
            self.fits_running += 1

    def fit_finished(self, rows, nbytes):
        with self._lock:
            self.fits_running -= 1
            self.results_held += rows
            self.results_held_bytes += nbytes

    def results_written(self, rows, nbytes):
        with self._lock:
            self.results_held -= rows
            self.results_held_bytes -= nbytes


search_tracker = SearchTracker()


def snapshot():
    """Footprint broken down by where the memory sits, plus search counters."""
    cur, peak = footprint()
    in_use, held = malloc_stats()
    return dict(footprint=cur, peak=peak, malloc_in_use=in_use, malloc_held=held,
                arrow=arrow_bytes(), swap=psutil.swap_memory().used,
                fits_running=search_tracker.fits_running,
                results_held=search_tracker.results_held,
                results_held_bytes=search_tracker.results_held_bytes)


def describe(snap, search=False):
    """One log fragment for a snapshot.

    The footprint splits into C-allocator memory in use (numpy, numba and
    sklearn arrays), pyarrow's pool, and the rest: Polars (its own allocator), Python objects (on 3.14t a
    separate mimalloc heap) plus memory allocators have freed but still hold.
    ``reserved`` is the C allocator's own total, freed blocks included; on
    macOS freed pages stop counting toward the footprint, so it can exceed it.
    """
    text = f"{snap['footprint'] / GB:.1f} GB"
    if snap["malloc_in_use"] is not None:
        rest = snap["footprint"] - snap["malloc_in_use"] - snap["arrow"]
        text += (f" (C arrays {snap['malloc_in_use'] / GB:.1f} GB in use, "
                 f"Arrow {snap['arrow'] / GB:.1f} GB, Polars, Python objects and freed-but-held "
                 f"{rest / GB:.1f} GB; C allocator reserved {snap['malloc_held'] / GB:.1f} GB)")
    text += f"; system swap {snap['swap'] / GB:.1f} GB"
    if search:
        text += (f"; {snap['fits_running']} fits running, {snap['results_held']:,} results "
                 f"held ({snap['results_held_bytes'] / GB:.2f} GB)")
    return text


def log_memory(stage, **objects):
    """Log the footprint breakdown and lifetime peak, and the array bytes of
    each named object."""
    snap = snapshot()
    parts = [f"{name.replace('_', ' ')} {array_bytes(obj) / GB:.1f} GB"
             for name, obj in objects.items() if obj is not None]
    held = f"; arrays held by {', '.join(parts)}" if parts else ""
    logger.info(f"Memory {stage}: {describe(snap)}; lifetime peak "
                f"{snap['peak'] / GB:.1f} GB{held}")


class MemoryWatch:
    """Watch a phase from a background thread.

    Samples the footprint every *interval* seconds and keeps a full snapshot
    from the moment of the phase's peak; logs a timeline line every
    *report_every* seconds while the phase runs, and the peak at the end.
    ``search=True`` adds the main-search counters.
    """

    def __init__(self, stage, interval=0.25, report_every=120, search=False):
        self.stage = stage
        self.interval = interval
        self.report_every = report_every
        self.search = search
        self.peak = 0
        self.at_peak = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _sample(self):
        cur = footprint()[0]
        # A full snapshot only when the peak grows by >1%: malloc statistics
        # take the allocator's locks, so not at every tick.
        if self.at_peak is None or cur > self.peak * 1.01:
            self.at_peak = snapshot()
            self.peak = max(cur, self.at_peak["footprint"])
        elif cur > self.peak:
            self.peak = cur

    def _run(self):
        import time
        t0 = last = time.monotonic()
        while not self._stop.wait(self.interval):
            self._sample()
            now = time.monotonic()
            if now - last >= self.report_every:
                last = now
                logger.info(f"Memory during {self.stage} ({(now - t0) / 60:.0f} min): "
                            f"{describe(snapshot(), self.search)}")

    def __enter__(self):
        self._sample()
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        self._sample()
        logger.info(f"Memory peak during {self.stage}: {describe(self.at_peak, self.search)}")
        return False


def hits_bytes(hits):
    """Approximate bytes one fit's results hold: their arrays, plus the list
    containers (the strings in them are shared with the library)."""
    total = 0
    for v in hits:
        if isinstance(v, np.ndarray):
            total += v.nbytes
        elif isinstance(v, list):
            total += sys.getsizeof(v)
    return total
