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

"""Make numba's compile timers safe when several threads compile at once.

numba's ``Dispatcher.compile`` installs two ``TimingListener``s
("numba:compiler_lock", "numba:llvm_lock") in a process-global listener list
*before* it takes the compiler lock, and every lock acquire/release broadcasts
its start/end event to every listener in that list, whichever thread installed
it.  If thread B installs its timer and thread A's *end* event arrives before
B's own *start*, B's timer gets an end without a start and raises
``AttributeError: 'TimingListener' object has no attribute '_ts'`` -- out of
thread A's compile, so A's fit fails and takes the run with it.

That needs two first-time compiles at once, which the main search's worker
threads do whenever a jitted function meets a new type signature mid-search.
Free-threaded Python makes the interleaving far more likely than under the
GIL; it killed two runs on 2026-11-03/04.

The fix: a compile timer only hears events from the thread that installed it,
which is what it measures anyway (that thread's wait for and hold of the
lock).  Listeners other than ``TimingListener`` are left global, and the
registry is locked and iterated over a snapshot, since free-threaded list
mutation during iteration can skip entries.  Only numba's ``event`` module
globals are replaced; numba calls ``register``/``unregister``/``broadcast``
through them, never by imported name.
"""

import threading

import numpy as np

_installed = False
_lock = threading.Lock()


def make_compile_timers_thread_safe():
    """Install the per-thread filtering for numba's compile timers.  Idempotent."""
    global _installed
    with _lock:
        if _installed:
            return
        from numba.core import event as ev

        registered = ev._registered
        registry_lock = threading.Lock()

        def register(kind, listener):
            assert isinstance(listener, ev.Listener)
            kind = ev._guard_kind(kind)
            if isinstance(listener, ev.TimingListener):
                listener._jmod_thread = threading.get_ident()
            with registry_lock:
                registered[kind].append(listener)

        def unregister(kind, listener):
            assert isinstance(listener, ev.Listener)
            kind = ev._guard_kind(kind)
            with registry_lock:
                registered[kind].remove(listener)

        def broadcast(event):
            me = threading.get_ident()
            with registry_lock:
                listeners = list(registered[event.kind])
            for listener in listeners:
                owner = getattr(listener, "_jmod_thread", None)
                if owner is not None and owner != me:
                    continue
                listener.notify(event)

        ev.register = register
        ev.unregister = unregister
        ev.broadcast = broadcast
        _installed = True


# --- numba's argument-type cache ---------------------------------------------
#
# Every call into a jitted function types its arguments in C.  Plain numbers
# and plain arrays (aligned, writable, a basic dtype) use fixed tables; every
# other argument -- a read-only array (the frozen spectral library, read-only
# memory maps), an unaligned array, a 0-d array, None, a bool -- goes through
# one process-wide hash table, keyed by a fingerprint (for an array: ndim,
# layout, writable, dtype).  The first argument with a new fingerprint inserts
# it.  The table has no lock, which under the GIL was fine; on free-threaded
# Python an insert that resizes it while another thread reads or inserts
# corrupts the heap (numba 0.68 _typeof.cpp; a 2ng 5plex run crashed in
# _Numba_hashtable_set on 2026-11-07, as the main search's workers started).
#
# The fix: insert every fingerprint the run can produce before any worker
# thread exists.  Workers then only read the table, and concurrent reads are
# safe.  For arrays that set is small and closed: each basic dtype, ndim 0-3,
# C/F/A layout, read-only or writable.  Tuples and lists are fingerprinted by
# their contents and are not covered; no jitted function is called with one.

_primed = False

_PRIME_DTYPES = ("?", "i1", "i2", "i4", "i8", "u1", "u2", "u4", "u8",
                 "f4", "f8", "c8", "c16")


def _layouts(make):
    """0-d, then C, F and non-contiguous arrays of ndim 1-3, from *make(shape)*."""
    out = [make(())]
    for ndim in (1, 2, 3):
        c = make((3,) * ndim)
        out.append(c)
        if ndim > 1:
            out.append(c.T)                                    # F
        wide = make((3,) * (ndim - 1) + (6,))
        out.append(wide[(slice(None),) * (ndim - 1) + (slice(None, None, 2),)])  # A
    return out


def _unaligned(dtype):
    """make(shape): writable arrays of *dtype* whose data starts one byte off
    alignment, so numba types them through the fingerprint table."""
    def make(shape):
        n = int(np.prod(shape)) if shape else 1
        buf = bytearray(n * dtype.itemsize + 1)
        return np.frombuffer(buf, dtype=dtype, count=n, offset=1).reshape(shape)
    return make


def _read_only(arrays):
    out = []
    for a in arrays:
        a = a.view()
        a.flags.writeable = False
        out.append(a)
    return out


def _prime_args():
    """The probe calls' arguments, one tuple per call: every fingerprint the
    type cache can be asked for with arrays of a basic dtype, plus None and
    bool."""
    calls = [(None, True)]
    for code in _PRIME_DTYPES:
        dtype = np.dtype(code)
        arrays = _layouts(lambda shape: np.zeros(shape, dtype))
        args = _read_only(arrays)
        if dtype.itemsize > 1:
            # writable arrays are typed without the table unless unaligned
            # (or 0-d); itemsize-1 dtypes cannot be unaligned
            unaligned = _layouts(_unaligned(dtype))
            args += unaligned + _read_only(unaligned)
        else:
            args.append(arrays[0])                             # writable 0-d
        calls.append(tuple(args))
    return calls


def prime_type_cache():
    """Insert every argument fingerprint into numba's type cache, on this
    thread, before worker threads call jitted functions.  Idempotent; call it
    with no other thread running numba.  Takes about a second."""
    global _primed
    with _lock:
        if _primed:
            return
        from numba import njit

        @njit(cache=False)
        def _probe(*args):
            return 0

        for args in _prime_args():
            _probe(*args)
        _primed = True
