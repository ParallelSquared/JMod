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
