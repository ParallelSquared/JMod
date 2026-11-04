"""numba compile timers must not crash when threads compile at the same time."""
import subprocess
import sys
import textwrap

# The precondition of the crash, forced deterministically: this thread installs
# a compile timer (as Dispatcher.compile does before taking the lock), then
# another thread's lock release broadcasts its END event to it.  Unpatched, the
# timer's depth goes to -1; two STARTs racing on that depth (a non-atomic += on
# free-threaded Python) can then leave depth 1 with no _ts, and the next END
# raises "TimingListener object has no attribute '_ts'".  The race on += itself
# cannot be forced from Python, so the test checks the depth.
_SEQUENCE = textwrap.dedent("""
    import threading
    {setup}
    from numba.core import event as ev
    errors = []
    def other_thread_releases_lock():
        try:
            ev.end_event("numba:compiler_lock")
        except AttributeError as e:
            errors.append(repr(e))
    durations = []
    with ev.install_timer("numba:compiler_lock", durations.append) as timer:
        t = threading.Thread(target=other_thread_releases_lock)
        t.start(); t.join()
        depth_after_foreign_end = timer._depth
        ev.start_event("numba:compiler_lock")   # this thread's own lock use
        ev.end_event("numba:compiler_lock")
    print("ERRORS", len(errors), "TIMED", len(durations), "DEPTH", depth_after_foreign_end)
""")


def _run(setup):
    out = subprocess.run([sys.executable, "-c", _SEQUENCE.format(setup=setup)],
                         capture_output=True, text=True, cwd=".")
    assert out.returncode == 0, out.stderr
    line = [l for l in out.stdout.splitlines() if l.startswith("ERRORS")][-1]
    _, errors, _, timed, _, depth = line.split()
    return int(errors), int(timed), int(depth)


def test_unpatched_numba_lets_other_threads_corrupt_a_timer():
    _, _, depth = _run("")
    if depth == 0:
        import pytest
        pytest.skip("numba no longer delivers other threads' events to compile "
                    "timers; src/utils/numba_threads.py may be removable")
    assert depth == -1


def test_patched_timers_ignore_other_threads():
    errors, timed, depth = _run("from src.utils.numba_threads import make_compile_timers_thread_safe\n"
                                "make_compile_timers_thread_safe()")
    assert depth == 0          # the other thread's END never reached this timer
    assert errors == 0
    assert timed == 1          # the installing thread's own start/end is still timed


def test_concurrent_first_compiles():
    import numpy as np
    import threading
    from numba import njit
    import src  # noqa: F401  (installs the patch)

    errors = []
    for r in range(3):
        fns = []
        for i in range(12):
            ns = {}
            exec(f"def f(x):\n    return x.sum() * {r * 100 + i}.0", ns)
            fns.append(njit(nogil=True)(ns["f"]))
        barrier = threading.Barrier(len(fns))

        def work(f):
            barrier.wait()
            try:
                f(np.ones(3))
            except Exception as e:  # noqa: BLE001
                errors.append(e)
        threads = [threading.Thread(target=work, args=(f,)) for f in fns]
        [t.start() for t in threads]
        [t.join() for t in threads]
    assert errors == []
