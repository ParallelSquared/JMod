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


def test_primed_fingerprints_cover_the_arrays_jmod_passes(tmp_path):
    """Every read-only, unaligned or 0-d array -- the arguments numba types
    through its unlocked fingerprint table -- has a fingerprint that
    prime_type_cache inserts, so worker threads never insert one."""
    import numpy as np
    from numba._dispatcher import compute_fingerprint
    from src.utils.numba_threads import _prime_args

    primed = {compute_fingerprint(a) for args in _prime_args() for a in args}

    def frozen(a):
        a = a.view()
        a.flags.writeable = False
        return a

    np.save(tmp_path / "x.npy", np.arange(10, dtype=np.float32))
    mapped = np.load(tmp_path / "x.npy", mmap_mode="r")        # library_mmap
    table = frozen(np.zeros((5, 4)))                           # frozen library
    cases = [mapped, mapped[2:7], mapped[::3], table, table[:, 1], table[1], table.T,
             frozen(np.zeros(0, dtype=np.int64)), frozen(np.zeros((), dtype=np.uint32)),
             np.frombuffer(b"\0" * 17, dtype=np.int64, count=2, offset=1),   # read-only, unaligned
             np.frombuffer(bytearray(17), dtype=np.float64, count=2, offset=1),
             np.zeros((), dtype=np.int32), frozen(np.zeros((2, 3, 4), dtype=np.bool_)),
             None, True, False]
    for a in cases:
        assert compute_fingerprint(a) in primed, (getattr(a, "dtype", a), getattr(a, "shape", None))


def test_prime_type_cache_runs_and_is_idempotent():
    from src.utils import numba_threads
    numba_threads.prime_type_cache()
    numba_threads.prime_type_cache()
    assert numba_threads._primed


def test_prime_type_cache_fills_the_table_so_first_calls_do_not_insert():
    """After priming, a new jitted function's first call with a read-only array,
    an unaligned array, None or a numpy bool finds its typecode in the table:
    numba's Python fallback (typeof_pyval) is not called from the C dispatcher.
    Run in a fresh process: the table is process-wide."""
    import subprocess, sys, textwrap
    code = textwrap.dedent("""
        import numpy as np, traceback
        from numba import njit
        from numba.core.dispatcher import _DispatcherBase
        from src.utils.numba_threads import prime_type_cache
        prime_type_cache()
        misses = []
        orig = _DispatcherBase.typeof_pyval
        def hook(self, v):
            if not any("_compile_for_args" in f.name for f in traceback.extract_stack()):
                misses.append(repr(type(v)))
            return orig(self, v)
        _DispatcherBase.typeof_pyval = hook
        @njit
        def f(a):
            return 0
        ro = np.arange(6, dtype=np.int64); ro.setflags(write=False)
        rof = np.zeros((3, 3), np.float32); rof.setflags(write=False)
        buf = bytearray(8 * 4 + 1); un = np.frombuffer(buf, np.float64, count=4, offset=1)
        for v in (ro, rof.T, un, None, np.bool_(True)):
            f(v)
        print("MISSES", misses)
    """)
    import os
    env = {k: v for k, v in os.environ.items() if k != "NUMBA_DISABLE_JIT"}   # conftest disables the JIT
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, env=env).stdout
    assert "MISSES []" in out, out
