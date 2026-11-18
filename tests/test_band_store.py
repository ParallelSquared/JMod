import gc
import weakref

import numpy as np

from src.utils.io.band_store import BandSpectrum


def _band():
    # three runs, each sorted by m/z: [1, 3], [2, 4], [0.5, 5]
    mz = np.array([1.0, 3.0, 2.0, 4.0, 0.5, 5.0], dtype=np.float32)
    it = np.arange(6, dtype=np.float32)
    mob = np.linspace(0.8, 0.9, 6).astype(np.float32)
    return BandSpectrum((mz, it, mob), (0, 2, 4, 6))


def test_band_reads_merged_runs():
    s = _band()
    np.testing.assert_array_equal(s.mz, [0.5, 1.0, 2.0, 3.0, 4.0, 5.0])
    np.testing.assert_array_equal(s.intens, [4, 0, 2, 1, 3, 5])


def test_partial_assignment_keeps_band_for_the_rest():
    s = _band()
    s.mz = np.array([9.0], dtype=np.float32)
    np.testing.assert_array_equal(s.mz, [9.0])
    np.testing.assert_array_equal(s.intens, [4, 0, 2, 1, 3, 5])


def test_full_assignment_releases_group_buffer():
    # use_centroided_ms2 assigns all three arrays: the raw group buffer must not
    # stay alive behind the centroids.
    buf = np.zeros(1000, dtype=np.float32)
    ref = weakref.ref(buf)
    s = BandSpectrum((buf, buf, buf), (0, 0, 0, 1000))
    del buf
    s.mz = np.array([1.0], dtype=np.float32)
    s.intens = np.array([2.0], dtype=np.float32)
    s.mobility = np.array([0.85], dtype=np.float32)
    gc.collect()
    assert ref() is None
    np.testing.assert_array_equal(s.mz, [1.0])
    np.testing.assert_array_equal(s.intens, [2.0])
    np.testing.assert_array_equal(s.mobility, np.array([0.85], dtype=np.float32))


def _random_group(rng):
    """A group buffer of three (m/z, mob, int)-sorted runs, with exact m/z ties
    across runs (raw TOF bins repeat m/z values)."""
    runs = []
    for n in rng.integers(0, 40, size=3):
        mz = rng.choice(np.arange(100, 140, dtype=np.float32), size=n)
        mob = rng.random(n).astype(np.float32)
        it = rng.random(n).astype(np.float32)
        o = np.lexsort((it, mob, mz))
        runs.append((mz[o], it[o], mob[o]))
    cuts = np.cumsum([0] + [len(r[0]) for r in runs])
    group = tuple(np.concatenate([r[k] for r in runs]) for k in range(3))
    return group, tuple(int(c) for c in cuts)


def test_gather_range_and_nearest_peak_match_merged_band():
    from src.utils.io.band_store import gather_range, merge_runs, nearest_peak
    rng = np.random.default_rng(0)
    for _ in range(300):
        (mz, it, mob), runs = _random_group(rng)
        bmz, bit, bmob = merge_runs(mz, it, mob, *runs)
        o = np.lexsort((bit, bmob, bmz))
        assert np.array_equal(o, np.arange(len(bmz)))          # merged in (mz, mob, int) order
        for lo, hi in [(105.0, 112.0), (99.0, 150.0), (120.0, 120.0), (200.0, 300.0)]:
            gmz, git, gmob = gather_range(mz, it, mob, *runs, lo, hi)
            a, b = np.searchsorted(bmz, lo), np.searchsorted(bmz, hi, side="right")
            np.testing.assert_array_equal(gmz, bmz[a:b])
            np.testing.assert_array_equal(git, bit[a:b])
            np.testing.assert_array_equal(gmob, bmob[a:b])
        for q in [99.5, 110.0, 110.5, 139.0, 141.0]:
            pred, succ = nearest_peak(mz, it, mob, *runs, q)
            pos = np.searchsorted(bmz, q)
            exp_pred = (bmz[pos - 1], bit[pos - 1]) if pos > 0 else None
            exp_succ = (bmz[pos], bit[pos]) if pos < len(bmz) else None
            assert (None if pred < 0 else (mz[pred], it[pred])) == exp_pred
            assert (None if succ < 0 else (mz[succ], it[succ])) == exp_succ
