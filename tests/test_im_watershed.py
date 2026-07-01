import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.io.im_watershed import segment_window


def _cluster(center, n, lo, hi, rng, spread=0.0015):
    """n peaks near `center` with intensities spread across [lo, hi]."""
    mob = rng.normal(center, spread, n)
    intens = rng.uniform(lo, hi, n)
    return mob, intens


def _background(n, rng, level=200.0, jitter=20.0):
    """Broad, near-uniform low-intensity noise floor across the IM range."""
    mob = rng.uniform(0.80, 1.30, n)
    intens = rng.uniform(level - jitter, level + jitter, n)
    return mob, intens


def _bands_sorted(bands):
    return sorted(bands, key=lambda b: b[0])


def test_two_separated_bands():
    rng = np.random.default_rng(0)
    bg = _background(400, rng)
    a = _cluster(0.95, 40, 2000, 60000, rng)
    b = _cluster(1.10, 40, 2000, 60000, rng)
    mob = np.concatenate([bg[0], a[0], b[0]])
    intens = np.concatenate([bg[1], a[1], b[1]])

    bands = _bands_sorted(segment_window(mob, intens))
    assert len(bands) == 2
    # bands are disjoint and ordered
    (lo0, hi0, _), (lo1, hi1, _) = bands
    assert lo0 < hi0 <= lo1 < hi1
    # apexes land near the two clusters
    assert lo0 <= 0.95 <= hi0
    assert lo1 <= 1.10 <= hi1


def test_spike_plus_small_neighbor_both_kept():
    # A dominant spike next to a much smaller real cluster must not swallow it
    # (the relative-merge case, cf. frame 11043).
    rng = np.random.default_rng(1)
    bg = _background(400, rng)
    big = _cluster(0.95, 40, 50000, 1_000_000, rng)
    small = _cluster(0.985, 40, 2000, 80000, rng)
    mob = np.concatenate([bg[0], big[0], small[0]])
    intens = np.concatenate([bg[1], big[1], small[1]])

    bands = _bands_sorted(segment_window(mob, intens))
    assert len(bands) == 2


def test_uniform_noise_rejected():
    # Broad, uniform-intensity peaks -> no band survives the CV filter.
    rng = np.random.default_rng(2)
    mob, intens = _background(600, rng, level=200.0, jitter=10.0)
    bands = segment_window(mob, intens)
    assert bands == []


def test_below_min_peaks_returns_empty():
    rng = np.random.default_rng(3)
    mob, intens = _cluster(0.95, 10, 2000, 60000, rng)
    assert segment_window(mob, intens) == []


def test_band_peak_indices_are_valid():
    rng = np.random.default_rng(4)
    bg = _background(400, rng)
    a = _cluster(0.95, 40, 2000, 60000, rng)
    mob = np.concatenate([bg[0], a[0]])
    intens = np.concatenate([bg[1], a[1]])

    bands = segment_window(mob, intens)
    assert len(bands) >= 1
    for im_lo, im_hi, peak_idx in bands:
        assert peak_idx.dtype.kind in "iu"
        # every returned peak truly falls inside the reported mobility band
        assert np.all(mob[peak_idx] >= im_lo)
        assert np.all(mob[peak_idx] <= im_hi)
        assert len(peak_idx) >= 20
