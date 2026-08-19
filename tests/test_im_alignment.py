import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.rt_alignment import fit_im_alignment


def _synthetic(rng, n=5000, noise=0.004):
    """Library IM plus a known monotone distortion -> 'observed' IM.

    The distortion is mild and monotone, like a real IM miscalibration: a slight
    gain/offset with a little curvature over the 1/K0 range timsTOF covers.
    """
    lib = rng.uniform(0.75, 1.30, size=n)
    obs = 0.97 * lib + 0.05 + 0.06 * (lib - 1.0) ** 2
    obs = obs + rng.normal(0.0, noise, size=n)
    return lib, obs


def _truth(lib):
    return 0.97 * lib + 0.05 + 0.06 * (lib - 1.0) ** 2


def test_recovers_known_distortion():
    rng = np.random.default_rng(0)
    lib, obs = _synthetic(rng)

    f = fit_im_alignment(lib, obs)
    assert f is not None

    # Evaluate away from the edges, where LOWESS anchors are sparse.
    xs = np.linspace(0.80, 1.25, 60)
    err = np.abs(f(xs) - _truth(xs))
    assert np.median(err) < 0.004
    assert np.max(err) < 0.012


def test_alignment_reaches_the_noise_floor():
    """Aligning should remove the systematic distortion, leaving only the noise.

    Not expressed as a ratio against the unaligned residual: the injected
    distortion here is comparable to the noise, so the achievable improvement is
    bounded by the noise floor rather than by any fixed factor.
    """
    rng = np.random.default_rng(1)
    noise = 0.004
    lib, obs = _synthetic(rng, noise=noise)

    f = fit_im_alignment(lib, obs)
    assert f is not None

    before = np.std(obs - lib)
    after = np.std(obs - f(lib))

    assert after < before                 # alignment helps
    assert after < 1.15 * noise           # and lands at the noise floor


def test_all_nan_library_im_returns_none():
    rng = np.random.default_rng(2)
    _, obs = _synthetic(rng)
    lib = np.full(obs.shape, np.nan)

    assert fit_im_alignment(lib, obs) is None


def test_all_nan_observed_im_returns_none():
    """mzML / non-IM data: observed mobility is absent, so no fit is possible."""
    rng = np.random.default_rng(3)
    lib, _ = _synthetic(rng)
    obs = np.full(lib.shape, np.nan)

    assert fit_im_alignment(lib, obs) is None


def test_too_few_anchors_returns_none():
    rng = np.random.default_rng(4)
    lib, obs = _synthetic(rng, n=6)

    assert fit_im_alignment(lib, obs) is None


def test_small_but_usable_anchor_count_still_fits():
    """Hundreds of anchors is the normal IM case and must not be rejected."""
    rng = np.random.default_rng(8)
    lib, obs = _synthetic(rng, n=400)

    assert fit_im_alignment(lib, obs) is not None


def test_partial_nans_are_dropped_not_fatal():
    """Per-row NaNs are normal (unmatched PSMs) and must not sink the fit."""
    rng = np.random.default_rng(5)
    lib, obs = _synthetic(rng, n=6000)
    lib = lib.copy()
    obs = obs.copy()
    lib[::7] = np.nan
    obs[::11] = np.nan

    f = fit_im_alignment(lib, obs)
    assert f is not None
    xs = np.linspace(0.80, 1.25, 40)
    assert np.median(np.abs(f(xs) - _truth(xs))) < 0.005


def test_length_mismatch_raises():
    rng = np.random.default_rng(7)
    lib, obs = _synthetic(rng, n=1000)

    with pytest.raises(ValueError):
        fit_im_alignment(lib[:-1], obs)
