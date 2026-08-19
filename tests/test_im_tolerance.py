import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.rt_alignment import (
    fit_zero_mean_laplace_uniform_1d,
    fit_im_tolerance,
)


def _synthetic_im_errors(rng, b=0.0026, n_core=40000, n_bg=8000,
                         n_outlier=6000, outlier_center=-0.23):
    """Sharp zero-centered Laplace core + broad uniform background + outlier bump."""
    core = rng.laplace(loc=0.0, scale=b, size=n_core)
    background = rng.uniform(-0.05, 0.05, size=n_bg)
    outliers = rng.normal(outlier_center, 0.01, size=n_outlier)
    return np.concatenate([core, background, outliers])


def test_laplace_uniform_recovers_scale():
    rng = np.random.default_rng(0)
    b_true = 0.0026
    core = rng.laplace(loc=0.0, scale=b_true, size=60000)
    background = rng.uniform(-0.05, 0.05, size=12000)
    x = np.concatenate([core, background])

    weight, b = fit_zero_mean_laplace_uniform_1d(x)

    # scale recovered close to truth, and a substantial Laplace weight
    assert abs(b - b_true) < 0.0006
    assert 0.5 < weight < 1.0


def test_tolerance_matches_laplace_4sd():
    rng = np.random.default_rng(1)
    b_true = 0.0026
    errors = _synthetic_im_errors(rng, b=b_true)

    tol = fit_im_tolerance(errors)
    expected = 4.0 * b_true * np.sqrt(2.0)  # 4*SD of the Laplace core

    assert tol is not None
    # within ~20% of the injected core's 4*SD
    assert abs(tol - expected) / expected < 0.2


def test_outliers_do_not_inflate_tolerance():
    rng = np.random.default_rng(2)
    b_true = 0.0026
    clean = _synthetic_im_errors(rng, b=b_true, n_outlier=0)
    dirty = _synthetic_im_errors(rng, b=b_true, n_outlier=12000)

    tol_clean = fit_im_tolerance(clean)
    tol_dirty = fit_im_tolerance(dirty)

    assert tol_clean is not None and tol_dirty is not None
    # the -0.23 outlier subpopulation is stripped before the fit
    assert abs(tol_dirty - tol_clean) / tol_clean < 0.2


def test_insufficient_data_returns_none():
    assert fit_im_tolerance(np.array([])) is None
    assert fit_im_tolerance(np.array([np.nan, np.nan])) is None
