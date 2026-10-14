"""
Tests for the --ms1_fwhm Gaussian MS1 peak-width columns.
"""

import numpy as np
import pandas as pd
import pytest

import src.ms1_fwhm as mf


CYCLE = 0.02                                   # minutes per MS1 scan
T = np.arange(0, 2.0, CYCLE)


def _peak(fwhm, mu=1.0, a=1e5):
    return mf._gauss(T, a, mu, fwhm / mf.SIGMA_TO_FWHM)


def _start(mu=1.0):
    return int(np.argmin(np.abs(T - mu)))


def _width(y, start):
    return mf._gaussian_fwhm(T, y, start)[0]


def _no_fit(y, start):
    return all(np.isnan(v) for v in mf._gaussian_fwhm(T, y, start))


class TestGaussianFwhm:
    @pytest.mark.parametrize("fwhm", [0.06, 0.1, 0.3])
    def test_recovers_width(self, fwhm):
        y = _peak(fwhm)
        y[y < 0.01 * y.max()] = 0.0
        assert _width(y, _start()) == pytest.approx(fwhm, rel=0.05)

    def test_start_off_apex_walks_uphill(self):
        y = _peak(0.1)
        assert _width(y, _start() - 2) == pytest.approx(0.1, rel=0.05)

    def test_neighbouring_peak_excluded(self):
        y = _peak(0.1, mu=1.0) + _peak(0.1, mu=1.25, a=2e5)
        assert _width(y, _start(1.0)) == pytest.approx(0.1, rel=0.15)

    def test_lone_dropout_ignored(self):
        y = _peak(0.12)
        y[_start() + 1] = 0.0
        assert _width(y, _start()) == pytest.approx(0.12, rel=0.1)

    def test_flat_background_is_nan(self):
        y = np.full(len(T), 5e4) + np.random.default_rng(0).normal(0, 5e3, len(T))
        assert _no_fit(y, _start())

    def test_no_signal_is_nan(self):
        assert _no_fit(np.zeros(len(T)), _start())

    def test_too_few_points_is_nan(self):
        y = np.zeros(len(T))
        y[_start()] = 1e5
        y[_start() + 1] = 5e4
        assert _no_fit(y, _start())


class TestR2:
    def test_clean_peak_r2_near_one(self):
        fwhm, r2 = mf._gaussian_fwhm(T, _peak(0.1), _start())
        assert fwhm == pytest.approx(0.1, rel=0.05)
        assert r2 == pytest.approx(1.0, abs=1e-3)

    def test_poor_fit_still_gets_width(self):
        # jagged top: resolved, returns to baseline, but far from Gaussian
        vals = [1.0, 0.55, 0.95, 0.6, 1.0, 0.55, 0.9]
        y = np.zeros(len(T))
        lo = _start() - len(vals) // 2
        y[lo:lo + len(vals)] = vals
        fwhm, r2 = mf._gaussian_fwhm(T, y, _start())
        assert r2 < 0.5                # used to be dropped as NaN
        assert np.isfinite(fwhm) and fwhm > 0


class TestAddColumn:
    def test_only_adds_two_columns_and_keeps_rows(self, monkeypatch):
        dat = pd.DataFrame({
            "untag_prec": ["A_2", "A_2", "B_2", "C_2"],
            "channel": [0, 1, 0, 0],
            "silac_channel": [0, 0, 0, 0],
            "is_decoy": [False, False, False, True],
            "BestChannel_Qvalue": [0.001, 0.001, 0.5, 0.001],
            "x": [1.0, 2.0, 3.0, 4.0],
        })
        before = dat.copy()
        monkeypatch.setattr(mf, "compute_ms1_fwhm",
                            lambda *a, **k: {("A_2", 0, 0): (0.08, 0.3),
                                             ("A_2", 1, 0): (np.nan, np.nan),
                                             ("B_2", 0, 0): (0.5, 0.9),
                                             ("C_2", 0, 0): (0.7, 0.9)})
        out = mf.add_ms1_fwhm(dat, spectra=None, mz_ppm=1e-5)
        assert list(out.columns) == list(before.columns) + [mf.FWHM_COL, mf.R2_COL]
        pd.testing.assert_frame_equal(out[before.columns], before)
        f, r = out[mf.FWHM_COL].tolist(), out[mf.R2_COL].tolist()
        assert f[0] == 0.08 and r[0] == 0.3                  # low R² kept, not NaN'd
        assert np.isnan(f[1]) and np.isnan(r[1])
        assert np.isnan(f[2]) and np.isnan(r[2])             # BestChannel_Qvalue >= 0.01
        assert np.isnan(f[3]) and np.isnan(r[3])             # decoy
