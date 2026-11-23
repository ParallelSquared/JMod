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


# --------------------------------------------------------------------------
# IM calibration (calibrate_im) and its application to the library
# (aligned_library_im).  JRIF is replaced by a stand-in whose error is set by
# the test; the real model is exercised in test_rt_predictor.py.
# --------------------------------------------------------------------------

import pandas as pd

import src.rt_predictor as rt_predictor
from src.rt_alignment import (MIN_IM_ANCHORS, IMCalibration, aligned_library_im,
                              calibrate_im)
from src.utils.errors import JModError


class _FakeJRIF:
    """Predicts a peptide's true 1/K0 plus a fixed per-peptide error."""

    def __init__(self, truth, noise, seed=0):
        rng = np.random.default_rng(seed)
        self.table = {k: v + rng.normal(0.0, noise) for k, v in truth.items()}

    def predict(self, mod_seqs, charges):
        return np.array([self.table.get((s, float(z)), np.nan)
                         for s, z in zip(mod_seqs, charges)])


def _anchors(rng, n=3000, obs_noise=0.003):
    """First-search anchors: observed IM, and library IM with a known distortion."""
    seqs = [f"PEPTIDE{i}K" for i in range(n)]
    z = rng.integers(2, 4, size=n).astype(float)
    truth = rng.uniform(0.75, 1.30, size=n)
    obs = truth + rng.normal(0.0, obs_noise, size=n)
    # library IM such that _truth(lib) == truth, approximately (inverse of the
    # mild distortion is close to linear over this range)
    lib = (truth - 0.05) / 0.97
    return seqs, z, truth, obs, lib


def _patch_jrif(monkeypatch, seqs, z, truth, noise):
    fake = _FakeJRIF({(s, float(c)): t for s, c, t in zip(seqs, z, truth)}, noise)
    monkeypatch.setattr(rt_predictor.JmodIMPredictor, "calibrate",
                        classmethod(lambda cls, train, tag=None: fake))
    return fake


def test_jrif_beats_a_noisy_library(monkeypatch):
    rng = np.random.default_rng(10)
    seqs, z, truth, obs, lib = _anchors(rng)
    _patch_jrif(monkeypatch, seqs, z, truth, noise=0.002)
    lib_noisy = lib + rng.normal(0.0, 0.03, size=lib.size)

    cal, holdout, model = calibrate_im(seqs, z, obs, lib_noisy)

    assert cal.winner == "JRIF"
    assert cal.spline is not None
    assert {"JRIF", "library"} <= set(holdout.columns)
    assert model["tolerance"] > 0


def test_a_good_library_beats_a_noisy_jrif(monkeypatch):
    rng = np.random.default_rng(11)
    seqs, z, truth, obs, lib = _anchors(rng)
    _patch_jrif(monkeypatch, seqs, z, truth, noise=0.03)

    cal, _, _ = calibrate_im(seqs, z, obs, lib)

    assert cal.winner == "library"


def test_no_library_im_means_jrif_alone(monkeypatch):
    rng = np.random.default_rng(12)
    seqs, z, truth, obs, _ = _anchors(rng)
    _patch_jrif(monkeypatch, seqs, z, truth, noise=0.03)

    cal, holdout, _ = calibrate_im(seqs, z, obs, np.full(len(seqs), np.nan))

    assert cal.winner == "JRIF"
    assert cal.spline is None
    assert "library" not in holdout.columns


def test_accuracy_comes_from_the_winners_held_out_error(monkeypatch):
    """A worse model must give a wider gate, not the default."""
    rng = np.random.default_rng(13)
    seqs, z, truth, obs, _ = _anchors(rng)
    nan_lib = np.full(len(seqs), np.nan)

    _patch_jrif(monkeypatch, seqs, z, truth, noise=0.003)
    tight = calibrate_im(seqs, z, obs, nan_lib)[2]["tolerance"]
    _patch_jrif(monkeypatch, seqs, z, truth, noise=0.03)
    wide = calibrate_im(seqs, z, obs, nan_lib)[2]["tolerance"]

    assert wide > 3 * tight


def test_too_few_anchors_is_an_error(monkeypatch):
    rng = np.random.default_rng(14)
    seqs, z, truth, obs, lib = _anchors(rng, n=MIN_IM_ANCHORS - 1)
    _patch_jrif(monkeypatch, seqs, z, truth, noise=0.003)

    with pytest.raises(JModError, match="anchors"):
        calibrate_im(seqs, z, obs, lib)


def test_anchors_without_observed_im_do_not_count(monkeypatch):
    rng = np.random.default_rng(15)
    seqs, z, truth, obs, lib = _anchors(rng, n=2 * MIN_IM_ANCHORS)
    obs[: MIN_IM_ANCHORS + 1] = np.nan
    _patch_jrif(monkeypatch, seqs, z, truth, noise=0.003)

    with pytest.raises(JModError, match="anchors"):
        calibrate_im(seqs, z, obs, lib)


def test_library_im_that_cannot_be_aligned_is_an_error(monkeypatch):
    rng = np.random.default_rng(16)
    seqs, z, truth, obs, lib = _anchors(rng)
    _patch_jrif(monkeypatch, seqs, z, truth, noise=0.003)
    monkeypatch.setattr("src.rt_alignment.fit_im_alignment", lambda *a: None)

    with pytest.raises(JModError, match="could not be aligned"):
        calibrate_im(seqs, z, obs, lib)


class _Library:
    """The four library-store fields aligned_library_im reads."""

    def __init__(self, mod_seq, prec_z, ion_mob, parent_idx):
        self.mod_seq = list(mod_seq)
        self.prec_z = np.asarray(prec_z, dtype=np.uint8)
        self.ion_mob = np.asarray(ion_mob, dtype=np.float32)
        self.parent_idx = np.asarray(parent_idx, dtype=np.int64)

    def __len__(self):
        return len(self.mod_seq)


@pytest.fixture
def library():
    # Three targets, then decoys of targets 0 and 2, then one parentless decoy.
    return _Library(
        mod_seq=["AAAK", "CCCK", "DDDK", "KAAA", "KDDD", "EEEK"],
        prec_z=[2, 2, 3, 2, 3, 2],
        ion_mob=[0.80, np.nan, 1.10, 0.80, 1.10, 0.95],
        parent_idx=[-1, -1, -1, 0, 2, -1],
    )


class _Lookup:
    def __init__(self, table):
        self.table = table
        self.calls = []

    def predict(self, mod_seqs, charges):
        self.calls.append(list(mod_seqs))
        return np.array([self.table.get(s, np.nan) for s in mod_seqs])


def test_no_calibration_means_no_im(library):
    assert np.isnan(aligned_library_im(library, None)).all()


def test_jrif_winner_predicts_and_decoys_copy_their_parent(library):
    jrif = _Lookup({"AAAK": 0.9, "CCCK": 1.0, "DDDK": 1.2, "EEEK": 0.7})
    out = aligned_library_im(library, IMCalibration("JRIF", jrif, spline=None))

    np.testing.assert_allclose(out, [0.9, 1.0, 1.2, 0.9, 1.2, 0.7])
    # decoys with a parent are never predicted from their own sequence
    assert "KAAA" not in jrif.calls[0] and "KDDD" not in jrif.calls[0]


def test_library_winner_is_filled_from_jrif_where_the_library_has_no_im(library):
    jrif = _Lookup({"CCCK": 1.0})
    out = aligned_library_im(
        library, IMCalibration("library", jrif, spline=lambda x: x + 0.01))

    np.testing.assert_allclose(out, [0.81, 1.0, 1.11, 0.81, 1.11, 0.96], rtol=1e-6)
    assert jrif.calls == [["CCCK"]]   # JRIF only for the gap


def test_jrif_winner_is_filled_from_the_library_where_jrif_cannot_predict(library):
    jrif = _Lookup({"AAAK": 0.9, "CCCK": 1.0, "EEEK": 0.7})   # DDDK unrepresentable
    out = aligned_library_im(
        library, IMCalibration("JRIF", jrif, spline=lambda x: x + 0.01))

    np.testing.assert_allclose(out, [0.9, 1.0, 1.11, 0.9, 1.11, 0.7], rtol=1e-6)


def test_library_cannot_win_without_an_alignment():
    with pytest.raises(ValueError):
        IMCalibration("library", object(), spline=None)
