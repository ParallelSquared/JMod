"""
Tests for functions in fragment_correlation.py.
"""

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

from __future__ import annotations

import threading
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.fragment_correlation import (
    FEATURE_COLUMNS,
    _N_KERNEL_FEATURES,
    _build_window_csr,
    _covering_scans_numba,
    _build_peak_lists,
    _scan_metadata,
    _match_and_fill_numba,
    _match_and_fill_im_numba,
    _pairwise_corr_and_features_numba,
    compute_fragment_correlations,
)
from src.utils.frag_encoding import encode_frag_names
from src.utils.io.load_files import SpectrumFile


# ---------------------------------------------------------------------------
# Small stub helpers shared by multiple tests
# ---------------------------------------------------------------------------


def _make_scan(mz, intens, rt, win_target, win_lower, win_upper):
    return SimpleNamespace(
        mz=np.asarray(mz, dtype=np.float64),
        intens=np.asarray(intens, dtype=np.float64),
        RT=float(rt),
        isolation_window={
            "isolation window target m/z": float(win_target),
            "isolation window lower offset": float(win_lower),
            "isolation window upper offset": float(win_upper),
        },
    )


def _make_im_scan(mz, intens, mob, rt, im_lo, im_hi,
                  win_target=500, win_lower=5, win_upper=5):
    """Like _make_scan but with per-peak mobility and IM-bin bounds."""
    return SimpleNamespace(
        mz=np.asarray(mz, dtype=np.float64),
        intens=np.asarray(intens, dtype=np.float64),
        mobility=np.asarray(mob, dtype=np.float64),
        RT=float(rt),
        im_lo=float(im_lo),
        im_hi=float(im_hi),
        isolation_window={
            "isolation window target m/z": float(win_target),
            "isolation window lower offset": float(win_lower),
            "isolation window upper offset": float(win_upper),
        },
    )


def _make_spectra(scans):
    """Wrap a list of scan stubs in a SpectrumFile-like holder."""
    return SimpleNamespace(ms2scans=list(scans))


def _make_library(entries):
    """Build a minimal SpectrumLibraryStore-like stub.

    ``entries`` is an iterable of dicts with ``seq``, ``z``, ``prec_mz``,
    ``frag_mz``, ``frag_names``.
    """
    key_to_idx = {}
    mzs, names_list, offsets, lengths, prec_mzs = [], [], [], [], []
    cursor = 0
    for i, e in enumerate(entries):
        key_to_idx[(e["seq"], int(e["z"]))] = i
        fmz = np.asarray(e["frag_mz"], dtype=np.float64)
        # sort library fragment m/z (not required, but mimics real data)
        order = np.argsort(fmz)
        fmz = fmz[order]
        fnames = [e["frag_names"][j] for j in order]
        mzs.append(fmz)
        names_list.append(encode_frag_names(fnames))
        offsets.append(cursor)
        lengths.append(len(fmz))
        prec_mzs.append(float(e["prec_mz"]))
        cursor += len(fmz)
    spectrum_mz = np.concatenate(mzs) if mzs else np.empty(0, dtype=np.float64)
    frag_names_data = (
        np.concatenate(names_list) if names_list else np.empty(0, dtype=np.int32)
    )
    return SimpleNamespace(
        key_to_idx=key_to_idx,
        spectrum_mz=spectrum_mz,
        spectrum_offsets=np.asarray(offsets, dtype=np.int64),
        spectrum_lengths=np.asarray(lengths, dtype=np.int32),
        frag_names_data=frag_names_data,
        prec_mz=np.asarray(prec_mzs, dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# _match_and_fill_numba
# ---------------------------------------------------------------------------


class TestMatchAndFill:
    def _call(self, scans, scan_idx, frag_mz, ppm_tol):
        pmz, pin, _mob = _build_peak_lists(list(scans), False)
        si = np.asarray(scan_idx, dtype=np.int64)
        fm = np.asarray(frag_mz, dtype=np.float64)
        out = np.empty((len(si), len(fm)), dtype=np.float64)
        _match_and_fill_numba(pmz, pin, si, len(si), fm, ppm_tol, out)
        return out

    def test_matched_query_returns_exact_intensity(self):
        scan = _make_scan([100.0, 200.0, 300.0], [10.0, 20.0, 30.0], 1.0, 500, 5, 5)
        out = self._call([scan], [0], [200.0], ppm_tol=20e-6)
        assert out[0, 0] == 20.0

    def test_unmatched_query_returns_zero(self):
        scan = _make_scan([100.0, 200.0, 300.0], [10.0, 20.0, 30.0], 1.0, 500, 5, 5)
        out = self._call([scan], [0], [250.0], ppm_tol=20e-6)
        assert out[0, 0] == 0.0

    def test_empty_scan_yields_all_zero_row(self):
        scan = _make_scan([], [], 1.0, 500, 5, 5)
        out = self._call([scan], [0], [100.0, 200.0], ppm_tol=20e-6)
        assert np.all(out == 0.0)

    def test_boundary_just_inside_is_matched(self):
        q = 500.0
        tol = 20e-6
        # peak sits exactly at (q + q*tol): diff == tol*q, so should match.
        peak_mz = q + q * tol
        scan = _make_scan([peak_mz], [42.0], 1.0, 500, 5, 5)
        out = self._call([scan], [0], [q], ppm_tol=tol)
        assert out[0, 0] == 42.0

    def test_boundary_just_outside_is_not_matched(self):
        q = 500.0
        tol = 20e-6
        peak_mz = q + q * tol * 2  # clearly outside
        scan = _make_scan([peak_mz], [42.0], 1.0, 500, 5, 5)
        out = self._call([scan], [0], [q], ppm_tol=tol)
        assert out[0, 0] == 0.0

    def test_multiple_candidates_picks_closest(self):
        q = 200.0
        # Two peaks both within tolerance; right one is closer.
        tol = 100e-6
        scan = _make_scan(
            [q - q * tol * 0.5, q + q * tol * 0.2],
            [10.0, 77.0],
            1.0, 500, 5, 5,
        )
        out = self._call([scan], [0], [q], ppm_tol=tol)
        assert out[0, 0] == 77.0


# ---------------------------------------------------------------------------
# _scan_metadata + _build_peak_lists + _build_window_csr
# ---------------------------------------------------------------------------


class TestCSRBuilders:
    def _build(self):
        scans = [
            _make_scan([100.0, 101.0], [1.0, 2.0], rt=5.0, win_target=500, win_lower=5, win_upper=5),
            _make_scan([150.0, 151.0, 152.0], [3.0, 4.0, 5.0], rt=1.0, win_target=500, win_lower=5, win_upper=5),
            _make_scan([200.0], [9.0], rt=3.0, win_target=510, win_lower=5, win_upper=5),
        ]
        return scans

    def test_peak_lists_round_trip(self):
        scans = self._build()
        pmz, pin, _mob = _build_peak_lists(list(scans), False)
        rt, lo, hi = _scan_metadata(list(scans))
        for i, s in enumerate(scans):
            np.testing.assert_array_equal(pmz[i], s.mz)
            np.testing.assert_array_equal(pin[i], s.intens)
        np.testing.assert_array_equal(rt, np.array([5.0, 1.0, 3.0]))
        np.testing.assert_array_equal(lo, np.array([495.0, 495.0, 505.0]))
        np.testing.assert_array_equal(hi, np.array([505.0, 505.0, 515.0]))

    def test_window_grouping_and_rt_sort(self):
        scans = self._build()
        rt, lo, hi = _scan_metadata(list(scans))
        wl, wh, offs, idx, rts = _build_window_csr(rt, lo, hi)
        # Two distinct windows: (495, 505) and (505, 515).
        assert wl.shape[0] == 2
        # Find the 495/505 window group and verify RT-sorted scan indices.
        for w in range(wl.shape[0]):
            group = idx[offs[w]:offs[w + 1]]
            group_rts = rts[offs[w]:offs[w + 1]]
            # Must be sorted ascending by RT.
            assert np.all(np.diff(group_rts) >= 0)
            # Scan indices must reference the real scans.
            for s_idx, s_rt in zip(group, group_rts):
                assert scans[s_idx].RT == s_rt


class TestMatchAndFillIM:
    """IM-gated matching kernel: keep only peaks within im_tol of the reference IM."""

    @staticmethod
    def _lists(mz_per_scan, int_per_scan, mob_per_scan):
        """Typed peak lists from per-scan values, via the real builder.

        Note this casts mobility to PEAK_MOB_DTYPE (float32), exactly as the
        production path does, so IM comparisons here carry float32 precision.
        """
        scans = [
            SimpleNamespace(mz=np.asarray(m, dtype=np.float64),
                            intens=np.asarray(i, dtype=np.float64),
                            mobility=np.asarray(b, dtype=np.float64))
            for m, i, b in zip(mz_per_scan, int_per_scan, mob_per_scan)
        ]
        return _build_peak_lists(scans, True)

    def _peaks(self):
        # 3 scans, one peak each at m/z 500; scans 0,1 at IM ~0.90, scan 2 an
        # interferer at IM 1.30.
        pmz, pin, pmob = self._lists(
            [[500.0], [500.0], [500.0]],
            [[10.0], [20.0], [99.0]],
            [[0.90], [0.91], [1.30]],
        )
        scan_idx = np.array([0, 1, 2], dtype=np.int64)
        frag_mz = np.array([500.0])
        return pmz, pin, pmob, scan_idx, frag_mz

    def test_median_reference_gates_interferer(self):
        pmz, pin, pmob, si, fm = self._peaks()
        out = np.empty((3, 1)); mob = np.empty((3, 1))
        pim = _match_and_fill_im_numba(pmz, pin, pmob, si, 3, fm,
                                       20e-6, 0.05, np.nan, out, mob)
        # median(0.90, 0.91, 1.30); float32 mobility, so not exact to 1e-9.
        assert abs(pim - 0.91) < 1e-6
        assert out[0, 0] == 10.0 and out[1, 0] == 20.0 and out[2, 0] == 0.0

    def test_fixed_reference(self):
        pmz, pin, pmob, si, fm = self._peaks()
        out = np.empty((3, 1)); mob = np.empty((3, 1))
        _match_and_fill_im_numba(pmz, pin, pmob, si, 3, fm,
                                 20e-6, 0.05, 1.30, out, mob)
        assert out[0, 0] == 0.0 and out[1, 0] == 0.0 and out[2, 0] == 99.0

    def test_co_matching_peaks_are_summed(self):
        # One scan with two peaks at m/z 500 within im_tol of the reference IM,
        # plus an out-of-tol interferer. The two in-tol peaks must be summed.
        pmz, pin, pmob = self._lists(          # single scan, 3 peaks
            [[500.0, 500.0, 500.0]], [[10.0, 20.0, 99.0]], [[0.90, 0.91, 1.30]])
        si = np.array([0], dtype=np.int64)
        fm = np.array([500.0])
        out = np.empty((1, 1)); mob = np.empty((1, 1))
        _match_and_fill_im_numba(pmz, pin, pmob, si, 1, fm,
                                 20e-6, 0.05, 0.905, out, mob)
        assert out[0, 0] == 30.0   # 10 + 20; the 1.30 interferer excluded

    def test_no_mobility_falls_back_to_no_gating(self):
        pmz, pin, pmob = self._lists(
            [[500.0], [500.0], [500.0]],
            [[10.0], [20.0], [99.0]],
            [[np.nan], [np.nan], [np.nan]],
        )
        si = np.array([0, 1, 2], dtype=np.int64)
        fm = np.array([500.0])
        out = np.empty((3, 1)); mob = np.empty((3, 1))
        r = _match_and_fill_im_numba(pmz, pin, pmob, si, 3, fm,
                                     20e-6, 0.05, np.nan, out, mob)
        assert np.isnan(r)
        assert out[0, 0] == 10.0 and out[1, 0] == 20.0 and out[2, 0] == 99.0


# ---------------------------------------------------------------------------
# _covering_scans_numba
# ---------------------------------------------------------------------------


class TestCoveringScans:
    def _build(self, scans):
        rt, lo, hi = _scan_metadata(list(scans))
        return _build_window_csr(rt, lo, hi)

    def test_window_coverage_only(self):
        scans = [
            _make_scan([100.0], [1.0], rt=10.0, win_target=500, win_lower=5, win_upper=5),
            _make_scan([100.0], [1.0], rt=10.0, win_target=600, win_lower=5, win_upper=5),
        ]
        wl, wh, offs, idx, rts = self._build(scans)
        out = np.empty(10, dtype=np.int64)
        n = _covering_scans_numba(
            500.0, 10.0, 100.0, wl, wh, offs, idx, rts, out,
        )
        # 500 is in (495, 505); 600 is in (595, 605). Only first covers.
        assert n == 1
        assert out[0] == 0

    def test_rt_boundary_inclusive(self):
        scans = [
            _make_scan([100.0], [1.0], rt=0.0, win_target=500, win_lower=5, win_upper=5),
            _make_scan([100.0], [1.0], rt=2.0, win_target=500, win_lower=5, win_upper=5),
            _make_scan([100.0], [1.0], rt=4.0, win_target=500, win_lower=5, win_upper=5),
        ]
        wl, wh, offs, idx, rts = self._build(scans)
        out = np.empty(10, dtype=np.int64)
        # apex=2, halfwidth=2 -> [0, 4]; all three scans included.
        n = _covering_scans_numba(500.0, 2.0, 2.0, wl, wh, offs, idx, rts, out)
        assert n == 3

    def test_outside_all_windows_returns_zero(self):
        scans = [
            _make_scan([100.0], [1.0], rt=1.0, win_target=500, win_lower=5, win_upper=5),
        ]
        wl, wh, offs, idx, rts = self._build(scans)
        out = np.empty(10, dtype=np.int64)
        n = _covering_scans_numba(700.0, 1.0, 5.0, wl, wh, offs, idx, rts, out)
        assert n == 0

    def test_buffer_capacity_respected(self):
        scans = [
            _make_scan([100.0], [1.0], rt=float(i), win_target=500, win_lower=5, win_upper=5)
            for i in range(5)
        ]
        wl, wh, offs, idx, rts = self._build(scans)
        out = np.empty(2, dtype=np.int64)  # only two slots
        n = _covering_scans_numba(500.0, 2.0, 10.0, wl, wh, offs, idx, rts, out)
        assert n == 2


# ---------------------------------------------------------------------------
# _pairwise_corr_and_features_numba
# ---------------------------------------------------------------------------


class TestPairwiseFeatures:
    def test_perfectly_correlated_pair_excluded(self):
        # r == 1.0 pairs are excluded (likely same fragment matched twice).
        # Two perfectly correlated columns -> 0 valid pairs -> all NaN / -1.
        matrix = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0], [4.0, 8.0]], dtype=np.float64)
        out = np.empty(_N_KERNEL_FEATURES, dtype=np.float64)
        _pairwise_corr_and_features_numba(matrix, out)
        assert np.all(np.isnan(out[:8]))
        assert np.all(out[8:] == -1.0)  # pairwise kernel output only

    def test_near_perfect_correlation_kept(self):
        # r very close to 1.0 but not exactly 1.0 should be kept.
        matrix = np.array(
            [[1.0, 2.01], [2.0, 4.0], [3.0, 6.0], [4.0, 8.0]], dtype=np.float64
        )
        out = np.empty(_N_KERNEL_FEATURES, dtype=np.float64)
        _pairwise_corr_and_features_numba(matrix, out)
        assert out[0] > 0.999  # mean should be very high but not exactly 1.0
        assert out[0] < 1.0
        assert out[28] > 0.999  # top1 pair

    def test_anti_correlated_pair(self):
        matrix = np.array([[1.0, 4.0], [2.0, 3.0], [3.0, 2.0], [4.0, 1.0]], dtype=np.float64)
        out = np.empty(_N_KERNEL_FEATURES, dtype=np.float64)
        _pairwise_corr_and_features_numba(matrix, out)
        assert out[0] == pytest.approx(-1.0, abs=1e-12)
        assert out[2] == pytest.approx(-1.0, abs=1e-12)
        assert out[3] == pytest.approx(-1.0, abs=1e-12)
        assert out[6] == pytest.approx(0.0, abs=1e-12)  # fraction > 0.5
        assert out[7] == pytest.approx(-1.0, abs=1e-12)
        assert out[28] == pytest.approx(-1.0, abs=1e-12)  # top1 pair

    def test_zero_variance_column_excluded(self):
        # column 0 is constant -> no variance; pair with column 1 excluded.
        # Remaining cols 1 and 2 are highly (not perfectly) correlated.
        matrix = np.array(
            [
                [5.0, 1.0, 2.1],
                [5.0, 2.0, 4.0],
                [5.0, 3.0, 6.0],
                [5.0, 4.0, 7.9],
            ],
            dtype=np.float64,
        )
        out = np.empty(_N_KERNEL_FEATURES, dtype=np.float64)
        _pairwise_corr_and_features_numba(matrix, out)
        # Only one valid pair (cols 1 & 2, col 0 excluded for zero variance).
        assert out[0] > 0.99  # mean should be very high
        assert out[0] < 1.0

    def test_single_column_nan(self):
        matrix = np.array([[1.0], [2.0], [3.0]], dtype=np.float64)
        out = np.empty(_N_KERNEL_FEATURES, dtype=np.float64)
        _pairwise_corr_and_features_numba(matrix, out)
        assert np.all(np.isnan(out[:8]))
        assert np.all(out[8:] == -1.0)  # pairwise kernel output only

    def test_frac_above_half_hand_counted(self):
        # Three columns: (col0, col1) r~0.98, (col0, col2) r~-0.98, (col1, col2) r~-0.98
        # Pairs with r > 0.5: 1/3 ≈ 0.3333
        matrix = np.array(
            [
                [1.0, 2.1, 4.0],
                [2.0, 4.0, 3.0],
                [3.0, 5.9, 2.0],
                [4.0, 8.0, 1.0],
            ],
            dtype=np.float64,
        )
        out = np.empty(_N_KERNEL_FEATURES, dtype=np.float64)
        _pairwise_corr_and_features_numba(matrix, out)
        assert out[6] == pytest.approx(1.0 / 3.0, abs=1e-6)

    def test_all_slots_written(self):
        # Fill scratch with sentinel then ensure nothing stays at the sentinel.
        matrix = np.array(
            [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0]], dtype=np.float64
        )
        sentinel = -123.456
        out = np.full(_N_KERNEL_FEATURES, sentinel, dtype=np.float64)
        _pairwise_corr_and_features_numba(matrix, out)
        assert not np.any(out == sentinel)


# ---------------------------------------------------------------------------
# compute_fragment_correlations end-to-end
# ---------------------------------------------------------------------------


def _build_coherent_scenario(with_isotope=False):
    """Build synthetic inputs where one precursor's fragments co-elute with a
    clean Gaussian so correlation features are predictable.
    """
    # Two library fragments, plus optional isotope copy at +1 Da to verify
    # that isotopes are filtered out before correlation.
    frag_names = ["b3_1", "y4_1"]
    frag_mz = [200.0, 400.0]
    if with_isotope:
        frag_names.append("b3_1_iso1")
        frag_mz.append(201.0)

    entries = [
        {
            "seq": "PEPTIDE",
            "z": 2,
            "prec_mz": 500.0,
            "frag_names": frag_names,
            "frag_mz": frag_mz,
        },
        {
            "seq": "DECOYSHUF",  # shuffled — different fragment m/z
            "z": 2,
            "prec_mz": 500.0,  # same isolation window
            "frag_names": ["b3_1", "y4_1"],
            "frag_mz": [210.0, 410.0],
        },
    ]
    library = _make_library(entries)

    # Build MS2 scans. Gaussian-ish intensity peak at apex RT ~ 10.
    rts = np.linspace(5.0, 15.0, 11)
    gauss = np.exp(-0.5 * ((rts - 10.0) / 1.0) ** 2)
    target_intens_b = 1000.0 * gauss
    # Small perturbation so b3/y4 are highly correlated but not r == 1.0
    target_intens_y = 1500.0 * gauss + np.array([0, 0, 1, 0, -1, 2, 0, -1, 0, 1, 0], dtype=np.float64)
    target_intens_iso = 300.0 * gauss  # would correlate perfectly if included

    scans = []
    for i, rt in enumerate(rts):
        mz = [200.0, 400.0]
        ints = [target_intens_b[i], target_intens_y[i]]
        if with_isotope:
            mz.append(201.0)
            ints.append(target_intens_iso[i])
        # Also include decoy fragments in the same scan so both precursors see data.
        mz.extend([210.0, 410.0])
        ints.extend([
            800.0 * gauss[i] * (1.0 + 0.1 * np.sin(i)),  # somewhat noisy
            500.0 * gauss[i] * (1.0 + 0.2 * np.cos(i)),
        ])
        order = np.argsort(mz)
        mz = np.array(mz)[order]
        ints = np.array(ints)[order]
        scans.append(_make_scan(mz, ints, rt=rt, win_target=500, win_lower=5, win_upper=5))

    spectra = _make_spectra(scans)
    return library, spectra


class TestPublicAPI:
    def test_output_shape_and_index(self):
        library, spectra = _build_coherent_scenario()
        fdc = pd.DataFrame(
            {
                "seq": ["PEPTIDE", "DECOYSHUF"],
                "z": [2, 2],
                "rt": [10.0, 10.0],
                "coeff": [5.0, 3.0],
            },
            index=[11, 22],
        )
        result = compute_fragment_correlations(
            spectra=spectra,
            library=library,
            fdc=fdc,
            fwhm=2.0,
            mz_tol=50e-6,
        )
        assert list(result.columns) == list(FEATURE_COLUMNS)
        assert list(result.index) == [11, 22]

    def test_negative_coeff_yields_nan_or_missing(self):
        library, spectra = _build_coherent_scenario()
        fdc = pd.DataFrame(
            {
                "seq": ["PEPTIDE"],
                "z": [2],
                "rt": [10.0],
                "coeff": [-1.0],
            }
        )
        result = compute_fragment_correlations(
            spectra=spectra, library=library, fdc=fdc, fwhm=2.0, mz_tol=50e-6,
        )
        row = result.iloc[0].values
        # Summary features are NaN
        assert np.all(np.isnan(row[:10]))
        # Ranked features are -1 (missing), precursor-frag features are NaN
        assert np.all(row[10:-2] == -1.0)
        assert np.all(np.isnan(row[-2:]))

    def test_target_high_correlation(self):
        library, spectra = _build_coherent_scenario()
        fdc = pd.DataFrame(
            {
                "seq": ["PEPTIDE"],
                "z": [2],
                "rt": [10.0],
                "coeff": [5.0],
            }
        )
        result = compute_fragment_correlations(
            spectra=spectra, library=library, fdc=fdc, fwhm=3.0, mz_tol=50e-6,
        )
        # Fragments are highly correlated (small perturbation) but not r == 1.0.
        assert result["mean_frag_corr"].iloc[0] > 0.99
        assert result["mean_frag_corr"].iloc[0] < 1.0
        assert result["n_corr_frags"].iloc[0] == 2
        assert result["n_corr_scans"].iloc[0] > 0
        assert result["top1_pair_corr"].iloc[0] > 0.99

    def test_isotope_fragments_filtered(self):
        library, spectra = _build_coherent_scenario(with_isotope=True)
        fdc = pd.DataFrame(
            {
                "seq": ["PEPTIDE"],
                "z": [2],
                "rt": [10.0],
                "coeff": [5.0],
            }
        )
        result = compute_fragment_correlations(
            spectra=spectra, library=library, fdc=fdc, fwhm=3.0, mz_tol=50e-6,
        )
        # Isotope column should not inflate the fragment count.
        assert result["n_corr_frags"].iloc[0] == 2

    def test_decoy_uses_own_fragments(self):
        library, spectra = _build_coherent_scenario()
        fdc = pd.DataFrame(
            {
                "seq": ["DECOYSHUF"],
                "z": [2],
                "rt": [10.0],
                "coeff": [3.0],
            }
        )
        result = compute_fragment_correlations(
            spectra=spectra, library=library, fdc=fdc, fwhm=3.0, mz_tol=50e-6,
        )
        # Decoy fragments live at 210 and 410 (not 200/400). Both present
        # in every scan, so n_corr_frags == 2 regardless of what the target
        # looks like.
        assert result["n_corr_frags"].iloc[0] == 2
        # And the correlation should NOT be a perfect 1.0 since we added
        # noise on the decoy traces.
        assert result["mean_frag_corr"].iloc[0] < 0.999

    def test_unknown_key_yields_nan_or_missing(self):
        library, spectra = _build_coherent_scenario()
        fdc = pd.DataFrame(
            {
                "seq": ["UNKNOWN"],
                "z": [2],
                "rt": [10.0],
                "coeff": [5.0],
            }
        )
        result = compute_fragment_correlations(
            spectra=spectra, library=library, fdc=fdc, fwhm=3.0, mz_tol=50e-6,
        )
        row = result.iloc[0].values
        assert np.all(np.isnan(row[:10]))
        assert np.all(row[10:-2] == -1.0)
        assert np.all(np.isnan(row[-2:]))

    def test_fwhm_none_returns_nan_or_missing(self):
        library, spectra = _build_coherent_scenario()
        fdc = pd.DataFrame(
            {
                "seq": ["PEPTIDE"],
                "z": [2],
                "rt": [10.0],
                "coeff": [5.0],
            }
        )
        result = compute_fragment_correlations(
            spectra=spectra, library=library, fdc=fdc, fwhm=None, mz_tol=50e-6,
        )
        row = result.iloc[0].values
        assert np.all(np.isnan(row[:10]))
        assert np.all(row[10:-2] == -1.0)
        assert np.all(np.isnan(row[-2:]))


# ---------------------------------------------------------------------------
# Thread-safety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    def test_concurrent_calls_bit_identical(self):
        library, spectra = _build_coherent_scenario()
        fdc = pd.DataFrame(
            {
                "seq": ["PEPTIDE", "DECOYSHUF", "PEPTIDE"],
                "z": [2, 2, 2],
                "rt": [10.0, 10.0, 9.5],
                "coeff": [5.0, 3.0, 4.0],
            }
        )
        reference = compute_fragment_correlations(
            spectra=spectra, library=library, fdc=fdc, fwhm=3.0, mz_tol=50e-6,
        ).to_numpy()

        results = []
        errors = []

        def worker():
            try:
                r = compute_fragment_correlations(
                    spectra=spectra, library=library, fdc=fdc,
                    fwhm=3.0, mz_tol=50e-6,
                ).to_numpy()
                results.append(r)
            except Exception as exc:  # pragma: no cover - defensive
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert len(results) == 4
        for r in results:
            np.testing.assert_array_equal(
                np.where(np.isnan(r), -1.0, r),
                np.where(np.isnan(reference), -1.0, reference),
            )


# ---------------------------------------------------------------------------
# IM-aware single-bin covering-scan selection
# ---------------------------------------------------------------------------


class TestIMBinSelection:
    """With IM data, each precursor's XIC must be built from ONE IM bin across RT,
    not the union of all overlapping IM bins at each RT."""

    def _scenario(self):
        library = _make_library([
            {"seq": "PEPTIDE", "z": 2, "prec_mz": 500.0,
             "frag_names": ["b3_1", "y4_1"], "frag_mz": [200.0, 400.0]},
        ])
        rts = np.linspace(5.0, 15.0, 11)
        g = np.exp(-0.5 * ((rts - 10.0) / 1.0) ** 2)
        b = 1000.0 * g
        y = 1500.0 * g + np.array([0, 0, 1, 0, -1, 2, 0, -1, 0, 1, 0], dtype=np.float64)
        scans = []
        for i, rt in enumerate(rts):
            # bin A [0.90, 0.95]: the precursor's real, coherent fragments
            scans.append(_make_im_scan([200.0, 400.0], [b[i], y[i]],
                                       [0.925, 0.925], rt, 0.90, 0.95))
            # bin B [0.95, 1.00]: interfering y4-only peak at a higher IM
            scans.append(_make_im_scan([400.0], [700.0 * g[i]],
                                       [0.975], rt, 0.95, 1.00))
        return library, _make_spectra(scans)

    def test_selects_single_im_bin(self):
        library, spectra = self._scenario()
        fdc = pd.DataFrame({"seq": ["PEPTIDE"], "z": [2], "rt": [10.0], "coeff": [5.0]})

        # IM on: precursor IM ~0.925 -> only bin A scans, one per RT (11), clean corr.
        on = compute_fragment_correlations(
            spectra=spectra, library=library, fdc=fdc,
            fwhm=3.0, mz_tol=50e-6, im_tol=0.02)
        assert on["n_corr_scans"].iloc[0] == 11
        assert on["mean_frag_corr"].iloc[0] > 0.99

        # IM off: all bins mixed -> both bins' scans used (22).
        off = compute_fragment_correlations(
            spectra=spectra, library=library, fdc=fdc,
            fwhm=3.0, mz_tol=50e-6, im_tol=0.0)
        assert off["n_corr_scans"].iloc[0] == 22
