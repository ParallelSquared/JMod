"""
This Source Code Form is subject to the terms of the Oxford Nanopore
Technologies, Ltd. Public License, v. 1.0.  Full licence can be found
at https://github.com/ParallelSquared/JMod/blob/main/LICENSE.txt
"""

"""
Tests for functions in spectral_fitting.py
"""

import numpy as np

# Import the functions we want to test
from src.spectral_fitting import (
    get_closest_ms1, get_scribe, get_residuals, max_matched_residual, get_manhattan_distance, hyperscore2, merge_spectrum_peaks
)

def test_get_closest_ms1():
    class FakeSpec:
        def __init__(self, RT):
            self.RT = RT
    spectra = [FakeSpec(1.0), FakeSpec(2.5), FakeSpec(5.0)]
    # Patch closest_ms1spec to just pick closest manually
    def fake_closest(target, arr):
        return np.argmin(np.abs(arr - target))
    closest_ms1spec = fake_closest
    
    ms1 = get_closest_ms1(2.0, spectra)
    assert ms1.RT == 2.5

def test_get_scribe():
    row_idx_split = [np.array([0,1])]
    col_idx_split = [np.array([0,1])]
    prec_val_split = [np.array([4,9])]
    val_obs = np.array([4,9])
    
    scores = get_scribe(row_idx_split, col_idx_split, prec_val_split, val_obs)
    assert scores.shape[0] == 1
    assert scores[0] >= 0

def test_get_residuals():
    ref_val = [np.array([1])]
    ref_row = [np.array([0])]
    ref_col = [np.array([0])]
    decoy_val = [np.array([2])]
    decoy_row = [np.array([1])]
    decoy_col = [np.array([0])]
    val_obs = np.array([1.0, 2.0])
    coeffs = np.array([1.0, 1.0])
    
    r, y_pred = get_residuals(ref_val, ref_row, ref_col,
                                          decoy_val, decoy_row, decoy_col,
                                          val_obs, coeffs, 0, 1)
    assert r.shape == val_obs.shape
    assert y_pred.shape == val_obs.shape

def test_max_matched_residual():
    row_idx_split = [np.array([0,1])]
    residuals = np.array([0.5, 0.2])
    max_res = max_matched_residual(row_idx_split, residuals)
    assert max_res[0] == 0.5

def test_get_manhattan_distance():
    row_idx_split = [np.array([0,1])]
    col_idx_split = [np.array([0,1])]
    prec_val_split = [np.array([1,2])]
    val_obs = np.array([1,2])
    y_pred = np.array([1,2])
    
    manhattan, sc = get_manhattan_distance(row_idx_split, col_idx_split, prec_val_split, val_obs, y_pred)
    assert manhattan.shape[0] == 1
    assert sc.shape[0] == 1

def test_hyperscore2():
    frags = {"b1": 100, "y1": 200, "b2":50}
    frag_names_matched = ["b1", "y1", "b2"]
    score, num_b, num_y = hyperscore2(frags, frag_names_matched)
    assert score >= 0
    assert num_b == 2
    assert num_y == 1

class DummySpec:
    def __init__(self, mz, intens):
        self.mz=mz
        self.intens=intens

    def peak_list(self):
        return(np.array([self.mz,self.intens]))
    
def test_merge_spectrum_peaks():
    dia_spec = DummySpec(
        mz=np.array([100.0, 100.01]), 
        intens=np.array([10.0, 20.0])
        )
    mz_ppm = 1e-6
    merge_spectrum_peaks(dia_spec, mz_ppm)
    expected = np.array([
        [100.0, 10.0],
        [100.01, 20.0]
        ])
    output = np.array(dia_spec.peak_list(), dtype=np.float64).T
    assert np.all(output == expected), f"Expected {expected}, got {output}"

    dia_spec = DummySpec(
        mz=np.array([100.000, 100.0001]),
        intens=np.array([10.0, 20.0])
        )
    mz_ppm = 1e-6
    merge_spectrum_peaks(dia_spec, mz_ppm)
    expected = np.array([
        [100.0, 30.0]
        ])
    output = np.array(dia_spec.peak_list(), dtype=np.float64).T
    assert np.all(output == expected), f"Expected {expected}, got {output}"


    dia_spec = DummySpec(
        mz=np.array([100.000, 100.0001, 100.0002]),
        intens=np.array([10.0, 20.0, 60.0])
        )
    mz_ppm = 1e-6
    merge_spectrum_peaks(dia_spec, mz_ppm)
    expected = np.array([[100.0, 30.0], [100.0001, 60.0]])
    output = np.array(dia_spec.peak_list(), dtype=np.float64).T
    assert np.all(output == expected), f"Expected {expected}, got {output}"

    dia_spec = DummySpec(
        mz=np.array([]),
        intens=np.array([])
        )
    mz_ppm = 1e-6
    merge_spectrum_peaks(dia_spec, mz_ppm)
    expected = np.array([]).reshape(0, 2)
    output = np.array(dia_spec.peak_list(), dtype=np.float64).T
    assert np.all(output == expected), f"Expected {expected}, got {output}"

    dia_spec = DummySpec(
        mz=np.array([50.0, 60.0, 100.000, 100.0001, 200.0]),
        intens=np.array([50, 60, 10.0, 20.0, 200])
    )
    mz_ppm = 1e-6
    merge_spectrum_peaks(dia_spec, mz_ppm)
    expected = np.array([[50.0, 50],[60.0, 60],[100.0, 30.0],[200.0, 200]])
    output = np.array(dia_spec.peak_list(), dtype=np.float64).T
    assert np.all(output == expected), f"Expected {expected}, got {output}"

    print("Merge_Spectrum_Peaks Tests Passing")
        