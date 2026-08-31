"""Spectrum.get_vals against raw mzML scan dicts.

Both cases here were broken by a merge resolution that stacked two branches'
versions of these lines instead of interleaving them, and neither the suite nor
the compiler noticed: the fixture mzML carries "ion injection time" on every
scan, and a dtype change is silent.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.io.load_files import (
    Spectrum,
    PEAK_MZ_DTYPE,
    PEAK_INT_DTYPE,
)


def _ms1_scan(*, injection_time=None):
    """A minimal MS1 scan dict, shaped like what pymzML yields."""
    scan_entry = {
        "scan start time": 12.5,
        "scanWindowList": {
            "scanWindow": [{
                "scan window lower limit": 350.0,
                "scan window upper limit": 1500.0,
            }]
        },
    }
    if injection_time is not None:
        scan_entry["ion injection time"] = injection_time

    return {
        "id": "controllerType=0 controllerNumber=1 scan=42",
        "ms level": 1,
        "scanList": {"scan": [scan_entry]},
        "total ion current": 60.0,
        # float64 in, so a missing cast leaves the wrong dtype rather than
        # coincidentally matching.
        "m/z array": np.array([400.1, 500.2, 600.3], dtype=np.float64),
        "intensity array": np.array([10.0, 20.0, 30.0], dtype=np.float64),
    }


def test_missing_injection_time_defaults_to_one():
    # Regression: an unguarded scan[...]["ion injection time"] raised KeyError
    # here, which is the whole point of the guard.
    spec = Spectrum(scan=_ms1_scan())
    assert spec.injection_time == 1.0


def test_injection_time_converted_from_milliseconds():
    spec = Spectrum(scan=_ms1_scan(injection_time=50.0))
    assert spec.injection_time == pytest.approx(0.05)


def test_peaks_normalized_to_peak_dtypes():
    # Regression: a later raw re-assignment overwrote the cast, silently
    # restoring float64 peaks and a second full copy downstream.
    spec = Spectrum(scan=_ms1_scan(injection_time=50.0))
    assert spec.mz.dtype == PEAK_MZ_DTYPE
    assert spec.intens.dtype == PEAK_INT_DTYPE
    assert spec.mz.flags["C_CONTIGUOUS"]
    assert spec.intens.flags["C_CONTIGUOUS"]


def test_peak_values_survive_the_cast():
    spec = Spectrum(scan=_ms1_scan(injection_time=50.0))
    np.testing.assert_allclose(spec.mz, [400.1, 500.2, 600.3], rtol=1e-6)
    np.testing.assert_allclose(spec.intens, [10.0, 20.0, 30.0], rtol=1e-6)


def test_scan_metadata_parsed():
    spec = Spectrum(scan=_ms1_scan(injection_time=50.0))
    assert spec.scan_num == 42
    assert spec.level == 1
    assert spec.RT == 12.5
    assert spec.scanwindow == [350.0, 1500.0]
