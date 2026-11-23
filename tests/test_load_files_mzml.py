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
    _mzml_scans,
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


def _write_mzml(path, rt, unit_accession, unit_name, source_format=None):
    """A one-spectrum MS1 mzML whose scan start time is *rt* in the given unit,
    converted from a source file of *source_format* (an accession), if given."""
    import base64
    arr = lambda vals: base64.b64encode(np.array(vals, dtype="<f8").tobytes()).decode()
    source = "" if source_format is None else f"""
  <fileDescription>
    <fileContent/>
    <sourceFileList count="1">
      <sourceFile id="S" name="run" location="file:///data">
        <cvParam cvRef="MS" accession="{source_format}" name="" value=""/>
      </sourceFile>
    </sourceFileList>
  </fileDescription>"""
    path.write_text(f"""<?xml version="1.0" encoding="utf-8"?>
<mzML xmlns="http://psi.hupo.org/ms/mzml" version="1.1.0">{source}
  <run id="r">
    <spectrumList count="1">
      <spectrum index="0" id="controllerType=0 controllerNumber=1 scan=7" defaultArrayLength="2">
        <cvParam cvRef="MS" accession="MS:1000511" name="ms level" value="1"/>
        <cvParam cvRef="MS" accession="MS:1000285" name="total ion current" value="30"/>
        <scanList count="1">
          <scan>
            <cvParam cvRef="MS" accession="MS:1000016" name="scan start time" value="{rt}" unitCvRef="UO" unitAccession="{unit_accession}" unitName="{unit_name}"/>
            <scanWindowList count="1"><scanWindow>
              <cvParam cvRef="MS" accession="MS:1000501" name="scan window lower limit" value="350"/>
              <cvParam cvRef="MS" accession="MS:1000500" name="scan window upper limit" value="1500"/>
            </scanWindow></scanWindowList>
          </scan>
        </scanList>
        <binaryDataArrayList count="2">
          <binaryDataArray encodedLength="0">
            <cvParam cvRef="MS" accession="MS:1000523" name="64-bit float" value=""/>
            <cvParam cvRef="MS" accession="MS:1000576" name="no compression" value=""/>
            <cvParam cvRef="MS" accession="MS:1000514" name="m/z array" value=""/>
            <binary>{arr([400.1, 500.2])}</binary>
          </binaryDataArray>
          <binaryDataArray encodedLength="0">
            <cvParam cvRef="MS" accession="MS:1000523" name="64-bit float" value=""/>
            <cvParam cvRef="MS" accession="MS:1000576" name="no compression" value=""/>
            <cvParam cvRef="MS" accession="MS:1000515" name="intensity array" value=""/>
            <binary>{arr([10.0, 20.0])}</binary>
          </binaryDataArray>
        </binaryDataArrayList>
      </spectrum>
    </spectrumList>
  </run>
</mzML>
""")


THERMO_RAW, BRUKER_TDF = "MS:1000563", "MS:1002817"
MINUTE, SECOND = ("UO:0000031", "minute"), ("UO:0000010", "second")


@pytest.mark.parametrize("source_format, rt, unit, expected", [
    (THERMO_RAW, "1.5", MINUTE, 1.5),   # Thermo: minutes, as .raw is read
    (THERMO_RAW, "90", SECOND, 1.5),
    (BRUKER_TDF, "90", SECOND, 90.0),   # Bruker: seconds, as .d is read
    (BRUKER_TDF, "1.5", MINUTE, 90.0),
    (None, "1.5", MINUTE, 1.5),         # other sources: as recorded
    (None, "90", SECOND, 90.0),
])
def test_mzml_scan_start_time_in_the_source_vendors_unit(tmp_path, source_format, rt, unit, expected):
    path = tmp_path / "one.mzML"
    _write_mzml(path, rt, *unit, source_format=source_format)
    (scan,) = list(_mzml_scans(str(path)))
    spec = Spectrum(scan=scan)
    assert spec.RT == pytest.approx(expected)
    assert spec.scan_num == 7
    np.testing.assert_allclose(spec.mz, [400.1, 500.2], rtol=1e-6)
    np.testing.assert_allclose(spec.intens, [10.0, 20.0], rtol=1e-6)
