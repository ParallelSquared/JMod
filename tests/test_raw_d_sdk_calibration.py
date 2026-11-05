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

"""A .d without peaks.parquet is read raw, with Bruker's calibration tables."""

import os

import numpy as np
import pytest

from src.utils.io import file_reader as fr


def _cal(n_tof=5, n_scan=3):
    return {"mz_values": np.linspace(100.0, 200.0, n_tof),
            "mobility_values": np.linspace(1.4, 0.6, n_scan)}


def _write_tables(tables_dir, mz, ook0):
    os.makedirs(tables_dir, exist_ok=True)
    np.save(os.path.join(tables_dir, "mz.npy"), mz)
    np.save(os.path.join(tables_dir, "ook0.npy"), ook0)
    with open(os.path.join(tables_dir, "calibration.json"), "w") as f:
        f.write("{}")


def test_tables_replace_the_estimates(tmp_path):
    cal = _cal()
    mz = np.linspace(100.1, 200.1, 5)
    ook0 = np.linspace(1.41, 0.61, 3)
    _write_tables(tmp_path, mz, ook0)
    fr._load_sdk_tables(str(tmp_path), cal)
    assert np.array_equal(cal["mz_values"], mz)
    assert np.array_equal(cal["mobility_values"], ook0)


@pytest.mark.parametrize("mz, ook0", [
    (np.zeros(4), np.zeros(3)),                       # too few TOF indices
    (np.zeros(5), np.zeros(4)),                       # too many scans
    (np.zeros(5, dtype=np.float32), np.zeros(3)),     # wrong dtype
])
def test_tables_must_match_the_estimates_shape(tmp_path, mz, ook0):
    _write_tables(tmp_path, mz, ook0)
    with pytest.raises(ValueError, match="expected float64"):
        fr._load_sdk_tables(str(tmp_path), _cal())


def test_cached_tables_are_reused(tmp_path, monkeypatch):
    tables = tmp_path / fr._SDK_TABLES_DIR
    _write_tables(tables, np.zeros(5), np.zeros(3))
    monkeypatch.setattr("subprocess.run", lambda *a, **k: pytest.fail("recomputed"))
    assert fr._sdk_calibration_tables(str(tmp_path)) == str(tables)


def test_wine_path():
    assert fr._wine_path("/Users/me/run 1.d") == r"Z:\Users\me\run 1.d"


def _fake_d(tmp_path, parquet):
    d = tmp_path / "run_1.d"
    d.mkdir()
    for name in ("analysis.tdf", "analysis.tdf_bin") + (("peaks.parquet",) if parquet else ()):
        (d / name).write_bytes(b"")
    return d


@pytest.fixture
def stub_loader(monkeypatch):
    """Stub everything _load_d calls; record which peak source it chose."""
    calls = {}
    monkeypatch.setattr(fr, "_load_calibration", lambda p: dict(
        _cal(), frame_ids=None, ms_level=None, im_range_lower=0.6, im_range_upper=1.4))
    monkeypatch.setattr(fr, "_load_dia_windows", lambda *a: None)
    monkeypatch.setattr(fr, "_sdk_calibration_tables",
                        lambda d, sdk=None: calls.setdefault("tables", d) and "TABLES")
    monkeypatch.setattr(fr, "_load_sdk_tables",
                        lambda t, cal: calls.setdefault("loaded", t))
    monkeypatch.setattr(fr, "_centroid_d", lambda *a: pytest.fail("must not centroid"))

    class SF:
        ms1scans = ms2scans = []

    def _build(filepath, peaks_path, cal, dia, bins, raw_d_path=None):
        calls["build"] = (peaks_path, raw_d_path)
        return SF()

    monkeypatch.setattr(fr, "_build_spectrum_file", _build)
    return calls


def test_without_peaks_parquet_reads_raw_with_sdk_tables(tmp_path, stub_loader):
    d = _fake_d(tmp_path, parquet=False)
    fr._load_d(str(d) + "/")
    assert stub_loader["tables"] == str(d)
    assert stub_loader["loaded"] == "TABLES"
    assert stub_loader["build"] == (None, str(d))
    assert not (d / "peaks.parquet").exists()


def test_a_peaks_parquet_is_ignored_and_the_raw_ions_are_read(tmp_path, stub_loader):
    # .d data is always read raw with Bruker's calibration; a centroided
    # peaks.parquet in the folder is not used.
    d = _fake_d(tmp_path, parquet=True)
    fr._load_d(str(d))
    assert stub_loader["tables"] == str(d)
    assert stub_loader["loaded"] == "TABLES"
    assert stub_loader["build"] == (None, str(d))
