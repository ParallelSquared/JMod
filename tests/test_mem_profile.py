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

"""Memory profiling helpers and the main-search fit profile."""

import time

import numpy as np
import polars as pl
import pytest

from src.utils import mem_profile as mp


def test_footprint_covers_a_live_allocation():
    # Not "grows by": earlier tests leave freed memory the allocator reuses.
    a = np.ones(512 * 2**20 // 8)  # 512 MB, touched
    cur, peak = mp.footprint()
    assert cur >= a.nbytes
    assert peak >= cur
    del a


def test_array_bytes_counts_each_buffer_once_and_skips_scalar_lists():
    class Holder:
        pass

    a = np.ones(1000)
    h = Holder()
    h.arrays = [a, a[::2], a.reshape(10, 100)]
    h.strings = ["x"] * 100_000
    h.nothing = None
    h.nested = {"k": [np.zeros(10, np.int32)]}
    assert mp.array_bytes(h) == a.nbytes + 40


def test_array_bytes_ignores_memmaps(tmp_path):
    m = np.lib.format.open_memmap(tmp_path / "m.npy", mode="w+", dtype=np.float64, shape=(1000,))
    assert mp.array_bytes([m, m[:10]]) == 0


def test_tracker_counts_running_fits_and_held_results():
    t = mp.SearchTracker()
    t.fit_started(); t.fit_started()
    t.fit_finished(10, 1000)
    assert (t.fits_running, t.results_held, t.results_held_bytes) == (1, 10, 1000)
    t.fit_finished(0, 0)
    t.results_written(10, 1000)
    assert (t.fits_running, t.results_held, t.results_held_bytes) == (0, 0, 0)


def test_describe_splits_the_footprint():
    snap = dict(footprint=10 * mp.GB, peak=12 * mp.GB, malloc_in_use=6 * mp.GB,
                malloc_held=7 * mp.GB, arrow=1 * mp.GB, swap=0, fits_running=3,
                results_held=1234, results_held_bytes=mp.GB // 2)
    text = mp.describe(snap, search=True)
    assert "10.0 GB (C arrays 6.0 GB in use, Arrow 1.0 GB, Polars, Python objects and freed-but-held 3.0 GB" in text
    assert "3 fits running, 1,234 results held (0.50 GB)" in text


def test_memory_watch_reports_the_phase_peak(caplog):
    with mp.MemoryWatch("test phase", interval=0.05, report_every=0.1) as w:
        a = np.ones(256 * 2**20 // 8)
        time.sleep(0.3)
        del a
    assert w.peak >= 256 * 2**20
    assert w.at_peak is not None


def test_hits_bytes_counts_arrays_and_list_containers():
    hits = (np.zeros(10), 5, ["a", "b"], "name", np.zeros(4, np.int32))
    assert mp.hits_bytes(hits) == 80 + 16 + __import__("sys").getsizeof(["a", "b"])


def test_fit_profile_handles_early_returns_and_no_mobility(tmp_path):
    from src.run_jmod import _log_fit_profile
    full = dict(scan=1, rt=10.0, prec_mz=500.0, im_lo=0.9, im_hi=1.0, raw_peaks=900,
                binned_peaks=700, candidates=400, rows=650, columns=60, nonzeros=2_000,
                fit_s=0.7, results=55, results_bytes=40_000, solve_gram_bytes=28_800,
                solve_internal_bytes=30_000)
    for i, step in enumerate(["bin_peaks", "query", "match_targets", "match_decoys", "assemble",
                              "solve", "unique_peaks", "fragment_columns", "features", "output"]):
        full[f"{step}_s"] = 0.05
        full[f"{step}_bytes"] = 1_000 * (i + 1) if step != "output" else 0
    records = [
        full,
        # no candidate matched: fit_to_lib2 returned before the solver
        dict(scan=2, rt=11.0, prec_mz=510.0, im_lo=0.9, im_hi=1.0, raw_peaks=10,
             binned_peaks=10, candidates=0, bin_peaks_s=0.01, bin_peaks_bytes=500,
             query_s=0.01, query_bytes=0, match_targets_s=0.0, match_targets_bytes=100,
             fit_s=0.02, results=0, results_bytes=0),
        # mzML: no mobility band
        dict(full, scan=3, im_lo=None, im_hi=None),
    ]
    _log_fit_profile(0, records, str(tmp_path), n_threads=2)
    saved = pl.read_parquet(tmp_path / "fit_profile" / "batch1.parquet")
    assert saved.height == 3
    # steps 1..9 bytes (45,000) + solver internals (30,000) + results (40,000)
    assert saved["working_bytes"].to_list() == [115_000, 600, 115_000]
