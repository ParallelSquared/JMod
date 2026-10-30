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

import numpy as np
import pandas as pd
import polars as pl
import os
import types
import pytest

from src.utils.errors import JModError
from src.fdr_analysis import (score_precursors, score_model, add_median_based_features, process_data,
                               compute_protein_FDR, log_df, untag_prec_qvalues,
                               area, walk_to_local_max, gauss_r2_joint, ms1_quality)

def test_score_model_minimal_fixed():
    # 12 samples, 2 classes
    X = pd.DataFrame({
        "feat1": np.arange(12),
        "feat2": np.arange(12, 24)
    })
    y = np.array([0, 1] * 6)  # 6 samples per class

    for model_type in ["rf", "lda", "nn"]:
        sm = score_model(model_type=model_type, n_splits=3)
        preds = sm.run_model(X, y)
        assert len(preds) == len(y)
        assert isinstance(preds, np.ndarray)

    # unsupported model type
    with pytest.raises(JModError, match="Unsupported model type"):
        sm = score_model("unsupported", n_splits=3)
        sm.run_model(X, y)

def test_score_precursors_minimal_fixed(tmp_path):
    # Need enough samples for 5-fold CV with iterative negative mining
    np.random.seed(42)
    n = 30
    is_decoy = [False, True] * (n // 2)
    fdc = pd.DataFrame({
        "coeff": np.random.uniform(0.5, 3.0, n),
        "is_decoy": is_decoy,
        "stripped_seq": [f"SEQ{i}" for i in range(n)],
        "rt_error": np.random.uniform(0.0, 1.0, n),
        "mz_error": np.random.uniform(0.0, 0.1, n),
        "feature1": np.random.uniform(1.0, 5.0, n),
        "feature2": np.random.uniform(1.0, 5.0, n),
    })

    out = score_precursors(
        fdc,
        model_type="xg",
        fdr_t=0.01,
        folder=str(tmp_path),  # ensures the plot-saving code runs
        target_decoy_ratio=1.0,
    )

    assert isinstance(out, pd.DataFrame)
    assert len(out) == len(fdc)


def test_add_median_based_features_basic():
    # minimal dummy dataframe
    df = pd.DataFrame({
        "untag_prec": ["A", "A", "B"],
        "channels_matched": [2, 2, 1],
        "m1": [10.0, 20.0, 30.0],
    })

    metric_columns = ["m1"]

    out = add_median_based_features(df, metric_columns)

    # basic: output must be DataFrame
    assert isinstance(out, pd.DataFrame)

    # check expected new columns exist
    assert "median_m1" in out.columns
    assert "diff_m1_from_median" in out.columns

    # check same number of rows
    assert len(out) == len(df)


def test_process_data_creates_output_files(tmp_path, monkeypatch):
    dummy_file = tmp_path / "input.mzML"
    dummy_file.write_text("dummy")  # create file
    (tmp_path.parent / "outputs").mkdir(exist_ok=True)

    spectra = None
    library = None

    monkeypatch.setattr("src.fdr_analysis.get_large_prec",
                        lambda *args, **kwargs: (None,
                                                 pd.DataFrame({
                                                     "seq": ["AAAA", "BBBB"],
                                                     "z": [2, 2],
                                                     "rt": [10.0, 20.0],
                                                     "mz": [500.0, 600.0],
                                                     "rt_error": [0.1, -0.2],
                                                     "mz_error": [5, -10],
                                                     "gof_stats": [0.1, 0.2],
                                                     "scribe_scores": [1, 2],
                                                     "max_matched_residuals": [0.5, 0.8],
                                                     "manhattan_distances": [5, 6],
                                                     "frag_errors": [[1.0, 2.0], [2.0, 3.0]],
                                                     "frac_lib_int": [1, 1],
                                                     "file_name": ["f", "f"],
                                                     "time_channel": [0, 0],
                                                     "silac_channel": [np.nan, np.nan],
                                                     "is_decoy": [False, False],
                                                     "stripped_seq": ["AAAA", "BBBB"],
                                                 }),
                                                 None))

    monkeypatch.setattr("src.fdr_analysis.score_precursors",
                        lambda fdc, *args, **kwargs: fdc.assign(
                            PredVal=[0.5, 0.7],
                            Qvalue=[0.01, 0.02],
                        ))

    monkeypatch.setattr("src.fdr_analysis.ms1_quant",
                        lambda *args, **kwargs: args[0].assign(plex_Area=[100.0, 0.0]))

    monkeypatch.setattr("src.fdr_analysis.compute_protein_FDR",
                        lambda df, results_folder=None, **kwargs: df.assign(
                            Protein_Qvalue=1.0,
                            protein="P12345"
                        ))

    monkeypatch.setattr(
        "src.fdr_analysis.unstring_floats",
        lambda x: x if isinstance(x, list) else [1.0, 2.0]
    )

    process_data(str(dummy_file), spectra, library,
                 ms1_tol=20e-6, rt_tol=1.0, im_tol=0.01, target_decoy_ratio=1.0)

    expected_files = [
        "outputs/all_IDs.csv",
        "filtered_IDs.csv",
        "outputs/all_IDs_filtered.parquet",
    ]

    for fname in expected_files:
        assert (tmp_path.parent / fname).exists()
    # Label-free: each untagged precursor is one row, so its q-value is the row's
    written = pd.read_parquet(tmp_path.parent / "outputs/all_IDs_filtered.parquet")
    assert written["untag_prec_Qvalue"].tolist() == written["Qvalue"].tolist()
    # gauss_r2_joint in all_IDs.csv; MS1_quality only with the filtered IDs
    assert "gauss_r2_joint" in pd.read_csv(tmp_path.parent / "outputs/all_IDs.csv").columns
    assert "MS1_quality" not in pd.read_csv(tmp_path.parent / "outputs/all_IDs.csv").columns
    assert "MS1_quality" in pd.read_csv(tmp_path.parent / "filtered_IDs.csv").columns
    assert "MS1_quality" in pd.read_parquet(tmp_path.parent / "filtered_IDs.parquet").columns


def test_untag_prec_qvalues_rank_each_plex_set_by_its_best_channel():
    # Two channels each: P1 0.9, P2 0.8, decoy D1 0.7, P3 0.6 as their best.
    # Counted down that order: 1/1, 1/2, 2/2, 2/3, monotonized
    fdx = pd.DataFrame({
        "file_name": ["f"] * 8,
        "untag_prec": ["P1_2", "P1_2", "P2_2", "P2_2", "D1_2", "D1_2", "P3_2", "P3_2"],
        "is_decoy": [False] * 4 + [True] * 2 + [False] * 2,
        "PredVal": [0.9, 0.1, 0.2, 0.8, 0.7, 0.05, 0.3, 0.6],
    })
    q = untag_prec_qvalues(fdx, target_decoy_ratio=1.0)
    assert q.tolist() == pytest.approx([0.5, 0.5, 0.5, 0.5, 2 / 3, 2 / 3, 2 / 3, 2 / 3])
        
        
def test_compute_protein_FDR_minimal(monkeypatch, tmp_path):
    fake_args = types.SimpleNamespace(plexDIA=False)

    df = pd.DataFrame({
        "file_name": ["A", "A"],
        "channel": [1, 1],
        "silac_channel": [0, 0],
        "Qvalue": [0.005, 0.005],
        "PredVal": [0.9, 0.8],
        "is_decoy": [False, True],
        "protein": ["P1", "DECOY_P1"],
        "untag_prec": ["p1", "p2"],
    })

    out = compute_protein_FDR(df, results_folder=None, target_decoy_ratio=1.0)

    assert isinstance(out, pd.DataFrame)
    assert "Protein_Qvalue" in out.columns  # main thing it produces
    assert out.shape[0] == 2  # still same number of rows


def test_log_df_smoke():
    """Smoke test for log_df function - just verify it runs without error."""
    df = pd.DataFrame({
        "file_name": ["sample1_001", "sample1_002", "sample2_001"],
        "metric1": [1.0, 2.0, 3.0],
        "metric2": [4.0, 5.0, 6.0],
    })

    # log_df should run without raising exceptions
    # It just logs to the logger, so we verify it doesn't crash
    log_df(df)


def test_log_df_multiple_files():
    """Test log_df with multiple different file prefixes."""
    df = pd.DataFrame({
        "file_name": ["fileA_01", "fileA_02", "fileB_01", "fileC_01"],
        "value": [10, 20, 30, 40],
    })

    # Should handle multiple unique file prefixes
    log_df(df)


def test_plex_area_sums_the_scan_radius_on_either_side_of_the_apex():
    x = np.array([1, 2, 5, 9, 4, 3, 1.])
    assert area(x, 3) == 5 + 9 + 4
    assert area(x, 3, area_radius=2) == 2 + 5 + 9 + 4 + 3
    # The apex stays area_radius scans in from the edges
    assert walk_to_local_max(np.array([9, 5, 4, 3, 2.]), 2, area_radius=2) == 2


SCANS = np.arange(100, 109)


def _ids(ms1_traces, apexes):
    return pd.DataFrame({
        "seq": [f"PEP{i}K" for i in range(len(ms1_traces))],
        "z": 2.0,
        "plexfittrace_spec_all": [";".join(map(str, SCANS)) if t is not None else 0 for t in ms1_traces],
        "plexfittrace_all": [";".join(map(str, t)) if t is not None else 0 for t in ms1_traces],
        "ms1_apex_scan": apexes,
    })


def _ms2(seq, trace):
    return pl.DataFrame({"seq": seq, "z": 2.0, "Ms1_spec_id": SCANS, "coeff": trace}).filter(pl.col("coeff") > 0)


def test_gaussian_traces_fit_and_ragged_ones_fit_worse():
    peak = 1000 * np.exp(-0.5 * ((SCANS - 104) / 1.5) ** 2)
    ragged = np.array([0, 900, 0, 700, 50, 800, 0, 600, 0.])
    ids = _ids([peak, ragged, None], [104, 104, 0])
    dc = pl.concat([_ms2("PEP0K", peak / 10), _ms2("PEP1K", peak / 10)])
    r2 = gauss_r2_joint(ids, dc)
    assert r2[0] > 0.999
    assert r2[1] < 0.8
    assert np.isnan(r2[2])  # no MS1 fit window


def test_no_ms2_coefficients_means_no_fit():
    peak = 1000 * np.exp(-0.5 * ((SCANS - 104) / 1.5) ** 2)
    r2 = gauss_r2_joint(_ids([peak], [104]), _ms2("OTHERK", peak))
    assert np.isnan(r2[0])


def test_ms1_quality_ranks_within_the_run_and_puts_no_ms1_quant_last():
    ids = pd.DataFrame({"PredVal": [0.9, 0.95, 0.99, 0.99],
                        "gauss_r2_joint": [0.5, 0.9, np.nan, 0.7],
                        "plex_Area": [10.0, 10.0, 10.0, 0.0]})
    qq = ms1_quality(ids)
    # percentiles: PredVal 1/4, 2/4, 3.5/4, 3.5/4; gauss_r2_joint 1/3, 3/3, missing = 0, 2/3
    assert np.allclose(qq, [np.sqrt(1 / 4 * 1 / 3), np.sqrt(2 / 4), 0.0, 0.01 * np.sqrt(3.5 / 4 * 2 / 3)])
