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


def test_score_precursors_scores_the_decoy_channel_but_never_trains_on_or_counts_it(monkeypatch):
    import xgboost
    np.random.seed(0)
    n = 60
    fdc = pd.DataFrame({
        "is_decoy": [False, True] * (n // 2),
        "is_decoy_channel": [False] * 40 + [True] * 20,
        "stripped_seq": [f"SEQ{i % 30}" for i in range(n)],
        "rt_error": np.random.uniform(0.0, 1.0, n),
        "mz_error": np.random.uniform(0.0, 0.1, n),
        "feature1": np.random.uniform(1.0, 5.0, n),
    })
    trained = []
    real_train = xgboost.train
    def spy(params, dtrain, **kwargs):
        trained.append(dtrain.num_row())
        assert "is_decoy_channel" not in dtrain.feature_names
        return real_train(params, dtrain, **kwargs)
    monkeypatch.setattr("src.fdr_analysis.xgb.train", spy)

    out = score_precursors(fdc, model_type="xg", target_decoy_ratio=1.0)

    real = ~fdc["is_decoy_channel"].to_numpy()
    # the first iteration's five folds train on each real row four times, never on the decoy channel
    assert sum(trained[:5]) == 4 * real.sum()
    assert all(sum(trained[i:i + 5]) <= 4 * real.sum() for i in (5, 10))
    assert out["PredVal"].notna().all()                                      # scored
    assert out.loc[~real, "Qvalue"].isna().all()                            # not counted
    # the real rows' Qvalue counts them alone
    sub = out.loc[real]
    order = np.argsort(-sub["PredVal"].to_numpy(), kind="stable")
    decoy = sub["is_decoy"].to_numpy()[order]
    with np.errstate(divide="ignore"):
        q_ordered = np.minimum.accumulate(((1 + np.cumsum(decoy)) / np.cumsum(~decoy))[::-1])[::-1]
    expected = np.empty(order.size)
    expected[order] = q_ordered
    assert np.allclose(sub["Qvalue"].to_numpy(), expected)


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


def test_untag_prec_qvalues_leave_out_the_decoy_channel():
    # P1's best is its decoy channel (0.95), so it ranks by its real 0.4; P2 is
    # seen only in the decoy channel.  Ranked: P3 0.9, D1 0.6, P1 0.4 -> 1/1, 2/1, 2/2
    fdx = pd.DataFrame({
        "file_name": ["f"] * 6,
        "untag_prec": ["P1_2", "P1_2", "P2_2", "P3_2", "D1_2", "D1_2"],
        "is_decoy": [False, False, False, False, True, True],
        "is_decoy_channel": [False, True, True, False, False, True],
        "PredVal": [0.4, 0.95, 0.99, 0.9, 0.6, 0.97],
    })
    q = untag_prec_qvalues(fdx, target_decoy_ratio=1.0)
    assert q.iloc[[0, 1, 3, 4, 5]].tolist() == pytest.approx([1.0, 1.0, 1.0, 1.0, 1.0])
    assert np.isnan(q.iloc[2])


def test_id_counts_and_summary_leave_out_the_decoy_channel(monkeypatch, tmp_path):
    from src import config
    monkeypatch.setattr(config.args, "plexDIA", True)
    df = pd.DataFrame({
        "file_name": ["A"] * 4, "channel": [0, 0, 8, 8], "silac_channel": [0] * 4,
        "Qvalue": [0.001, 0.002, np.nan, np.nan], "BestChannel_Qvalue": [0.001, 0.002, 0.001, 0.002],
        "PredVal": [0.9, 0.8, 0.95, 0.85], "is_decoy": [False] * 4,
        "is_decoy_channel": [False, False, True, True],
        "protein": ["P1", "P2", "P1", "P2"], "untag_prec": ["p1", "p2", "p1", "p2"],
    })
    compute_protein_FDR(df, results_folder=str(tmp_path), target_decoy_ratio=1.0)
    summary = (tmp_path / "Summary.txt").read_text()
    # the precursors at Qvalue and at BestChannel_Qvalue (too few proteins for a protein FDR)
    assert summary.count("All Channels: 2") == 2
    rows = [line.split() for line in summary.splitlines() if line.split()[:1] == ["A"]]
    assert rows and {row[1] for row in rows} == {"0"}                         # no channel 8

        
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


def test_process_data_flags_and_filters_the_decoy_channel(tmp_path, monkeypatch):
    """The tag's empty channel is scored like any other, so it is in all_IDs with
    is_decoy_channel set, but it is never an identification: out of the filtered
    IDs, which is what the combined results and the MBR library are built from."""
    from src.mass_tags import available_tags

    mass_tag = available_tags["PSMtag_5plex"].select_channels(["0", "4"]).with_decoy_channel()
    assert mass_tag.decoy_channel == "8"
    seqs = [f"(PSMtag_5plex-{c})AAAA" for c in (0, 8)] + [f"(PSMtag_5plex-{c})BBBB" for c in (0, 8)]

    dummy_file = tmp_path / "input.mzML"
    dummy_file.write_text("dummy")
    (tmp_path.parent / "outputs").mkdir(exist_ok=True)

    monkeypatch.setattr("src.fdr_analysis.get_large_prec",
                        lambda *args, **kwargs: (None,
                                                 pd.DataFrame({
                                                     "seq": seqs,
                                                     "z": [2] * 4,
                                                     "rt": [10.0] * 4,
                                                     "mz": [500.0] * 4,
                                                     "rt_error": [0.1] * 4,
                                                     "mz_error": [5] * 4,
                                                     "gof_stats": [0.1] * 4,
                                                     "scribe_scores": [1] * 4,
                                                     "max_matched_residuals": [0.5] * 4,
                                                     "manhattan_distances": [5] * 4,
                                                     "frag_errors": [[1.0, 2.0]] * 4,
                                                     "frag_mz": [[200.0, 300.0]] * 4,
                                                     "frag_int": [[1.0, 2.0]] * 4,
                                                     "frac_lib_int": [1] * 4,
                                                     "file_name": ["f"] * 4,
                                                     "time_channel": [0] * 4,
                                                     "silac_channel": [np.nan] * 4,
                                                     "is_decoy": [False] * 4,
                                                     "stripped_seq": ["AAAA", "AAAA", "BBBB", "BBBB"],
                                                 }),
                                                 None))
    monkeypatch.setattr("src.fdr_analysis.score_precursors",
                        lambda fdc, *args, **kwargs: fdc.assign(PredVal=0.9, Qvalue=0.001))
    monkeypatch.setattr("src.fdr_analysis.ms1_quant",
                        lambda *args, **kwargs: args[0].assign(plex_Area=100.0))
    monkeypatch.setattr("src.fdr_analysis.compute_protein_FDR",
                        lambda df, results_folder=None, **kwargs: df.assign(Protein_Qvalue=1.0,
                                                                            protein="P12345"))
    monkeypatch.setattr("src.fdr_analysis.unstring_floats",
                        lambda x: x if isinstance(x, list) else [1.0, 2.0])

    process_data(str(dummy_file), None, None, ms1_tol=20e-6, rt_tol=1.0, im_tol=0.01,
                 target_decoy_ratio=1.0, mass_tag=mass_tag)

    all_ids = pd.read_csv(tmp_path.parent / "outputs/all_IDs.csv")
    assert dict(zip(all_ids["channel"], all_ids["is_decoy_channel"])) == {0: False, 8: True}
    filtered = pd.read_csv(tmp_path.parent / "filtered_IDs.csv")
    assert filtered["channel"].tolist() == [0, 0]
    assert "is_decoy_channel" in pd.read_parquet(tmp_path.parent / "outputs/all_IDs_filtered.parquet").columns
