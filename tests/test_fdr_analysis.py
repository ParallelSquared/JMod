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
                               compute_protein_FDR, id_counts, id_table_lines, report_id_counts, target_decoy_qvalues,
                               untag_prec_qvalues, fill_ms1_quant_gaps,
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


def test_target_decoy_qvalues_is_jmods_estimator():
    # ranked: T T D T D T -> (1 + decoys) / targets, then the minimum from below
    score = np.array([6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
    is_decoy = np.array([False, False, True, False, True, False])
    raw = np.array([1 / 1, 1 / 2, 2 / 2, 2 / 3, 3 / 3, 3 / 4])
    assert target_decoy_qvalues(score, is_decoy) == pytest.approx(np.minimum.accumulate(raw[::-1])[::-1])
    # by score: T(3) D(2) T(1)
    assert target_decoy_qvalues(np.array([1.0, 3.0, 2.0]), np.array([False, False, True]),
                                target_decoy_ratio=2.0) == pytest.approx([2.0, 2.0, 2.0])


def test_target_decoy_qvalues_of_a_subset():
    score = np.array([6.0, 5.0, 4.0, 3.0])
    is_decoy = np.array([False, True, False, False])
    rows = np.array([True, False, True, True])            # the decoy left out
    q = target_decoy_qvalues(score, is_decoy, rows=rows)
    # as the rows ranked alone: no decoy, so 1/1, 1/2, 1/3, the lowest from below
    assert np.isnan(q[1]) and q[rows] == pytest.approx([1 / 3] * 3)
    assert q[rows] == pytest.approx(target_decoy_qvalues(score[rows], is_decoy[rows]))


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
    report_id_counts(compute_protein_FDR(df, target_decoy_ratio=1.0), str(tmp_path))
    # the precursors at Qvalue and at BestChannel_Qvalue; too few proteins for a protein FDR
    assert (tmp_path / "Summary.txt").read_text().splitlines() == [
        "IDs at 1% FDR, A",
        "          Precursors            Proteins",
        "channel   own Q   best chan Q   own Q   best chan Q",
        "      0       2             2       0             0",
        "  total       2             2       0             0"]               # no channel 8

        
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

    out = compute_protein_FDR(df, target_decoy_ratio=1.0)

    assert isinstance(out, pd.DataFrame)
    assert "Protein_Qvalue" in out.columns  # main thing it produces
    assert out.shape[0] == 2  # still same number of rows


def _counted(file_name, channels, silac_channels=None):
    n = len(channels)
    df = pd.DataFrame({"file_name": file_name, "channel": channels,
                       "silac_channel": silac_channels if silac_channels is not None else [np.nan] * n,
                       "is_decoy": False, "Qvalue": 0.001, "BestChannel_Qvalue": 0.001,
                       "Protein_Qvalue": 0.001, "untag_prec": [f"p{i}" for i in range(n)],
                       "protein": [f"P{i}" for i in range(n)]})
    return df


def test_id_table_label_free_is_one_level():
    lines = id_table_lines(id_counts(_counted("C:/data/run.mzML", [0, 0, 0])))
    assert lines == ["IDs at 1% FDR, run.mzML",
                     "channel   Precursors   Proteins",
                     "      0            3          3",
                     "  total            3          3"]


def test_id_table_sorts_the_channels_as_numbers_and_splits_timeplex():
    table = id_counts(_counted("f", ["1_4", "0_12", "0_4", "0_4"]), plexDIA=True)
    lines = id_table_lines(table, timeplex=True)
    rows = [line.split() for line in lines[3:]]
    assert lines[2].split() == ["time", "channel", "own", "Q", "best", "chan", "Q", "own", "Q", "best", "chan", "Q"]
    assert rows == [["0", "4", "2", "2", "2", "2"], ["0", "12", "1", "1", "1", "1"],
                    ["1", "4", "1", "1", "1", "1"], ["total", "4", "4", "4", "4"]]


def test_id_counts_channel_fdr_between_qvalue_and_best_channel():
    # p0 passes everything; p1 only through its channel FDR and best channel;
    # p2 only best channel.  P0 is in the run and the channel, P1 in the run
    # but not the channel, P2 in the channel but not the run
    df = pd.DataFrame({"file_name": "f", "channel": 0, "silac_channel": np.nan, "is_decoy": False,
                       "untag_prec": ["p0", "p1", "p2"], "protein": ["P0", "P1", "P2"],
                       "Qvalue": [0.001, 0.5, 0.5], "Channel_Qvalue": [0.001, 0.001, 0.5],
                       "BestChannel_Qvalue": [0.001, 0.001, 0.001],
                       "Protein_Qvalue": [0.001, 0.001, 0.5], "Protein_Channel_Qvalue": [0.001, 0.5, 0.001]})
    table = id_counts(df, plexDIA=True)
    assert [label for _, label in table.columns] == ["own Q", "chan Q", "best chan Q"] * 2
    assert table.iloc[0].tolist() == [1, 2, 3, 1, 1, 2]
    # without a fitted channel FDR, no such column
    assert "chan Q" not in {label for _, label in id_counts(df.assign(Channel_Qvalue=np.nan)).columns}


def test_id_table_one_per_file_and_silac():
    df = pd.concat([_counted("a", [0, 0, 0], [0.0, 1.0, np.nan]), _counted("b", [4], [0.0])])
    lines = id_table_lines(id_counts(df), silac=True)
    assert [line for line in lines if line.startswith("IDs")] == [
        "IDs at 1% FDR, a", "IDs at 1% FDR, b"]
    assert lines[1].split() == ["channel", "SILAC", "Precursors", "Proteins"]
    assert [line.split()[:2] for line in lines[2:5]] == [["0", "0"], ["0", "1"], ["0", "-"]]   # no label: "-"
    assert id_table_lines(id_counts(df.iloc[:0])) == ["IDs at 1% FDR: none"]


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


def test_ms1_quant_fill_leaves_the_decoy_channels_qvalues_unscored():
    """A set found only in the decoy channel has no best channel: as 0 it would
    pass the run's FDR and be scored by the channel FDR and the protein FDR."""
    dat = pd.DataFrame({"is_decoy_channel": [False, True, True],
                        "Qvalue": [0.001, np.nan, np.nan],
                        "BestChannel_Qvalue": [0.001, 0.001, np.nan],      # the last: decoy channel alone
                        "untag_prec_Qvalue": [0.001, 0.001, np.nan],
                        "plex_Area": [5.0, np.nan, np.nan], "ms1_cor": [0.9, np.nan, 0.2]})
    out = fill_ms1_quant_gaps(dat)
    assert out["Qvalue"].isna().tolist() == [False, True, True]
    assert out["BestChannel_Qvalue"].isna().tolist() == [False, False, True]
    assert out["untag_prec_Qvalue"].isna().tolist() == [False, False, True]
    assert out["plex_Area"].tolist() == [5.0, 0.0, 0.0]                   # the rest as before
    assert out["ms1_cor"].tolist() == [0.9, 0.0, 0.2]
    assert not (out["BestChannel_Qvalue"] < 0.01).iloc[2]


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
