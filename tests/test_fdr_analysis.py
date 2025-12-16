import numpy as np
import pandas as pd
import os
import types
import pytest

from src.fdr_analysis import score_precursors, score_model, add_median_based_features, process_data, compute_protein_FDR

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
    try:
        sm = score_model("unsupported", n_splits=3)
        sm.run_model(X, y)
    except ValueError:
        pass

def test_score_precursors_minimal_fixed(tmp_path):
    # Minimal valid input
    fdc = pd.DataFrame({
        "coeff": [2.0, 0.5, 2.1, 0.7, 1.5],
        "decoy": [False, True, False, True, False],
        "stripped_seq": ["AAA", "BBB", "CCC", "DDD", "EEE"],
        "rt_error": [0.1, 0.2, 0.3, 0.4, 0.5],
        "mz_error": [0.01, 0.02, 0.03, 0.04, 0.05],
        "feature1": [1.0, 2.0, 3.0, 4.0, 5.0],
        "feature2": [5.0, 4.0, 3.0, 2.0, 1.0],
    })

    out = score_precursors(
        fdc,
        model_type="lda",
        fdr_t=0.01,
        folder=str(tmp_path)  # ensures the plot-saving code runs
    )

    assert isinstance(out, pd.DataFrame)
    assert len(out) == len(fdc)

    for col in ["PredVal", "Qvalue"]:
        assert col in out.columns

    expected_files = [
        "ModelScore.png",
        "RT_error.png",
        "mz_error.png"
    ]
    for fname in expected_files:
        f = tmp_path / fname
        assert f.exists()


def test_add_median_based_features_basic():
    # minimal dummy dataframe
    df = pd.DataFrame({
        "untag_prec": ["A", "A", "B"],
        "channels_matched": [2, 2, 1],
        "m1": [10.0, 20.0, 30.0],
    })

    metric_columns = ["m1"]

    out = add_median_based_features(df, metric_columns, verbose=False)

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

    spectra = None
    library = None

    monkeypatch.setattr("src.fdr_analysis.get_large_prec",
                        lambda *args, **kwargs: (None,
                                                 pd.DataFrame({
                                                     "seq": ["AAAA", "BBBB"],
                                                     "z": [2, 2],
                                                     "rt_error": [0.1, -0.2],
                                                     "mz_error": [5, -10],
                                                     "gof_stats": [0.1, 0.2],
                                                     "scribe_scores": [1, 2],
                                                     "max_matched_residuals": [0.5, 0.8],
                                                     "manhattan_distances": [5, 6],
                                                     "frag_errors": [[1.0, 2.0], [2.0, 3.0]],
                                                     "frac_lib_int": [1, 1],
                                                     "file_name": ["f", "f"],
                                                     "time_channel": [0, 0]
                                                 }),
                                                 None))

    monkeypatch.setattr("src.fdr_analysis.score_precursors",
                        lambda *args, **kwargs: pd.DataFrame({
                            "untag_prec": ["AAAA_2", "BBBB_2"],
                            "channel": [0, 0],
                            "PredVal": [0.5, 0.7],
                            "Qvalue": [0.01, 0.02],
                        }))

    monkeypatch.setattr("src.fdr_analysis.ms1_quant",
                        lambda *args, **kwargs: args[0])

    monkeypatch.setattr("src.fdr_analysis.compute_protein_FDR",
                        lambda df, results_folder=None: df.assign(
                            Protein_Qvalue=1.0,
                            protein="P12345"
                        ))

    monkeypatch.setattr(
        "src.fdr_analysis.unstring_floats",
        lambda x: x if isinstance(x, list) else [1.0, 2.0]
    )

    process_data(str(dummy_file), spectra, library)

    expected_files = [
        "all_IDs.csv",
        "filtered_IDs.csv",
        "all_IDs_filtered.parquet",
    ]

    for fname in expected_files:
        assert (tmp_path / fname).exists()
        
        
def test_compute_protein_FDR_minimal(monkeypatch, tmp_path):
    fake_args = types.SimpleNamespace(plexDIA=False)

    df = pd.DataFrame({
        "file_name": ["A", "A"],
        "channel": [1, 1],
        "Qvalue": [0.005, 0.005],
        "PredVal": [0.9, 0.8],
        "decoy": [False, True],
        "protein": ["P1", "DECOY_P1"],
        "untag_prec": ["p1", "p2"],
    })

    out = compute_protein_FDR(df, results_folder=None)

    assert isinstance(out, pd.DataFrame)
    assert "Protein_Qvalue" in out.columns  # main thing it produces
    assert out.shape[0] == 2  # still same number of rows