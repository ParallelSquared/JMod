import numpy as np
import pandas as pd
import os
import types
import pytest

from src.fdr_analysis import score_precursors, score_model, add_median_based_features, process_data, compute_protein_FDR, log_df

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
        folder=str(tmp_path)  # ensures the plot-saving code runs
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
        "outputs/all_IDs.csv",
        "filtered_IDs.csv",
        "outputs/all_IDs_filtered.parquet",
    ]

    for fname in expected_files:
        assert (tmp_path.parent / fname).exists()
        
        
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

    out = compute_protein_FDR(df, results_folder=None)

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