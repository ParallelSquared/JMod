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


def test_wtd_frag_len_weights_by_ms2_signal():
    """sum((b/x)*a): mean fragment length weighted by observed MS2 signal."""
    import polars as pl
    from src.utils.frag_encoding import encode_frag_names
    from src.fdr_analysis import (_compute_wtd_frag_len,
                                  _compute_wtd_frag_len_from_polars)

    def codes(*names):
        return list(map(int, encode_frag_names(list(names))))

    frag_names = [
        codes("y3_1", "y7_1", "b2_1"),
        codes("y10-H2O_2", "b4_1_iso1"),  # loss/charge/iso bits must not leak in
        [],                               # nothing matched
        codes("y5_1", "y6_1"),            # matched but no signal
    ]
    obs_int = [[100.0, 300.0, 100.0], [50.0, 50.0], [], [0.0, 0.0]]
    expected = [(100 * 3 + 300 * 7 + 100 * 2) / 500.0,
                (50 * 10 + 50 * 4) / 100.0,
                np.nan,
                np.nan]

    assert np.allclose(_compute_wtd_frag_len(frag_names, obs_int), expected,
                       equal_nan=True)

    # the polars path, plus a null-list row (the parquet placeholder default)
    lf = pl.DataFrame(
        {"__fdc_idx": np.arange(len(frag_names) + 1, dtype=np.int64),
         "frag_names": frag_names + [None],
         "obs_int": obs_int + [None]},
        schema={"__fdc_idx": pl.Int64,
                "frag_names": pl.List(pl.Int32),
                "obs_int": pl.List(pl.Float64)},
    )
    got = _compute_wtd_frag_len_from_polars(lf)["wtd_frag_len"].to_numpy()
    assert np.allclose(got, expected + [np.nan], equal_nan=True)


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


def test_plex_set_key_scopes_to_time_channel():
    """The plexDIA-set key separates time channels but keeps mTRAQ channels together."""
    from src.fdr_analysis import _plex_set_key

    # same precursor, 2 mTRAQ channels x 2 time channels
    fdc = pd.DataFrame({
        "untag_prec": ["PEPTIDE_2"] * 4,
        "time_channel": np.float32([0, 0, 1, 1]),
    })

    key = _plex_set_key(fdc, timeplex=True)
    # the two rows in time channel 0 are one set, the two in channel 1 another
    assert key.iloc[0] == key.iloc[1]
    assert key.iloc[2] == key.iloc[3]
    assert key.iloc[0] != key.iloc[2]

    # off timeplex there is a single acquisition -- key is untag_prec unchanged
    flat = _plex_set_key(fdc, timeplex=False)
    assert flat.nunique() == 1
    assert (flat == "PEPTIDE_2").all()

    # a null time_channel forms its own group rather than being dropped
    fdc_nan = pd.DataFrame({
        "untag_prec": ["A_2", "A_2"],
        "time_channel": np.float32([0, np.nan]),
    })
    key_nan = _plex_set_key(fdc_nan, timeplex=True)
    assert key_nan.iloc[0] != key_nan.iloc[1]
    assert key_nan.notna().all()


def test_median_features_do_not_pool_across_time_channels():
    """diff_*_from_median compares within a plexDIA set, not across acquisitions."""
    from src.fdr_analysis import _plex_set_key, add_median_based_features, PLEX_SET_COL

    # One precursor, 2 mTRAQ channels x 2 time channels. Within each time channel
    # the two channels straddle that channel's own median; the two time channels
    # sit at very different absolute levels (a run-to-run offset).
    fdc = pd.DataFrame({
        "untag_prec": ["A_2"] * 4,
        "time_channel": np.float32([0, 0, 1, 1]),
        "m1": [10.0, 20.0, 110.0, 120.0],
    })
    fdc[PLEX_SET_COL] = _plex_set_key(fdc, timeplex=True)
    fdc["channels_matched"] = fdc.groupby(PLEX_SET_COL)[PLEX_SET_COL].transform("size")

    out = add_median_based_features(fdc, ["m1"], group_col=PLEX_SET_COL, verbose=False)

    # each time channel has its own median (15 and 115), so the run offset cancels
    assert list(out["median_m1"]) == [15.0, 15.0, 115.0, 115.0]
    assert list(out["diff_m1_from_median"]) == [-5.0, 5.0, -5.0, 5.0]

    # pooling across time channels would instead use one global median of 65
    pooled = add_median_based_features(fdc, ["m1"], group_col="untag_prec", verbose=False)
    assert list(pooled["median_m1"]) == [65.0, 65.0, 65.0, 65.0]


def test_frac_shared_intensity_scoped_to_plex_set():
    """A precursor is compared only against channels it was co-isolated with."""
    import polars as pl
    from src.fdr_analysis import (_compute_frac_shared_intensity,
                                  _compute_frac_shared_intensity_from_polars,
                                  PLEX_SET_COL)

    # Time channel 0: the two rows share no fragment m/z -> frac 0.
    # Time channel 1: the two rows share every fragment m/z -> frac 1.
    # Pooling the two time channels would let channel 0's rows find their
    # own m/z values again in channel 1 and report spurious sharing.
    fdc = pd.DataFrame({
        "__fdc_idx": np.arange(4, dtype=np.int64),
        "untag_prec": ["A_2"] * 4,
        "time_channel": np.float32([0, 0, 1, 1]),
        "frag_mz": [[100.0, 200.0], [300.0, 400.0],
                    [100.0, 200.0], [100.0, 200.0]],
        "frag_int": [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
    })
    fdc[PLEX_SET_COL] = ["A_2|tc0.0", "A_2|tc0.0", "A_2|tc1.0", "A_2|tc1.0"]

    got = _compute_frac_shared_intensity(fdc, mz_tol=20e-6, group_col=PLEX_SET_COL)
    assert list(got) == [0.0, 0.0, 1.0, 1.0]

    # pooling across time channels inflates the two channel-0 rows
    pooled = _compute_frac_shared_intensity(fdc, mz_tol=20e-6, group_col="untag_prec")
    assert pooled[0] == 1.0

    # the polars path must agree with the pandas path
    lf = pl.DataFrame(
        {"__fdc_idx": fdc["__fdc_idx"].to_numpy(),
         "frag_mz": fdc["frag_mz"].tolist(),
         "frag_int": fdc["frag_int"].tolist()},
        schema={"__fdc_idx": pl.Int64,
                "frag_mz": pl.List(pl.Float64),
                "frag_int": pl.List(pl.Float64)},
    )
    got_pl = _compute_frac_shared_intensity_from_polars(
        fdc, lf, mz_tol=20e-6, group_col=PLEX_SET_COL)
    assert list(got_pl) == [0.0, 0.0, 1.0, 1.0]


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