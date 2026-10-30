import numpy as np
import polars as pl
import pytest

from src.rt_normalization import (_collapse, _with_log2, apply_correction, lowess_rt_correction,
                                  paired_rt_correction, rt_normalize)

FRAC = 0.3
RT_BINS = np.linspace(0, 100, 11)


def _runs(n=5000, gain=None, keep=None, seed=0):
    """Two runs of the same *n* precursors at the same RTs (0-100), each
    precursor with an abundance of mean 20 and SD 2 (log2), plus noise.

    *gain(rt)* is added to run 1's log2 quant; *keep(rt, abundance)* says
    which of run 1's precursors are identified (all of run 0's are)."""
    rng = np.random.default_rng(seed)
    rt = rng.uniform(0, 100, n)
    abundance = rng.normal(20, 2, n)
    frames = []
    for run in (0, 1):
        y = abundance + rng.normal(0, 0.2, n)
        kept = np.ones(n, bool)
        if run == 1:
            y = y + (gain(rt) if gain else 0)
            kept = keep(rt, abundance) if keep else kept
        frames.append(pl.DataFrame({
            "run": run, "prec": np.arange(n)[kept], "rt": rt[kept], "quant": np.exp2(y[kept]),
        }))
    return pl.concat(frames, how="vertical_relaxed")


def _binned_medians(rt, values):
    which = np.digitize(rt, RT_BINS[1:-1])
    return np.array([np.median(values[which == b]) for b in range(len(RT_BINS) - 1)])


def _ratio_by_rt(df, corr):
    """Binned medians vs RT of each shared precursor's corrected log2 ratio, run 1 over run 0."""
    corrected = df.with_columns(pl.Series("y", np.log2(df["quant"].to_numpy()) - corr.to_numpy()))
    wide = (corrected.pivot(on="run", index="prec", values="y")
            .join(corrected.filter(pl.col("run") == 0).select("prec", "rt"), on="prec")
            .drop_nulls())
    return _binned_medians(wide["rt"].to_numpy(), (wide["1"] - wide["0"]).to_numpy())


def test_sample_stage_follows_drift_and_keeps_median():
    df = _runs(gain=lambda rt: np.sin(rt / 100 * np.pi)).filter(pl.col("run") == 1)
    rt = df["rt"].to_numpy()
    corr = lowess_rt_correction(df, "quant", ["run"], "rt", frac=FRAC).to_numpy()
    y = np.log2(df["quant"].to_numpy())
    # The correction is the drift (a swing of 1) up to a constant, less fit noise
    assert np.ptp(_binned_medians(rt, corr - np.sin(rt / 100 * np.pi))) < 0.3
    assert abs(np.median(y - corr) - np.median(y)) < 0.1


def test_run_stage_removes_rt_gain_and_keeps_levels():
    df = _runs(gain=lambda rt: 0.3 + (rt - 50) / 50)
    raw = _ratio_by_rt(df, pl.Series(np.zeros(df.height)))
    assert np.ptp(raw) > 1.5
    corr = paired_rt_correction(df, "quant", ["run"], ["prec"], "rt", frac=FRAC)
    ratio = _ratio_by_rt(df, corr)
    assert np.ptp(ratio) < 0.15
    # The level difference between the runs stays, and each run's curve is centred
    assert abs(np.median(ratio) - np.median(raw)) < 0.05
    for run in (0, 1):
        assert abs(np.median(corr.filter(df["run"] == run))) < 0.05


def test_selection_bias_trend_is_undone_by_run_stage():
    # Run 1 misses the low-abundance precursors after RT 50; the true ratio is 0
    # everywhere.  The sample stage turns that into a false trend of ratio with
    # RT, which the run stage flattens; the level it leaves is not the run
    # stage's to change
    df = _runs(keep=lambda rt, abundance: (rt < 50) | (abundance > 20))
    sample_only = rt_normalize(df, "quant", ("sample",), "rt", sample_frac=FRAC, sample_cols=["run"])
    false_trend = np.ptp(_ratio_by_rt(df, sample_only))
    assert false_trend > 1
    both = rt_normalize(df, "quant", ("sample", "run"), "rt", sample_frac=FRAC, run_frac=FRAC,
                        sample_cols=["run"], run_cols=["run"], pair_cols=["prec"])
    # Not flat to fit noise like a smooth trend: LOWESS rounds off the step at RT 50
    assert np.ptp(_ratio_by_rt(df, both)) < 0.2 * false_trend


def test_collapse_takes_median_of_log2():
    df = pl.DataFrame({"run": [0, 0, 0], "prec": [1, 1, 1], "rt": [1.0, 2.0, 30.0], "quant": [1.0, 2.0, 100.0]})
    collapsed = _collapse(_with_log2(df, "quant", "rt"), ["run"], ["prec"])
    assert collapsed.height == 1
    assert collapsed["__log2_quant"][0] == pytest.approx(1.0)
    assert collapsed["__rt"][0] == pytest.approx(2.0)


def test_steps_compose_in_either_order_each_with_its_own_frac():
    df = _runs(gain=lambda rt: (rt - 50) / 50)
    kwargs = dict(sample_frac=FRAC, run_frac=0.1, sample_cols=["run"], run_cols=["run"], pair_cols=["prec"])
    for steps in (("sample", "run"), ("run", "sample")):
        corr = rt_normalize(df, "quant", steps, "rt", **kwargs)
        assert corr.len() == df.height and corr.is_finite().all()
    assert rt_normalize(df, "quant", ("sample",), "rt", **kwargs).equals(
        lowess_rt_correction(df, "quant", ["run"], "rt", frac=FRAC), check_names=False)
    assert rt_normalize(df, "quant", ("run",), "rt", **kwargs).equals(
        paired_rt_correction(df, "quant", ["run"], ["prec"], "rt", frac=0.1), check_names=False)


def test_step_without_its_columns_raises():
    df = _runs()
    with pytest.raises(ValueError):
        rt_normalize(df, "quant", ("sample",), "rt", run_cols=["run"], pair_cols=["prec"])
    with pytest.raises(ValueError):
        rt_normalize(df, "quant", ("run",), "rt", run_cols=["run"])


def test_curve_held_flat_beyond_fitted_rt_and_unfitted_rows_corrected():
    df = _runs(gain=lambda rt: 2 * (rt - 50) / 50).filter(pl.col("run") == 1, pl.col("rt").is_between(10, 90))
    # Rows outside the fitted RTs whose quant can't be fitted (NaN, 0)
    df = pl.concat([df, pl.DataFrame({"run": [1, 1], "prec": [-1, -2], "rt": [0.0, 100.0],
                                      "quant": [float("nan"), 0.0]})], how="vertical_relaxed")
    corr = lowess_rt_correction(df, "quant", ["run"], "rt", frac=FRAC).to_numpy()
    rt = df["rt"].to_numpy()
    fitted = rt[:-2]
    assert corr[-2] == pytest.approx(corr[np.argmin(fitted)])
    assert corr[-1] == pytest.approx(corr[np.argmax(fitted)])
    assert corr[-2] < -1 and corr[-1] > 1


def test_small_group_left_uncorrected_with_warning():
    df = _runs(gain=lambda rt: (rt - 50) / 50).filter(pl.col("run") == 1)
    small = pl.DataFrame({"run": [2] * 10, "prec": range(10), "rt": np.linspace(0, 100, 10),
                          "quant": np.exp2(np.linspace(18, 22, 10))})
    df = pl.concat([df, small], how="vertical_relaxed")
    with pytest.warns(UserWarning, match="fewer than 50"):
        corr = lowess_rt_correction(df, "quant", ["run"], "rt", frac=FRAC)
    assert (corr.filter(df["run"] == 2) == 0).all()
    assert (corr.filter(df["run"] == 1) != 0).any()


def test_correction_follows_row_order_and_leaves_df_unchanged():
    df = _runs(gain=lambda rt: (rt - 50) / 50)
    before = df.clone()
    kwargs = dict(sample_frac=FRAC, run_frac=FRAC, sample_cols=["run"], run_cols=["run"], pair_cols=["prec"])
    corr = rt_normalize(df, "quant", ("sample", "run"), "rt", **kwargs)
    assert df.equals(before)
    order = np.random.default_rng(1).permutation(df.height)
    shuffled = rt_normalize(df[order], "quant", ("sample", "run"), "rt", **kwargs)
    np.testing.assert_allclose(shuffled.to_numpy(), corr.to_numpy()[order])


def test_apply_correction():
    np.testing.assert_allclose(apply_correction(np.array([8.0, 8.0]), np.array([1.0, -1.0])), [4.0, 16.0])
