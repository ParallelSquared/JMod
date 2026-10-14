import os

import numpy as np
import pandas as pd
import pytest

from src.quant_normalization import (RTNormConfig, apply_rt_channel_normalization,
                                     ms1_tic_curve)


def _synthetic(n_per_channel=3000, channels=(0.0, 1.0), drift=None, seed=0):
    """Precursors with a known per-channel RT-dependent gain applied to a
    common ground-truth abundance, so the correction has something to recover."""
    rng = np.random.default_rng(seed)
    drift = drift or {0.0: (lambda rt: 0.0 * rt),
                      1.0: (lambda rt: 1.5 * np.sin(rt / 10.0))}
    # One ground-truth abundance and RT per precursor, shared by every channel,
    # so a channel/channel ratio is 1 apart from the drift and the noise.
    rt = np.sort(rng.uniform(5, 60, n_per_channel))
    true_log2 = rng.normal(17, 2.0, n_per_channel)
    rows = []
    for ch in channels:
        obs = np.exp2(true_log2 + drift[ch](rt) + rng.normal(0, 0.15, n_per_channel))
        rows.append(pd.DataFrame({
            "untag_prec": [f"P{i}_2" for i in range(n_per_channel)],
            "time_channel": ch,
            "rt": rt,
            "plex_Area": obs,
            "coeff": obs / 10.0,
            # MS2_PRIMARY is coeff_ChannelUnique, and a level whose primary
            # column is absent is skipped entirely -- without this the MS2 half
            # of the normalizer never runs and no MS2 *_rtnorm column appears.
            "coeff_ChannelUnique": obs / 10.0,
            "Qvalue": 0.001,
            "is_decoy": False,
            "_true": np.exp2(true_log2),
        }))
    return pd.concat(rows, ignore_index=True)


def test_removes_known_per_channel_drift():
    df = _synthetic()
    out = apply_rt_channel_normalization(df.copy(), timeplex=True,
                                         cfg=RTNormConfig(window=200))
    assert "plex_Area_rtnorm" in out.columns
    assert "coeff_rtnorm" in out.columns

    def channel_ratio_spread(col):
        # Ratio of channel 1 to channel 0 for the same precursor: flat if the
        # per-channel drift was removed.
        w = out.pivot_table(index="untag_prec", columns="time_channel", values=col)
        r = np.log2(w[1.0] / w[0.0]).dropna()
        return np.std(r)

    assert channel_ratio_spread("plex_Area_rtnorm") < 0.5 * channel_ratio_spread("plex_Area")
    assert channel_ratio_spread("coeff_rtnorm") < 0.5 * channel_ratio_spread("coeff")


def test_factors_are_multiplicative_and_shared_within_channel():
    df = _synthetic(n_per_channel=800)
    out = apply_rt_channel_normalization(df.copy(), timeplex=True)
    assert np.allclose(out["plex_Area_rtnorm"],
                       out["plex_Area"] * out["rtnorm_factor_MS1"])
    assert np.allclose(out["coeff_rtnorm"], out["coeff"] * out["rtnorm_factor_MS2"])
    assert (out["rtnorm_factor_MS1"] > 0).all()


def test_plexdia_label_ratios_are_untouched():
    """The factor is per time channel, so ratios between label channels of one
    time channel must survive the correction exactly."""
    base = _synthetic(n_per_channel=1500, channels=(0.0,))
    labels = []
    for lab, mult in [("0", 1.0), ("4", 3.0), ("8", 0.4)]:
        d = base.copy()
        d["channel"] = f"0_{lab}"
        d["plex_Area"] = d["plex_Area"] * mult
        d["coeff"] = d["coeff"] * mult
        labels.append(d)
    df = pd.concat(labels, ignore_index=True)

    out = apply_rt_channel_normalization(df.copy(), timeplex=True, plexDIA=True)
    w = out.pivot_table(index="untag_prec", columns="channel",
                        values="plex_Area_rtnorm")
    assert np.allclose(w["0_4"] / w["0_0"], 3.0)
    assert np.allclose(w["0_8"] / w["0_0"], 0.4)


def test_single_channel_data_is_still_corrected():
    df = _synthetic(n_per_channel=3000, channels=(0.0,),
                    drift={0.0: (lambda rt: 0.04 * (rt - 30))})
    out = apply_rt_channel_normalization(df.copy(), timeplex=False,
                                         cfg=RTNormConfig(window=200))
    # The running median of log2 quant should be much flatter afterwards.
    def trend_spread(col):
        s = out.sort_values("rt")
        return np.nanstd(pd.Series(np.log2(s[col].to_numpy()))
                         .rolling(300, center=True, min_periods=100).median())

    assert trend_spread("plex_Area_rtnorm") < 0.5 * trend_spread("plex_Area")


def test_disabled_and_missing_columns_are_no_ops():
    df = _synthetic(n_per_channel=200)
    out = apply_rt_channel_normalization(df.copy(), cfg=RTNormConfig(method="none"))
    assert "plex_Area_rtnorm" not in out.columns

    no_rt = df.drop(columns=["rt"])
    assert "plex_Area_rtnorm" not in apply_rt_channel_normalization(no_rt).columns


def test_too_few_reference_precursors_leaves_data_alone():
    df = _synthetic(n_per_channel=10)
    out = apply_rt_channel_normalization(df.copy(), timeplex=True,
                                         cfg=RTNormConfig(min_ref=50))
    assert np.allclose(out["rtnorm_factor_MS1"], 1.0)


def test_inplace_keeps_raw_copy():
    df = _synthetic(n_per_channel=800)
    raw = df["plex_Area"].to_numpy().copy()
    out = apply_rt_channel_normalization(df.copy(), timeplex=True, inplace_cols=True)
    assert np.allclose(out["plex_Area_raw"], raw)
    assert not np.allclose(out["plex_Area"], raw)


def test_decoys_are_corrected_but_do_not_set_the_curve():
    df = _synthetic(n_per_channel=2000)
    df.loc[df.index[::2], "is_decoy"] = True
    # Blow the decoy quantities up; the curve must not follow them.
    df.loc[df["is_decoy"], "plex_Area"] *= 1000.0
    out = apply_rt_channel_normalization(df.copy(), timeplex=True)
    assert (out["rtnorm_factor_MS1"] > 0).all()
    # Same RT and channel => same factor, decoy or not.
    a = out[out["is_decoy"]].sort_values("rt")
    b = out[~out["is_decoy"]].sort_values("rt")
    assert np.isclose(np.median(a["rtnorm_factor_MS1"]),
                      np.median(b["rtnorm_factor_MS1"]), rtol=0.2)


@pytest.mark.parametrize("method", ["rolling_median", "rolling_mean",
                                    "rt_window", "loess"])
def test_all_smoothers_run(method):
    df = _synthetic(n_per_channel=1200)
    out = apply_rt_channel_normalization(
        df.copy(), timeplex=True,
        cfg=RTNormConfig(method=method, window=200, rt_window=2.0, loess_frac=0.2))
    assert np.isfinite(out["plex_Area_rtnorm"]).all()


def test_tic_method_uses_ms1_scans():
    class _S:
        def __init__(self, rt, tic):
            self.RT, self.TIC = rt, tic

    class _Spectra:
        ms1scans = [_S(rt, 1e6 * (2 + np.sin(rt / 10.0)))
                    for rt in np.linspace(0, 70, 2000)]

    spectra = _Spectra()
    assert ms1_tic_curve(spectra) is not None
    df = _synthetic(n_per_channel=1500)
    out = apply_rt_channel_normalization(
        df.copy(), timeplex=True, spectra=spectra,
        cfg=RTNormConfig(method="tic_rolling", window=200))
    assert np.isfinite(out["plex_Area_rtnorm"]).all()


def test_center_scope_global_equalizes_channels():
    """``center_scope='global'`` also removes the between-channel level
    difference; the default 'channel' scope must leave it intact."""
    df = _synthetic(n_per_channel=2000)
    df.loc[df["time_channel"] == 1.0, "plex_Area"] *= 4.0
    df.loc[df["time_channel"] == 1.0, "coeff"] *= 4.0

    def level_gap(out, col):
        m = out.groupby("time_channel")[col].apply(lambda v: np.median(np.log2(v)))
        return abs(m.loc[1.0] - m.loc[0.0])

    kept = apply_rt_channel_normalization(df.copy(), timeplex=True)
    equalized = apply_rt_channel_normalization(
        df.copy(), timeplex=True, cfg=RTNormConfig(center_scope="global"))
    assert level_gap(kept, "plex_Area_rtnorm") > 1.5
    assert level_gap(equalized, "plex_Area_rtnorm") < 0.2


def test_paired_stage_is_immune_to_rt_dependent_detection_bias():
    """The failure mode this design exists to avoid.

    Identification gets harder late in the gradient, so the precursors that
    still clear the q-value gate there are the brighter ones. That makes the
    *detected population* drift upward with RT even when the instrument is
    perfectly stable. An unpaired running-centre estimator reads that as drift
    and divides it out; the paired estimator, which compares each precursor to
    itself, does not.
    """
    rng = np.random.default_rng(3)
    n = 6000
    rt = np.sort(rng.uniform(5, 60, n))
    true_log2 = rng.normal(17, 2.0, n)
    rows = []
    for ch in (0.0, 1.0):
        obs = np.exp2(true_log2 + rng.normal(0, 0.1, n))   # no real drift at all
        # Detection threshold rises linearly with RT: 0 % lost at the start,
        # ~50 % lost at the end, and the survivors are the bright ones.
        thresh = np.quantile(true_log2, 0.0) + (true_log2.std() * (rt - 5) / 55) * 1.6
        qv = np.where(true_log2 > thresh, 0.001, 0.5)
        rows.append(pd.DataFrame({
            "untag_prec": [f"P{i}_2" for i in range(n)],
            "time_channel": ch, "rt": rt, "plex_Area": obs, "coeff": obs / 10.0,
            "Qvalue": qv, "is_decoy": False,
        }))
    df = pd.concat(rows, ignore_index=True)

    def worst_shift(cfg):
        out = apply_rt_channel_normalization(df.copy(), timeplex=True, cfg=cfg)
        return np.max(np.abs(np.log2(out["rtnorm_factor_MS1"].to_numpy())))

    unpaired = worst_shift(RTNormConfig(two_stage=False, stage_order="paired_first",
                                        run_stage="always"))
    paired = worst_shift(RTNormConfig(two_stage=True, stage_order="paired_first",
                                      run_stage="never"))
    assert unpaired > 0.3           # invents a large trend that is not there
    assert paired < 0.05            # sees the truth: nothing to correct
    assert paired < 0.2 * unpaired


def test_paired_stage_preserves_channel_levels():
    """Stage 1 corrects shape only; a pure level difference must survive."""
    df = _synthetic(n_per_channel=3000)
    df.loc[df["time_channel"] == 1.0, "plex_Area"] *= 4.0
    df.loc[df["time_channel"] == 1.0, "coeff"] *= 4.0
    out = apply_rt_channel_normalization(df.copy(), timeplex=True)
    m = out.groupby("time_channel")["plex_Area_rtnorm"].apply(
        lambda v: np.median(np.log2(v)))
    assert abs((m.loc[1.0] - m.loc[0.0]) - 2.0) < 0.25


def test_default_order_leaves_channel_ratios_undistorted():
    """The shipped order is unpaired-per-channel then paired.

    The unpaired stage does chase the detection-selection artefact — it has no
    way not to — but the paired stage runs afterwards and removes whatever of
    that leaked into the *between-channel* comparison, which is the part that
    corrupts a ratio. Here the two channels are given selection thresholds of
    different steepness, so per-channel flattening alone distorts their ratio.
    """
    rng = np.random.default_rng(11)
    n = 6000
    rt = np.sort(rng.uniform(5, 60, n))
    true_log2 = rng.normal(17, 2.0, n)
    rows = []
    for ch, steep in [(0.0, 0.8), (1.0, 2.2)]:
        obs = np.exp2(true_log2 + rng.normal(0, 0.1, n))     # no real drift
        thresh = true_log2.min() + true_log2.std() * steep * (rt - 5) / 55
        rows.append(pd.DataFrame({
            "untag_prec": [f"P{i}_2" for i in range(n)],
            "time_channel": ch, "rt": rt, "plex_Area": obs, "coeff": obs / 10.0,
            "Qvalue": np.where(true_log2 > thresh, 0.001, 0.5), "is_decoy": False,
        }))
    df = pd.concat(rows, ignore_index=True)

    def ratio_drift(cfg):
        out = apply_rt_channel_normalization(df.copy(), timeplex=True, cfg=cfg)
        w = out.pivot_table(index="untag_prec", columns="time_channel",
                            values="plex_Area_rtnorm")
        r = np.log2(w[1.0] / w[0.0])
        rt_p = out.groupby("untag_prec")["rt"].first().reindex(w.index)
        # spread of the ratio across RT: 0 if the two channels were corrected
        # consistently, large if per-channel flattening pulled them apart
        b = np.digitize(rt_p, np.arange(5, 60, 5))
        meds = [np.median(r[b == k]) for k in np.unique(b) if (b == k).sum() > 100]
        return np.std(meds)

    unpaired_only = ratio_drift(RTNormConfig(two_stage=False,
                                             stage_order="paired_first",
                                             run_stage="always"))
    shipped = ratio_drift(RTNormConfig())        # unpaired then paired
    assert unpaired_only > 0.1
    assert shipped < 0.4 * unpaired_only


def test_run_stage_auto_skips_when_paired_and_fires_when_not():
    multi = _synthetic(n_per_channel=2500)
    single = _synthetic(n_per_channel=2500, channels=(0.0,),
                        drift={0.0: (lambda rt: 0.04 * (rt - 30))})

    def spread(out):
        return np.std(np.log2(out["rtnorm_factor_MS1"].to_numpy()))

    # Multi-channel: the paired stage handles it, the run stage stays out.
    a = apply_rt_channel_normalization(
        multi.copy(), timeplex=True,
        cfg=RTNormConfig(stage_order="paired_first", run_stage="auto"))
    b = apply_rt_channel_normalization(
        multi.copy(), timeplex=True,
        cfg=RTNormConfig(stage_order="paired_first", run_stage="always"))
    assert spread(b) > spread(a)

    # Single channel: nothing to pair against, so the unpaired stage must fire
    # under either ordering.
    for order in ("paired_first", "unpaired_first"):
        c = apply_rt_channel_normalization(
            single.copy(), timeplex=False,
            cfg=RTNormConfig(stage_order=order, run_stage="auto"))
        assert spread(c) > 0.1


def test_intensity_axis_shrinks_real_fold_changes():
    """Guard rail for a correction that looks great and is wrong.

    Smoothing the paired residual against *this channel's own* quantity
    regresses every precursor toward its across-channel mean, because the
    residual is that quantity minus a mean of which it is a term. Every spread
    metric improves — you are literally shrinking the data — while the real fold
    changes are eaten. On the benchmark, with its bimodal human/yeast design,
    it takes ratio compression from 0.75 to 0.54; this synthetic is cleaner and
    so shrinks less, which is why the assertions here pin the *direction and
    ordering* rather than a magnitude. Only detectable against ground truth.
    """
    rng = np.random.default_rng(5)
    n = 6000
    rt = np.sort(rng.uniform(5, 60, n))
    base = rng.normal(17, 2.0, n)
    # Each precursor has its OWN true fold change between the channels. A
    # uniform offset would survive any of these corrections (they are centred on
    # their own median); it is the precursor-to-precursor spread of real fold
    # changes that regression to the mean destroys.
    # Mirror the benchmark's design: a constant background plus a spiked-in
    # minority at a known fold change. Compression is the shrinkage of the gap
    # between the two groups.
    spiked = rng.random(n) < 0.22
    truth_gap = 2.0
    delta = np.where(spiked, truth_gap, 0.0) + rng.normal(0, 0.1, n)
    rows = []
    for ch, d in [(0.0, np.zeros(n)), (1.0, delta)]:
        obs = np.exp2(base + d + rng.normal(0, 0.2, n))
        rows.append(pd.DataFrame({
            "untag_prec": [f"P{i}_2" for i in range(n)],
            "time_channel": ch, "rt": rt, "plex_Area": obs, "coeff": obs / 10.0,
            "Qvalue": 0.001, "is_decoy": False,
        }))
    df = pd.concat(rows, ignore_index=True)

    prec_spiked = pd.Series(spiked, index=[f"P{i}_2" for i in range(n)])

    def recovered_gap(cfg):
        out = apply_rt_channel_normalization(df.copy(), timeplex=True, cfg=cfg)
        w = out.pivot_table(index="untag_prec", columns="time_channel",
                            values="plex_Area_rtnorm")
        r = np.log2(w[1.0] / w[0.0])
        m = prec_spiked.reindex(r.index).to_numpy()
        return float(np.median(r[m]) - np.median(r[~m]))

    default = recovered_gap(RTNormConfig())
    own_mean = recovered_gap(RTNormConfig(intensity_stage=True,
                                          intensity_axis="own_mean"))
    own_chan = recovered_gap(RTNormConfig(intensity_stage=True,
                                          intensity_axis="own_channel"))
    assert abs(default - truth_gap) < 0.10        # real fold change preserved
    # Each extra intensity axis shrinks it, own-channel worst.
    assert own_chan < own_mean < default
    assert own_chan < 0.95 * truth_gap


def test_shipped_defaults():
    """Pin the configuration the benchmark actually selected.

    Every one of these was measured on the mixed-species ground truth; changing
    one silently would invalidate the numbers in bench/RESULTS.md.
    """
    c = RTNormConfig()
    assert c.method == "loess_cap"       # span = min(loess_frac * n, window)
    assert c.loess_frac == 0.05
    assert c.window == 400
    assert c.min_span == 50
    assert c.qvalue_cutoff == 0.05
    assert c.stage_order == "unpaired_first"   # per-time-channel, then paired
    assert c.two_stage is True
    assert c.plex_agg == "mean"
    assert c.center_scope == "channel"   # channel levels left alone
    assert c.mz_stage is False           # measured: no gain
    assert c.intensity_stage is False    # measured: manufactures compression


# --------------------------------------------------------------------------
# Per-scan traces and per-fragment intensities
# --------------------------------------------------------------------------

def _vector_frame(n=600):
    """A quant table carrying every shape of intensity column JMod reports."""
    rng = np.random.default_rng(7)
    d = pd.DataFrame({
        "rt": rng.uniform(5, 55, n),
        "untag_prec": [f"P{i % 150}" for i in range(n)],
        "time_channel": rng.integers(0, 3, n).astype(float),
        "Qvalue": rng.uniform(0, 0.004, n),
        "is_decoy": False,
        "plex_Area": rng.lognormal(10, 0.6, n),
        "coeff": rng.lognormal(10, 0.6, n),
        # MS2_PRIMARY; without it the MS2 level is skipped and there is no
        # rtnorm_factor_MS2 for the fragment vectors to be scaled by.
        "coeff_ChannelUnique": rng.lognormal(10, 0.6, n),
    })
    d["all_ms1_iso0vals"] = [";".join(str(v) for v in rng.lognormal(9, 0.3, 3))
                             for _ in range(n)]
    d["plexfittrace_all"] = d["all_ms1_iso0vals"]
    d["plexfittrace_ps_all"] = ";".join(["0.9"] * 3)   # Pearson, not intensity
    d["all_ms1_specs"] = ";".join(["100", "111", "122"])  # spectrum ids
    d["tic"] = rng.lognormal(20, 0.2, n)               # scan property
    d["obs_int"] = [list(rng.lognormal(8, 0.5, 5)) for _ in range(n)]
    d["frag_int"] = [list(rng.random(5)) for _ in range(n)]  # library, predicted
    return d


def test_traces_take_the_ms1_factor_and_fragments_the_ms2_factor():
    """Vector normalization is opt-in, so the columns must be passed in."""
    from src.quant_normalization import MS1_TRACE_COLS, MS2_TRACE_COLS, SUFFIX
    d = _vector_frame()
    before_iso = [np.fromstring(s, sep=";") for s in d["all_ms1_iso0vals"]]
    before_obs = [np.asarray(v, dtype=float) for v in d["obs_int"]]
    out = apply_rt_channel_normalization(d, timeplex=True, cfg=RTNormConfig(),
                                         ms1_trace_cols=MS1_TRACE_COLS,
                                         ms2_trace_cols=MS2_TRACE_COLS)
    f1 = out["rtnorm_factor_MS1"].to_numpy()
    f2 = out["rtnorm_factor_MS2"].to_numpy()
    for i in (0, 5, 123, 599):
        np.testing.assert_allclose(
            np.fromstring(out["all_ms1_iso0vals" + SUFFIX][i], sep=";"),
            before_iso[i] * f1[i], rtol=1e-9)
        np.testing.assert_allclose(
            np.asarray(out["obs_int" + SUFFIX][i], dtype=float),
            before_obs[i] * f2[i], rtol=1e-9)


@pytest.mark.parametrize("col", ["frag_int", "tic", "plexfittrace_ps_all",
                                 "all_ms1_specs"])
def test_library_and_non_intensity_columns_are_left_alone(col):
    """Predicted intensities, Pearson stats, ids and scan totals are not quant."""
    from src.quant_normalization import MS1_TRACE_COLS, MS2_TRACE_COLS, SUFFIX
    out = apply_rt_channel_normalization(_vector_frame(), timeplex=True,
                                         cfg=RTNormConfig(),
                                         ms1_trace_cols=MS1_TRACE_COLS,
                                         ms2_trace_cols=MS2_TRACE_COLS)
    assert col + SUFFIX not in out.columns


def test_vectors_are_not_normalized_by_default():
    """Opt-in: the copies roughly double the trace columns in all_IDs.csv."""
    from src.quant_normalization import SUFFIX
    out = apply_rt_channel_normalization(_vector_frame(), timeplex=True,
                                         cfg=RTNormConfig())
    for c in ("all_ms1_iso0vals", "plexfittrace_all", "obs_int"):
        assert c + SUFFIX not in out.columns


def test_vector_length_and_shape_are_preserved():
    from src.quant_normalization import MS1_TRACE_COLS, SUFFIX
    d = _vector_frame()
    before = [np.fromstring(s, sep=";") for s in d["all_ms1_iso0vals"]]
    out = apply_rt_channel_normalization(d, timeplex=True, cfg=RTNormConfig(),
                                         ms1_trace_cols=MS1_TRACE_COLS)
    for i in (0, 42, 300):
        a = before[i]
        b = np.fromstring(out["all_ms1_iso0vals" + SUFFIX][i], sep=";")
        assert len(a) == len(b)
        # One factor for the whole vector, so every element ratio is identical
        # and any correlation computed from the trace is unchanged.
        assert float(np.std(b / a)) == pytest.approx(0.0, abs=1e-9)


def test_scale_vector_column_handles_awkward_input():
    from src.quant_normalization import _scale_vector_column
    s = pd.Series(["1.0;2.0", "", None, "not;numbers", [1.0, 2.0]], dtype=object)
    out = _scale_vector_column(s, np.array([2.0, 2.0, 2.0, 2.0, np.nan]))
    np.testing.assert_allclose(np.fromstring(out[0], sep=";"), [2.0, 4.0])
    assert out[1] == ""
    assert out[2] is None
    assert out[3] == "not;numbers"          # unparseable -> untouched
    assert list(out[4]) == [1.0, 2.0]       # non-finite factor -> untouched


def test_diagnostic_plot_is_written(tmp_path):
    p = str(tmp_path / "rtnorm_qc.png")
    apply_rt_channel_normalization(_vector_frame(), timeplex=True,
                                   cfg=RTNormConfig(), plot_path=p)
    assert os.path.exists(p) and os.path.getsize(p) > 5000


def test_a_broken_plot_path_does_not_fail_the_search():
    out = apply_rt_channel_normalization(
        _vector_frame(), timeplex=True, cfg=RTNormConfig(),
        plot_path="/nonexistent-dir-xyz/qc.png")
    assert "plex_Area_rtnorm" in out.columns


# --------------------------------------------------------------------------
# Intensity floor on the reference set
# --------------------------------------------------------------------------

def _sentinel_frame(n=900, n_sentinel=300, seed=11):
    """Half the channel's rows are JMod's 0.001 no-signal placeholder.

    Left in the reference set they drag the local centre down hard, because a
    running median of log2 quant does not care that they are placeholders.
    """
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({
        "rt": rng.uniform(5, 55, n),
        "untag_prec": [f"P{i}" for i in range(n)],
        "time_channel": 0.0,
        "Qvalue": rng.uniform(0, 0.004, n),
        "is_decoy": False,
        "plex_Area": rng.lognormal(14, 0.4, n),
        "coeff": rng.lognormal(14, 0.4, n),
    })
    d.loc[d.index[:n_sentinel], "plex_Area"] = 0.001
    return d


def test_placeholder_rows_are_kept_out_of_the_reference_curve():
    d = _sentinel_frame()
    strict = apply_rt_channel_normalization(d.copy(), cfg=RTNormConfig(min_intensity=1.0))
    loose = apply_rt_channel_normalization(d.copy(), cfg=RTNormConfig(min_intensity=0.0))
    # The 0.001 rows are ~14 log2 below the real ones; letting them define the
    # curve moves the factor by orders of magnitude.
    fs = strict["rtnorm_factor_MS1"].to_numpy()
    fl = loose["rtnorm_factor_MS1"].to_numpy()
    assert np.median(np.abs(np.log2(fl / fs))) > 1.0


def test_every_row_is_still_corrected_including_the_placeholders():
    d = _sentinel_frame()
    out = apply_rt_channel_normalization(d, cfg=RTNormConfig(min_intensity=1.0))
    f = out["rtnorm_factor_MS1"].to_numpy()
    assert np.isfinite(f).all() and (f > 0).all()
    assert len(out) == len(d)          # nothing dropped, only down-weighted


def test_min_intensity_default_is_one():
    assert RTNormConfig().min_intensity == 1.0
    from src.default_dict import default_dict
    assert default_dict["rt_norm_min_intensity"]["default"] == 1.0


def test_min_intensity_is_reported_in_describe():
    assert "int>1" in RTNormConfig().describe()
    assert "int>" not in RTNormConfig(min_intensity=0.0).describe()
