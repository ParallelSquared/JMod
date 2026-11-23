"""RT-dependent normalization (src/rt_normalization.py) on synthetic data
with a known injected RT bias."""

import numpy as np
import polars as pl
import pytest

from src.rt_normalization import (normalize_rt, running_median_curve, select_references, unit_factor,
                                  window_size)

NOISE_SD = 0.15
FDR = 0.01


def _bias(rng, rt):
    """A smooth RT bias: three random sinusoids and a step, about ±1 log2."""
    out = np.zeros_like(rt)
    for _ in range(3):
        period = rng.uniform(15, 60)
        out += rng.uniform(0.15, 0.35) * np.sin(2 * np.pi * rt / period + rng.uniform(0, 2 * np.pi))
    out += np.where(rt > rng.uniform(15, 45), rng.uniform(-0.4, 0.4), 0.0)
    return out


def _table(log2, rt, run_idx, channel, untag_prec, qvalue=None):
    """A combined-IDs-shaped table: NaN in *log2* is a missing plex_Area."""
    n = len(log2)
    return pl.DataFrame({
        "run_idx": np.asarray(run_idx, dtype=np.int32),
        "channel": np.asarray(channel, dtype=np.int64),
        "silac_channel": np.full(n, np.nan),
        "untag_prec": untag_prec,
        "rt": rt,
        "plex_Area": np.where(np.isnan(log2), 0.0, np.exp2(np.nan_to_num(log2))),
        "Qvalue": np.full(n, 0.001) if qvalue is None else qvalue,
        "is_decoy": np.zeros(n, dtype=bool),
    })


def _runs_data(seed=0, n_prec=6000, n_runs=10, n_channels=1, de=None, missing=0.1, rt_jitter=0.2):
    """Runs as samples: log2 = mu_p + offset_r + bias_r(rt) + channel offset + noise.

    *de* maps precursor index -> (runs, log2 change).  Returns the table and
    the truth (bias per run as a function, mu, offsets).
    """
    rng = np.random.default_rng(seed)
    mu = rng.normal(20, 2, n_prec)
    rt_p = rng.uniform(0, 60, n_prec)
    # Each run's bias, on a fine grid to interpolate at any RT
    grid = np.linspace(-5, 65, 7001)
    bias_grid = [_bias(np.random.default_rng(seed * 100 + r), grid) for r in range(n_runs)]
    offsets = rng.normal(0, 0.5, n_runs)
    channel_offsets = rng.normal(0, 0.3, n_channels)
    cols = {k: [] for k in ("log2", "rt", "run", "channel", "prec")}
    precs = np.array([f"P{i}_2" for i in range(n_prec)])
    for r in range(n_runs):
        rt = rt_p + rng.normal(0, rt_jitter, n_prec)
        bias = np.interp(rt, grid, bias_grid[r])
        for c in range(n_channels):
            log2 = mu + offsets[r] + channel_offsets[c] + bias + rng.normal(0, NOISE_SD, n_prec)
            if de:
                for p, (runs, change) in de.items():
                    if r in runs:
                        log2[p] += change
            log2[rng.random(n_prec) < missing] = np.nan
            cols["log2"].append(log2)
            cols["rt"].append(rt)
            cols["run"].append(np.full(n_prec, r + 1))
            cols["channel"].append(np.full(n_prec, c))
            cols["prec"].append(precs)
    df = _table(*(np.concatenate(cols[k]) for k in ("log2", "rt", "run", "channel", "prec")))
    truth = {"grid": grid, "bias": bias_grid, "mu": mu, "offsets": offsets}
    return df, truth


def _curve(curves, unit):
    return next(c for c in curves if c.unit == unit)


def _centered_bias(truth, rt, run):
    """The identifiable bias of *run* at *rt*: relative to the mean over runs."""
    all_runs = np.array([np.interp(rt, truth["grid"], b) for b in truth["bias"]])
    return all_runs[run - 1] - all_runs.mean(axis=0)


class TestCore:
    def test_window_size(self):
        assert window_size(1000) == 200
        assert window_size(20_000) == 200
        assert window_size(50_000) == 500

    def test_running_median_of_a_constant_is_the_constant(self):
        rt = np.linspace(0, 10, 500)
        curve_rt, curve = running_median_curve(rt, np.full(500, 2.5), 200)
        assert np.allclose(curve, 2.5)
        assert np.all(np.diff(curve_rt) > 0)

    def test_running_median_follows_a_step_and_holds_flat_at_the_ends(self):
        rt = np.linspace(0, 60, 4000)
        d = np.where(rt > 30, 1.0, 0.0)
        curve_rt, curve = running_median_curve(rt, d, 200)
        assert curve[0] == 0 and curve[-1] == 1
        assert np.interp(-10, curve_rt, curve) == 0 and np.interp(100, curve_rt, curve) == 1

    def test_running_median_does_not_depend_on_input_order(self):
        rng = np.random.default_rng(1)
        rt = np.round(rng.uniform(0, 60, 3000), 1)  # ties in RT
        d = rng.normal(size=3000)
        perm = rng.permutation(3000)
        a = running_median_curve(rt, d, 200)
        b = running_median_curve(rt[perm], d[perm], 200)
        assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])

    def test_unit_factor_needs_two_windows(self):
        rt = np.linspace(0, 60, 399)
        assert unit_factor(rt, np.zeros(399), rt) is None
        assert unit_factor(np.linspace(0, 60, 400), np.zeros(400), rt) is not None

    def test_unit_factor_removes_the_level(self):
        rng = np.random.default_rng(2)
        rt = rng.uniform(0, 60, 3000)
        d = 3.0 + np.sin(rt / 10) + rng.normal(0, 0.1, 3000)
        rows = rng.uniform(0, 60, 5000)
        row_factor, _, _, _ = unit_factor(rt, d, rows)
        assert abs(np.median(row_factor)) < 1e-9

    def test_select_references(self):
        rng = np.random.default_rng(3)
        matrix = rng.normal(20, 1, (100, 4)) + rng.normal(0, 1, (100, 1)) * np.array([0, 1, 2, 3])
        matrix[:10, 1:] = np.nan  # observed once: never a reference
        reference, levels = select_references(matrix, "median")
        assert not reference[:10].any()
        assert reference.sum() == 36  # 40% of the 90 candidates
        assert np.allclose(levels[reference], np.nanmedian(matrix[reference], axis=1))
        assert np.isnan(levels[~reference]).all()


@pytest.fixture(scope="module")
def runs_result():
    """The spec's setting: 20k precursors over 60 min, 10 runs."""
    df, truth = _runs_data(n_prec=20_000)
    out, curves = normalize_rt(df, "run_rt", FDR)
    return df, truth, out, curves


class TestRunsMode:
    @pytest.fixture
    def result(self, runs_result):
        return runs_result

    def test_curve_follows_the_injected_bias(self, result):
        _, truth, _, curves = result
        assert len(curves) == 10
        for curve in curves:
            run = curve.unit[0]
            expected = _centered_bias(truth, curve.ref_rt, run)
            assert np.corrcoef(curve.ref_factor, expected)[0, 1] > 0.95

    def test_residual_sd_close_to_the_noise(self, result):
        df, truth, out, _ = result
        n_prec = len(truth["mu"])
        before = np.log2(out["plex_Area"].replace(0.0, None).to_numpy())
        after = np.log2(out["plex_Area_norm"].to_numpy())
        prec = np.tile(np.arange(n_prec), 10)
        run = out["run_idx"].to_numpy()

        def residual_sd(values):
            resid = values - truth["mu"][prec] - truth["offsets"][run - 1]
            # each run's level is left alone: take it out
            for r in range(1, 11):
                in_run = (run == r) & np.isfinite(resid)
                resid[in_run] -= np.median(resid[in_run])
            # the bias common to all runs is not identifiable: take it out per precursor
            resid = resid.reshape(10, n_prec)
            resid = resid - np.nanmean(resid, axis=0)
            return np.nanstd(resid)

        assert residual_sd(before) > 2 * NOISE_SD
        assert residual_sd(after) < 1.3 * NOISE_SD

    def test_rows_and_missing_values_kept_and_plex_area_unchanged(self, result):
        df, _, out, _ = result
        assert out.height == df.height
        assert out["plex_Area"].equals(df["plex_Area"])
        missing = (df["plex_Area"] == 0).to_numpy()
        assert missing.any()
        assert out["plex_Area_norm"].is_null().to_numpy()[missing].all()
        assert out["plex_Area_norm"].is_not_null().to_numpy()[~missing].all()

    def test_each_runs_level_is_kept(self, result):
        _, _, out, _ = result
        assert _median_factor(out, "run_idx").abs().max() < 0.05

    def test_output_does_not_depend_on_row_order(self, result):
        df, _, out, _ = result
        shuffled = df.sample(fraction=1.0, shuffle=True, seed=7)
        out2, _ = normalize_rt(shuffled, "run_rt", FDR)
        key = ["run_idx", "channel", "untag_prec"]
        a = out.sort(key)["plex_Area_norm"].to_numpy()
        b = out2.sort(key)["plex_Area_norm"].to_numpy()
        assert np.allclose(a, b, equal_nan=True)

    def test_fold_changes_spread_over_rt_are_recovered(self):
        n_prec = 6000
        de_precs = np.arange(0, n_prec, 20)  # 5%, spread evenly over RT (RTs are random)
        de = {p: (range(5, 10), 1.0) for p in de_precs}
        df, _ = _runs_data(seed=4, de=de)
        out, _ = normalize_rt(df, "run_rt", FDR)
        assert _fold_change(out, de_precs, n_prec) == pytest.approx(1.0, abs=0.1)

    def test_change_concentrated_in_one_rt_region_is_attenuated(self):
        """The documented failure mode: when many precursors in one RT region
        change together, the curve takes part of the change for bias."""
        n_prec = 6000
        df, _ = _runs_data(seed=5)
        rt_p = df.filter(pl.col("run_idx") == 1)["rt"].to_numpy()  # close to the precursor RTs
        region = np.flatnonzero((rt_p > 20) & (rt_p < 30))
        de_precs = region[: int(0.8 * len(region))]
        de = {p: (range(5, 10), 1.0) for p in de_precs}
        df, _ = _runs_data(seed=5, de=de)
        out, _ = normalize_rt(df, "run_rt", FDR)
        assert _fold_change(out, de_precs, n_prec) < 0.7

    def test_channel_ratios_within_a_run_are_unchanged(self):
        df, _ = _runs_data(seed=6, n_prec=3000, n_runs=4, n_channels=3)
        out, curves = normalize_rt(df, "run_rt", FDR)
        assert len(curves) == 4
        ratio = out.filter(pl.col("plex_Area") > 0).with_columns(
            (pl.col("plex_Area_norm") / pl.col("plex_Area")).alias("r"))
        # every channel of a precursor in a run (same RT) gets the same factor
        spread = ratio.group_by("run_idx", "untag_prec").agg((pl.col("r").max() / pl.col("r").min()).alias("s"))
        assert spread["s"].max() == pytest.approx(1.0, abs=1e-9)

    def test_one_channel_with_a_large_change_does_not_move_the_curve(self):
        rng = np.random.default_rng(8)
        n_prec, n_runs = 4000, 4
        mu = rng.normal(20, 2, n_prec)
        rt = rng.uniform(0, 60, n_prec)
        frames = []
        for r in range(1, n_runs + 1):
            for c in range(3):
                log2 = mu + rng.normal(0, NOISE_SD, n_prec)
                if r == 1 and c == 2:
                    log2 = log2 + np.where((rt > 20) & (rt < 40), 3.0, 0.0)
                frames.append(_table(log2, rt, np.full(n_prec, r), np.full(n_prec, c),
                                     [f"P{i}_2" for i in range(n_prec)]))
        out, curves = normalize_rt(pl.concat(frames), "run_rt", FDR)
        curve = _curve(curves, (1,))
        in_region = (curve.curve_rt > 20) & (curve.curve_rt < 40)
        assert np.abs(curve.curve_factor[in_region]).max() < 0.2

    def test_a_negative_channel_does_not_stop_its_run_being_normalized(self):
        rng = np.random.default_rng(9)
        n_prec, n_runs = 4000, 4
        mu = rng.normal(20, 2, n_prec)
        rt = rng.uniform(0, 60, n_prec)
        frames = []
        for r in range(1, n_runs + 1):
            for c in range(3):
                log2 = mu + np.sin(rt / 8) * (r == 1) + rng.normal(0, NOISE_SD, n_prec)
                if r == 1 and c == 2:
                    # background only: a few low precursors
                    log2 = np.where(rng.random(n_prec) < 0.05, log2 - 6, np.nan)
                frames.append(_table(log2, rt, np.full(n_prec, r), np.full(n_prec, c),
                                     [f"P{i}_2" for i in range(n_prec)]))
        _, curves = normalize_rt(pl.concat(frames), "run_rt", FDR)
        assert (1,) in [c.unit for c in curves]

    def test_one_run_is_not_normalized(self):
        df, _ = _runs_data(seed=10, n_prec=2000, n_runs=1)
        out, curves = normalize_rt(df, "run_rt", FDR)
        assert curves == []
        assert out["plex_Area_norm"].is_null().all()


def _median_factor(out, by):
    """Per *by*, the median log2 factor applied (plex_Area / plex_Area_norm)."""
    return (out.filter(pl.col("plex_Area_norm").is_not_null())
            .group_by(by).agg((pl.col("plex_Area") / pl.col("plex_Area_norm")).log(2).median().alias("f"))["f"])


def _fold_change(out, de_precs, n_prec):
    """Mean log2 change of *de_precs* between runs 6-10 and 1-5, relative to
    the precursors that did not change."""
    values = out.with_columns(pl.col("plex_Area_norm").log(2).alias("v")).filter(pl.col("v").is_finite())
    change = (values.with_columns((pl.col("run_idx") > 5).alias("b"))
              .group_by("untag_prec", "b").agg(pl.col("v").mean())
              .pivot("b", index="untag_prec", values="v")
              .with_columns((pl.col("true") - pl.col("false")).alias("change")))
    is_de = pl.col("untag_prec").is_in([f"P{p}_2" for p in de_precs])
    de_change = change.filter(is_de)["change"].mean()
    rest = change.filter(~is_de)["change"].mean()
    return de_change - rest


@pytest.fixture(scope="module")
def single_run():
    """One run, 5 channels, 20k precursors.  Channels 0 and 1 carry their own RT bias and
    are set far above and below the rest, so the across-channel median is
    always channel 3, which has no bias."""
    rng = np.random.default_rng(11)
    n_prec = 20_000
    mu = rng.normal(20, 2, n_prec)
    rt = rng.uniform(0, 60, n_prec)
    offsets = [5.0, -5.0, -0.5, 0.0, 0.5]
    biases = [_bias(rng, rt), _bias(rng, rt), 0 * rt, 0 * rt, 0 * rt]
    frames = []
    for c in range(5):
        log2 = mu + offsets[c] + biases[c] + rng.normal(0, NOISE_SD, n_prec)
        log2[rng.random(n_prec) < 0.1] = np.nan
        frames.append(_table(log2, rt, np.full(n_prec, 1), np.full(n_prec, c),
                             [f"P{i}_2" for i in range(n_prec)]))
    df = pl.concat(frames)
    out, curves = normalize_rt(df, "channel_rt", FDR)
    return df, out, curves, rt, biases


class TestChannelsMode:
    def test_channel_specific_bias_is_recovered_on_a_single_file(self, single_run):
        _, _, curves, rt, biases = single_run
        assert len(curves) == 5
        for c in (0, 1):
            curve = _curve(curves, (1, c))
            expected = np.interp(curve.ref_rt, np.sort(rt), biases[c][np.argsort(rt)])
            assert np.corrcoef(curve.ref_factor, expected)[0, 1] > 0.95

    def test_channel_offsets_are_left_in_place(self, single_run):
        _, out, _, _, _ = single_run
        assert _median_factor(out, "channel").abs().max() < 0.05
        levels = (out.filter(pl.col("plex_Area") > 0).group_by("channel")
                  .agg(pl.col("plex_Area_norm").log(2).median()).sort("channel")["plex_Area_norm"])
        # the +5 and -5 channels are still about +5 and -5 from the unbiased one
        assert levels[0] - levels[3] == pytest.approx(5, abs=0.2)
        assert levels[1] - levels[3] == pytest.approx(-5, abs=0.2)

    def test_a_negative_channel_with_background_ids_stays_low(self):
        rng = np.random.default_rng(12)
        n_prec = 6000
        mu = rng.normal(20, 2, n_prec)
        rt = rng.uniform(0, 60, n_prec)
        frames = []
        for c in range(4):
            log2 = mu + rng.normal(0, NOISE_SD, n_prec)
            if c == 3:  # background in a third of the precursors, far below
                log2 = np.where(rng.random(n_prec) < 0.33, log2 - 6, np.nan)
            frames.append(_table(log2, rt, np.full(n_prec, 1), np.full(n_prec, c),
                                 [f"P{i}_2" for i in range(n_prec)]))
        out, curves = normalize_rt(pl.concat(frames), "channel_rt", FDR)
        # Its precursors are far from the other channels, so none is among the
        # least variable: it gets no references and is left empty.  Either way
        # it must not be scaled up
        negative = out.filter((pl.col("channel") == 3) & (pl.col("plex_Area") > 0))
        normalized = negative.filter(pl.col("plex_Area_norm").is_not_null())
        if normalized.height:
            assert abs((normalized["plex_Area_norm"] / normalized["plex_Area"]).log(2).median()) < 0.1
        assert (1, 0) in [c.unit for c in curves]

    def test_a_channel_with_too_few_references_is_left_empty(self):
        rng = np.random.default_rng(13)
        n_prec = 3000
        mu = rng.normal(20, 2, n_prec)
        rt = rng.uniform(0, 60, n_prec)
        frames = []
        for c in range(3):
            log2 = mu + rng.normal(0, NOISE_SD, n_prec)
            if c == 2:
                log2[300:] = np.nan  # 300 precursors, so far fewer than 400 references
            frames.append(_table(log2, rt, np.full(n_prec, 1), np.full(n_prec, c),
                                 [f"P{i}_2" for i in range(n_prec)]))
        out, curves = normalize_rt(pl.concat(frames), "channel_rt", FDR)
        assert (1, 2) not in [c.unit for c in curves]
        assert out.filter(pl.col("channel") == 2)["plex_Area_norm"].is_null().all()
        assert out.filter(pl.col("channel") == 0)["plex_Area_norm"].is_not_null().all()

    def test_label_free_has_nothing_to_compare(self):
        df, _ = _runs_data(seed=14, n_prec=2000, n_runs=2)
        out, curves = normalize_rt(df, "channel_rt", FDR)
        assert curves == []
        assert out["plex_Area_norm"].is_null().all()


class TestSkipped:
    def test_none_leaves_the_table_unchanged(self):
        df, _ = _runs_data(seed=15, n_prec=500, n_runs=2)
        out, curves = normalize_rt(df, None, FDR)
        assert out.equals(df) and curves == []

    def test_timeplex_is_not_normalized(self):
        df, _ = _runs_data(seed=16, n_prec=2000, n_runs=3)
        df = df.with_columns((pl.col("untag_prec").str.len_chars() % 2).alias("time_channel"))
        out, curves = normalize_rt(df, "run_rt", FDR)
        assert curves == []
        assert out["plex_Area_norm"].is_null().all()

    def test_decoys_and_failing_qvalues_are_not_references(self):
        df, _ = _runs_data(seed=17, n_prec=3000, n_runs=3)
        out, curves = normalize_rt(df.with_columns(pl.lit(0.5).alias("Qvalue")), "run_rt", FDR)
        assert curves == []
        decoys = df.with_columns(pl.lit(True).alias("is_decoy"))
        out, curves = normalize_rt(decoys, "run_rt", FDR)
        assert curves == [] and out["plex_Area_norm"].is_null().all()

    def test_unknown_mode_is_an_error(self):
        df, _ = _runs_data(seed=18, n_prec=100, n_runs=2)
        with pytest.raises(Exception, match="--norm"):
            normalize_rt(df, "bins", FDR)


class TestCombineRuns:
    def test_combined_table_gets_plex_area_norm_and_the_plots(self, tmp_path):
        from src.multi_run import combine_runs
        df, _ = _runs_data(seed=19, n_prec=3000, n_runs=3)
        runs = {}
        for (run_idx,), rows in df.partition_by("run_idx", as_dict=True).items():
            folder = tmp_path / f"run{run_idx}"
            (folder / "outputs").mkdir(parents=True)
            n = rows.height
            seqs = rows["untag_prec"].str.replace("_2", "")
            rows.drop("run_idx").with_columns(
                seqs.alias("seq"), seqs.alias("stripped_seq"), seqs.alias("untag_seq"),
                pl.lit(2.0).alias("z"), pl.lit(f"run{run_idx}").alias("file_name"),
                pl.lit("P1").alias("protein"), pl.lit(0.9).alias("PredVal"),
                pl.lit(0.001).alias("BestChannel_Qvalue"), pl.lit(0.001).alias("Protein_Qvalue"),
                pl.lit(500.0).alias("mz"),
            ).write_parquet(folder / "outputs" / "all_IDs_filtered.parquet")
            runs[run_idx] = str(folder)
        experiment_dir = tmp_path / "experiment"
        experiment_dir.mkdir()
        combined = combine_runs(runs, str(experiment_dir), target_decoy_ratio=1.0, fdr_threshold=0.01,
                                norm="run_rt")
        assert combined.height == df.height
        assert combined["plex_Area_norm"].is_not_null().sum() > 0
        written = pl.read_parquet(experiment_dir / "combined_filtered_IDs.parquet")
        assert "plex_Area_norm" in written.columns
        names = {p.name for p in (experiment_dir / "experiment_results").iterdir()}
        assert {"13_rt_norm_factors.png", "14_rt_norm_deviation.png"} <= names

    def test_plots_for_channel_units(self, tmp_path, single_run):
        from src.multi_run import plot_rt_norm_deviation, plot_rt_norm_factors
        _, _, curves, _, _ = single_run
        plot_rt_norm_factors(curves, str(tmp_path))
        plot_rt_norm_deviation(curves, str(tmp_path))
        assert (tmp_path / "13_rt_norm_factors.png").is_file()
        assert (tmp_path / "14_rt_norm_deviation.png").is_file()
