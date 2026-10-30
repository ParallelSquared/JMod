#!/usr/bin/env python
"""Evaluate RT normalization (src/rt_normalization.py) on a JMod experiment.

Reads a combined_filtered_IDs.parquet and, for plex_Area and coeff, compares
raw quant with each arm: sample only, run only, sample then run, and run then
sample.  Per arm it prints:

- cv: the median, over precursors and channels, of the CV of quant across
  units (run, or run and time channel).  Every run is treated as a replicate.
- cv_heldout: cv over a random half of the precursors, with the curves fitted
  to the other half only.  The run stage fits the very residuals cv measures,
  so cv keeps falling as the span shrinks even once the curves fit noise;
  where cv falls and cv_heldout doesn't, the span is too small.  It is over
  other precursors than cv, so compare it with raw's cv_heldout, not with cv.
  The curves are fitted to half the points, so are a little noisier than cv's.
  The gap shows only when a span holds few points (~10 or fewer per window,
  or the 1% RT step LOWESS interpolates over, whichever is wider).
- sample_trend: the median, over samples (run and channel), of the range of
  the binned median log2 quant vs RT (10 RT-quantile bins).  The sample stage
  should bring it down.
- run_trend: the median, over units, of the range of the binned median of
  each precursor's log2 quant less its mean over units, vs RT.  The run stage
  should bring it down.
- level_shift: the largest change from raw, over units, of a unit's level:
  the median of each of its shared precursors' log2 quant less the
  precursor's mean over units.  A shift means the arm moved a unit's overall
  level, e.g. the sample stage normalizing each run to which precursors it
  identified.  The run stage alone moves it a little where it removes a
  skewed trend (it centres each curve on the curve's median), so compare
  the arms with the run arm, not with 0.  It says that a level moved, not
  which level is right.

Usage::

    python scripts/eval_rt_normalization.py path/to/combined_filtered_IDs.parquet
    python scripts/eval_rt_normalization.py path/to/combined_filtered_IDs.parquet --sample-frac 0.3 --run-frac 0.1
"""

import argparse
import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.rt_normalization import apply_correction, rt_normalize  # noqa: E402

ARMS = {"raw": (), "sample": ("sample",), "run": ("run",),
        "sample>run": ("sample", "run"), "run>sample": ("run", "sample")}
N_BINS = 10


def evaluate(df, quant_col, sample_frac, run_frac, it, min_points):
    """One row of metrics per arm for *quant_col*."""
    time_channel = ["time_channel"] if "time_channel" in df.columns else []
    run_cols = ["run_idx", *time_channel]
    pair_cols = ["untag_prec", *time_channel]
    sample_cols = ["run_idx", "channel"]
    df = df.filter(pl.col(quant_col).is_finite(), pl.col(quant_col) > 0, pl.col("rt").is_finite())
    # A random half of the precursors, by hash; their quant is hidden from the
    # held-out fit, which leaves NaN quant out but still corrects its rows
    heldout = df["untag_prec"].hash(seed=0) % 2 == 1
    hidden = df.with_columns(pl.when(heldout).then(np.nan).otherwise(pl.col(quant_col)).alias(quant_col))

    def corrected(fit_df, steps):
        # df with column q, its quant corrected by the curves fitted to *fit_df*
        corr = rt_normalize(fit_df, quant_col, steps, "rt", sample_frac=sample_frac, run_frac=run_frac,
                            it=it, min_points=min_points,
                            sample_cols=sample_cols, run_cols=run_cols, pair_cols=pair_cols)
        return df.with_columns(apply_correction(df[quant_col].cast(pl.Float64), corr).alias("q"))

    rows = []
    for arm, steps in ARMS.items():
        full = corrected(df, steps)
        residuals = _residuals(full, run_cols, pair_cols)
        levels = residuals.group_by(run_cols).agg(pl.col("v").median().alias("level"))
        if arm == "raw":
            raw_levels = levels
        rows.append({"quant": quant_col, "arm": arm,
                     "cv": _median_cv(full, run_cols, pair_cols),
                     "cv_heldout": _median_cv(corrected(hidden, steps).filter(heldout), run_cols, pair_cols),
                     "sample_trend": _median_trend(full.with_columns(pl.col("q").log(2).alias("v")),
                                                   sample_cols),
                     "run_trend": _median_trend(residuals, run_cols),
                     "level_shift": levels.join(raw_levels, on=run_cols, suffix="_raw")
                     .select((pl.col("level") - pl.col("level_raw")).abs().max()).item()})
    return pl.DataFrame(rows)


def _median_cv(df, run_cols, pair_cols):
    keys = list(dict.fromkeys([*pair_cols, "channel"]))
    return (
        df.group_by(*keys, *[c for c in run_cols if c not in keys]).agg(pl.col("q").median())
        .group_by(keys).agg((pl.col("q").std() / pl.col("q").mean()).alias("cv"), pl.len())
        .filter(pl.col("len") >= 2)["cv"].median()
    )


def _residuals(df, run_cols, pair_cols):
    # Each precursor's median log2 quant per unit, less its mean over units, at its median RT
    keys = list(dict.fromkeys([*pair_cols, *run_cols]))
    return (
        df.group_by(keys).agg(pl.col("q").log(2).median().alias("v"), pl.col("rt").median())
        .filter(pl.len().over(pair_cols) >= 2)
        .with_columns(pl.col("v") - pl.col("v").mean().over(pair_cols))
    )


def _median_trend(df, group_cols):
    # Per group, the range of the median of v in RT-quantile bins; the median over groups
    ranges = []
    for _, group in df.partition_by(group_cols, as_dict=True).items():
        if group.height < N_BINS * 10:
            continue
        rt, v = group["rt"].to_numpy(), group["v"].to_numpy()
        which = np.digitize(rt, np.quantile(rt, np.linspace(0, 1, N_BINS + 1)[1:-1]))
        ranges.append(np.ptp([np.median(v[which == b]) for b in range(N_BINS)]))
    return float(np.median(ranges)) if ranges else float("nan")


def main():
    parser = argparse.ArgumentParser(description="Evaluate RT normalization on combined_filtered_IDs.parquet")
    parser.add_argument("ids", help="path to combined_filtered_IDs.parquet")
    parser.add_argument("--sample-frac", type=float, default=0.2, help="sample stage LOWESS span (default 0.2)")
    parser.add_argument("--run-frac", type=float, default=0.05, help="run stage LOWESS span (default 0.05)")
    parser.add_argument("--it", type=int, default=3, help="LOWESS robustness iterations (default 3)")
    parser.add_argument("--min-points", type=int, default=50, help="fewest points to fit (default 50)")
    args = parser.parse_args()

    df = pl.read_parquet(args.ids)
    results = pl.concat([evaluate(df, quant_col, args.sample_frac, args.run_frac, args.it, args.min_points)
                         for quant_col in ("plex_Area", "coeff") if quant_col in df.columns])
    with pl.Config(tbl_rows=-1, float_precision=3):
        print(results)


if __name__ == "__main__":
    main()
