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

"""LOWESS normalization of RT-dependent trends in quant.

Two stages, each returning a per-row log2 correction in the table's row
order; the corrected quant is apply_correction(quant, corr):

- lowess_rt_correction: flattens log2(quant) vs RT within each sample.
- paired_rt_correction: per precursor, its log2 quant less its mean over runs,
  fitted against RT in each run and centred, so the RT-dependent differences
  between runs are removed without changing any run's level.

rt_normalize runs any sequence of them.  Nothing here depends on the rest of
JMod; every column is named by the caller.
"""

import warnings

import numpy as np
import polars as pl
from statsmodels.nonparametric.smoothers_lowess import lowess

_ROW = "__row"
_Y = "__log2_quant"
_RT = "__rt"


def lowess_rt_correction(df, quant_col, sample_cols, rt_col, *, frac, it=3, min_points=50):
    """Per row, the log2 correction that flattens log2(*quant_col*) vs
    *rt_col* within each *sample_cols* group: the LOWESS curve less the
    median log2 quant.  A sample with fewer than *min_points* points to fit
    is left uncorrected (0), with a warning."""
    corr = np.zeros(df.height)
    for sample, group in _with_log2(df, quant_col, rt_col).partition_by(sample_cols, as_dict=True).items():
        rt, y = group[_RT].to_numpy(), group[_Y].to_numpy()
        curve = _fit_curve(rt, y, frac, it, min_points)
        if curve is None:
            warnings.warn(f"RT normalization: sample {sample} has fewer than {min_points} points "
                          f"to fit, so it is left uncorrected")
            continue
        fitted = np.isfinite(rt) & np.isfinite(y)
        corr[group[_ROW].to_numpy()] = _interp(curve, np.median(y[fitted]), rt)
    return pl.Series("rt_correction", corr)


def paired_rt_correction(df, quant_col, run_cols, pair_cols, rt_col, *, frac, it=3, min_points=50):
    """Per row, the log2 correction that removes RT-dependent differences
    between the units of *run_cols* (e.g. run, or run and time channel).

    Each precursor (*pair_cols*) seen in at least 2 units has a residual in
    each, its log2 quant less its mean over the units.  The residuals are
    fitted against RT in each unit, centred on the fit's median.  A unit with
    fewer than *min_points* residuals is left uncorrected (0), with a warning.
    """
    rows = _with_log2(df, quant_col, rt_col)
    resid = (
        _collapse(rows, run_cols, pair_cols)
        .filter(pl.len().over(pair_cols) >= 2)
        .with_columns((pl.col(_Y) - pl.col(_Y).mean().over(pair_cols)).alias(_Y))
    )
    fits = {unit: _fit_curve(group[_RT].to_numpy(), group[_Y].to_numpy(), frac, it, min_points)
            for unit, group in resid.partition_by(run_cols, as_dict=True).items()}

    corr = np.zeros(df.height)
    for unit, group in rows.partition_by(run_cols, as_dict=True).items():
        curve = fits.get(unit)
        if curve is None:
            warnings.warn(f"RT normalization: unit {unit} shares fewer than {min_points} precursors "
                          f"with the other units, so it is left uncorrected")
            continue
        corr[group[_ROW].to_numpy()] = _interp(curve, np.median(curve[:, 1]), group[_RT].to_numpy())
    return pl.Series("rt_correction", corr)


def apply_correction(quant, corr):
    """*quant* corrected by the log2 correction *corr*."""
    return quant * np.exp2(-corr)


def rt_normalize(df, quant_col, steps, rt_col, *, frac, it=3, min_points=50,
                 sample_cols=None, run_cols=None, pair_cols=None):
    """The total log2 correction of running *steps* in order, each on the
    quant as corrected by those before it.  A step is "sample"
    (lowess_rt_correction, needs *sample_cols*) or "run"
    (paired_rt_correction, needs *run_cols* and *pair_cols*)."""
    total = pl.Series("rt_correction", np.zeros(df.height))
    for step in steps:
        if step == "sample" and sample_cols is not None:
            corr = lowess_rt_correction(df, quant_col, sample_cols, rt_col,
                                        frac=frac, it=it, min_points=min_points)
        elif step == "run" and run_cols is not None and pair_cols is not None:
            corr = paired_rt_correction(df, quant_col, run_cols, pair_cols, rt_col,
                                        frac=frac, it=it, min_points=min_points)
        else:
            raise ValueError(f"RT normalization step {step!r} is unknown or its columns weren't given")
        total = total + corr
        df = df.with_columns(apply_correction(df[quant_col], corr).alias(quant_col))
    return total


def _with_log2(df, quant_col, rt_col):
    # Row index, RT and log2 quant (NaN where the quant isn't positive) as floats
    quant = pl.col(quant_col).cast(pl.Float64)
    return df.with_row_index(_ROW).with_columns(
        pl.col(rt_col).cast(pl.Float64).fill_null(np.nan).alias(_RT),
        pl.when(quant > 0).then(quant.log(2)).otherwise(np.nan).alias(_Y),
    )


def _collapse(rows, run_cols, pair_cols):
    # One row per (precursor, unit): the median log2 quant and median RT of
    # its rows with a finite RT and quant
    keys = list(dict.fromkeys([*pair_cols, *run_cols]))
    return (
        rows.filter(pl.col(_Y).is_finite(), pl.col(_RT).is_finite())
        .group_by(keys)
        .agg(pl.col(_Y).median(), pl.col(_RT).median())
    )


def _fit_curve(x, y, frac, it, min_points):
    # LOWESS of y on x over the finite points, sorted by x; None if too few
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < min_points:
        return None
    x, y = x[keep], y[keep]
    return lowess(y, x, frac=frac, it=it, delta=0.01 * np.ptp(x), return_sorted=True)


def _interp(curve, centre, rt):
    # The curve less *centre* at each RT, flat beyond its ends; 0 where RT isn't finite
    finite = np.isfinite(rt)
    return np.where(finite, np.interp(np.where(finite, rt, 0), curve[:, 0], curve[:, 1] - centre), 0)
