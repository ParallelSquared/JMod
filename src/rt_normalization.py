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

"""RT-dependent normalization of plex_Area.

Corrects intensity bias that varies along the gradient within a unit (a run,
or a run's channel): spray instability, gradient-dependent transmission,
channel-specific losses that depend on hydrophobicity.  A single factor per
unit cannot fix this.

Method, per unit:

1. Reference precursors: those observed in at least half of the samples,
   keeping the 40% least variable across them.  Each has a reference level
   across the samples.
2. Deviation d = log2 plex_Area - reference level, for each reference
   precursor the unit has.
3. A running median of d over the k reference points nearest in RT order,
   k = max(200, 1% of the unit's reference points).  The window is a number of
   points, not minutes, so it is wide where references are sparse and narrow
   where they are dense.  At the ends it takes the first or last k points, so
   the curve is held flat beyond them rather than extrapolated.
4. Every row of the unit takes the curve at its RT (linear interpolation).
   The curve's median over the unit's rows is subtracted first: only the RT
   shape is corrected, and each unit keeps its overall level.

Two modes, by what the samples are.  The same sample is never assumed to be
in the same channel in different runs.

- "channel_rt": each channel of a run is a sample.  Each run is normalized on
  its own (one file is enough), with one curve per (run, channel).  Channels
  co-elute, so spray fluctuation is shared and mostly cancels between them;
  this corrects what is specific to a channel and depends on RT.
- "run_rt": each run is a sample.  A precursor's value in a run is the median
  log2 over its observed channels; one curve per run, applied to every channel
  of the run, so channel ratios within a run are unchanged.

Assumption: within each RT neighbourhood most precursors are unchanged
between samples.  A biological change concentrated in one RT region (e.g. a
class of hydrophobic peptides all changing together) is partly removed.

The overall level of a unit is left alone on purpose.  Equalizing it is
loading normalization, which would scale an unrecognized negative (an empty
cell, a blank) up to look like a real sample; negatives are not known during
processing.

The RT is each row's observed apex RT: the bias follows the instrument clock,
and each curve belongs to one run, so no alignment between runs is needed.

A unit needs at least 2k reference points (two non-overlapping windows);
with fewer its plex_Area_norm is left empty and a warning names it.  Units
are never given a single global factor instead.

##TODO references pass Qvalue; Channel_Qvalue is the intended column
##TODO timeplex is skipped: its time channels are separate injections at
#  different acquisition times, so it needs normalization within each injection
##TODO further normalization settings that each correct one thing (loading,
#  RT shape, ...), chosen independently
"""

import math
from typing import NamedTuple

import numpy as np
import polars as pl
from numpy.lib.stride_tricks import sliding_window_view

from src.logger import logger
from src.utils.errors import JModError

RT_NORM_MODES = ("run_rt", "channel_rt")

# The running median's window: at least this many reference points, or this
# fraction of the unit's reference points when that is more
_MIN_WINDOW = 200
_WINDOW_FRACTION = 0.01
# Reference precursors: observed in at least this fraction of the samples
# (and at least 2, for a spread), keeping this fraction of the least variable
_MIN_OBSERVED_FRACTION = 0.5
_REFERENCE_FRACTION = 0.4
# Elements of the running median's windows taken at once (~32 MB of float64)
_WINDOW_CHUNK_ELEMENTS = 1 << 22

# The channel columns a unit is told apart by in channel_rt mode
_CHANNEL_COLUMNS = ("silac_channel", "channel")


class RtCurve(NamedTuple):
    """One unit's correction, for the plots."""
    unit: tuple          # (run_idx,) or (run_idx, *channel values)
    curve_rt: np.ndarray
    curve_factor: np.ndarray   # log2, level removed
    ref_rt: np.ndarray
    ref_d: np.ndarray          # the reference deviations before correction
    ref_factor: np.ndarray     # the factor applied at each reference point


# ── Core (numpy only) ───────────────────────────────────────────────────────


def window_size(n_ref):
    """The running median's window for a unit of *n_ref* reference points."""
    return max(_MIN_WINDOW, int(n_ref * _WINDOW_FRACTION))


def running_median_curve(rt, d, k):
    """Running median of *d* over the *k* points nearest in RT order.

    Points are sorted by RT (ties by d, so the result does not depend on the
    input order); each point's value is the median of the k points centred on
    it, the window clipped at the ends so it always holds k points.  Points
    sharing an RT are averaged, so the curve's RTs strictly increase.

    Returns (curve_rt, curve_value).  Needs len(rt) >= k.
    """
    order = np.lexsort((d, rt))
    rt_sorted = rt[order]
    d_sorted = d[order]
    n = len(rt_sorted)
    starts = np.clip(np.arange(n) - k // 2, 0, n - k)
    windows = sliding_window_view(d_sorted, k)
    medians = np.empty(n)
    chunk = max(1, _WINDOW_CHUNK_ELEMENTS // k)
    for i in range(0, n, chunk):
        medians[i:i + chunk] = np.median(windows[starts[i:i + chunk]], axis=1)
    curve_rt, inverse = np.unique(rt_sorted, return_inverse=True)
    curve_value = np.bincount(inverse, weights=medians) / np.bincount(inverse)
    return curve_rt, curve_value


def unit_factor(ref_rt, ref_d, rt):
    """A unit's log2 correction at each RT in *rt* (the RTs of all of the
    unit's quantified rows), from its reference points.

    The curve's median over *rt* is removed, so the median factor the unit's
    rows get is 0: its overall level is kept.  (Not the median over the
    reference points: references are chosen for a low spread, which the bias
    itself changes along RT, so they need not be spread over RT like the rows.)

    Returns (factor at *rt*, factor at the reference points, curve_rt,
    curve_factor), or None when the unit has fewer than two windows of
    reference points.
    """
    n_ref = len(ref_rt)
    k = window_size(n_ref)
    if n_ref < 2 * k:
        return None
    curve_rt, curve_value = running_median_curve(ref_rt, ref_d, k)
    level = np.median(np.interp(rt, curve_rt, curve_value))
    curve_factor = curve_value - level
    return (np.interp(rt, curve_rt, curve_factor), np.interp(ref_rt, curve_rt, curve_factor),
            curve_rt, curve_factor)


def select_references(matrix, center):
    """The reference precursors of a precursor x sample matrix of log2 values
    (NaN where missing), and their reference levels.

    A reference is observed in at least half of the samples (and at least 2);
    of those, the 40% with the least SD across samples are kept (ties by row,
    so sort the rows for a result independent of input order).  Its level is
    the *center* ("median" or "mean") of its observed values.

    Returns (boolean mask over rows, levels: NaN where not a reference).
    """
    n_samples = matrix.shape[1]
    n_observed = np.sum(~np.isnan(matrix), axis=1)
    candidate = n_observed >= max(2, math.ceil(_MIN_OBSERVED_FRACTION * n_samples))
    reference = np.zeros(matrix.shape[0], dtype=bool)
    levels = np.full(matrix.shape[0], np.nan)
    if not candidate.any():
        return reference, levels
    rows = np.flatnonzero(candidate)
    sub = matrix[rows]
    sd = np.nanstd(sub, axis=1, ddof=1)
    n_keep = math.ceil(_REFERENCE_FRACTION * len(rows))
    kept = rows[np.argsort(sd, kind="stable")[:n_keep]]
    reference[kept] = True
    if center == "median":
        levels[kept] = np.nanmedian(matrix[kept], axis=1)
    elif center == "mean":
        levels[kept] = np.nanmean(matrix[kept], axis=1)
    else:
        raise ValueError(f"center must be 'median' or 'mean', not {center!r}")
    return reference, levels


# ── The combined table (polars) ─────────────────────────────────────────────


def normalize_rt(df, mode, fdr_threshold):
    """*df* (the combined IDs) with plex_Area_norm, and each normalized unit's
    RtCurve for the plots.

    *mode* is "run_rt" or "channel_rt" (see the module docstring); with
    None (--norm not given) *df* is returned unchanged.  Otherwise plex_Area_norm is
    plex_Area / 2**factor, empty where plex_Area is missing or 0, for decoys,
    and in units that could not be normalized.  plex_Area is not changed and
    no row is dropped.  References pass Qvalue < *fdr_threshold*.
    """
    if mode is None:
        return df, []
    if mode not in RT_NORM_MODES:
        raise JModError(f"--norm must be one of {', '.join(RT_NORM_MODES)}, not {mode!r}")
    empty = pl.lit(None, dtype=pl.Float64).alias("plex_Area_norm")
    if "time_channel" in df.columns and df["time_channel"].n_unique() > 1:
        logger.info("RT normalization: not applied to timeplex")
        return df.with_columns(empty), []

    # The channel columns that tell channels apart here (a constant one, e.g.
    # silac_channel without SILAC, says nothing)
    channel_columns = [c for c in _CHANNEL_COLUMNS if c in df.columns and df[c].n_unique() > 1]
    if mode == "channel_rt" and not channel_columns:
        logger.info("RT normalization (channel_rt): one channel per run, nothing to compare; not applied")
        return df.with_columns(empty), []
    area = df["plex_Area"].cast(pl.Float64).fill_nan(None).to_numpy()
    valid = ((~df["is_decoy"].fill_null(True)).to_numpy()
             & np.isfinite(area) & (area > 0)
             & df["rt"].cast(pl.Float64).fill_nan(None).is_not_null().to_numpy())
    work = (
        df.select(["run_idx", *channel_columns, "untag_prec",
                   pl.col("rt").cast(pl.Float64), pl.col("Qvalue")])
        .with_columns(
            pl.Series("_row", np.arange(df.height)),
            pl.Series("_valid", valid),
            pl.Series("_log2", np.where(valid, np.log2(np.where(valid, area, 1.0)), np.nan)),
        )
        .filter(pl.col("_valid"))
    )
    candidates = work.filter(pl.col("Qvalue") < fdr_threshold)

    if mode == "channel_rt":
        factor, curves, n_units, skipped = _channels_mode(work, candidates, channel_columns, df.height)
        noun = "(run, channel) units"
    else:
        factor, curves, n_units, skipped = _runs_mode(work, candidates, channel_columns, df.height)
        noun = "runs"
    for message in skipped:
        logger.warning(f"RT normalization: {message}; its plex_Area_norm is left empty")
    logger.info(f"RT normalization ({mode}): {len(curves)} of {n_units} {noun} normalized")

    norm = np.where(np.isnan(factor), np.nan, area / np.exp2(np.nan_to_num(factor)))
    return df.with_columns(pl.Series("plex_Area_norm", norm).fill_nan(None)), curves


def _channels_mode(work, candidates, channel_columns, n_rows):
    """Channels as samples: one curve per (run, channel), each run on its own."""
    factor = np.full(n_rows, np.nan)
    curves, skipped = [], []
    unit_columns = ["run_idx", *channel_columns]
    n_units = work.select(unit_columns).unique().height
    # One value per (unit, precursor): rows of the same precursor in a unit
    # (rare) are taken together
    per_unit = (candidates.group_by(*unit_columns, "untag_prec")
                .agg(pl.col("_log2").median(), pl.col("rt").median()))
    for (run_idx,), run_rows in sorted(work.partition_by("run_idx", as_dict=True).items()):
        channels = run_rows.select(channel_columns).unique().sort(channel_columns)
        if channels.height < 2:
            skipped.append(f"run {run_idx} has one channel, nothing to compare it with")
            continue
        run_values = per_unit.filter(pl.col("run_idx") == run_idx)
        precursors = run_values["untag_prec"].unique().sort()
        matrix = _matrix(run_values, precursors, channels, channel_columns)
        reference, levels = select_references(matrix, "median")
        rts = _matrix(run_values, precursors, channels, channel_columns, value="rt")
        for c, channel in enumerate(channels.iter_rows()):
            unit = (run_idx, *channel)
            observed = reference & ~np.isnan(matrix[:, c])
            unit_rows = run_rows.filter(*[pl.col(col).eq_missing(v) for col, v in zip(channel_columns, channel)])
            curve = _fit_unit(unit, rts[observed, c], matrix[observed, c] - levels[observed],
                              unit_rows, factor, skipped)
            if curve is not None:
                curves.append(curve)
    return factor, curves, n_units, skipped


def _runs_mode(work, candidates, channel_columns, n_rows):
    """Runs as samples: one curve per run, applied to all of its channels."""
    factor = np.full(n_rows, np.nan)
    curves, skipped = [], []
    run_idxs = sorted(work["run_idx"].unique().to_list())
    if len(run_idxs) < 2:
        return factor, curves, len(run_idxs), ["run_rt needs two or more runs"]
    # A run's channels: those with any quantified row in it
    n_channels = work.select("run_idx", *channel_columns).unique().group_by("run_idx").len("_n_channels")
    per_channel = (candidates.group_by("run_idx", *channel_columns, "untag_prec")
                   .agg(pl.col("_log2").median(), pl.col("rt").median()))
    # A precursor's value in a run: the median over its observed channels,
    # where it is observed in at least half of the run's channels
    per_run = (
        per_channel.group_by("run_idx", "untag_prec")
        .agg(pl.col("_log2").median(), pl.col("rt").median(), pl.len().alias("_n_observed"))
        .join(n_channels, on="run_idx")
        .filter(pl.col("_n_observed") * 2 >= pl.col("_n_channels"))
    )
    precursors = per_run["untag_prec"].unique().sort()
    runs = pl.DataFrame({"run_idx": run_idxs}, schema={"run_idx": work.schema["run_idx"]})
    matrix = _matrix(per_run, precursors, runs, ["run_idx"])
    rts = _matrix(per_run, precursors, runs, ["run_idx"], value="rt")
    reference, levels = select_references(matrix, "mean")
    for r, run_idx in enumerate(run_idxs):
        observed = reference & ~np.isnan(matrix[:, r])
        curve = _fit_unit((run_idx,), rts[observed, r], matrix[observed, r] - levels[observed],
                          work.filter(pl.col("run_idx") == run_idx), factor, skipped)
        if curve is not None:
            curves.append(curve)
    return factor, curves, len(run_idxs), skipped


def _matrix(values, precursors, samples, sample_columns, value="_log2"):
    """A precursor x sample matrix of *value* (NaN where missing), rows in the
    order of *precursors*, columns in the order of *samples*' rows."""
    indexed = (values
               .join(pl.DataFrame({"untag_prec": precursors}).with_row_index("_p"), on="untag_prec")
               .join(samples.with_row_index("_s"), on=sample_columns, nulls_equal=True))
    matrix = np.full((len(precursors), samples.height), np.nan)
    matrix[indexed["_p"].to_numpy(), indexed["_s"].to_numpy()] = indexed[value].to_numpy()
    return matrix


def _fit_unit(unit, ref_rt, ref_d, unit_rows, factor, skipped):
    """Fit *unit*'s curve and write its factor into *factor* at its rows."""
    fit = unit_factor(ref_rt, ref_d, unit_rows["rt"].to_numpy())
    if fit is None:
        skipped.append(f"{_unit_name(unit)} has {len(ref_rt)} reference precursors, "
                       f"fewer than the {2 * window_size(len(ref_rt))} needed")
        return None
    row_factor, ref_factor, curve_rt, curve_factor = fit
    factor[unit_rows["_row"].to_numpy()] = row_factor
    return RtCurve(unit, curve_rt, curve_factor, ref_rt, ref_d, ref_factor)


def _unit_name(unit):
    # e.g. "run 3", or "run 3, channel 8"
    if len(unit) == 1:
        return f"run {unit[0]}"
    return f"run {unit[0]}, channel {'·'.join(f'{v:g}' if isinstance(v, float) else str(v) for v in unit[1:])}"
