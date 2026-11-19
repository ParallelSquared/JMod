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

"""Plex-set scoring: a separate model of the run's IDs, one row per plex set,
giving PlexSetScore and PlexSet_Qvalue -- a candidate to replace the precursor
scoring (score_precursors) and the q-values built on it.

A plex set is one untagged precursor in one run: the same peptide searched once
per channel, targets and decoys apart (each channel's target has its own decoy,
and the decoys of a set share their untagged precursor, as the targets do).
untag_prec_Qvalue ranks each set by its best channel's PredVal alone.  This
model sees every channel of the set at once: the measurements of each channel
side by side, and whether the channels agree with each other -- real channels
of one peptide co-elute and share their errors; the channels of a false set are
independent draws and need not.

It is built from the same per-channel measurements the precursor scoring
trains on (fdr_analysis.NOT_FEATURES says which are not), and from nothing the
precursor scoring outputs (FIRST_MODEL_OUTPUTS: PredVal and the q-values from
it), so the two can be compared as alternatives.  Nothing from MS1 quant
either: it quantifies targets alone, so anything it computes would tell the
targets from the decoys.  The decoy channel's rows are left out, as they are
of the precursor scoring's training and q-values: a set seen only in the decoy
channel has no PlexSetScore or PlexSet_Qvalue.

Fitted as the precursor scoring is: cross-validated, grouped by peptide, with
two semi-supervised refits on this model's own q-values.  The q-values use the
run's target/decoy ratio, as untag_prec_Qvalue's do.  Nothing is filtered by
them.
"""

import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl

from src import config
from src.logger import logger

FOLDER = "plex_scoring"
FEATURE_FILE = "plex_set_features.parquet"
FEATURE_IMPORTANCE_FILE = "feature_importance.csv"
IDS_PLOT = "ids_vs_qvalue.png"
SCORE_PLOT = "plex_set_score.png"
SCORE_COLUMNS = ["PlexSetScore", "PlexSet_Qvalue"]

# What a plex set is, as for untag_prec_Qvalue (fdr_analysis.untag_prec_qvalues)
SET_KEYS = ["file_name", "untag_prec", "is_decoy"]
# Carried into the feature table so it stands on its own; not features
_ID_COLUMNS = [*SET_KEYS, "stripped_seq", "untag_prec_Qvalue"]
# The CV groups: a peptide's charges in one fold, as for the precursor scoring
GROUP_COLUMN = "stripped_seq"

# The precursor scoring's outputs, and what is built on them, never features
# here (most are made after this model runs; listed so they never can be)
FIRST_MODEL_OUTPUTS = ["PredVal", "Qvalue", "BestChannel_Qvalue", "untag_prec_Qvalue", "Protein_Qvalue",
                       "ChannelScore", "Channel_Qvalue", "Protein_Channel_Qvalue", *SCORE_COLUMNS,
                       "Protein_PlexSet_Qvalue"]
# The channels of a set are laid side by side in this order, highest first:
# <feature>_rank<i> is the i-th channel's.  By rank, not by channel: the model
# is to learn how many channels agree and how well, not anything about a
# particular channel (its mass offset, how often it is empty), which would also
# not carry over from one plex to another.  A raw match score, not PredVal; ties
# keep the row order.  As many ranks as the run's largest set has channels, up
# to MAX_RANKS; a set with fewer has null in the rest
RANK_BY = "hyperscore"
MAX_RANKS = 16
# Where each channel eluted, and its m/z error: real channels of one peptide
# agree, the channels of a false set need not.  Their range over the set
# (<feature>_range), and each channel's distance from the first-ranked one
# (<feature>_from_best_rank<i>)
AGREEMENT = ["rt", "prec_im", "mz_error"]
# Each measurement summarised over the set's channels (<feature>_<summary>):
# any of mean, median, min, max, std.  RANK_ARRAYS adds each measurement
# channel by channel, by rank (<feature>_rank<i>).  On a 5-plex run (25k sets)
# the summaries alone gave the most sets at 1% FDR: 2,126, against 2,105 with
# the rank arrays too and 1,950 with the rank arrays alone -- a tree splits on
# one column at a time, so it cannot take the minimum over the rank columns
# itself, and hundreds of sparse rank columns give it more noise to fit
SUMMARIES = ["mean", "median", "min", "max"]
RANK_ARRAYS = False
# Cross-channel elution (elution_features): each channel's MS2 coefficient
# trace over the DIA cycles, from the search's per-scan coefficients.  Real
# channels of one peptide co-elute, so their traces rise and fall together;
# the channels of a false set are independent draws.  A correlation needs at
# least this many cycles in the set's traces, and two traces that vary
ELUTION_MIN_CYCLES = 3
# Summaries over the set's channels, or pairs of them, of the elution and
# fragment comparisons
ELUTION_SUMMARIES = ["mean", "median", "min", "max"]
# The decoy channel (decoy_channel_features): these measurements of its match,
# against the real channels'
DECOY_CHANNEL_COMPARED = ["hyperscore", "gof_stats", "scribe_scores", "n_scans", "mean_frag_corr",
                          "frac_int_matched"]
# Cross-channel fragments (fragment_features): a correlation between two
# channels' channel-specific fragment intensities needs this many fragments
# both matched
FRAGMENT_MIN_COMMON = 3

# Too few decoy sets and the q-values are too coarse to mean much
MIN_DECOYS = 100
N_ITERATIONS = 3
TRAIN_Q = 0.01          # a target set this confident is a positive for the refit
PEP_CUT = 0.90          # a target set this decoy-like is relabelled a decoy for it
# Models that take a missing value as it is; a set of one channel has no spread
_NAN_MODELS = ("xg", "rf")


def channel_columns(rows):
    """The per-channel measurements of *rows* the model is built from: the
    precursor scoring's inputs (every numeric column but fdr_analysis.NOT_FEATURES
    and the internal "__" ones), with its predicted_rt in place of rt, and none
    of FIRST_MODEL_OUTPUTS."""
    from src.fdr_analysis import NOT_FEATURES
    excluded = set(NOT_FEATURES) | set(FIRST_MODEL_OUTPUTS) | {"rt"}
    columns = [c for c in rows.columns
               if c not in excluded and not str(c).startswith("__")
               and pd.api.types.is_numeric_dtype(rows[c]) and not pd.api.types.is_bool_dtype(rows[c])]
    return columns + (["predicted_rt"] if {"rt", "rt_error"} <= set(rows.columns) else [])


def placeholder_rules(columns):
    """{column: the rows where its value is one of JMod's placeholders} for the
    columns in *columns*: the channel FDR's rules (channel_fdr._PLACEHOLDERS,
    each checked against a real run), plus sq_mz_error, the square of
    mz_error's -1.  Only the rules whose column, and every column the rule
    reads, are in *columns*."""
    from src.channel_fdr import _PLACEHOLDERS
    rules = {**_PLACEHOLDERS, "sq_mz_error": pl.col("mz_error") == -1}
    return {c: mask for c, mask in rules.items()
            if c in columns and set(mask.meta.root_names()) <= set(columns)}


def set_features(rows, scan_coeffs=None, scan_cycles=None, list_cols=None, label_name=None):
    """One row per plex set of *rows* (pandas, one row per precursor-channel),
    sorted by "__set" (the set's number, rows.groupby(SET_KEYS).ngroup()): its
    _ID_COLUMNS, n_channels, n_channels_ms1 (those with an MS1 peak: an
    mz_error), for every channel_columns measurement its SUMMARIES over the set
    and (RANK_ARRAYS) its value in each channel by rank (RANK_BY), the
    AGREEMENT features and the decoy_channel_features.  With *scan_coeffs* (the
    search's per-scan coefficients) and *scan_cycles* (channel_fdr.ms2_cycles'
    table), the elution_features; with *list_cols* (the apex scans' matched
    fragments, by __fdc_idx) and *label_name* (the mass tag or SILAC label on
    the sequences), the fragment_features.

    The decoy channel's rows are left out of every feature but the
    decoy_channel_ ones (a set seen only in it has no row)."""
    group = rows.groupby(SET_KEYS, sort=False).ngroup().to_numpy()
    real = (~rows["is_decoy_channel"].to_numpy(bool) if "is_decoy_channel" in rows.columns
            else np.ones(len(rows), bool))
    measured = channel_columns(rows)
    agreement = [c for c in AGREEMENT if c in rows.columns and rows.loc[real, c].nunique() > 1]
    rule_columns = [c for mask in placeholder_rules(rows.columns).values() for c in mask.meta.root_names()]
    read = [c for c in dict.fromkeys([*_ID_COLUMNS, *measured, *agreement, "rt", "rt_error", *rule_columns,
                                      "seq", "z", "time_channel", "spec_id", "__fdc_idx"])
            if c in rows.columns]

    def channel_frame(which):
        frame = pl.from_pandas(rows.loc[which, read].reset_index(drop=True)).with_columns(
            pl.Series("__set", group[which]))
        # A placeholder (mz_error's -1 for no MS1 peak, say) is not a
        # measurement: null, so the means, ranges and distances between
        # channels skip it, rather than read it as a real value.  All
        # evaluated on the values as written
        frame = frame.with_columns(pl.when(mask).then(None).otherwise(pl.col(c)).alias(c)
                                   for c, mask in placeholder_rules(frame.columns).items())
        if "predicted_rt" in measured:
            frame = frame.with_columns(predicted_rt=pl.col("rt") - pl.col("rt_error"))
        return frame.with_columns(_rank=pl.col(RANK_BY).fill_null(-np.inf).fill_nan(-np.inf)
                                  .rank("ordinal", descending=True).over("__set"))

    df = channel_frame(real).filter(pl.col("_rank") <= MAX_RANKS)
    n_ranks = int(df["_rank"].max() or 0)
    # The decoy channel: its best row in each set (one, but for a set that
    # spans time channels), in a slot of its own after the real channels'
    decoy = channel_frame(~real).filter(pl.col("_rank") == 1).with_columns(
        pl.lit(n_ranks + 1, dtype=df["_rank"].dtype).alias("_rank"))
    with_decoy = pl.concat([df, decoy], how="diagonal_relaxed") if decoy.height else df

    first = pl.col("_rank") == 1
    # each channel's distance from the first-ranked one, for AGREEMENT
    df = df.with_columns((pl.col(c) - pl.col(c).filter(first).first().over("__set")).abs().alias(f"{c}_from_best")
                         for c in agreement)
    many = pl.len() > 1
    others = ~first
    features = df.group_by("__set").agg(
        *[pl.col(c).first() for c in _ID_COLUMNS if c in df.columns],
        pl.len().alias("n_channels"),
        *([pl.col("mz_error").is_not_null().sum().alias("n_channels_ms1")] if "mz_error" in df.columns else []),
        *[getattr(pl.col(c), stat)().alias(f"{c}_{stat}") for stat in SUMMARIES for c in measured],
        *[pl.when(many).then(pl.col(c).max() - pl.col(c).min()).alias(f"{c}_range") for c in agreement],
        *[getattr(pl.col(f"{c}_from_best").filter(others), stat)().alias(f"{c}_from_best_{stat}")
          for stat in SUMMARIES for c in agreement],
    )
    features = features.join(decoy_channel_features(df, decoy), on="__set", how="left").with_columns(
        pl.col("decoy_channel_seen").fill_null(0))
    if scan_coeffs is not None:
        features = features.join(elution_features(with_decoy, scan_coeffs, scan_cycles, n_ranks),
                                 on="__set", how="left")
    if list_cols is not None and label_name is not None:
        features = features.join(fragment_features(with_decoy, list_cols, label_name, n_ranks),
                                 on="__set", how="left")
    if RANK_ARRAYS:
        # each measurement channel by channel, side by side
        values = [*measured, *[f"{c}_from_best" for c in agreement]]
        wide = df.pivot(on="_rank", index="__set", values=values, sort_columns=True)
        wide = wide.rename({(f"{v}_{i}" if len(values) > 1 else str(i)): f"{v}_rank{i}"
                            for v in values for i in range(1, n_ranks + 1)})
        # the first-ranked channel's distance from itself is 0 by definition
        wide = wide.drop([f"{c}_from_best_rank1" for c in agreement])
        features = features.join(wide, on="__set", how="left")
    return features.sort("__set")


def decoy_channel_features(real, decoy):
    """How the decoy channel -- the tag channel left empty in the sample, but
    searched like any other -- compares with the set's real channels, one row
    per set that has it in *decoy* (set_features' frames).  In a real set it is
    empty, so whatever it matched is noise next to the real channels; in a false
    set every channel is noise, and it is as good as any of them.

      - decoy_channel_seen: 1 (0 in set_features for a set without it)
      - decoy_channel_outscores: the share of the real channels it beats on RANK_BY
      - decoy_channel_<m>_from_best and _from_median: its DECOY_CHANNEL_COMPARED
        measurements less the first-ranked real channel's and the real
        channels' median
    The elution and fragment features compare it too (decoy_channel_elution_,
    decoy_channel_frag_)."""
    compared = [c for c in DECOY_CHANNEL_COMPARED if c in decoy.columns]
    if decoy.height == 0:
        return pl.DataFrame({"__set": pl.Series([], dtype=real["__set"].dtype),
                             "decoy_channel_seen": pl.Series([], dtype=pl.Int32)})
    reference = real.group_by("__set").agg(
        *[pl.col(c).filter(pl.col("_rank") == 1).first().alias(f"__best_{c}") for c in compared],
        *[pl.col(c).median().alias(f"__median_{c}") for c in compared])
    outscores = (real.select("__set", pl.col(RANK_BY).alias("__real"))
                 .join(decoy.select("__set", pl.col(RANK_BY).alias("__decoy")), on="__set", how="inner")
                 .group_by("__set").agg((pl.col("__real") < pl.col("__decoy")).mean()
                                        .alias("decoy_channel_outscores")))
    return (decoy.select("__set", *[pl.col(c).alias(f"__decoy_{c}") for c in compared])
            .join(reference, on="__set", how="inner")
            .join(outscores, on="__set", how="left")
            .with_columns(
                pl.lit(1, dtype=pl.Int32).alias("decoy_channel_seen"),
                *[(pl.col(f"__decoy_{c}") - pl.col(f"__best_{c}")).alias(f"decoy_channel_{c}_from_best")
                  for c in compared],
                *[(pl.col(f"__decoy_{c}") - pl.col(f"__median_{c}")).alias(f"decoy_channel_{c}_from_median")
                  for c in compared])
            .select("__set", "decoy_channel_seen", "decoy_channel_outscores",
                    *[f"decoy_channel_{c}_from_{ref}" for c in compared for ref in ("best", "median")]))


def channel_traces(ranked, scan_coeffs, scan_cycles):
    """Each channel's MS2 coefficient trace: (__set, _rank, cycle, coeff), one
    row per cycle in which it has a coefficient, sorted.  *ranked* is
    set_features' frame (seq, z, __set, _rank, per channel); a channel seen
    twice in one cycle (overlapping windows) takes its larger coefficient."""
    keys = _scan_keys(ranked, scan_coeffs)
    channels = ranked.select(*_as_keys(keys), "__set", "_rank")
    return (scan_coeffs.select(*_as_keys(keys), pl.col("spec_id").cast(pl.Int64), "coeff")
            .join(channels, on=keys, how="inner")
            .join(_cycles(scan_cycles), on="spec_id", how="inner")
            .group_by("__set", "_rank", "cycle").agg(pl.col("coeff").max())
            .sort("__set", "_rank", "cycle"))


def _scan_keys(ranked, scan_coeffs):
    """What names one precursor-channel in the per-scan coefficients."""
    return ["seq", "z"] + (["time_channel"] if "time_channel" in ranked.columns
                           and "time_channel" in scan_coeffs.columns else [])


def _as_keys(keys):
    return [pl.col("z").cast(pl.Int64)] + [pl.col(c) for c in keys if c != "z"]


def _cycles(scan_cycles):
    return scan_cycles.select(pl.col("spec_id").cast(pl.Int64), "cycle")


def _correlation(a, b):
    """Pearson r of two traces, or NaN when either is flat."""
    a, b = a - a.mean(), b - b.mean()
    denominator = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denominator) if denominator > 0 else np.nan


def _overlap(a, b):
    """The share of the cycles in which either has a coefficient where both do."""
    either = (a > 0) | (b > 0)
    return float(((a > 0) & (b > 0)).sum() / either.sum()) if either.any() else np.nan


def _blocks(sets):
    """(start, end) of each run of one value in the sorted *sets*."""
    starts = np.flatnonzero(np.r_[True, sets[1:] != sets[:-1]]) if sets.size else np.zeros(0, np.int64)
    return starts, np.r_[starts[1:], sets.size]


def _summarise(out, prefix, values, ranks_from):
    """Into *out*: *values* (sets x ranks, NaN where undefined) as its
    ELUTION_SUMMARIES over the channels (<prefix>_<summary>) and, with
    RANK_ARRAYS, by rank from *ranks_from* on (<prefix>_rank<i>)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # all-NaN rows: nothing to compare
        for stat in ELUTION_SUMMARIES:
            out[f"{prefix}_{stat}"] = getattr(np, f"nan{stat}")(values, axis=1)
    if RANK_ARRAYS:
        for i in range(ranks_from, values.shape[1] + 1):
            out[f"{prefix}_rank{i}"] = values[:, i - 1]


def _frame(sets, out, ranked):
    return (pl.DataFrame({"__set": sets, **out})
            .with_columns(pl.col("__set").cast(ranked["__set"].dtype))
            .with_columns(pl.col(c).fill_nan(None) for c in out))


def _ratio_sd(a, b):
    """The spread (SD) of log(b / a) over the cycles both have a coefficient
    in, or NaN with fewer than two: real channels of one peptide keep their
    abundance ratio across the peak, whatever its height."""
    both = (a > 0) & (b > 0)
    return float(np.std(np.log(b[both] / a[both]), ddof=1)) if both.sum() >= 2 else np.nan


def elution_features(ranked, scan_coeffs, scan_cycles, n_ranks):
    """Cross-channel elution and apex agreement, one row per set (__set) with a
    trace: its real channels' MS2 coefficient traces (channel_traces) over the
    cycles any of them has one in, 0 where a channel has none (JMod writes no
    zero coefficients).  *ranked* holds the real channels, ranks 1 to
    *n_ranks*, and the decoy channel, if any, at rank n_ranks + 1.

      - elution_cycles: those cycles; elution_shared_cycles: the share of them
        in which more than one channel has a coefficient
      - elution_corr_<summary>: the ELUTION_SUMMARIES of every pair's Pearson r
      - per channel: r with the first-ranked channel (corr_to_best); the share
        of the cycles either has a coefficient in where both do (overlap_to_best,
        which says something even of traces too short to correlate); r with the
        sum of the set's other channels (corr_to_rest); the spread of its log
        ratio to the first-ranked channel over the cycles both are seen in
        (ratio_sd_to_best, _ratio_sd)
      - apex agreement.  Each channel's apex is its own: the scan get_large_prec
        chose for it (spec_id), independently of the other channels.  The set's
        apex is the peak of its summed trace, where all its channels together
        are highest.  Per channel: how many cycles its apex is from the
        first-ranked channel's (apex_from_best) and from the set's
        (apex_from_set), and its coefficient at the set's apex over its own
        largest (height_at_set_apex: 1 if it peaks there, 0 if it is not seen
        there).  apex_agree: the share of the channels whose apex is in the
        cycle most of them share
      - the decoy channel, against the real ones, over the cycles any channel
        is seen in: decoy_channel_elution_corr_to_rest, _overlap_to_best,
        _ratio_sd_to_best, _apex_from_set and _height_at_set_apex

    Each per-channel value as its ELUTION_SUMMARIES over the channels
    (elution_<name>_<summary>) and, with RANK_ARRAYS, by rank.  A correlation
    is null with fewer than ELUTION_MIN_CYCLES cycles, or a flat trace.
    """
    traces = channel_traces(ranked, scan_coeffs, scan_cycles)
    sets = traces["__set"].to_numpy()
    ranks = traces["_rank"].to_numpy().astype(np.int64) - 1
    cycles = traces["cycle"].to_numpy()
    coeffs = traces["coeff"].to_numpy().astype(float)
    starts, ends = _blocks(sets)
    n_sets = starts.size
    slot = n_ranks                       # the decoy channel's row

    # each channel's own apex, as a cycle: sets x (ranks + the decoy channel)
    apex = np.full((n_sets, n_ranks + 1), np.nan)
    apexes = (ranked.select("__set", "_rank", pl.col("spec_id").cast(pl.Int64))
              .join(_cycles(scan_cycles), on="spec_id", how="inner")
              if "spec_id" in ranked.columns else pl.DataFrame({"__set": [], "_rank": [], "cycle": []}))
    row_of = {s: k for k, s in enumerate(sets[starts])}
    for s, r, c in zip(apexes["__set"].to_numpy(), apexes["_rank"].to_numpy(), apexes["cycle"].to_numpy()):
        k = row_of.get(s)
        if k is not None:
            apex[k, int(r) - 1] = c

    out = {"elution_cycles": np.full(n_sets, np.nan), "elution_shared_cycles": np.full(n_sets, np.nan),
           **{f"elution_corr_{stat}": np.full(n_sets, np.nan) for stat in ELUTION_SUMMARIES},
           "elution_apex_agree": np.full(n_sets, np.nan),
           **{f"decoy_channel_elution_{name}": np.full(n_sets, np.nan)
              for name in ("corr_to_rest", "overlap_to_best", "ratio_sd_to_best", "apex_from_set",
                           "height_at_set_apex")}}
    per_channel = {name: np.full((n_sets, n_ranks), np.nan)
                   for name in ("corr_to_best", "overlap_to_best", "corr_to_rest", "ratio_sd_to_best",
                                "apex_from_best", "apex_from_set", "height_at_set_apex")}
    for k, (start, end) in enumerate(zip(starts, ends)):
        cycle_index, at = np.unique(cycles[start:end], return_inverse=True)
        everything = np.zeros((n_ranks + 1, cycle_index.size))
        everything[ranks[start:end], at] = coeffs[start:end]
        # the real channels over their own cycles: the decoy channel's must not
        # stretch them
        own_cycles = (everything[:n_ranks] > 0).any(axis=0)
        if not own_cycles.any():
            continue
        matrix = everything[:n_ranks, own_cycles]
        seen = matrix > 0
        present = np.flatnonzero(seen.any(axis=1))
        out["elution_cycles"][k] = matrix.shape[1]
        out["elution_shared_cycles"][k] = (seen.sum(axis=0) > 1).mean()

        # apexes
        total = matrix.sum(axis=0)
        set_apex = cycle_index[own_cycles][np.argmax(total)]
        own = apex[k, :n_ranks]
        has_apex = np.isfinite(own)
        per_channel["apex_from_set"][k, has_apex] = np.abs(own[has_apex] - set_apex)
        per_channel["height_at_set_apex"][k, present] = (matrix[present, np.argmax(total)]
                                                         / matrix[present].max(axis=1))
        if has_apex.sum() > 1:
            _, counts = np.unique(own[has_apex], return_counts=True)
            out["elution_apex_agree"][k] = counts.max() / has_apex.sum()
            if has_apex[0]:
                later = has_apex.copy()
                later[0] = False
                per_channel["apex_from_best"][k, later] = np.abs(own[later] - own[0])

        # the decoy channel, over every cycle any channel is seen in
        decoy = everything[slot]
        if decoy.any():
            real_total = everything[:n_ranks].sum(axis=0)
            if cycle_index.size >= ELUTION_MIN_CYCLES:
                out["decoy_channel_elution_corr_to_rest"][k] = _correlation(decoy, real_total)
            out["decoy_channel_elution_overlap_to_best"][k] = _overlap(everything[0], decoy)
            out["decoy_channel_elution_ratio_sd_to_best"][k] = _ratio_sd(everything[0], decoy)
            at_set_apex = np.flatnonzero(cycle_index == set_apex)[0]
            out["decoy_channel_elution_height_at_set_apex"][k] = decoy[at_set_apex] / decoy.max()
            if np.isfinite(apex[k, slot]):
                out["decoy_channel_elution_apex_from_set"][k] = abs(apex[k, slot] - set_apex)

        if present.size < 2:
            continue
        long_enough = matrix.shape[1] >= ELUTION_MIN_CYCLES
        if long_enough:
            pairs = np.asarray([_correlation(matrix[a], matrix[b])
                                for n, a in enumerate(present) for b in present[n + 1:]])
            if np.isfinite(pairs).any():
                for stat in ELUTION_SUMMARIES:
                    out[f"elution_corr_{stat}"][k] = getattr(np, f"nan{stat}")(pairs)
        for r in present:
            if long_enough:
                per_channel["corr_to_rest"][k, r] = _correlation(matrix[r], total - matrix[r])
            if r > 0 and seen[0].any():
                per_channel["overlap_to_best"][k, r] = _overlap(matrix[0], matrix[r])
                per_channel["ratio_sd_to_best"][k, r] = _ratio_sd(matrix[0], matrix[r])
                if long_enough:
                    per_channel["corr_to_best"][k, r] = _correlation(matrix[0], matrix[r])
    for name, values in per_channel.items():
        _summarise(out, f"elution_{name}", values, 2 if name.endswith("best") else 1)
    return _frame(sets[starts], out, ranked)


def fragment_features(ranked, list_cols, label_name, n_ranks):
    """Cross-channel fragment agreement at the apex, one row per set (__set).
    *ranked* holds the real channels, ranks 1 to *n_ranks*, and the decoy
    channel, if any, at rank n_ranks + 1.

    Each channel's matched fragments at its apex scan (list_cols: frag_names,
    obs_int, frag_int, frag_errors), only the channel-specific ones: those
    that span a tag site, so their m/z moves from channel to channel
    (channel_fdr.channel_specific_mask, sites read off the searched sequence,
    channel_fdr.tag_sites).  The rest have the same m/z in every channel, so
    every channel matches the same peak and agrees with the others trivially.
    A fragment of one channel stands for the same fragment of another when its
    identity (ion type, index, charge, loss, isotope: the frag code) is the
    same.

      - frag_corr_<summary>: the ELUTION_SUMMARIES, over every pair of
        channels, of the Pearson r of the log intensities of the fragments both
        matched (at least FRAGMENT_MIN_COMMON).  Corresponding fragments sit at
        nearly the same m/z, and intensity depends on where in the spectrum a
        peak is, so even a false set's channels correlate (a median of 0.83 on
        a 5-plex run, against 0.90 for real sets)
      - frag_resid_corr_<summary>: the same r of log(observed / library
        intensity).  It runs the other way (0.88 for false sets, 0.81 for real
        ones): where the observed intensities do not follow the library, the
        residuals are the library's pattern turned over, and every channel
        shares it
      - frag_overlap_<summary>: the same of the share of the channel-specific
        fragments either matched that both did
      - frag_error_range: the spread over the channels of their signed median
        fragment mass error.  Real channels of one peptide share the run's
        calibration; a false match's errors are anywhere within the tolerance
      - per channel against the first-ranked one: frag_corr_to_best,
        frag_resid_corr_to_best, frag_overlap_to_best and frag_error_from_best
        (the difference in median error), as their ELUTION_SUMMARIES and, with
        RANK_ARRAYS, by rank
      - frag_n_specific_<summary>: each channel's channel-specific fragments
        matched
      - the decoy channel, against the first-ranked real one:
        decoy_channel_frag_corr_to_best, _resid_corr_to_best,
        _overlap_to_best and _error_from_best
    """
    from src.channel_fdr import channel_specific_mask, first_of_each, tag_sites
    frags = (ranked.select("__set", "_rank", "seq", "__fdc_idx")
             .join(list_cols.select("__fdc_idx", "frag_names", "obs_int", "frag_int", "frag_errors"),
                   on="__fdc_idx", how="left", maintain_order="left")
             .sort("__set", "_rank"))
    sets = frags["__set"].to_numpy()
    starts, ends = _blocks(sets)
    n_sets = starts.size
    ranks = frags["_rank"].to_numpy().astype(np.int64) - 1
    slot = n_ranks                       # the decoy channel

    # each row's channel-specific fragments: {code: (log observed, log
    # library)}, and their median mass error
    specific, median_error = [], []
    for seq, codes, obs, lib, err in zip(frags["seq"], frags["frag_names"], frags["obs_int"],
                                         frags["frag_int"], frags["frag_errors"]):
        sites, n_tags, n_residues = tag_sites(seq, label_name)
        if codes is None or n_tags == 0 or len(codes) == 0:
            specific.append({})
            median_error.append(np.nan)
            continue
        codes, values, _ = first_of_each(np.asarray(codes, dtype=np.int32),
                                         np.column_stack([np.asarray(obs, dtype=float), np.asarray(lib, dtype=float),
                                                          np.asarray(err, dtype=float)]))
        keep = (channel_specific_mask(codes, n_residues, sites[0], sites[-1])
                & (values[:, 0] > 0) & (values[:, 1] > 0))
        specific.append(dict(zip(codes[keep].tolist(), np.log(values[keep, :2]).tolist())))
        median_error.append(float(np.median(values[keep, 2])) if keep.any() else np.nan)

    def compare(a, b):
        """(r of the log intensities, r of their residuals from the library,
        the overlap) of two channels' fragments."""
        common = sorted(a.keys() & b.keys())
        either = len(a.keys() | b.keys())
        overlap = len(common) / either if either else np.nan
        if len(common) < FRAGMENT_MIN_COMMON:
            return np.nan, np.nan, overlap
        x, y = np.array([a[c] for c in common]), np.array([b[c] for c in common])
        return _correlation(x[:, 0], y[:, 0]), _correlation(x[:, 0] - x[:, 1], y[:, 0] - y[:, 1]), overlap

    compared = ("corr_to_best", "resid_corr_to_best", "overlap_to_best", "error_from_best")
    out = {**{f"frag_{name}_{stat}": np.full(n_sets, np.nan)
              for name in ("corr", "resid_corr", "overlap") for stat in ELUTION_SUMMARIES},
           "frag_error_range": np.full(n_sets, np.nan),
           **{f"decoy_channel_frag_{name}": np.full(n_sets, np.nan) for name in compared}}
    per_channel = {name: np.full((n_sets, n_ranks), np.nan) for name in (*compared, "n_specific")}
    for k, (start, end) in enumerate(zip(starts, ends)):
        channel = {ranks[i]: specific[i] for i in range(start, end)}
        error = {ranks[i]: median_error[i] for i in range(start, end)}
        order = sorted(r for r in channel if r < slot)
        for r in order:
            per_channel["n_specific"][k, r] = len(channel[r])
        errors = np.array([error[r] for r in order], dtype=float)
        if np.isfinite(errors).sum() > 1:
            out["frag_error_range"][k] = np.nanmax(errors) - np.nanmin(errors)
        pairs = [compare(channel[a], channel[b]) for n, a in enumerate(order) for b in order[n + 1:]]
        if pairs:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                for name, values in zip(("corr", "resid_corr", "overlap"), np.array(pairs, dtype=float).T):
                    for stat in ELUTION_SUMMARIES:
                        out[f"frag_{name}_{stat}"][k] = getattr(np, f"nan{stat}")(values)
        if 0 not in channel:
            continue
        against = [r for r in order if r > 0] + ([slot] if slot in channel else [])
        for r in against:
            values = (*compare(channel[0], channel[r]), abs(error[r] - error[0]))
            for name, value in zip(compared, values):
                if r == slot:
                    out[f"decoy_channel_frag_{name}"][k] = value
                else:
                    per_channel[name][k, r] = value
    for name, values in per_channel.items():
        _summarise(out, f"frag_{name}", values, 2 if name.endswith("best") else 1)
    return _frame(sets[starts], out, ranked)

def model_columns(features):
    """The columns of the plex-set feature table the model is fitted on."""
    return [c for c in features.columns if c not in ["__set", *_ID_COLUMNS, *SCORE_COLUMNS]]


def fit_plex_scoring(features, columns, target_decoy_ratio, model_type=None,
                     n_iterations=N_ITERATIONS, train_q=TRAIN_Q, pep_cut=PEP_CUT):
    """PlexSetScore and PlexSet_Qvalue for every row of *features* (set_features'
    table), fitted on its *columns*.  Returns (score, q-values, feature
    importance), aligned with the rows; or None, with a warning, when there are
    too few decoy sets or no target ones."""
    from src.channel_fdr import feature_importance, model_matrix
    from src.fdr_analysis import estimate_pep, score_model, target_decoy_qvalues
    is_decoy = features["is_decoy"].to_numpy().astype(bool)
    if is_decoy.sum() < MIN_DECOYS or is_decoy.all():
        logger.warning(f"Plex-set scoring: {int(is_decoy.sum())} decoy and {int((~is_decoy).sum())} target "
                       f"plex sets; needs at least {MIN_DECOYS} decoy ones and a target. No PlexSet_Qvalue")
        return None
    groups = features[GROUP_COLUMN].to_numpy()
    X, dead = model_matrix(features, columns)
    model_type = model_type or config.score_model
    if model_type not in _NAN_MODELS:
        logger.warning(f"Plex-set scoring: the {model_type!r} model cannot take missing features; using 'xg'")
        model_type = "xg"
    left_out = f" ({len(dead)} left out, no information on this run: {', '.join(dead)})" if dead else ""
    logger.info(f"{X.shape[1]} plex-set features{left_out}")
    y = (~is_decoy).astype(int)

    score = q = model = None
    for iteration in range(n_iterations):
        logger.info(f"  scoring iteration {iteration + 1}/{n_iterations}")
        fitter = score_model(model_type)
        if iteration == 0:
            score = fitter.run_model(X, y, groups=groups)
        else:
            mined = ~is_decoy & (estimate_pep(score, is_decoy) >= pep_cut)
            keep = is_decoy | (~mined & (q <= train_q)) | mined
            y_refit = np.where(mined, 0, y)
            if np.unique(y_refit[keep]).size < 2:
                logger.warning(f"Plex-set scoring: iteration {iteration + 1} would train on one class; "
                               f"keeping iteration {iteration}")
                break
            logger.info(f"    Kept {int(keep.sum())}/{keep.size} sets, relabeled {int(mined.sum())} as decoys")
            score = fitter.run_model_filtered(X, y, keep, y_refit, groups=groups)
        model = fitter
        q = target_decoy_qvalues(score, is_decoy, target_decoy_ratio)
    model.columns = list(X.columns)
    return score, q, feature_importance(model)


def run_plex_scoring(rows, target_decoy_ratio, results_folder, fdr_threshold, scan_coeffs=None,
                     scan_cycles=None, list_cols=None, label_name=None):
    """The plex-set scoring for one run, into <results_folder>/plex_scoring/:
    FEATURE_FILE (the feature table, with PlexSetScore and PlexSet_Qvalue),
    FEATURE_IMPORTANCE_FILE and the plots.

    *rows* is the run's scored precursor-channels (pandas, with PredVal,
    Qvalue and untag_prec_Qvalue); *scan_coeffs* and *scan_cycles*, when
    given, add the elution features, and *list_cols* and *label_name* the
    fragment features (set_features).  Returns (PlexSetScore, PlexSet_Qvalue), two
    Series aligned with *rows*: every row gets its set's, the decoy channel's
    rows too; NaN for a set seen only in the decoy channel, and everywhere when
    nothing was fitted.
    """
    logger.info("")
    logger.info("Plex-Set Scoring")
    group = rows.groupby(SET_KEYS, sort=False).ngroup().to_numpy()
    features = set_features(rows, scan_coeffs, scan_cycles, list_cols, label_name)
    folder = os.path.join(results_folder, FOLDER)
    os.makedirs(folder, exist_ok=True)

    score_by_set = np.full(group.max() + 1 if group.size else 0, np.nan)
    q_by_set = score_by_set.copy()
    fitted = fit_plex_scoring(features, model_columns(features), target_decoy_ratio) if features.height else None
    if fitted is not None:
        score, q, importance = fitted
        sets = features["__set"].to_numpy()
        score_by_set[sets], q_by_set[sets] = score, q
        features = features.with_columns(pl.Series("PlexSetScore", score, dtype=pl.Float64),
                                         pl.Series("PlexSet_Qvalue", q, dtype=pl.Float64))
        importance.to_csv(os.path.join(folder, FEATURE_IMPORTANCE_FILE))
        from src.channel_fdr_plots import plot_feature_importance
        plot_feature_importance(importance, folder)
        plot_ids_vs_qvalue(features, folder, fdr_threshold)
        plot_score(features, folder)
        report_counts(features, fdr_threshold)
    features.drop("__set").write_parquet(os.path.join(folder, FEATURE_FILE))
    return (pd.Series(score_by_set[group], index=rows.index),
            pd.Series(q_by_set[group], index=rows.index))


def protein_qvalues(ids, target_decoy_ratio):
    """Protein_PlexSet_Qvalue for every row of the run's IDs (pandas): its
    protein's q-value by the plex-set model.

    Each protein, targets and decoys apart (a decoy shares its target's protein
    name), is scored by the best PlexSetScore of its plex sets in the real
    channels, and ranked with JMod's estimator (fdr_analysis.
    target_decoy_qvalues) and the run's *target_decoy_ratio* -- as
    fdr_analysis.compute_protein_FDR ranks them by the best PredVal for
    Protein_Qvalue, so the two are alternatives.  Unlike it, no precursor
    q-value is required first: that would tie the proteins to the precursor
    scoring's IDs.  Ranked in (protein, target before decoy) order, so ties
    fall the same way every run.

    Returns one value per row, in its order and with its index: its protein's
    q-value for the target rows of the real channels; null for the library
    decoys, the decoy channel, a row without a protein, and everywhere when
    there is no PlexSetScore.
    """
    from src.fdr_analysis import target_decoy_qvalues
    q = pd.Series(np.nan, index=ids.index)
    if "PlexSetScore" not in ids.columns:
        return q
    real = (~ids["is_decoy_channel"].astype(bool) if "is_decoy_channel" in ids.columns
            else pd.Series(True, index=ids.index))
    scored = ids[real & ids["protein"].notna() & ids["PlexSetScore"].notna()]
    if scored.empty:
        return q
    best = (scored.groupby(["protein", "is_decoy"], sort=True)["PlexSetScore"].max().reset_index())
    best["q"] = target_decoy_qvalues(best["PlexSetScore"].to_numpy(), best["is_decoy"].to_numpy(bool),
                                     target_decoy_ratio)
    by_protein = best[~best["is_decoy"].astype(bool)].set_index("protein")["q"]
    targets = real & ~ids["is_decoy"].astype(bool)
    q[targets] = ids.loc[targets, "protein"].map(by_protein).to_numpy(dtype=float)
    return q


def log_protein_counts(ids, fdr=0.01):
    """TEMPORARY: one log line of the run's proteins at *fdr* by the plex-set
    model (a precursor at PlexSet_Qvalue and the protein at
    Protein_PlexSet_Qvalue) and by the best channel (BestChannel_Qvalue and
    Protein_Qvalue, as the ID table's "best chan Q"): summed over the real
    channels, and unique."""
    real = ~ids["is_decoy_channel"].astype(bool) if "is_decoy_channel" in ids.columns else True
    targets = ids[~ids["is_decoy"].astype(bool) & real & ids["protein"].notna()]
    keys = [c for c in ("file_name", "channel", "silac_channel", "time_channel") if c in targets.columns]
    counts = []
    for label, precursor, protein in (("plex set Q", "PlexSet_Qvalue", "Protein_PlexSet_Qvalue"),
                                      ("best chan Q", "BestChannel_Qvalue", "Protein_Qvalue")):
        passing = targets[(targets[precursor] < fdr) & (targets[protein] < fdr)]
        summed = len(passing.drop_duplicates(subset=keys + ["protein"]))
        counts.append(f"{label} {summed:,} summed over channels, {passing['protein'].nunique():,} unique")
    logger.info(f"Proteins at {fdr:.0%} FDR: " + "; ".join(counts))


def set_counts(features, fdr_threshold):
    """{q-value column: target plex sets at or below *fdr_threshold*}, for
    untag_prec_Qvalue and PlexSet_Qvalue, over the sets of *features*."""
    targets = features.filter(~pl.col("is_decoy"))
    return {c: int((targets[c] <= fdr_threshold).sum())
            for c in ("untag_prec_Qvalue", "PlexSet_Qvalue") if c in targets.columns}


def report_counts(features, fdr_threshold):
    counts = set_counts(features, fdr_threshold)
    logger.info(f"Plex sets at {fdr_threshold:g} FDR: "
                + ", ".join(f"{n} by {column}" for column, n in counts.items()))


# The x axis of IDS_PLOT runs from 0 to this
Q_RANGE = 0.1


def plot_ids_vs_qvalue(features, folder, fdr_threshold=0.01):
    """Target plex sets at or below each q-value, by untag_prec_Qvalue (best
    channel) and by PlexSet_Qvalue (the set model), with the threshold marked.
    Into *folder*/IDS_PLOT."""
    targets = features.filter(~pl.col("is_decoy"))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for column, label in (("untag_prec_Qvalue", "best channel (untag_prec_Qvalue)"),
                          ("PlexSet_Qvalue", "plex-set model (PlexSet_Qvalue)")):
        if column not in targets.columns:
            continue
        q = targets[column].drop_nulls().drop_nans().to_numpy()
        shown = np.sort(q[q <= Q_RANGE])
        n_at = int((q <= fdr_threshold).sum())
        ax.step(np.append(shown, Q_RANGE), np.append(np.arange(1, shown.size + 1), shown.size),
                where="post", label=f"{label}: {n_at} at {fdr_threshold:g}")
    ax.axvline(fdr_threshold, color="grey", linestyle=":", linewidth=1)
    ax.set_xlim(0, Q_RANGE)
    ax.set_xlabel("q-value")
    ax.set_ylabel("Target plex sets at or below it")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("Plex sets vs q-value")
    fig.tight_layout()
    _save(fig, os.path.join(folder, IDS_PLOT))


def plot_score(features, folder, bins=50):
    """PlexSetScore of the target and decoy plex sets.  Into *folder*/SCORE_PLOT."""
    score = features["PlexSetScore"].to_numpy()
    is_decoy = features["is_decoy"].to_numpy().astype(bool)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    edges = np.histogram_bin_edges(score, bins)
    ax.hist(score[~is_decoy], edges, alpha=0.5, label=f"Target sets ({int((~is_decoy).sum())})")
    ax.hist(score[is_decoy], edges, alpha=0.5, label=f"Decoy sets ({int(is_decoy.sum())})")
    ax.set_xlabel("PlexSetScore")
    ax.set_ylabel("Plex sets")
    ax.legend()
    fig.tight_layout()
    _save(fig, os.path.join(folder, SCORE_PLOT))


def _save(fig, path):
    try:
        fig.savefig(path, dpi=300)
    except OSError as error:
        # e.g. the file is open in a viewer on Windows
        logger.warning(f"Plex-set scoring: could not write {path} ({error})")
    finally:
        plt.close(fig)
