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

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import polars as pl

from src.logger import logger
from src.utils.misc_functions import datestamped

# Chart colours: two validated categorical slots, plus recessive greys for
# text, reference lines and gridlines
_TARGET_COLOR = "#2a78d6"
_DECOY_COLOR = "#eb6834"
_TEXT_COLOR = "#0b0b0b"
_MUTED_COLOR = "#52514e"
_GRID_COLOR = "#e4e3df"


def combine_runs(completed_run_folders, experiment_dir, target_decoy_ratio, fdr_threshold):
    """Combine the completed runs into one table in the experiment folder, and plot it.

    *completed_run_folders* maps each completed run's index to its results
    folder.  Writes grouped_results.parquet: every run's IDs that pass the
    run-level FDR, with a global q-value per untagged precursor.  Nothing is
    filtered on the global q-value; that is left to the user.
    """
    logger.info("")
    logger.info(f"Combining the results of {len(completed_run_folders)} run(s)")
    df, untag_prec_global_qs = collect_results(completed_run_folders, target_decoy_ratio, fdr_threshold)

    results_path = datestamped(os.path.join(experiment_dir, "grouped_results.parquet"))
    df.write_parquet(results_path)
    logger.info(f"Combined results written to {os.path.abspath(results_path)}")

    run_idxs = sorted(completed_run_folders)
    plot_lost_to_global_q(df, len(run_idxs), fdr_threshold, experiment_dir)
    if len(run_idxs) >= 2:
        plot_best_score_run(untag_prec_global_qs, run_idxs, experiment_dir)


def collect_results(folder_path_dict, target_decoy_ratio, fdr_threshold):
    """Every run's IDs, with a global q-value for each untagged precursor.

    The global q-value ranks each untagged precursor by its best PredVal across
    runs and channels, and counts targets and decoys as the per-run q-value
    does.  Returns (the run-level IDs that pass *fdr_threshold*, sorted by
    precursor; one row per untagged precursor with its best score, the run it
    came from and its global q-value).
    """
    df = pl.concat([
        pl.scan_parquet(os.path.join(folder_path, "outputs", "all_IDs_filtered.parquet"))
        .with_columns(pl.lit(run_idx).alias("run_idx"))
        for run_idx, folder_path in folder_path_dict.items()
    ], how="vertical_relaxed")  # a column's type can differ between runs (e.g. channel int vs float)

    df = (
        df
        .select(["run_idx", "file_name", "protein", "seq", "z", "channel", "silac_channel",
                 "PredVal", "Qvalue", "BestChannel_Qvalue", "Protein_Qvalue",
                 "plex_Area", "stripped_seq", "untag_seq", "untag_prec",
                 "mz", "rt", "is_decoy"])  #TODO need to add prec_im, coeff
        .with_columns(
            (pl.col("seq") + pl.lit("_") + pl.col("z").cast(pl.Int16).cast(pl.String)).alias("Prec")
        )
    )

    untag_prec_scores = (
        df.group_by("untag_prec")  ##across run, channel, SILAC channel
        .agg(
            pl.col("PredVal").max().alias("MaxPredVal"),
            pl.col("is_decoy").first().alias("is_decoy"), ##This assumes that no target decoy collisions have made it this far
            # Ties go to the earliest run
            pl.col("run_idx").sort_by(["PredVal", "run_idx"], descending=[True, False]).first().alias("BestRun"),
            pl.col("run_idx").n_unique().alias("n_scored_runs"),
        )
    )

    untag_prec_global_qs = (
        untag_prec_scores
        .sort("MaxPredVal", descending=True)
        .with_columns(
            (~pl.col("is_decoy")).cast(pl.Int64).cum_sum().alias("n_target"),
            pl.col("is_decoy").cast(pl.Int64).cum_sum().alias("n_decoy"),
        )
        # Tied scores are a single threshold step: give every precursor in a tie
        # the counts from its end, so the q-value doesn't depend on row order
        .with_columns(
            pl.col("n_target").max().over("MaxPredVal"),
            pl.col("n_decoy").max().over("MaxPredVal"),
        )
    )

    untag_prec_global_qs = (
        untag_prec_global_qs
        .with_columns(
            # As the per-run q-value (fdr_analysis.score_precursors)
            ((1 + pl.col("n_decoy")) / pl.col("n_target") * target_decoy_ratio).alias("FDR")
        )
        .with_columns(
            pl.col("FDR")
            .reverse()
            .cum_min()
            .reverse()
            .alias("untag_prec_Global_Qvalue")
        )
        .collect()
    )

    df = (
        df
        .filter(pl.col("BestChannel_Qvalue") < fdr_threshold)
        .filter(pl.col("is_decoy") == False)
        .join(
            untag_prec_global_qs.lazy().select(["untag_prec", "untag_prec_Global_Qvalue"]),
            on="untag_prec",
            how="left",
        )
        .sort(["untag_prec", "Prec", "run_idx"])
        .collect()
    )

    return df, untag_prec_global_qs


def plot_lost_to_global_q(df, n_runs, fdr_threshold, experiment_dir):
    """Column chart of the precursors the global q-value would remove, by the
    number of runs they pass the run-level FDR in."""
    per_precursor = (
        df.group_by("untag_prec")
        .agg(
            pl.col("run_idx").n_unique().alias("n_runs"),
            pl.col("untag_prec_Global_Qvalue").first().alias("global_q"),
        )
    )
    lost = per_precursor.filter(pl.col("global_q") >= fdr_threshold)
    lost_by_n_runs = dict(lost.group_by("n_runs").len().iter_rows())

    x = np.arange(1, n_runs + 1)
    counts = np.array([lost_by_n_runs.get(n, 0) for n in x])

    fig, ax = plt.subplots(figsize=(_figure_width(n_runs), 4))
    ax.bar(x, counts, width=_bar_width(fig, n_runs), color=_TARGET_COLOR)
    _label_columns(ax, x, counts, "{:,}")
    _style_axes(ax, x)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    if not counts.any():
        ax.set_ylim(0, 1)
        ax.text(0.5, 0.5, "None", transform=ax.transAxes, ha="center", va="center", color=_MUTED_COLOR)
    ax.set_xlabel("Runs the precursor passes the run-level FDR in", color=_MUTED_COLOR)
    ax.set_ylabel("Precursors", color=_MUTED_COLOR)
    ax.set_title(f"Precursors failing the global q-value ({fdr_threshold:g}): "
                 f"{lost.height:,} of {per_precursor.height:,}",
                 loc="left", color=_TEXT_COLOR)
    _save(fig, experiment_dir, "grouped_lost_to_global_q.png")


def plot_best_score_run(untag_prec_global_qs, run_idxs, experiment_dir):
    """Which run each precursor's best score, the one the global q-value uses,
    came from: one panel for targets, one for decoys.

    Only precursors scored in two or more runs are counted; for the rest there
    is no choice of run.  If no run's scores run higher than the others',
    each run holds about an even share.
    """
    shared = untag_prec_global_qs.filter(pl.col("n_scored_runs") >= 2)
    even_share = 100 / len(run_idxs)

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(_figure_width(len(run_idxs)), 6))
    for ax, is_decoy, name, color in ((axes[0], False, "Targets", _TARGET_COLOR),
                                      (axes[1], True, "Decoys", _DECOY_COLOR)):
        precursors = shared.filter(pl.col("is_decoy") == is_decoy)
        best_by_run = dict(precursors.group_by("BestRun").len().iter_rows())
        shares = np.array([100 * best_by_run.get(r, 0) / max(precursors.height, 1) for r in run_idxs])

        ax.bar(run_idxs, shares, width=_bar_width(fig, len(run_idxs)), color=color)
        ax.axhline(even_share, color=_MUTED_COLOR, linewidth=1)
        ax.annotate("even share", xy=(1, even_share), xycoords=("axes fraction", "data"),
                    xytext=(0, 3), textcoords="offset points", ha="right", va="bottom",
                    fontsize=8, color=_MUTED_COLOR)
        _style_axes(ax, run_idxs)
        ax.set_ylabel("% of precursors", color=_MUTED_COLOR)
        ax.set_title(f"{name} ({precursors.height:,} scored in 2+ runs)",
                     loc="left", fontsize=10, color=_TEXT_COLOR)
    axes[1].set_xlabel("Run", color=_MUTED_COLOR)
    fig.suptitle("Run each precursor's best score came from", x=0.01, ha="left", color=_TEXT_COLOR)
    _save(fig, experiment_dir, "grouped_best_score_run.png")


def _figure_width(n_columns):
    # Wide enough that 90 runs stay readable, capped so a figure stays printable
    return min(16, max(6, 2 + 0.14 * n_columns))


def _bar_width(fig, n_columns):
    # A fraction of each slot, but never wider than about a quarter of an inch
    slot_inches = fig.get_figwidth() * 0.8 / n_columns
    return min(0.8, 0.25 / slot_inches)


def _label_columns(ax, x, values, fmt):
    # Value on top of each non-empty column, only while there are few enough to read
    nonzero = [(xi, v) for xi, v in zip(x, values) if v]
    if len(nonzero) > 25:
        return
    for xi, v in nonzero:
        ax.annotate(fmt.format(v), xy=(xi, v), xytext=(0, 2), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8, color=_MUTED_COLOR)


def _style_axes(ax, x):
    ax.set_xlim(min(x) - 0.6, max(x) + 0.6)
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(_GRID_COLOR)
    ax.tick_params(colors=_MUTED_COLOR, length=0)
    ax.grid(axis="y", color=_GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)


def _save(fig, experiment_dir, file_name):
    path = datestamped(os.path.join(experiment_dir, file_name))
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Plot written to {os.path.abspath(path)}")
