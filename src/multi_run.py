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

import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator
import numpy as np
import polars as pl

from src.logger import logger
from src.utils.misc_functions import datestamped

# Chart colours, from a validated palette: blue and orange categorical slots,
# a lighter blue for the second part of a stack, the blue sequential ramp for
# the heatmap, and recessive greys for text, reference lines and gridlines
_BLUE = "#2a78d6"
_LIGHT_BLUE = "#9ec5f4"
_DARK_BLUE = "#184f95"
_ORANGE = "#eb6834"
_BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
_TEXT_COLOR = "#0b0b0b"
_MUTED_COLOR = "#52514e"
_GRID_COLOR = "#e4e3df"
_SURFACE_COLOR = "#ffffff"


def combine_runs(completed_run_folders, experiment_dir, target_decoy_ratio, fdr_threshold):
    """Combine the completed runs into one table, and plot it.

    *completed_run_folders* maps each completed run's index to its results
    folder.  Writes an experiment_results folder in *experiment_dir*,
    datestamped if the name is taken, holding combined_filtered_IDs.parquet:
    every run's IDs that pass the run-level FDR, with a global q-value per
    untagged precursor.  Nothing is filtered on the global q-value; that is
    left to the user.  The plots go there too; those that compare runs only
    when there are enough runs to compare.
    """
    logger.info("")
    logger.info(f"Combining the results of {len(completed_run_folders)} run(s)")
    df, untag_prec_global_qs = collect_results(completed_run_folders, target_decoy_ratio, fdr_threshold)

    results_dir = datestamped(os.path.join(experiment_dir, "experiment_results"))
    os.makedirs(results_dir)
    df.write_parquet(os.path.join(results_dir, "combined_filtered_IDs.parquet"))
    logger.info(f"Experiment results written to {os.path.abspath(results_dir)}")

    run_idxs = sorted(completed_run_folders)
    precursors = precursors_per_run(df, fdr_threshold)
    # The plots' file names are numbered in reading order: identifications,
    # completeness, quantity, then the global q-value's diagnostics
    plot_ids_per_run(precursors, run_idxs, fdr_threshold, results_dir)
    plot_proteins_per_run(df, run_idxs, fdr_threshold, results_dir)
    plot_summed_intensity_per_run(precursors, run_idxs, results_dir)
    plot_intensity_per_run(precursors, run_idxs, results_dir)
    # These compare runs
    if len(run_idxs) >= 2:
        plot_lost_to_global_q(precursors, len(run_idxs), fdr_threshold, results_dir)
        plot_best_score_run(untag_prec_global_qs, run_idxs, results_dir)
        plot_data_completeness(precursors, len(run_idxs), fdr_threshold, results_dir)
    if len(run_idxs) >= 3:
        plot_run_correlation(precursors, run_idxs, results_dir)


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
                 "plex_Area", "coeff", "stripped_seq", "untag_seq", "untag_prec",
                 "mz", "rt", "prec_im", "is_decoy"])
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


def precursors_per_run(df, fdr_threshold):
    """One row per run and untagged precursor it identifies, for the plots.

    ``area`` is the summed plex_Area (0 when there is none), ``log2_area`` its
    log2 (null when there is none), and ``retained`` whether the precursor
    passes the global q-value.
    """
    ##TODO plexDIA: the channels of a precursor are summed into one row here; split them
    return (
        df.group_by("run_idx", "untag_prec")
        .agg(
            pl.col("plex_Area").fill_nan(None).sum().alias("area"),
            (pl.col("untag_prec_Global_Qvalue").first() < fdr_threshold).alias("retained"),
        )
        .with_columns(
            pl.when(pl.col("area") > 0).then(pl.col("area").log(2)).alias("log2_area")
        )
    )


def plot_ids_per_run(precursors, run_idxs, fdr_threshold, results_dir):
    """Stacked columns: each run's precursors at the run-level FDR, split into
    those the global q-value keeps and, on top, those it removes."""
    counts = {r: (0, 0) for r in run_idxs}
    for run_idx, retained, lost in (precursors.group_by("run_idx")
                                    .agg(pl.col("retained").sum(), (~pl.col("retained")).sum().alias("lost"))
                                    .iter_rows()):
        counts[run_idx] = (retained, lost)
    retained = np.array([counts[r][0] for r in run_idxs])
    lost = np.array([counts[r][1] for r in run_idxs])

    fig, ax = plt.subplots(figsize=(_figure_width(len(run_idxs)), 4))
    width = _bar_width(fig, len(run_idxs))
    # A thin surface-coloured edge keeps the two segments apart
    ax.bar(run_idxs, retained, width=width, color=_BLUE, edgecolor=_SURFACE_COLOR, linewidth=0.8,
           label=f"Pass the global q-value ({fdr_threshold:g})")
    ax.bar(run_idxs, lost, bottom=retained, width=width, color=_LIGHT_BLUE, edgecolor=_SURFACE_COLOR,
           linewidth=0.8, label="Removed by the global q-value")
    _label_columns(ax, run_idxs, retained + lost, "{:,}")
    _style_axes(ax, run_idxs)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("Run", color=_MUTED_COLOR)
    ax.set_ylabel("Precursors", color=_MUTED_COLOR)
    ax.set_title(f"Precursors per run (run-level q-value < {fdr_threshold:g})", loc="left", color=_TEXT_COLOR)
    _legend(ax)
    _save(fig, results_dir, "01_ids_per_run.png")


def plot_proteins_per_run(df, run_idxs, fdr_threshold, results_dir):
    """Columns: each run's proteins at the run-level protein q-value, among its
    run-level precursor IDs."""
    ##TODO stack by the global protein q-value, as plot_ids_per_run does for
    ##precursors, once there is one
    counts = dict(df.filter(pl.col("Protein_Qvalue") < fdr_threshold)
                  .group_by("run_idx").agg(pl.col("protein").n_unique())
                  .iter_rows())
    proteins = np.array([counts.get(r, 0) for r in run_idxs])

    fig, ax = plt.subplots(figsize=(_figure_width(len(run_idxs)), 4))
    ax.bar(run_idxs, proteins, width=_bar_width(fig, len(run_idxs)), color=_BLUE)
    _label_columns(ax, run_idxs, proteins, "{:,}")
    _style_axes(ax, run_idxs)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("Run", color=_MUTED_COLOR)
    ax.set_ylabel("Proteins", color=_MUTED_COLOR)
    ax.set_title(f"Proteins per run (run-level protein q-value < {fdr_threshold:g})",
                 loc="left", color=_TEXT_COLOR)
    _save(fig, results_dir, "02_proteins_per_run.png")


def plot_summed_intensity_per_run(precursors, run_idxs, results_dir):
    """Columns: log2 of each run's summed plex_Area over its run-level IDs."""
    summed = dict(precursors.group_by("run_idx").agg(pl.col("area").sum()).iter_rows())
    log2_summed = np.array([math.log2(summed[r]) if summed.get(r, 0) > 0 else 0 for r in run_idxs])
    n_unquantified = precursors.filter(pl.col("log2_area").is_null()).height

    fig, ax = plt.subplots(figsize=(_figure_width(len(run_idxs)), 4))
    ax.bar(run_idxs, log2_summed, width=_bar_width(fig, len(run_idxs)), color=_BLUE)
    _label_columns(ax, run_idxs, log2_summed, "{:.1f}")
    _style_axes(ax, run_idxs)
    ax.set_xlabel("Run", color=_MUTED_COLOR)
    ax.set_ylabel("log2 summed plex_Area", color=_MUTED_COLOR)
    ax.set_title("Summed intensity per run", loc="left", color=_TEXT_COLOR)
    _subtitle(ax, f"{n_unquantified:,} IDs without a plex_Area left out")
    _save(fig, results_dir, "04_summed_intensity_per_run.png")


def plot_intensity_per_run(precursors, run_idxs, results_dir):
    """Box plots: the distribution of log2 plex_Area in each run.

    Runs on the same intensity scale have their medians level.  Whiskers reach
    1.5 x IQR; points beyond them are not drawn, so 90 runs stay readable.
    """
    by_run = dict(precursors.filter(pl.col("log2_area").is_not_null())
                  .group_by("run_idx").agg(pl.col("log2_area")).iter_rows())
    data = [np.asarray(by_run.get(r, [np.nan])) for r in run_idxs]

    fig, ax = plt.subplots(figsize=(_figure_width(len(run_idxs)), 4))
    ax.boxplot(data, positions=run_idxs, widths=_bar_width(fig, len(run_idxs)),
               showfliers=False, patch_artist=True, manage_ticks=False,
               boxprops=dict(facecolor=_LIGHT_BLUE, edgecolor=_BLUE),
               whiskerprops=dict(color=_BLUE), capprops=dict(color=_BLUE),
               medianprops=dict(color=_DARK_BLUE, linewidth=1.5))
    _style_axes(ax, run_idxs, zero_baseline=False)
    ax.set_xlabel("Run", color=_MUTED_COLOR)
    ax.set_ylabel("log2 plex_Area", color=_MUTED_COLOR)
    ax.set_title("Intensity per run", loc="left", color=_TEXT_COLOR)
    _subtitle(ax, "Whiskers at 1.5 x IQR; outliers not drawn")
    _save(fig, results_dir, "05_intensity_per_run.png")


def plot_lost_to_global_q(precursors, n_runs, fdr_threshold, results_dir):
    """Columns: the precursors the global q-value removes, by the number of
    runs they pass the run-level FDR in."""
    per_precursor = (
        precursors.group_by("untag_prec")
        .agg(pl.col("run_idx").n_unique().alias("n_runs"), pl.col("retained").first())
    )
    lost = per_precursor.filter(~pl.col("retained"))
    lost_by_n_runs = dict(lost.group_by("n_runs").len().iter_rows())

    x = np.arange(1, n_runs + 1)
    counts = np.array([lost_by_n_runs.get(n, 0) for n in x])

    fig, ax = plt.subplots(figsize=(_figure_width(n_runs), 4))
    ax.bar(x, counts, width=_bar_width(fig, n_runs), color=_BLUE)
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
    _save(fig, results_dir, "07_lost_to_global_q.png")


def plot_best_score_run(untag_prec_global_qs, run_idxs, results_dir):
    """Which run each precursor's best score, the one the global q-value uses,
    came from: one panel for targets, one for decoys.

    Only precursors scored in two or more runs are counted; for the rest there
    is no choice of run.  If no run's scores run higher than the others',
    each run holds about an even share.
    """
    shared = untag_prec_global_qs.filter(pl.col("n_scored_runs") >= 2)
    even_share = 100 / len(run_idxs)

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(_figure_width(len(run_idxs)), 6))
    for ax, is_decoy, name, color in ((axes[0], False, "Targets", _BLUE),
                                      (axes[1], True, "Decoys", _ORANGE)):
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
    _save(fig, results_dir, "08_best_score_run.png")


def plot_data_completeness(precursors, n_runs, fdr_threshold, results_dir):
    """Lines: how many precursors are identified in at least k runs, for every
    k, for all run-level IDs and for those that also pass the global q-value."""
    per_precursor = (
        precursors.group_by("untag_prec")
        .agg(pl.col("run_idx").n_unique().alias("n_runs"), pl.col("retained").first())
    )
    k = np.arange(1, n_runs + 1)
    n_all = per_precursor["n_runs"].to_numpy()
    n_retained = per_precursor.filter(pl.col("retained"))["n_runs"].to_numpy()
    at_least_all = np.array([(n_all >= i).sum() for i in k])
    at_least_retained = np.array([(n_retained >= i).sum() for i in k])

    fig, ax = plt.subplots(figsize=(_figure_width(n_runs), 4))
    marker = "o" if n_runs <= 30 else None
    ax.plot(k, at_least_all, color=_LIGHT_BLUE, linewidth=2, marker=marker, markersize=5,
            markeredgecolor=_SURFACE_COLOR, label="All run-level IDs")
    ax.plot(k, at_least_retained, color=_BLUE, linewidth=2, marker=marker, markersize=5,
            markeredgecolor=_SURFACE_COLOR, label=f"Also pass the global q-value ({fdr_threshold:g})")
    _style_axes(ax, k)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("Identified in at least this many runs", color=_MUTED_COLOR)
    ax.set_ylabel("Precursors", color=_MUTED_COLOR)
    ax.set_title("Data completeness", loc="left", color=_TEXT_COLOR)
    _legend(ax)
    _save(fig, results_dir, "03_data_completeness.png")


def plot_run_correlation(precursors, run_idxs, results_dir):
    """Heatmap: Pearson correlation of log2 plex_Area between every pair of
    runs, over the precursors both quantify.  An outlier run shows as a pale
    row and column."""
    # Pairs sharing fewer precursors than this are left blank
    min_shared = 10
    wide = (precursors.filter(pl.col("log2_area").is_not_null())
            .pivot(on="run_idx", index="untag_prec", values="log2_area")
            .to_pandas())
    wide.columns = [str(c) for c in wide.columns]
    wide = wide.reindex(columns=[str(r) for r in run_idxs])
    corr = wide.corr(min_periods=min_shared).to_numpy()

    n = len(run_idxs)
    off_diagonal = corr[~np.eye(n, dtype=bool)]
    lowest = np.nanmin(off_diagonal) if np.isfinite(off_diagonal).any() else 0.0
    cmap = LinearSegmentedColormap.from_list("blues", _BLUE_RAMP)
    cmap.set_bad(_GRID_COLOR)

    size = min(12, max(5, 3 + 0.1 * n))
    fig, ax = plt.subplots(figsize=(size + 1, size))
    image = ax.imshow(np.ma.masked_invalid(corr), cmap=cmap, vmin=lowest, vmax=1)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Pearson r", color=_MUTED_COLOR)
    colorbar.ax.tick_params(colors=_MUTED_COLOR, length=0)
    colorbar.outline.set_visible(False)

    step = math.ceil(n / 20)  # at most about 20 labels per axis
    ticks = list(range(0, n, step))
    for set_ticks, set_labels in ((ax.set_xticks, ax.set_xticklabels), (ax.set_yticks, ax.set_yticklabels)):
        set_ticks(ticks)
        set_labels([run_idxs[i] for i in ticks])
    ax.tick_params(colors=_MUTED_COLOR, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if n <= 12:
        midpoint = (lowest + 1) / 2
        for i in range(n):
            for j in range(n):
                if np.isfinite(corr[i, j]):
                    ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=8,
                            color=_SURFACE_COLOR if corr[i, j] > midpoint else _TEXT_COLOR)
    ax.set_xlabel("Run", color=_MUTED_COLOR)
    ax.set_ylabel("Run", color=_MUTED_COLOR)
    ax.set_title("Run-to-run correlation of log2 plex_Area", loc="left", color=_TEXT_COLOR)
    _save(fig, results_dir, "06_run_correlation.png")


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


def _subtitle(ax, text):
    ax.annotate(text, xy=(0, 1), xycoords="axes fraction", xytext=(0, 2), textcoords="offset points",
                ha="left", va="bottom", fontsize=8, color=_MUTED_COLOR)
    ax.set_title(ax.get_title(loc="left"), loc="left", color=_TEXT_COLOR, pad=16)


def _legend(ax):
    legend = ax.legend(loc="upper left", bbox_to_anchor=(1, 1), frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color(_MUTED_COLOR)


def _style_axes(ax, x, zero_baseline=True):
    ax.set_xlim(min(x) - 0.6, max(x) + 0.6)
    if zero_baseline:
        ax.set_ylim(bottom=0)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(_GRID_COLOR)
    ax.tick_params(colors=_MUTED_COLOR, length=0)
    ax.grid(axis="y", color=_GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)


def _save(fig, results_dir, file_name):
    path = os.path.join(results_dir, file_name)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
