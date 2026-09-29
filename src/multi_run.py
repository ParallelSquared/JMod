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
from typing import NamedTuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator
import numpy as np
import polars as pl

from statsmodels.nonparametric.smoothers_lowess import lowess

from src.logger import logger
from src.utils.parse_peptides import untag_sequences
from src.utils.errors import JModError
from src.utils.misc_functions import datestamped

# The first pass's IDs that go into the match-between-runs library: those with
# this q-value below the FDR threshold.  "untag_prec_Global_Qvalue" is the
# stricter choice, keeping the union of the runs' IDs at the FDR threshold.
MBR_QVALUE_COLUMN = "BestChannel_Qvalue"

# A run is aligned to the reference only if they share at least this many precursors
_MIN_SHARED_FOR_ALIGNMENT = 20

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


class _Level(NamedTuple):
    """What the global q-value plots of one level (precursor or protein) are about."""
    key: str             # the column the global q-value is computed per
    noun: str            # e.g. "precursor", in the labels
    global_q: str        # its global q-value, in the labels
    ids_file: str        # plot_ids_per_run's file
    lost_file: str       # plot_lost_to_global_q's file
    best_run_file: str   # plot_best_score_run's file
    completeness_file: str  # plot_data_completeness's file


_PRECURSOR_LEVEL = _Level("untag_prec", "precursor", "global q-value", "01_precursors_per_run.png",
                          "08_precursors_lost_to_global_q.png", "09_precursor_best_score_run.png",
                          "03_precursor_data_completeness.png")
_PROTEIN_LEVEL = _Level("protein", "protein", "global protein q-value", "02_proteins_per_run.png",
                        "10_proteins_lost_to_global_q.png", "12_protein_best_score_run.png",
                        "04_protein_data_completeness.png")

# plot_proteins_lost_by_precursor_count's last column holds this many
# precursors and more, so a long tail of large proteins stays readable
_MAX_PRECURSOR_COLUMN = 10


def combine_runs(completed_run_folders, experiment_dir, target_decoy_ratio, fdr_threshold):
    """Combine the completed runs into one table, and plot it.

    *completed_run_folders* maps each completed run's index to its results
    folder.  Writes an experiment_results folder in *experiment_dir*,
    datestamped if the name is taken, holding combined_filtered_IDs.parquet:
    every run's IDs that pass the run-level FDR, with a global q-value per
    untagged precursor and per protein.  Nothing is filtered on the global
    q-values; that is left to the user.  The plots go there too; those that compare runs only
    when there are enough runs to compare.  Returns the combined table.
    """
    logger.info("")
    logger.info(f"Combining the results of {len(completed_run_folders)} run(s)")
    df, untag_prec_global_qs, protein_global_qs = collect_results(completed_run_folders, target_decoy_ratio,
                                                                  fdr_threshold)

    results_dir = datestamped(os.path.join(experiment_dir, "experiment_results"))
    os.makedirs(results_dir)
    df.write_parquet(os.path.join(results_dir, "combined_filtered_IDs.parquet"))
    logger.info(f"Experiment results written to {os.path.abspath(results_dir)}")

    run_idxs = sorted(completed_run_folders)
    precursors = precursors_per_run(df, fdr_threshold)
    proteins = proteins_per_run(df, fdr_threshold)
    # Made in the order of their numbers: identifications, completeness,
    # quantity, then the global q-values' diagnostics, precursors before
    # proteins.  Those that compare runs are made only when there are enough runs
    n_runs = len(run_idxs)
    plot_ids_per_run(precursors, _PRECURSOR_LEVEL, run_idxs, fdr_threshold, results_dir)
    plot_ids_per_run(proteins, _PROTEIN_LEVEL, run_idxs, fdr_threshold, results_dir)
    if n_runs >= 2:
        plot_data_completeness(precursors, _PRECURSOR_LEVEL, n_runs, fdr_threshold, results_dir)
        plot_data_completeness(proteins, _PROTEIN_LEVEL, n_runs, fdr_threshold, results_dir)
    plot_summed_intensity_per_run(precursors, run_idxs, results_dir)
    plot_intensity_per_run(precursors, run_idxs, results_dir)
    if n_runs >= 3:
        plot_run_correlation(precursors, run_idxs, results_dir)
    if n_runs >= 2:
        plot_lost_to_global_q(precursors, _PRECURSOR_LEVEL, n_runs, fdr_threshold, results_dir)
        plot_best_score_run(untag_prec_global_qs, _PRECURSOR_LEVEL, run_idxs, results_dir)
        plot_lost_to_global_q(proteins, _PROTEIN_LEVEL, n_runs, fdr_threshold, results_dir)
        plot_proteins_lost_by_precursor_count(df, fdr_threshold, results_dir)
        plot_best_score_run(protein_global_qs, _PROTEIN_LEVEL, run_idxs, results_dir)
    return df


def collect_results(folder_path_dict, target_decoy_ratio, fdr_threshold):
    """Every run's IDs, with a global q-value for each untagged precursor and
    for each protein (global_qvalues).

    Returns (the run-level IDs that pass *fdr_threshold*, sorted by precursor,
    with both global q-values; global_qvalues' table for untagged precursors;
    and for proteins).
    """
    df = pl.concat([
        pl.scan_parquet(os.path.join(folder_path, "outputs", "all_IDs_filtered.parquet"))
        .with_columns(pl.lit(run_idx).alias("run_idx"))
        for run_idx, folder_path in folder_path_dict.items()
    ], how="vertical_relaxed")  # a column's type can differ between runs (e.g. channel int vs float)

    # Kept when the runs' results carry them: timeplex runs have each ID's time
    # channel, and coeff and prec_im are there when the results were written with them
    available = df.collect_schema().names()
    time_channel, coeff, prec_im = ([c] if c in available else [] for c in ("time_channel", "coeff", "prec_im"))
    df = (
        df
        .select(["run_idx", "file_name", "protein", "seq", "z", "channel", "silac_channel",
                 *time_channel,
                 "PredVal", "Qvalue", "BestChannel_Qvalue", "Protein_Qvalue",
                 "plex_Area", *coeff, "stripped_seq", "untag_seq", "untag_prec",
                 "mz", "rt", *prec_im, "is_decoy"])
        .with_columns(
            (pl.col("seq") + pl.lit("_") + pl.col("z").cast(pl.Int16).cast(pl.String)).alias("Prec")
        )
    )

    untag_prec_global_qs = global_qvalues(df, "untag_prec", target_decoy_ratio, "untag_prec_Global_Qvalue")
    # Each protein by its best precursor; rows without a protein are left out, as
    # the run-level protein q-value leaves them out (fdr_analysis.compute_protein_FDR)
    protein_global_qs = global_qvalues(df.filter(pl.col("protein").is_not_null()), "protein",
                                       target_decoy_ratio, "Protein_Global_Qvalue")

    # Only targets are kept, so only the target proteins' q-values are joined
    # (a decoy shares its target's protein name)
    df = (
        df
        .filter(pl.col("BestChannel_Qvalue") < fdr_threshold)
        .filter(pl.col("is_decoy") == False)
        .join(
            untag_prec_global_qs.lazy().select(["untag_prec", "untag_prec_Global_Qvalue"]),
            on="untag_prec",
            how="left",
        )
        .join(
            protein_global_qs.lazy().filter(~pl.col("is_decoy")).select(["protein", "Protein_Global_Qvalue"]),
            on="protein",
            how="left",
        )
        .sort(["untag_prec", "Prec", "run_idx"])
        .collect()
    )

    return df, untag_prec_global_qs, protein_global_qs


def global_qvalues(df, key, target_decoy_ratio, qvalue_name):
    """A global q-value for each *key* (untag_prec, or protein), over every run.

    Each key's target and decoy are ranked by their best PredVal across runs,
    channels and SILAC channels (and a protein's precursors), and targets and
    decoys are counted as the per-run q-value does.  Returns one row per key and
    is_decoy with its best score, the run it came from, and the q-value, named
    *qvalue_name*.
    """
    scores = (
        # A decoy shares its target's protein name, so they are told apart by is_decoy
        df.group_by(key, "is_decoy")
        .agg(
            pl.col("PredVal").max().alias("MaxPredVal"),
            # Ties go to the earliest run
            pl.col("run_idx").sort_by(["PredVal", "run_idx"], descending=[True, False]).first().alias("BestRun"),
            pl.col("run_idx").n_unique().alias("n_scored_runs"),
        )
    )

    global_qs = (
        scores
        .sort("MaxPredVal", descending=True)
        .with_columns(
            (~pl.col("is_decoy")).cast(pl.Int64).cum_sum().alias("n_target"),
            pl.col("is_decoy").cast(pl.Int64).cum_sum().alias("n_decoy"),
        )
        # Tied scores are a single threshold step: give every key in a tie the
        # counts from its end, so the q-value doesn't depend on row order
        .with_columns(
            pl.col("n_target").max().over("MaxPredVal"),
            pl.col("n_decoy").max().over("MaxPredVal"),
        )
    )

    return (
        global_qs
        .with_columns(
            # As the per-run q-value (fdr_analysis.score_precursors)
            ((1 + pl.col("n_decoy")) / pl.col("n_target") * target_decoy_ratio).alias("FDR")
        )
        .with_columns(
            pl.col("FDR")
            .reverse()
            .cum_min()
            .reverse()
            .alias(qvalue_name)
        )
        .collect()
    )


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


def proteins_per_run(df, fdr_threshold):
    """One row per run and protein it identifies at the run-level protein
    q-value, among its run-level precursor IDs, for the plots.

    ``retained`` is whether the protein passes the global protein q-value.
    """
    return (
        df.filter(pl.col("Protein_Qvalue") < fdr_threshold)
        .group_by("run_idx", "protein")
        .agg((pl.col("Protein_Global_Qvalue").first() < fdr_threshold).alias("retained"))
    )


def mbr_library_rts(combined_ids, fdr_threshold, mbr_dir, qvalue_column=MBR_QVALUE_COLUMN):
    """One aligned RT for every untagged precursor the first pass identified.

    The IDs are those with *qvalue_column* below *fdr_threshold*.  Each run's
    RTs are aligned to a reference run, the one with the most IDs: a LOWESS fit
    of the RT difference over the precursors the two share, after dropping
    differences more than 4 SD from their mean, is added to every RT of the
    run.  A precursor's library RT is then its reference-run RT, or else the
    median of its aligned RTs.  Within a run, a precursor identified in several
    channels takes the RT of its best-scoring one.

    On timeplex each time channel of each run is aligned as a unit of its own,
    so the channels' offsets are removed along with the drift between runs.
    ##TODO test on timeplex data once the timeplex changes are merged

    Writes the plots to *mbr_dir*.  Returns one row per untag_prec with its rt.
    """
    ids = combined_ids.filter(pl.col(qvalue_column) < fdr_threshold, ~pl.col("is_decoy"))
    if ids.is_empty():
        raise JModError("The first pass identified no precursors, so there is no "
                        "match-between-runs library to build")
    unit_columns = ["run_idx"] + (["time_channel"] if "time_channel" in ids.columns else [])
    unit_rts = (
        ids.group_by(*unit_columns, "untag_prec")
        .agg(pl.col("rt").sort_by(["PredVal", "rt"], descending=[True, False]).first().cast(pl.Float64))
    )
    units = {unit: frame.drop(unit_columns)
             for unit, frame in unit_rts.partition_by(unit_columns, as_dict=True).items()}
    reference = max(sorted(units), key=lambda unit: units[unit].height)
    logger.info(f"MBR library: aligning RTs to {_unit_name(reference, unit_columns)}, "
                f"which has the most IDs ({units[reference].height:,})")
    reference_rts = units[reference].rename({"rt": "reference_rt"})

    lowess_dir = os.path.join(mbr_dir, "lowess")
    os.makedirs(lowess_dir, exist_ok=True)
    aligned = [units[reference].with_columns(pl.lit(True).alias("is_reference"))]
    for unit in sorted(units):
        if unit == reference:
            continue
        name = _unit_name(unit, unit_columns)
        shared = units[unit].join(reference_rts, on="untag_prec")
        if shared.height < _MIN_SHARED_FOR_ALIGNMENT:
            logger.warning(f"MBR library: {name} shares only {shared.height} precursors with the "
                           f"reference, too few to align; its RTs are left out")
            continue
        shared_rt = shared["rt"].to_numpy()
        delta, kept, curve = _fit_rt_alignment(shared_rt, shared["reference_rt"].to_numpy())
        plot_rt_alignment(shared_rt, delta, kept, curve, name, lowess_dir)
        rt = units[unit]["rt"].to_numpy()
        aligned.append(units[unit].with_columns(
            pl.Series("rt", rt + np.interp(rt, curve[:, 0], curve[:, 1])),
            pl.lit(False).alias("is_reference"),
        ))

    plot_library_size_by_run(combined_ids.filter(~pl.col("is_decoy")), fdr_threshold, qvalue_column, mbr_dir)
    return (
        pl.concat(aligned)
        .group_by("untag_prec")
        .agg(
            pl.when(pl.col("is_reference").any())
            .then(pl.col("rt").filter(pl.col("is_reference")).first())
            .otherwise(pl.col("rt").median())
            .alias("rt")
        )
        .sort("untag_prec")
    )


def mbr_targets(targets, library_rts, mass_tag, SILAC):
    """The library targets the first pass identified, with their aligned RTs.

    *targets* is the input library as loaded, before decoys and tagging.
    Tagged or detagged, it is matched on untag_prec.  Returns a new target
    store of the matched entries, with their iRT replaced by the aligned RT
    (on the reference run's time scale).
    """
    untag_prec = (pl.Series(untag_sequences(targets.mod_seq, mass_tag, SILAC), dtype=pl.String)
                  + "_" + pl.Series(targets.prec_z).cast(pl.Int64).cast(pl.String))
    rt = untag_prec.replace_strict(library_rts["untag_prec"], library_rts["rt"],
                                   default=None, return_dtype=pl.Float64)
    matched = rt.is_not_null()
    if not matched.any():
        raise JModError("No first-pass ID matches an entry of the spectral library, "
                        "so there is no match-between-runs library to build")
    logger.info(f"MBR library: {int(matched.sum()):,} of {len(targets):,} library precursors "
                f"were identified in the first pass")
    mbr = targets.subset_entries(matched.to_numpy())
    mbr.iRT = rt.filter(matched).to_numpy().astype(targets.iRT.dtype)
    return mbr


def _fit_rt_alignment(rt, reference_rt):
    """LOWESS fit of the difference reference_rt - rt against rt.

    Differences more than 4 SD from their mean are left out of the fit.
    Returns (delta, kept, curve); the curve's columns are rt and the
    correction to add to it.
    """
    delta = reference_rt - rt
    # <= rather than <: runs with no spread at all keep every point
    kept = np.abs(delta - delta.mean()) <= 4 * np.std(delta, ddof=1)
    curve = lowess(delta[kept], rt[kept], frac=0.2, return_sorted=True)
    return delta, kept, curve


def _unit_name(unit, unit_columns):
    # e.g. "run 3", or "run 3, time channel 1" on timeplex
    parts = [f"run {unit[0]}"]
    if len(unit_columns) > 1:
        parts.append(f"time channel {int(unit[1])}")
    return ", ".join(parts)


def plot_ids_per_run(per_run, level, run_idxs, fdr_threshold, results_dir):
    """Stacked columns: each run's precursors (or proteins, per *level*) at the
    run-level q-value, split into those the global q-value keeps and, on top,
    those it removes.

    *per_run* has a row per run and precursor (protein) it identifies, as
    precursors_per_run (proteins_per_run) gives.
    """
    counts = {r: (0, 0) for r in run_idxs}
    for run_idx, retained, lost in (per_run.group_by("run_idx")
                                    .agg(pl.col("retained").sum(), (~pl.col("retained")).sum().alias("lost"))
                                    .iter_rows()):
        counts[run_idx] = (retained, lost)
    retained = np.array([counts[r][0] for r in run_idxs])
    lost = np.array([counts[r][1] for r in run_idxs])

    noun = level.noun.capitalize()
    fig, ax = plt.subplots(figsize=(_figure_width(len(run_idxs)), 4))
    width = _bar_width(fig, len(run_idxs))
    # A thin surface-coloured edge keeps the two segments apart
    ax.bar(run_idxs, retained, width=width, color=_BLUE, edgecolor=_SURFACE_COLOR, linewidth=0.8,
           label=f"Global Qvalue < {fdr_threshold:g}")
    _unstick(ax.bar(run_idxs, lost, bottom=retained, width=width, color=_LIGHT_BLUE, edgecolor=_SURFACE_COLOR,
                    linewidth=0.8, label=f"Qvalue < {fdr_threshold:g}"))
    _label_columns(ax, run_idxs, retained + lost, "{:,}")
    _style_axes(ax, run_idxs)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("Run", color=_MUTED_COLOR)
    ax.set_ylabel(f"{noun}s", color=_MUTED_COLOR)
    ax.set_title(f"{noun}s Per Run", loc="center", color=_TEXT_COLOR)
    _legend(ax, below=True)
    _save(fig, results_dir, level.ids_file)


def plot_summed_intensity_per_run(precursors, run_idxs, results_dir):
    """Columns: log2 of each run's summed plex_Area over its run-level IDs."""
    summed = dict(precursors.group_by("run_idx").agg(pl.col("area").sum()).iter_rows())
    log2_summed = np.array([math.log2(summed[r]) if summed.get(r, 0) > 0 else 0 for r in run_idxs])

    fig, ax = plt.subplots(figsize=(_figure_width(len(run_idxs)), 4))
    ax.bar(run_idxs, log2_summed, width=_bar_width(fig, len(run_idxs)), color=_BLUE)
    _label_columns(ax, run_idxs, log2_summed, "{:.1f}")
    _style_axes(ax, run_idxs)
    if log2_summed.max() > 0:
        ax.set_ylim(top=log2_summed.max() * 1.1)  # room above the column labels
    ax.set_xlabel("Run", color=_MUTED_COLOR)
    ax.set_ylabel("log2 sum plex_Area", color=_MUTED_COLOR)
    ax.set_title("Summed intensity per run", loc="center", color=_TEXT_COLOR)
    _save(fig, results_dir, "05_summed_intensity_per_run.png")


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
    ax.set_title("Intensity per run", loc="center", color=_TEXT_COLOR)
    _save(fig, results_dir, "06_intensity_per_run.png")


def plot_lost_to_global_q(per_run, level, n_runs, fdr_threshold, results_dir):
    """Columns: the precursors (or proteins, per *level*) the global q-value
    removes, by the number of runs they pass the run-level FDR in.

    *per_run* has a row per run and precursor (protein) it identifies, as
    precursors_per_run (proteins_per_run) gives.
    """
    per_key = (
        per_run.group_by(level.key)
        .agg(pl.col("run_idx").n_unique().alias("n_runs"), pl.col("retained").first())
    )
    lost = per_key.filter(~pl.col("retained"))
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
    ax.set_xlabel(f"Number of Runs with Q < {fdr_threshold:g}", color=_MUTED_COLOR)
    ax.set_ylabel(f"{level.noun.capitalize()}s", color=_MUTED_COLOR)
    ax.set_title(f"{level.noun.capitalize()}s failing the {level.global_q} ({fdr_threshold:g}): "
                 f"{lost.height:,} of {per_key.height:,}",
                 loc="center", color=_TEXT_COLOR)
    _save(fig, results_dir, level.lost_file)


def plot_best_score_run(global_qs, level, run_idxs, results_dir):
    """Which run each precursor's (or protein's, per *level*) best score, the
    one the global q-value uses, came from: one panel for targets, one for
    decoys.  *global_qs* is global_qvalues' table.

    Only those scored in every run are counted, so each run has the same
    chance to hold the best score: if no run's scores run higher than the
    others', each run holds about an even share.
    """
    in_every_run = global_qs.filter(pl.col("n_scored_runs") == len(run_idxs))
    even_share = 100 / len(run_idxs)

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(_figure_width(len(run_idxs)), 6))
    for ax, is_decoy, name, color in ((axes[0], False, "Targets", _BLUE),
                                      (axes[1], True, "Decoys", _ORANGE)):
        subset = in_every_run.filter(pl.col("is_decoy") == is_decoy)
        best_by_run = dict(subset.group_by("BestRun").len().iter_rows())
        shares = np.array([100 * best_by_run.get(r, 0) / max(subset.height, 1) for r in run_idxs])

        ax.bar(run_idxs, shares, width=_bar_width(fig, len(run_idxs)), color=color)
        ax.axhline(even_share, color=_MUTED_COLOR, linewidth=1)
        ax.annotate("even share", xy=(1, even_share), xycoords=("axes fraction", "data"),
                    xytext=(0, 3), textcoords="offset points", ha="right", va="bottom",
                    fontsize=8, color=_MUTED_COLOR)
        _style_axes(ax, run_idxs)
        ax.set_ylabel(f"% of {level.noun}s", color=_MUTED_COLOR)
        ax.set_title(f"{name} ({subset.height:,} scored in all runs)",
                     loc="left", fontsize=10, color=_TEXT_COLOR)
    axes[1].set_xlabel("Run", color=_MUTED_COLOR)
    fig.suptitle(f"Run each {level.noun}'s best score came from", color=_TEXT_COLOR)
    _save(fig, results_dir, level.best_run_file)


def plot_data_completeness(per_run, level, n_runs, fdr_threshold, results_dir):
    """Lines: how many precursors (or proteins, per *level*) are identified in
    at least k runs, for every k, for all run-level IDs and for those that also
    pass the global q-value.

    *per_run* has a row per run and precursor (protein) it identifies, as
    precursors_per_run (proteins_per_run) gives.
    """
    per_key = (
        per_run.group_by(level.key)
        .agg(pl.col("run_idx").n_unique().alias("n_runs"), pl.col("retained").first())
    )
    k = np.arange(1, n_runs + 1)
    n_all = per_key["n_runs"].to_numpy()
    n_retained = per_key.filter(pl.col("retained"))["n_runs"].to_numpy()
    at_least_all = np.array([(n_all >= i).sum() for i in k])
    at_least_retained = np.array([(n_retained >= i).sum() for i in k])

    noun = level.noun.capitalize()
    fig, ax = plt.subplots(figsize=(_figure_width(n_runs), 4))
    marker = "o" if n_runs <= 30 else None
    # Drawn first so its legend entry comes first, as in plot_ids_per_run, and
    # raised so it stays on top where the lines meet
    ax.plot(k, at_least_retained, color=_BLUE, linewidth=2, marker=marker, markersize=5, zorder=3,
            markeredgecolor=_SURFACE_COLOR, label=f"Global Qvalue < {fdr_threshold:g}")
    ax.plot(k, at_least_all, color=_LIGHT_BLUE, linewidth=2, marker=marker, markersize=5,
            markeredgecolor=_SURFACE_COLOR, label=f"Qvalue < {fdr_threshold:g}")
    _style_axes(ax, k)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("Identified in at least K runs", color=_MUTED_COLOR)
    ax.set_ylabel(f"{noun}s", color=_MUTED_COLOR)
    ax.set_title(f"{noun} Data Completeness", loc="center", color=_TEXT_COLOR)
    _legend(ax, below=True)
    _save(fig, results_dir, level.completeness_file)


def plot_proteins_lost_by_precursor_count(df, fdr_threshold, results_dir):
    """Columns: the proteins the global protein q-value removes, by how many
    untagged precursors identify them.

    Over the proteins at the run-level protein q-value in some run, counting
    each one's untagged precursors among its run-level IDs in every run.  From
    _MAX_PRECURSOR_COLUMN on, the counts share one column.
    """
    per_protein = (
        df.filter(pl.col("Protein_Qvalue") < fdr_threshold)
        .group_by("protein")
        .agg(pl.col("untag_prec").n_unique().alias("n_precursors"),
             (pl.col("Protein_Global_Qvalue").first() < fdr_threshold).alias("retained"))
    )
    lost = per_protein.filter(~pl.col("retained"))
    lost_by_count = dict(lost.group_by(pl.col("n_precursors").clip(upper_bound=_MAX_PRECURSOR_COLUMN))
                         .len().iter_rows())

    x = np.arange(1, _MAX_PRECURSOR_COLUMN + 1)
    counts = np.array([lost_by_count.get(n, 0) for n in x])

    fig, ax = plt.subplots(figsize=(_figure_width(len(x)), 4))
    ax.bar(x, counts, width=_bar_width(fig, len(x)), color=_BLUE)
    _label_columns(ax, x, counts, "{:,}")
    _style_axes(ax, x)
    ax.set_xticks(x, [str(n) for n in x[:-1]] + [f"{x[-1]}+"])
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    if not counts.any():
        ax.set_ylim(0, 1)
        ax.text(0.5, 0.5, "None", transform=ax.transAxes, ha="center", va="center", color=_MUTED_COLOR)
    ax.set_xlabel(f"Number of Untag Precs with Q < {fdr_threshold:g}", color=_MUTED_COLOR)
    ax.set_ylabel("Proteins", color=_MUTED_COLOR)
    ax.set_title(f"Proteins failing the global protein q-value ({fdr_threshold:g}): "
                 f"{lost.height:,} of {per_protein.height:,}",
                 loc="left", color=_TEXT_COLOR)
    _save(fig, results_dir, "11_proteins_lost_by_precursor_count.png")


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
    ax.set_title("Run-to-run correlation of log2 plex_Area", loc="center", color=_TEXT_COLOR)
    _save(fig, results_dir, "07_run_correlation.png")


def plot_library_size_by_run(targets, fdr_threshold, qvalue_column, mbr_dir):
    """Step plot: how large the MBR library has grown after each run, in file
    order, counting every precursor identified in that run or an earlier one.

    One line for the run-level IDs (*targets*: the first pass's combined
    targets) and one for those that also pass the global q-value; the legend
    marks the one the library was built from (*qvalue_column*).
    """
    run_idxs = sorted(targets["run_idx"].unique().to_list())
    fig, ax = plt.subplots(figsize=(_figure_width(len(run_idxs)), 4))
    largest = 0
    for column, name, color in (("BestChannel_Qvalue", "Run-level q-value", _BLUE),
                                ("untag_prec_Global_Qvalue", "Global untag_prec q-value", _LIGHT_BLUE)):
        ids = targets.filter(pl.col(column) < fdr_threshold)
        by_run = dict(ids.group_by("run_idx").agg(pl.col("untag_prec").unique()).iter_rows())
        seen, library_size = set(), []
        for r in run_idxs:
            seen |= set(by_run.get(r, []))
            library_size.append(len(seen))
        largest = max(largest, library_size[-1])
        label = f"{name} < {fdr_threshold:g}" + (" (the library)" if column == qvalue_column else "")
        ax.step(run_idxs, library_size, where="post", color=color, linewidth=2, label=label)

    _style_axes(ax, run_idxs)
    ax.set_ylim(0, max(largest, 1) * 1.05)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    if len(run_idxs) <= 30:
        ax.set_xticks(run_idxs)  # every run labelled while they fit
    ax.set_xlabel("Run", color=_MUTED_COLOR)
    ax.set_ylabel("Precursors", color=_MUTED_COLOR)
    ax.set_title("MBR library size by run", loc="left", color=_TEXT_COLOR)
    _legend(ax)
    _save(fig, mbr_dir, "library_size_by_run.png")


def plot_rt_alignment(rt, delta, kept, curve, name, lowess_dir):
    """Two panels for one run: its RT difference to the reference with the
    LOWESS fit, and what is left of the difference after the correction."""
    rt, delta = rt[kept], delta[kept]
    residual = delta - np.interp(rt, curve[:, 0], curve[:, 1])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, values, title in ((axes[0], delta, "Before"), (axes[1], residual, "After correction")):
        ax.scatter(rt, values, s=2, alpha=0.4, color=_BLUE, linewidths=0)
        ax.axhline(0, color=_MUTED_COLOR, linewidth=1)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(_GRID_COLOR)
        ax.tick_params(colors=_MUTED_COLOR, length=0)
        ax.grid(axis="y", color=_GRID_COLOR, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_xlabel(f"RT, {name}", color=_MUTED_COLOR)
        ax.set_title(title, loc="left", fontsize=10, color=_TEXT_COLOR)
    axes[0].plot(curve[:, 0], curve[:, 1], color=_ORANGE, linewidth=2, label="LOWESS fit")
    axes[0].set_ylabel("RT difference (reference - run)", color=_MUTED_COLOR)
    axes[0].legend(loc="upper right", frameon=False, fontsize=9, labelcolor=_MUTED_COLOR)
    fig.suptitle(f"RT alignment of {name} to the reference ({int(kept.sum()):,} shared precursors)",
                 x=0.01, ha="left", color=_TEXT_COLOR)
    _save(fig, lowess_dir, name.replace(", ", "_").replace(" ", "_") + ".png")


def _figure_width(n_columns):
    # Wide enough that 90 runs stay readable, capped so a figure stays printable
    return min(16, max(6, 2 + 0.14 * n_columns))


def _bar_width(fig, n_columns):
    # A fraction of each slot, but never wider than about a quarter of an inch
    slot_inches = fig.get_figwidth() * 0.8 / n_columns
    return min(0.8, 0.25 / slot_inches)


def _unstick(bars):
    # A stacked segment's base is a sticky edge, which autoscaling will not pad
    # past: sitting on the tallest column, it would leave no room above it for
    # the column's label
    for bar in bars:
        bar.sticky_edges.y.clear()


def _label_columns(ax, x, values, fmt):
    # Value on top of each non-empty column, only while there are few enough to read
    nonzero = [(xi, v) for xi, v in zip(x, values) if v]
    if len(nonzero) > 25:
        return
    for xi, v in nonzero:
        ax.annotate(fmt.format(v), xy=(xi, v), xytext=(0, 2), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8, color=_MUTED_COLOR)


def _legend(ax, below=False):
    # Right of the plot, one entry per line; or *below* it, in one row in the
    # order the entries were drawn, anchored to the figure's bottom edge so it
    # clears the x-axis label
    if below:
        handles, _ = ax.get_legend_handles_labels()
        legend = ax.figure.legend(loc="upper center", bbox_to_anchor=(0.5, 0), ncol=len(handles),
                                  frameon=False, fontsize=9)
    else:
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
