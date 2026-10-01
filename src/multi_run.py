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
from matplotlib.patches import Patch
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
# One colour per channel of a run, in this fixed order: the palette's eight
# categorical slots and a ninth, purple, validated with them (adjacent pairs,
# light surface).  More channels than this fall back to _BLUE for all
_CHANNEL_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
                   "#008300", "#4a3aa7", "#e34948", "#9b4dca"]


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
                          "07_precursors_lost_to_global_q.png", "11_precursor_best_score_run.png",
                          "03_precursor_data_completeness.png")
_PROTEIN_LEVEL = _Level("protein", "protein", "global protein q-value", "02_proteins_per_run.png",
                        "08_proteins_lost_to_global_q.png", "12_protein_best_score_run.png",
                        "04_protein_data_completeness.png")

# The columns that can tell an ID's channel apart within a run, and their names
# in the plots, outermost first: within a run, time channels group channels.
# Those that vary in an experiment make its (run, channel) units
_CHANNEL_COLUMNS = {"time_channel": "time channel", "silac_channel": "SILAC channel", "channel": "channel"}


class _IdLevel(NamedTuple):
    """One q-value an ID can pass in a (run, channel), for the ID and
    completeness plots."""
    column: str          # the per-run table's column saying whether the ID passes; None: every ID does
    label: str           # in the legends
    shade: str           # its shade of the channel's colour: "dark", "full" or "light"


# From the strictest.  The run-level IDs pass the best channel's q-value;
# the global q-value is per ID over the experiment, so it can sit anywhere
# below them.  ##TODO Channel Qvalue, the strictest, once JMod has
# channel-level FDR: add _IdLevel("channel_qvalue_pass", "Channel Qvalue",
# "darker") first, a "darker" shade, and its column in precursors_per_run and
# proteins_per_run
_ID_LEVELS = [
    _IdLevel("qvalue_pass", "Qvalue", "dark"),
    _IdLevel("retained", "Global Qvalue", "full"),
    _IdLevel(None, "BestChannel Qvalue", "light"),
]

# plot_proteins_lost_by_precursor_count's last column holds this many
# precursors and more, so a long tail of large proteins stays readable
_MAX_PRECURSOR_COLUMN = 10


def combine_runs(completed_run_folders, experiment_dir, target_decoy_ratio, fdr_threshold):
    """Combine the completed runs into one table, and plot it.

    *completed_run_folders* maps each completed run's index to its results
    folder.  Writes combined_filtered_IDs.parquet in *experiment_dir*: every
    run's IDs that pass the run-level FDR, with a global q-value per untagged
    precursor and per protein.  Nothing is filtered on the global q-values;
    that is left to the user.  The plots and run_index.txt go in an
    experiment_results folder beside it; those that compare runs only when
    there are enough runs to compare.  Each is datestamped if its name is
    taken.  Returns the combined table.
    """
    logger.info("")
    logger.info(f"Combining the results of {len(completed_run_folders)} run(s)")
    df, untag_prec_global_qs, protein_global_qs = collect_results(completed_run_folders, target_decoy_ratio,
                                                                  fdr_threshold)

    combined_path = datestamped(os.path.join(experiment_dir, "combined_filtered_IDs.parquet"))
    df.write_parquet(combined_path)
    logger.info(f"Combined IDs written to {os.path.abspath(combined_path)}")
    results_dir = datestamped(os.path.join(experiment_dir, "experiment_results"))
    os.makedirs(results_dir)
    write_run_index(df, completed_run_folders, results_dir)
    logger.info(f"Experiment results written to {os.path.abspath(results_dir)}")

    # Each run's channels (mass tag, SILAC or time channels) are plotted apart
    units = _Units(df, sorted(completed_run_folders))
    precursors = precursors_per_run(df, units, fdr_threshold)
    proteins = proteins_per_run(df, units, fdr_threshold)
    # Made in the order of their numbers: identifications, completeness,
    # quantity, then the global q-values' diagnostics, precursors before
    # proteins.  Those that compare units are made only when there are enough
    # of them.  The channels' colours have a key of their own
    if units.columns:
        plot_channel_colors(units, results_dir)
    plot_ids_per_run(precursors, _PRECURSOR_LEVEL, units, fdr_threshold, results_dir)
    plot_ids_per_run(proteins, _PROTEIN_LEVEL, units, fdr_threshold, results_dir)
    if units.n_units >= 2:
        plot_data_completeness(precursors, _PRECURSOR_LEVEL, units, fdr_threshold, results_dir)
        plot_data_completeness(proteins, _PROTEIN_LEVEL, units, fdr_threshold, results_dir)
    plot_intensity_per_run(precursors, units, results_dir)
    if units.n_units >= 3:
        plot_run_correlation(precursors, units, results_dir)
    if units.n_units >= 2:
        plot_lost_to_global_q(precursors, _PRECURSOR_LEVEL, units, fdr_threshold, results_dir)
        plot_lost_to_global_q(proteins, _PROTEIN_LEVEL, units, fdr_threshold, results_dir)
        plot_proteins_lost_by_precursor_count(df, units, fdr_threshold, results_dir)
    # Not made for now (10-12): summed intensity, and which run the best scores came from
    # plot_summed_intensity_per_run(precursors, units, results_dir)
    # if units.n_units >= 2:
    #     plot_best_score_run(untag_prec_global_qs, _PRECURSOR_LEVEL, units, results_dir)
    #     plot_best_score_run(protein_global_qs, _PROTEIN_LEVEL, units, results_dir)
    return df


def write_run_index(df, completed_run_folders, results_dir):
    """run_index.txt: each run's index in the plots and tables (run_idx), its
    data file and its results folder, tab-separated."""
    data_files = dict(df.group_by("run_idx").agg(pl.col("file_name").first()).iter_rows())
    with open(os.path.join(results_dir, "run_index.txt"), "w") as f:
        f.write("run_idx\tdata_file\tresults_folder\n")
        for run_idx in sorted(completed_run_folders):
            f.write(f"{run_idx}\t{data_files.get(run_idx, '')}\t{completed_run_folders[run_idx]}\n")


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
    is_decoy with its best score, the run it came from (BestRun) and that run's
    channel (Best_channel and so on, one per channel column), the number of runs
    and of (run, channel) units it was scored in, and the q-value, named
    *qvalue_name*.
    """
    channel_columns = [c for c in _CHANNEL_COLUMNS if c in df.collect_schema().names()]
    # Ties go to the earliest run, then the lowest channel
    best_first = ["PredVal", "run_idx", *channel_columns]
    descending = [True] + [False] * (1 + len(channel_columns))
    scores = (
        # A decoy shares its target's protein name, so they are told apart by is_decoy
        df.group_by(key, "is_decoy")
        .agg(
            pl.col("PredVal").max().alias("MaxPredVal"),
            pl.col("run_idx").sort_by(best_first, descending=descending).first().alias("BestRun"),
            *(pl.col(c).sort_by(best_first, descending=descending).first().alias(f"Best_{c}")
              for c in channel_columns),
            pl.col("run_idx").n_unique().alias("n_scored_runs"),
            pl.struct("run_idx", *channel_columns).n_unique().alias("n_scored_units"),
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


def precursors_per_run(df, units, fdr_threshold):
    """One row per (run, channel) unit and untagged precursor it identifies,
    for the plots, with the unit's index in *units* (``unit``).

    ``area`` is the plex_Area (0 when there is none), ``log2_area`` its log2
    (null when there is none), ``retained`` whether the precursor passes the
    global q-value, and ``qvalue_pass`` whether it passes its own channel's
    Qvalue, not only through its best channel.
    """
    return units.index(
        df.group_by("run_idx", *units.columns, "untag_prec")
        .agg(
            pl.col("plex_Area").fill_nan(None).sum().alias("area"),
            (pl.col("untag_prec_Global_Qvalue").first() < fdr_threshold).alias("retained"),
            (pl.col("Qvalue") < fdr_threshold).any().alias("qvalue_pass"),
        )
        .with_columns(
            pl.when(pl.col("area") > 0).then(pl.col("area").log(2)).alias("log2_area")
        )
    )


def proteins_per_run(df, units, fdr_threshold):
    """One row per (run, channel) unit and protein it identifies at the
    run-level protein q-value, among its run-level precursor IDs, for the
    plots, with the unit's index in *units* (``unit``).

    ``retained`` is whether the protein passes the global protein q-value, and
    ``qvalue_pass`` whether one of its precursors passes its own channel's
    Qvalue there.  The protein q-value is the run's, over every channel.
    """
    return units.index(
        df.filter(pl.col("Protein_Qvalue") < fdr_threshold)
        .group_by("run_idx", *units.columns, "protein")
        .agg((pl.col("Protein_Global_Qvalue").first() < fdr_threshold).alias("retained"),
             (pl.col("Qvalue") < fdr_threshold).any().alias("qvalue_pass"))
    )


class _Units:
    """The (run, channel) units the plots show, in order: each run's channels.

    The channel columns are those that vary in the experiment (time channel,
    SILAC channel, mass tag channel), outermost first.  With none, as
    label-free, a unit is a run.  Every run is given every channel, so one with
    no IDs in a channel still shows it, empty.  Each unit's colour follows its
    innermost channel.
    """

    def __init__(self, df, run_idxs):
        self.run_idxs = run_idxs
        self.columns = [c for c in _CHANNEL_COLUMNS if c in df.columns and df[c].n_unique() > 1]
        runs = pl.DataFrame({"run_idx": run_idxs}, schema={"run_idx": df.schema["run_idx"]})
        if self.columns:
            channels = df.select(self.columns).unique().sort(self.columns)
            runs = runs.join(channels, how="cross")
        self.frame = runs.sort("run_idx", *self.columns).with_row_index("unit")
        self.channels = [tuple(row) for row in channels.rows()] if self.columns else [()]
        self.n_units = self.frame.height
        # e.g. "channel", and "(run, channel) pairs" for the axis labels
        self.channel_name = ", ".join(_CHANNEL_COLUMNS[c] for c in self.columns)
        self.plural = f"(run, {self.channel_name}) pairs" if self.columns else "runs"
        self.axis_name = ", ".join(["Run", *(_CHANNEL_COLUMNS[c].capitalize() for c in self.columns)])
        self.labels = [self._label(row) for row in self.frame.select("run_idx", *self.columns).rows()]

        # The colour of each channel of a run, by its innermost channel
        innermost = sorted({ch[-1] for ch in self.channels}) if self.columns else [None]
        colors = _CHANNEL_COLORS if len(innermost) <= len(_CHANNEL_COLORS) else [_BLUE] * len(innermost)
        self.inner_colors = dict(zip(innermost, colors))
        self.channel_colors = [self.inner_colors[ch[-1] if ch else None] for ch in self.channels]
        self.colors = self.channel_colors * len(run_idxs)

    def index(self, table, run_column="run_idx", channel_columns=None):
        """*table* with each row's unit index (``unit``), matched on its run and
        channel columns (by default named as in the IDs)."""
        on = [run_column, *(channel_columns or self.columns)]
        keys = self.frame.select("unit", "run_idx", *self.columns)
        return table.join(keys, left_on=on, right_on=["run_idx", *self.columns], how="inner")

    def counts(self, table, column=None):
        """Per unit, in order: *table*'s rows, or its rows where *column* is true."""
        rows = table if column is None else table.filter(pl.col(column))
        by_unit = dict(rows.group_by("unit").len().iter_rows())
        return np.array([by_unit.get(u, 0) for u in range(self.n_units)])

    def id_levels(self):
        """The q-value levels to plot.  Without channels an ID's Qvalue is its
        best channel's, so the two are one level, named Qvalue."""
        if self.columns:
            return _ID_LEVELS
        return [level if level.column else level._replace(label="Qvalue")
                for level in _ID_LEVELS if level.column != "qvalue_pass"]

    def channel_legend_handles(self):
        """A legend entry per innermost channel colour, when there are channels."""
        if not self.columns:
            return []
        name = _CHANNEL_COLUMNS[self.columns[-1]].capitalize()
        return [Patch(facecolor=color, label=f"{name} {self._value(value)}")
                for value, color in self.inner_colors.items()]

    def _label(self, row):
        # e.g. "3", or "3·8" for run 3's channel 8
        return "·".join(self._value(v) for v in row)

    @staticmethod
    def _value(v):
        return f"{v:g}" if isinstance(v, float) else str(v)


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
    # delta: fit at points 1% of the RT range apart and interpolate between
    # them, the same curve in a fraction of the time
    curve = lowess(delta[kept], rt[kept], frac=0.2, delta=0.01 * np.ptp(rt[kept]), return_sorted=True)
    return delta, kept, curve


def _unit_name(unit, unit_columns):
    # e.g. "run 3", or "run 3, time channel 1" on timeplex
    parts = [f"run {unit[0]}"]
    if len(unit_columns) > 1:
        parts.append(f"time channel {int(unit[1])}")
    return ", ".join(parts)


def plot_ids_per_run(per_run, level, units, fdr_threshold, results_dir):
    """Columns, grouped by run: each (run, channel)'s precursors (or proteins,
    per *level*) at each q-value level (units.id_levels).

    Each level's count is its own column from zero, drawn largest first, so
    every level's top is at its count and the smaller ones sit in front: the
    run-level IDs (the best channel's q-value) at the top, and below them the
    channel's own Qvalue and the global q-value, in whichever order their
    counts fall.  *per_run* is precursors_per_run's (proteins_per_run's) table.
    """
    levels = units.id_levels()
    counts = np.array([units.counts(per_run, lvl.column) for lvl in levels])  # level x unit

    noun = level.noun.capitalize()
    fig, ax = plt.subplots(figsize=(_figure_width(units.n_units), 4))
    x, width = _unit_positions(fig, units)
    # Largest first at every unit: rank r draws each unit's r-th largest level
    order = np.argsort(-counts, axis=0, kind="stable")
    for rank in range(len(levels)):
        at_rank = order[rank]
        ax.bar(x, counts[at_rank, np.arange(units.n_units)], width=width,
               color=[_shade(units.colors[u], levels[i].shade) for u, i in enumerate(at_rank)],
               # A thin surface-coloured edge keeps the levels and neighbouring columns apart
               edgecolor=_SURFACE_COLOR, linewidth=0.8)
    if not units.columns:
        _label_columns(ax, x, counts.max(axis=0), "{:,}")
    _style_axes(ax, units.run_idxs)
    _unit_axis(ax, units, x)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylabel(f"{noun}s", color=_MUTED_COLOR)
    ax.set_title(f"{noun}s Per Run", loc="center", color=_TEXT_COLOR)
    # The levels' shades, shown in the channels' first colour
    level_handles = [Patch(facecolor=_shade(units.channel_colors[0], lvl.shade), edgecolor=_SURFACE_COLOR,
                           linewidth=0.8, label=f"{lvl.label} < {fdr_threshold:g}") for lvl in levels]
    _legend(ax, below=True, handles=level_handles)
    _save(fig, results_dir, level.ids_file)


def plot_summed_intensity_per_run(precursors, units, results_dir):
    """Columns, grouped by run: log2 of each (run, channel)'s summed plex_Area
    over its run-level IDs."""
    summed = dict(precursors.group_by("unit").agg(pl.col("area").sum()).iter_rows())
    log2_summed = np.array([math.log2(summed[u]) if summed.get(u, 0) > 0 else 0
                            for u in range(units.n_units)])

    fig, ax = plt.subplots(figsize=(_figure_width(units.n_units), 4))
    x, width = _unit_positions(fig, units)
    ax.bar(x, log2_summed, width=width, color=units.colors, edgecolor=_SURFACE_COLOR, linewidth=0.8)
    _label_columns(ax, x, log2_summed, "{:.1f}", vertical=bool(units.columns))
    _style_axes(ax, units.run_idxs)
    _unit_axis(ax, units, x)
    if log2_summed.max() > 0:
        ax.set_ylim(top=log2_summed.max() * 1.1)  # room above the column labels
    ax.set_ylabel("log2 sum plex_Area", color=_MUTED_COLOR)
    ax.set_title("Summed intensity per run", loc="center", color=_TEXT_COLOR)
    _save(fig, results_dir, "10_summed_intensity_per_run.png")


def plot_intensity_per_run(precursors, units, results_dir):
    """Box plots, grouped by run: the distribution of log2 plex_Area in each
    (run, channel).

    Units on the same intensity scale have their medians level.  Whiskers reach
    1.5 x IQR; points beyond them are not drawn, so 90 runs stay readable.
    """
    by_unit = dict(precursors.filter(pl.col("log2_area").is_not_null())
                   .group_by("unit").agg(pl.col("log2_area")).iter_rows())
    data = [np.asarray(by_unit.get(u, [np.nan])) for u in range(units.n_units)]

    fig, ax = plt.subplots(figsize=(_figure_width(units.n_units), 4))
    x, width = _unit_positions(fig, units)
    boxes = ax.boxplot(data, positions=x, widths=width * 0.9,
                       showfliers=False, patch_artist=True, manage_ticks=False,
                       medianprops=dict(linewidth=1.5))
    # Each box in its channel's colour: a light fill, the colour's edge, a dark median
    for u, color in enumerate(units.colors):
        boxes["boxes"][u].set(facecolor=_shade(color, "light"), edgecolor=color)
        for part in ("whiskers", "caps"):
            for line in boxes[part][2 * u:2 * u + 2]:
                line.set_color(color)
        boxes["medians"][u].set_color(_shade(color, "dark"))
    _style_axes(ax, units.run_idxs, zero_baseline=False)
    _unit_axis(ax, units, x)
    ax.set_ylabel("log2 plex_Area", color=_MUTED_COLOR)
    ax.set_title("Intensity per run", loc="center", color=_TEXT_COLOR)
    _save(fig, results_dir, "05_intensity_per_run.png")


def plot_lost_to_global_q(per_run, level, units, fdr_threshold, results_dir):
    """Columns: the precursors (or proteins, per *level*) the global q-value
    removes, by the number of (run, channel) units they pass their own
    channel's Qvalue in.

    Counted on each channel's own Qvalue, not its best channel's, which would
    count every channel of a run as soon as one passes.  *per_run* is
    precursors_per_run's (proteins_per_run's) table.
    """
    per_key = (
        per_run.group_by(level.key)
        .agg(pl.col("unit").filter(pl.col("qvalue_pass")).n_unique().alias("n_units"),
             pl.col("retained").first())
    )
    lost = per_key.filter(~pl.col("retained"))
    lost_by_n_units = dict(lost.group_by("n_units").len().iter_rows())

    x = np.arange(1, units.n_units + 1)
    counts = np.array([lost_by_n_units.get(n, 0) for n in x])

    fig, ax = plt.subplots(figsize=(_figure_width(units.n_units), 4))
    ax.bar(x, counts, width=_bar_width(fig, units.n_units), color=_BLUE)
    _label_columns(ax, x, counts, "{:,}")
    _style_axes(ax, x)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    if not counts.any():
        ax.set_ylim(0, 1)
        ax.text(0.5, 0.5, "None", transform=ax.transAxes, ha="center", va="center", color=_MUTED_COLOR)
    ax.set_xlabel(f"Number of {units.plural.title()} with Qvalue < {fdr_threshold:g}", color=_MUTED_COLOR)
    ax.set_ylabel(f"{level.noun.capitalize()}s", color=_MUTED_COLOR)
    ax.set_title(f"{level.noun.capitalize()}s failing the {level.global_q} ({fdr_threshold:g}): "
                 f"{lost.height:,} of {per_key.height:,}",
                 loc="center", color=_TEXT_COLOR)
    _save(fig, results_dir, level.lost_file)


def plot_best_score_run(global_qs, level, units, results_dir):
    """Which (run, channel) each precursor's (or protein's, per *level*) best
    score, the one the global q-value uses, came from: one panel for targets,
    one for decoys.  *global_qs* is global_qvalues' table.

    Only those scored in every (run, channel) are counted, so each has the same
    chance to hold the best score: if none scores higher than the others, each
    holds about an even share.
    """
    in_every_unit = units.index(global_qs.filter(pl.col("n_scored_units") == units.n_units),
                                run_column="BestRun", channel_columns=[f"Best_{c}" for c in units.columns])
    even_share = 100 / units.n_units

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(_figure_width(units.n_units), 6))
    x, width = _unit_positions(fig, units)
    for ax, is_decoy, name, color in ((axes[0], False, "Targets", _BLUE),
                                      (axes[1], True, "Decoys", _ORANGE)):
        subset = in_every_unit.filter(pl.col("is_decoy") == is_decoy)
        shares = 100 * units.counts(subset) / max(subset.height, 1)

        ax.bar(x, shares, width=width, color=color, edgecolor=_SURFACE_COLOR, linewidth=0.8)
        ax.axhline(even_share, color=_MUTED_COLOR, linewidth=1)
        ax.annotate("even share", xy=(1, even_share), xycoords=("axes fraction", "data"),
                    xytext=(0, 3), textcoords="offset points", ha="right", va="bottom",
                    fontsize=8, color=_MUTED_COLOR)
        _style_axes(ax, units.run_idxs)
        ax.tick_params(axis="x", which="both", length=0)  # the panels share _unit_axis's ticks
        ax.set_ylabel(f"% of {level.noun}s", color=_MUTED_COLOR)
        ax.set_title(f"{name} ({subset.height:,} scored in all {units.plural})",
                     loc="left", fontsize=10, color=_TEXT_COLOR)
    _unit_axis(axes[1], units, x)
    fig.suptitle(f"Run each {level.noun}'s best score came from", color=_TEXT_COLOR)
    _save(fig, results_dir, level.best_run_file)


def plot_data_completeness(per_run, level, units, fdr_threshold, results_dir):
    """Lines: how many precursors (or proteins, per *level*) are identified in
    at least k (run, channel) units, for every k, at each q-value level
    (units.id_levels).  At the global q-value, an ID counts in the units it
    is a run-level ID in, if it passes.

    *per_run* is precursors_per_run's (proteins_per_run's) table.
    """
    k = np.arange(1, units.n_units + 1)
    noun = level.noun.capitalize()
    fig, ax = plt.subplots(figsize=(_figure_width(units.n_units), 4))
    marker = "o" if units.n_units <= 30 else None
    for lvl in units.id_levels():
        passing = per_run if lvl.column is None else per_run.filter(pl.col(lvl.column))
        n_units = passing.group_by(level.key).agg(pl.col("unit").n_unique())["unit"].to_numpy()
        at_least = np.array([(n_units >= i).sum() for i in k])
        # The global q-value's line stays on top where the lines meet
        ax.plot(k, at_least, color=_shade(_BLUE, lvl.shade), linewidth=2, marker=marker, markersize=5,
                zorder=3 if lvl.column == "retained" else 2,
                markeredgecolor=_SURFACE_COLOR, label=f"{lvl.label} < {fdr_threshold:g}")
    _style_axes(ax, k)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel(f"Identified in at least K {units.plural}", color=_MUTED_COLOR)
    ax.set_ylabel(f"{noun}s", color=_MUTED_COLOR)
    ax.set_title(f"{noun} Data Completeness", loc="center", color=_TEXT_COLOR)
    _legend(ax, below=True)
    _save(fig, results_dir, level.completeness_file)


def plot_proteins_lost_by_precursor_count(df, units, fdr_threshold, results_dir):
    """Columns: the proteins the global protein q-value removes, by how many
    untagged precursors identify them at their own channel's Qvalue.

    Over the proteins at the run-level protein q-value in some run, counting
    each one's untagged precursors that pass their own channel's Qvalue in some
    run.  From _MAX_PRECURSOR_COLUMN on, the counts share one column.
    """
    per_protein = (
        df.filter(pl.col("Protein_Qvalue") < fdr_threshold)
        .group_by("protein")
        .agg(pl.col("untag_prec").filter(pl.col("Qvalue") < fdr_threshold).n_unique().alias("n_precursors"),
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
    ax.set_xlabel(f"Number of Untag Precs with Qvalue < {fdr_threshold:g}", color=_MUTED_COLOR)
    ax.set_ylabel("Proteins", color=_MUTED_COLOR)
    ax.set_title(f"Proteins failing the global protein q-value ({fdr_threshold:g}): "
                 f"{lost.height:,} of {per_protein.height:,}",
                 loc="left", color=_TEXT_COLOR)
    _save(fig, results_dir, "09_proteins_lost_by_precursor_count.png")


def plot_run_correlation(precursors, units, results_dir):
    """Heatmap: Pearson correlation of log2 plex_Area between every pair of
    (run, channel) units, over the precursors both quantify.  An outlier shows
    as a pale row and column."""
    # Pairs sharing fewer precursors than this are left blank
    min_shared = 10
    wide = (precursors.filter(pl.col("log2_area").is_not_null())
            .pivot(on="unit", index="untag_prec", values="log2_area")
            .to_pandas())
    wide.columns = [str(c) for c in wide.columns]
    wide = wide.reindex(columns=[str(u) for u in range(units.n_units)])
    corr = wide.corr(min_periods=min_shared).to_numpy()

    n = units.n_units
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
        set_labels([units.labels[i] for i in ticks])
    if units.columns:
        ax.tick_params(axis="x", labelrotation=90)  # run·channel labels are too wide to sit side by side
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
    ax.set_xlabel(units.axis_name, color=_MUTED_COLOR)
    ax.set_ylabel(units.axis_name, color=_MUTED_COLOR)
    ax.set_title("Run-to-run correlation of log2 plex_Area", loc="center", color=_TEXT_COLOR)
    _save(fig, results_dir, "06_run_correlation.png")


def plot_library_size_by_run(targets, fdr_threshold, qvalue_column, mbr_dir):
    """Lines: how large the MBR library grows as runs are added, counting every
    untagged precursor identified in that run or an earlier one.

    The runs are added in the order that grows the library fastest: first the
    one with the most IDs (MBR's reference run), then each time the one adding
    the most not yet in it, on *qvalue_column* (the library's q-value).  One
    line for the run-level IDs of *targets* (the first pass's combined
    targets), one for those that also pass the global q-value, each labelled
    with its final size.
    """
    run_idxs = sorted(targets["run_idx"].unique().to_list())
    ids_by_run = {}
    for column in ("untag_prec_Global_Qvalue", "BestChannel_Qvalue"):
        by_run = dict(targets.filter(pl.col(column) < fdr_threshold)
                      .group_by("run_idx").agg(pl.col("untag_prec").unique()).iter_rows())
        ids_by_run[column] = {r: set(by_run.get(r, [])) for r in run_idxs}
    order = _greedy_run_order(ids_by_run[qvalue_column])

    # On channels the run-level IDs are the best channel's, as in the other plots
    channels = any(c in targets.columns and targets[c].n_unique() > 1 for c in _CHANNEL_COLUMNS)
    run_level = "BestChannel Qvalue" if channels else "Qvalue"
    x = np.arange(1, len(order) + 1)
    fig, ax = plt.subplots(figsize=(_figure_width(len(order)), 4))
    finals = {}
    for column, name, color in (("untag_prec_Global_Qvalue", "Global Qvalue", _BLUE),
                                ("BestChannel_Qvalue", run_level, _LIGHT_BLUE)):
        seen, library_size = set(), []
        for r in order:
            seen |= ids_by_run[column][r]
            library_size.append(len(seen))
        finals[column] = library_size[-1]
        ax.plot(x, library_size, color=color, linewidth=2, marker="o" if len(order) <= 30 else None,
                markersize=5, markeredgecolor=_SURFACE_COLOR, zorder=3 if column == "untag_prec_Global_Qvalue" else 2,
                label=f"{name} < {fdr_threshold:g}")
    # Each line's final size beside its last point; the larger above, the smaller below
    for column, size in finals.items():
        above = size >= max(finals.values())
        ax.annotate(f"{size:,}", xy=(x[-1], size), xytext=(6, 3 if above else -3), textcoords="offset points",
                    ha="left", va="bottom" if above else "top", fontsize=8, color=_MUTED_COLOR)

    _style_axes(ax, x)
    ax.set_xlim(0.4, len(order) + 0.9)  # room for the final sizes on the right
    ax.set_ylim(0, max(max(finals.values()), 1) * 1.08)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    step = math.ceil(len(order) / 30)  # at most about 30 run labels
    ax.set_xticks(x[::step], [str(r) for r in order[::step]])
    ax.set_xlabel("Run", color=_MUTED_COLOR)
    ax.set_ylabel("Untagged Precursors", color=_MUTED_COLOR)
    ax.set_title("MBR library size by run", loc="center", color=_TEXT_COLOR)
    _legend(ax, below=True)
    _save(fig, mbr_dir, "library_size_by_run.png")


def _greedy_run_order(ids_by_run):
    """The runs of *ids_by_run* ({run: set of IDs}) in the order that grows
    their union fastest: first the one with the most IDs, then each time the
    one adding the most not yet included.  Ties go to the lower run index."""
    remaining = {r: set(ids) for r, ids in ids_by_run.items()}
    order = []
    while remaining:
        best = max(sorted(remaining), key=lambda r: len(remaining[r]))
        added = remaining.pop(best)
        order.append(best)
        for ids in remaining.values():
            ids -= added
    return order


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


def _unit_positions(fig, units):
    """The x of each (run, channel)'s column, grouped side by side around its
    run, and the columns' width.

    Within a run the channels are grouped by their outer channels (e.g. a time
    channel's mass tag channels), half a column apart.  A run's group takes
    most of its slot, its columns at most about a quarter of an inch each;
    past the figure's widest they only grow thinner.  With one channel this is
    _bar_width's column centred on the run.
    """
    n_channels = len(units.channels)
    # Each column's place in its run's group, a half-column gap before each new outer group
    places, place = [], 0.0
    for i, ch in enumerate(units.channels):
        if i and ch[:-1] != units.channels[i - 1][:-1]:
            place += 0.5
        places.append(place)
        place += 1
    span = places[-1] + 1
    slot_inches = fig.get_figwidth() * 0.8 / len(units.run_idxs)
    width = min(0.8 / span, 0.25 / slot_inches)
    offsets = (np.array(places) + 0.5 - span / 2) * width
    return np.array([r + o for r in units.run_idxs for o in offsets]), width


def _unit_axis(ax, units, x):
    """The run axis: each run under its group of columns, and when a run's
    channels are grouped by outer channels (e.g. time channels), each outer
    channel beneath its group while they fit.  The innermost channel is shown
    by colour only (plot_channel_colors is the key)."""
    if not units.columns:
        ax.set_xlabel("Run", color=_MUTED_COLOR)
        return
    ax.tick_params(axis="x", which="both", colors=_MUTED_COLOR, length=0)
    ax.set_xlabel(units.axis_name, color=_MUTED_COLOR)
    ax.set_xticks(units.run_idxs, [str(r) for r in units.run_idxs])
    if len(units.columns) == 1 or units.n_units > 60:
        return
    # One label under each run's group of columns sharing their outer channels
    outer = [tuple(label.split("·")[1:-1]) for label in units.labels]
    start = 0
    for i in range(1, len(outer) + 1):
        if i == len(outer) or outer[i] != outer[start]:
            ax.annotate(outer[start][-1], xy=((x[start] + x[i - 1]) / 2, 0),
                        xycoords=("data", "axes fraction"), xytext=(0, -3), textcoords="offset points",
                        ha="center", va="top", fontsize=7, color=_MUTED_COLOR)
            start = i
    ax.tick_params(axis="x", which="major", pad=14)


def plot_channel_colors(units, results_dir):
    """The key to the channels' colours in the other plots: a swatch per
    innermost channel."""
    handles = units.channel_legend_handles()
    n_columns = min(len(handles), 5)
    n_rows = math.ceil(len(handles) / n_columns)
    fig = plt.figure(figsize=(1.6 * n_columns, 0.5 + 0.35 * n_rows))
    # Row by row: matplotlib fills a legend's columns first, so reorder the handles to match
    ordered = [handles[r * n_columns + c] for c in range(n_columns) for r in range(n_rows)
               if r * n_columns + c < len(handles)]
    legend = fig.legend(handles=ordered, loc="center", ncol=n_columns, frameon=False, fontsize=9,
                        title="Channel colours", title_fontsize=10)
    for label in legend.get_texts():
        label.set_color(_MUTED_COLOR)
    legend.get_title().set_color(_TEXT_COLOR)
    _save(fig, results_dir, "00_channel_colors.png")


def _label_columns(ax, x, values, fmt, vertical=False):
    # Value on top of each non-empty column, only while there are few enough to
    # read; *vertical* for narrow grouped columns, whose labels would overlap
    nonzero = [(xi, v) for xi, v in zip(x, values) if v]
    if len(nonzero) > 25:
        return
    if vertical and nonzero:
        # Standing labels are tall: room for them above the tallest column
        ax.set_ylim(top=max(v for _, v in nonzero) * 1.2)
    for xi, v in nonzero:
        ax.annotate(fmt.format(v), xy=(xi, v), xytext=(0, 2), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8, color=_MUTED_COLOR,
                    rotation=90 if vertical else 0)


def _legend(ax, below=False, handles=None):
    # Right of the plot, one entry per line; or *below* it, in one row in the
    # order the entries were drawn (or *handles*' order), anchored to the
    # figure's bottom edge so it clears the x-axis label
    if below:
        if handles is None:
            handles, _ = ax.get_legend_handles_labels()
        # Below the axes' labels: when rows of channel labels reach past the
        # figure's bottom edge, start under them instead
        fig = ax.figure
        axes_bottom = ax.get_tightbbox(fig.canvas.get_renderer()).y0
        axes_bottom = fig.transFigure.inverted().transform((0, axes_bottom))[1]
        top = 0.0 if axes_bottom >= 0 else axes_bottom - 0.02
        legend = fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, top),
                            ncol=len(handles), frameon=False, fontsize=9)
    else:
        legend = ax.legend(loc="upper left", bbox_to_anchor=(1, 1), frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color(_MUTED_COLOR)


def _shade(color, shade):
    """*color* in one of the q-value levels' shades: "full", "dark" or "light".

    Blue takes the palette's own steps (as label-free plots always have); the
    other channel colours are mixed towards white or black to match.
    """
    if shade == "full":
        return color
    if color == _BLUE:
        return {"dark": _DARK_BLUE, "light": _LIGHT_BLUE}[shade]
    rgb = np.array(matplotlib.colors.to_rgb(color))
    rgb = rgb * 0.6 if shade == "dark" else rgb + (1 - rgb) * 0.55
    return matplotlib.colors.to_hex(rgb)


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
