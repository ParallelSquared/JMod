"""The channel-FDR plots of a run, into <run>/channel_fdr/.

The best channel by PredVal in each plex set, IDs vs Channel_Qvalue, each
channel's IDs retained at the FDR threshold, the fitted model's feature
importance, and an ECDF of every channel feature (into ECDF/), one line per
channel, the decoy channel set apart: a look at which features tell a channel
that is there from one that is not.  --channel_fdr_plots "all" (the default)
plots every precursor together; an amino acid, such as "K", the peptides
ending in it and the rest side by side, in every plot but the importance --
under a K-tagging tag every y ion of a K-ending peptide carries the tag, so
the two behave differently; "off" makes none of them.  Whatever it says, a
fitted run always gets logit(ChannelScore) per channel with each channel's
score cutoff at the FDR threshold (plot_channel_scores).  For a written feature
table on its own: scripts/channel_fdr_plots.py.
"""
import multiprocessing
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from src.channel_fdr import (IMPORTANCE_COLUMN, IMPORTANCE_SPREAD_COLUMN, SCORE_COLUMNS, _AMINO_ACIDS,
                             _ID_COLUMNS, read_feature_importance)
from src.logger import logger

ECDF_FOLDER = "ECDF"
# The x axis shows the central 99% of the values: a handful of extreme ones
# would otherwise squash every channel into a corner.  Only a zoom: the
# fractions count every value
CLIP_PCT = 99.5
DPI = 120
# An ECDF panel's size, and the figure's margins (left, right, bottom, top, and
# between the panels), in inches
ECDF_PANEL_SIZE = (8.0, 5.0)
ECDF_MARGINS = (0.75, 0.25, 0.65, 0.75, 0.85)
# An ECDF line is drawn through at most this many of its steps: a panel is
# under 1,000 pixels wide, so more draw the same line, only slower
ECDF_MAX_POINTS = 2000


def feature_columns(features):
    """Every column of the table that is a feature: all but the identity ones
    and the channel FDR's own outputs (ChannelScore, Channel_Qvalue), the
    model's fit to the features rather than evidence of its own."""
    return [c for c in features.columns if c not in _ID_COLUMNS + SCORE_COLUMNS]


def panels(features, split):
    """[(panel title, the rows it plots)]: one panel of every row for "all"; for
    an amino acid ("K", "E"), the peptides ending in it and all the others."""
    if split == "all":
        return [("", features)]
    ends = (pl.col("last_aa_code") == _AMINO_ACIDS.index(split)).fill_null(False)
    return [(f"C-terminal {split}", features.filter(ends)), ("other C-terminus", features.filter(~ends))]


def _by_channel(features, column):
    """[(channel, is the decoy channel, finite values, number null)], in channel order."""
    out = []
    # the column alone, not the table: grouping copies every column it is given
    rows = features.select("channel", "is_decoy_channel", column)
    for (channel, decoy), group in sorted(rows.group_by(["channel", "is_decoy_channel"]),
                                          key=lambda kv: kv[0][0]):
        values = group[column].cast(pl.Float64).to_numpy()
        finite = values[np.isfinite(values)]          # null is NaN here
        out.append((channel, bool(decoy), finite, values.size - finite.size))
    return out


def _styles(groups):
    """The real channels on one colour ramp, the decoy channel a dashed black line,
    so "the real channels together, the decoy apart" reads without the legend."""
    real = [g for g in groups if not g[1]]
    ramp = plt.get_cmap("viridis")(np.linspace(0.05, 0.85, max(len(real), 1)))
    colours = iter(ramp)
    return [dict(color="black", linestyle="--", linewidth=2.0) if decoy
            else dict(color=next(colours), linestyle="-", linewidth=1.4)
            for _, decoy, _, _ in groups]


def _label(channel, decoy, values, n_null):
    return (f"{channel:g}{' (decoy)' if decoy else ''}: n={values.size}"
            + (f", {n_null} null" if n_null else ""))


def ks_distance(a, b):
    """Two-sample Kolmogorov-Smirnov statistic, max |ECDF_a - ECDF_b|; NaN if
    either is empty."""
    if a.size == 0 or b.size == 0:
        return np.nan
    a, b = np.sort(a), np.sort(b)
    both = np.concatenate([a, b])
    return float(np.max(np.abs(np.searchsorted(a, both, side="right") / a.size
                                - np.searchsorted(b, both, side="right") / b.size)))


def _legend(ax, groups):
    """The legend, titled with the KS distance between the real channels pooled
    and the decoy channel; a note instead in a panel with nothing drawn."""
    if not any(v.size for _, _, v, _ in groups):
        ax.text(0.5, 0.5, "no values", transform=ax.transAxes, ha="center", va="center", color="grey")
        return
    real = [v for _, decoy, v, _ in groups if not decoy]
    decoy = [v for _, is_decoy, v, _ in groups if is_decoy]
    ks = ks_distance(np.concatenate(real) if real else np.zeros(0),
                     np.concatenate(decoy) if decoy else np.zeros(0))
    ax.legend(fontsize=8, title_fontsize=8,
              title=None if np.isnan(ks) else f"KS real vs decoy: {ks:.2f}")


def _range(values):
    """The x axis: the values' central CLIP_PCT, and half a unit either side
    when they are whole numbers (counts, a yes/no), so the steps at either end
    are not drawn on the axes' edges."""
    lo, hi = np.percentile(values, [100 - CLIP_PCT, CLIP_PCT])
    return (lo - 0.5, hi + 0.5) if lo == hi or np.all(values == np.round(values)) else (lo, hi)


def ecdf_steps(values):
    """The ECDF of *values* as (x, y) steps, through at most ECDF_MAX_POINTS of
    them, evenly spaced in rank with the first and last kept: off by at most
    1 / ECDF_MAX_POINTS in y, less than a pixel."""
    x = np.sort(values)
    keep = (np.arange(x.size) if x.size <= ECDF_MAX_POINTS
            else np.unique(np.linspace(0, x.size - 1, ECDF_MAX_POINTS).round().astype(int)))
    return x[keep], (keep + 1) / x.size


def _draw_ecdf(ax, groups, x_range):
    """Monotone lines do not hide one another, and the largest vertical gap
    between two of them is their KS distance."""
    for (channel, decoy, values, n_null), style in zip(groups, _styles(groups)):
        if values.size == 0:
            continue
        ax.plot(*ecdf_steps(values), drawstyle="steps-post",
                label=_label(channel, decoy, values, n_null), **style)
    ax.set_xlim(*x_range)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.set_ylabel("Cumulative fraction")


def _ecdf_figure(column, panel_groups, x_range, path):
    """*column*'s ECDFs, one panel per entry of *panel_groups* [(title,
    groups)], side by side on the same x axis, into *path*; left unwritten if
    it cannot be (open in a viewer, say).

    The layout is fixed, not tight_layout's, which took half of the time: the
    y axis is always 0 to 1, so its labels are always as wide, and the x
    axis's, with any offset ("1e-6"), fit the bottom margin.
    """
    n = len(panel_groups)
    width = ECDF_PANEL_SIZE[0] * n
    left, right, bottom, top, gap = ECDF_MARGINS
    fig, axes = plt.subplots(1, n, figsize=(width, ECDF_PANEL_SIZE[1]), squeeze=False, sharex=True)
    fig.subplots_adjust(left=left / width, right=1 - right / width, bottom=bottom / ECDF_PANEL_SIZE[1],
                        top=1 - top / ECDF_PANEL_SIZE[1],
                        wspace=gap * n / (width - left - right - gap * (n - 1)))
    for ax, (title, groups) in zip(axes[0], panel_groups):
        _draw_ecdf(ax, groups, x_range)
        ax.set_xlabel(column)
        if title:
            ax.set_title(title, fontsize=10)
        _legend(ax, groups)
    fig.suptitle(column)
    try:
        fig.savefig(path, dpi=DPI)
    except OSError:
        pass
    finally:
        plt.close(fig)


def _ecdf_job(job):
    return _ecdf_figure(*job)


def _split(split):
    """*split*, whatever its case, as "all" or one amino acid's letter; None
    for "off" (no plots), and None with a warning for anything else."""
    value = str(split).strip()
    if value.lower() in ("all", "off"):
        return None if value.lower() == "off" else "all"
    if len(value) == 1 and value.upper() in _AMINO_ACIDS:
        return value.upper()
    logger.warning(f"Channel FDR plots: not made, the split must be all, off or one amino acid "
                   f"({_AMINO_ACIDS}), not {split!r}")
    return None


def plot_feature_ecdfs(features, folder, split="all", workers=1):
    """An ECDF of every feature of *features* (the table run_channel_fdr
    writes), into *folder*/ECDF.  *split*: "all", or an amino acid ("K") for
    the peptides ending in it and the rest as two panels.  A feature with no value in any panel
    is skipped, and a file that cannot be written (open in a viewer, say) is
    left as it is; nothing is logged.  With *workers* > 1 the figures are drawn
    and saved in that many processes (the saving is most of the time).
    Returns the number of features plotted."""
    split = _split(split)
    if split is None:
        return 0
    ecdf_folder = os.path.join(folder, ECDF_FOLDER)
    os.makedirs(ecdf_folder, exist_ok=True)
    parts = panels(features, split)
    jobs = []
    for column in feature_columns(features):
        panel_groups = [(title, _by_channel(rows, column)) for title, rows in parts]
        values = [v for _, groups in panel_groups for _, _, v, _ in groups if v.size]
        if values:
            jobs.append((column, panel_groups, _range(np.concatenate(values)),
                         os.path.join(ecdf_folder, f"{column}.png")))
    workers = max(1, min(workers, os.cpu_count() or 1, len(jobs)))
    if workers == 1:
        for job in jobs:
            _ecdf_job(job)
    else:
        with multiprocessing.get_context("spawn").Pool(workers) as pool:
            pool.map(_ecdf_job, jobs)
    return len(jobs)


IMPORTANCE_PLOT = "feature_importance.png"
# The importance plot shows this many features, the most important; the rest
# are in the CSV
TOP_FEATURES = 30


def plot_feature_importance(importance, folder, top=TOP_FEATURES):
    """The *top* features of *importance* (channel_fdr.feature_importance's
    table), the most important at the top: each one's mean share of the gain
    over the CV folds' models, with their standard deviation as error bars.  In
    the style of the run's own scoring plots (score_model), which draw one of
    these per fold.  Into *folder*/IMPORTANCE_PLOT."""
    shown = importance.sort_values(IMPORTANCE_COLUMN, ascending=False).head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, max(3.0, len(shown) * 0.3)))
    ax.barh(shown.index, shown[IMPORTANCE_COLUMN], xerr=shown[IMPORTANCE_SPREAD_COLUMN], capsize=2,
            error_kw=dict(elinewidth=0.8))
    ax.set_title("Feature Importance" + (f" (top {len(shown)} of {len(importance)})"
                                         if len(shown) < len(importance) else ""))
    ax.set_xlabel("Share of gain, mean +/- SD over the CV folds")
    fig.tight_layout()
    path = os.path.join(folder, IMPORTANCE_PLOT)
    _save(fig, path)
    return path


def _save(fig, path):
    try:
        fig.savefig(path, dpi=DPI)
    except OSError as error:
        # e.g. the file is open in a viewer on Windows
        logger.warning(f"Channel FDR: could not write {path} ({error})")
    finally:
        plt.close(fig)


BEST_CHANNEL_PLOT = "best_channel_by_predval.png"
IDS_PLOT = "ids_vs_channel_qvalue.png"
RETAINED_PLOT = "ids_retained_at_channel_qvalue.png"
# The x axis of IDS_PLOT runs from 0 to this
Q_RANGE = 0.1


def _set_keys(features):
    """The columns that name a plex set: its untagged precursor (in a time
    channel, on timeplex)."""
    return ["untag_prec"] + (["time_channel"] if "time_channel" in features.columns else [])


def best_channel_counts(features):
    """{channel: the number of plex sets in which it has the highest PredVal}.
    A set where several channels tie for it (PredVal saturates near 1) counts
    1/k for each of the k, so the counts add up to the number of sets."""
    best = (features
            .with_columns(pl.col("PredVal").max().over(_set_keys(features)).alias("__best"))
            .filter(pl.col("PredVal") == pl.col("__best"))
            .with_columns((1.0 / pl.len().over(_set_keys(features))).alias("__credit")))
    totals = best.group_by("channel").agg(pl.col("__credit").sum())
    return {channel: credit for channel, credit in sorted(totals.iter_rows())}


def tied_sets(features):
    """The number of plex sets in which more than one channel has the highest PredVal."""
    keys = _set_keys(features)
    at_best = pl.col("PredVal") == pl.col("PredVal").max().over(keys)
    return int(features.filter(at_best).group_by(keys).len().filter(pl.col("len") > 1).height)


def plot_best_channel(features, folder, split="all"):
    """Bars: in how many plex sets each channel has the highest PredVal, of the
    sets the channel FDR scored.  If one channel wins far more often than the
    others, PredVal carries which channel a row is, as well as whether it is
    there.  Into *folder*/BEST_CHANNEL_PLOT."""
    parts = panels(features, split)
    fig, axes = plt.subplots(1, len(parts), figsize=(7 * len(parts), 4.5), squeeze=False)
    for ax, (title, rows) in zip(axes[0], parts):
        groups = _by_channel(rows, "PredVal")
        counts = best_channel_counts(rows)
        n_sets = sum(counts.values())
        for i, ((channel, decoy, _, _), style) in enumerate(zip(groups, _styles(groups))):
            count = counts.get(channel, 0.0)
            ax.bar(i, count, color="white" if decoy else style["color"], edgecolor="black" if decoy else None,
                   hatch="//" if decoy else None, linewidth=1.2 if decoy else 0)
            ax.text(i, count, f"{count / n_sets:.0%}" if n_sets else "", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(range(len(groups)),
                      [f"{channel:g}{' (decoy)' if decoy else ''}" for channel, decoy, _, _ in groups])
        ax.set_xlabel("Channel")
        ax.set_ylabel("Plex sets where it has the highest PredVal")
        n_tied = tied_sets(rows)
        ties = f", {n_tied} tied (split between the tied channels)" if n_tied else ""
        ax.set_title(f"{title + ': ' if title else ''}{n_sets:.0f} sets{ties}", fontsize=10)
    fig.suptitle("Best channel by PredVal")
    fig.tight_layout()
    _save(fig, os.path.join(folder, BEST_CHANNEL_PLOT))


def plot_ids_vs_channel_qvalue(features, folder, split="all", fdr_threshold=0.01):
    """Each real channel's number of precursors at Channel_Qvalue <= q, against
    q, with the run's threshold marked.  Into *folder*/IDS_PLOT; nothing if
    there are no channel q-values."""
    if not _has_channel_qvalues(features):
        logger.info("Channel FDR plots: no Channel_Qvalue, so no IDs vs Channel_Qvalue plot")
        return
    parts = panels(features, split)
    fig, axes = plt.subplots(1, len(parts), figsize=(7 * len(parts), 4.5), squeeze=False, sharey=True)
    for ax, (title, rows) in zip(axes[0], parts):
        groups = _by_channel(rows, "Channel_Qvalue")
        for (channel, decoy, q, _), style in zip(groups, _styles(groups)):
            if decoy or q.size == 0:            # the decoy channel has no q-values
                continue
            n_at = int((q <= fdr_threshold).sum())
            # a step at each q-value on the axis, carried on to its edge
            shown = np.sort(q[q <= Q_RANGE])
            ax.step(np.append(shown, Q_RANGE), np.append(np.arange(1, shown.size + 1), shown.size),
                    where="post", label=f"{channel:g}: {n_at} at {fdr_threshold:g}", **style)
        ax.axvline(fdr_threshold, color="grey", linestyle=":", linewidth=1)
        ax.set_xlim(0, Q_RANGE)
        ax.set_xlabel("Channel_Qvalue")
        ax.set_ylabel("Precursors at or below it")
        ax.grid(alpha=0.3)
        if title:
            ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8, loc="lower right")       # the curves start high: that corner is empty
    fig.suptitle("IDs vs Channel_Qvalue")
    fig.tight_layout()
    _save(fig, os.path.join(folder, IDS_PLOT))


def _has_channel_qvalues(features):
    return "Channel_Qvalue" in features.columns and features["Channel_Qvalue"].drop_nulls().len() > 0


def retained_counts(features, fdr_threshold):
    """{real channel: (its rows at Channel_Qvalue <= *fdr_threshold*, all its
    rows)}, in channel order.  The rows are the IDs the channel FDR scored."""
    return {channel: (int((q <= fdr_threshold).sum()), q.size + n_null)
            for channel, decoy, q, n_null in _by_channel(features, "Channel_Qvalue") if not decoy}


def _lighter(colour, amount=0.55):
    """*colour* mixed towards white by *amount*."""
    return tuple(c + (1 - c) * amount for c in colour[:3])


def _is_dark(colour):
    """Whether white text reads on *colour* (by its luma); else black."""
    r, g, b = colour[:3]
    return 0.299 * r + 0.587 * g + 0.114 * b < 0.6


def plot_ids_retained(features, folder, split="all", fdr_threshold=0.01):
    """A stacked bar per real channel: of its IDs, those retained at
    Channel_Qvalue <= *fdr_threshold* (the channel's colour, with the share
    retained) and those removed (a lighter shade of it).  Into
    *folder*/RETAINED_PLOT; nothing if there are no channel q-values."""
    if not _has_channel_qvalues(features):
        return
    parts = panels(features, split)
    fig, axes = plt.subplots(1, len(parts), figsize=(7 * len(parts), 4.5), squeeze=False, sharey=True)
    for ax, (title, rows) in zip(axes[0], parts):
        counts = retained_counts(rows, fdr_threshold)
        # the real channels' colours, as in the other plots
        colours = [s["color"] for s in _styles([(c, False, None, None) for c in counts])]
        for i, ((kept, n), colour) in enumerate(zip(counts.values(), colours)):
            ax.bar(i, kept, color=colour, label="retained" if i == 0 else None)
            ax.bar(i, n - kept, bottom=kept, color=_lighter(colour), label="removed" if i == 0 else None)
            if n:
                ax.text(i, kept / 2, f"{kept / n:.0%}", ha="center", va="center", fontsize=8,
                        color="white" if _is_dark(colour) else "black")
        ax.set_xticks(range(len(counts)), [f"{channel:g}" for channel in counts])
        ax.set_xlabel("Channel")
        ax.set_ylabel("IDs")
        if title:
            ax.set_title(title, fontsize=10)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.12)       # room for the legend above the bars
        ax.legend(fontsize=8, loc="upper right", ncol=2)
    fig.suptitle(f"IDs retained at Channel_Qvalue <= {fdr_threshold:g}")
    fig.tight_layout()
    _save(fig, os.path.join(folder, RETAINED_PLOT))


SCORE_PLOT = "channel_score_by_channel.png"
# logit(ChannelScore) is drawn with the score clipped to [SCORE_CLIP,
# 1 - SCORE_CLIP]: it saturates near 0 and 1, and the logit of those runs off
SCORE_CLIP = 1e-4
# A channel with fewer scores than this gets its median but no violin
MIN_VIOLIN = 20


def score_cutoffs(features, fdr_threshold):
    """{real channel: its lowest ChannelScore at Channel_Qvalue <=
    *fdr_threshold*, None if no row gets there}, in channel order: the score
    the channel's q-values cut at."""
    rows = features.filter(~pl.col("is_decoy_channel"))
    passed = (rows.filter(pl.col("Channel_Qvalue") <= fdr_threshold)
              .group_by("channel").agg(pl.col("ChannelScore").min()))
    lowest = dict(passed.iter_rows())
    return {channel: lowest.get(channel) for channel in sorted(rows["channel"].unique().to_list())}


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), SCORE_CLIP, 1 - SCORE_CLIP)
    return np.log(p / (1 - p))


def _violin(ax, values, x, colour, side=None):
    """A violin of *values* at *x*, its median as a white line; with *side*
    "left" or "right", only that half (two halves make a split violin)."""
    if values.size >= MIN_VIOLIN:
        body = ax.violinplot(values, positions=[x], widths=0.8, showextrema=False, bw_method=0.15)["bodies"][0]
        if side is not None:
            verts = body.get_paths()[0].vertices
            verts[:, 0] = np.minimum(verts[:, 0], x) if side == "left" else np.maximum(verts[:, 0], x)
        body.set_facecolor(colour)
        body.set_edgecolor("none")
        body.set_alpha(0.9)
    if values.size:
        lo, hi = {None: (x - 0.18, x + 0.18), "left": (x - 0.25, x), "right": (x, x + 0.25)}[side]
        ax.plot([lo, hi], [np.median(values)] * 2, color="white", linewidth=1.6, solid_capstyle="butt")


def _passing(rows, channel, fdr_threshold):
    """(rows of *channel* at Channel_Qvalue <= *fdr_threshold*, its rows with a score)."""
    rows = rows.filter((pl.col("channel") == channel) & pl.col("ChannelScore").is_not_null())
    return int((rows["Channel_Qvalue"] <= fdr_threshold).sum()), rows.height


def plot_channel_scores(features, folder, split="all", fdr_threshold=0.01):
    """A violin per channel of logit(ChannelScore), the decoy channel in grey,
    with each channel's median (white) and, for a real channel, the score its
    Channel_Qvalue cuts at at *fdr_threshold* (black; score_cutoffs), and its
    rows passing out of those scored above it.  Where the cutoff sits against
    the scores shows how much the decoy channel holds each channel back.
    *split*: an amino acid ("K"): split violins, the peptides ending in it on
    the left (the channel's colour) and the rest on the right (a lighter shade
    of it), each half with its own median and counts, the cutoff across both
    (one per channel: the q-values are the channel's); anything else, whole
    violins.  Into *folder*/SCORE_PLOT; nothing if there are no scores (no fit)."""
    if "ChannelScore" not in features.columns or features["ChannelScore"].drop_nulls().len() == 0:
        logger.info("Channel FDR plots: no ChannelScore, so no score by channel plot")
        return
    value = str(split).strip().upper()
    halves = panels(features, value if len(value) == 1 and value in _AMINO_ACIDS else "all")
    sides = [None] if len(halves) == 1 else ["left", "right"]
    # the halves' names: "K-term", and the rest, which under trypsin end in R
    names = [] if len(halves) == 1 else [f"{value}-term", {"K": "R-term", "R": "K-term"}.get(value, "other")]
    has_q = _has_channel_qvalues(features)
    cutoffs = score_cutoffs(features, fdr_threshold) if has_q else {}
    groups = _by_channel(features, "ChannelScore")
    fig, ax = plt.subplots(figsize=(max(5.0, (0.9 if len(sides) == 1 else 1.3) * len(groups) + 1.5), 4.8))
    for i, ((channel, decoy, _, _), style) in enumerate(zip(groups, _styles(groups))):
        colour = (0.5, 0.5, 0.5) if decoy else style["color"]
        notes = []
        for side, (_, rows) in zip(sides, halves):
            scores = rows.filter(pl.col("channel") == channel)["ChannelScore"].drop_nulls().to_numpy()
            _violin(ax, _logit(scores), i, _lighter(colour, 0.5) if side == "right" else colour, side)
            if decoy or not has_q:
                notes.append(f"n={scores.size:,}")
            else:
                kept, n = _passing(rows, channel, fdr_threshold)
                # one line per half when split; whole violins have room for two
                notes.append((f"{kept:,} / {n:,}" + (" " if names else "\n") + f"({kept / n:.0%})") if n else "0")
        if cutoffs.get(channel) is not None and not decoy:
            ax.plot([i - 0.42, i + 0.42], [_logit(cutoffs[channel])] * 2, color="black", linewidth=1.6)
        ax.annotate("\n".join(notes), (i, 1.0), xycoords=("data", "axes fraction"), ha="center", va="bottom",
                    fontsize=7 if names else 7.5)
    if names:     # each line of counts' half, to their left
        ax.annotate("\n".join(f"{name}:" for name in names), (0, 1.0), xycoords="axes fraction", ha="right",
                    va="bottom", fontsize=7, fontweight="bold")
    ax.set_xticks(range(len(groups)), [f"{channel:g}{chr(10) + '(decoy)' if decoy else ''}"
                                       for channel, decoy, _, _ in groups])
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.set_xlabel("Channel")
    ax.set_ylabel("logit(ChannelScore)")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    handles = [plt.Line2D([0], [0], color="black", linewidth=1.6, label=f"Channel_Qvalue = {fdr_threshold:g}")]
    if names:
        from matplotlib.patches import Patch
        handles += [Patch(facecolor=(0.35, 0.35, 0.35), label=names[0]),
                    Patch(facecolor=_lighter((0.35, 0.35, 0.35), 0.5), label=names[1])]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=8, frameon=False)
    fig.suptitle("ChannelScore", fontsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    _save(fig, os.path.join(folder, SCORE_PLOT))


def plot_channel_fdr(features, folder, importance=None, split="all", fdr_threshold=0.01, workers=1):
    """The channel-FDR plots of a run into *folder*: each feature's ECDF,
    drawn in *workers* processes, the best channel by PredVal, IDs vs
    Channel_Qvalue, the IDs it retains at *fdr_threshold*, and the feature
    importance if there is one.  *split*: "all"; an amino acid ("K") for the
    peptides ending in it and the rest as two panels (not the feature
    importance, which is one model's); or "off", no plots."""
    split = _split(split)
    if split is None:
        return
    plot_feature_ecdfs(features, folder, split=split, workers=workers)
    plot_best_channel(features, folder, split=split)
    plot_ids_vs_channel_qvalue(features, folder, split=split, fdr_threshold=fdr_threshold)
    plot_ids_retained(features, folder, split=split, fdr_threshold=fdr_threshold)
    if importance is not None:
        plot_feature_importance(importance, folder)
