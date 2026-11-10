"""The channel-FDR plots of a run, into <run>/channel_fdr/.

Always: the best channel by PredVal in each plex set, IDs vs Channel_Qvalue,
each channel's IDs retained at the FDR threshold, and the fitted model's
feature importance.  With --channel_fdr_plots also an ECDF of every channel
feature (into ECDF/), one line per channel, the decoy channel set apart: a
look at which features tell a channel that is there from one that is not.
"all" (the flag alone)
plots every precursor together, "KR" the peptides ending in K and in R side by
side, in every plot but the importance -- under a K-tagging tag every y ion of
a K-ending peptide carries the tag, so the two behave differently.  Also runs
on its own, on any written feature table, making all of them:

    python -m src.channel_fdr_plots <run>/channel_fdr/channel_features.parquet [KR]
"""
import os
import sys

if __name__ == "__main__":
    # JMod's config parses the command line when it is imported (through
    # src.channel_fdr), and these arguments are not its options
    _ARGS, sys.argv[1:] = sys.argv[1:], []

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from src.channel_fdr import (IMPORTANCE_COLUMN, IMPORTANCE_SPREAD_COLUMN, _AMINO_ACIDS, _ID_COLUMNS,
                             read_feature_importance)
from src.logger import logger

ECDF_FOLDER = "ECDF"
SPLITS = ("all", "KR")
# The x axis shows the central 99% of the values: a handful of extreme ones
# would otherwise squash every channel into a corner.  Only a zoom: the
# fractions count every value
CLIP_PCT = 99.5
DPI = 120


def feature_columns(features):
    """Every column of the table that is a feature: all but the identity ones."""
    return [c for c in features.columns if c not in _ID_COLUMNS]


def panels(features, split):
    """[(panel title, the rows it plots)]: one panel of every row for "all"; for
    "KR", the peptides ending in K and those ending in R (any other C-terminus
    in neither)."""
    if split == "all":
        return [("", features)]
    return [(f"C-terminal {aa}", features.filter(pl.col("last_aa_code") == _AMINO_ACIDS.index(aa)))
            for aa in "KR"]


def _by_channel(features, column):
    """[(channel, is the decoy channel, finite values, number null)], in channel order."""
    out = []
    for (channel, decoy), group in sorted(features.group_by(["channel", "is_decoy_channel"]),
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
    and the decoy channel."""
    real = [v for _, decoy, v, _ in groups if not decoy]
    decoy = [v for _, is_decoy, v, _ in groups if is_decoy]
    ks = ks_distance(np.concatenate(real) if real else np.zeros(0),
                     np.concatenate(decoy) if decoy else np.zeros(0))
    ax.legend(fontsize=8, title_fontsize=8,
              title=None if np.isnan(ks) else f"KS real vs decoy: {ks:.2f}")


def _range(values):
    lo, hi = np.percentile(values, [100 - CLIP_PCT, CLIP_PCT])
    return (lo - 0.5, hi + 0.5) if lo == hi else (lo, hi)


def _draw_ecdf(ax, groups, x_range):
    """Monotone lines do not hide one another, and the largest vertical gap
    between two of them is their KS distance."""
    for (channel, decoy, values, n_null), style in zip(groups, _styles(groups)):
        if values.size == 0:
            continue
        x = np.sort(values)
        ax.plot(x, np.arange(1, x.size + 1) / x.size, drawstyle="steps-post",
                label=_label(channel, decoy, values, n_null), **style)
    ax.set_xlim(*x_range)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.set_ylabel("Cumulative fraction")


def _ecdf_figure(column, panel_groups, x_range, path):
    """*column*'s ECDFs, one panel per entry of *panel_groups* [(title,
    groups)], side by side on the same x axis."""
    fig, axes = plt.subplots(1, len(panel_groups), figsize=(8 * len(panel_groups), 5), squeeze=False,
                             sharex=True)
    for ax, (title, groups) in zip(axes[0], panel_groups):
        _draw_ecdf(ax, groups, x_range)
        ax.set_xlabel(column)
        if title:
            ax.set_title(title, fontsize=10)
        _legend(ax, groups)
    fig.suptitle(column)
    fig.tight_layout()
    _save(fig, path)


def _split(split):
    """*split* as one of SPLITS, whatever its case; None, with a warning, if it is none of them."""
    known = next((s for s in SPLITS if s.lower() == str(split).lower()), None)
    if known is None:
        logger.warning(f"Channel FDR plots: not made, the split must be one of {', '.join(SPLITS)}, "
                       f"not {split!r}")
    return known


def plot_feature_ecdfs(features, folder, split="all"):
    """An ECDF of every feature of *features* (the table run_channel_fdr
    writes), into *folder*/ECDF.  *split*: "all", or "KR" for the K- and
    R-ending peptides as two panels.  A feature with no value in any panel is
    skipped.  Returns the number of features plotted."""
    split = _split(split)
    if split is None:
        return 0
    ecdf_folder = os.path.join(folder, ECDF_FOLDER)
    os.makedirs(ecdf_folder, exist_ok=True)
    parts = panels(features, split)
    if split == "KR":
        n_other = features.height - sum(rows.height for _, rows in parts)
        if n_other:
            logger.info(f"Channel feature plots: {n_other} precursor-channels ending in neither K nor R "
                        f"are in neither panel")
    empty = []
    for column in feature_columns(features):
        panel_groups = [(title, _by_channel(rows, column)) for title, rows in parts]
        values = [v for _, groups in panel_groups for _, _, v, _ in groups if v.size]
        if not values:
            empty.append(column)
            continue
        _ecdf_figure(column, panel_groups, _range(np.concatenate(values)),
                     os.path.join(ecdf_folder, f"{column}.png"))
    if empty:
        logger.info(f"Channel feature plots: no values to plot for {', '.join(empty)}")
    n = len(feature_columns(features)) - len(empty)
    logger.info(f"Channel feature ECDFs ({split}) for {n} features written to {os.path.abspath(ecdf_folder)}")
    return n


IMPORTANCE_PLOT = "feature_importance.png"
# The importance plot shows this many features, the most important; the rest
# are in the CSV
TOP_FEATURES = 30


def plot_feature_importance(importance, folder, top=TOP_FEATURES):
    """The *top* features of *importance* (channel_fdr_fit.feature_importance's
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
        ax.set_title(f"{title + ': ' if title else ''}{n_sets:.0f} sets", fontsize=10)
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


def plot_channel_fdr(features, folder, importance=None, split="all", fdr_threshold=0.01, ecdfs=True):
    """The channel-FDR plots of a run into *folder*: the best channel by
    PredVal, IDs vs Channel_Qvalue, the IDs it retains at *fdr_threshold*, the
    feature importance if there is one, and with *ecdfs* each feature's ECDF.
    *split*: "all", or "KR" for the K- and R-ending peptides as two panels (not
    the feature importance, which is one model's)."""
    split = _split(split)
    if split is None:
        return
    if ecdfs:
        plot_feature_ecdfs(features, folder, split=split)
    plot_best_channel(features, folder, split=split)
    plot_ids_vs_channel_qvalue(features, folder, split=split, fdr_threshold=fdr_threshold)
    plot_ids_retained(features, folder, split=split, fdr_threshold=fdr_threshold)
    if importance is not None:
        plot_feature_importance(importance, folder)


if __name__ == "__main__":
    if len(_ARGS) not in (1, 2):
        sys.exit("usage: python -m src.channel_fdr_plots <channel_features.parquet> [all|KR]")
    parquet = _ARGS[0]
    folder = os.path.dirname(os.path.abspath(parquet))
    # with the fit's feature importance, if it was written beside the table
    plot_channel_fdr(pl.read_parquet(parquet), folder, importance=read_feature_importance(folder),
                     split=_ARGS[1] if len(_ARGS) == 2 else "all")
