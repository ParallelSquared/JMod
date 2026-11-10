"""Histograms and ECDFs of every channel feature, one line per channel, the
decoy channel set apart: a look at which features tell a channel that is there
from one that is not, before anything is fitted.

Written by the pipeline with --channel_fdr_plots, into <run>/channel_fdr/
histograms and <run>/channel_fdr/ECDF: "all" (the flag alone) plots every
precursor together, "KR" plots the peptides ending in K and in R side by side --
under a K-tagging tag every y ion of a K-ending peptide carries the tag, so
the two behave differently.  Also runs on its own, on any written feature
table:

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

from src.channel_fdr import _AMINO_ACIDS, _ID_COLUMNS
from src.logger import logger

HIST_FOLDER = "histograms"
ECDF_FOLDER = "ECDF"
SPLITS = ("all", "KR")
# The axes show the central 99% of the values: a handful of extreme ones would
# otherwise squash every channel into a bin or two.  The ECDF only zooms (its
# fractions count every value); the histogram leaves the tails out
CLIP_PCT = 99.5
N_BINS = 40
# Integer-valued features over at most this many values get one bin per value
MAX_INTEGER_BINS = 60
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


def _bins(values):
    integer = np.all(values == np.round(values))
    if integer and values.max() - values.min() <= MAX_INTEGER_BINS:
        return np.arange(values.min() - 0.5, values.max() + 1.5)
    return np.linspace(*_range(values), N_BINS + 1)


def _draw_histogram(ax, groups, bins):
    for (channel, decoy, values, n_null), style in zip(groups, _styles(groups)):
        if values.size == 0:
            continue
        counts, _ = np.histogram(values, bins=bins)
        density = counts / max(counts.sum(), 1) / np.diff(bins)
        # one step outline per channel: filled bars from several channels hide each other
        ax.stairs(density, bins, label=_label(channel, decoy, values, n_null), **style)
    ax.set_ylabel("Density")


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


def _figure(column, panel_groups, draw, path):
    """One panel per entry of *panel_groups* [(title, groups)], side by side on
    the same x axis."""
    fig, axes = plt.subplots(1, len(panel_groups), figsize=(8 * len(panel_groups), 5), squeeze=False,
                             sharex=True)
    for ax, (title, groups) in zip(axes[0], panel_groups):
        draw(ax, groups)
        ax.set_xlabel(column)
        if title:
            ax.set_title(title, fontsize=10)
        _legend(ax, groups)
    fig.suptitle(column)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def plot_channel_features(features, folder, split="all"):
    """A histogram and an ECDF of every feature of *features* (the table
    write_channel_features writes), into *folder*/histograms and *folder*/ECDF.
    *split*: "all", or "KR" for the K- and R-ending peptides as two panels.  A
    feature with no value in any panel is skipped.  Returns the number of
    features plotted."""
    split = next((s for s in SPLITS if s.lower() == str(split).lower()), None)
    if split is None:
        logger.warning(f"Channel feature plots: not made, the split must be one of {', '.join(SPLITS)}")
        return 0
    hist_folder, ecdf_folder = os.path.join(folder, HIST_FOLDER), os.path.join(folder, ECDF_FOLDER)
    os.makedirs(hist_folder, exist_ok=True)
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
        everything = np.concatenate(values)
        bins, x_range = _bins(everything), _range(everything)
        _figure(column, panel_groups, lambda ax, groups: _draw_histogram(ax, groups, bins),
                os.path.join(hist_folder, f"{column}.png"))
        _figure(column, panel_groups, lambda ax, groups: _draw_ecdf(ax, groups, x_range),
                os.path.join(ecdf_folder, f"{column}.png"))
    if empty:
        logger.info(f"Channel feature plots: no values to plot for {', '.join(empty)}")
    n = len(feature_columns(features)) - len(empty)
    logger.info(f"Channel feature histograms and ECDFs ({split}) for {n} features written to "
                f"{os.path.abspath(hist_folder)} and {os.path.abspath(ecdf_folder)}")
    return n


if __name__ == "__main__":
    if len(_ARGS) not in (1, 2):
        sys.exit("usage: python -m src.channel_fdr_plots <channel_features.parquet> [all|KR]")
    parquet = _ARGS[0]
    plot_channel_features(pl.read_parquet(parquet), os.path.dirname(os.path.abspath(parquet)),
                          split=_ARGS[1] if len(_ARGS) == 2 else "all")
