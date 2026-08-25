"""
This Source Code Form is subject to the terms of the Oxford Nanopore
Technologies, Ltd. Public License, v. 1.0.  Full licence can be found
at https://github.com/ParallelSquared/JMod/blob/main/LICENSE.txt
"""

"""
Retention-time-dependent quantitative normalization, applied per time channel.

Rationale
---------
Signal delivered to the detector drifts across the gradient (spray stability,
space-charge in the trap, differential ion suppression).  Under timePlex the
drift is *channel specific*: each emitter has its own spray history, so the
same peptide measured in two time channels picks up two different RT-dependent
gains.  Any ratio taken across time channels (or across runs) inherits that
gain difference and shows up as a ratio that trends with RT.

The correction assumes the running centre of precursor quantities within one
time channel is flat across the gradient — i.e. the *typical* precursor does
not systematically get brighter or dimmer with RT — and divides out whatever
smooth RT-dependent deviation is observed.  It is applied independently to
each time channel (and, for non-timePlex data, to the single channel).

Both the MS1 and the MS2 quantities are normalized, each with its own factor
curve, because the two readouts drift differently (MS1 is dominated by
space-charge in the survey scan, MS2 by isolation-window competition).

plexDIA
-------
When labelled channels (mTRAQ / SILAC) are present, the *label* channels of one
time channel share the same physical spray and therefore the same RT-dependent
gain.  The reference curve is built once per time channel from the mean quantity
across label channels of confidently identified precursors (Qvalue < cutoff),
and the resulting factor is applied to every label channel of that time channel.
This is deliberate: ratios *within* a plexDIA set are mathematically untouched
by the correction, so labelled-channel quantitation cannot be degraded by it.

Public entry point
------------------
``apply_rt_channel_normalization(df, ...)`` adds ``*_rtnorm`` columns plus the
two factor columns.  Original columns are left untouched.
"""

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd

try:  # keep this module importable outside the JMod package (offline sweeps)
    from src.logger import logger
except Exception:  # pragma: no cover
    import logging
    logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Columns
# --------------------------------------------------------------------------
# The MS1 and MS2 quantities JMod reports.  ``primary`` is the one used to build
# the reference curve for that level; the rest ride along on the same factor so
# every reported quantity stays on one consistent scale.
MS1_PRIMARY = "plex_Area"
# The MS2 correction curve is estimated from this column; every other MS2
# quantity is multiplied by the resulting factor.
#
# ``coeff_ChannelUnique`` rather than ``coeff`` because ``coeff`` is
# ratio-compressed (0.65 of the designed fold change on JD0413 against 0.97),
# and the paired stage measures precisely the per-precursor across-channel
# deviation that compression attenuates -- so a compressed primary
# under-corrects.  Measured on JD0413, switching the primary cut the
# across-time-channel RT drift of ``coeff`` by 22% (0.184 -> 0.143) and of
# ``coeff_ChannelUnique`` by 16% (0.339 -> 0.284), and improved yeast
# fold-change accuracy for both.  Requires the ungated pre-scoring pass:
# gated, the column covers only 91% of the reference set against 99% ungated.
# Falls back automatically -- ``apply_rt_channel_normalization`` skips a level
# whose primary column is absent.
MS2_PRIMARY = "coeff_ChannelUnique"

MS1_QUANT_COLS = ("plex_Area", "MS1_Area")
MS2_QUANT_COLS = ("coeff", "coeff_ChannelUnique",
                  "coeff_Area", "coeff_ChannelUnique_Area",
                  # single-scan value at the group's shared apex --
                  # normalised like any other reported abundance
                  "coeff_CommonApex", "coeff_ChannelUnique_CommonApex")

# Per-scan and per-fragment intensity vectors.  Normalizing these is opt-in
# (--rt_norm_traces): the copies roughly double the size of the trace columns in
# all_IDs.csv and only the scalar quantities are used downstream.  Only
# *observed* intensities would ever belong here -- never frag_int or
# unique_frag_mz (library predictions), plexfittrace_ps_all (a Pearson
# statistic), all_ms1_specs (spectrum ids) or tic (a property of the scan).
MS1_TRACE_COLS = ("plexfittrace", "plexfittrace_all") + tuple(
    f"all_ms1_iso{i}vals" for i in range(6))
MS2_TRACE_COLS = ("obs_int", "unique_obs_int")

# Sentinels for the intensity axes (see _paired_channel_residuals).
OWN_MEAN = "__own_mean__"
OWN_CHAN = "__own_channel__"

FACTOR_MS1 = "rtnorm_factor_MS1"
FACTOR_MS2 = "rtnorm_factor_MS2"
SUFFIX = "_rtnorm"


def _scale_vector_column(series, factor):
    """Multiply every element of a per-row intensity vector by that row's factor.

    JMod carries these in two shapes: ``";"``-joined strings (the MS1 traces and
    isotope vectors) and real list/array columns (the fragment intensities).
    Returns the same shape it was given so the new column can sit beside the
    original without changing how anything downstream reads it.

    Non-finite factors leave the row untouched rather than blanking it -- a row
    the normalizer could not place is still a valid measurement.
    """
    vals = series.to_numpy() if hasattr(series, "to_numpy") else np.asarray(series)
    f = np.asarray(factor, dtype=float)
    out = np.empty(len(vals), dtype=object)
    for i, v in enumerate(vals):
        fi = f[i]
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            out[i] = v
            continue
        if not np.isfinite(fi) or fi <= 0:
            out[i] = v
            continue
        if isinstance(v, str):
            if not v:
                out[i] = v
                continue
            try:
                out[i] = ";".join(
                    repr(float(tok) * fi) for tok in v.split(";"))
            except ValueError:
                out[i] = v                    # not numeric; leave it alone
        else:
            try:
                arr = np.asarray(v, dtype=float)
            except (TypeError, ValueError):
                out[i] = v
                continue
            out[i] = (arr * fi).tolist() if arr.ndim else v
    return out


@dataclass
class RTNormConfig:
    """Knobs for the RT-dependent per-channel normalization.

    Attributes
    ----------
    method
        ``"rolling_linear"``  local linear fit over a fixed *count* of precursors
        ``"rolling_median"``  running median over a fixed count of precursors
        ``"rolling_mean"``    running mean over a fixed count of precursors
        ``"loess_n"``         lowess with the span set from ``window`` precursors
        ``"loess_cap"``       lowess spanning ``min(loess_frac * n, window)``
                              precursors — a fraction of the channel, capped at
                              a fixed count so a well-covered channel does not
                              get an unnecessarily coarse curve, and a sparse
                              one does not get a window covering most of its
                              gradient
        ``"rt_window"``       median over a fixed *RT width* (minutes)
        ``"loess"``           lowess smoother of log2(quant) vs RT
        ``"tic"``             per-scan MS1 TIC curve only (needs ``tic_rt``/``tic``)
        ``"tic_rolling"``     TIC correction first, then the running median
        ``"none"``            disable
    window
        Number of precursors in the running window (count-based methods).  The
        window is centred, so it spans ``window//2`` precursors either side.
    rt_window
        Width in minutes for ``rt_window``.
    loess_frac
        Fraction of the data used per local fit for ``loess``.
    tic_window
        Number of MS1 scans in the running mean used to smooth the TIC curve.
    qvalue_cutoff
        Only precursors at or below this q-value contribute to the reference
        curve.  All rows are corrected regardless.  A looser cutoff than the
        reporting threshold is deliberate: the curve wants *coverage across RT*
        far more than it wants purity, and a handful of false positives moves a
        local median or a local line very little.
    qvalue_col
        Which q-value column gates the reference set.
    plex_agg
        How label channels of one time channel are collapsed into a single
        reference value per precursor: ``"mean"`` (arithmetic, the default and
        what the spec asks for), ``"geomean"``, or ``"median"``.
    ref_top_frac
        Restrict the reference set to the brightest fraction of precursors
        (1.0 = use all).  Mirrors DIA-NN's "normalise on high-abundance
        precursors" behaviour.
    min_intensity
        Floor on the quantity for a row to *define* the reference curve.  JMod
        writes 0.001 into a trace position with no observed signal, so columns
        derived from a trace carry placeholder values that are positive but
        meaningless; a bare ``> 0`` test lets them set the local centre.  Every
        row is still corrected -- this only decides which rows are trusted to
        fit the curve.
    min_ref
        A channel with fewer reference precursors than this is left uncorrected.
    min_span
        Floor on the local fit's span in precursors, so a fraction rule cannot
        collapse to a handful of points on a very shallow channel.
    max_log2_shift
        Safety clamp on the correction, in log2 units.  Guards against a wild
        extrapolation at the very edges of a channel's RT range.
    center
        Statistic defining each channel's target level: ``"median"`` or
        ``"mean"`` of log2(quant) over the reference set.
    two_stage
        Estimate the between-channel part of the drift precursor-by-precursor
        first (see ``_paired_channel_residuals``), then the part common to the
        whole run.  Only the paired stage can change a channel-vs-channel ratio,
        and it is immune to the detection-selection bias that makes the unpaired
        estimator invent a trend at the ends of the gradient.  Ignored when
        there is only one channel.
    mz_stage
        Run a second paired correction smoothed against precursor *m/z* after
        the RT one, for any channel-specific transmission difference that RT
        could not explain.  Off by default: on the benchmark the leftover m/z
        structure is ~0.05 log2, about twice a shuffled-label null and smaller
        than what the RT correction itself leaves behind, and correcting it does
        not improve accuracy.
    intensity_stage
        Run a paired correction smoothed against precursor abundance, for a
        channel gain that depends on how bright a precursor is.
        ``intensity_axis="own_mean"`` uses the precursor's mean across channels
        -- the only defensible choice, because it is the same number in every
        channel.  ``"own_channel"`` uses each channel's own quantity and is
        provided only to demonstrate that it manufactures ratio compression.
        Off by default.
    stage_order
        ``"unpaired_first"`` (default): flatten each time channel's own running
        centre — the classic single-stage correction, one curve *per time
        channel* — and then run the paired stage on what is left.  Order matters:
        per-channel flattening also distorts the ratios *between* channels,
        because the channels are staggered in RT so "flat in its own frame" is
        not the same correction for two of them, and because each channel has
        its own detection-selection bias.  Running paired last lets it measure
        and remove exactly that damage.  Single-channel data simply gets the
        unpaired stage, which is all that is available and is what helps
        run-to-run agreement.

        ``"paired_first"`` runs the paired stage first and then consults
        ``run_stage``; kept for comparison.
    run_stage
        Only consulted when ``stage_order == "paired_first"``.  ``"auto"``
        applies a run-level curve only when the paired stage could not run;
        ``"always"`` / ``"never"`` force it.
    center_scope
        ``"channel"`` (default) flattens each time channel around *its own*
        level, so between-channel loading differences survive untouched and the
        correction is purely a within-channel RT correction.  ``"global"`` also
        pulls every channel onto one shared level, i.e. median normalization
        across time channels on top of the RT flattening — only correct if the
        channels are meant to carry equal total load.
    """

    method: str = "loess_cap"
    window: int = 400
    rt_window: float = 2.0
    loess_frac: float = 0.05
    tic_window: int = 20
    qvalue_cutoff: float = 0.05
    qvalue_col: str = "Qvalue"
    plex_agg: str = "mean"
    ref_top_frac: float = 1.0
    min_ref: int = 50
    min_intensity: float = 1.0
    min_span: int = 50
    max_log2_shift: float = 4.0
    center: str = "median"
    center_scope: str = "channel"
    two_stage: bool = True
    mz_stage: bool = False
    mz_col: str = "mz"
    intensity_stage: bool = False
    intensity_axis: str = "own_mean"
    run_stage: str = "auto"
    stage_order: str = "unpaired_first"

    def describe(self) -> str:
        if self.method == "loess_cap":
            w = f"min({self.loess_frac}*n, {self.window})"
        elif self.method in ("rolling_linear", "rolling_median", "rolling_mean",
                             "loess_n", "tic_rolling"):
            w = f"window={self.window}"
        elif self.method == "rt_window":
            w = f"rt_window={self.rt_window}min"
        elif self.method == "loess":
            w = f"frac={self.loess_frac}"
        else:
            w = f"tic_window={self.tic_window}"
        extra = "" if self.ref_top_frac >= 1.0 else f" top={self.ref_top_frac}"
        extra += "" if self.min_intensity <= 0 else f" int>{self.min_intensity:g}"
        return f"{self.method}({w}{extra}, agg={self.plex_agg})"


# --------------------------------------------------------------------------
# Smoothers.  Each returns the *log2 deviation from the channel centre* sampled
# at ``rt_ref``; callers interpolate that onto the rows they want to correct.
# --------------------------------------------------------------------------

def _rolling_stat(y: np.ndarray, window: int, how: str) -> np.ndarray:
    """Centred running median/mean over a fixed number of points."""
    window = max(3, int(window))
    if window > len(y):
        window = len(y) if len(y) % 2 else len(y) - 1
        window = max(3, window)
    s = pd.Series(y)
    r = s.rolling(window=window, center=True, min_periods=max(3, window // 4))
    out = (r.median() if how == "median" else r.mean()).to_numpy()
    # rolling() leaves NaN wherever min_periods was not met (only possible at
    # the very ends for tiny inputs); carry the nearest fitted value outward so
    # the interpolator downstream never sees a NaN.
    return _fill_edges(out)


def _fill_edges(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    ok = np.isfinite(v)
    if not ok.any():
        return np.zeros_like(v)
    idx = np.arange(len(v))
    return np.interp(idx, idx[ok], v[ok])


def _rolling_linear(rt: np.ndarray, y: np.ndarray, window: int) -> np.ndarray:
    """Centred local *linear* fit over a fixed number of points, evaluated at
    each point's own RT.

    Same idea as loess with a boxcar kernel and degree 1, but computed from
    running sums, so it is O(n) rather than O(n * window).  The linear term is
    what matters: a running median is flat inside its window, which biases it
    wherever the trend is sloped — most visibly at the two ends of the gradient,
    exactly where the drift is largest.
    """
    n = len(y)
    window = max(5, int(window))
    if window >= n:
        # Degenerate to a single global line.
        b, a = np.polyfit(rt, y, 1)
        return a + b * rt
    half = window // 2

    def csum(v):
        return np.concatenate([[0.0], np.cumsum(v)])

    x = rt - rt.mean()  # centre for conditioning
    S1, Sx, Sxx = csum(np.ones(n)), csum(x), csum(x * x)
    Sy, Sxy = csum(y), csum(x * y)

    i = np.arange(n)
    lo = np.clip(i - half, 0, n)
    hi = np.clip(i + half + 1, 0, n)

    def w(c):
        return c[hi] - c[lo]

    n_w, sx, sxx, sy, sxy = w(S1), w(Sx), w(Sxx), w(Sy), w(Sxy)
    denom = n_w * sxx - sx * sx
    # Where the window is degenerate (all points at one RT) fall back to the
    # window mean instead of dividing by ~0.
    slope = np.where(np.abs(denom) > 1e-12, (n_w * sxy - sx * sy) / np.where(
        np.abs(denom) > 1e-12, denom, 1.0), 0.0)
    intercept = (sy - slope * sx) / n_w
    return intercept + slope * x


def _rt_window_median(rt: np.ndarray, y: np.ndarray, width: float,
                      n_knots: int = 400) -> tuple:
    """Median of ``y`` within +/- width/2 minutes, evaluated on an RT grid."""
    lo, hi = float(rt[0]), float(rt[-1])
    if hi <= lo:
        return rt, np.full_like(y, np.median(y))
    knots = np.linspace(lo, hi, min(n_knots, max(10, len(y) // 10)))
    half = width / 2.0
    left = np.searchsorted(rt, knots - half, side="left")
    right = np.searchsorted(rt, knots + half, side="right")
    vals = np.empty(len(knots))
    for i, (a, b) in enumerate(zip(left, right)):
        vals[i] = np.median(y[a:b]) if b - a >= 3 else np.nan
    return knots, _fill_edges(vals)


def loess_span_frac(n: int, loess_frac: float = 0.05, window: int = 400,
                    min_span: int = 50) -> float:
    """Local-fit span as a fraction of ``n``: ``min(loess_frac*n, window)``.

    Span is expressed in *points*, not in RT, so one setting means the same
    thing on a sparse channel and a dense one; the fraction wins when coverage
    is low, the cap wins when it is high, and ``min_span`` floors both. Shared
    by the intensity normalization and by the width/shape models so all three
    curves have the same RT resolution on the same run.
    """
    n = max(int(n), 1)
    span = min(float(loess_frac) * n, float(window))
    return float(min(1.0, max(span, float(min_span)) / n))


def _loess(rt: np.ndarray, y: np.ndarray, frac: float) -> tuple:
    from statsmodels.nonparametric.smoothers_lowess import lowess
    # ``it=0`` keeps it to a single pass: the reference set is already q-value
    # filtered, and robustifying iterations cost time without changing a curve
    # this heavily averaged.
    fit = lowess(y, rt, frac=frac, it=0, return_sorted=True)
    return fit[:, 0], fit[:, 1]


# --------------------------------------------------------------------------
# TIC curve
# --------------------------------------------------------------------------

def ms1_tic_curve(spectra) -> Optional[tuple]:
    """(rt, tic) for the MS1 scans of a loaded SpectrumFile, or None."""
    ms1 = getattr(spectra, "ms1scans", None)
    if not ms1:
        return None
    rt = np.array([s.RT for s in ms1], dtype=float)
    tic = np.array([(s.TIC if s.TIC is not None else np.nan) for s in ms1], dtype=float)
    ok = np.isfinite(rt) & np.isfinite(tic) & (tic > 0)
    if ok.sum() < 10:
        return None
    order = np.argsort(rt[ok])
    return rt[ok][order], tic[ok][order]


def _tic_deviation(tic_rt: np.ndarray, tic: np.ndarray, window: int) -> tuple:
    """log2 deviation of the smoothed TIC from its global mean."""
    sm = pd.Series(tic).rolling(window=max(3, 2 * int(window) + 1),
                                center=True, min_periods=3).mean().to_numpy()
    sm = _fill_edges(sm)
    sm = np.clip(sm, 1e-12, None)
    return tic_rt, np.log2(sm) - np.log2(np.mean(tic))


# --------------------------------------------------------------------------
# Reference-curve construction
# --------------------------------------------------------------------------

def _channel_reference(sub: pd.DataFrame, quant_col: str, cfg: RTNormConfig,
                       group_col: Optional[str]) -> Optional[tuple]:
    """Reference (rt, log2 quant) pairs for one time channel.

    ``group_col`` is the precursor identity used to collapse label channels
    (plexDIA / SILAC).  Pass ``None`` to use rows as-is.
    """
    q = sub[quant_col].to_numpy(dtype=float)
    rt = sub["rt"].to_numpy(dtype=float)
    keep = np.isfinite(q) & (q > cfg.min_intensity) & np.isfinite(rt)
    if cfg.qvalue_col in sub.columns:
        qv = pd.to_numeric(sub[cfg.qvalue_col], errors="coerce").to_numpy(dtype=float)
        keep &= np.isfinite(qv) & (qv < cfg.qvalue_cutoff)
    if keep.sum() < cfg.min_ref:
        return None

    if group_col is not None and group_col in sub.columns:
        g = pd.DataFrame({"g": sub[group_col].to_numpy()[keep],
                          "rt": rt[keep], "q": q[keep]})
        if cfg.plex_agg == "geomean":
            g["q"] = np.log2(g["q"])
            agg = g.groupby("g", sort=False).agg(rt=("rt", "median"), q=("q", "mean"))
            ref_rt = agg["rt"].to_numpy()
            ref_y = agg["q"].to_numpy()
        else:
            how = "median" if cfg.plex_agg == "median" else "mean"
            agg = g.groupby("g", sort=False).agg(rt=("rt", "median"), q=("q", how))
            ref_rt = agg["rt"].to_numpy()
            ref_y = np.log2(np.clip(agg["q"].to_numpy(), 1e-12, None))
    else:
        ref_rt = rt[keep]
        ref_y = np.log2(q[keep])

    if cfg.ref_top_frac < 1.0 and len(ref_y) > cfg.min_ref:
        # Brightest fraction only.  Intensity rank is taken globally rather than
        # per RT window on purpose: a per-window top-N would re-introduce the
        # very RT-dependent selection bias the correction is meant to measure.
        n_keep = max(cfg.min_ref, int(round(len(ref_y) * cfg.ref_top_frac)))
        sel = np.argsort(ref_y)[-n_keep:]
        ref_rt, ref_y = ref_rt[sel], ref_y[sel]

    if len(ref_y) < cfg.min_ref:
        return None
    order = np.argsort(ref_rt, kind="stable")
    return ref_rt[order], ref_y[order]


def _center_of(y: np.ndarray, cfg: RTNormConfig) -> float:
    return float(np.median(y) if cfg.center == "median" else np.mean(y))


def _paired_channel_residuals(slim: pd.DataFrame, chan: np.ndarray,
                              keep: np.ndarray, quant_col: str,
                              group_col: str, cfg: RTNormConfig,
                              x_col: str = "rt") -> dict:
    """Per-channel RT residuals measured against each precursor's own mean.

    The unpaired estimator (running centre of raw log2 quant) inherits whatever
    the *detected population* does across the gradient.  That population is not
    constant: identification gets harder late in the gradient, so the precursors
    that still clear the q-value gate there are the brighter ones.  On the
    benchmark data that selection alone drifts the reference upward by ~0.7 log2
    from the start of the gradient to the end — a trend with no instrumental
    cause, which an unpaired correction then dutifully divides out of every
    precursor.

    Pairing removes it.  For a precursor seen in two or more channels, take its
    deviation from its own across-channel mean; the selection that let it be
    detected is common to those channels and cancels.  What is left is exactly
    the between-channel gain difference, which is the channel-specific part of
    the drift.  The residuals sum to zero across channels by construction, so
    this step cannot move a channel's overall level — only its shape.

    Returns {channel: (rt, residual)} for channels with enough paired
    precursors, or {} when nothing can be paired (single-channel data).
    """
    if len(np.unique(chan)) < 2:
        return {}

    idx = np.flatnonzero(keep)
    # ``x_col`` is the axis the residual will be smoothed against.  Two special
    # values refer to the intensity itself:
    #   OWN_MEAN  the precursor's mean log2 quantity across channels -- the same
    #             number for every channel, so it is a legitimate covariate
    #   OWN_CHAN  this channel's own log2 quantity -- NOT legitimate, see below
    x_special = x_col in (OWN_MEAN, OWN_CHAN)
    d = pd.DataFrame({
        "g": slim[group_col].to_numpy()[idx],
        "ch": chan[idx],
        "rt": (np.zeros(idx.size) if x_special
               else slim[x_col].to_numpy()[idx]),
        "y": np.log2(np.clip(slim[quant_col].to_numpy()[idx], 1e-12, None)),
    })
    # Collapse label channels (and any duplicate rows) to one value per
    # (precursor, time channel) before pairing.
    how = {"mean": "mean", "geomean": "mean", "median": "median"}[cfg.plex_agg]
    if cfg.plex_agg == "mean":
        # Arithmetic mean of the quantities, per the spec, then back to log2.
        d["lin"] = np.exp2(d["y"])
        g = d.groupby(["g", "ch"], sort=False).agg(rt=("rt", "median"),
                                                   lin=("lin", "mean"))
        g["y"] = np.log2(np.clip(g["lin"].to_numpy(), 1e-12, None))
        g = g.drop(columns=["lin"])
    else:
        g = d.groupby(["g", "ch"], sort=False).agg(rt=("rt", "median"),
                                                   y=("y", how))
    g = g.reset_index()

    n_ch = g.groupby("g")["ch"].transform("size")
    g = g[n_ch >= 2]
    if len(g) < 2 * cfg.min_ref:
        return {}
    g["resid"] = g["y"] - g.groupby("g")["y"].transform("mean")
    if x_col == OWN_MEAN:
        # Abundance of the precursor, shared by all its channels.  Under
        # homoscedastic noise the residual is uncorrelated with it by
        # construction, so any structure here is a real abundance-dependent
        # channel effect.
        g["rt"] = g.groupby("g")["y"].transform("mean")
    elif x_col == OWN_CHAN:
        # Deliberately available so the benchmark can show why it is wrong:
        # the residual is y minus the mean *of which y is a term*, so a
        # precursor that happened to read high in this channel has a positive
        # residual by construction.  Regressing that away shrinks every ratio
        # toward zero -- it manufactures ratio compression.
        g["rt"] = g["y"]

    out = {}
    for ch, sub in g.groupby("ch"):
        if len(sub) < cfg.min_ref:
            continue
        sub = sub.sort_values("rt", kind="stable")
        out[ch] = (sub["rt"].to_numpy(), sub["resid"].to_numpy())
    return out


def _channel_correction(ref_rt: np.ndarray, ref_y: np.ndarray, cfg: RTNormConfig,
                        tic: Optional[tuple],
                        center: Optional[float] = None) -> Optional[tuple]:
    """Knot RTs and the log2 correction to *subtract* at those RTs."""
    span_pts = 0
    if center is None:
        center = _center_of(ref_y, cfg)

    tic_knots = tic_dev = None
    if cfg.method in ("tic", "tic_rolling"):
        if tic is None:
            logger.warning("RT normalization: TIC method requested but no MS1 TIC "
                           "available; falling back to the local linear fit")
            cfg = RTNormConfig(**{**cfg.__dict__, "method": "loess_n"})
        else:
            tic_knots, tic_dev = _tic_deviation(tic[0], tic[1], cfg.tic_window)

    if cfg.method == "tic":
        return tic_knots, tic_dev

    y = ref_y
    if cfg.method == "tic_rolling":
        # Remove the TIC trend from the reference values first, so the running
        # median only has to explain what the TIC could not.
        y = ref_y - np.interp(ref_rt, tic_knots, tic_dev)
        center = _center_of(y, cfg)

    if cfg.method in ("rolling_linear", "tic_rolling"):
        knots, fit = ref_rt, _rolling_linear(ref_rt, y, cfg.window)
    elif cfg.method == "rolling_median":
        knots, fit = ref_rt, _rolling_stat(y, cfg.window, "median")
    elif cfg.method == "rolling_mean":
        knots, fit = ref_rt, _rolling_stat(y, cfg.window, "mean")
    elif cfg.method in ("loess_n", "loess_cap"):
        n_ref = max(len(y), 1)
        # ``loess_n``: span expressed in precursors, so the same setting means
        # the same thing on a 5k-precursor channel and a 60k one.
        # ``loess_cap``: whichever is *fewer* precursors, a fixed fraction of
        # the channel or the fixed count. On a sparse channel the fraction wins,
        # which keeps the curve's RT resolution proportional to the coverage
        # instead of letting 400 precursors swallow half the gradient.
        frac = loess_span_frac(
            n_ref, 1.0 if cfg.method == "loess_n" else cfg.loess_frac,
            cfg.window, cfg.min_span)
        span_pts = int(round(frac * n_ref))
        knots, fit = _loess(ref_rt, y, frac)
    elif cfg.method == "rt_window":
        knots, fit = _rt_window_median(ref_rt, y, cfg.rt_window)
    elif cfg.method == "loess":
        knots, fit = _loess(ref_rt, y, cfg.loess_frac)
    else:
        raise ValueError(f"unknown RT normalization method: {cfg.method!r}")

    dev = fit - center
    if cfg.method == "tic_rolling":
        # np.interp clamps outside the TIC range, matching approxfun(rule = 2).
        dev = dev + np.interp(knots, tic_knots, tic_dev)
    return _clamp_to_supported_span(knots, dev, ref_rt, cfg, span_pts)


def _clamp_to_supported_span(knots, dev, ref_rt, cfg, span_pts=0):
    """Trim the curve to the RT span that ``min_span`` reference points support.

    A local fit keeps producing values past its last well-supported point, and at
    the edges the neighbourhood it draws on is entirely one-sided, so the fit
    extrapolates.  That is worst exactly where it does most damage: the first and
    last time channels run out of precursors at opposite ends of the gradient.
    Measured on JD0435 tc0, the 64-68 min bin held 128 reference precursors
    against ~5,000 mid-gradient, and the curve there applied a +2.34 log2 (5x)
    correction to the least trustworthy precursors in the run -- turning a raw
    running median of -2.2 into +1.2.

    Cutting the knots back to ``[rt of the min_span'th point, rt of the
    min_span'th from the end]`` makes ``np.interp`` hold the last supported value
    flat beyond it, which is the honest answer: outside that span there is no
    evidence about the drift, so do not invent any.  Applied at both ends.
    """
    if len(knots) < 2 or len(ref_rt) < 2:
        return knots, dev
    # Half a span, floored at min_span. Half a span is the point at which the
    # local neighbourhood stops being two-sided: past it the fit is drawing
    # entirely on earlier (or later) precursors and extrapolating a slope into a
    # region with no evidence. The bare min_span floor is not enough on its own
    # -- tc0 on JD0435 has ~55,000 reference precursors, so its 50th-from-the-end
    # still sits inside the bad bin and trimming to it moved the tail correction
    # only 2.34 -> 2.11 log2.
    k = max(int(getattr(cfg, "min_span", 50)), int(span_pts) // 2)
    if k < 1 or 2 * k >= len(ref_rt):
        return knots, dev
    rt_sorted = np.sort(np.asarray(ref_rt, dtype=float))
    lo, hi = rt_sorted[k - 1], rt_sorted[-k]
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
        return knots, dev
    keep = (knots >= lo) & (knots <= hi)
    if keep.sum() < 2:
        return knots, dev
    return knots[keep], dev[keep]


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def apply_rt_channel_normalization(df: pd.DataFrame,
                                   timeplex: bool = False,
                                   plexDIA: bool = False,
                                   cfg: Optional[RTNormConfig] = None,
                                   spectra=None,
                                   ms1_quant_cols: Sequence[str] = MS1_QUANT_COLS,
                                   ms2_quant_cols: Sequence[str] = MS2_QUANT_COLS,
                                   inplace_cols: bool = False,
                                   ref_mask: Optional[np.ndarray] = None,
                                   ms1_trace_cols: Sequence[str] = (),
                                   ms2_trace_cols: Sequence[str] = (),
                                   plot_path: Optional[str] = None) -> pd.DataFrame:
    """Normalize MS1 and MS2 quantities per time channel as a function of RT.

    Adds ``rtnorm_factor_MS1`` / ``rtnorm_factor_MS2`` (multiplicative, linear
    space) and a ``<col>_rtnorm`` copy of every quant column.  With
    ``inplace_cols=True`` the original columns are overwritten instead and the
    pre-normalization values are preserved as ``<col>_raw``.

    The per-scan MS1 traces and isotope vectors, and the per-fragment observed
    intensities, get the same treatment on the same factor, so a normalized
    quantity can be traced back to normalized evidence.

    ``plot_path`` writes a per-channel diagnostic figure of the correction.

    Returns ``df`` (modified in place and also returned, matching the style of
    the rest of ``process_data``).
    """
    cfg = cfg or RTNormConfig()
    if cfg.method == "none":
        return df
    if "rt" not in df.columns:
        logger.warning("RT normalization skipped: no 'rt' column")
        return df

    n = len(df)
    if n == 0:
        return df

    # --- channel grouping -------------------------------------------------
    # Under timePlex each time channel is a separate spray with its own drift.
    # Without timePlex the whole run is one channel — the correction still runs,
    # which is the point: it flattens the gradient trend for plain LF/plexDIA
    # data too.
    if timeplex and "time_channel" in df.columns:
        chan = df["time_channel"].to_numpy()
    else:
        chan = np.zeros(n)

    # Precursor identity. Needed twice: to collapse label channels inside a time
    # channel (plexDIA only), and to pair a precursor against itself across time
    # channels (always, when there is more than one channel).
    prec_col = "untag_prec" if "untag_prec" in df.columns else None
    if plexDIA and prec_col is None:
        logger.warning("RT normalization: plexDIA set but no 'untag_prec' "
                       "column; using rows without collapsing label channels")
    group_col = prec_col if plexDIA else None

    # Decoys never contribute to the reference curve but still get corrected so
    # target and decoy quantities stay on one scale.  ``ref_mask`` lets a caller
    # (the strategy sweep) hold rows out of the curve while still correcting
    # them, which is how window size is chosen without fitting the evaluation.
    ref_mask = (np.ones(n, dtype=bool) if ref_mask is None
                else np.asarray(ref_mask, dtype=bool).copy())
    if "is_decoy" in df.columns:
        ref_mask &= ~df["is_decoy"].to_numpy(dtype=bool)

    tic = ms1_tic_curve(spectra) if (spectra is not None and
                                     cfg.method in ("tic", "tic_rolling")) else None

    levels = []
    if MS1_PRIMARY in df.columns:
        levels.append((MS1_PRIMARY, FACTOR_MS1,
                       [c for c in ms1_quant_cols if c in df.columns],
                       [c for c in ms1_trace_cols if c in df.columns]))
    if MS2_PRIMARY in df.columns:
        levels.append((MS2_PRIMARY, FACTOR_MS2,
                       [c for c in ms2_quant_cols if c in df.columns],
                       [c for c in ms2_trace_cols if c in df.columns]))
    if not levels:
        logger.warning("RT normalization skipped: no quant columns found")
        return df

    # Kept for the diagnostic figure: the per-row log2 correction of each level.
    shifts = {}

    rt_all = df["rt"].to_numpy(dtype=float)
    uniq = pd.unique(chan)

    # The reference curve only ever looks at rt / quant / q-value / the group
    # key, so slice a slim frame once instead of copying the (very wide) quant
    # table once per channel per level.
    slim_cols = {"rt": rt_all}
    if cfg.qvalue_col in df.columns:
        slim_cols[cfg.qvalue_col] = df[cfg.qvalue_col].to_numpy()
    if prec_col is not None:
        slim_cols[prec_col] = df[prec_col].to_numpy()
    if cfg.mz_stage and cfg.mz_col in df.columns:
        slim_cols[cfg.mz_col] = pd.to_numeric(df[cfg.mz_col],
                                              errors="coerce").to_numpy(dtype=float)
    for primary, _, _, _ in levels:
        slim_cols[primary] = pd.to_numeric(df[primary], errors="coerce").to_numpy(dtype=float)
    slim = pd.DataFrame(slim_cols)

    # Rows eligible to define any reference curve: a real quantity, a real RT,
    # not a decoy, not held out, and confident enough to be worth trusting.
    def _eligible(primary):
        keep = ref_mask.copy()
        q = slim[primary].to_numpy()
        keep &= np.isfinite(q) & (q > cfg.min_intensity) & np.isfinite(rt_all)
        if cfg.qvalue_col in slim.columns:
            qv = pd.to_numeric(slim[cfg.qvalue_col], errors="coerce").to_numpy(dtype=float)
            keep &= np.isfinite(qv) & (qv < cfg.qvalue_cutoff)
        return keep

    def _clip(v):
        return np.clip(v, -cfg.max_log2_shift, cfg.max_log2_shift)

    for primary, factor_col, cols, vec_cols in levels:
        log2_corr = np.zeros(n, dtype=float)
        n_done = 0

        def _corrected(rows):
            """Reference frame with the corrections applied so far divided out,
            so each stage only has to explain what the previous ones left."""
            sub = slim.iloc[rows].copy()
            sub[primary] = sub[primary].to_numpy() * np.exp2(-log2_corr[rows])
            return sub

        # ---- paired stage --------------------------------------------------
        # Only the *differences* between time channels, measured precursor by
        # precursor so no composition or detection-selection effect can leak in.
        # A no-op on single-channel data, where there is nothing to pair against.
        def run_paired(x_col="rt"):
            nonlocal n_done
            if not (cfg.two_stage and prec_col is not None and len(uniq) > 1):
                return False
            if x_col not in slim.columns and x_col not in (OWN_MEAN, OWN_CHAN):
                return False
            sub = _corrected(np.arange(n))
            paired = _paired_channel_residuals(sub, chan, _eligible(primary),
                                               primary, prec_col, cfg, x_col=x_col)
            if x_col in (OWN_MEAN, OWN_CHAN):
                y_all = np.log2(np.clip(sub[primary].to_numpy(), 1e-12, None))
                if x_col == OWN_CHAN:
                    x_apply = y_all
                else:
                    x_apply = pd.Series(y_all).groupby(
                        slim[prec_col].to_numpy()).transform("mean").to_numpy()
            else:
                x_apply = (rt_all if x_col == "rt"
                           else slim[x_col].to_numpy(dtype=float))
            for ch, (prt, pres) in paired.items():
                # center=None => the curve is centred on its own median, so this
                # stage changes each channel's *shape* and never its level.
                # Between-channel loading differences are a property of the
                # samples, not of the drift, and are left alone.
                out = _channel_correction(prt, pres, cfg, tic)
                if out is None:
                    continue
                sel = np.flatnonzero(chan == ch)
                log2_corr[sel] += _clip(np.interp(x_apply[sel], out[0], out[1]))
            if paired:
                n_done = max(n_done, len(paired))
                logger.info(f"RT normalization: paired stage over {x_col} corrected "
                            f"{len(paired)} of {len(uniq)} time channels for {primary}")
            return bool(paired)

        # ---- unpaired stage ------------------------------------------------
        # The running centre of the detected population. ``pooled`` fits one
        # curve for the whole run (cannot touch a channel-vs-channel ratio);
        # otherwise one curve per channel, which is the classic single-stage
        # correction and the only option when there is a single channel.
        def run_unpaired(pooled):
            nonlocal n_done
            groups = ([(None, np.arange(n))] if pooled
                      else [(ch, np.flatnonzero(chan == ch)) for ch in uniq])
            for key, sel in groups:
                rows = sel[ref_mask[sel]]
                ref = _channel_reference(_corrected(rows), primary, cfg, group_col)
                if ref is None:
                    logger.warning(
                        f"RT normalization: {'run' if key is None else f'channel {key}'}"
                        f" has too few reference precursors for {primary}; "
                        f"left uncorrected")
                    continue
                out = _channel_correction(ref[0], ref[1], cfg, tic)
                if out is None:
                    continue
                # np.interp clamps at the edges (approxfun rule = 2), so
                # precursors outside the reference RT span get the nearest
                # fitted correction rather than a runaway extrapolation.
                log2_corr[sel] += _clip(np.interp(rt_all[sel], out[0], out[1]))
                n_done = max(n_done, len(uniq) if key is None else n_done + 1)

        if cfg.stage_order in ("unpaired_first", "unpaired_paired_unpaired"):
            # Flatten every channel's own running centre first, then let the
            # paired stage repair the between-channel damage that per-channel
            # flattening does (the channels are staggered in RT, so "flat in its
            # own frame" is not the same correction for two of them).
            run_unpaired(pooled=False)
            paired_ok = run_paired()
            if cfg.intensity_stage:
                run_paired(x_col=(OWN_CHAN if cfg.intensity_axis == "own_channel"
                                  else OWN_MEAN))
            if cfg.mz_stage:
                # Same paired contrast, smoothed against precursor m/z instead
                # of RT, for whatever channel-specific transmission difference
                # RT could not explain.
                run_paired(x_col=cfg.mz_col)
            if cfg.stage_order == "unpaired_paired_unpaired":
                # Re-flattening each channel after the paired stage largely
                # *undoes* it: the paired correction is exactly what displaced
                # each channel's running centre away from flat, so removing that
                # displacement removes the correction. Kept only so the
                # benchmark can show it.
                run_unpaired(pooled=False)
        else:
            paired_ok = run_paired()
            if cfg.run_stage == "always" or (cfg.run_stage == "auto" and not paired_ok):
                run_unpaired(pooled=paired_ok)

        # ---- optional: put every channel on one level ----------------------
        # A separate, explicitly opt-in step. The stages above only remove RT
        # *shape*; this equalises the channels' absolute levels, which is right
        # only when they are meant to carry the same total load.
        if cfg.center_scope == "global" and len(uniq) > 1:
            elig = _eligible(primary)
            centers = {}
            for ch in uniq:
                rows = np.flatnonzero((chan == ch) & elig)
                if len(rows) >= cfg.min_ref:
                    centers[ch] = _center_of(
                        np.log2(slim[primary].to_numpy()[rows]) - log2_corr[rows], cfg)
            if len(centers) > 1:
                target = float(np.mean(list(centers.values())))
                for ch, c in centers.items():
                    log2_corr[chan == ch] += c - target

        factor = np.exp2(-log2_corr)
        df[factor_col] = factor
        shifts[primary] = log2_corr.copy()
        for c in cols:
            vals = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float) * factor
            if inplace_cols:
                df[c + "_raw"] = df[c]
                df[c] = vals
            else:
                df[c + SUFFIX] = vals

        # Per-scan and per-fragment intensities ride the same factor. The factor
        # is one number per precursor, so every element of a row's vector is
        # scaled by it: the scale moves and the shape does not, which leaves
        # every correlation, cosine and Pearson computed from these columns
        # identical before and after.
        for c in vec_cols:
            out = _scale_vector_column(df[c], factor)
            if inplace_cols:
                df[c + "_raw"] = df[c]
                df[c] = out
            else:
                df[c + SUFFIX] = out
        if vec_cols:
            logger.info(f"RT normalization: scaled per-scan/per-fragment "
                        f"intensities {', '.join(vec_cols)}")

        logger.info(f"RT normalization [{cfg.describe()}] applied to {primary} "
                    f"across {n_done}/{len(uniq)} channel(s); "
                    f"median |log2 shift| = {np.median(np.abs(log2_corr)):.3f}")

    if plot_path:
        try:
            plot_rtnorm_diagnostics(df, chan, shifts, plot_path, cfg,
                                    ref_mask=ref_mask)
        except Exception as e:          # a figure must never fail a search
            logger.warning(f"RT normalization diagnostic plot failed: {e}")

    return df


def plot_rtnorm_diagnostics(df, chan, shifts, path, cfg=None, ref_mask=None,
                            bins=40):
    """Figure showing what the normalization did, per time channel.

    Top row: the correction itself -- log2 factor against RT, one line per
    channel. This is the curve that was fitted, and it is the thing to look at
    when a run behaves oddly.

    Bottom row: the running median of log2 quant against RT, before and after.
    The point of the correction is that the *after* trace is flat, so a bottom
    panel that still slopes means the curve did not take.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not shifts:
        return
    rt = df["rt"].to_numpy(dtype=float)
    keep = np.isfinite(rt)
    if ref_mask is not None:
        keep = keep & np.asarray(ref_mask, dtype=bool)
    chans = [c for c in pd.unique(chan) if c == c]
    levels = [(p, FACTOR_MS1 if p == MS1_PRIMARY else FACTOR_MS2)
              for p in shifts]

    # Rows share a y-axis: with independent scaling, two panels showing the same
    # absolute drift look completely different, and the more-zoomed one reads as
    # a failed correction when it is nothing of the kind.
    fig, axes = plt.subplots(2, len(levels), figsize=(6.2 * len(levels), 7.2),
                             squeeze=False, sharey="row")
    edges = np.linspace(np.nanmin(rt[keep]), np.nanmax(rt[keep]), bins + 1)
    mids = 0.5 * (edges[:-1] + edges[1:])
    cmap = plt.get_cmap("tab10")

    for j, (primary, _) in enumerate(levels):
        corr = shifts[primary]
        q = pd.to_numeric(df[primary], errors="coerce").to_numpy(dtype=float)
        ok = keep & np.isfinite(q) & (q > 0)
        lo = np.log2(np.where(ok, q, np.nan))

        ax_c, ax_q = axes[0][j], axes[1][j]
        for i, ch in enumerate(sorted(chans)):
            m = ok & (chan == ch)
            if m.sum() < 20:
                continue
            colour = cmap(i % 10)
            lab = f"time channel {ch:g}" if len(chans) > 1 else "run"
            o = np.argsort(rt[m])
            ax_c.plot(rt[m][o], -corr[m][o], lw=1.6, color=colour, label=lab)

            idx = np.clip(np.digitize(rt[m], edges) - 1, 0, bins - 1)
            before = np.array([np.nanmedian(lo[m][idx == b]) if (idx == b).sum() > 5
                               else np.nan for b in range(bins)])
            after = np.array([np.nanmedian((lo[m] - corr[m])[idx == b])
                              if (idx == b).sum() > 5 else np.nan
                              for b in range(bins)])
            # Centred so the two traces are comparable as shapes, which is what
            # is being judged -- the level is deliberately left alone.
            ax_q.plot(mids, before - np.nanmedian(before), lw=1.2, ls="--",
                      color=colour, alpha=0.55,
                      label=f"{lab} before" if len(chans) <= 3 else None)
            ax_q.plot(mids, after - np.nanmedian(after), lw=1.9, color=colour,
                      label=f"{lab} after" if len(chans) <= 3 else None)

        ax_c.axhline(0, color="k", lw=0.8, ls=":")
        ax_c.set_title(f"{primary}: correction applied", fontsize=11)
        ax_c.set_ylabel("log2 factor")
        ax_c.legend(fontsize=8)
        ax_c.grid(alpha=0.25)

        ax_q.axhline(0, color="k", lw=0.8, ls=":")
        ax_q.set_title(f"{primary}: running median, dashed = before, "
                       f"solid = after", fontsize=11)
        ax_q.set_xlabel("retention time (min)")
        ax_q.set_ylabel("centred log2 quantity")
        ax_q.legend(fontsize=8)
        ax_q.grid(alpha=0.25)

    fig.suptitle("RT-dependent quantitative normalization"
                 + (f"  [{cfg.describe()}]" if cfg is not None else ""),
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"RT normalization diagnostic plot -> {path}")


def config_from_args(args) -> RTNormConfig:
    """Build an RTNormConfig from parsed JMod CLI args."""
    return RTNormConfig(
        method=getattr(args, "rt_norm_method", "loess_cap"),
        window=int(getattr(args, "rt_norm_window", 400)),
        rt_window=float(getattr(args, "rt_norm_rt_window", 2.0)),
        loess_frac=float(getattr(args, "rt_norm_loess_frac", 0.05)),
        qvalue_cutoff=float(getattr(args, "rt_norm_qvalue", 0.05)),
        plex_agg=getattr(args, "rt_norm_plex_agg", "mean"),
        ref_top_frac=float(getattr(args, "rt_norm_top_frac", 1.0)),
        two_stage=bool(getattr(args, "rt_norm_two_stage", True)),
        run_stage=getattr(args, "rt_norm_run_stage", "auto"),
        stage_order=getattr(args, "rt_norm_stage_order", "unpaired_first"),
        mz_stage=bool(getattr(args, "rt_norm_mz_stage", False)),
        min_ref=int(getattr(args, "rt_norm_min_prec", 50)),
        min_span=int(getattr(args, "rt_norm_min_span", 50)),
        min_intensity=float(getattr(args, "rt_norm_min_intensity", 1.0)),
        center_scope=getattr(args, "rt_norm_center_scope", "channel"),
    )
