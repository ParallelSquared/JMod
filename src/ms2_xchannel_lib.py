"""
This Source Code Form is subject to the terms of the Oxford Nanopore
Technologies, Ltd. Public License, v. 1.0.  Full licence can be found
at https://github.com/ParallelSquared/JMod/blob/main/LICENSE.txt
"""

"""
Rescore MS2 library-fitting features against an *empirical* library.

The spectral library is generic: it knows nothing about this instrument, this
gradient or this collision energy, so every library-fitting feature the scorer
uses (scribe, goodness-of-fit, Manhattan, spectral contrast, the residual
extremes) is measured against an expectation that is systematically a little
wrong -- for targets and decoys alike.  Under plexDIA the same peptide is
measured once per label channel in the same run, so the channels themselves
supply an empirical expectation that carries all of that run-specific detail.

This builds that empirical library from the IDs the original library produced,
refits each channel against it *per precursor* (never re-running the search-time
NNLS over whole scans), and recomputes the same feature family.  The original
features are kept; these are additions, and the per-feature deltas are emitted
too because "how much better does the empirical library fit" is the quantity of
interest.

Feature definitions are taken verbatim from
``spectral_fitting._compute_candidate_features_jit`` so the two families are
computed identically and only the library differs:

    scribe          sum_k (sqrt(h_k)/S_h - sqrt(x_k)/S_x)^2      lower = better
    gof             log2(sum|res| / sum_fitted)                  lower = better
    max_matched     log2(max|res| where obs>0 / sum_fitted)      lower = better
    max_unmatched   log2(max|res| where obs~0 / sum_fitted)      lower = better
    manhattan       -log2(sum|pred-obs| / sum obs)               higher = better
    contrast        sqrt(uv) / (sqrt(u2)*sqrt(v2))               higher = better

One difference from production, stated rather than hidden: production's
residuals are the *joint* residuals of every candidate fitted in the window,
while these are single-precursor residuals (pred = c * lib for this precursor
alone).  That is forced -- an empirical library exists only for this precursor
-- so ``orig_*`` is recomputed the same isolated way rather than read from the
report.  Only ``orig_*`` vs ``emp_*`` is a clean library contrast; comparing
``emp_*`` against the report's own scribe/gof would confound library with fit.

Leakage, two kinds, both closed
-------------------------------
1. **Label.**  A channel's empirical library is built only from *other* channels
   of the same precursor, never from itself.  Targets and decoys are handled
   identically and never mix.
2. **Self-signal.**  mTRAQ tags the N-terminus and lysine, so for an
   R-terminating peptide the y-ions of every channel land at one m/z and the
   observed peak is their superposition.  Building channel i's library from such
   a fragment in channel j would feed i's own intensity back to it.  So both the
   library and the scoring use **channel-resolved fragments only**.

Shared (non-resolvable) fragments
---------------------------------
mTRAQ tags the N-terminus and lysine, so an R-terminating peptide's y-ions land
at one m/z in every label channel and the recorded peak is their superposition.
Those fragments used to be **deleted** -- with the resolved survivors
renormalised among themselves, an R-terminating precursor kept ~36% of its
library intensity and 1.7% of rows kept none at all.

They are now kept, with the peak divided among the channels in proportion to the
quants estimated from their **unique** fragments:

    share_i = O_k * (c_i f_i) / sum_j (c_j f_j)

The true relative intensity of fragment k is a property of the peptide, so it is
common to every channel and CANCELS out of that ratio -- the split needs no
library value at all, which matters because the library's relative intensities
are exactly what this module exists to improve on.  ``c_j`` comes from the
sibling's own resolved fragments, never from the shared peak, so nothing
circular enters.  Sharing is decided per fragment m/z within one time channel
AND requires the same isolation window: mTRAQ moves the precursor ~4 Da per
channel, groups straddle two windows 37.9% of the time, and channels in
different windows never co-populate one MS2 scan.

Measured on JD0413_re against nine alternatives (absolute library-predicted
subtraction, a consensus MS2 apex, apex +/-1 averaging, sibling contribution as
features, and combinations): this is the only variant that beat the previous
behaviour on distinct precursors -- 9,955 against 9,902, with rows, best-channel
and AUC all moving up with it.  Every other variant traded peptide coverage for
AUC.  The gain is +0.54% on one run with one seed and has no error bar.

Superposed positions keep their ORIGINAL library relative intensity; the
empirical evidence reshapes only the block the siblings can speak to, rescaled
so the original library's resolved-vs-shared intensity split is preserved -- that
split is the one quantity no channel can measure.

Rows with no sibling channel
----------------------------
A precursor seen in only one channel has no empirical library.  Those rows do
**not** get a sentinel: ``xchan_emp_*`` falls back to the original-library value,
so the column means "the best library available for this row" and is continuous
across the whole table.  The deltas are then 0 by construction, which is the
correct reading -- no improvement was available.

The sentinel would have been actively harmful.  48.1% of targets have a sibling
channel against 43.0% of decoys, so a missingness pattern is itself weak
evidence of the label; encoding it as a distinct -1 value hands the scorer that
correlation through the back door instead of through ``xchan_n_informing``,
where it can be weighed honestly.

One residual asymmetry, stated rather than hidden: for a row that *has* an
empirical library the features are computed over the fragments the siblings
support, and for a row that does not they are computed over all of its
channel-resolved fragments.  The two populations therefore rest on slightly
different fragment subsets.  That is unavoidable and is far milder than a
sentinel.
"""
import math
import os

import numpy as np
import pandas as pd

try:
    from src.logger import logger
except Exception:  # pragma: no cover
    import logging
    logger = logging.getLogger(__name__)

SENTINEL = -1.0
_EPS = 1e-10

BASE_FEATS = ("scribe", "gof", "max_matched", "max_unmatched", "manhattan",
              "contrast", "cos")
FEATURES = (tuple(f"xchan_emp_{f}" for f in BASE_FEATS)
            + tuple(f"xchan_orig_{f}" for f in BASE_FEATS)
            + tuple(f"xchan_d_{f}" for f in BASE_FEATS)
            + ("xchan_n_informing", "xchan_n_frag", "xchan_lib_dispersion",
               "xchan_coeff_ratio", "xchan_w_top_frac"))

# The channel-unique fit is already performed here -- it is the sibling weight --
# so the quantity and the features that fall out of it are emitted rather than
# recomputed by a second pass over the same spectra.  ``ms2_unique_quant``
# reports only the quantity and discards the fragment count and library share
# after logging them; the residual and the ratio to ``coeff`` were never
# computed at all, which is why they had to live in a bench script.
CU_QUANT = "coeff_ChannelUnique"

CU_FEATURES = ("cu_n_frag_unique", "cu_frac_lib_unique", "cu_resid_frac",
               "cu_ratio_over_coeff")

# DIAGNOSTIC ONLY -- never scoring features (they are in fdr_analysis's drop
# list).  coeff_ChannelUnique recomputed from the cleanest and dirtiest half of
# the channel-unique fragments, where "dirty" means a wide isotope envelope and
# so more crosstalk per unit of neighbour abundance (a b3 carries far less than
# a b10).  Both halves come from the SAME observation vector and the SAME split,
# and the split is fixed once per precursor group from a single reference
# sequence -- identical for every channel, so the two coefficients stay
# comparable across channels and any ratio built from them is unbiased.  This
# exists to test whether weighting fragments by purity is worth building.
# Each half is emitted at BOTH ends of the subtraction scale: "_a0" is the
# uncorrected fit, the bare name is the fully-corrected one (alpha=1).  Since
# jmod_coefficient is linear in the observation, those two pin the whole line
#     coeff(alpha) = coeff_a0 + alpha * (coeff_a1 - coeff_a0)
# so alpha can be calibrated per RUN afterwards, from the requirement that the
# clean and dirty halves -- two estimates of the same abundance -- agree.  That
# needs no ground truth, so it works on ordinary samples and adapts to window
# width and dynamic range instead of baking in a constant fitted on one file.
PURITY_DIAG = ("coeff_CU_cleanFrags", "coeff_CU_dirtyFrags",
               "coeff_CU_cleanFrags_a0", "coeff_CU_dirtyFrags_a0",
               "cu_purity_median_s")

# Fragment-purity weighting swept in ONE run.  w_k = 1/(1+(kappa*s_k)^2) with
# s_k the fragment's M+2 share, so kappa sets where the crossover from
# noise-limited to interference-limited sits.  The form is inverse-variance if
# the residual over-subtraction scales with s (it should: the amount subtracted
# is proportional to s), but the SCALE has to be measured rather than guessed --
# emitting several at once costs one extra closed-form evaluation each and
# avoids a run per value.
PURITY_KAPPAS = (1.0, 2.0, 4.0, 8.0, 16.0)
# ...plus the two parameter-free forms.  "inv" is w = 1/s, "inv2" is w = 1/s^2.
# 1/s^2 is the formal inverse-variance weight if the residual error scales with
# s; 1/s is milder; the 1/(1+(kappa*s)^2) family is 1/s^2 with a noise floor,
# which is what a real fragment has.  Only measurement settles which.
PURITY_SWEEP = (tuple(f"coeff_CU_pw{int(k)}" for k in PURITY_KAPPAS)
                + ("coeff_CU_pwInv", "coeff_CU_pwInv2"))


def _helpers():
    try:
        from src.ms2_unique_quant import (_as_array, _extract,
                                          channel_unique_masks, jmod_coefficient)
        return _as_array, _extract, channel_unique_masks, jmod_coefficient
    except Exception:
        import importlib.util
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "ms2_unique_quant.py")
        spec = importlib.util.spec_from_file_location("_muq_for_xchan", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return (mod._as_array, mod._extract, mod.channel_unique_masks,
                mod.jmod_coefficient)


def spectral_features(lib, obs, coeff):
    """The six library-fitting features, formulas as in spectral_fitting.

    ``lib`` is the normalized expected pattern, ``obs`` the observed intensities
    at the same fragments, ``coeff`` the fitted scale.  Single-precursor
    residuals: pred = coeff * lib.

    Written as one Python loop over plain floats rather than a dozen numpy
    reductions.  These vectors hold 3-10 fragments, where a ``np.sum`` call
    spends microseconds of dispatch to add five numbers: profiling the caller
    showed 1.3 million such calls costing 4 s, more than the extraction of the
    spectra themselves.  Same arithmetic, same results to 1e-12.
    """
    out = {f: np.nan for f in BASE_FEATS}
    n = len(lib)
    if n == 0 or n != len(obs) or not np.isfinite(coeff):
        return out

    s_h = s_x = 0.0
    sum_res = sum_fit = x_sum = man = 0.0
    u2 = v2 = uv = l2 = o2 = lo = 0.0
    mm = mu = 0.0
    for k in range(n):
        L = lib[k]
        o = obs[k]
        pred = coeff * L
        res = o - pred
        ares = res if res >= 0.0 else -res
        if L > 0.0:
            s_h += math.sqrt(L)
        if o > 0.0:
            s_x += math.sqrt(o)
        sum_res += ares
        sum_fit += pred if pred >= 0.0 else -pred
        if o > 1e-6:
            if ares > mm:
                mm = ares
        elif o < 1e-6:
            if ares > mu:
                mu = ares
        x_sum += o
        d = pred - o
        man += d if d >= 0.0 else -d
        u2 += pred * pred
        v2 += o * o
        uv += pred * o
        l2 += L * L
        o2 += o * o
        lo += L * o

    if s_h > 0.0 and s_x > 0.0:
        scribe = 0.0
        for k in range(n):
            L = lib[k]
            o = obs[k]
            h = (math.sqrt(L) / s_h) if L > 0.0 else 0.0
            x = (math.sqrt(o) / s_x) if o > 0.0 else 0.0
            scribe += (h - x) ** 2
        out["scribe"] = scribe

    if sum_res == 0.0:
        sum_res = 1e-6
    if sum_fit == 0.0:
        sum_fit = 1e-6
    out["gof"] = math.log2(sum_res / sum_fit)
    out["max_matched"] = math.log2(mm / (sum_fit + _EPS) + _EPS)
    out["max_unmatched"] = math.log2(mu / (sum_fit + _EPS) + _EPS)

    # Plain cosine between the expected pattern and the observation.  JMod's
    # ``spectral_contrast`` is sqrt(uv)/(sqrt(u2)*sqrt(v2)), which is not scale
    # free and collapses to ~1e-4 on raw intensities; this is the quantity a
    # reader means by "cosine similarity", so it is reported alongside.
    if l2 > 0.0 and o2 > 0.0:
        out["cos"] = lo / (math.sqrt(l2) * math.sqrt(o2))

    if x_sum > 0.0 and man > 0.0:
        out["manhattan"] = -math.log2(man / x_sum)
        out["contrast"] = math.sqrt(uv if uv > 0.0 else 0.0) / (
            math.sqrt(u2) * math.sqrt(v2) + _EPS)
    return out


def _unit(v):
    s = float(np.sum(v))
    return v / s if s > 0 else None


def _coeff_fast(lib, obs, frac_lib_int, mask=None):
    """``jmod_coefficient`` as a plain loop.

    Identical arithmetic to the shared helper -- including the lumped penalty
    row ``(1 - f)^2`` -- but without a numpy reduction per term.  It is called
    ~3x per row on 3-10 element vectors, where dispatch dominates: 98k calls
    cost 2.3 s in the profile.  Verified equal to the shared version to 1e-12.
    """
    n = len(lib)
    total = 0.0
    for k in range(n):
        L = lib[k]
        o = obs[k]
        if L > 0.0 and L == L and o == o:
            total += L
    if total <= 0.0:
        return float("nan")
    f = float(frac_lib_int)
    if not (f == f and 0.0 < f <= 1.0):
        f = 1.0
    scale = f / total
    num = 0.0
    den = 0.0
    used = False
    for k in range(n):
        L = lib[k]
        o = obs[k]
        if not (L > 0.0 and L == L and o == o):
            continue
        if mask is not None and not mask[k]:
            continue
        a = L * scale
        num += a * o
        den += a * a
        used = True
    if not used:
        return float("nan")
    den += (1.0 - f) ** 2
    if den <= 0.0:
        return float("nan")
    return num / den


def _extract_fast(spec, mzs, ppm):
    """``_extract`` with the two searchsorted calls hoisted out of the loop.

    The shared helper calls ``np.searchsorted`` twice *per fragment*; profiling
    showed 553k such calls costing 2.7 s.  This does two vectorized calls per
    spectrum instead.  A running cumulative sum would be worse, not better --
    that is O(peaks) per row against O(fragments x log peaks) here -- so the
    per-fragment totals are taken directly, with the overwhelmingly common
    single-peak case short-circuited.
    """
    n = len(mzs)
    out = np.zeros(n)
    if spec is None or n == 0:
        return out
    arr_mz, arr_int = spec.mz, spec.intens
    tol = mzs * (ppm * 1e-6)
    lo = np.searchsorted(arr_mz, mzs - tol)
    hi = np.searchsorted(arr_mz, mzs + tol)
    for i in range(n):
        a = lo[i]
        b = hi[i]
        if b > a:
            out[i] = arr_int[a] if b - a == 1 else float(arr_int[a:b].sum())
    return out


def add_xchannel_features(df: pd.DataFrame, spectra=None, ppm: float = 10.0,
                          mz_tol: float = 0.01, min_frag: int = 3,
                          min_score_frag: int = 1, weight_by: str = "cu",
                          sibling_scope: str = "precursor",
                          group_cols=("untag_prec", "time_channel"),
                          purity_diag: bool = False,
                          min_siblings: int = 0,
                          purity_kappa: float = 0.0,
                          row_mask=None, progress_every: int = 0,
                          shared_frag_mode: str = "proportional",
                          tag=None, isotope_correct: bool = True,
                          n_iso: int = 5,
                          mz_tol_da: float = 0.015) -> pd.DataFrame:
    """Add the ``xchan_*`` family.  Returns ``df``, modified in place.

    ``coeff_ChannelUnique`` (``CU_QUANT``) is also produced here -- this is
    the code path production actually uses when ``ms2_prescoring`` is on
    (the default), NOT ``ms2_unique_quant.add_channel_unique_coeff``, whose
    own call in ``fdr_analysis.py`` is skipped once this has already filled
    the column.  ``isotope_correct`` therefore has to be threaded through
    here too, or it silently never runs. See ``src.ms2_isotope_xtalk``.
    """
    need = ["frag_names", "frag_mz", "frag_int", "frac_lib_int", "spec_id"]
    if isotope_correct:
        need = need + ["seq"]
    missing = [c for c in need if c not in df.columns]
    if missing or spectra is None:
        logger.warning(f"cross-channel MS2 features skipped: "
                       f"missing {missing or 'spectra'}")
        return df
    gcols = [c for c in group_cols if c in df.columns]
    if not gcols:
        logger.warning("cross-channel MS2 features skipped: no grouping column")
        return df
    if isotope_correct and tag is None:
        logger.warning(f"{CU_QUANT}: isotope_correct requested but no tag was "
                       f"given -- falling back to the plain extraction")
        isotope_correct = False

    _as_array, _extract, channel_unique_masks, jmod_coefficient = _helpers()
    _extract_raw = _extract
    _extract = _extract_fast          # same result, two searchsorted calls per spectrum
    # _coeff_fast is the hot path but takes no weights; keep the reference
    # implementation for the weighted variants, which run far less often.
    _coeff_ref = jmod_coefficient
    jmod_coefficient = _coeff_fast    # same arithmetic, no numpy dispatch per term

    n = len(df)
    names = [_as_array(v, dtype=str) for v in df["frag_names"]]
    mzs = [_as_array(v) for v in df["frag_mz"]]
    libs = [_as_array(v) for v in df["frag_int"]]
    seqs = df["seq"].astype(str).to_numpy() if isotope_correct else None
    iso_cache = {} if isotope_correct else None
    # Precursor m/z and charge, for working out how much of a contaminating
    # channel's envelope the isolation window actually admits. Optional: a
    # frame without them just falls back to assuming the whole envelope got in,
    # which is what a wide window does anyway.
    win_iso_cache = {} if isotope_correct else None
    prec_mzs = prec_zs = None
    if isotope_correct:
        mz_col = next((c for c in ("prec_mz", "mz") if c in df.columns), None)
        z_col = next((c for c in ("z", "charge") if c in df.columns), None)
        if mz_col is not None and z_col is not None:
            prec_mzs = pd.to_numeric(df[mz_col], errors="coerce").to_numpy()
            prec_zs = pd.to_numeric(df[z_col], errors="coerce").to_numpy()
        else:
            logger.warning(
                f"{CU_QUANT}: no precursor m/z + charge columns "
                f"(looked for prec_mz/mz and z/charge) -- isolation-window "
                f"truncation of the contaminating envelope will be skipped")
    if isotope_correct:
        from src.ms2_isotope_xtalk import normalize_frag_name
        names = [np.array([normalize_frag_name(x) for x in nm]) for nm in names]
    fracs = pd.to_numeric(df["frac_lib_int"], errors="coerce").to_numpy()
    specid = pd.to_numeric(df["spec_id"], errors="coerce").to_numpy()
    # Only channels co-isolated in the SAME window can superpose.
    wins = (pd.to_numeric(df["window_mz"], errors="coerce").to_numpy()
            if "window_mz" in df.columns else np.zeros(len(df)))

    out = {f: np.full(n, SENTINEL, dtype=float) for f in FEATURES}
    pur = {c: np.full(n, np.nan) for c in PURITY_DIAG} if purity_diag else None
    swp = ({c: np.full(n, np.nan) for c in PURITY_SWEEP}
           if (purity_diag and tag is not None) else None)
    cu_q = np.full(n, np.nan)
    cu_nfrag = np.zeros(n, dtype=float)
    cu_frac = np.full(n, np.nan)
    cu_resid = np.full(n, np.nan)
    out["xchan_n_informing"] = np.zeros(n, dtype=float)
    out["xchan_n_frag"] = np.zeros(n, dtype=float)

    scan_pos = getattr(spectra, "scan_pos", {}) or {}
    ms2 = getattr(spectra, "ms2scans", []) or []
    fit_me = (np.ones(n, dtype=bool) if row_mask is None
              else np.asarray(row_mask, dtype=bool))

    # Two different groupings, and conflating them is the trap.
    #
    # The uniqueness mask asks "does this fragment collide with another channel
    # *in the same spectrum*", which happens only among the mTRAQ labels of one
    # time channel, so it is always computed over (untag_prec, time_channel).
    #
    # The sibling pool asks "where else was this peptide measured", and that is
    # every channel of the precursor.  Time channels are separate emitters
    # staggered in RT -- measured on JD0413, the same precursor+label in two time
    # channels is a median of 8.2 min apart and 0.00% are within 1 min -- so they
    # are wholly independent measurements and safe to pool.  Pooling only within
    # a time channel leaves a mean of 2.27 channels; pooling across the precursor
    # gives 5.90, and 48,079 precursors have all nine.
    #
    # Widening the *mask* grouping instead would be wrong: the same label in two
    # time channels has identical fragment m/z, so every fragment would be called
    # shared and the mask would empty out.
    mask_key = df[gcols].astype(str).agg("|".join, axis=1).to_numpy()
    pool_key = (df["untag_prec"].astype(str).to_numpy()
                if sibling_scope == "precursor" else mask_key)

    groups = list(pd.Series(np.arange(n)).groupby(pool_key))
    for gi, (_, idx) in enumerate(groups):
        if progress_every and gi % progress_every == 0:
            logger.info(f"  xchannel group {gi:,}/{len(groups):,}")
        rows = idx.to_numpy()
        # uniqueness decided per time channel, inside the pool
        masks_by_row = {}
        # Same-scan sibling rows only (one time channel's worth), for the
        # isotope correction below: contamination is a same-spectrum effect,
        # so pooling across time channels (like the sibling_scope="precursor"
        # pool above) would compare signal from different points in time.
        same_scan_siblings = {}
        for _, sub in pd.Series(rows).groupby(mask_key[rows]):
            sr = sub.to_numpy()
            for i, m in zip(sr, channel_unique_masks([names[i] for i in sr],
                                                     [mzs[i] for i in sr], mz_tol)):
                masks_by_row[i] = m
                same_scan_siblings[i] = sr
        masks = [masks_by_row[i] for i in rows]

        # NOTE: do NOT narrow this to rows sharing an isolation window. It is
        # tempting -- on JD1187's 250 narrow windows only 48.2% of
        # immediate-predecessor pairs have their MONOISOTOPIC precursors in the
        # same window (94.2% on JD0588's 20 wide ones) -- but the monoisotopic
        # peak is the wrong thing to test. The predecessor contaminates through
        # its precursor ISOTOPE ENVELOPE, and the tag step is built to be
        # 2 x 13C per tag site, so the predecessor's M+2*n_tags precursor
        # isotopologue sits on the target channel's own precursor m/z to within
        # 0.0001 mDa. Whenever the target is inside the window that isotopologue
        # is too, by construction. Worse, every isotopologue able to reach the
        # target's fragment m/z needs at least 2*n_tags neutrons, so they all
        # sit at or ABOVE the target's precursor -- squarely inside the window.
        # Gating on the monoisotopic peak would therefore delete the dominant,
        # always-present contamination path, not a spurious one.

        # Contamination only runs from lower to higher mass, so processing
        # channels in that order lets each one use its lower-mass siblings'
        # ALREADY-fitted coefficients as known contamination sources -- see
        # ms2_isotope_xtalk.cascading_channel_fit. known_coeffs accumulates
        # across this group only.
        # Purity split for the diagnostic, keyed by fragment NAME rather than
        # by position.  Channels of one precursor match different fragment
        # SUBSETS, so requiring identical arrays fired on only ~1% of groups.
        # What the premise actually needs is that a GIVEN fragment carries the
        # same weight in every channel -- i.e. the rule is a function of
        # fragment identity -- which a name-keyed map satisfies whatever subset
        # each channel happens to have.  Susceptibility and the median are both
        # computed once per group from a single reference sequence, so nothing
        # here varies by channel.
        s_by_name, med_s = None, np.nan
        if purity_diag and tag is not None and seqs is not None and len(rows):
            try:
                from src.ms2_isotope_xtalk import (fragment_isotope_pattern,
                                                   base_frag_key)
                ref = rows[0]
                s_by_name = {}
                for r in rows:
                    for nmk, keep_k in zip(np.asarray(names[r]),
                                           masks_by_row.get(r, [])):
                        k = str(nmk)
                        if not keep_k or "_iso" in k or k in s_by_name:
                            continue
                        try:
                            _o, _r = fragment_isotope_pattern(
                                seqs[ref], base_frag_key(k), tag,
                                n_iso=max(3, n_iso), cache=iso_cache)
                            s_by_name[k] = float(_r[2])   # M+2: envelope width
                        except Exception:
                            pass
                if len(s_by_name) >= 4:
                    med_s = float(np.median(list(s_by_name.values())))
                else:
                    s_by_name = None
            except Exception:
                s_by_name = None

        # Fragment-purity weights: down-weight fragments with a WIDE isotope
        # envelope, which receive proportionally more crosstalk from the
        # neighbouring channel (a b12 carries ~6x the M+2 share of a b3).
        # Measured on JD0588 with an equal-count split at the group median:
        # the clean half gives compression 0.915 and sd 1.287, the dirty half
        # 1.177 and 1.696 -- clean wins on accuracy AND precision, and beats
        # the all-fragment coefficient (0.876) using half the evidence.
        # Keyed by fragment NAME from one reference sequence so a given
        # fragment carries the SAME weight in every channel; otherwise each
        # channel's coefficient is a different functional and ratios bias.
        w_by_name = None
        if purity_kappa and tag is not None and seqs is not None and len(rows):
            try:
                from src.ms2_isotope_xtalk import (fragment_isotope_pattern,
                                                   base_frag_key)
                ref = rows[0]
                w_by_name = {}
                for r in rows:
                    for nmk, keep_k in zip(np.asarray(names[r]),
                                           masks_by_row.get(r, [])):
                        k = str(nmk)
                        if not keep_k or "_iso" in k or k in w_by_name:
                            continue
                        try:
                            _o, _r = fragment_isotope_pattern(
                                seqs[ref], base_frag_key(k), tag,
                                n_iso=max(3, n_iso), cache=iso_cache)
                            sk = float(_r[2])
                            w_by_name[k] = 1.0 / (1.0 + (purity_kappa * sk) ** 2)
                        except Exception:
                            pass
                if len(w_by_name) < 2:
                    w_by_name = None
            except Exception:
                w_by_name = None

        known_coeffs = {}
        if isotope_correct:
            from src.ms2_isotope_xtalk import channel_label_from_seq
            mass_order = sorted(range(len(rows)),
                                key=lambda k: tag.mass_dict[
                                    f"{tag.name}-{channel_label_from_seq(seqs[rows[k]], tag)}"])
            rows = rows[mass_order]
            masks = [masks[k] for k in mass_order]

        # Which rows share each fragment peak.  Keyed on (rounded m/z, window):
        # decided inside one time channel, and only among channels the same
        # isolation window put into the same MS2 scan.
        share = {}
        if shared_frag_mode == "proportional":
            for _, sub in pd.Series(rows).groupby(mask_key[rows]):
                for i in sub.to_numpy():
                    wk = round(float(wins[i]), 2) if np.isfinite(wins[i]) else 0.0
                    for k, mzv in enumerate(mzs[i]):
                        share.setdefault((round(float(mzv) / mz_tol), wk),
                                         []).append((i, k))

        pattern, obs_of, mask_of, chan_w = {}, {}, {}, {}
        # Leave-one-out by subtraction.  Averaging each row against its siblings
        # directly is O(channels^2 x fragments) -- on a 9-plex that is 9 rows x 8
        # siblings.  Accumulating every channel once and then removing the row's
        # own contribution gives the identical leave-one-out average in
        # O(channels x fragments), an ~8x saving at 9 channels.  ``tot_n`` counts
        # contributors so a fragment supported only by the row itself is
        # correctly treated as having no sibling evidence, rather than relying on
        # a weight difference that has cancelled to a rounding residue.
        tot_v, tot_w, tot_n = {}, {}, {}
        for i, m in zip(rows, masks):
            mask_of[i] = m
            lib, mz = libs[i], mzs[i]
            if lib.size == 0 or lib.size != m.size or mz.size != m.size:
                continue
            pos = scan_pos.get(int(specid[i])) if np.isfinite(specid[i]) else None
            if pos is None or pos[0] != 2 or pos[1] >= len(ms2):
                continue
            # The observed vector is kept even when nothing is channel-resolved,
            # so such a row can still be scored against the original library.
            # This is always the plain (uncorrected) extraction: the isotope
            # correction below produces coeff_ChannelUnique directly, and does
            # not feed the separate cross-channel empirical-library features
            # (``pattern``/``chan_w``) built from ``obs`` here -- those are a
            # different, unrelated scoring mechanism this change leaves alone.
            obs = _extract(ms2[pos[1]], mz, ppm)
            obs_of[i] = obs
            if m.any():
                u = _unit(obs[m])
                if u is not None:
                    # keyed by fragment NAME: the same fragment sits at a
                    # different m/z in every channel
                    pattern[i] = dict(zip(np.asarray(names[i])[m], u))
                    # This channel's abundance, i.e. coeff_ChannelUnique: how
                    # much weight its spectrum deserves in the average.  A bright
                    # channel measures the shape better, and if the per-fragment
                    # noise is Poisson then weights proportional to intensity are
                    # the right ones for averaging shapes.
                    w = jmod_coefficient(lib, obs, fracs[i], mask=m)
                    obs_corr = obs_raw0 = None
                    chan_w[i] = float(w) if np.isfinite(w) and w > 0 else 0.0
                    # This *is* coeff_ChannelUnique -- unless isotope_correct,
                    # in which case it is recomputed instead (see
                    # ms2_isotope_xtalk.cascading_channel_fit): channels are
                    # processed in ascending mass order (above), so every
                    # lower-mass sibling's OWN coefficient is already known by
                    # the time this row is reached, and isotope-envelope
                    # contamination from it is subtracted before refitting
                    # this channel's own jmod_coefficient -- replacing both
                    # the plain extraction above and this combination for
                    # this row only.
                    if isotope_correct:
                        from src.ms2_isotope_xtalk import cascading_channel_fit
                        _sc = ms2[pos[1]]
                        _win = getattr(_sc, "ms1window", None)
                        w, obs_corr, obs_raw0 = cascading_channel_fit(
                            i, same_scan_siblings[i].tolist(), seqs, names,
                            mzs, libs, masks_by_row, fracs, known_coeffs,
                            lambda mzarr, sc=_sc: _extract_raw(sc, mzarr, ppm),
                            tag, n_iso=n_iso, mz_tol_da=mz_tol_da,
                            cache=iso_cache,
                            prec_mz=(prec_mzs[i] if prec_mzs is not None else None),
                            prec_z=(prec_zs[i] if prec_zs is not None else None),
                            window_hi=(float(_win[1]) if _win is not None
                                       and len(_win) > 1 else None),
                            window_lo=(float(_win[0]) if _win is not None
                                       and len(_win) > 0 else None),
                            win_cache=win_iso_cache, return_obs=True,
                            min_siblings=min_siblings,
    )
                    # Emit it, plus the three quantities that fall out of the
                    # same fit for free.
                    if swp is not None and s_by_name is not None:
                        sv = np.array([s_by_name.get(str(x), np.nan)
                                       for x in np.asarray(names[i])])
                        if np.isfinite(sv).sum() >= 2:
                            o_w = (obs_corr if (isotope_correct
                                                and obs_corr is not None)
                                   else obs)
                            sf = np.where(np.isfinite(sv),
                                          np.maximum(sv, 1e-3), np.nan)
                            forms = [(cn, 1.0 / (1.0 + (kap * np.nan_to_num(
                                          sf, nan=0.0)) ** 2))
                                     for kap, cn in zip(PURITY_KAPPAS,
                                                        PURITY_SWEEP)]
                            forms.append(("coeff_CU_pwInv", 1.0 / sf))
                            forms.append(("coeff_CU_pwInv2", 1.0 / sf ** 2))
                            for cn, wv in forms:
                                wv = np.where(np.isfinite(sv),
                                              np.nan_to_num(wv, nan=0.0), 0.0)
                                vv = _coeff_ref(lib, o_w, fracs[i],
                                                mask=m, weights=wv)
                                if np.isfinite(vv):
                                    swp[cn][i] = float(vv)
                    if w_by_name is not None:
                        wv = np.array([w_by_name.get(str(x), np.nan)
                                       for x in np.asarray(names[i])])
                        if np.isfinite(wv).sum() >= 2:
                            wv = np.where(np.isfinite(wv), wv, 0.0)
                            o_w = (obs_corr if (isotope_correct
                                                and obs_corr is not None)
                                   else obs)
                            w2 = _coeff_ref(lib, o_w, fracs[i], mask=m,
                                            weights=wv)
                            if np.isfinite(w2):
                                w = w2
                    if np.isfinite(w):
                        cu_q[i] = float(w)
                    if pur is not None:
                        # same observation vector, same split -> the only thing
                        # differing between the two is fragment purity
                        o_use = obs_corr if (isotope_correct
                                             and obs_corr is not None) else obs
                        pur["cu_purity_median_s"][i] = med_s
                        if s_by_name is not None and o_use is not None:
                            sv = np.array([s_by_name.get(str(x), np.nan)
                                           for x in np.asarray(names[i])])
                            good = np.isfinite(sv)
                            cm = m & good & (sv <= med_s)
                            dm = m & good & (sv > med_s)
                            o_raw = obs_raw0 if obs_raw0 is not None else obs
                            if cm.any():
                                pur["coeff_CU_cleanFrags"][i] = jmod_coefficient(
                                    lib, o_use, fracs[i], mask=cm)
                                pur["coeff_CU_cleanFrags_a0"][i] = jmod_coefficient(
                                    lib, o_raw, fracs[i], mask=cm)
                            if dm.any():
                                pur["coeff_CU_dirtyFrags"][i] = jmod_coefficient(
                                    lib, o_use, fracs[i], mask=dm)
                                pur["coeff_CU_dirtyFrags_a0"][i] = jmod_coefficient(
                                    lib, o_raw, fracs[i], mask=dm)
                    if isotope_correct:
                        known_coeffs[i] = w
                    tot_lib = 0.0
                    sub_lib = 0.0
                    nfrag = 0
                    for k in range(len(lib)):
                        L = lib[k]
                        if L > 0.0 and L == L and obs[k] == obs[k]:
                            tot_lib += L
                            if m[k]:
                                sub_lib += L
                                nfrag += 1
                    cu_nfrag[i] = nfrag
                    if tot_lib > 0.0:
                        cu_frac[i] = sub_lib / tot_lib
                    if np.isfinite(w) and tot_lib > 0.0 and nfrag:
                        f_ = fracs[i] if (fracs[i] == fracs[i]
                                          and 0.0 < fracs[i] <= 1.0) else 1.0
                        num = 0.0
                        den = 0.0
                        for k in range(len(lib)):
                            L = lib[k]
                            if not (L > 0.0 and L == L and obs[k] == obs[k]) or not m[k]:
                                continue
                            r_ = obs[k] - w * (L / tot_lib) * f_
                            num += r_ * r_
                            den += obs[k] * obs[k]
                        if den > 0.0:
                            cu_resid[i] = math.sqrt(num) / math.sqrt(den)

        # one pass over every channel: running totals per fragment name
        def _w_of(j):
            if weight_by == "cu":
                return chan_w.get(j, 0.0)
            if weight_by == "sqrt_cu":
                return math.sqrt(max(chan_w.get(j, 0.0), 0.0))
            return 1.0

        for j, pj in pattern.items():
            wj = _w_of(j)
            if wj <= 0:
                continue
            for nmk, v in pj.items():
                tot_v[nmk] = tot_v.get(nmk, 0.0) + wj * v
                tot_w[nmk] = tot_w.get(nmk, 0.0) + wj
                tot_n[nmk] = tot_n.get(nmk, 0) + 1

        for i in rows:
            if not fit_me[i] or i not in obs_of:
                continue
            m = mask_of[i]
            # A precursor whose every matched fragment is superposed across
            # channels -- an R-terminating peptide with only y-ions matched, 1.7%
            # of rows on JD0413 and 99.6% of them R-terminating -- has no
            # channel-resolved fragment at all.  It can never take an empirical
            # library, because building one would feed it its own signal back
            # through the shared peak.  But the original-library fallback has no
            # cross-channel component to contaminate, so score it over all of
            # its matched fragments rather than leaving it empty.
            # Every matched fragment is scored: the shared ones are recovered
            # by the proportional split below rather than deleted.
            score_mask = (np.ones(m.size, dtype=bool)
                          if shared_frag_mode == "proportional"
                          else (m if m.any() else np.ones(m.size, dtype=bool)))
            nm_i = np.asarray(names[i])[score_mask]
            obs_i = obs_of[i][score_mask]
            lib_i = libs[i][score_mask]
            others = [j for j in pattern if j != i] if m.any() else []
            out["xchan_n_informing"][i] = len(others)

            # Which fragments the siblings can speak to.  With no sibling, or
            # too little overlap, the row still gets its original-library
            # features over all of its own channel-resolved fragments.
            use = np.zeros(nm_i.size, dtype=bool)
            ref = np.zeros(nm_i.size)
            if others:
                # Weight per fragment, not per channel: siblings do not all
                # resolve the same fragments, so each position is normalized by
                # the weight that actually contributed to it.  The running totals
                # cover every channel, so this row's own contribution is removed
                # here -- that subtraction IS the leave-one-out.
                pi = pattern.get(i)
                wi = _w_of(i) if pi is not None else 0.0
                cnt = np.zeros(nm_i.size)
                for k, nmk in enumerate(nm_i):
                    n_other = tot_n.get(nmk, 0)
                    v = tot_v.get(nmk, 0.0)
                    w = tot_w.get(nmk, 0.0)
                    if wi > 0 and pi is not None and nmk in pi:
                        n_other -= 1
                        v -= wi * pi[nmk]
                        w -= wi
                    if n_other > 0 and w > 0:
                        ref[k] = v / w
                        cnt[k] = w
                use = cnt > 0
                if use.sum() < min_frag:
                    use = np.zeros(nm_i.size, dtype=bool)
                if use.any():
                    ws = [chan_w.get(j, 0.0) for j in others]
                    tot = float(sum(ws))
                    if tot > 0:
                        # 1.0 means one sibling supplied the whole library
                        out["xchan_w_top_frac"][i] = float(max(ws) / tot)

            has_emp = bool(use.any())
            if shared_frag_mode == "proportional":
                # Both libraries live on the full fragment set.  Siblings speak
                # only to the positions in ``use``; the resolved-vs-shared
                # intensity split is kept from the original library, since no
                # channel can measure it (a sibling's value at a shared peak
                # contains this channel's own signal).
                sel = np.ones(nm_i.size, dtype=bool)
                orig = _unit(lib_i)
                emp_vec = None
                if orig is not None and has_emp:
                    w_res = float(orig[use].sum())
                    e = ref[use]
                    se = float(e.sum())
                    if se > 0.0 and w_res > 0.0:
                        emp_vec = orig.copy()
                        emp_vec[use] = w_res * e / se
                    else:
                        has_emp = False

                # Divide each shared peak among its channels in proportion to
                # the quants fitted on their UNIQUE fragments.
                o = obs_i.copy()
                idx_of = np.flatnonzero(score_mask)
                f_i = fracs[i] if np.isfinite(fracs[i]) else 1.0
                own_w = chan_w.get(i, 0.0) * f_i
                if own_w > 0.0:
                    wk_i = round(float(wins[i]), 2) if np.isfinite(wins[i]) else 0.0
                    for k in range(nm_i.size):
                        kk = int(idx_of[k])
                        sh = share.get((round(float(mzs[i][kk]) / mz_tol), wk_i))
                        if not sh or len(sh) < 2:
                            continue
                        denom = 0.0
                        for (j, _kj) in sh:
                            cj = chan_w.get(j, 0.0)
                            if cj <= 0.0:
                                continue
                            fj = fracs[j] if np.isfinite(fracs[j]) else 1.0
                            denom += cj * fj
                        if denom > 0.0:
                            o[k] = float(obs_i[k]) * (own_w / denom)
            else:
                sel = use if has_emp else np.ones(nm_i.size, dtype=bool)
                o = obs_i[sel]
                orig = _unit(lib_i[sel])
                emp_vec = None
            # ``min_frag`` gates only whether the siblings can supply a library.
            # Scoring against the ORIGINAL library needs no such floor -- scribe,
            # gof and Manhattan are all defined on one or two fragments -- and
            # applying it here left 99,530 rows (8.3%) with no features at all
            # for no reason.
            if orig is None or o.size < min_score_frag:
                continue
            out["xchan_n_frag"][i] = int(sel.sum())

            f = fracs[i] if np.isfinite(fracs[i]) else 1.0
            c_org = jmod_coefficient(orig, o, f)
            fo = spectral_features(orig, o, c_org)

            if has_emp:
                # Same estimator, same fragments, same scan -- only the expected
                # pattern differs, so every delta is a pure library contrast.
                emp = (emp_vec if shared_frag_mode == "proportional"
                       else _unit(ref[use]))
                if emp is None:
                    has_emp = False
            if has_emp:
                c_emp = jmod_coefficient(emp, o, f)
                fe = spectral_features(emp, o, c_emp)
                if np.isfinite(c_emp) and np.isfinite(c_org) and c_org > 0:
                    out["xchan_coeff_ratio"][i] = float(c_emp / c_org)
            else:
                # No sibling evidence: the best library for this row IS the
                # original one, so the empirical column carries that value and
                # the delta is 0.  See the module docstring on why this beats a
                # sentinel.
                fe = fo
                out["xchan_coeff_ratio"][i] = 1.0

            for k in BASE_FEATS:
                if np.isfinite(fe[k]):
                    out[f"xchan_emp_{k}"][i] = fe[k]
                if np.isfinite(fo[k]):
                    out[f"xchan_orig_{k}"][i] = fo[k]
                if np.isfinite(fe[k]) and np.isfinite(fo[k]):
                    out[f"xchan_d_{k}"][i] = fe[k] - fo[k]

            if len(others) >= 2:
                cs = []
                for a in range(len(others)):
                    for b in range(a + 1, len(others)):
                        pa, pb = pattern[others[a]], pattern[others[b]]
                        sh = [k for k in pa if k in pb]
                        if len(sh) >= min_frag:
                            va = np.array([pa[k] for k in sh])
                            vb = np.array([pb[k] for k in sh])
                            na, nb = np.linalg.norm(va), np.linalg.norm(vb)
                            if na > 0 and nb > 0:
                                cs.append(float(np.dot(va, vb) / (na * nb)))
                if cs:
                    out["xchan_lib_dispersion"][i] = float(np.median(cs))

    for f in FEATURES:
        df[f] = out[f]
    df[CU_QUANT] = cu_q
    df["cu_n_frag_unique"] = cu_nfrag
    df["cu_frac_lib_unique"] = cu_frac
    df["cu_resid_frac"] = cu_resid
    if "coeff" in df.columns:
        _c = pd.to_numeric(df["coeff"], errors="coerce").to_numpy(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            df["cu_ratio_over_coeff"] = np.where(_c > 0, cu_q / _c, np.nan)
    else:
        df["cu_ratio_over_coeff"] = np.nan
    scored = out["xchan_n_frag"] > 0
    emp = scored & (out["xchan_d_gof"] != 0.0)
    logger.info(
        f"cross-channel empirical-library features: {int(scored.sum()):,} of "
        f"{n:,} rows scored; {int(emp.sum()):,} against an empirical library "
        f"(median {int(np.median(out['xchan_n_informing'][emp])) if emp.any() else 0} "
        f"sibling channels), the rest fall back to their original-library values")
    if swp is not None:
        for c in PURITY_SWEEP:
            df[c] = swp[c]
    if pur is not None:
        for c in PURITY_DIAG:
            df[c] = pur[c]
        got = np.isfinite(pur["coeff_CU_cleanFrags"]) & np.isfinite(
            pur["coeff_CU_dirtyFrags"])
        logger.info(
            f"fragment-purity diagnostic: {int(got.sum()):,} of {n:,} rows got "
            f"both halves; median M+2 susceptibility "
            f"{np.nanmedian(pur['cu_purity_median_s']):.4f}")
    if isotope_correct:
        from src.ms2_isotope_xtalk import dump_stats
        dump_stats(logger)
    return df
