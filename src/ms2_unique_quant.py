"""
This Source Code Form is subject to the terms of the Oxford Nanopore
Technologies, Ltd. Public License, v. 1.0.  Full licence can be found
at https://github.com/ParallelSquared/JMod/blob/main/LICENSE.txt
"""

"""
MS2 quantitation from fragments that can be assigned to one plexDIA channel.

mTRAQ labels the peptide N-terminus and lysine (``rules = "nK"``).  A b-ion
always contains the N-terminus so it always carries a tag; a y-ion carries one
only when a lysine falls inside it.  For a tryptic peptide ending in R the
y-ions of every channel therefore land at the *same* m/z, and the observed peak
is the superposition of all channels.  Measured on JD0435: 99.6% of y-ions from
R-terminating precursors share an m/z across channels against 0.0% for
K-terminating ones, and R-terminating precursors are 46.9% of confident IDs.
Those precursors keep only ~44% of their library intensity in channel-resolved
peaks.

``coeff`` is fitted over all matched fragments, superposed ones included.  This
module refits over the channel-resolved fragments only and reports one column,
``coeff_ChannelUnique``, at the same MS2 scan JMod fitted ``coeff`` at.

Summed variants over the surrounding scans were tried and dropped: on JD0435
they recovered fold change worse than the single apex value (0.79 / 0.75 / 0.90
against 0.93) at ~40% worse spread.  An R-terminating precursor is fitted from
3-4 fragments, so summing scans adds their errors rather than averaging them.

Fit
---
Each precursor's channels are fitted alone against the observed peaks -- the
co-isolated competitors that JMod's search-time NNLS carries as extra columns
are *not* included.  That is deliberate and cheaper, and it costs less than it
sounds: "channel-unique" means the fragment m/z sets of the channels are
disjoint, so the design matrix is block diagonal and the fit decouples into one
independent least-squares scale per channel,
``c = sum(lib*obs) / sum(lib^2)``, which for non-negative inputs is already the
NNLS solution.  The trade is that interference from *other peptides* landing on
a channel-unique fragment is not deconvolved away.

Observed intensities are re-extracted from the MS2 spectra here rather than read
from ``obs_int``: that column is populated at every library m/z whether or not a
peak was matched (verified in the source parquet -- zero fraction 0.000, and
``frac_int_matched`` recomputed from it is 1.000 against a reported 0.079), so
fitting through it fits through noise.

Cost is controlled by fitting only rows that passed FDR
(``BestChannel_Qvalue < q_gate`` and not a decoy); everything else gets NaN.
"""

from typing import Optional

import numpy as np
import pandas as pd

try:
    from src.logger import logger
except Exception:  # pragma: no cover
    import logging
    logger = logging.getLogger(__name__)

NAME_COL = "frag_names"
MZ_COL = "frag_mz"
LIB_COL = "frag_int"
FRACLIB_COL = "frac_lib_int"
PREC_COL = "untag_prec"
TC_COL = "time_channel"
WIN_COL = "window_mz"
RT_COL = "rt"
SPEC_COL = "spec_id"
APEX_COL = "ms1_apex_scan"

SEQ_COL = "seq"

QUANT_COL = "coeff_ChannelUnique"
UNIQUE_COLS = (QUANT_COL,)


def _as_array(v, dtype=float):
    """Fragment list columns arrive as lists, arrays or bracketed strings."""
    if v is None:
        return np.empty(0, dtype=dtype)
    if isinstance(v, str):
        toks = v.replace("[", " ").replace("]", " ").replace(",", " ").split()
        try:
            return np.asarray([dtype(t) for t in toks], dtype=dtype)
        except ValueError:
            return np.empty(0, dtype=dtype)
    try:
        return np.asarray(list(v), dtype=dtype)
    except (TypeError, ValueError):
        return np.empty(0, dtype=dtype)


def jmod_coefficient(lib, obs, frac_lib_int, mask=None, weights=None):
    """The coefficient JMod's MS2 NNLS solves for, with rows optionally dropped.

    ``spectral_fitting`` fits sum-to-1 normalized library patterns against the
    matched DIA peaks, and appends one extra row per candidate holding the
    library intensity that was NOT matched, against an observed value of 0.
    That row penalises a candidate whose predicted peaks are largely absent.
    For a single candidate the whole thing is closed form:

        a_i   = (lib_i / sum(lib)) * frac_lib_int      matched rows
        P     = 1 - frac_lib_int                       penalty row
        coeff = sum(a_i * obs_i) / (sum(a_i^2) + P^2)

    ``frac_lib_int`` rescales because ``frag_int`` holds only the *matched*
    fragments: their share of a sum-to-1 full library is ``frac_lib_int``, and
    the rest is the penalty.

    Checked against the reported ``coeff`` on K-terminating precursors -- where
    no fragment is superposed, so this must agree with JMod: r = 0.908 in log2,
    median ratio 0.78.  The gap that remains is the joint deconvolution against
    co-isolated competitors, which this deliberately does not do.

    ``mask`` drops rows from the matched sum -- exactly what removing a
    superposed fragment from the fit means.  The penalty row is kept: library
    intensity that was never matched is still unmatched.
    """
    lib = np.asarray(lib, dtype=float)
    obs = np.asarray(obs, dtype=float)
    ok = np.isfinite(lib) & np.isfinite(obs) & (lib > 0)
    if not ok.any():
        return np.nan
    total = float(np.sum(lib[ok]))
    if total <= 0:
        return np.nan
    f = float(frac_lib_int) if np.isfinite(frac_lib_int) and 0 < frac_lib_int <= 1 else 1.0
    a = (lib / total) * f
    use = ok if mask is None else (ok & np.asarray(mask, dtype=bool))
    if not use.any():
        return np.nan
    if weights is None:
        denom = float(np.sum(a[use] ** 2) + (1.0 - f) ** 2)
        if denom <= 0:
            return np.nan
        return float(np.sum(a[use] * obs[use]) / denom)
    # Weighted least squares over the same rows.  Weights must be a function of
    # the FRAGMENT only, identical in every channel of a precursor, or the
    # coefficient becomes a different linear functional per channel and every
    # ratio is biased.  Normalised to mean 1 over the used rows so the balance
    # against the unmatched-library penalty row is preserved and equal weights
    # reproduce the unweighted result exactly.
    w = np.asarray(weights, dtype=float)
    if w.shape != lib.shape:
        denom = float(np.sum(a[use] ** 2) + (1.0 - f) ** 2)
        return float(np.sum(a[use] * obs[use]) / denom) if denom > 0 else np.nan
    w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
    mw = float(np.mean(w[use])) if use.any() else 0.0
    if mw <= 0:
        return np.nan
    w = w / mw
    denom = float(np.sum(w[use] * a[use] ** 2) + (1.0 - f) ** 2)
    if denom <= 0:
        return np.nan
    return float(np.sum(w[use] * a[use] * obs[use]) / denom)


def channel_unique_masks(names, mzs, mz_tol=0.01):
    """Per-row mask of fragments whose m/z differs across a precursor's channels.

    Decided from the recorded m/z rather than from the tag rules, so it stays
    correct for missed cleavages (an internal K makes some y-ions resolvable),
    for other tags, and for SILAC.  A precursor seen in one channel only has
    nothing to compare against, so every fragment counts as unique.
    """
    if len(names) < 2:
        return [np.ones(len(n), dtype=bool) for n in names]
    seen = {}
    for n, z in zip(names, mzs):
        for key, val in zip(n, z):
            seen.setdefault(key, []).append(val)
    shared = {k for k, v in seen.items()
              if len(v) > 1 and (max(v) - min(v)) <= mz_tol}
    return [np.array([k not in shared for k in n], dtype=bool) for n in names]


def _extract(spec, mzs, ppm):
    """Summed intensity within ppm of each m/z; 0 where nothing is there."""
    out = np.zeros(len(mzs))
    if spec is None:
        return out
    arr_mz, arr_int = spec.mz, spec.intens
    for i, mz in enumerate(mzs):
        tol = mz * ppm * 1e-6
        lo = np.searchsorted(arr_mz, mz - tol)
        hi = np.searchsorted(arr_mz, mz + tol)
        if hi > lo:
            out[i] = float(np.sum(arr_int[lo:hi]))
    return out


def _window_scans(spectra, window_mz, tol=0.51):
    """MS2 scans of one isolation window, in acquisition order."""
    out = []
    for s in getattr(spectra, "ms2scans", []) or []:
        pm = getattr(s, "prec_mz", None)
        if pm is not None and abs(float(pm) - float(window_mz)) <= tol:
            out.append(s)
    return out




def add_channel_unique_coeff(df: pd.DataFrame, spectra=None,
                             plexDIA: bool = True, ppm: float = 10.0,
                             n_adjacent: int = 1, q_gate: float = 0.01,
                             mz_tol: float = 0.01, tag=None,
                             isotope_correct: bool = False,
                             n_iso: int = 5,
                             mz_tol_da: float = 0.015) -> pd.DataFrame:
    """Add the ``coeff_ChannelUnique*`` family to a JMod quant table.

    ``isotope_correct`` (requires ``tag``, and a ``seq`` column carrying the
    modified/tagged sequence) additionally removes the isotope-envelope
    crosstalk a sibling channel's isotope tail leaves in this channel's
    "unique" fragment window -- see ``src.ms2_isotope_xtalk``.  Off by
    default: it adds a joint NNLS solve per channel-unique fragment key and
    is only worth the cost for tags whose channel step is numerically close
    to a multiple of the natural isotope spacing (mTRAQ-style tags are;
    check a new tag's ``delta`` before assuming it needs this).
    """
    need = (NAME_COL, MZ_COL, LIB_COL, PREC_COL, WIN_COL, RT_COL)
    if isotope_correct:
        need = need + (SEQ_COL,)
    missing = [c for c in need if c not in df.columns]
    if missing:
        logger.warning(f"{QUANT_COL} skipped: missing {', '.join(missing)}")
        return df
    if not plexDIA:
        logger.info(f"{QUANT_COL} skipped (plexDIA off): with no labelled "
                    f"channels no fragment is superposed across channels")
        return df
    if spectra is None:
        logger.warning(f"{QUANT_COL} skipped: no spectra to refit against")
        return df
    if isotope_correct and tag is None:
        logger.warning(f"{QUANT_COL}: isotope_correct requested but no tag "
                       f"was given -- falling back to the plain extraction")
        isotope_correct = False

    n = len(df)
    # ``q_gate=None`` fits every candidate, decoys included.  That is required
    # when the column feeds the scorer or the RT-normalization curve: a value
    # that exists only for confident targets predicts the label, and a
    # normalization reference restricted to them inherits the same selection.
    fit_me = np.ones(n, dtype=bool)
    if q_gate is not None:
        if "BestChannel_Qvalue" in df.columns:
            fit_me &= pd.to_numeric(df["BestChannel_Qvalue"],
                                    errors="coerce").to_numpy() < q_gate
        if "is_decoy" in df.columns:
            fit_me &= ~df["is_decoy"].to_numpy(dtype=bool)

    names = [_as_array(v, dtype=str) for v in df[NAME_COL]]
    mzs = [_as_array(v) for v in df[MZ_COL]]
    libs = [_as_array(v) for v in df[LIB_COL]]
    seqs = df[SEQ_COL].astype(str).to_numpy() if isotope_correct else None
    iso_cache = {} if isotope_correct else None
    if isotope_correct:
        from src.ms2_isotope_xtalk import normalize_frag_name
        names = [np.array([normalize_frag_name(x) for x in nm]) for nm in names]
    fracs = (pd.to_numeric(df[FRACLIB_COL], errors="coerce").to_numpy()
             if FRACLIB_COL in df.columns else np.ones(len(df)))
    win = pd.to_numeric(df[WIN_COL], errors="coerce").to_numpy()
    rts = pd.to_numeric(df[RT_COL], errors="coerce").to_numpy()
    specid = (pd.to_numeric(df[SPEC_COL], errors="coerce").to_numpy()
              if SPEC_COL in df.columns else np.full(n, np.nan))
    apex = (pd.to_numeric(df[APEX_COL], errors="coerce").to_numpy()
            if APEX_COL in df.columns else np.full(n, np.nan))

    q = np.full(n, np.nan)
    a_ = np.full(n, np.nan)
    p_ = np.full(n, np.nan)
    c_ = np.full(n, np.nan)
    nfr = np.zeros(n, dtype=int)
    frac = np.full(n, np.nan)

    key = df[PREC_COL].astype(str)
    if TC_COL in df.columns:
        key = key + "|" + df[TC_COL].astype(str)

    win_cache = {}
    n_groups = 0
    for _, idx in pd.Series(np.arange(n)).groupby(key.to_numpy()):
        # Uniqueness is a property of the plex GROUP, so it must be decided
        # from every channel present -- including ones the q-value gate will
        # exclude from the refit. Comparing only the surviving channels would
        # call a superposed fragment "unique" whenever its partners happened to
        # miss the cutoff (measured: R-terminating precursors jump from 44% to
        # 91% of library intensity retained, which is the bug not the signal).
        all_rows = idx.to_numpy()
        all_masks = channel_unique_masks([names[i] for i in all_rows],
                                         [mzs[i] for i in all_rows], mz_tol)
        mask_of = dict(zip(all_rows.tolist(), all_masks))
        keep = fit_me[all_rows]
        rows = all_rows[keep]
        masks = [m for m, k in zip(all_masks, keep) if k]
        if rows.size == 0:
            continue
        # Contamination only runs from lower to higher mass, so processing
        # channels in that order lets each one use its lower-mass siblings'
        # ALREADY-fitted coefficients as known contamination sources -- see
        # ms2_isotope_xtalk.cascading_channel_fit for why that beats a single
        # joint solve. known_coeffs accumulates across this group only.
        known_coeffs = {}
        if isotope_correct:
            from src.ms2_isotope_xtalk import channel_label_from_seq
            order = sorted(range(len(rows)),
                           key=lambda k: tag.mass_dict[
                               f"{tag.name}-{channel_label_from_seq(seqs[rows[k]], tag)}"])
            rows = rows[order]
            masks = [masks[k] for k in order]
        w = win[rows[0]]
        if not np.isfinite(w):
            continue
        wkey = round(float(w), 2)
        if wkey not in win_cache:
            win_cache[wkey] = _window_scans(spectra, w)
        scans = win_cache[wkey]
        if not scans:
            continue
        srt = np.array([s.RT for s in scans])
        n_groups += 1

        # One shared lookup: MS1 scan number -> RT, so an MS1-defined apex can
        # be located among this window's MS2 scans.
        def _pos_of_ms1(scan_num):
            if not np.isfinite(scan_num):
                return None
            t = getattr(spectra, "scan_pos", {}).get(int(scan_num))
            if t is None or t[0] != 1:
                return None
            return int(np.argmin(np.abs(srt - spectra.ms1scans[t[1]].RT)))

        def _pos_of_ms2(scan_num):
            if not np.isfinite(scan_num):
                return None
            for k, sc in enumerate(scans):
                if sc.scan_num == int(scan_num):
                    return k
            return None

        for i, m in zip(rows, masks):
            lib, mz = libs[i], mzs[i]
            if lib.size != m.size or mz.size != m.size or not m.any():
                continue
            nfr[i] = int(m.sum())
            tot = float(np.nansum(lib))
            frac[i] = float(np.nansum(lib[m]) / tot) if tot > 0 else np.nan

            # Anchors are per ROW, not per group: each channel of a precursor is
            # fitted at its own spec_id and can have its own voted apex.  The
            # extraction span is built to cover every anchor this row needs plus
            # a full +/-n either side, so no window is ever clipped.  Centring
            # the span on the group's first row instead left 10.7% of rows with
            # their own scan outside the trace and clipped the +/-1 window for
            # 21.9% -- summing 2 scans where a neighbour got 3.
            # Headline value: the single scan JMod fitted coeff at, resolved
            # through scan_pos rather than by searching this isolation window's
            # scan list.  The search misses whenever window_mz does not match
            # prec_mz within tolerance and then silently falls back to a
            # neighbouring scan -- that alone cost 0.931 -> 0.823 of fold-change
            # recovery.  scan_pos cannot miss.
            def _coeff_at(sc):
                if isotope_correct:
                    from src.ms2_isotope_xtalk import cascading_channel_fit
                    return cascading_channel_fit(
                        i, all_rows.tolist(), seqs, names, mzs, libs, mask_of,
                        fracs, known_coeffs,
                        lambda mzarr: _extract(sc, mzarr, ppm), tag,
                        n_iso=n_iso, mz_tol_da=mz_tol_da, cache=iso_cache)
                return jmod_coefficient(lib, _extract(sc, mz, ppm), fracs[i], mask=m)

            q[i] = np.nan
            pos_direct = getattr(spectra, "scan_pos", {}).get(int(specid[i])) \
                if np.isfinite(specid[i]) else None
            if pos_direct is not None and pos_direct[0] == 2:
                q[i] = _coeff_at(spectra.ms2scans[pos_direct[1]])

            if not np.isfinite(q[i]):
                # Fallback only.  The trace costs an extraction and a fit at
                # every scan of the span -- about five times the work of the
                # headline value -- and is consulted for the ~1.7% of rows whose
                # spec_id does not resolve, so it is built lazily rather than for
                # every row.  Identical result, and it is what makes this
                # affordable on the full ungated candidate table.
                p_spec = _pos_of_ms2(specid[i])
                p_apex = _pos_of_ms1(apex[i])
                anchors = [p for p in (p_spec, p_apex) if p is not None]
                if not anchors:
                    anchors = [int(np.argmin(np.abs(srt - rts[i])))]
                pad = max(n_adjacent, 1) + 1
                lo = max(0, min(anchors) - pad)
                hi = min(len(scans), max(anchors) + pad + 1)
                sel = list(range(lo, hi))
                trace = np.array([_coeff_at(scans[k]) for k in sel])
                if not np.isfinite(trace).any():
                    continue
                cen = anchors[0] - lo
                q[i] = float(trace[cen]) if np.isfinite(trace[cen]) else np.nan

            if isotope_correct:
                known_coeffs[i] = q[i]

    df[QUANT_COL] = q

    ok = np.isfinite(q)
    logger.info(
        f"{QUANT_COL}: refitted {int(ok.sum()):,} of {int(fit_me.sum()):,} "
        f"FDR-passing rows across {n_groups:,} plex groups; median "
        f"{int(np.median(nfr[ok])) if ok.any() else 0} channel-resolved "
        f"fragments carrying a median "
        f"{100*np.nanmedian(frac[ok]) if ok.any() else 0:.0f}% of the library "
        f"intensity")
    return df
