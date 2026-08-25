"""
This Source Code Form is subject to the terms of the Oxford Nanopore
Technologies, Ltd. Public License, v. 1.0.  Full licence can be found
at https://github.com/ParallelSquared/JMod/blob/main/LICENSE.txt
"""

"""
MS2 quantitation anchored on a *common* apex shared by a precursor's channels.

``plex_Area`` sums the MS1 fit around one apex that every channel of a plex
group shares.  The MS2 quantities do not: ``coeff`` and ``coeff_ChannelUnique``
are each evaluated at whatever MS2 scan that channel happened to be fitted at.
Measured on JD0413_re, only **31.3%** of plex groups have all their channels
fitted at the same MS2 scan; the spread between them is a median of 26 spectrum
ids and 238 at the 90th percentile.  Two channels of one precursor are therefore
routinely read at different points on the elution profile, which is a direct
channel-vs-channel ratio error and is exactly what a common apex removes.

The anchor cannot be a scan *number*, because mTRAQ shifts the precursor m/z by
~4 Da per channel and the channels of one group are split across two isolation
windows 37.9% of the time (measured, same run).  So it is defined in **retention
time**, and each channel is read at the nearest MS2 scan *of its own isolation
window* to that time.

Which time, in practice: the **median of the group's channels' own fitted RTs**.
The code below prefers the RT of ``ms1_apex_scan`` and falls back to that median,
but the fallback is what always runs -- ``ms1_apex_scan`` is not in
``ms2_prescoring.SCALARS``, so the column never reaches this module and ``apex``
is all-NaN for every row.  Even if it were passed, it would not help: the column
is written during MS1 quant, *after* scoring, so at the point this runs it is 0
for 87.0% of rows (measured on JD0413_re; the 13.0% that are non-zero are all
valid MS1 scans, and their count is exactly the best-channel ID total).

The property these columns actually depend on -- one common time for the whole
group, so a channel-vs-channel ratio is not read off two different points of the
elution profile -- holds either way.  But the median of per-channel argmax picks
is a weaker anchor than an MS1 apex would be: it is not robust to a channel
fitted on a shoulder, and not robust to an interferent inflating one scan.

An MS2 analogue of ``plex_Area``'s rule was measured as a replacement -- apex =
argmax over scans of Pearson(predicted, observed) for the whole plex group, with
predicted at fragment k summed over the channels sharing that peak.  It cuts
ratio compression appreciably (at a designed -3 within a plexDIA set, 94.8% ->
99.4% of the designed fold change) but widens the human 1:1 control IQR by
~8-10%.  Not adopted: the accuracy/precision trade was judged not worth changing
a shipped quant column over.  See bench/ms2_apex_quant.py.

Four columns, all optional and additive.  Every one of them is anchored on the
GROUP'S COMMON APEX -- the ``_Area`` pair are simply the same quantity summed
over anchor +/- ``ms2_area_adjacent_scans`` instead of taken at the anchor scan
alone.  The shipped names are shortened (there is no separate
``*_CommonApex_Area`` column; ``coeff_ChannelUnique_Area`` IS that quantity):

    coeff_CommonApex             all matched fragments, at the common anchor
    coeff_Area                   the same, summed over anchor +/- n_adjacent
    coeff_ChannelUnique_CommonApex   channel-resolved fragments, common anchor
    coeff_ChannelUnique_Area         the same, summed over anchor +/- n_adjacent

n_adjacent is worth setting per acquisition geometry: measured on JD1187
(2.0 m/z windows, channel RT spread 0.33 min) +/-1 scan takes median
per-precursor error 1.206 -> 1.122 with the slope unchanged (1.042 -> 1.027),
while on JD0588 (18 m/z windows, spread 0.055 min) every extra scan hurts both
metrics monotonically.  Hence the flag rather than a constant.

Note on windows: the scan list is built per **row**, from that row's own
``window_mz``.  ``add_channel_unique_coeff`` builds one list for the whole group
from ``win[rows[0]]``, which is the wrong window for every channel of the 37.9%
of groups that straddle two.  That only reaches its trace/fallback path, not its
headline value, but any area built on that trace inherits it.
"""
from typing import Optional

import numpy as np
import pandas as pd

try:
    from src.logger import logger
except Exception:  # pragma: no cover
    import logging
    logger = logging.getLogger(__name__)

try:
    from src.ms2_unique_quant import (_as_array, _extract, _window_scans,
                                      channel_unique_masks, jmod_coefficient)
except Exception:  # pragma: no cover - allows standalone import by file path
    from ms2_unique_quant import (_as_array, _extract, _window_scans,
                                  channel_unique_masks, jmod_coefficient)

try:
    from src.ms2_isotope_xtalk import (cascading_channel_fit,
                                       channel_label_from_seq,
                                       normalize_frag_name)
except Exception:  # pragma: no cover - allows standalone import by file path
    from ms2_isotope_xtalk import (cascading_channel_fit,
                                   channel_label_from_seq, normalize_frag_name)

NAME_COL, MZ_COL, LIB_COL = "frag_names", "frag_mz", "frag_int"
FRACLIB_COL, PREC_COL, TC_COL = "frac_lib_int", "untag_prec", "time_channel"
WIN_COL, RT_COL, SPEC_COL, APEX_COL = "window_mz", "rt", "spec_id", "ms1_apex_scan"

ALL_APEX = "coeff_CommonApex"
CU_APEX = "coeff_ChannelUnique_CommonApex"
# The two reported area quantities.  Named after what they are to a user -- the
# MS2 analogue of ``plex_Area`` -- rather than after the apex rule that makes
# them work.
ALL_AREA = "coeff_Area"
CU_AREA = "coeff_ChannelUnique_Area"
COMMON_APEX_COLS = (ALL_APEX, ALL_AREA, CU_APEX, CU_AREA)

# Measured on JD0413 under two different libraries: summing over the common apex
# leaves fold-change accuracy unchanged and costs precision.  Across time
# channels at a designed -3, coeff_ChannelUnique_Area gives -2.95 against -2.94
# for the single-scan value, with IQR 0.82 against 0.79 on the human control;
# coeff_Area's yeast spread is markedly worse (2.81 against 1.70).  They are
# reported because they were asked for, and are excluded from scoring for the
# same reason every other raw abundance is.
#
# Why the plex_Area analogy does not carry: plex_Area sums fitted isotope
# envelopes, where each scan's value rests on a full envelope fit and summing
# averages.  An MS2 coefficient rests on a handful of fragments per scan -- 3-4
# for an R-terminating precursor -- so summing scans adds their errors instead.


def _scan_rt(spectra, scan_num):
    """Retention time of an MS1 scan number, via ``scan_pos``."""
    if scan_num is None or not np.isfinite(scan_num):
        return None
    t = getattr(spectra, "scan_pos", {}).get(int(scan_num))
    if t is None or t[0] != 1:
        return None
    return float(spectra.ms1scans[t[1]].RT)


def add_common_apex_coeff(df: pd.DataFrame, spectra=None, plexDIA: bool = True,
                          ppm: float = 10.0, n_adjacent: int = 1,
                          q_gate: float = 0.01, mz_tol: float = 0.01,
                          row_mask: Optional[np.ndarray] = None,
                          tag=None, isotope_correct: bool = True,
                          n_iso: int = 5,
                          mz_tol_da: float = 0.015,
                          min_siblings: int = 0,
                          weighted_apex: bool = True) -> pd.DataFrame:
    """Add the ``*_CommonApex*`` family.  A no-op without spectra or fragments."""
    need = (NAME_COL, MZ_COL, LIB_COL, PREC_COL, WIN_COL, RT_COL)
    missing = [c for c in need if c not in df.columns]
    if missing:
        logger.warning(f"common-apex MS2 quant skipped: missing {', '.join(missing)}")
        return df
    if spectra is None:
        logger.warning("common-apex MS2 quant skipped: no spectra to refit against")
        return df

    n = len(df)
    fit_me = np.ones(n, dtype=bool) if row_mask is None else np.asarray(row_mask, bool).copy()
    if row_mask is None:
        # ``q_gate=None`` lifts the confidence gate -- used when this runs before
        # scoring, where BestChannel_Qvalue does not exist yet.  Decoys stay out
        # either way: these are report-only abundances, never scoring features,
        # so fitting them would be wasted work.
        if q_gate is not None and "BestChannel_Qvalue" in df.columns:
            fit_me &= pd.to_numeric(df["BestChannel_Qvalue"],
                                    errors="coerce").to_numpy() < q_gate
        if "is_decoy" in df.columns:
            fit_me &= ~df["is_decoy"].to_numpy(dtype=bool)

    names = [_as_array(v, dtype=str) for v in df[NAME_COL]]
    mzs = [_as_array(v) for v in df[MZ_COL]]
    libs = [_as_array(v) for v in df[LIB_COL]]
    fracs = (pd.to_numeric(df[FRACLIB_COL], errors="coerce").to_numpy()
             if FRACLIB_COL in df.columns else np.ones(n))

    # Isotope-crosstalk correction for the channel-unique quantities. Needs the
    # modified sequence (per-channel tag composition) and, for the
    # isolation-window truncation, the precursor m/z and charge -- any of them
    # missing just falls back to the uncorrected fit rather than failing.
    seqs = prec_mzs = prec_zs = iso_cache = win_iso_cache = None
    if isotope_correct and tag is not None:
        if "seq" not in df.columns:
            logger.warning(f"{CU_AREA}: isotope_correct requested but no 'seq' "
                           f"column -- areas stay uncorrected")
            isotope_correct = False
        else:
            seqs = df["seq"].astype(str).to_numpy()
            names = [np.array([normalize_frag_name(x) for x in nm])
                     for nm in names]
            iso_cache, win_iso_cache = {}, {}
            mz_col = next((c for c in ("prec_mz", "mz") if c in df.columns), None)
            z_col = next((c for c in ("z", "charge") if c in df.columns), None)
            if mz_col is not None and z_col is not None:
                prec_mzs = pd.to_numeric(df[mz_col], errors="coerce").to_numpy()
                prec_zs = pd.to_numeric(df[z_col], errors="coerce").to_numpy()
            else:
                logger.warning(
                    f"{CU_AREA}: no precursor m/z + charge -- isolation-window "
                    f"truncation skipped for the areas")
    elif isotope_correct and tag is None:
        logger.warning(f"{CU_AREA}: isotope_correct requested but no tag given")
        isotope_correct = False
    specid = (pd.to_numeric(df[SPEC_COL], errors="coerce").to_numpy()
              if SPEC_COL in df.columns else np.full(n, np.nan))
    scan_pos = getattr(spectra, "scan_pos", {}) or {}
    ms2list = getattr(spectra, "ms2scans", []) or []
    win = pd.to_numeric(df[WIN_COL], errors="coerce").to_numpy()
    rts = pd.to_numeric(df[RT_COL], errors="coerce").to_numpy()
    apex = (pd.to_numeric(df[APEX_COL], errors="coerce").to_numpy()
            if APEX_COL in df.columns else np.full(n, np.nan))

    out = {c: np.full(n, np.nan) for c in COMMON_APEX_COLS}
    n_scan = np.zeros(n, dtype=int)

    key = df[PREC_COL].astype(str)
    if TC_COL in df.columns:
        key = key + "|" + df[TC_COL].astype(str)

    win_cache = {}

    def _scans_for(w):
        wk = round(float(w), 2)
        if wk not in win_cache:
            sc = _window_scans(spectra, w)
            win_cache[wk] = (sc, np.array([s.RT for s in sc]) if sc else np.empty(0))
        return win_cache[wk]

    n_groups = 0
    for _, idx in pd.Series(np.arange(n)).groupby(key.to_numpy()):
        all_rows = idx.to_numpy()
        # Uniqueness is a property of the whole group, including channels the
        # q-value gate will drop -- otherwise a superposed fragment is called
        # unique whenever its partners missed the cutoff.
        all_masks = channel_unique_masks([names[i] for i in all_rows],
                                         [mzs[i] for i in all_rows], mz_tol)
        keep = fit_me[all_rows]
        rows, masks = all_rows[keep], [m for m, k in zip(all_masks, keep) if k]
        if rows.size == 0:
            continue
        mask_of = dict(zip(all_rows.tolist(), all_masks))

        # The one time every channel agrees on.  In practice always the median
        # of the channels' own observed RTs: ms1_apex_scan does not reach this
        # module (not in ms2_prescoring.SCALARS) and is a post-scoring column
        # anyway, so _scan_rt returns None for every row.  See the module
        # docstring.
        t_apex = None
        for i in all_rows:
            t_apex = _scan_rt(spectra, apex[i])
            if t_apex is not None:
                break
        if t_apex is None:
            finite = rts[all_rows][np.isfinite(rts[all_rows])]
            if finite.size == 0:
                continue
            if not weighted_apex:
                t_apex = float(np.median(finite))
            else:
                # Weight each channel's RT by how much signal it actually has.
                # A plain median treats a channel sitting on noise the same as
                # one carrying the peak, and on JD1187 the channels of a group
                # disagree by a median of 0.33 min (p90 1.7) -- so the anchor
                # is only as good as the worst half of them. Measured on that
                # run, precursors whose channels agree on RT quantify at 0.719
                # compression against 0.263 for those that do not, so the
                # anchor is the dominant term, not a detail.
                wts, rr = [], []
                for i in all_rows:
                    if not np.isfinite(rts[i]):
                        continue
                    v = 0.0
                    pos = (scan_pos.get(int(specid[i]))
                           if np.isfinite(specid[i]) else None)
                    if (pos is not None and pos[0] == 2
                            and pos[1] < len(ms2list)):
                        lib_i, mz_i = libs[i], mzs[i]
                        m_i = mask_of.get(i)
                        if (lib_i.size and lib_i.size == mz_i.size
                                and m_i is not None and m_i.size == lib_i.size
                                and m_i.any()):
                            o = _extract(ms2list[pos[1]], mz_i, ppm)
                            c = jmod_coefficient(lib_i, o, fracs[i], mask=m_i)
                            if np.isfinite(c) and c > 0:
                                v = float(c)
                    wts.append(v)
                    rr.append(float(rts[i]))
                if rr and sum(wts) > 0:
                    o = np.argsort(rr)
                    r_s = np.asarray(rr)[o]
                    w_s = np.asarray(wts)[o]
                    cw = np.cumsum(w_s) / w_s.sum()
                    t_apex = float(r_s[int(np.searchsorted(cw, 0.5))])
                else:
                    t_apex = float(np.median(finite))
        n_groups += 1

        # Channel-unique quant here is subject to exactly the isotope-envelope
        # crosstalk ms2_isotope_xtalk corrects for the single-scan value, so the
        # reported areas need the same treatment or they silently disagree with
        # coeff_ChannelUnique.  The cascade runs PER SCAN: each summed scan is a
        # separate spectrum with its own contamination, and a contaminating
        # channel's abundance at scan p-1 is not its abundance at p.  So the
        # loops invert -- scan offset outside, channels in ascending tag mass
        # inside -- and every scan builds its own known_coeffs.
        order = list(zip(rows, masks))
        if isotope_correct and tag is not None and seqs is not None:
            try:
                order = sorted(
                    order,
                    key=lambda rm: tag.mass_dict[
                        f"{tag.name}-{channel_label_from_seq(seqs[rm[0]], tag)}"])
            except Exception:
                order = list(zip(rows, masks))   # unparseable: keep input order

        geom = {}
        srt_of = {}
        for i, m in order:
            lib, mz = libs[i], mzs[i]
            if lib.size != m.size or mz.size != m.size or not m.any():
                continue
            scans, srt = _scans_for(win[i])       # this row's OWN window
            if not scans:
                continue
            p = int(np.argmin(np.abs(srt - t_apex)))
            lo, hi = max(0, p - n_adjacent), min(len(scans), p + n_adjacent + 1)
            n_scan[i] = hi - lo
            geom[i] = (scans, p, lo, hi)
            srt_of[i] = srt

        s_all = {i: 0.0 for i in geom}
        s_cu = {i: 0.0 for i in geom}
        for d in range(-n_adjacent, n_adjacent + 1):
            known = {}                            # per-scan, never shared
            for i, m in order:
                if i not in geom:
                    continue
                scans, p, lo, hi = geom[i]
                k = p + d
                if k < lo or k >= hi:
                    continue
                lib, mz = libs[i], mzs[i]
                o = _extract(scans[k], mz, ppm)
                va = jmod_coefficient(lib, o, fracs[i], mask=None)
                if isotope_correct and tag is not None and seqs is not None:
                    _w = getattr(scans[k], "ms1window", None)
                    vu = cascading_channel_fit(
                        i, all_rows.tolist(), seqs, names, mzs, libs, mask_of,
                        fracs, known,
                        lambda mzarr, sc=scans[k]: _extract(sc, mzarr, ppm),
                        tag, n_iso=n_iso, mz_tol_da=mz_tol_da, cache=iso_cache,
                        prec_mz=(prec_mzs[i] if prec_mzs is not None else None),
                        prec_z=(prec_zs[i] if prec_zs is not None else None),
                        window_hi=(float(_w[1]) if _w is not None and len(_w) > 1
                                   else None),
                        window_lo=(float(_w[0]) if _w is not None and len(_w) > 0
                                   else None),
                        win_cache=win_iso_cache,
                        min_siblings=min_siblings,
)
                    known[i] = vu
                else:
                    vu = jmod_coefficient(lib, o, fracs[i], mask=m)
                if d == 0:
                    out[ALL_APEX][i] = va
                    out[CU_APEX][i] = vu
                s_all[i] += va if np.isfinite(va) else 0.0
                s_cu[i] += vu if np.isfinite(vu) else 0.0
        for i in geom:
            out[ALL_AREA][i] = s_all[i]
            out[CU_AREA][i] = s_cu[i]

    for c in COMMON_APEX_COLS:
        df[c] = out[c]
    df["CommonApex_n_scans"] = n_scan
    ok = np.isfinite(out[CU_APEX])
    logger.info(
        f"common-apex MS2 quant: {int(ok.sum()):,} of {int(fit_me.sum()):,} rows "
        f"across {n_groups:,} plex groups; window +/-{n_adjacent} scans "
        f"(median {int(np.median(n_scan[n_scan > 0])) if (n_scan > 0).any() else 0} summed)")
    return df
