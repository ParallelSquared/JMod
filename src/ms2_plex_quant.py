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

"""MS2 quantification of a plex group: coeff_plex.

The main search's ``coeff`` is fitted per channel at that channel's own best
MS2 scan.  For a faint channel that scan is the one where the noise happened
to be highest, so faint channels read high and ratios compress.  This refits
the whole plex group jointly, the way the MS1 quant does, at the group's
voted apex:

* rows are the channels' fragment isotope peaks (library M+0..M+2, and
  M+3..M+5 only where they fall on another channel's peak -- a lighter
  channel's M+4 sits on the next channel's monoisotope under a 4 Da tag step)
  and peaks of different channels within tolerance share a row, so crosstalk
  is modelled and every ion is counted once;
* fragments shared by all channels (untagged y-ions) stay in as sum rows;
* the on-grid chemical background around each fragment (other peptides'
  fragment ladders on the same mass-defect grid) is estimated from the grid
  positions +-10 isotope spacings away that hold no theoretical peak, and
  subtracted before the fit;
* fragment ions are gated to the precursor's mobility (per-ion 1/K0 within
  the fitted fragment spread), which removes most of that background;
* rows are weighted by their expected signal share, so background-dominated
  rows carry little weight;
* each channel is read from its own isolation window (a 5plex spans ~8 Th,
  so about 1 group in 4 straddles two windows), and a window is fitted with
  only the channels whose precursor it isolates;
* the fit is repeated over the 5 scans of the window nearest the group's
  consensus apex -- the MS1 quant's vote over the channels' apex scans
  (ms1_apex_scan), the same anchor plex_Area uses -- and combined with weights
  from the window's background-subtracted total, i.e. integrated over the
  elution.

On the 200pg 5plex (run 9642) this took the yeast ratio slope of the MS2
quant from 0.29 (coeff) to 0.55 unfiltered and 0.72 with the top half by
MS2_quant_quality.  The spread of the MS2 ratios stays ~2x the MS1 quant's:
a faint channel's fragments hold a few ions each at this load.

Columns: ``coeff_plex``, ``ms2_bg`` (background per row at the reference
scan), ``ms2_snr``, ``ms2_nfrag`` (fragments with signal above background)
and, on the filtered IDs, ``MS2_quant_quality``.
"""

import re
from concurrent.futures import ThreadPoolExecutor
from functools import reduce

import numpy as np
import pandas as pd
from scipy import optimize

from src.logger import logger
from src.utils.frag_encoding import decode_frag_names
from src.iso_functions import fragment_seq, get_seq_comp
from src.isotope_distribution import isotopic_variants

ISOTOPE_SPACING_DA = 1.00235   # averagine peptide isotope spacing, as ms1_cor_channels / JrMod
N_BG_SPACINGS = 10       # background windows: grid positions +-N spacings from each fragment
N_SCANS = 5              # scans of each window integrated around the reference scan
N_ISO = 6                # isotope orders computed per fragment (M+0..M+5)
MIN_REL = 0.005


def _frag_charge(name):
    """Charge of a fragment named like "y5_2"."""
    return int(name.split("_")[1])


def fragment_isotope_pattern(mod_seq, base_ion_type, tag, n_iso=5, cache=None):
    """Theoretical isotope envelope of one fragment, one tagged channel.

    Parameters
    ----------
    mod_seq : str
        Full modified peptide sequence, tag annotations included, e.g.
        ``"E(PSMtag_9plex-4)MAAAPPGR"``.
    base_ion_type : str
        Fragment name without any "_iso" suffix, e.g. ``"b3_1"``.
    tag : massTag
        Needed for ``tag.name`` and ``tag.channel_comp``.
    n_iso : int
        Number of isotope orders to compute (0 = monoisotopic only).
    cache : dict or None
        Keyed by (composition string, charge, n_iso); isotope patterns are a
        pure function of elemental composition, so this is safe to share
        across an entire run.

    Returns
    -------
    mz_offset : np.ndarray, shape (n_iso,)
        Isotope order i's m/z minus the monoisotopic m/z (offset, not
        absolute, so the caller adds it to whatever m/z the library/report
        actually recorded -- keeps this consistent under small calibration
        differences between the two).
    rel_int : np.ndarray, shape (n_iso,)
        Isotope order i's intensity relative to order 0 (rel_int[0] == 1).
    """
    split_frag_seq, frag_info = fragment_seq(mod_seq, base_ion_type)
    frag_type, _frag_idx, loss, frag_z = frag_info
    frag_z = int(frag_z)
    frag_comp = get_seq_comp(split_frag_seq, frag_type,
                             neutral_loss=(loss or None))

    tags = [t for aa in split_frag_seq
            for t in re.findall(f"\\(({tag.name}.*?)\\)", aa)]
    if tag.channel_comp is not None and tags:
        tag_comp = reduce(lambda x, y: x + y,
                          [tag.channel_comp[re.findall(f"{tag.name}-(\\d+)", t)[0]]
                           for t in tags])
        frag_comp = frag_comp + tag_comp

    key = None
    if cache is not None:
        key = (str(sorted(frag_comp.items())), frag_z, n_iso)
        hit = cache.get(key)
        if hit is not None:
            return hit

    isotopes = isotopic_variants(frag_comp, npeaks=n_iso, charge=frag_z)
    mono = isotopes[0]
    mz_offset = np.array([p.mz - mono.mz for p in isotopes])
    rel_int = np.array([p.intensity / mono.intensity for p in isotopes])

    if cache is not None:
        cache[key] = (mz_offset, rel_int)
    return mz_offset, rel_int


def build_group(rows, tag, ppm, iso_cache):
    """Rows for one plex group's MS2 fit.

    Returns (M, meta) or None: M is rows x channels of theoretical intensity;
    meta holds the row m/z members, which channels each row belongs to, the
    fragment keys per row, the background windows per (fragment, channel)
    and the row index of each (fragment, channel).
    """
    K = len(rows)
    peaks = []                       # (mz, channel, intensity, key, isotope order)
    key_mz = {}
    for j in range(K):
        seq = rows["seq"].iat[j]
        # frag_names are still the packed int32 codes here: the pipeline
        # decodes them only when it writes the filtered IDs
        names = decode_frag_names(np.asarray(rows["frag_names"].iat[j], dtype=np.int32))
        mzs = np.asarray(rows["frag_mz"].iat[j], dtype=float)
        libs = np.asarray(rows["frag_int"].iat[j], dtype=float)
        for name, mz, lib in zip(names, mzs, libs):
            if "_iso" in name or not lib > 0:
                continue
            off, rel = fragment_isotope_pattern(seq, name, tag, n_iso=N_ISO, cache=iso_cache)
            for order, (o, r) in enumerate(zip(off, rel)):
                if r >= MIN_REL:
                    peaks.append((float(mz + o), j, float(lib * r), name, order))
            key_mz.setdefault(name, {})[j] = float(mz)
    core = [p for p in peaks if p[4] <= 2]
    core_mz = np.array(sorted(p[0] for p in core))
    # higher isotopes only where they overlap another channel's peaks
    extra = [p for p in peaks if p[4] > 2
             and np.min(np.abs(core_mz - p[0])) <= 1.5 * ppm * p[0]]
    peaks = sorted(core + extra, key=lambda p: p[0])
    theo = np.array([p[0] for p in peaks])
    unique = {name: (max(v.values()) - min(v.values()) > 0.01) if len(v) > 1 else True
              for name, v in key_mz.items()}
    bg_windows = {}
    for name, v in key_mz.items():
        sp = ISOTOPE_SPACING_DA / _frag_charge(name)
        for j, mz in v.items():
            cand = [mz + s * sp for s in range(-N_BG_SPACINGS, N_BG_SPACINGS + 1) if s != 0]
            bg_windows[(name, j)] = [q for q in cand if np.min(np.abs(theo - q)) > 2 * ppm * q]
    groups, cur = [], [peaks[0]]
    for p in peaks[1:]:
        if p[0] - cur[-1][0] <= ppm * p[0]:
            cur.append(p)
        else:
            groups.append(cur)
            cur = [p]
    groups.append(cur)
    n = len(groups)
    M = np.zeros((n, K))
    members = [[m[0] for m in g] for g in groups]
    row_keys = [set() for _ in range(n)]
    row_channels = [set() for _ in range(n)]
    for r, g in enumerate(groups):
        for mz, j, intensity, name, _order in g:
            M[r, j] += intensity
            row_keys[r].add((name, j))
            row_channels[r].add(j)
    row_unique = np.array([len(row_channels[r]) == 1
                           and all(unique.get(name, True) for name, _ in row_keys[r])
                           for r in range(n)])
    all_mz = theo.tolist() + [q for v in bg_windows.values() for q in v]
    # Row windows as intervals: peaks in a row are within tolerance of each
    # other, so their windows overlap and their union is one interval; adjacent
    # rows that still overlap are split at the midpoint so every ion is counted
    # once.  Background windows are intervals too, grouped by (fragment,
    # channel) key; a row's background is the mean over its keys of the
    # median over that key's windows.
    row_lo = np.array([min(m) * (1 - ppm) for m in members])
    row_hi = np.array([max(m) * (1 + ppm) for m in members])
    for r in range(1, n):
        if row_lo[r] < row_hi[r - 1]:
            mid = 0.5 * (row_lo[r] + row_hi[r - 1])
            row_hi[r - 1] = mid
            row_lo[r] = mid
    keys = sorted(bg_windows)
    key_index = {k: i for i, k in enumerate(keys)}
    width = max((len(v) for v in bg_windows.values()), default=1) or 1
    bg_lo = np.full((len(keys), width), np.nan)
    bg_hi = np.full((len(keys), width), np.nan)
    for k, wins in bg_windows.items():
        for c, q in enumerate(wins):
            bg_lo[key_index[k], c] = q * (1 - ppm)
            bg_hi[key_index[k], c] = q * (1 + ppm)
    key_to_row = np.zeros((n, len(keys)))
    for r in range(n):
        for k in row_keys[r]:
            if k in key_index:
                key_to_row[r, key_index[k]] = 1.0
    key_to_row /= np.maximum(key_to_row.sum(1, keepdims=True), 1.0)
    meta = dict(members=members, row_keys=row_keys, row_channels=row_channels, row_unique=row_unique,
                bg_windows=bg_windows, lo=min(all_mz) - 0.1, hi=max(all_mz) + 0.1,
                row_lo=row_lo, row_hi=row_hi, bg_lo=bg_lo, bg_hi=bg_hi, key_to_row=key_to_row)
    return M, meta


def _ions(spec, meta, prec_im, im_tol):
    a, b = np.searchsorted(spec.mz, [meta["lo"], meta["hi"]])
    mz, it = spec.mz[a:b], spec.intens[a:b]
    mob = getattr(spec, "mobility", None)
    if mob is not None and im_tol and im_tol > 0 and np.isfinite(prec_im):
        keep = np.abs(mob[a:b] - prec_im) <= im_tol
        mz, it = mz[keep], it[keep]
    return mz, it


def observe(spec, meta, ppm, prec_im, im_tol):
    """(row intensities, background per row) of one MS2 spectrum."""
    mz, it = _ions(spec, meta, prec_im, im_tol)
    csum = np.r_[0.0, np.cumsum(it, dtype=np.float64)]
    def window_sums(lo, hi):
        a = np.searchsorted(mz, lo)
        b = np.searchsorted(mz, hi)
        return csum[b] - csum[a]
    y = window_sums(meta["row_lo"], meta["row_hi"])
    bg_lo, bg_hi = meta["bg_lo"], meta["bg_hi"]
    if bg_lo.size:
        w = window_sums(np.nan_to_num(bg_lo, nan=0.0), np.nan_to_num(bg_hi, nan=0.0))
        w = np.where(np.isfinite(bg_lo), w, np.nan)
        with np.errstate(all="ignore"):
            key_med = np.nanmedian(w, axis=1)
        key_med = np.nan_to_num(key_med, nan=0.0)
        bg = meta["key_to_row"] @ key_med
    else:
        bg = np.zeros(y.size)
    return y, bg


def fit_scan(M, y, bg):
    """All channels at one scan: plain NNLS for the scale, then NNLS weighted by
    each row's expected signal share.  Returns (coefficients, signal total)."""
    yy = np.clip(y - bg, 0, None)
    if not yy.any():
        return np.zeros(M.shape[1]), 0.0
    c0, _ = optimize.nnls(M, yy)
    scale = max(float(c0.sum()), 1.0)
    expected = M.sum(1) * scale
    w = np.sqrt(expected / (expected + bg + 1.0))
    c, _ = optimize.nnls(M * w[:, None], yy * w)
    return c, float(yy.sum())


def quantify_group(rows, spectra, tag, ppm, im_tol, iso_cache):
    """coeff_plex, ms2_bg, ms2_snr, ms2_nfrag for the rows of one plex group."""
    K = len(rows)
    out = np.full((K, 4), np.nan)
    M, meta = build_group(rows, tag, ppm, iso_cache)
    prec_im = float(np.nanmedian(rows["prec_im"])) if np.isfinite(rows["prec_im"]).any() else np.nan
    scan_pos = spectra.scan_pos
    pos = [scan_pos[int(sid)][1] for sid in rows["spec_id"]]
    ms2 = spectra.ms2scans
    # Anchor: the plex group's consensus apex -- the MS1 quant's Gaussian vote
    # over the channels' apex scans (ms1_cor_channels._quant_group), written to
    # every row as ms1_apex_scan.  The channels co-elute, so the group is read
    # at one time.
    # (the middle element, not the median: with --apex_jitter or --free_apex the
    # channels' apexes can differ, and the median of an even count can fall
    # between two MS1 scans)
    apex = int(np.sort(rows["ms1_apex_scan"].to_numpy(dtype=np.int64))[len(rows) // 2])
    ref_rt = spectra.ms1scans[scan_pos[apex][1]].RT
    # The channels' isolation windows / IM bands: each gets its own 5 scans
    # nearest the reference scan.  A window is fitted with only the channels
    # whose precursor it isolates (a 5plex spans ~8 Th, so ~1 in 4 groups
    # straddles two windows) and only the rows those channels have: a channel
    # that is not isolated there must not take a share of the untagged
    # fragments.
    win_mz = [round(float(ms2[p].prec_mz), 3) for p in pos]
    results = {}
    for j in range(K):
        s = ms2[pos[j]]
        wkey = (win_mz[j], round(float(s.im_lo), 5))
        if wkey in results:
            continue
        active = np.array([win_mz[i] == wkey[0] for i in range(K)])
        use = np.any(M[:, active] > 0, axis=1)
        Ma = M[use][:, active]
        cand = spectra.ms2_window_scans(wkey[0], s.im_lo)
        near = cand[np.argsort(np.abs(np.array([ms2[q].RT for q in cand]) - ref_rt))[:N_SCANS]]
        fits, totals, ref_obs = [], [], None
        for q in near:
            y, bg = observe(ms2[q], meta, ppm, prec_im, im_tol)
            ca, total = fit_scan(Ma, y[use], bg[use])
            c = np.zeros(K)
            c[active] = ca
            fits.append(c)
            totals.append(total)
            if ref_obs is None or abs(ms2[q].RT - ref_rt) < abs(ref_obs[2] - ref_rt):
                ref_obs = (y, bg, ms2[q].RT)
        totals = np.array(totals)
        # TODO: weight the scans by a fixed Gaussian of the fitted elution width
        # around the voted apex (as the MS1 quant integrates a fixed window)
        # instead of by each scan's measured total intensity.
        if totals.sum() > 0:
            g = totals / totals.sum()
            results[wkey] = (sum(c * gi for c, gi in zip(fits, g)), ref_obs)
    for j in range(K):
        s = ms2[pos[j]]
        res = results.get((round(float(s.prec_mz), 3), round(float(s.im_lo), 5)))
        if res is None:
            continue
        c, (y, bg, _rt) = res
        own_unique = (M[:, j] > 0) & meta["row_unique"]
        signal = float(np.clip(y[own_unique] - bg[own_unique], 0, None).sum())
        background = float(bg[own_unique].sum())
        keys_with_signal = {name for r in np.flatnonzero(own_unique) if y[r] > bg[r]
                            for name, _ in meta["row_keys"][r]}
        # TODO: ms2_snr is signal over background, and the background is 0 in
        # ~40% of rows (empty fragment windows), where it becomes the raw signal
        # and MS2_quant_quality ranks those rows by intensity.  Needs a noise
        # floor measured from the background windows.
        out[j] = (c[j], float(np.mean(bg[M[:, j] > 0])) if (M[:, j] > 0).any() else 0.0,
                  signal / max(background, 1.0), len(keys_with_signal))
    return out


def add_plex_coeff(df, spectra, tag, ppm, im_tol, q_gate=0.01, n_threads=1):
    """Add coeff_plex, ms2_bg, ms2_snr and ms2_nfrag to *df* for the non-decoy
    rows with BestChannel_Qvalue < *q_gate*.  Rows are grouped by untag_prec
    (and time_channel when present)."""
    fit_me = (~df["is_decoy"].to_numpy(dtype=bool)) & (pd.to_numeric(df["BestChannel_Qvalue"], errors="coerce").to_numpy() < q_gate)
    idx = np.flatnonzero(fit_me)
    sub = df.iloc[idx]
    keys = sub["untag_prec"].astype(str)
    if "time_channel" in sub.columns:
        keys = keys + "|" + sub["time_channel"].astype(str)
    groups = [g.to_numpy() for _, g in pd.Series(np.arange(len(sub))).groupby(keys.to_numpy())]
    spectra.index_ms2_windows()
    iso_cache = {}
    out = np.full((len(df), 4), np.nan)

    # Every group's rows are cut here, before the pool: slicing one shared
    # DataFrame from many threads races in pandas' copy-on-write reference
    # tracking on free-threaded Python (a segfault in BlockValuesRefs).
    frames = [sub.iloc[g] for g in groups]

    def work(g, rows):
        out[idx[g]] = quantify_group(rows, spectra, tag, ppm, im_tol, iso_cache)

    with ThreadPoolExecutor(max_workers=max(1, int(n_threads))) as pool:
        list(pool.map(work, groups, frames))
    df["coeff_plex"], df["ms2_bg"], df["ms2_snr"], df["ms2_nfrag"] = out.T
    ok = np.isfinite(out[:, 0])
    logger.info(f"coeff_plex: {int(ok.sum()):,} of {len(idx):,} rows in {len(groups):,} plex groups; "
                f"median {int(np.nanmedian(out[ok, 3])) if ok.any() else 0} fragments above background")
    return df


def ms2_quant_quality(ids):
    """MS2_quant_quality, 0-1, from MS2 evidence only: the geometric mean of the
    within-run percentile ranks of the fragment-trace coherence
    (top3_frag_mean_corr), ms2_snr and ms2_nfrag.  On run 9642 the top half by
    this score took coeff_plex from slope 0.57 / human IQR 1.94 to 0.73 / 1.32."""
    def pct(values):
        return values.rank(method="average") / values.count()
    parts = [pct(ids[c]).fillna(0.0) for c in ("top3_frag_mean_corr", "ms2_snr", "ms2_nfrag")]
    return np.power(np.prod(parts, axis=0), 1.0 / len(parts))
