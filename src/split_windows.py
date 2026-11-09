#  Split-window search (--split_win).
#
#  A precursor's isotope envelope can straddle two adjacent DIA isolation
#  windows: with 2 Th windows the window holding M0 often isolates only M0 (or
#  M0+M1), and the rest of the envelope is fragmented in the next window up,
#  in the same cycle.  JMod normally searches a precursor only in the window
#  holding its M0, with a fragment-isotope template that assumes every
#  precursor isotope was isolated.
#
#  With --split_win, each MS2 scan (window A) is fitted jointly with the scan
#  of the window directly above it in the same cycle (window B):
#
#    * the two spectra are combined into one "joint spectrum": B's peaks have
#      their m/z multiplied by SCALE, which keeps ppm matching exact and puts
#      them far above any real fragment m/z, so A and B peaks never match each
#      other;
#    * a precursor with M0 in A gets one column whose template covers both
#      scans: in A the fragment isotopes allowed by the precursor isotopes A
#      isolated, in B those allowed by the isotopes B isolated (fragment
#      isotope k of fragment f, when precursor isotopes j in [a, b] are
#      isolated: L_fk * P(complement heavy count in [a-k, b-k]); L_fk is the
#      library's untruncated isotope pattern).  Its B part is only included
#      when B holds >= --split_min_frac of the precursor's isotope envelope.
#      So the precursor's coefficient is one value fitted to both scans;
#    * competitors are added so the joint fit is not biased: precursors with
#      M0 in the window below A whose upper part (>= split_min_frac) lies in A
#      ("satellites"), and the precursors with M0 in B (their own window-B
#      template).  Only the precursors with M0 in A are reported, so the
#      output keeps one row per precursor per scan, as before.
#
#  The isotope maths uses exact elemental compositions (C, H, N, O, S natural
#  abundances); other elements are treated as monoisotopic.

import re
import time
import numpy as np
from numba import njit

from src.logger import logger
from src.utils.parse_peptides import parse_peptide

ISO_DIFF = 1.0033548
NB = 10                 # isotopes tracked for precursor / complement envelopes
SCALE = 1000.0          # window-B m/z multiplier in the joint spectrum
UNSCALE_ABOVE = 2.0e4   # any m/z above this in a joint spectrum is a scaled window-B m/z
B_FLAG = np.int32(1 << 21)   # spare frag-code bit: peak predicted in window B

KIND_A = 0          # M0 in A, template in A only
KIND_AB = 1         # M0 in A, template in A and B (joint coefficient)
KIND_SAT = 2        # M0 in the window below A, its upper part in A (competitor)
KIND_UP = 3         # M0 in B, its own template in B (competitor)

_ABUND = {'C': (0.9893, 0.0107), 'H': (0.999885, 0.000115), 'N': (0.99636, 0.00364),
          'O': (0.99757, 0.00038, 0.00205), 'S': (0.9499, 0.0075, 0.0425, 0.0, 0.0001)}
_ELEMS = ('C', 'H', 'N', 'O', 'S')
_NTERM = re.compile(r'^((?:\([^)]*\))+)([A-Z])(.*)$')
_MAXN = (800, 1400, 300, 300, 60)


def _power_tables():
    tabs = []
    for e, mx in zip(_ELEMS, _MAXN):
        p = np.array(_ABUND[e])
        t = np.zeros((mx + 1, NB))
        t[0, 0] = 1.0
        for n in range(1, mx + 1):
            t[n] = np.convolve(t[n - 1], p)[:NB]
        tabs.append(t)
    return tabs


_POW = _power_tables()


@njit(nogil=True, cache=False)
def _env(c, pC, pH, pN, pO, pS, out):
    """Isotope distribution (truncated at NB, renormalised) of composition c[5] into out[NB]."""
    nb = out.shape[0]
    v = np.zeros(nb)
    w = np.zeros(nb)
    tmp = np.zeros(nb)
    n0 = min(max(c[0], 0), pC.shape[0] - 1)
    for k in range(nb):
        v[k] = pC[n0, k]
    for e in range(1, 5):
        if e == 1:
            n = min(max(c[1], 0), pH.shape[0] - 1)
            for k in range(nb):
                w[k] = pH[n, k]
        elif e == 2:
            n = min(max(c[2], 0), pN.shape[0] - 1)
            for k in range(nb):
                w[k] = pN[n, k]
        elif e == 3:
            n = min(max(c[3], 0), pO.shape[0] - 1)
            for k in range(nb):
                w[k] = pO[n, k]
        else:
            n = min(max(c[4], 0), pS.shape[0] - 1)
            for k in range(nb):
                w[k] = pS[n, k]
        for k in range(nb):
            s = 0.0
            for i in range(k + 1):
                s += v[i] * w[k - i]
            tmp[k] = s
        for k in range(nb):
            v[k] = tmp[k]
    tot = 0.0
    for k in range(nb):
        tot += v[k]
    for k in range(nb):
        out[k] = v[k] / tot if tot > 0 else 0.0


@njit(nogil=True, cache=False)
def _window_factors(comp_c, iso_k, ent, jA_hi, jB_lo, jB_hi, lib_int, pC, pH, pN, pO, pS, int_A, int_B, dst):
    """Per library peak: template intensity in the M0 window (isotopes 0..jA_hi) and in the
    upper window (isotopes jB_lo..jB_hi), = lib_int * P(complement heavy count in [a-k, b-k])."""
    env = np.zeros(NB)
    cdf = np.zeros(NB)
    for i in range(comp_c.shape[0]):
        _env(comp_c[i], pC, pH, pN, pO, pS, env)
        s = 0.0
        for k in range(NB):
            s += env[k]
            cdf[k] = s
        k = iso_k[i]
        e = ent[i]
        # window A: precursor isotopes 0..jA_hi -> complement in [0, jA_hi - k]
        hi = jA_hi[e] - k
        fa = 0.0
        if hi >= 0:
            fa = cdf[min(hi, NB - 1)]
        int_A[dst[i]] = lib_int[i] * fa
        # window B: precursor isotopes jB_lo..jB_hi -> complement in [jB_lo - k, jB_hi - k]
        fb = 0.0
        if jB_lo[e] <= jB_hi[e]:
            hi = jB_hi[e] - k
            lo = jB_lo[e] - k
            if hi >= 0:
                up = cdf[min(hi, NB - 1)]
                if lo - 1 < 0:
                    dn = 0.0
                elif lo - 1 >= NB:
                    dn = 1.0
                else:
                    dn = cdf[lo - 1]
                fb = up - dn
                if fb < 0.0:
                    fb = 0.0
        int_B[dst[i]] = lib_int[i] * fb


@njit(nogil=True, cache=False)
def _prec_fracs(comp_p, jA_hi, jB_lo, jB_hi, pC, pH, pN, pO, pS, f_A, f_B):
    env = np.zeros(NB)
    for i in range(comp_p.shape[0]):
        _env(comp_p[i], pC, pH, pN, pO, pS, env)
        a = 0.0
        for j in range(0, min(jA_hi[i], NB - 1) + 1):
            a += env[j]
        b = 0.0
        if jB_lo[i] <= jB_hi[i]:
            for j in range(jB_lo[i], min(jB_hi[i], NB - 1) + 1):
                b += env[j]
        f_A[i] = a
        f_B[i] = b


class SplitWindowTemplates:
    """Per-run split-window data: per library peak the window-A and window-B template
    intensities, per entry the isolated envelope fractions and window indices."""

    def __init__(self, int_A, int_B, f_A, f_B, has_B, win_A, win_B, grid_lo, grid_hi, codes, min_frac):
        self.int_A = int_A
        self.int_B = int_B
        self.f_A = f_A
        self.f_B = f_B
        self.has_B = has_B
        self.win_A = win_A
        self.win_B = win_B
        self.grid_lo = grid_lo
        self.grid_hi = grid_hi
        self.codes = codes
        self.min_frac = min_frac
        self.upper = {}          # MS2 scan_num -> spectrum of the window directly above, same cycle
        self.scan_grid = {}      # MS2 scan_num -> index of its window in the grid

    def grid_index(self, lo, hi):
        i = int(np.searchsorted(self.grid_lo, lo - 0.01))
        if i < len(self.grid_lo) and abs(self.grid_lo[i] - lo) < 0.01 and abs(self.grid_hi[i] - hi) < 0.01:
            return i
        return -1


def window_grid(ms2scans):
    """Unique isolation windows (lo, hi), sorted by lo.  Returns None when windows overlap."""
    w = np.unique(np.round(np.array([s.ms1window for s in ms2scans], dtype=np.float64), 3), axis=0)
    w = w[np.argsort(w[:, 0])]
    if len(w) > 1 and np.any(w[1:, 0] < w[:-1, 1] - 0.05):
        return None
    return w


def _residue_vectors(mod_seqs, tag):
    """Per entry: (n_res, 5) per-residue C,H,N,O,S counts with mods/tags folded in."""
    from src.iso_functions import _residue_comp, _hashable_channel_comps
    tag_name = tag.name if tag is not None else None
    cc = _hashable_channel_comps(tag)
    cache = {}

    def rv(res):
        v = cache.get(res)
        if v is None:
            r = res
            m = _NTERM.match(r)
            if m:   # N-terminal mod written before the first residue: fold it onto that residue
                r = m.group(2) + m.group(1) + m.group(3)
            comp = _residue_comp(r, tag_name, cc)
            v = np.array([comp.get(e, 0) for e in _ELEMS], dtype=np.int32)
            cache[res] = v
        return v

    for ms in mod_seqs:
        yield np.array([rv(r) for r in parse_peptide(ms)], dtype=np.int32)


def build_templates(library, rt_mz, ms2scans, ms1scans, tag, min_frac):
    """Build the split-window templates for this run.  Returns None (split search off) when
    the run's windows overlap or the library has no isotope expansion."""
    t0 = time.time()
    grid = window_grid(ms2scans)
    if grid is None:
        logger.warning("--split_win: isolation windows overlap; split-window search is only "
                       "implemented for non-overlapping windows. Running without it.")
        return None
    if getattr(library, 'spectrum_perm', None) is not None or library.frag_names_data is None:
        logger.warning("--split_win needs --iso (isotope-expanded library). Running without it.")
        return None
    lo, hi = grid[:, 0], grid[:, 1]
    n = len(library.mod_seq)
    mz0 = np.asarray(rt_mz[:n, 1], dtype=np.float64)
    z = np.asarray(library.prec_z, dtype=np.float64)[:n]
    # window holding M0 (non-overlapping grid)
    wa = np.searchsorted(lo, mz0, side='right') - 1
    ok = (wa >= 0) & (mz0 < hi[np.clip(wa, 0, len(hi) - 1)])
    wa = np.where(ok, wa, -1)
    # window directly above (adjacent)
    wb = np.full(n, -1)
    nxt = np.clip(wa + 1, 0, len(lo) - 1)
    adj = ok & (wa + 1 < len(lo)) & (np.abs(lo[nxt] - hi[np.clip(wa, 0, len(hi) - 1)]) < 0.05)
    wb[adj] = nxt[adj]
    sp = ISO_DIFF / z
    jA_hi = np.where(ok, np.floor((hi[np.clip(wa, 0, len(hi) - 1)] - mz0) / sp), NB - 1)
    jA_hi = np.clip(jA_hi, 0, NB - 1).astype(np.int64)
    jB_lo = np.where(wb >= 0, jA_hi + 1, NB).astype(np.int64)
    jB_hi = np.where(wb >= 0, np.floor((hi[np.clip(wb, 0, len(hi) - 1)] - mz0) / sp), -1)
    jB_hi = np.clip(jB_hi, -1, NB - 1).astype(np.int64)

    so, sl = library.spectrum_offsets, library.spectrum_lengths
    lib_int_all = np.asarray(library.spectrum_int, dtype=np.float64)
    codes_all = np.asarray(library.frag_names_data, dtype=np.int32)
    int_A = np.zeros(len(lib_int_all), dtype=np.float32)
    int_B = np.zeros(len(lib_int_all), dtype=np.float32)
    comp_p = np.zeros((n, 5), dtype=np.int64)

    ion = (codes_all >> 0) & 0x7
    idx = (codes_all >> 3) & 0xFF
    loss = (codes_all >> 11) & 0x7
    isok = ((codes_all >> 17) & 0xF).astype(np.int64)
    LOSS = {0: np.zeros(5, np.int32), 1: np.array([0, 2, 0, 1, 0], np.int32), 2: np.array([0, 3, 1, 0, 0], np.int32),
            3: np.array([0, 3, 0, 4, 0], np.int32)}
    loss_vec = np.stack([LOSS.get(i, LOSS[0]) for i in range(8)])
    H2O = np.array([0, 2, 0, 1, 0], np.int32)

    CH = 20000
    buf_c, buf_k, buf_e, buf_i, buf_d = [], [], [], [], []
    n_buf = 0
    n_unk = 0

    def flush():
        if not buf_c:
            return
        _window_factors(np.vstack(buf_c).astype(np.int64), np.concatenate(buf_k), np.concatenate(buf_e),
                        jA_hi, jB_lo, jB_hi, np.concatenate(buf_i), *_POW, int_A, int_B, np.concatenate(buf_d))
        buf_c.clear(); buf_k.clear(); buf_e.clear(); buf_i.clear(); buf_d.clear()

    for e, R in enumerate(_residue_vectors(library.mod_seq[:n], tag)):
        prec = R.sum(0) + H2O
        comp_p[e] = prec
        s, L = int(so[e]), int(sl[e])
        if L == 0:
            continue
        sl_ = slice(s, s + L)
        it, ix = ion[sl_], idx[sl_].astype(np.int64)
        nres = len(R)
        cumb = np.cumsum(R, 0)
        cumy = np.cumsum(R[::-1], 0) + H2O
        ixc = np.clip(ix - 1, 0, nres - 1)
        frag = np.where((it == 0)[:, None], cumb[ixc], np.where((it == 1)[:, None], cumy[ixc], cumb[ixc]))
        n_unk += int(((it != 0) & (it != 1)).sum())
        frag = frag - loss_vec[loss[sl_]]
        buf_c.append(prec[None, :] - frag)
        buf_k.append(isok[sl_])
        buf_e.append(np.full(L, e, dtype=np.int64))
        buf_i.append(lib_int_all[sl_])
        buf_d.append(np.arange(s, s + L, dtype=np.int64))
        n_buf += 1
        if n_buf >= CH:
            flush()
            n_buf = 0
    flush()

    f_A = np.zeros(n)
    f_B = np.zeros(n)
    _prec_fracs(comp_p, jA_hi, jB_lo, jB_hi, *_POW, f_A, f_B)
    has_B = (wb >= 0) & (f_B >= min_frac)
    # entries outside every window keep the plain library template
    for e in np.where(~ok)[0]:
        s, L = int(so[e]), int(sl[e])
        int_A[s:s + L] = lib_int_all[s:s + L]
    ctx = SplitWindowTemplates(int_A, int_B, f_A.astype(np.float32), f_B.astype(np.float32), has_B,
                               wa.astype(np.int32), wb.astype(np.int32), lo, hi, codes_all, min_frac)
    nt = library.n_targets
    logger.info(f"--split_win: {len(grid)} windows; templates for {n:,} entries in {time.time() - t0:.0f}s. "
                f"Targets: mean envelope fraction isolated in the M0 window {f_A[:nt][ok[:nt]].mean():.3f}; "
                f"{100 * has_B[:nt].mean():.1f}% have >= {min_frac:g} of it in the window above"
                + (f"; {n_unk} non-b/y peaks treated as b" if n_unk else ""))
    return ctx


def attach_neighbours(ctx, ms2scans, max_dt_s):
    """MS2 scan_num -> the scan of the window directly above, nearest in RT, when it was
    acquired within *max_dt_s* seconds.  Methods may interleave windows (e.g. two sweeps
    per cycle), so the window above is not always acquired next to this one; a shared
    coefficient is only sound when the two scans are (near-)simultaneous."""
    by_lo = {}
    for s in ms2scans:
        lo, hi = float(s.ms1window[0]), float(s.ms1window[1])
        by_lo.setdefault(round(lo, 2), []).append(s)
        ctx.scan_grid[s.scan_num] = ctx.grid_index(round(lo, 3), round(hi, 3))
    lookup = {}
    for k, lst in by_lo.items():
        lst.sort(key=lambda x: x.RT)
        lookup[k] = (np.array([x.RT for x in lst], dtype=np.float64), lst)
    n_pair = n_far = 0
    max_dt = max_dt_s / 60.0     # spectrum RT is in minutes
    for s in ms2scans:
        up = lookup.get(round(float(s.ms1window[1]), 2))
        if up is None:
            continue
        rts, lst = up
        j = int(np.searchsorted(rts, s.RT))
        best, bd = None, np.inf
        for k in (j - 1, j):
            if 0 <= k < len(rts) and abs(rts[k] - s.RT) < bd:
                best, bd = lst[k], abs(rts[k] - s.RT)
        if best is not None and bd <= max_dt:
            ctx.upper[s.scan_num] = best
            n_pair += 1
        elif best is not None:
            n_far += 1
    logger.info(f"--split_win: {n_pair:,} of {len(ms2scans):,} MS2 scans have the window above acquired within "
                f"{max_dt_s:g} s (joint fit); {n_far:,} have it further away (fitted alone, with "
                f"window-truncated templates)")


@njit(nogil=True, cache=False)
def build_joint_library(cand, kind, spec_mz, int_A, int_B, so, sl, codes, top_n, scale, b_flag):
    """Per-candidate template spectra for the joint (A + scaled B) spectrum.
    Returns flat mz/int/code arrays, per-candidate offsets/lengths (int64/int32) and top-N
    (local indices), laid out like the library backing arrays."""
    n = cand.shape[0]
    lens = np.zeros(n, dtype=np.int32)
    for i in range(n):
        c = cand[i]
        s = so[c]
        L = sl[c]
        cnt = 0
        for k in range(L):
            if kind[i] == 0 or kind[i] == 1 or kind[i] == 3:
                if int_A[s + k] > 0.0:
                    cnt += 1
            if kind[i] == 1 or kind[i] == 2:
                if int_B[s + k] > 0.0:
                    cnt += 1
        lens[i] = cnt
    offs = np.zeros(n, dtype=np.int64)
    tot = 0
    for i in range(n):
        offs[i] = tot
        tot += lens[i]
    mz = np.empty(tot, dtype=np.float64)
    it = np.empty(tot, dtype=np.float64)
    cd = np.empty(tot, dtype=np.int32)
    for i in range(n):
        c = cand[i]
        s = so[c]
        L = sl[c]
        p = offs[i]
        kd = kind[i]
        if kd == 0 or kd == 1 or kd == 3:
            f = scale if kd == 3 else 1.0
            for k in range(L):
                if int_A[s + k] > 0.0:
                    mz[p] = spec_mz[s + k] * f
                    it[p] = int_A[s + k]
                    cd[p] = codes[s + k]
                    p += 1
        if kd == 1 or kd == 2:
            f = scale if kd == 1 else 1.0
            for k in range(L):
                if int_B[s + k] > 0.0:
                    mz[p] = spec_mz[s + k] * f
                    it[p] = int_B[s + k]
                    cd[p] = codes[s + k] | b_flag
                    p += 1
    tn_len = np.zeros(n, dtype=np.int32)
    for i in range(n):
        tn_len[i] = min(top_n, lens[i])
    tn_off = np.zeros(n, dtype=np.int64)
    t = 0
    for i in range(n):
        tn_off[i] = t
        t += tn_len[i]
    tn = np.empty(t, dtype=np.int32)
    for i in range(n):
        o = offs[i]
        L = lens[i]
        if L == 0:
            continue
        order = np.argsort(-it[o:o + L])
        for k in range(tn_len[i]):
            tn[tn_off[i] + k] = order[k]
    return mz, it, cd, offs, lens, tn, tn_off, tn_len


def gather_candidates(ctx, query, a_mz, b_mz, lo_A, hi_A, lo_B, hi_B, scan_grid_idx, n_targets):
    """Candidate library indices and kinds for one joint fit.
    query(dia_mz, win_lo, win_hi) -> global indices (fragment index or m/z+RT fallback).
    Returns (target_idx, target_kind, decoy_idx, decoy_kind)."""
    has_B, win_B = ctx.has_B, ctx.win_B
    q_A = np.asarray(query(a_mz, lo_A, hi_A), dtype=np.int64)
    parts = [q_A]
    if b_mz is not None:
        q_BA = np.asarray(query(b_mz, lo_A, hi_A), dtype=np.int64)
        parts.append(q_BA[has_B[q_BA]])
    prim = np.unique(np.concatenate(parts)) if len(parts) > 1 else np.unique(q_A)
    kprim = np.where(has_B[prim] & (b_mz is not None), KIND_AB, KIND_A).astype(np.int8)
    sat = np.empty(0, np.int64)
    if scan_grid_idx >= 1:
        lo_Z = ctx.grid_lo[scan_grid_idx - 1]
        q_Z = np.asarray(query(a_mz, lo_Z, lo_A), dtype=np.int64)
        q_Z = q_Z[has_B[q_Z] & (win_B[q_Z] == scan_grid_idx)]
        sat = np.setdiff1d(q_Z, prim)
    up = np.empty(0, np.int64)
    if b_mz is not None:
        q_B = np.asarray(query(b_mz, lo_B, hi_B), dtype=np.int64)
        up = np.setdiff1d(np.setdiff1d(q_B, prim), sat)
    idx = np.concatenate([prim, sat, up])
    kind = np.concatenate([kprim, np.full(len(sat), KIND_SAT, np.int8), np.full(len(up), KIND_UP, np.int8)])
    t = idx < n_targets
    return idx[t], kind[t], idx[~t], kind[~t]


def unscale_mz(a):
    a = np.asarray(a, dtype=np.float64)
    return np.where(a > UNSCALE_ABOVE, a / SCALE, a)
