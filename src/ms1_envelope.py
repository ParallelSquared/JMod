#  MS1 isotope-envelope features (--ms1_envelope).
#
#  JMod's only MS1 evidence is `mz_error`: the monoisotopic peak in the single nearest MS1 scan.
#  Heavy precursors (e.g. mass-tagged peptides, median ~2 kDa) carry only ~30% of their
#  signal in M0, and M1 is the tallest isotope for most of them, so an M0-only check misses
#  real precursors and says nothing about envelope shape.  For every fitted candidate these
#  features look at M0..M3 in the nearest MS1 scan and the --ms1_envelope_scans scans either
#  side, against the exact theoretical envelope of the precursor's composition:
#
#    ms1env_n          isotopes (of M0..M3 with >= 5% of the envelope) found, nearest scan
#    ms1env_cos        cosine of observed M0..M3 intensities to the theoretical envelope, nearest scan
#    ms1env_logint     log10(1 + summed M0..M3 intensity), nearest scan
#    ms1env_ppm        intensity-weighted mean |ppm error| of the found isotopes, nearest scan
#    ms1env_n_nb       max of ms1env_n over the nearest and neighbouring scans
#    ms1env_cos_nb     max of ms1env_cos over them
#    ms1env_logint_nb  max of ms1env_logint over them
#
#  They are written as extra columns of the search output and so enter the scorer like
#  every other feature.

import numpy as np
from numba import njit

from src.logger import logger

ISO_DIFF = 1.0033548
N_ISO = 4
FEATURES = ['ms1env_n', 'ms1env_cos', 'ms1env_logint', 'ms1env_ppm', 'ms1env_n_nb', 'ms1env_cos_nb', 'ms1env_logint_nb']


def precursor_envelopes(library, tag):
    """(n_entries, N_ISO) theoretical M0..M3 abundances (fractions of the whole envelope)."""
    from src.split_windows import _residue_vectors, _env, _POW, NB
    n = len(library.mod_seq)
    H2O = np.array([0, 2, 0, 1, 0])
    comp = np.zeros((n, 5), dtype=np.int64)
    for e, R in enumerate(_residue_vectors(library.mod_seq[:n], tag)):
        comp[e] = R.sum(0) + H2O
    out = np.zeros((n, N_ISO), dtype=np.float64)
    _envs(comp, *_POW, out, NB)
    return out


@njit(nogil=True, cache=False)
def _envs(comp, pC, pH, pN, pO, pS, out, nb):
    from_env = np.zeros(nb)
    for i in range(comp.shape[0]):
        _env_local(comp[i], pC, pH, pN, pO, pS, from_env)
        for k in range(out.shape[1]):
            out[i, k] = from_env[k]


@njit(nogil=True, cache=False)
def _env_local(c, pC, pH, pN, pO, pS, out):
    nb = out.shape[0]
    v = np.zeros(nb)
    w = np.zeros(nb)
    tmp = np.zeros(nb)
    for k in range(nb):
        v[k] = pC[min(max(c[0], 0), pC.shape[0] - 1), k]
    for e in range(1, 5):
        if e == 1:
            tab = pH
        elif e == 2:
            tab = pN
        elif e == 3:
            tab = pO
        else:
            tab = pS
        n = min(max(c[e], 0), tab.shape[0] - 1)
        for k in range(nb):
            w[k] = tab[n, k]
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
def _features(pmz, pz, env, scan_mz, scan_int, scan_off, centre, tol, out):
    n = pmz.shape[0]
    n_scans = scan_off.shape[0] - 1
    I = np.zeros(N_ISO)
    E = np.zeros(N_ISO)
    for i in range(n):
        pn = 0.0
        for j in range(N_ISO):
            pn += env[i, j] * env[i, j]
        pn = np.sqrt(pn)
        best_cos = 0.0
        best_li = 0.0
        best_n = 0.0
        for s in range(n_scans):
            a = scan_off[s]
            b = scan_off[s + 1]
            for j in range(N_ISO):
                I[j] = 0.0
                E[j] = 0.0
                q = pmz[i] + j * ISO_DIFF / pz[i]
                if b > a:
                    k = np.searchsorted(scan_mz[a:b], q) + a
                    bd = tol
                    for kk in (k - 1, k):
                        if a <= kk < b:
                            d = abs(scan_mz[kk] - q) / q
                            if d <= bd:
                                bd = d
                                I[j] = scan_int[kk]
                                E[j] = d
            nf = 0.0
            dot = 0.0
            on = 0.0
            tot = 0.0
            ew = 0.0
            for j in range(N_ISO):
                if I[j] > 0.0 and env[i, j] >= 0.05:
                    nf += 1.0
                dot += I[j] * env[i, j]
                on += I[j] * I[j]
                tot += I[j]
                ew += I[j] * E[j]
            cos = dot / (np.sqrt(on) * pn) if on > 0.0 and pn > 0.0 else 0.0
            li = np.log10(1.0 + tot)
            if s == centre:
                out[i, 0] = nf
                out[i, 1] = cos
                out[i, 2] = li
                out[i, 3] = (ew / tot) * 1e6 if tot > 0.0 else 0.0
            if cos > best_cos:
                best_cos = cos
            if li > best_li:
                best_li = li
            if nf > best_n:
                best_n = nf
        out[i, 4] = best_n
        out[i, 5] = best_cos
        out[i, 6] = best_li


def features(lib_idx, pmz, library, env, ms1_spectra, ms1_rt, prec_rt, tol, n_nb):
    """Feature matrix (len(lib_idx), len(FEATURES)) for candidates at library indices lib_idx,
    precursor m/z pmz (calibrated, decoy offset applied), for the MS2 scan at prec_rt."""
    out = np.zeros((len(lib_idx), len(FEATURES)))
    if len(lib_idx) == 0 or ms1_spectra is None or len(ms1_spectra) == 0:
        return out
    c = int(np.searchsorted(ms1_rt, prec_rt))
    if c > 0 and (c == len(ms1_rt) or abs(ms1_rt[c - 1] - prec_rt) <= abs(ms1_rt[c] - prec_rt)):
        c -= 1
    lo, hi = max(0, c - n_nb), min(len(ms1_spectra), c + n_nb + 1)
    scans = ms1_spectra[lo:hi]
    off = np.zeros(len(scans) + 1, dtype=np.int64)
    for k, s in enumerate(scans):
        off[k + 1] = off[k] + (len(s.mz) if s.mz is not None else 0)
    mz = np.concatenate([np.asarray(s.mz, np.float64) for s in scans if s.mz is not None and len(s.mz)]) if off[-1] else np.zeros(0)
    it = np.concatenate([np.asarray(s.intens, np.float64) for s in scans if s.mz is not None and len(s.mz)]) if off[-1] else np.zeros(0)
    idx = np.asarray(lib_idx, dtype=np.int64)
    _features(np.asarray(pmz, np.float64), np.asarray(library.prec_z, np.float64)[idx], env[idx],
              mz, it, off, c - lo, float(tol), out)
    return out
