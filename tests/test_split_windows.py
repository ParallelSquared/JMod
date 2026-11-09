"""--split_win template maths against an independent exact calculation.

Builds a two-entry fake library (one heavy-tagged peptide, one CAM peptide) with JMod's
own isotope expansion, places each precursor's M0 at a chosen spot in a 2 Th window, and checks
split_windows.build_templates: per peak, window-A and window-B intensities over the library
intensity must equal P(complement heavy count in [a-k, b-k]) computed by brute-force
convolution of natural abundances; the isolated envelope fractions must equal the precursor
envelope summed over the isolated isotopes.  Also checks build_joint_library's layout.

Run: python tests/test_split_windows.py
"""
import os, sys, types
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pyteomics import mass
from src.mass_tags import massTag
import src.iso_functions as iso_f
from src.utils.frag_encoding import encode_frag_names
from src import split_windows as sw

ABUND = {'C': [0.9893, 0.0107], 'H': [0.999885, 0.000115], 'N': [0.99636, 0.00364],
         'O': [0.99757, 0.00038, 0.00205], 'S': [0.9499, 0.0075, 0.0425, 0.0, 0.0001]}
TAGC = {'C': 23, 'H': 30, 'N': 6, 'O': 7}
RES = {'G': (2, 3, 1, 1, 0), 'A': (3, 5, 1, 1, 0), 'S': (3, 5, 1, 2, 0), 'P': (5, 7, 1, 1, 0), 'V': (5, 9, 1, 1, 0),
       'T': (4, 7, 1, 2, 0), 'C': (3, 5, 1, 1, 1), 'L': (6, 11, 1, 1, 0), 'N': (4, 6, 2, 2, 0), 'E': (5, 7, 1, 3, 0),
       'K': (6, 12, 2, 1, 0), 'R': (6, 12, 4, 1, 0)}


def exact_env(comp, n=14):
    v = np.array([1.0])
    for e, c in zip('CHNOS', comp):
        for _ in range(int(c)):
            v = np.convolve(v, ABUND[e])[:n]
    out = np.zeros(n); out[:len(v)] = v
    return out


def residues(seq_spec):
    out = []
    for aa, mods in seq_spec:
        r = np.array(RES[aa], float)
        for m in mods:
            if m == 'tag':
                r += [TAGC['C'], TAGC['H'], TAGC['N'], TAGC['O'], 0]
            if m == 'cam':
                r += [2, 3, 1, 1, 0]
        out.append(r)
    return np.array(out)


tag = massTag(rules='nK', base_mass=502.21759732782, delta=[0.0], channel_names=['0'], name='HeavyTag',
              compositions={'0': mass.Composition(TAGC)})
peps = [
    # (mod_seq, residue spec, z, position of M0 inside its 2 Th window as fraction from the lower edge)
    ('S(HeavyTag-0)GEPLSTLAELNGK(HeavyTag-0)', [('S', ['tag'])] + [(a, []) for a in 'GEPLSTLAELNG'] + [('K', ['tag'])], 2, 0.85),
    ('ELVSC(UniMod:4)LR', [(a, []) for a in 'ELVS'] + [('C', ['cam'])] + [(a, []) for a in 'LR'], 2, 0.60),
    ('S(HeavyTag-0)GEPLSTLAELNGK(HeavyTag-0)', [('S', ['tag'])] + [(a, []) for a in 'GEPLSTLAELNG'] + [('K', ['tag'])], 3, 0.95),
]
H2O = np.array([0, 2, 0, 1, 0])
spec_mz, spec_int, codes, offs, lens, mod_seq, prec_z, mz0s = [], [], [], [], [], [], [], []
o = 0
for ms, spec, z, pos in peps:
    L = len(spec)
    frags = {}
    for k in (2, 4, L - 2):
        frags[f'b{k}_1'] = [200.0 + k, 1.0 + k]
        frags[f'y{k}_1'] = [300.0 + k, 2.0 + k]
    sp, names = iso_f.gen_isotopes_dict(ms, frags, tag, 4)
    spec_mz.append(sp[:, 0]); spec_int.append(sp[:, 1]); codes.append(encode_frag_names(names))
    offs.append(o); lens.append(len(sp)); o += len(sp)
    mod_seq.append(ms); prec_z.append(float(z))
    lo = 800.0 if z == 2 else 700.0
    mz0s.append(lo + 2.0 * pos)

lib = types.SimpleNamespace(
    mod_seq=np.array(mod_seq, dtype=object), prec_z=np.array(prec_z),
    spectrum_mz=np.concatenate(spec_mz), spectrum_int=np.concatenate(spec_int).astype(np.float32),
    frag_names_data=np.concatenate(codes).astype(np.int32),
    spectrum_offsets=np.array(offs, dtype=np.int64), spectrum_lengths=np.array(lens, dtype=np.int32),
    spectrum_perm=None, n_targets=len(peps))
scans = [types.SimpleNamespace(ms1window=np.array([lo, lo + 2.0])) for lo in np.arange(600.0, 1000.0, 2.0)]
rt_mz = np.column_stack([np.zeros(len(peps)), np.array(mz0s), np.full(len(peps), np.nan)])
ctx = sw.build_templates(lib, rt_mz, scans, [], tag, 0.4)
assert ctx is not None

ISO = 1.0033548
worst = 0.0
for e, (ms, spec, z, pos) in enumerate(peps):
    R = residues(spec)
    prec = R.sum(0) + H2O
    Pp = exact_env(prec)
    mz0 = mz0s[e]; lo = np.floor(mz0 / 2) * 2; hi = lo + 2
    jA = int(np.floor((hi - mz0) * z / ISO)); jB_hi = int(np.floor((hi + 2 - mz0) * z / ISO))
    fA, fB = Pp[:jA + 1].sum(), Pp[jA + 1:jB_hi + 1].sum()
    assert abs(ctx.f_A[e] - fA) < 1e-5 and abs(ctx.f_B[e] - fB) < 1e-5, (ms, ctx.f_A[e], fA, ctx.f_B[e], fB)
    assert ctx.has_B[e] == (fB >= 0.4)
    s, L = offs[e], lens[e]
    for p in range(s, s + L):
        c = int(lib.frag_names_data[p])
        t, ix, k = c & 7, (c >> 3) & 0xFF, (c >> 17) & 0xF
        frag = R[:ix].sum(0) if t == 0 else R[::-1][:ix].sum(0) + H2O
        Pc = np.cumsum(exact_env(prec - frag))
        a = Pc[jA - k] if jA - k >= 0 else 0.0
        b = (Pc[jB_hi - k] if jB_hi - k >= 0 else 0.0) - (Pc[jA - k] if jA - k >= 0 else 0.0)
        gotA = ctx.int_A[p] / lib.spectrum_int[p]
        gotB = ctx.int_B[p] / lib.spectrum_int[p]
        worst = max(worst, abs(gotA - a), abs(gotB - b))
        assert abs(gotA - a) < 1e-5 and abs(gotB - b) < 1e-5, (ms, p, k, gotA, a, gotB, b)
    print(f'{ms:42s} z={z} M0 at {pos:.2f} of window: f_A {fA:.3f} f_B {fB:.3f} (J_A={jA}) has_B={bool(ctx.has_B[e])}')
print(f'window factors match exact calculation (max abs diff {worst:.1e})')

# joint library layout: kinds 0..3 for entry 0
cand = np.array([0, 0, 0, 0], dtype=np.int64)
kind = np.array([0, 1, 2, 3], dtype=np.int8)
mz, it, cd, of, ln, tn, tno, tnl = sw.build_joint_library(cand, kind, lib.spectrum_mz, ctx.int_A, ctx.int_B,
                                                          lib.spectrum_offsets, lib.spectrum_lengths, ctx.codes, 10, sw.SCALE, sw.B_FLAG)
nA = int((ctx.int_A[:lens[0]] > 0).sum()); nB = int((ctx.int_B[:lens[0]] > 0).sum())
assert list(ln) == [nA, nA + nB, nB, nA], (list(ln), nA, nB)
seg = lambda i: slice(of[i], of[i] + ln[i])
assert np.all(mz[seg(0)] < 2e4) and np.all(mz[seg(3)] > 2e4)
assert np.all(mz[seg(1)][:nA] < 2e4) and np.all(mz[seg(1)][nA:] > 2e4)
assert np.all((cd[seg(1)][nA:] & sw.B_FLAG) != 0) and np.all((cd[seg(1)][:nA] & sw.B_FLAG) == 0)
assert np.all((cd[seg(2)] & sw.B_FLAG) != 0) and np.all(mz[seg(2)] < 2e4)
for i in range(4):
    top = it[seg(i)][tn[tno[i]:tno[i] + tnl[i]]]
    assert np.all(np.diff(top) <= 0) and tnl[i] == min(10, ln[i])
assert np.allclose(sw.unscale_mz(mz[seg(3)]), mz[seg(0)])
print('build_joint_library layout OK')
