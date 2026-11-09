"""--ms1_envelope features on synthetic MS1 scans.  Run: python tests/test_ms1_envelope.py"""
import os, sys, types
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pyteomics import mass
from src.mass_tags import massTag
from src import ms1_envelope as me

tag = massTag(rules='nK', base_mass=502.21759732782, delta=[0.0], channel_names=['0'], name='HeavyTag',
              compositions={'0': mass.Composition({'C': 23, 'H': 30, 'N': 6, 'O': 7})})
lib = types.SimpleNamespace(mod_seq=np.array(['(HeavyTag-0)SGEPLSTLAELNGK(HeavyTag-0)', 'ELVSC(UniMod:4)LR'], dtype=object),
                            prec_z=np.array([2.0, 2.0]))
env = me.precursor_envelopes(lib, tag)
# independent check against pyteomics/brainpy-free exact convolution
ABUND = {'C': [0.9893, 0.0107], 'H': [0.999885, 0.000115], 'N': [0.99636, 0.00364], 'O': [0.99757, 0.00038, 0.00205], 'S': [0.9499, 0.0075, 0.0425, 0.0, 0.0001]}
def exact(comp):
    v = np.array([1.0])
    for e in 'CHNOS':
        for _ in range(comp.get(e, 0)):
            v = np.convolve(v, ABUND[e])[:12]
    return v[:4] / v.sum()
c0 = dict(mass.Composition(sequence='SGEPLSTLAELNGK')); c0 = {k: c0.get(k, 0) + 2 * {'C': 23, 'H': 30, 'N': 6, 'O': 7}.get(k, 0) for k in set(c0) | {'C', 'H', 'N', 'O'}}
c1 = dict(mass.Composition(sequence='ELVSCLR')); c1 = {k: c1.get(k, 0) + {'C': 2, 'H': 3, 'N': 1, 'O': 1}.get(k, 0) for k in set(c1) | {'C', 'H', 'N', 'O'}}
assert np.allclose(env[0], exact(c0), atol=2e-3) and np.allclose(env[1], exact(c1), atol=2e-3), (env, exact(c0), exact(c1))
print('theoretical envelopes OK:', np.round(env, 3))

pmz = np.array([900.40, 451.70]); ISO = me.ISO_DIFF
def scan(rt, peaks):
    mz = np.array(sorted(p[0] for p in peaks)); it = np.array([p[1] for p in sorted(peaks)])
    return types.SimpleNamespace(RT=rt, mz=mz, intens=it)
scans, rts = [], []
for k, rt in enumerate([10.00, 10.01, 10.02, 10.03, 10.04]):
    scale = [0.2, 0.6, 1.0, 0.6, 0.2][k]
    pk = [((900.40 + j * ISO / 2) * (1 + 1e-6), 1e6 * scale * env[0, j]) for j in range(4)]           # exact envelope, +1 ppm
    if k == 2:
        pk = [p for j, p in enumerate(pk) if j != 0]                                               # M0 missing in the nearest scan
    pk += [(451.70 + j * ISO / 2, 1e5 * scale * w) for j, w in enumerate([0.2, 0.8, 0.0, 0.0])]   # distorted envelope for entry 1
    pk += [(500.0, 5e4), (950.0, 1e4)]
    scans.append(scan(rt, pk)); rts.append(rt)
F = me.features(np.array([0, 1]), pmz, lib, env, scans, np.array(rts), 10.021, 3e-6, 2)
f = dict(zip(me.FEATURES, F.T))
print({k: np.round(v, 3) for k, v in f.items()})
assert f['ms1env_n'][0] == 3 and f['ms1env_n_nb'][0] == 4          # M0 missing in the nearest scan, all 4 in neighbours
assert 0.85 < f['ms1env_cos'][0] < 1.0 and f['ms1env_cos_nb'][0] > 0.999
assert abs(f['ms1env_ppm'][0] - 1.0) < 0.05
assert f['ms1env_cos'][1] < 0.8                                     # wrong shape scores low
assert f['ms1env_logint_nb'][0] >= f['ms1env_logint'][0]
print('MS1 envelope features OK')
