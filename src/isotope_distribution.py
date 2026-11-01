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

"""Aggregated isotope distributions: a drop-in for brainpy's isotopic_variants.

Each element's natural isotopes form a short polynomial in "extra neutrons
over its lightest isotope".  The composition's distribution is the product of
each element's polynomial raised to its count, truncated to the peaks asked
for -- binary exponentiation of truncated convolutions, as JrMod does
(src/library/isotopes.rs).  A second track carries each peak's
abundance-weighted exact mass shift, so a peak's center mass is the
monoisotopic mass plus shift_sum / probability.

Fixed isotopes ("C[13]") are single-isotope elements: they add their exact
mass and leave the shape alone.  Masses and abundances are NIST's
(src/mass_data.py, pyteomics' table, which brainpy ships too).  Output follows brainpy: peaks +0 ...
+(npeaks-1), intensities normalised over the peaks returned, m/z =
(mass + z * proton) / |z| (the neutral mass for z = 0), sorted by m/z.

The only shared state is the element table, replaced whole (never mutated)
when a new element first appears, so it is safe to call from many threads.
"""

import re
from typing import NamedTuple

import numpy as np
from numba import njit
from src.mass import nist_mass

PROTON = nist_mass["H+"][0][0]

# A peak span can not exceed this many nominal neutrons; enough for any npeaks
# JMod asks for.
MAX_PEAKS = 16


class Peak:
    """One isotope peak.  Mutable, as brainpy's are: callers shift mz in place."""
    __slots__ = ("mz", "intensity", "charge")

    def __init__(self, mz, intensity, charge):
        self.mz = mz
        self.intensity = intensity
        self.charge = charge

    def __repr__(self):
        return f"Peak(mz={self.mz:f}, intensity={self.intensity:f}, charge={self.charge})"


_FIXED_ISOTOPE = re.compile(r"^([A-Z][a-z]*)\[(\d+)\]$")


def _element(symbol):
    """(neutron offsets, abundances, mass shifts, lightest mass) of an element,
    or of a fixed isotope written "C[13]"."""
    m = _FIXED_ISOTOPE.match(symbol)
    if m:
        mass = nist_mass[m.group(1)][int(m.group(2))][0]
        return (np.zeros(1, np.int64), np.ones(1), np.zeros(1), mass)
    isotopes = sorted((n, mass, ab) for n, (mass, ab) in nist_mass[symbol].items()
                      if n != 0 and ab > 0)
    lightest_n, lightest_mass = isotopes[0][0], isotopes[0][1]
    return (np.array([n - lightest_n for n, _, _ in isotopes], np.int64),
            np.array([ab for _, _, ab in isotopes]),
            np.array([mass - lightest_mass for _, mass, _ in isotopes]),
            lightest_mass)


class _Table(NamedTuple):
    """Every element seen so far, flat: element e's isotopes are
    starts[e]:starts[e+1] of offsets/abundances/shifts."""
    index: dict
    offsets: np.ndarray
    abundances: np.ndarray
    shifts: np.ndarray
    starts: np.ndarray
    lightest: np.ndarray


def _build_table(symbols):
    parts = [_element(sym) for sym in symbols]
    starts = np.zeros(len(parts) + 1, np.int64)
    for i, e in enumerate(parts):
        starts[i + 1] = starts[i] + e[0].shape[0]
    return _Table({sym: i for i, sym in enumerate(symbols)},
                  np.concatenate([e[0] for e in parts]),
                  np.concatenate([e[1] for e in parts]),
                  np.concatenate([e[2] for e in parts]),
                  starts,
                  np.array([e[3] for e in parts]))


# Replaced whole (never mutated) when a new element appears, so a thread that
# has read _table keeps a consistent snapshot
_table = _build_table(["C", "H", "N", "O", "S", "P", "C[13]", "H[2]", "N[15]", "O[18]"])


def _table_for(symbols):
    global _table
    t = _table
    missing = [sym for sym in symbols if sym not in t.index]
    if missing:
        t = _build_table(list(t.index) + missing)
        _table = t
    return t


@njit(nogil=True)
def _convolve(a_p, a_s, b_p, b_s, k):
    """Truncated convolution of two (probability, shift-sum) distributions."""
    out_p = np.zeros(k)
    out_s = np.zeros(k)
    for i in range(k):
        if a_p[i] == 0.0 and a_s[i] == 0.0:
            continue
        for j in range(k - i):
            out_p[i + j] += a_p[i] * b_p[j]
            out_s[i + j] += a_p[i] * b_s[j] + a_s[i] * b_p[j]
    return out_p, out_s


@njit(nogil=True)
def _peaks(elements, counts, offsets, abundances, shifts, starts, lightest, k, charge, proton):
    """m/z and normalised intensity of peaks +0 .. +(k-1) of the composition
    counts[i] x element elements[i], and how many of them exist: as brainpy,
    no more than the most extra neutrons the composition can carry, plus one."""
    p = np.zeros(k)
    s = np.zeros(k)
    p[0] = 1.0
    mono = 0.0
    max_shift = 0
    for i in range(elements.shape[0]):
        e = elements[i]
        n = counts[i]
        mono += n * lightest[e]
        max_shift += n * offsets[starts[e + 1] - 1]   # offsets ascend within an element
        base_p = np.zeros(k)
        base_s = np.zeros(k)
        for t in range(starts[e], starts[e + 1]):
            o = offsets[t]
            if o < k:
                base_p[o] += abundances[t]
                base_s[o] += abundances[t] * shifts[t]
        # binary exponentiation of the element's distribution to its count
        acc_p = np.zeros(k)
        acc_s = np.zeros(k)
        acc_p[0] = 1.0
        while n > 0:
            if n & 1:
                acc_p, acc_s = _convolve(acc_p, acc_s, base_p, base_s, k)
            n >>= 1
            if n > 0:
                base_p, base_s = _convolve(base_p, base_s, base_p, base_s, k)
        p, s = _convolve(p, s, acc_p, acc_s, k)
    total = p.sum()
    mz = np.empty(k)
    intensity = np.empty(k)
    for i in range(k):
        mass = mono + (s[i] / p[i] if p[i] > 0 else 0.0)
        mz[i] = (mass + charge * proton) / abs(charge) if charge != 0 else mass
        intensity[i] = p[i] / total
    return mz, intensity, min(k, max_shift + 1)


def isotopic_variants(composition, npeaks, charge=0):
    """brainpy.isotopic_variants for a {element: count} composition: the
    first npeaks aggregated isotope peaks as Peak(mz, intensity, charge)."""
    k = int(npeaks)
    if not 1 <= k <= MAX_PEAKS:
        raise ValueError(f"npeaks must be 1..{MAX_PEAKS}, got {npeaks}")
    items = [(sym, n) for sym, n in composition.items() if n != 0]
    if any(n < 0 for _, n in items):
        raise ValueError(f"negative element count in {dict(composition)}")
    t = _table_for([sym for sym, _ in items])
    mz, intensity, n_peaks = _peaks(np.array([t.index[sym] for sym, _ in items], np.int64),
                           np.array([n for _, n in items], np.int64),
                           t.offsets, t.abundances, t.shifts, t.starts, t.lightest,
                           k, int(charge), PROTON)
    peaks = [Peak(m, i, charge) for m, i in zip(mz[:n_peaks].tolist(), intensity[:n_peaks].tolist())
             if m >= 1 and i >= 0]
    peaks.sort(key=lambda pk: pk.mz)
    return tuple(peaks)
