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
"""MS2 IM bands with each peak stored once.

The IM bands overlap by 50% (stride = width / 2), so almost every peak belongs to
two adjacent bands.  Copying each band's peaks out doubled the banded MS2 (~69 GB
on a 20 ng 5plex).  Instead each (frame, window) group keeps one buffer of its
peaks, ordered by which bands contain them:

    key 2b      peaks in band b only (edges of the band range)
    key 2b + 1  peaks in bands b and b + 1

and within a key by (m/z, mobility, intensity).  Band b is then exactly the keys
2b - 1, 2b and 2b + 1: three consecutive runs of the buffer, each sorted.  A
band's peak arrays (``BandSpectrum.mz`` / ``intens`` / ``mobility``) are those
runs merged by (m/z, mobility, intensity) -- the band's peaks in that order, the
same arrays a per-band copy sorted on that key holds -- built on demand and
cached per thread for a few bands.  Kernels that look up ions in many bands
(fragment correlations) read the runs directly instead (``gather_range``).

Membership is the test the per-band copies used: ``lo <= mob < hi`` on the
float64 mobility, for the two candidate bins of each peak (the 50% overlap,
``width = 2 * stride``, puts a peak in at most two).
"""
import threading
from collections import OrderedDict

import numba as nb
import numpy as np

from src.utils.io.load_files import Spectrum


@nb.njit(cache=False, nogil=True)
def band_keys(mob_arr, im_bins):
    """Band-membership key per peak (see the module docstring), -1 if in no band."""
    n = mob_arr.shape[0]
    out = np.full(n, -1, dtype=np.int64)
    bin_lo0 = im_bins[0, 0]
    stride = im_bins[1, 0] - im_bins[0, 0]
    n_bins = im_bins.shape[0]
    for i in range(n):
        m = np.float64(mob_arr[i])
        r = np.int64((m - bin_lo0) / stride)
        in_r = r >= 0 and r < n_bins and im_bins[r, 0] <= m and m < im_bins[r, 1]
        q = r - 1
        in_q = q >= 0 and q < n_bins and im_bins[q, 0] <= m and m < im_bins[q, 1]
        if in_r and in_q:
            out[i] = 2 * q + 1
        elif in_r:
            out[i] = 2 * r
        elif in_q:
            out[i] = 2 * q
    return out


@nb.njit(cache=False, nogil=True, inline="always")
def _less(mz, mob, it, a, b):
    """Peak a before peak b in (m/z, mobility, intensity) order."""
    if mz[a] != mz[b]:
        return mz[a] < mz[b]
    if mob[a] != mob[b]:
        return mob[a] < mob[b]
    return it[a] < it[b]


@nb.njit(cache=False, nogil=True)
def _merge3(mz, it, mob, a, a1, b, b1, c, c1):
    """The peaks [a,a1), [b,b1), [c,c1) of a group buffer, each (m/z, mob, int)-
    sorted, merged into new arrays in that order.  ``mob`` may be empty (no
    mobility), only when at most one range is non-empty (then nothing is compared)."""
    n = (a1 - a) + (b1 - b) + (c1 - c)
    omz = np.empty(n, dtype=mz.dtype)
    oit = np.empty(n, dtype=it.dtype)
    omob = np.empty(n if mob.shape[0] else 0, dtype=mob.dtype)
    for k in range(n):
        # which range supplies the next peak (by range, not position: an exhausted
        # range's cursor can equal the next range's start)
        run = -1
        best = -1
        if a < a1:
            run, best = 0, a
        if b < b1 and (best < 0 or _less(mz, mob, it, b, best)):
            run, best = 1, b
        if c < c1 and (best < 0 or _less(mz, mob, it, c, best)):
            run, best = 2, c
        omz[k] = mz[best]
        oit[k] = it[best]
        if mob.shape[0]:
            omob[k] = mob[best]
        if run == 0:
            a += 1
        elif run == 1:
            b += 1
        else:
            c += 1
    return omz, oit, omob


@nb.njit(cache=False, nogil=True)
def merge_runs(mz, it, mob, s0, s1, s2, s3):
    """The band of runs [s0,s1), [s1,s2), [s2,s3), merged (see ``_merge3``)."""
    return _merge3(mz, it, mob, s0, s1, s1, s2, s2, s3)


@nb.njit(cache=False, nogil=True)
def gather_range(gmz, git, gmob, s0, s1, s2, s3, lo, hi):
    """The band's peaks with ``lo <= mz <= hi`` in the band's order.

    Exactly the merged band's ``[searchsorted(mz, lo), searchsorted(mz, hi,
    side='right'))`` slice.  When the matches lie in one run (always, for
    spectra stored the usual way, passed as s0 = s1 = 0, s2 = s3 = n) they are
    returned as views; otherwise the runs' matches are merged (they are few).
    ``gmob`` may be empty (no mobility); the mobility returned is then empty."""
    p0 = s0 + np.searchsorted(gmz[s0:s1], lo)
    p1 = s0 + np.searchsorted(gmz[s0:s1], hi, side="right")
    q0 = s1 + np.searchsorted(gmz[s1:s2], lo)
    q1 = s1 + np.searchsorted(gmz[s1:s2], hi, side="right")
    r0 = s2 + np.searchsorted(gmz[s2:s3], lo)
    r1 = s2 + np.searchsorted(gmz[s2:s3], hi, side="right")
    nonempty = int(p1 > p0) + int(q1 > q0) + int(r1 > r0)
    if nonempty <= 1:
        # one range (or none): a view; the empty ranges add nothing
        a = p0 if p1 > p0 else (q0 if q1 > q0 else r0)
        b = p1 if p1 > p0 else (q1 if q1 > q0 else r1)
        return gmz[a:b], git[a:b], gmob[a:b] if gmob.shape[0] else gmob
    return _merge3(gmz, git, gmob, p0, p1, q0, q1, r0, r1)


@nb.njit(cache=False, nogil=True)
def nearest_peak(gmz, git, gmob, s0, s1, s2, s3, q):
    """The merged band's neighbours of ``q``: (index of the last peak with
    mz < q, of the first with mz >= q) as buffer positions, -1 if none.  The
    merged band's ``pos - 1`` and ``pos`` for ``pos = searchsorted(mz, q)``."""
    pred = -1
    succ = -1
    for k in range(3):
        if k == 0:
            a, b = s0, s1
        elif k == 1:
            a, b = s1, s2
        else:
            a, b = s2, s3
        if b <= a:
            continue
        pos = a + np.searchsorted(gmz[a:b], q)
        if pos > a:
            if pred < 0 or _less(gmz, gmob, git, pred, pos - 1):
                pred = pos - 1
        if pos < b:
            if succ < 0 or _less(gmz, gmob, git, pos, succ):
                succ = pos
    return pred, succ


_tls = threading.local()
_CACHE_BANDS = 64


class BandSpectrum(Spectrum):
    """An MS2 IM band whose peaks are three runs of its (frame, window) group's
    buffer (see the module docstring).  ``mz`` / ``intens`` / ``mobility`` are the
    merged band, built on first use and cached per thread (``_CACHE_BANDS``
    bands), so reading all three costs one merge.  Assigning one of them
    replaces it for this spectrum (the others still read the band); once all
    three are assigned the spectrum no longer refers to the group buffer, so the
    buffer is freed when no band uses it (``use_centroided_ms2``)."""

    def __init__(self, group, runs):
        # Spectrum's attributes, without its mz/intens/mobility = None (properties here)
        self.id = None
        self.level = None
        self.RT = None
        self.collision_energy = None
        self.TIC = None
        self.isolation_window = None
        self.im_lo = None
        self.im_hi = None
        self._group = group          # (mz, intens, mob) buffers of the (frame, window) group
        self._runs = runs            # (s0, s1, s2, s3)
        self._own = None             # [mz, intens, mob], each set once assigned

    def _band(self):
        """The merged band's (mz, intens, mob)."""
        s0, s1, s2, s3 = self._runs
        mz, it, mob = self._group
        # one non-empty run: the band is a view of the buffer
        nonempty = (s1 > s0) + (s2 > s1) + (s3 > s2)
        if nonempty <= 1:
            return mz[s0:s3], it[s0:s3], mob[s0:s3]
        cache = getattr(_tls, "bands", None)
        if cache is None:
            cache = _tls.bands = OrderedDict()
        key = id(self)
        hit = cache.get(key)
        if hit is not None and hit[0] is self:
            cache.move_to_end(key)
            return hit[1]
        arrays = merge_runs(mz, it, mob, s0, s1, s2, s3)
        cache[key] = (self, arrays)
        if len(cache) > _CACHE_BANDS:
            cache.popitem(last=False)
        return arrays

    def _arrays(self):
        if self._own is None:
            return self._band()
        if self._group is None:
            return tuple(self._own)
        return tuple(o if o is not None else b for o, b in zip(self._own, self._band()))

    def _assign(self, i, value):
        if self._own is None:
            self._own = [None, None, None]
        self._own[i] = value
        if all(a is not None for a in self._own):
            self._group = self._runs = None      # own arrays only: release the group buffer

    @property
    def mz(self):
        return self._arrays()[0]

    @mz.setter
    def mz(self, value):
        self._assign(0, value)

    @property
    def intens(self):
        return self._arrays()[1]

    @intens.setter
    def intens(self, value):
        self._assign(1, value)

    @property
    def mobility(self):
        return self._arrays()[2]

    @mobility.setter
    def mobility(self, value):
        self._assign(2, value)
