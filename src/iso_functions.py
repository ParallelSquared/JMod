
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

import warnings

# TODO: remove once we figure out what imports psims at startup on Windows (jmod never writes mzMLb)
warnings.filterwarnings("ignore", message="hdf5plugin is missing")

from src.isotope_distribution import isotopic_variants
import re
from pyteomics import mass
import src.config as config
import tqdm
import os
import copy
import functools
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from functools import reduce
import numpy as np

from src.utils.parse_peptides import parse_peptide, split_frag_name
from src.utils.parse_peptides import parse_peptide, split_frag_name

from src.utils.misc_functions import frag_to_peak

from src.logger import logger



# ## split up the fragment name (b/y)(frag index)(-loss)_charge
# def split_frag_name(ion_type):
#     frag_name,frag_z = ion_type.split("_")
#     loss_check = frag_name.split("-")
#     loss = ""
#     if len(loss_check)>1:
#         frag_name,loss = loss_check
#     frag_type = frag_name[0]
#     frag_idx = int(frag_name[1:])
    
#     return frag_type,frag_idx,loss,frag_z

# def parse_peptide(seq):
#     close_d = {"[": "]", "(": ")"}
#     open_set = set(close_d.keys())
#     close_set = set(close_d.values())
    
#     new_seq = []
#     current = ""
#     s_idx = 0

#     while s_idx < len(seq):
#         s = seq[s_idx]

#         if s in open_set:
#             # Begin collecting the bracketed modification
#             opener = s
#             closer = close_d[opener]
#             mod = s
#             stack = [closer]
#             s_idx += 1

#             while s_idx < len(seq) and stack:
#                 c = seq[s_idx]
#                 mod += c

#                 if c in open_set:
#                     stack.append(close_d[c])
#                 elif c in close_set:
#                     if stack and c == stack[-1]:
#                         stack.pop()
#                 s_idx += 1

#             current += mod  # Append full modification to current letter

#         elif s.isalpha():
#             if current:
#                 new_seq.append(current)
#             current = s
#             s_idx += 1

#         else:
#             # If somehow an unexpected char, just add it
#             current += s
#             s_idx += 1

#     if current:
#         new_seq.append(current)

#     return new_seq
    
### First get the AA sequence and modifications of the fragment
def fragment_seq(peptide, ion_type):
    """
   Get the parsed peptide sequence and parsed ion information

    Parameters
    ----------
    peptide : str or list of str
        A list of individual amino acid strings, as well as their modifications or the string altogether
        i.e. ["A(PSMtag_5plex-4)", "C(Unimod:4), "R"] or "A(PSMtag_5plex-4)C(Unimod:4)R"
    ion_type : str
        ion_type: the ion type, i.e. b5_2, y3-H2O_1. See split_frag_name

    Returns
    -------
    peptide : list of str
        A parsed list of each amino acid with its modification. See parse_peptide
    [frag_type,frag_idx,loss,frag_z]: [str, int, str, str]
        A list of ion type information. See split_frag_name
    
    """
    
    peptide = "".join(peptide)
    # split_peptide = re.findall("([A-Z](?:\(.*?\))?)",peptide)
    split_peptide = parse_peptide(peptide)
    
    ### capture anything in brakets as a modification
    mods = re.finditer("\((.*?)\)",peptide)

    stripped_peptide = re.sub("\(.*?\)","",peptide)
    
    frag_type,frag_idx,loss,frag_z = split_frag_name(ion_type)
    
    assert int(frag_idx)<len(stripped_peptide)
    if frag_type in 'abc':
        seq = split_peptide[:int(frag_idx)]
    elif frag_type in 'xyz':
        seq = split_peptide[-int(frag_idx):]
    else:
        raise(ValueError("Invalid ion type"))
        
    return seq, [frag_type,frag_idx,loss,frag_z]


# def split_peptide(peptide):
    
#     return re.findall("([A-Z](?:\(.*?\))?)",peptide)

### all unimod modifications are stored here: the copy JMod ships, so no
### network is needed (scripts/update_unimod.py refreshes it)
UNIMOD_XML = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "unimod.xml")


@functools.cache
def unimods():
    """Unimod's modifications, read from UNIMOD_XML the first time a process needs them."""
    with open(UNIMOD_XML, "rb") as f:
        return mass.Unimod(f)

# [i["title"] for i in unimods.mods[:10]]

# unimods.by_id(4)["composition"]


## ## get the compostion of the fragment

mod_pattern = re.compile(r"\([A-z]+\:(\d+)\)")
def get_seq_comp(split_seq,ion_type,neutral_loss=None):
    """
   Get the sequence composition of a parsed pepetide sequence

    Parameters
    ----------
    split_seq : list of str
        A list of individual amino acid strings, as well as their modifications
        i.e. ["A(PSMtag_5plex-4)", "C(Unimod:4), "R"]
    ion_type : str
        Ion type: "M" (intact peptide), "b", or "y" among others

    Returns
    -------
    seq_comp : mass.Compostion object
        Pyteomics mass.Composition object that contains the amount of each element within the peptide
    
    """
    
    stripped_seq = "".join([i[0] for i in split_seq]) ## assumes AA comes first before mods
    
    # mods = [int(j) for i in split_seq for j in re.findall("\([A-z]+\:(\d+)\)",i) if len(i)>1]
    mods = [int(j) for i in split_seq for j in mod_pattern.findall(i) if len(i)>1] ### TODO Make work for more mod types
    # tags = [t for aa in split_seq for t in re.findall("(\(.*?\))",aa)]
    seq_comp = mass.Composition(sequence=stripped_seq,ion_type=ion_type)
    if neutral_loss:
        seq_comp -= mass.Composition(neutral_loss)
    for unimod_idx in mods:
        seq_comp += unimods().by_id(unimod_idx)["composition"]
    return seq_comp


# def frag_isotope(frag,seq):
#     # mz,intensity = frags[frag]
#     split_frag_seq,frag_info = fragment_seq(seq,frag)
#     loss = "-"+frag_info[2] if frag_info[2] else frag_info[2]
#     ion_type = frag_info[0] + loss
#     frag_comp = get_seq_comp(split_frag_seq, ion_type)
    
#     isotopes = isotopic_variants(frag_comp,
#                                  npeaks=config.num_iso_peaks,
#                                  charge = int(frag_info[3]))
#     mono_iso_peak = isotopes[0]
#     return isotopes


# def gen_isotopes(seq,frags):
#     new_frags = []
#     for frag in frags:
#         mz,intensity = frags[frag]
#         split_frag_seq,frag_info = fragment_seq(seq,frag)
#         loss = "-"+frag_info[2] if frag_info[2] else frag_info[2]
#         ion_type = frag_info[0] + loss
#         frag_comp = get_seq_comp(split_frag_seq, ion_type)
        
#         isotopes = isotopic_variants(frag_comp,
#                                      npeaks=config.num_iso_peaks,
#                                      charge = int(frag_info[3]))
#         mono_iso_peak = isotopes[0]
#         for iso in isotopes:
#             new_intensity = intensity*(iso.intensity/mono_iso_peak.intensity)
#             if True:#new_intensity > config.min_iso_intensity:
#                 new_frags.append([iso.mz,new_intensity])
    
#     new_frags = np.array(new_frags)
#     sorted_frags = new_frags[np.argsort(new_frags[:,0])]
#     return sorted_frags/[1,np.max(np.array(new_frags)[:,1])]

## Fragment compositions, built from each peptide's residues once instead of
## parsing every fragment's sequence: a fragment's composition is its
## residues' (with their UniMod mods and tags, as get_seq_comp and the tag
## channels count them) plus what its ion type and neutral loss add.  The
## caches below hold what repeats across peptides, per process.

_split_frag_name = functools.cache(split_frag_name)  # "b3_1", "y4-H2O_2", ... repeat in every peptide
_BRACKETS = re.compile(r"\(.*?\)")


@functools.cache
def _ion_comp(ion_type, loss):
    """What a fragment of *ion_type*, less neutral *loss*, adds to its residues' composition."""
    comp = mass.Composition(sequence="G", ion_type=ion_type) - mass.std_aa_comp["G"]
    if loss:
        comp -= mass.Composition(loss)
    return dict(comp)


@functools.cache
def _residue_comp(residue, tag_name, channel_comps):
    """A parsed residue's composition, e.g. "C(UniMod:4)" or "K(mTRAQ-4)": its
    amino acid, its UniMod mods, and its tags' channel compositions
    (*channel_comps*: _hashable_channel_comps of the tag's channel_comp)."""
    comp = mass.Composition(mass.std_aa_comp[residue[0]])
    if len(residue) > 1:
        for unimod_idx in mod_pattern.findall(residue):
            comp += unimods().by_id(int(unimod_idx))["composition"]
        if channel_comps is not None:
            for t in re.findall(rf"\(({tag_name}.*?)\)", residue):
                comp += dict(dict(channel_comps)[re.findall(rf"{tag_name}-(\d+)", t)[0]])
    return dict(comp)


def _hashable_channel_comps(tag):
    # The tag's channel compositions as a cache key: the residue cache outlives
    # the tag object, which a worker unpickles afresh with every chunk
    if tag is None or tag.channel_comp is None:
        return None
    return tuple((channel, tuple(comp.items())) for channel, comp in tag.channel_comp.items())


# Each fragment's isotope envelope by (n_iso, charge, composition): its peaks'
# m/z offsets from the monoisotopic peak and intensities relative to it, as
# isotopic_variants gives them.  Cleared when full, which bounds a worker's memory
# (about 110 MB at 2 isotopes) while the fragments every peptide shares stay cached
_ENVELOPES = {}
_MAX_ENVELOPES = 2 ** 18


def _envelope(n_iso, frag_z, comp):
    key = (n_iso, frag_z, "".join(f"{e}{n}" for e, n in comp))
    envelope = _ENVELOPES.get(key)
    if envelope is None:
        isotopes = isotopic_variants(dict(comp), npeaks=n_iso, charge=frag_z)
        mono = isotopes[0]
        envelope = tuple((iso.mz - mono.mz, iso.intensity / mono.intensity) for iso in isotopes)
        if len(_ENVELOPES) >= _MAX_ENVELOPES:
            _ENVELOPES.clear()
        _ENVELOPES[key] = envelope
    return envelope


def gen_isotopes_dict(seq,frags, tag, n_iso):
    seq = "".join(seq)
    residues = parse_peptide(seq)
    n_stripped = len(_BRACKETS.sub("", seq))
    names = [_split_frag_name(frag) for frag in frags]
    channel_comps = _hashable_channel_comps(tag)
    residue_comps = [_residue_comp(r, tag.name if tag else None, channel_comps) for r in residues]
    ion_comps = {(t, loss): _ion_comp(t, loss) for t, _, loss, _ in names}

    # Element counts: prefix[k] is the first k residues' composition
    elements = sorted({e for c in residue_comps for e in c} | {e for c in ion_comps.values() for e in c})
    column = {e: i for i, e in enumerate(elements)}
    counts = np.zeros((len(residues) + 1, len(elements)), dtype=np.int64)
    for i, comp in enumerate(residue_comps, start=1):
        for e, n in comp.items():
            counts[i, column[e]] = n
    prefix = np.cumsum(counts, axis=0)
    ion_counts = {}
    for ion, comp in ion_comps.items():
        ion_counts[ion] = np.zeros(len(elements), dtype=np.int64)
        for e, n in comp.items():
            ion_counts[ion][column[e]] = n

    new_frags = {}
    for frag, (frag_type, frag_idx, loss, frag_z) in zip(frags, names):
        mz,intensity = frags[frag]
        k = int(frag_idx)
        assert k < n_stripped
        # The residues fragment_seq takes: residues[:k], or residues[-k:]
        if frag_type in 'abc':
            frag_counts = prefix[min(k, len(residues))]
        elif frag_type in 'xyz':
            frag_counts = prefix[-1] - prefix[max(len(residues) - k, 0) if k else 0]
        else:
            raise(ValueError("Invalid ion type"))
        frag_counts = (frag_counts + ion_counts[(frag_type, loss)]).tolist()
        comp = [(e, n) for e, n in zip(elements, frag_counts) if n]

        for iso_idx, (iso_diff, iso_ratio) in enumerate(_envelope(n_iso, int(frag_z), comp)):
            frag_iso = "" if iso_idx==0 else "_iso"+str(iso_idx)
            new_frags[frag+frag_iso] = [mz+iso_diff, intensity*iso_ratio]

    return frag_to_peak(new_frags,return_frags=True)


# ---------------------------------------------------------------------------
# Fragment isotopes for a chunk of library entries in one GIL-free kernel.
#
# JrMod's construction (src/library/isotopes.rs, fragment_isotope_envelopes):
# per peptide, a prefix walk convolves each residue's isotope distribution in
# turn (a/b/c ions) and a suffix walk does the same from the C terminus
# (x/y/z), so every fragment's envelope comes from one pass over the
# residues.  Unlike JrMod, residues carry their modifications and tag
# channels (_residue_comp), and each distribution carries an abundance-
# weighted mass track (isotope_distribution), so isotope m/z offsets are
# exact, not nominal.  Neutral losses are ignored: a loss fragment gets its
# parent ion's envelope.  a and z ions remove atoms, which a walk can not, so
# they are computed from the fragment's exact element counts.
# ---------------------------------------------------------------------------

from numba import njit as _njit
from src import isotope_distribution as _isodist
from src.utils import frag_encoding as _fe

# Ion type codes (frag_encoding._ION_TYPE_TO_INT): b 0, y 1, a 2, c 3, x 4, z 5
_ION_ORDER = ("b", "y", "a", "c", "x", "z")
assert all(_fe._ION_TYPE_TO_INT[t] == i for i, t in enumerate(_ION_ORDER))
# Frag code layout (frag_encoding), as plain ints for the kernel
_ION_SHIFT, _ION_MASK = _fe._ION_SHIFT, _fe._ION_MASK
_IDX_SHIFT, _IDX_MASK = _fe._IDX_SHIFT, _fe._IDX_MASK
_CHG_SHIFT, _CHG_MASK = _fe._CHG_SHIFT, _fe._CHG_MASK
_ISO_SHIFT = _fe._ISO_SHIFT


@_njit(nogil=True)
def _dense_dist(counts, offsets, abundances, shifts, starts, k):
    """(probability, shift sum) of peaks +0..+(k-1) for element counts over
    the element table (counts[e] of table element e)."""
    p = np.zeros(k)
    s = np.zeros(k)
    p[0] = 1.0
    for e in range(counts.shape[0]):
        n = counts[e]
        if n == 0:
            continue
        base_p = np.zeros(k)
        base_s = np.zeros(k)
        for t in range(starts[e], starts[e + 1]):
            o = offsets[t]
            if o < k:
                base_p[o] += abundances[t]
                base_s[o] += abundances[t] * shifts[t]
        acc_p = np.zeros(k)
        acc_s = np.zeros(k)
        acc_p[0] = 1.0
        while n > 0:
            if n & 1:
                acc_p, acc_s = _isodist._convolve(acc_p, acc_s, base_p, base_s, k)
            n >>= 1
            if n > 0:
                base_p, base_s = _isodist._convolve(base_p, base_s, base_p, base_s, k)
        p, s = _isodist._convolve(p, s, acc_p, acc_s, k)
    return p, s


@_njit(nogil=True)
def _conv_into(a_p, a_s, b_p, b_s, out_p, out_s, k):
    """_isodist._convolve written into out_p/out_s (no allocation)."""
    for i in range(k):
        out_p[i] = 0.0
        out_s[i] = 0.0
    for i in range(k):
        if a_p[i] == 0.0 and a_s[i] == 0.0:
            continue
        for j in range(k - i):
            out_p[i + j] += a_p[i] * b_p[j]
            out_s[i + j] += a_p[i] * b_s[j] + a_s[i] * b_p[j]


@_njit(nogil=True)
def _iso_chunk_kernel(rows, res_ids, res_offsets, uniq_counts, n_stripped,
                      frag_offsets, frag_lengths, frag_mz, frag_int, frag_codes,
                      spec_offsets, out_mz, out_int, out_codes,
                      ion_counts, offsets, abundances, shifts, starts, n_iso):
    """Isotope peaks of every fragment of library entries rows[i], written to
    out_* at spec_offsets[rows[i]]: n_iso peaks per fragment, sorted by m/z
    (stable), codes carrying the isotope index.  Entry i's residues are
    uniq_counts[res_ids[res_offsets[i]:res_offsets[i+1]]] (element counts).
    Everything per residue and per fragment reuses buffers sized once."""
    k = n_iso
    n_el = ion_counts.shape[1]
    # Each distinct residue's distribution, once
    n_u = uniq_counts.shape[0]
    u_p = np.zeros((n_u, k))
    u_s = np.zeros((n_u, k))
    for u in range(n_u):
        pp, ss = _dense_dist(uniq_counts[u], offsets, abundances, shifts, starts, k)
        u_p[u] = pp
        u_s[u] = ss
    ion_p = np.zeros((6, k))
    ion_s = np.zeros((6, k))
    for t in range(6):
        if t != 2 and t != 5:
            pp, ss = _dense_dist(ion_counts[t], offsets, abundances, shifts, starts, k)
            ion_p[t] = pp
            ion_s[t] = ss
    max_res = 0
    max_peaks = 0
    for i in range(rows.shape[0]):
        max_res = max(max_res, res_offsets[i + 1] - res_offsets[i])
        max_peaks = max(max_peaks, frag_lengths[rows[i]] * k)
    pre_p = np.zeros((max_res + 1, k))
    pre_s = np.zeros((max_res + 1, k))
    suf_p = np.zeros((max_res + 1, k))
    suf_s = np.zeros((max_res + 1, k))
    fp = np.zeros(k)
    fs = np.zeros(k)
    counts = np.zeros(n_el, dtype=np.int64)
    mz = np.empty(max_peaks)
    inten = np.empty(max_peaks)
    codes = np.empty(max_peaks, dtype=np.int32)
    order = np.empty(max_peaks, dtype=np.int64)

    for i in range(rows.shape[0]):
        row = rows[i]
        r0 = res_offsets[i]
        n_res = res_offsets[i + 1] - r0
        # prefix[j] / suffix[j]: the first / last j residues
        for q in range(k):
            pre_p[0, q] = 0.0
            pre_s[0, q] = 0.0
            suf_p[0, q] = 0.0
            suf_s[0, q] = 0.0
        pre_p[0, 0] = 1.0
        suf_p[0, 0] = 1.0
        for j in range(n_res):
            u = res_ids[r0 + j]
            _conv_into(pre_p[j], pre_s[j], u_p[u], u_s[u], pre_p[j + 1], pre_s[j + 1], k)
            u = res_ids[r0 + n_res - 1 - j]
            _conv_into(suf_p[j], suf_s[j], u_p[u], u_s[u], suf_p[j + 1], suf_s[j + 1], k)

        f0 = frag_offsets[row]
        nf = frag_lengths[row]
        for f in range(nf):
            code = frag_codes[f0 + f]
            ion = (code >> _ION_SHIFT) & _ION_MASK
            idx = (code >> _IDX_SHIFT) & _IDX_MASK
            charge = ((code >> _CHG_SHIFT) & _CHG_MASK) + 1
            if idx >= n_stripped[i]:
                raise ValueError("fragment index beyond the peptide")
            if ion == 2 or ion == 5:
                # a / z remove atoms: from the fragment's exact element counts
                for e in range(n_el):
                    counts[e] = ion_counts[ion, e]
                if ion == 2:
                    lo, hi = 0, min(idx, n_res)
                else:
                    lo, hi = (max(n_res - idx, 0) if idx else 0), n_res
                for j in range(lo, hi):
                    u = res_ids[r0 + j]
                    for e in range(n_el):
                        counts[e] += uniq_counts[u, e]
                dp, ds = _dense_dist(counts, offsets, abundances, shifts, starts, k)
                for q in range(k):
                    fp[q] = dp[q]
                    fs[q] = ds[q]
            elif ion == 0 or ion == 3:
                j = min(idx, n_res)
                _conv_into(pre_p[j], pre_s[j], ion_p[ion], ion_s[ion], fp, fs, k)
            else:
                j = min(idx, n_res) if idx else n_res
                _conv_into(suf_p[j], suf_s[j], ion_p[ion], ion_s[ion], fp, fs, k)
            if fp[0] <= 0:
                raise ValueError("fragment with no monoisotopic peak")
            shift0 = fs[0] / fp[0]
            base_mz = np.float64(frag_mz[f0 + f])
            base_int = np.float64(frag_int[f0 + f])
            for q in range(k):
                shift = fs[q] / fp[q] if fp[q] > 0 else 0.0
                mz[f * k + q] = base_mz + (shift - shift0) / charge
                inten[f * k + q] = base_int * (fp[q] / fp[0])
                codes[f * k + q] = code | (q << _ISO_SHIFT)

        # stable insertion sort of this entry's peaks by m/z
        n_peaks = nf * k
        for q in range(n_peaks):
            order[q] = q
        for q in range(1, n_peaks):
            cur = order[q]
            v = mz[cur]
            w = q - 1
            while w >= 0 and mz[order[w]] > v:
                order[w + 1] = order[w]
                w -= 1
            order[w + 1] = cur
        o = spec_offsets[row]
        for q in range(n_peaks):
            out_mz[o + q] = mz[order[q]]
            out_int[o + q] = inten[order[q]]
            out_codes[o + q] = codes[order[q]]


_iso_tls = threading.local()


def _iso_chunk(library, rows, tag, n_iso, spec_offsets, out_mz, out_int, out_codes):
    """Fill out_* for library entries *rows* (see _iso_chunk_kernel).  The
    Python here is per residue, with per-thread caches; everything per
    fragment runs in the kernel."""
    cache = getattr(_iso_tls, "residues", None)
    if cache is None:
        cache = _iso_tls.residues = {}
    tag_name = tag.name if tag else None
    channel_comps = _hashable_channel_comps(tag)
    uniq, res_ids, res_offsets, n_stripped = {}, [], [0], []
    for row in rows:
        seq = library.mod_seq[row]
        residues = parse_peptide(seq)
        for r in residues:
            u = uniq.get(r)
            if u is None:
                u = uniq[r] = len(uniq)
            res_ids.append(u)
        res_offsets.append(len(res_ids))
        n_stripped.append(len(_BRACKETS.sub("", seq)))
    comps = []
    for r in uniq:
        key = (r, tag_name, channel_comps)
        comp = cache.get(key)
        if comp is None:
            comp = cache[key] = dict(_residue_comp.__wrapped__(r, tag_name, channel_comps))
        comps.append(comp)
    ion_comps = [_ion_comp(t, None) for t in _ION_ORDER]
    symbols = {e for c in comps for e in c} | {e for c in ion_comps for e in c}
    table = _isodist._table_for(sorted(symbols))
    n_el = len(table.index)
    uniq_counts = np.zeros((len(comps), n_el), dtype=np.int64)
    for u, comp in enumerate(comps):
        for e, n in comp.items():
            uniq_counts[u, table.index[e]] = n
    ions = np.zeros((6, n_el), dtype=np.int64)
    for t, comp in enumerate(ion_comps):
        for e, n in comp.items():
            ions[t, table.index[e]] = n
    _iso_chunk_kernel(np.asarray(rows, dtype=np.int64),
                      np.array(res_ids, dtype=np.int64),
                      np.array(res_offsets, dtype=np.int64),
                      uniq_counts,
                      np.array(n_stripped, dtype=np.int64),
                      library.frag_offsets, library.frag_lengths,
                      library.frag_mz, library.frag_int, library.frag_keys_data,
                      spec_offsets, out_mz, out_int, out_codes,
                      ions, table.offsets, table.abundances, table.shifts, table.starts, n_iso)


def _alloc_iso_buffers(library, all_keys, n_iso):
    """Compute the exact output layout and preallocate the flat spectrum arrays.

    ``gen_isotopes_dict`` emits exactly ``n_iso`` peaks per fragment (the
    ``min_iso_intensity`` cut at the top of this module is disabled), so every
    entry's peak count is known before any isotope math runs. That lets the
    isotope pass write straight into preallocated arrays instead of building a
    list of per-entry arrays and concatenating -- which on a multi-million entry
    library costs several times the final payload in transient copies.

    Parameters
    ----------
    library : SpectrumLibraryStore
        Spectral library. Read-only here.
    all_keys : list
        ``list(library)``, in the order the caller will enumerate.
    n_iso : int
        Number of isotopes per fragment.

    Returns
    -------
    tuple
        ``(spec_offsets, spec_lengths, spectrum_mz, spectrum_int, frag_names_data)``
    """
    n = len(all_keys)

    idxs = np.fromiter((library.key_to_idx[k] for k in all_keys), dtype=np.int64, count=n)
    if n and not np.array_equal(idxs, np.arange(n, dtype=np.int64)):
        raise RuntimeError(
            "Library iteration order does not match store row order; "
            "spectrum offsets would be written to the wrong entries."
        )

    lengths64 = library.frag_lengths[idxs].astype(np.int64) * n_iso
    total = int(lengths64.sum())

    spec_offsets = np.zeros(n, dtype=np.int64)
    if n > 1:
        np.cumsum(lengths64[:-1], out=spec_offsets[1:])
    spec_lengths = lengths64.astype(np.int32)

    return (spec_offsets,
            spec_lengths,
            np.empty(total, dtype=np.float32),
            np.empty(total, dtype=np.float32),
            np.empty(total, dtype=np.int32))


def _store_iso_entry(i, spectrum, ordered_frags, spec_offsets, spec_lengths,
                     spectrum_mz, spectrum_int, frag_names_data, key):
    """Write one entry's isotope peaks into the preallocated arrays."""
    length = int(spec_lengths[i])
    if len(spectrum) != length:
        raise RuntimeError(
            f"Entry {i} {key!r} produced {len(spectrum)} peaks, expected {length}. "
            "Preallocation requires the min_iso_intensity filter in "
            "gen_isotopes_dict to stay disabled."
        )
    off = int(spec_offsets[i])
    spec = np.asarray(spectrum, dtype=np.float64)
    spectrum_mz[off:off + length] = spec[:, 0]
    spectrum_int[off:off + length] = spec[:, 1]
    frag_names_data[off:off + length] = ordered_frags


def iso_library(library, tag, n_iso):
    """
    Generate isotopes for library fragments (single-threaded).

    Only rebuilds spectrum_data and frag_names_data on the store.

    Parameters
    ----------
    library : SpectrumLibraryStore
        Spectral library.
    tag : massTag
        A massTag instance.
    n_iso : int
        Number of isotopes per fragment to generate.

    Returns
    -------
    SpectrumLibraryStore
        The same store with updated spectrum/frag_names arrays.
    """
    from src.utils.frag_encoding import encode_frag_names

    all_keys = list(library)

    (spec_offsets, spec_lengths,
     spectrum_mz, spectrum_int, frag_names_data) = _alloc_iso_buffers(library, all_keys, n_iso)

    # The old spectra are never read (isotopes are rebuilt from frags), so
    # release them before filling the n_iso-times-larger replacements
    library.spectrum_mz = None
    library.spectrum_int = None
    library.frag_names_data = None
    library.spectrum_perm = None  # iso rebuilds spectra; acts as the finalizer

    logger.info("Generating isotopes for library:")
    for i, key in enumerate(tqdm.tqdm(all_keys)):
        frags = library[key]["frags"]
        spectrum, ordered_frags = gen_isotopes_dict(key[0], frags, tag, n_iso)
        _store_iso_entry(i, spectrum, encode_frag_names(ordered_frags),
                         spec_offsets, spec_lengths,
                         spectrum_mz, spectrum_int, frag_names_data, key)

    library.spectrum_mz = spectrum_mz
    library.spectrum_int = spectrum_int
    library.frag_names_data = frag_names_data
    library.spectrum_offsets = spec_offsets
    library.spectrum_lengths = spec_lengths

    return library

def iso_library_multi(library, tag, n_iso, n_workers):
    """
    Generate isotopes for library fragments, in parallel.

    Only rebuilds spectrum_data and frag_names_data on the store.
    All other arrays (frag_data, frag_keys_data, scalars) are unchanged.

    Fragments get their isotope envelopes from _iso_chunk_kernel (JrMod's
    per-peptide prefix/suffix walk; neutral losses take their parent ion's
    envelope).  Entries are processed in chunks by a thread pool; each chunk
    writes its own entries' slices of the preallocated arrays.  The kernel
    runs without the GIL, so the threads overlap on any Python build, and
    fully on free-threaded Python.

    Parameters
    ----------
    library : SpectrumLibraryStore
        Spectral library.
    tag : massTag
        A massTag instance.
    n_iso : int
        Number of isotopes per fragment to generate.
    n_workers : int
        Threads to generate them with (--iso_workers), at most one per CPU.

    Returns
    -------
    SpectrumLibraryStore
        The same store with updated spectrum/frag_names arrays.
    """
    n = len(library)
    all_keys = list(library)

    (spec_offsets, spec_lengths,
     spectrum_mz, spectrum_int, frag_names_data) = _alloc_iso_buffers(library, all_keys, n_iso)

    # The old spectra are never read (isotopes are rebuilt from frags), so
    # release them before filling the n_iso-times-larger replacements
    library.spectrum_mz = None
    library.spectrum_int = None
    library.frag_names_data = None
    library.spectrum_perm = None  # iso rebuilds spectra; acts as the finalizer

    logger.info("Generating isotopes for library:")
    chunksize = 1000
    chunks = [range(i, min(i + chunksize, n)) for i in range(0, n, chunksize)]
    n_workers = max(1, min(os.cpu_count() or 1, n_workers))
    with ThreadPoolExecutor(n_workers) as pool, tqdm.tqdm(total=n) as bar:
        futures = {pool.submit(_iso_chunk, library, chunk, tag, n_iso, spec_offsets,
                               spectrum_mz, spectrum_int, frag_names_data): len(chunk)
                   for chunk in chunks}
        for f in as_completed(futures):
            f.result()
            bar.update(futures[f])

    library.spectrum_mz = spectrum_mz
    library.spectrum_int = spectrum_int
    library.frag_names_data = frag_names_data
    library.spectrum_offsets = spec_offsets
    library.spectrum_lengths = spec_lengths

    return library


# def calculate_mz(sequence,charge):
    
#     split_seq = split_peptide(sequence)
    
#     seq_comp = get_seq_comp(split_seq, "M")
#     return mass.calculate_mass(seq_comp,charge=charge)


def precursor_isotopes(sequence,charge,tags,n_isotopes=2, decoys=True):
    """
    Return a list of isotope Peak objects (src.isotope_distribution): Peak(p.mz, p.intensity, p.charge)

    Parameters
    ----------
    sequence : str
        Peptide sequence including tags and PTMs
    charge : int or float
        Peptide Charge
    tag : massTag
        massTag Object
    n_isotopes : int
        The number of isotopes to be returned
    decoys: bool
        True by default, can be set to false if there will be no decoys passed into func

    Returns
    -------
    isotopes : list of isotope Peaks
        i.e. Peak(p.mz, p.intensity, p.charge), Peak(p.mz, p.intensity, p.charge)]
    
    """
    if decoys:
        sequence = re.sub("Decoy_","",sequence)
    #split_seq = split_peptide(sequence)
    split_seq = parse_peptide(sequence)
    
    seq_comp = get_seq_comp(split_seq, "M", neutral_loss=None)
    
    if tags:
        for tag in tags:
            if tag is not None:
                pattern_tag = re.compile(rf"\(({tag.name}.*?)\)")
                pattern_channel = re.compile(rf"{tag.name}-(\d+)")
        
                matched_tags = [t for aa in split_seq for t in pattern_tag.findall(aa)]
                if tag.channel_comp is not None and len(matched_tags)>0:
                        tag_comp = reduce(lambda x, y: x + y, [tag.channel_comp[pattern_channel.findall(t)[0]] for t in matched_tags])
                        seq_comp+=tag_comp
            
    
    isotopes = isotopic_variants(seq_comp,
                                 npeaks=n_isotopes,
                                 charge = int(charge))
    
    
    return isotopes 

####################################################################################
##################   PLexDIA  code    ##########################################
####################################################################################
####################################################################################


# def iso_distr(temp):
#     hydrogen = int(temp[1])

#     carbon = int(temp[0])

#     nitrogen = int(temp[2])

#     oxygen = int(temp[3])

#     sulfur = int(temp[4])

#     pH = [0.999885, 0.0001157]
#     pC = [0.9893, 0.0107]
#     pN = [0.99632, 0.00368]
#     pO = [0.99757, 0.00038, 0.00205]
#     pS = [0.9493, 0.0076, 0.0429, 0.0002]

#     p = convolve(carbon, pC)
#     p = np.convolve(p, convolve(oxygen, pO))
#     p = np.convolve(p, convolve(hydrogen, pH))
#     p = np.convolve(p, convolve(nitrogen, pN))
#     p = np.convolve(p, convolve(sulfur, pS))
    
#     iso = np.array(cut(p / np.max(p)),dtype="float64")
#     return iso

# def my_iso_distr(comp):
#     hydrogen = int(comp["H"])

#     carbon = int(comp["C"])

#     nitrogen = int(comp["N"])

#     oxygen = int(comp["O"])

#     sulfur = int(comp["S"])

#     pH = [0.999885, 0.0001157]
#     pC = [0.9893, 0.0107]
#     pN = [0.99632, 0.00368]
#     pO = [0.99757, 0.00038, 0.00205]
#     pS = [0.9493, 0.0076, 0.0429, 0.0002]

#     p = convolve(carbon, pC)
#     p = np.convolve(p, convolve(oxygen, pO))
#     p = np.convolve(p, convolve(hydrogen, pH))
#     p = np.convolve(p, convolve(nitrogen, pN))
#     p = np.convolve(p, convolve(sulfur, pS))
    
#     iso = np.array(cut(p / np.max(p)),dtype="float64")
#     return iso


# def bits1(n):
#     b = []
#     while n:
#         b = [n & 1] + b
#         n >>= 1
#     return b or [0]


# def convolve(number, probability):
#     bitarray = bits1(number)
#     pi = probability
#     p = [1]
#     for i, b in enumerate(bitarray[::-1]):
#         p = cut(np.convolve(p, pi)) if b == 1 else p
#         pi = cut(np.convolve(pi, pi))

#     return p


# def cut(array,tr=0.00001):

#     index = np.where(array > tr)[0][-1]

#     if (len(array) > index):
#         return array[:index + 1]
#     else:
#         return (array)
      


