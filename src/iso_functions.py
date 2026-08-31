
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

from brainpy import isotopic_variants
import re
from pyteomics import mass
import src.config as config
import tqdm
import os
import copy
from functools import reduce
import copy
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

### all unimod modifications are stored here
unimods = mass.Unimod()

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
        seq_comp += unimods.by_id(unimod_idx)["composition"]
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

def gen_isotopes_dict(seq,frags, tag, n_iso):
    new_frags = {}
    for frag in frags:
        mz,intensity = frags[frag]
        split_frag_seq,frag_info = fragment_seq(seq,frag)
        frag_comp = get_seq_comp(split_frag_seq, frag_info[0], neutral_loss=frag_info[2])
        frag_z = int(frag_info[3])
        
        
        if tag:
            tags = [t for aa in split_frag_seq for t in re.findall(f"\(({tag.name}.*?)\)",aa)]
            tag_mz = np.sum([tag.mass_dict[t] for t  in tags])/frag_z
            if tag.channel_comp is not None and len(tags)>0:
                tag_comp = reduce(lambda x, y: x + y, [tag.channel_comp[re.findall(f"{tag.name}-(\d+)",t)[0]] for t in tags])
                frag_comp+=tag_comp
        else:
            tag_mz = 0
        
        isotopes = isotopic_variants(frag_comp,
                                     npeaks=n_iso,
                                     charge = frag_z)
        mono_iso_peak = isotopes[0]
        for iso_idx,iso in enumerate(isotopes):
            new_intensity = intensity*(iso.intensity/mono_iso_peak.intensity)
            if True:#new_intensity > config.min_iso_intensity:
                frag_iso = "" if iso_idx==0 else "_iso"+str(iso_idx)
                # new_frags[frag+frag_iso] = [mz+iso_diff+tag_mz,new_intensity]
                iso_diff = iso.mz - mono_iso_peak.mz
                new_frags[frag+frag_iso] = [mz+iso_diff,new_intensity]
                
    return frag_to_peak(new_frags,return_frags=True)


def _gen_isotopes_encoded(args):
    """Wrapper that encodes frag names to int32 inside the worker process."""
    seq, frags, tag, n_iso = args
    from src.utils.frag_encoding import encode_frag_names
    spectrum, ordered_frags = gen_isotopes_dict(seq, frags, tag, n_iso)
    return spectrum, encode_frag_names(ordered_frags)


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

    n = len(library)
    all_keys = list(library)

    all_spec_peaks = []
    all_frag_codes = []
    spec_offsets = np.empty(n, dtype=np.int64)
    spec_lengths = np.empty(n, dtype=np.int32)
    cursor = 0

    logger.info("Generating isotopes for library:")
    for i, key in enumerate(tqdm.tqdm(all_keys)):
        frags = library[key]["frags"]
        spectrum, ordered_frags = gen_isotopes_dict(key[0], frags, tag, n_iso)
        n_peaks = len(spectrum)
        spec_offsets[i] = cursor
        spec_lengths[i] = n_peaks
        all_spec_peaks.append(np.asarray(spectrum, dtype=np.float64))
        all_frag_codes.append(encode_frag_names(ordered_frags))
        cursor += n_peaks

    library.spectrum_data = np.concatenate(all_spec_peaks, axis=0) if all_spec_peaks else np.empty((0, 2), dtype=np.float64)
    library.frag_names_data = np.concatenate(all_frag_codes) if all_frag_codes else np.empty(0, dtype=np.int32)
    library.spectrum_offsets = spec_offsets
    library.spectrum_lengths = spec_lengths

    return library

import multiprocessing
def _iso_arg_gen(library, all_keys, tag, n_iso):
    """Lazily yield (seq, frags, tag, n_iso) tuples one at a time.

    This avoids materializing all fragment dicts in memory at once.
    """
    for k in all_keys:
        yield (k[0], library[k]["frags"], tag, n_iso)


def iso_library_multi(library, tag, n_iso):
    """
    Generate isotopes for library fragments (multiprocessed).

    Only rebuilds spectrum_data and frag_names_data on the store.
    All other arrays (frag_data, frag_keys_data, scalars) are unchanged.

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

    n = len(library)
    all_keys = list(library)

    logger.info("Generating isotopes for library:")
    p = multiprocessing.get_context('spawn').Pool(min(multiprocessing.cpu_count(), 61),
                                                  initializer=config.limit_blas_threads)

    # Process results incrementally via imap to avoid holding all inputs
    # and outputs in memory simultaneously.
    all_spec_peaks = []
    all_frag_codes = []
    spec_offsets = np.empty(n, dtype=np.int64)
    spec_lengths = np.empty(n, dtype=np.int32)
    cursor = 0

    chunksize = 1000
    arg_gen = _iso_arg_gen(library, all_keys, tag, n_iso)
    for i, (spectrum, ordered_frags) in enumerate(
        tqdm.tqdm(p.imap(_gen_isotopes_encoded, arg_gen, chunksize=chunksize), total=n)
    ):
        n_peaks = len(spectrum)
        spec_offsets[i] = cursor
        spec_lengths[i] = n_peaks
        all_spec_peaks.append(np.asarray(spectrum, dtype=np.float64))
        all_frag_codes.append(ordered_frags)
        cursor += n_peaks

    p.close()
    p.join()

    spectrum_data = np.concatenate(all_spec_peaks, axis=0) if all_spec_peaks else np.empty((0, 2), dtype=np.float64)
    del all_spec_peaks
    frag_names_data = np.concatenate(all_frag_codes) if all_frag_codes else np.empty(0, dtype=np.int32)
    del all_frag_codes

    # Replace the spectrum arrays on the store.
    # The old arrays are dereferenced and freed by GC.
    library.spectrum_data = spectrum_data
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
    Return a list of brainpy theoretical peak objects: Peak(p.mz, p.intensity, p.charge)

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
    isotopes : list of brainpy theoretical peaks
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
      


