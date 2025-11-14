"""
This Source Code Form is subject to the terms of the Oxford Nanopore
Technologies, Ltd. Public License, v. 1.0.  Full licence can be found
at https://github.com/ParallelSquared/JMod/blob/main/LICENSE.txt
"""


from brainpy import isotopic_variants
import re
from pyteomics import mass
import src.config as config
import tqdm
import os
from functools import reduce
import copy
import numpy as np

from src.utils.parse_peptides import parse_peptide, split_frag_name

from src.utils.misc_functions import frag_to_peak

from src.logger import logger

    
### First get the AA sequence and modifications of the fragment
def fragment_seq(peptide, ion_type):
    
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


def split_peptide(peptide):
    
    return re.findall("([A-Z](?:\(.*?\))?)",peptide)

### all unimod modifications are stored here
unimods = mass.Unimod()

# [i["title"] for i in unimods.mods[:10]]

# unimods.by_id(4)["composition"]


## ## get the compostion of the fragment

mod_pattern = re.compile(r"\([A-z]+\:(\d+)\)")
def get_seq_comp(split_seq,ion_type):
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
    mods = [int(j) for i in split_seq for j in mod_pattern.findall(i) if len(i)>1]
    # tags = [t for aa in split_seq for t in re.findall("(\(.*?\))",aa)]
    seq_comp = mass.Composition(sequence=stripped_seq,ion_type=ion_type)
    for unimod_idx in mods:
        seq_comp += unimods.by_id(unimod_idx)["composition"]
    return seq_comp




import copy

def frag_isotope(frag,seq):
    # mz,intensity = frags[frag]
    split_frag_seq,frag_info = fragment_seq(seq,frag)
    loss = "-"+frag_info[2] if frag_info[2] else frag_info[2]
    ion_type = frag_info[0] + loss
    frag_comp = get_seq_comp(split_frag_seq, ion_type)
    
    isotopes = isotopic_variants(frag_comp,
                                 npeaks=config.num_iso_peaks,
                                 charge = int(frag_info[3]))
    mono_iso_peak = isotopes[0]
    return isotopes


def gen_isotopes(seq,frags):
    new_frags = []
    for frag in frags:
        mz,intensity = frags[frag]
        split_frag_seq,frag_info = fragment_seq(seq,frag)
        loss = "-"+frag_info[2] if frag_info[2] else frag_info[2]
        ion_type = frag_info[0] + loss
        frag_comp = get_seq_comp(split_frag_seq, ion_type)
        
        isotopes = isotopic_variants(frag_comp,
                                     npeaks=config.num_iso_peaks,
                                     charge = int(frag_info[3]))
        mono_iso_peak = isotopes[0]
        for iso in isotopes:
            new_intensity = intensity*(iso.intensity/mono_iso_peak.intensity)
            if True:#new_intensity > config.min_iso_intensity:
                new_frags.append([iso.mz,new_intensity])
    
    new_frags = np.array(new_frags)
    sorted_frags = new_frags[np.argsort(new_frags[:,0])]
    return sorted_frags/[1,np.max(np.array(new_frags)[:,1])]

def gen_isotopes_dict(seq,frags, tag = None):
    new_frags = {}
    for frag in frags:
        mz,intensity = frags[frag]
        split_frag_seq,frag_info = fragment_seq(seq,frag)
        loss = "-"+frag_info[2] if frag_info[2] else frag_info[2]
        ion_type = frag_info[0] + loss
        frag_comp = get_seq_comp(split_frag_seq, ion_type)
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
                                     npeaks=config.num_iso_peaks,
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

def iso_library(library):
    ## add n isotpic peaks to the "spectrum" portio of each library entry
    logger.info("Creating Copy of Library...")
    new_library = copy.deepcopy(library)
    
    logger.info("Generating isotopes for library:")
    for key in tqdm.tqdm(new_library):
        frags = new_library[key]["frags"]
        
        # new_library[key]["spectrum"] = gen_isotopes(key[0],frags)
        new_library[key]["spectrum"],new_library[key]["ordered_frags"] = gen_isotopes_dict(key[0],frags)
        
    return new_library

import multiprocessing
def iso_library_multi(library):
    ## add n isotpic peaks to the "spectrum" portio of each library entry
    logger.info("Creating Copy of Library...")
    new_library = copy.deepcopy(library)
    
    logger.info("Generating isotopes for library:")
    all_keys = list(new_library)
    all_seqs = [i[0] for i in all_keys]
    all_frags = [new_library[i]["frags"] for i in new_library]
    with multiprocessing.Pool(8) as p:
        iso_out = p.starmap(gen_isotopes_dict,tqdm.tqdm(zip(all_seqs,all_frags),total=len(all_seqs)))
    for key,out in zip(all_keys,iso_out):
        new_library[key]["spectrum"],new_library[key]["ordered_frags"] = out
        
        # new_library[key]["spectrum"] = gen_isotopes(key[0],frags)
        # new_library[key]["spectrum"],new_library[key]["ordered_frags"] = gen_isotopes_dict(key[0],frags)
        
    return new_library


def calculate_mz(sequence,charge):
    
    split_seq = split_peptide(sequence)
    
    seq_comp = get_seq_comp(split_seq, "M")
    return mass.calculate_mass(seq_comp,charge=charge)


def precursor_isotopes(sequence,charge,tag,n_isotopes=2, decoys=True):
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
    
    seq_comp = get_seq_comp(split_seq, "M")
    
    if tag:
        pattern_tag = re.compile(rf"\(({tag.name}.*?)\)")
        pattern_channel = re.compile(rf"{tag.name}-(\d+)")

        tags = [t for aa in split_seq for t in pattern_tag.findall(aa)]
        if tag.channel_comp is not None and len(tags)>0:
                tag_comp = reduce(lambda x, y: x + y, [tag.channel_comp[pattern_channel.findall(t)[0]] for t in tags])
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
      


