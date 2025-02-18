

from brainpy import isotopic_variants
import re
from pyteomics import mass
import config
import tqdm
import os

import copy
import numpy as np

from miscFunctions import frag_to_peak



## split up the fragment name (b/y)(frag index)(-loss)_charge
def split_frag_name(ion_type):
    frag_name,frag_z = ion_type.split("_")
    loss_check = frag_name.split("-")
    loss = ""
    if len(loss_check)>1:
        frag_name,loss = loss_check
    frag_type = frag_name[0]
    frag_idx = int(frag_name[1:])
    
    return frag_type,frag_idx,loss,frag_z
    
### First get the AA sequence and modifications of the fragment
def fragment_seq(peptide, ion_type):
    
    peptide = "".join(peptide)
    split_peptide = re.findall("([A-Z](?:\(.*?\))?)",peptide)
    
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

def get_seq_comp(split_seq,ion_type):
    
    stripped_seq = "".join([i[0] for i in split_seq]) ## assumes AA comes first before mods
    
    mods = [int(j) for i in split_seq for j in re.findall("\([A-z]+\:(\d+)\)",i) if len(i)>1]
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


def gen_isotopes_dict(seq,frags):
    new_frags = {}
    for frag in frags:
        mz,intensity = frags[frag]
        split_frag_seq,frag_info = fragment_seq(seq,frag)
        loss = "-"+frag_info[2] if frag_info[2] else frag_info[2]
        ion_type = frag_info[0] + loss
        frag_comp = get_seq_comp(split_frag_seq, ion_type)
        frag_z = int(frag_info[3])
        
        
        if config.tag:
            tags = [t for aa in split_frag_seq for t in re.findall(f"\(({config.tag.name}.*?)\)",aa)]
            tag_mz = np.sum([config.tag.mass_dict[t] for t  in tags])/frag_z
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
                new_frags[frag+frag_iso] = [iso.mz+tag_mz,new_intensity]
                
    return frag_to_peak(new_frags,return_frags=True)

def iso_library(library):
    ## add n isotpic peaks to the "spectrum" portio of each library entry
    print("Creating Copy of Library...")
    new_library = copy.deepcopy(library)
    
    print("Generating isotopes for library:")
    for key in tqdm.tqdm(new_library):
        frags = new_library[key]["frags"]
        
        # new_library[key]["spectrum"] = gen_isotopes(key[0],frags)
        new_library[key]["spectrum"],new_library[key]["ordered_frags"] = gen_isotopes_dict(key[0],frags)
        
    return new_library

import multiprocessing
def iso_library_multi(library):
    ## add n isotpic peaks to the "spectrum" portio of each library entry
    print("Creating Copy of Library...")
    new_library = copy.deepcopy(library)
    
    print("Generating isotopes for library:")
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

def precursor_isotopes(sequence,charge,n_isotopes=2):
    
    split_seq = split_peptide(sequence)
    seq_comp = get_seq_comp(split_seq, "M")
    
    isotopes = isotopic_variants(seq_comp,
                                 npeaks=n_isotopes,
                                 charge = int(charge))
    
    return isotopes







####################################################################################
##################   PLexDIA  code    ##########################################
####################################################################################
####################################################################################


def iso_distr(temp):
    hydrogen = int(temp[1])

    carbon = int(temp[0])

    nitrogen = int(temp[2])

    oxygen = int(temp[3])

    sulfur = int(temp[4])

    pH = [0.999885, 0.0001157]
    pC = [0.9893, 0.0107]
    pN = [0.99632, 0.00368]
    pO = [0.99757, 0.00038, 0.00205]
    pS = [0.9493, 0.0076, 0.0429, 0.0002]

    p = convolve(carbon, pC)
    p = np.convolve(p, convolve(oxygen, pO))
    p = np.convolve(p, convolve(hydrogen, pH))
    p = np.convolve(p, convolve(nitrogen, pN))
    p = np.convolve(p, convolve(sulfur, pS))
    
    iso = np.array(cut(p / np.max(p)),dtype="float64")
    return iso

def my_iso_distr(comp):
    hydrogen = int(comp["H"])

    carbon = int(comp["C"])

    nitrogen = int(comp["N"])

    oxygen = int(comp["O"])

    sulfur = int(comp["S"])

    pH = [0.999885, 0.0001157]
    pC = [0.9893, 0.0107]
    pN = [0.99632, 0.00368]
    pO = [0.99757, 0.00038, 0.00205]
    pS = [0.9493, 0.0076, 0.0429, 0.0002]

    p = convolve(carbon, pC)
    p = np.convolve(p, convolve(oxygen, pO))
    p = np.convolve(p, convolve(hydrogen, pH))
    p = np.convolve(p, convolve(nitrogen, pN))
    p = np.convolve(p, convolve(sulfur, pS))
    
    iso = np.array(cut(p / np.max(p)),dtype="float64")
    return iso


def bits1(n):
    b = []
    while n:
        b = [n & 1] + b
        n >>= 1
    return b or [0]


def convolve(number, probability):
    bitarray = bits1(number)
    pi = probability
    p = [1]
    for i, b in enumerate(bitarray[::-1]):
        p = cut(np.convolve(p, pi)) if b == 1 else p
        pi = cut(np.convolve(pi, pi))

    return p


def cut(array,tr=0.00001):

    index = np.where(array > tr)[0][-1]

    if (len(array) > index):
        return array[:index + 1]
    else:
        return (array)
      


