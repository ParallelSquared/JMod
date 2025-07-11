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
import numpy.typing as npt
import brainpy as bp
#$from . import config 

from .misc_functions import frag_to_peak



## split up the fragment name (b/y)(frag index)(-loss)_charge
def split_frag_name(ion_type: str) -> tuple[str, int, str, int]:
    """
    Splits the string name for an ion into a tuple (ion type, fragment index, loss, charge)

    Parameters
    ----------
    ion_type : str
        The name for the ion, e.g. 'y6_1' or 'b3-H2O_2'

    Returns
    -------
    tuple[str, int, str, int]
       (ion type, fragment index, loss, charge)
    
    Example
    -------
    >>> ion_type = 'y6_1'
    >>> split_frag_name(ion_type)
    ('y', 6, '', 1)
    """
    frag_name,frag_z = ion_type.split("_")
    loss_check = frag_name.split("-")
    loss = ""
    if len(loss_check)>1:
        frag_name,loss = loss_check
    frag_type = frag_name[0]
    frag_idx = int(frag_name[1:])
    
    return frag_type,frag_idx,loss,frag_z

def parse_peptide(seq):
    """
    Splits a peptide sequence string into a list of single amino acid strings possibly containing modifications 

    Parameters
    ----------
    seq : str
        The original peptide sequence 

    Returns
    -------
    list[str] the amino acid sequence with modifications as separate strings
    
    Example
    -------
    >>> parse_peptide("PEP(+10.0)TID[another mod in square brackets]E")
    ['P', 'E', 'P(+10.0)', 'T', 'I', 'D[another mod in square brackets]', 'E']
    """
    close_d = {"[": "]", "(": ")"}
    open_set = set(close_d.keys())
    close_set = set(close_d.values())
    
    new_seq = []
    current = ""
    s_idx = 0

    while s_idx < len(seq):
        s = seq[s_idx]

        if s in open_set:
            # Begin collecting the bracketed modification
            opener = s
            closer = close_d[opener]
            mod = s
            stack = [closer]
            s_idx += 1

            while s_idx < len(seq) and stack:
                c = seq[s_idx]
                mod += c

                if c in open_set:
                    stack.append(close_d[c])
                elif c in close_set:
                    if stack and c == stack[-1]:
                        stack.pop()
                s_idx += 1

            current += mod  # Append full modification to current letter

        elif s.isalpha():
            if current:
                new_seq.append(current)
            current = s
            s_idx += 1

        else:
            # If somehow an unexpected char, just add it
            current += s
            s_idx += 1

    if current:
        new_seq.append(current)

    return new_seq
    
### First get the AA sequence and modifications of the fragment
def fragment_seq(peptide: str, ion_type: str) -> tuple[list[str], list[str]]:
    """
    Splits a peptide sequence string into a list of single amino acid strings possibly containing modifications 

    Parameters
    ----------
    peptide : str
        The original peptide sequence 
    ion_type : str
        The name for the ion, e.g. 'y6_1' or 'b3-H2O_2'

    Returns
    -------
    tuple[list[str], list[str]] First element is the amino acid sequence with modifications as separate strings,
    for the fragment specied by `ion_type`. The second element is a list containing the ion type, fragment index, 
    loss, and charge as strings.
    
    Example
    -------
    >>> peptide
    'PEP(+10.0)TIDE'
    >>> ion_type
    'y5_2'
    >>> fragment_seq(peptide, ion_type)
    (['P(+10.0)', 'T', 'I', 'D', 'E'], ['y', 5, '', '2'])
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

### all unimod modifications are stored here
unimods = mass.Unimod()

# [i["title"] for i in unimods.mods[:10]]

# unimods.by_id(4)["composition"]


## ## get the compostion of the fragment

def get_seq_comp(split_seq: list[str], ion_type: str) -> mass.Composition:
    """
    Gets the elemental composition of a peptide fragment sequence, including modifications.

    Parameters
    ----------
    split_seq : str
        Sequence of the fragment ast list of amino acids 
    ion_type : str
        Whether the ion is b/y etc. 

    Returns
    -------
    Elemental composition of the fragment as a `pyteomics.mass.Composition` object.
    Composition({'H': 53, 'C': 31, 'O': 11, 'N': 11})
    Type: <class 'pyteomics.mass.mass.Composition'>

    
    Example
    -------
    >>> splie_seq
   ['E', 'Q', 'A', 'I', 'S', 'V', 'R']
    >>> ion_type
    'y'
    >>> get_seq_comp(split_seq, ion_type)
    Composition({'H': 59, 'C': 33, 'O': 12, 'N': 11})
    """
    stripped_seq = "".join([i[0] for i in split_seq]) ## assumes AA comes first before mods
    
    mods = [int(j) for i in split_seq for j in re.findall("\([A-z]+\:(\d+)\)",i) if len(i)>1]
    # tags = [t for aa in split_seq for t in re.findall("(\(.*?\))",aa)]
    seq_comp = mass.Composition(sequence=stripped_seq,ion_type=ion_type)
    for unimod_idx in mods:
        seq_comp += unimods.by_id(unimod_idx)["composition"]

    return seq_comp

import copy

def gen_isotopes_dict(seq: str,frags: dict[str,list[float]], tag = None) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.string_]]:
    """
    Gets the elemental composition of a peptide fragment sequence, including modifications.

    Parameters
    ----------
    seq : str
        Peptide sequence 
    frags : str
        Dictionary of fragment names and their corresponding m/z and intensity values.
    tag : Tag, optional

    Returns
    -------
    Tuple of numpy arrays. First array has two columns: m/z and intensity of the isotopic peaks.
    The second array contains the names of the fragments with isotopic peaks.

    Example
    -------
    >>> seq
    'ELYAQFLR'
    >>> frags:
    {'b3_1': [406.19726206442, 0.12509464],
    'b4_1': [477.23437584913, 0.06771548],
    'b5_1': [605.29295335441, 0.011354789],
    'y3_1': [435.27143006419, 0.17212126],
    'y4_1': [563.33000756947, 0.27101088],
    'y5_1': [634.36712135418, 1.0],
    'y6_1': [797.43044988673, 0.8603392],
    'y7_1': [910.51451386386, 0.41480112]}
    >>> gen_isotopes_dict(seq,frags)
    (array([[4.06197262e+02, 1.25094640e-01],
        [4.07200367e+02, 2.91051908e-02],
        [4.35271430e+02, 1.72121260e-01],
        [4.36274291e+02, 4.38021471e-02],
        [4.77234376e+02, 6.77154800e-02],
        [4.78237437e+02, 1.82643442e-02],
        [5.63330008e+02, 2.71010880e-01],
        [5.64332831e+02, 8.60598922e-02],
        [6.05292953e+02, 1.13547890e-02],
        [6.06295941e+02, 3.77874933e-03],
        [6.34367121e+02, 1.00000000e+00],
        [6.35369941e+02, 3.54607904e-01],
        [7.97430450e+02, 8.60339200e-01],
        [7.98433348e+02, 3.93518915e-01],
        [9.10514514e+02, 4.14801120e-01],
        [9.11517436e+02, 2.18846360e-01]]),
    array(['b3_1', 'b3_1_iso1', 'y3_1', 'y3_1_iso1', 'b4_1', 'b4_1_iso1',
        'y4_1', 'y4_1_iso1', 'b5_1', 'b5_1_iso1', 'y5_1', 'y5_1_iso1',
        'y6_1', 'y6_1_iso1', 'y7_1', 'y7_1_iso1'], dtype='<U9'))
    """
    new_frags = {}
    #Get isotopes for each fragment 
    for frag in frags:

        frag_mono_mz,intensity = frags[frag]
        split_frag_seq,frag_info = fragment_seq(seq,frag)
        loss = "-"+frag_info[2] if frag_info[2] else frag_info[2]
        ion_type = frag_info[0] + loss

        #Elemental composition of the fragment and charge 
        frag_comp = get_seq_comp(split_frag_seq, ion_type)
        frag_z = int(frag_info[3])
        
        #Get elemental composition of the tag 
        if tag:
            #Get tags in the fragment 
            tags = [t for aa in split_frag_seq for t in re.findall(f"\(({tag.name}.*?)\)",aa)]

            #Add elemental composition of the tag to the fragment 
            #If the tag has a fixed isotopic distribution, there is no 'channel_comp' 
            if tag.channel_comp is not None and len(tags)>0:
                tag_comp = reduce(lambda x, y: x + y, [tag.channel_comp[re.findall(f"{tag.name}-(\d+)",t)[0]] for t in tags])
                frag_comp+=tag_comp
        
        #Calculate isotopic masses given the 
        isotopes = isotopic_variants(frag_comp,
                                     npeaks=config.num_iso_peaks,
                                     charge = frag_z)

        #Mono-isotopic peak 
        mono_iso_peak = isotopes[0]
        #Add each isotope 
        for iso_idx,iso in enumerate(isotopes):
            #Relative intensity of the current isotopic peak 
            new_intensity = intensity*(iso.intensity/mono_iso_peak.intensity)
            #Name of the fragment isotope 
            frag_iso = "" if iso_idx==0 else "_iso"+str(iso_idx)
            #Difference in m/z between the isotopic peak and the mono-isotopic peak
            iso_mz_diff = iso.mz - mono_iso_peak.mz
            new_frags[frag+frag_iso] = [frag_mono_mz+iso_mz_diff,new_intensity]

    return frag_to_peak(new_frags,return_frags=True)

import multiprocessing
def iso_library_multi(library: dict) -> dict:
    """
    Adds fragment isotopic peaks to each entry in the spectral library.

    Parameters
    ----------
    library : dict
        Spectral library where each key is a tuple (peptide sequence, charge) and the value is a dictionary of data for that precursor 

    Returns
    -------
    A library formatted and keyed just as the input library but with isotopic peaks added to the "spectrum" key of each entry.

    Example
    -------
    >>> library.items()[:1]
    Structure Example: [(('AAAEQAISVR', 2.0), {'mod_seq': 'AAAEQAISVR', 'seq': 'AAAEQAISVR', 'prec_mz': 508.28018033366, 'prec_z': 2.0, 'iRT': 0.296130418777466, 'frags': {'y7_1': [802.44174284642, 1.0], 'y8_1': [873.47885663113, 0.8503218], 'y6_1': [673.39914975845, 0.5639957], 'y3_1': [361.21939449133, 0.51684165], 'y9_1': [944.51597041584, 0.49269193], 'b3_1': [214.1186178209, 0.48869178], 'y5_1': [545.34057225317, 0.35331511], 'b4_1': [343.16121090887, 0.29596102], 'y4_1': [474.30345846846, 0.22480424], 'b6_1': [542.25690219886, 0.1957216], 'b5_1': [471.21978841415, 0.13230422]}, 'protein_group': 'Q01780', 'protein_name': 'Q01780', 'genes': 'EXOSC10', 'spectrum': array([[2.14118618e+02, 4.88691780e-01],
        [3.43161211e+02, 2.95961020e-01],
        [3.61219394e+02, 5.16841650e-01],
        [4.71219788e+02, 1.32304220e-01],
        [4.74303458e+02, 2.24804240e-01],
        [5.42256902e+02, 1.95721600e-01],
        [5.45340572e+02, 3.53315110e-01],
        [6.73399150e+02, 5.63995700e-01],
        [8.02441743e+02, 1.00000000e+00],
        [8.73478857e+02, 8.50321800e-01],
        [9.44515970e+02, 4.92691930e-01]]), 'ordered_frags': array(['b3_1', 'b4_1', 'y3_1', 'b5_1', 'y4_1', 'b6_1', 'y5_1', 'y6_1',
        'y7_1', 'y8_1', 'y9_1'], dtype='<U4')})]
    >>>  iso_library_multi(library).items()[:1]
    [(('AAAEQAISVR', 2.0), {'mod_seq': 'AAAEQAISVR', 'seq': 'AAAEQAISVR', 'prec_mz': 508.28018033366, 'prec_z': 2.0, 'iRT': 0.296130418777466, 'frags': {'y7_1': [802.44174284642, 1.0], 'y8_1': [873.47885663113, 0.8503218], 'y6_1': [673.39914975845, 0.5639957], 'y3_1': [361.21939449133, 0.51684165], 'y9_1': [944.51597041584, 0.49269193], 'b3_1': [214.1186178209, 0.48869178], 'y5_1': [545.34057225317, 0.35331511], 'b4_1': [343.16121090887, 0.29596102], 'y4_1': [474.30345846846, 0.22480424], 'b6_1': [542.25690219886, 0.1957216], 'b5_1': [471.21978841415, 0.13230422]}, 'protein_group': 'Q01780', 'protein_name': 'Q01780', 'genes': 'EXOSC10', 'spectrum': array([[2.14118618e+02, 4.88691780e-01],
       [2.15121404e+02, 5.43275837e-02],
       [3.43161211e+02, 2.95961020e-01],
       [3.44164080e+02, 5.05647133e-02],
       [3.61219394e+02, 5.16841650e-01],
       [3.62222035e+02, 9.22381361e-02],
       [4.71219788e+02, 1.32304220e-01],
       [4.72222605e+02, 3.09481305e-02],
       [4.74303458e+02, 2.24804240e-01],
       [4.75306224e+02, 5.58995349e-02],
       [5.42256902e+02, 1.95721600e-01],
       [5.43259715e+02, 5.30352500e-02],
       [5.45340572e+02, 3.53315110e-01],
       [5.46343340e+02, 1.00947496e-01],
       [6.73399150e+02, 5.63995700e-01],
       [6.74401901e+02, 1.96711769e-01],
       [8.02441743e+02, 1.00000000e+00],
       [8.03444534e+02, 4.08462200e-01],
       [8.73478857e+02, 8.50321800e-01],
       [8.74481647e+02, 3.78834241e-01],
       [9.44515970e+02, 4.92691930e-01],
       [9.45518761e+02, 2.37760882e-01]]), 'ordered_frags': array(['b3_1', 'b3_1_iso1', 'b4_1', 'b4_1_iso1', 'y3_1', 'y3_1_iso1',
       'b5_1', 'b5_1_iso1', 'y4_1', 'y4_1_iso1', 'b6_1', 'b6_1_iso1',
       'y5_1', 'y5_1_iso1', 'y6_1', 'y6_1_iso1', 'y7_1', 'y7_1_iso1',
       'y8_1', 'y8_1_iso1', 'y9_1', 'y9_1_iso1'], dtype='<U9')})]

    """
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


def precursor_isotopes(sequence: str,charge: int,n_isotopes: int=2) -> list[bp._c.isotopic_distribution.TheoreticalPeak]:
    """
    Get precursor isotopic peaks for a given peptide sequence and charge.
    This function uses the `isotopic_variants` function from the `brainpy` library to generate isotopic peaks.

    Parameters
    ----------
    sequence : str
        Peptide sequence for which to generate isotopic peaks.
    charge : int
        Charge state of the peptide.
    n_isotopes : int, optional
        Number of isotopic peaks to generate, by default 2.

    Returns
    -------
    List of isotopic peaks, where each peak is a `brainpy._c.isotopic_distribution.TheoreticalPeak` object.
    Each peak contains the m/z, relative intensity, and charge of the isotopic variant.
    Example
    -------
    >>> sequence
    'DVPNSQLR'
    >>>  charge
    2
    >>> precursor_isotopes(sequence, charge, n_isotopes = 5)
    [Peak(mz=464.745973, intensity=0.607952, charge=2),
    Peak(mz=465.247360, intensity=0.286527, charge=2),
    Peak(mz=465.748606, intensity=0.083606, charge=2),
    Peak(mz=466.249827, intensity=0.018192, charge=2),
    Peak(mz=466.751025, intensity=0.003233, charge=2),
    Peak(mz=467.252209, intensity=0.000491, charge=2)]
    """
    sequence = re.sub("Decoy_","",sequence)
    split_seq = parse_peptide(sequence)
    
    seq_comp = get_seq_comp(split_seq, "M")
    
    if config.tag:
        tags = [t for aa in split_seq for t in re.findall(f"\(({config.tag.name}.*?)\)",aa)]
        if config.tag.channel_comp is not None and len(tags)>0:
                tag_comp = reduce(lambda x, y: x + y, [config.tag.channel_comp[re.findall(f"{config.tag.name}-(\d+)",t)[0]] for t in tags])
                seq_comp+=tag_comp
            
    
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

    index = np.where(array >= tr)[0][-1]

    if (len(array) > index):
        return array[:index + 1]
    else:
        return (array)
      


