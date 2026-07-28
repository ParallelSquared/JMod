"""
This Source Code Form is subject to the terms of the Oxford Nanopore
Technologies, Ltd. Public License, v. 1.0.  Full licence can be found
at https://github.com/ParallelSquared/JMod/blob/main/LICENSE.txt
"""
import numpy as np
import os
import csv
import pandas as pd
import sqlite3
import struct
import zlib
import pickle
from src.utils.misc_functions import  frag_to_peak
from src.utils.parse_peptides import change_seq, convert_frags
import tqdm
import copy
import src.config as config
from src.iso_functions import gen_isotopes_dict
from src.logger import logger
import re
from pyteomics import mass


# load in spec library (tsv)
# file = "/Users/kevinmcdonnell/Programming/Data/SpecLibs/HeLa+K562-1prcGlobProt-5prcLocPep-PeakViewConverted.txt"
# spec_lib = pd.read_csv(file,delimiter="\t")


def create_python_lib(spec_lib):
    python_lib = {}
    for idx,row in spec_lib.iterrows():
        unique_id = (row["modification_sequence"],row["prec_z"])
        python_lib.setdefault(unique_id,{})
        python_lib[unique_id]["mod_seq"] = row["modification_sequence"]
        python_lib[unique_id]["seq"] = row["stripped_sequence"]
        python_lib[unique_id]["prec_mz"] = row["Q1"]
        python_lib[unique_id]["prec_z"] = row["prec_z"]
        python_lib[unique_id]["iRT"] = row["iRT"]
        python_lib[unique_id].setdefault("frags",{})
        frag_type = str(row["frg_type"])+str(row["frg_nr"])+"_"+str(row["frg_z"])
        python_lib[unique_id]["frags"][frag_type]=[row["Q3"],row["relative_intensity"]]
        
    return python_lib

    
#####################################################

# def load_tsv_lib(spec_lib_file):
#     with open(spec_lib_file,newline="") as tsv_file:
#         csv_reader = csv.DictReader(tsv_file,delimiter="\t")
#         python_lib = {}
#         idx = 0
#         for row in csv_reader:
#             unique_id = (row["modification_sequence"],float(row["prec_z"]))
#             python_lib.setdefault(unique_id,{})
#             python_lib[unique_id]["mod_seq"] = row["modification_sequence"]
#             python_lib[unique_id]["seq"] = row["stripped_sequence"]
#             python_lib[unique_id]["prec_mz"] = float(row["Q1"])
#             python_lib[unique_id]["prec_z"] = float(row["prec_z"]) 
#             rt = row["iRT"]
#             python_lib[unique_id]["iRT"] = None if rt=="" else float(rt)
#             python_lib[unique_id].setdefault("frags",{})
#             frag_type = str(row["frg_type"])+str(row["frg_nr"])+"_"+str(row["frg_z"])
#             python_lib[unique_id]["frags"][frag_type]=[float(row["Q3"]),float(row["relative_intensity"])]
#             # idx+=1
#             # if idx>1117038:
#             #     break
#         for key in python_lib:
#             python_lib[key]["spectrum"] = frag_to_peak(python_lib[key]["frags"])
#         return python_lib

# def load_tsv_lib_sp(spec_lib_file):
#     with open(spec_lib_file,newline="") as tsv_file:
#         csv_reader = csv.DictReader(tsv_file,delimiter="\t")
#         python_lib = {}
#         idx = 0
#         for row in csv_reader:
#             unique_id = (row["modification_sequence"],float(row["prec_z"]))
#             python_lib.setdefault(unique_id,{})
#             python_lib[unique_id]["mod_seq"] = row["modification_sequence"]
#             python_lib[unique_id]["seq"] = row["stripped_sequence"]
#             python_lib[unique_id]["PrecursorMZ"] = float(row["Q1"])
#             python_lib[unique_id]["prec_z"] = float(row["prec_z"]) 
#             rt = row["iRT"]
#             python_lib[unique_id]["PrecursorRT"] = None if rt=="" else float(rt)
#             python_lib[unique_id].setdefault("frags",{})
#             frag_type = str(row["frg_type"])+str(row["frg_nr"])+"_"+str(row["frg_z"])
#             python_lib[unique_id]["frags"][frag_type]=[float(row["Q3"]),float(row["relative_intensity"])]
#             # idx+=1
#             # if idx>1117038:
#             #     break
#         for key in python_lib:
#             python_lib[key]["Spectrum"] = frag_to_peak(python_lib[key]["frags"])
#         return python_lib
    

### FileName	PrecursorMz	ProductMz	Tr_recalibrated	IonMobility	transition_name	LibraryIntensity	transition_group_id	decoy	
# PeptideSequence	Proteotypic	QValue	PGQValue	Ms1ProfileCorr	ProteinGroup	ProteinName	Genes	FullUniModPeptideName	
# ModifiedPeptide	PrecursorCharge	PeptideGroupLabel	UniprotID	NTerm	CTerm	FragmentType	FragmentCharge	FragmentSeriesNumber	FragmentLossType	ExcludeFromAssay

diann_names = ['PrecursorMz',
                 "ModifiedPeptide",
                 "PrecursorCharge",
                 "Tr_recalibrated",
                 "PeptideSequence",
                 "IonMobility",
                 "ProductMz",
                 "LibraryIntensity",
                 'ProteinGroup', 
                 'ProteinName',
                 'Genes',
                 "FragmentType"	,
                 "FragmentCharge",
                 "FragmentSeriesNumber"	,
                 "FragmentLossType"]

### DIann names to our names converter
diann_to_jmod = {'PrecursorMz':'prec_mz',
                 "ModifiedPeptide":"mod_seq",
                 "PrecursorCharge":"prec_z",
                 "Tr_recalibrated":"iRT",
                 "PeptideSequence":"seq",
                 "IonMobility":"IonMob",
                 'ProteinGroup':"protein_group", 
                 'ProteinName':"protein_name",
                 'Genes':"genes"
                 }

jmod_to_diann = {j:i for i,j in diann_to_jmod.items()}


def load_tsv_speclib(spec_lib_file):
    # load speclib files from DIA-NN
    logger.info("using: load_tsv_speclib")
    with open(spec_lib_file,newline="") as tsv_file:
        csv_reader = csv.DictReader(tsv_file,delimiter="\t")
        all_columns  = csv_reader.fieldnames
        python_lib = {}
        idx = 0
        for row in csv_reader:
            if "ModifiedPeptide" in row:
                row["ModifiedPeptide"] = row["ModifiedPeptide"].strip("_")
            elif "ModifiedSequence" in row:
                row["ModifiedPeptide"] = row["ModifiedSequence"].strip("_")
            unique_id = (row["ModifiedPeptide"],float(row["PrecursorCharge"]))
            python_lib.setdefault(unique_id,{})
            python_lib[unique_id]["mod_seq"] = row["ModifiedPeptide"]
            if "StrippedPeptide" in row:
                python_lib[unique_id]["seq"] = row["StrippedPeptide"]
            else:
                python_lib[unique_id]["seq"] = row["PeptideSequence"]
            python_lib[unique_id]["prec_mz"] = float(row["PrecursorMz"])
            python_lib[unique_id]["prec_z"] = float(row["PrecursorCharge"]) 
            
            if "Tr_recalibrated" in row:
                rt = row["Tr_recalibrated"]
            elif "RT" in row:
                rt = row["RT"]
            elif "iRT" in row:
                rt = row["iRT"]
            else:
                from src.utils.gui_utils import send_raise_to_TK
                send_raise_to_TK("ValueError - Unknown Retention Time Column")
                raise ValueError("Unknown retention time column")
            python_lib[unique_id]["iRT"] = None if rt=="" else float(rt)
            python_lib[unique_id].setdefault("frags",{})
            loss=""
            if "FragmentLossType" in row:
                loss = str(row["FragmentLossType"])
                if loss in ["unknown","noloss",""]:
                    loss=""
                else:
                    loss = "-"+loss
            if "FragmentNumber" in row:
                frag_type = str(row["FragmentType"])+str(row["FragmentNumber"])+loss+"_"+str(row["FragmentCharge"])
            else:
                frag_type = str(row["FragmentType"])+str(row["FragmentSeriesNumber"])+loss+"_"+str(row["FragmentCharge"])
            
            if "FragmentMz" in row:
                python_lib[unique_id]["frags"][frag_type]=[float(row["FragmentMz"]),float(row["RelativeIntensity"])]
            else:
                python_lib[unique_id]["frags"][frag_type]=[float(row["ProductMz"]),float(row["LibraryIntensity"])]
            if "IonMobility" in row:
                if row["IonMobility"]!="":
                    python_lib[unique_id]["IonMob"] = float(row["IonMobility"]) 
            elif "IM" in row:
                if row["IM"]!="":
                    python_lib[unique_id]["IonMob"] = float(row["IM"]) 
            
            ### Protein info
            if "ProteinGroup" in row:
                python_lib[unique_id]["protein_group"] = row["ProteinGroup"]
            if "ProteinName" in row:
                python_lib[unique_id]["protein_name"] = row["ProteinName"]
            elif "ProteinID" in row:
                python_lib[unique_id]["protein_name"] = row["ProteinID"]
            elif "ProteinId" in row:
                python_lib[unique_id]["protein_name"] = row["ProteinId"]
            if "Genes" in row:
                python_lib[unique_id]["genes"] = row["Genes"]
            if "GeneName" in row:
                python_lib[unique_id]["genes"] = row["GeneName"]
            if "UniprotID" in row:
                python_lib[unique_id]["UniprotID"] = row["UniprotID"]
            
            
            # idx+=1
            # if idx>111703:
            #     break
        for key in python_lib:
            python_lib[key]["spectrum"],python_lib[key]["ordered_frags"] = frag_to_peak(python_lib[key]["frags"],return_frags=True)
            # python_lib[key]["spec_frags"] = specific_frags(python_lib[key]["frags"]) # Note: does not work if only one frag in entry
        return python_lib






#  Generate Spec lib from .blib file (specter)
def load_blib(spec_lib_file):
    
    python_lib = {}
    sql_lib = sqlite3.connect(spec_lib_file)
    
    Precursors = pd.read_sql("SELECT * FROM RefSpectra",sql_lib)
    
    for i in range(len(Precursors)):
        precID = str(Precursors["id"][i])
        precKey = (Precursors["peptideModSeq"][i],Precursors["precursorCharge"][i])
        NumPeaks = pd.read_sql("SELECT numPeaks FROM RefSpectra WHERE id = "+precID,sql_lib)['numPeaks'][0]
            
        SpectrumMZ = pd.read_sql("SELECT peakMZ FROM RefSpectraPeaks WHERE RefSpectraID = " + precID,sql_lib)['peakMZ'][0]
        SpectrumIntensities = pd.read_sql("SELECT peakIntensity FROM RefSpectraPeaks WHERE RefSpectraID = "+precID,sql_lib)['peakIntensity'][0]
        
        ## Copied from Specter
        if len(SpectrumMZ) == 8*NumPeaks and len(SpectrumIntensities) == 4*NumPeaks:
            python_lib.setdefault(precKey,{})
            SpectrumMZ = struct.unpack('d'*NumPeaks,SpectrumMZ)
            SpectrumIntensities = struct.unpack('f'*NumPeaks,SpectrumIntensities)
            python_lib[precKey]['spectrum'] = np.array((SpectrumMZ,SpectrumIntensities)).T
            python_lib[precKey]['prec_mz'] = Precursors['precursorMZ'][i]
            python_lib[precKey]['iRT'] = Precursors['retentionTime'][i]      #The library retention time is given in minutes
        elif len(SpectrumIntensities) == 4*NumPeaks:
            python_lib.setdefault(precKey,{})
            SpectrumMZ = struct.unpack('d'*NumPeaks,zlib.decompress(SpectrumMZ))
            SpectrumIntensities = struct.unpack('f'*NumPeaks,SpectrumIntensities)
            python_lib[precKey]['spectrum'] = np.array((SpectrumMZ,SpectrumIntensities)).T
            python_lib[precKey]['prec_mz'] = Precursors['precursorMZ'][i]
            python_lib[precKey]['iRT'] = Precursors['retentionTime'][i]
        elif len(SpectrumMZ) == 8*NumPeaks:
            python_lib.setdefault(precKey,{})
            SpectrumMZ = struct.unpack('d'*NumPeaks,SpectrumMZ)
            SpectrumIntensities = struct.unpack('f'*NumPeaks,zlib.decompress(SpectrumIntensities))
            python_lib[precKey]['spectrum'] = np.array((SpectrumMZ,SpectrumIntensities)).T
            python_lib[precKey]['prec_mz'] = Precursors['precursorMZ'][i]
            python_lib[precKey]['iRT'] = Precursors['retentionTime'][i]
        elif len(zlib.decompress(SpectrumMZ)) == 8*NumPeaks and len(zlib.decompress(SpectrumIntensities)) == 4*NumPeaks:
            python_lib.setdefault(precKey,{})
            SpectrumMZ = struct.unpack('d'*NumPeaks,zlib.decompress(SpectrumMZ))
            SpectrumIntensities = struct.unpack('f'*NumPeaks,zlib.decompress(SpectrumIntensities))
            python_lib[precKey]['spectrum'] = np.array((SpectrumMZ,SpectrumIntensities)).T
            python_lib[precKey]['prec_mz'] = Precursors['precursorMZ'][i]
            python_lib[precKey]['iRT'] = Precursors['retentionTime'][i]
        
    sql_lib.close()

    return python_lib

    
# lib = load_blib("/Volumes/One Touch/PTI/Specter/EcoliSpectralLibrary.blib")    
_MOD_PATTERN = re.compile(r"\(([^)]+)\)")

def has_mass_tag(modified_peptides, prec_mzs, prec_zs):
    """
    Returns True if any ModifiedPeptide string contains a non-UniMod
    parenthetical modification (a mass tag),
    along with the back-calculated mass of that tag.

    Parameters
    ----------
    modified_peptides : Iterable[str]
        ModifiedPeptide strings, e.g. "P(PSMtag_5plex-0)EPTIDEK"
    prec_mzs : Iterable[float]
        PrecursorMz values, same order/length as modified_peptides
    prec_zs : Iterable[float]
        PrecursorCharge values, same order/length as modified_peptides

    Returns
    -------
    (source_channel_mass, library_tag_bool, library_tag_name) : (float, bool, str)
        Back-calculated mass of the tag channel from the first tagged
        peptide found, and whether any tag was found at all. Returns
        (0, False, None) if no tag is present.
    """
    diann_mods = config.diann_mods
    for pep, prec_mz, prec_z in zip(modified_peptides, prec_mzs, prec_zs):
        if not pep:
            continue
        all_mods = [m.strip() for m in _MOD_PATTERN.findall(pep)]
        tag_mods = [m for m in all_mods if m not in diann_mods]
        if not tag_mods:
            continue

        known_mods_attached = [m for m in all_mods if m in diann_mods]
        stripped_peptide = re.sub(r"\([^)]*\)", "", pep)
        stripped_mass = mass.fast_mass(stripped_peptide)
        known_mods_mass = sum(diann_mods[m] for m in known_mods_attached)

        untagged_mz = (stripped_mass + known_mods_mass + prec_z * 1.00727647) / prec_z
        mz_delta = prec_mz - untagged_mz
        source_channel_mass = (mz_delta * prec_z) / len(tag_mods)

        return source_channel_mass, True, tag_mods[0]

    return 0, False, None

def loadSpecLib(lib_file):
    from src.models.spec_lib.library_store import SpectrumLibraryStore

    lib_ext = lib_file.rsplit(".")[-1]

    logger.info("Loading Library...")
    store_file = lib_file + "_store.npz"
    python_lib_file = lib_file + "_pythonlib"

    if os.path.exists(store_file):
        logger.info("Loading Library... from binary cache")
        spec_lib = SpectrumLibraryStore.load(store_file)
    elif os.path.exists(python_lib_file):
        logger.info("Loading Library... from pickle")
        with open(python_lib_file, "rb") as read_file:
            old_lib = pickle.load(read_file)
        spec_lib = SpectrumLibraryStore.from_dict(old_lib)
        spec_lib.save(store_file)
    else:
        logger.info("Loading Library... from file")
        if lib_ext == "blib":
            spec_lib = SpectrumLibraryStore.from_blib(lib_file)
        elif lib_ext == "tsv":
            spec_lib = SpectrumLibraryStore.from_tsv(lib_file)
        elif lib_ext == "parquet":
            spec_lib = SpectrumLibraryStore.from_parquet(lib_file)
        spec_lib.save(store_file)

    source_channel_mass, library_tag_bool, library_tag_name = has_mass_tag(spec_lib.mod_seq, spec_lib.prec_mz, spec_lib.prec_z)
    if library_tag_bool:
        logger.info("Spectral Library Contains Tagged Peptides")

    logger.info(f"Loaded {len(spec_lib)} library precursors")
    return spec_lib, library_tag_bool, source_channel_mass, library_tag_name


# TODO add a test for this, make sure decoys are being generated correctly
import multiprocessing

def _decoy_worker(args):
    """Worker function for parallel decoy generation."""
    seq, frags, rules, tag, n_iso, use_iso = args
    new_seq = change_seq(seq, rules, tag=tag)
    new_frags = convert_frags(seq, frags, rules, tag=tag)
    if use_iso:
        spectrum, ordered_frags = gen_isotopes_dict(new_seq, new_frags, tag, n_iso)
    else:
        spectrum, ordered_frags = frag_to_peak(new_frags, return_frags=True)
    return new_seq, new_frags, spectrum, ordered_frags


def _decoy_arg_gen(library, all_keys, rules, tag, n_iso):
    """Lazily yield decoy worker args to avoid materializing all frag dicts."""
    for key in all_keys:
        yield (key[0], library[key]["frags"], rules, tag, n_iso, False)


def create_decoy_lib(library, rules, tag=None, n_iso=0):
    """Generate decoys and return a combined target+decoy SpectrumLibraryStore.

    Decoys are generated without tagging or isotope expansion — those
    operations should be applied to the combined store afterward.

    Returns a combined store with targets at [0, N) and decoys at [N, N+M).
    """
    from src.models.spec_lib.library_store import SpectrumLibraryStore
    import gc

    all_keys = list(library.keys())
    n_total = len(all_keys)

    p = multiprocessing.Pool(min(multiprocessing.cpu_count(), 61))
    results = []
    chunksize = 1000
    arg_gen = _decoy_arg_gen(library, all_keys, rules, tag, n_iso)
    for result in tqdm.tqdm(p.imap(_decoy_worker, arg_gen, chunksize=chunksize), total=n_total):
        results.append(result)
    p.close()
    p.join()

    gc.collect()

    store = SpectrumLibraryStore.from_target_and_decoy_results(library, all_keys, results)

    del results
    gc.collect()

    return store
            
            
# spec_lib = loadSpecLib("/Volumes/Lab/KMD/SpectralLibraries/8ng_LF_24nce.tsv")

def write_speclib_tsv(library,filename):
    ## create a new library from a library dictionary created by the above functions
    
    with open(filename,"w",newline="") as write_file:
        writer = csv.writer(write_file, delimiter='\t',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        
        lib_keys = list(library.keys())
        
        # write columns assuming they are always the same
        # each entry is also a dict
        col_names = list(library[lib_keys[0]].keys())
        
        writer.writerow(diann_names)
        # writer.writerow([i for i in diann_names if diann_to_jmod[i] in col_names])
        
        for key in lib_keys:
            precursor={i:library[key][j] for i,j in diann_to_jmod.items() if j in col_names}
            for frag in library[key]["frags"]:
                # logger.info(frag)
                frag_name,frag_z = frag.split("_")
                loss_check = frag_name.split("-")
                loss = "noloss"
                if len(loss_check)>1:
                    frag_name,loss = loss_check
                frag_type = frag_name[0]
                frag_idx = int(frag_name[1:])
                precursor["ProductMz"]=library[key]["frags"][frag][0]
                precursor["LibraryIntensity"]=library[key]["frags"][frag][1]
                precursor["FragmentType"]=frag_type
                precursor["FragmentCharge"]=int(frag_z)
                precursor["FragmentSeriesNumber"]=frag_idx
                precursor["FragmentLossType"]=loss
                
                # logger.info(list(precursor.values()))
                writer.writerow([precursor[i] if i in precursor else "" for i in diann_names])
                



class LibrarySpectrum():
    
    def __init__(self,seq,z):
        
        self.seq= seq
        self.z = z
        
        self.__data__ = {}
        
    def __repr__(self):
        return f"({self.seq},{self.z})"
        
    def __str__(self):
        return f"({self.seq},{self.z})"
        
    def __getattr__(self, name):
       return self[name]
   
    ### note this way of doing things may not work as eacvh line is a fragment not a precursor
    def read_entry(self,row):
        unique_id = (row["ModifiedPeptide"],float(row["PrecursorCharge"]))
        
        
        self.__data__["mod_seq"] = row["ModifiedPeptide"]
        self.__data__["seq"] = row["PeptideSequence"]
        self.__data__["prec_mz"] = float(row["PrecursorMz"])
        self.__data__["prec_z"] = float(row["PrecursorCharge"]) 
        rt = row["Tr_recalibrated"]
        self.__data__["iRT"] = None if rt=="" else float(rt)
        self.__data__.setdefault("frags",{})
        if "FragmentLossType" in row:
            loss = str(row["FragmentLossType"])
            if loss in ["unknown","noloss"]:
                loss=""
            else:
                loss = "-"+loss
        frag_type = str(row["FragmentType"])+str(row["FragmentSeriesNumber"])+loss+"_"+str(row["FragmentCharge"])
        self.__data__["frags"][frag_type]=[float(row["ProductMz"]),float(row["LibraryIntensity"])]
        if "IonMobility" in row:
            self.__data__["IonMob"] = float(row["IonMobility"]) 
        
        ### Protein info
        self.__data__["protein_group"] = row["ProteinGroup"]
        self.__data__["protein_name"] = row["ProteinName"]
        self.__data__["genes"] = row["Genes"]
        
    
class SpectrumLibrary():
    
    def __init__(self,lib_file):
        
        self.filename = lib_file
        
        
    
    def loadSpecLib(self,lib_file):
        
        lib_ext = lib_file.rsplit(".")[-1]
        
        logger.info("Loading Library...")
        python_lib_file = lib_file+"_pythonlib"
        if not os.path.exists(python_lib_file):
            logger.info("Loading Library... from file")
            if lib_ext=="blib":
                spec_lib = load_blib(lib_file)
            else:
                # spec_lib = load_tsv_lib(lib_file)
                spec_lib = load_tsv_speclib(lib_file)
            with open(python_lib_file,"wb") as write_file:
                pickle.dump(spec_lib, write_file)
        else:
            logger.info("Loading Library... from pickle")
            with open(python_lib_file,"rb") as read_file:
                spec_lib = pickle.load(read_file)
        
        logger.info(f"Loaded {len(spec_lib)} library spectra")
        logger.info("finished")
        return spec_lib
        
    