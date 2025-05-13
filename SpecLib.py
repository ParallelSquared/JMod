import subprocess
import numpy as np
from pyteomics import mzml, auxiliary
import os
import matplotlib.pyplot as plt
import csv
import pandas as pd
import sqlite3
import time
import cProfile
import struct
import zlib
import pickle
from miscFunctions import split_frag_name, frag_to_peak, specific_frags
import tqdm
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

def load_tsv_lib(spec_lib_file):
    with open(spec_lib_file,newline="") as tsv_file:
        csv_reader = csv.DictReader(tsv_file,delimiter="\t")
        python_lib = {}
        idx = 0
        for row in csv_reader:
            unique_id = (row["modification_sequence"],float(row["prec_z"]))
            python_lib.setdefault(unique_id,{})
            python_lib[unique_id]["mod_seq"] = row["modification_sequence"]
            python_lib[unique_id]["seq"] = row["stripped_sequence"]
            python_lib[unique_id]["prec_mz"] = float(row["Q1"])
            python_lib[unique_id]["prec_z"] = float(row["prec_z"]) 
            rt = row["iRT"]
            python_lib[unique_id]["iRT"] = None if rt=="" else float(rt)
            python_lib[unique_id].setdefault("frags",{})
            frag_type = str(row["frg_type"])+str(row["frg_nr"])+"_"+str(row["frg_z"])
            python_lib[unique_id]["frags"][frag_type]=[float(row["Q3"]),float(row["relative_intensity"])]
            # idx+=1
            # if idx>1117038:
            #     break
        for key in python_lib:
            python_lib[key]["spectrum"] = frag_to_peak(python_lib[key]["frags"])
        return python_lib

def load_tsv_lib_sp(spec_lib_file):
    with open(spec_lib_file,newline="") as tsv_file:
        csv_reader = csv.DictReader(tsv_file,delimiter="\t")
        python_lib = {}
        idx = 0
        for row in csv_reader:
            unique_id = (row["modification_sequence"],float(row["prec_z"]))
            python_lib.setdefault(unique_id,{})
            python_lib[unique_id]["mod_seq"] = row["modification_sequence"]
            python_lib[unique_id]["seq"] = row["stripped_sequence"]
            python_lib[unique_id]["PrecursorMZ"] = float(row["Q1"])
            python_lib[unique_id]["prec_z"] = float(row["prec_z"]) 
            rt = row["iRT"]
            python_lib[unique_id]["PrecursorRT"] = None if rt=="" else float(rt)
            python_lib[unique_id].setdefault("frags",{})
            frag_type = str(row["frg_type"])+str(row["frg_nr"])+"_"+str(row["frg_z"])
            python_lib[unique_id]["frags"][frag_type]=[float(row["Q3"]),float(row["relative_intensity"])]
            # idx+=1
            # if idx>1117038:
            #     break
        for key in python_lib:
            python_lib[key]["Spectrum"] = frag_to_peak(python_lib[key]["frags"])
        return python_lib
    

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
    print("using: load_tsv_speclib")
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


def loadSpecLib(lib_file):
    
    lib_ext = lib_file.rsplit(".")[-1]
    
    print("Loading Library",end=" ")
    python_lib_file = lib_file+"_pythonlib"
    if not os.path.exists(python_lib_file):
        print("... from file")
        if lib_ext=="blib":
            spec_lib = load_blib(lib_file)
        else:
            # spec_lib = load_tsv_lib(lib_file)
            spec_lib = load_tsv_speclib(lib_file)
        with open(python_lib_file,"wb") as write_file:
            pickle.dump(spec_lib, write_file)
    else:
        print("... from pickle")
        with open(python_lib_file,"rb") as read_file:
            spec_lib = pickle.load(read_file)
    
    print(f"Loaded {len(spec_lib)} library precursors")
    print("finished")
    return spec_lib
    

import copy
import config
from miscFunctions import change_seq, convert_frags, frag_to_peak
from iso_functions import gen_isotopes_dict


def create_decoy_lib(library,rules):
    ## keep keys the same but change seq, mz and frags
    
    decoy_lib =copy.deepcopy(library) # create copy so we do not change the original
    
    for key in tqdm.tqdm(decoy_lib):
        entry = decoy_lib[key]
        
        entry["seq"] = change_seq(key[0],rules)
        #!!! To change;
        # if config.args.decoy=="rev": ## this will have the same mz as many correct matches and therefore a really good ms1 isotope corr
        #     entry["prec_mz"] -= config.decoy_mz_offset
            
        entry["frags"] = convert_frags(key[0], entry["frags"],rules)
        
        if config.args.iso:
            entry["spectrum"], entry["ordered_frags"] = gen_isotopes_dict(entry["seq"], entry["frags"])
        else:
            entry["spectrum"], entry["ordered_frags"] = frag_to_peak(entry["frags"],return_frags=True)
            
            
    return decoy_lib
            
            
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
                # print(frag)
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
                
                # print(list(precursor.values()))
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
        
        print("Loading Library",end=" ")
        python_lib_file = lib_file+"_pythonlib"
        if not os.path.exists(python_lib_file):
            print("... from file")
            if lib_ext=="blib":
                spec_lib = load_blib(lib_file)
            else:
                # spec_lib = load_tsv_lib(lib_file)
                spec_lib = load_tsv_speclib(lib_file)
            with open(python_lib_file,"wb") as write_file:
                pickle.dump(spec_lib, write_file)
        else:
            print("... from pickle")
            with open(python_lib_file,"rb") as read_file:
                spec_lib = pickle.load(read_file)
        
        print(f"Loaded {len(spec_lib)} library spectra")
        print("finished")
        return spec_lib
        
    