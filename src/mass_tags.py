"""
This Source Code Form is subject to the terms of the Oxford Nanopore
Technologies, Ltd. Public License, v. 1.0.  Full licence can be found
at https://github.com/ParallelSquared/JMod/blob/main/LICENSE.txt
"""

import re
from pyteomics import mass
import tqdm
import os
import src.config as config
import numpy as np
import copy
import pandas as pd

from src.iso_functions import split_frag_name, fragment_seq

from src.utils.misc_functions import frag_to_peak, specific_frags
from src.utils.parse_peptides import parse_peptide
"""
## Load in the library
from SpecLib import loadSpecLib, write_speclib_tsv
# library = loadSpecLib("/Users/kevinmcdonnell/Programming/Data/SpecLibs/tims_library_dec_23.tsv")
library = loadSpecLib("/Users/kevinmcdonnell/Programming/Data/SpecLibs/timstof60Kprosit_speclib.tsv")
lib_file = "/Users/kevinmcdonnell/Programming/Data/SpecLibs/tims_library_dec_23_PrositFrags.tsv"
library = loadSpecLib(lib_file)

all_keys = list(library)

library[all_keys[4]]

"""
#### Diann inputs
# --fixed-mod mTRAQ, 140.0949630177, nK
# --lib-fixed-mod mTRAQ
# --channels mTRAQ,0,nK,0:0; mTRAQ,4,nK,4.0070994:4.0070994;mTRAQ,8,nK,8.0141988132:8.0141988132

## Also defines a "decoy: channel at mTRAQ-12


# ## define mtraq tags
# tag_name = "mTRAQ"
# base_mass = 140.0949630177
# channel_delta = 4.0070994
# n_channels = 3
# channel_names = ["0","4","8"]
# rules = "nK"
 
# tag_masses = (np.arange(n_channels)*channel_delta)+base_mass
class tag_isotope:
    
    def __init__(self,idxs,ints,mzs=None,min_intensity=0):
        assert len(idxs)==len(ints)
        
        # self.mzs = mzs
        # self.idxs = idxs
        # self.ints = ints
        # self.int_dict = {i:j for i,j in zip(idxs,ints)}
        # self.len = len(idxs)
        self.int_dict = {}
        self.idxs = []
        self.ints = []
        exceeds_min = np.ones_like(ints)
        i = 0
        while i<len(ints) and ints[i]<=min_intensity:
            exceeds_min[i] = 0
            i+=1
        
        i = len(ints)-1
        while i>0 and ints[i]<=min_intensity:
            exceeds_min[i] = 0
            i-=1
        
        for idx,i,keep in zip(idxs,ints,exceeds_min):
            if keep:
                self.int_dict[idx]=i
                self.idxs.append(idx)
                self.ints.append(i)
        
        
        
    # @property
    # def mz(self):
    #     return self.mzs
    def __repr__(self):
        return "\n".join(["idxs: "+",\t".join(map(str,self.idxs)),
                          "ints: "+",\t".join(map(str,self.ints)),
                          ])

    def __len__(self):
        return self.len
    
    # def plot(self,**kwargs):
    #     plt.vlines(self.idxs,0,self.ints,**kwargs)
        
        
class emp_tag_isotopes:

    def __init__(self,channel_names,file,sep=",",max_num_tags=4,min_intensity=0):
        
        ### read in data
        data = pd.read_csv(file,sep=sep,  index_col=0)
        ## convert tcols to integers (these are isotope indices)
        data.columns = data.columns.map(int)
        ### create dictionary to call data from
        data_dict = {str(i):tag_isotope(list(j.index), list(j),min_intensity=min_intensity)
                         for i,j in data.iterrows()}
        
        self.data = data_dict
        
        ##########################################
        ## get emp values for relevant channels
        ##########################################
        
        ### create the holder for the isotopes
        self.channel_dict = {i:{j:None for j in range(1,max_num_tags+1)} for i in channel_names}
        
        ### loop throught the channels, adding one tag each time
        for channel in channel_names:
            for num_tags in range(1,max_num_tags+1):
                if num_tags==1:
                    self.channel_dict[channel][num_tags] = data_dict[channel]
                else:
                    ## add 1 tag to previous tag isotope
                    self.channel_dict[channel][num_tags] = self.comb_iso(self.channel_dict[channel][num_tags-1], 
                                                                         self.channel_dict[channel][1],
                                                                         min_intensity=min_intensity)
        
        
        
    ### combine tag_isotope classes
    def comb_iso(self, a_iso,b_iso,min_intensity=0):
        idx_array = np.array(a_iso.idxs)[:,np.newaxis]+np.array([b_iso.idxs])
        min_idx = np.min(idx_array)
        max_idx = np.max(idx_array)
        
        new_ints = [0]*(max_idx-min_idx+1)
        new_idxs = [0]*(max_idx-min_idx+1)
        new_iso_dict = {}
        
        for i in a_iso.idxs:
            a_int = a_iso.int_dict[i]
            for j in b_iso.idxs:
                idx = i+j-min_idx
                new_iso_dict.setdefault(i+j,0)
                # if idx>=new_len:
                #     break
                new_iso_dict[i+j]+=a_int*b_iso.int_dict[j]
        sorted_idxs = sorted(new_iso_dict.keys())
        return tag_isotope(idxs=sorted_idxs, 
                           ints=[new_iso_dict[i] for i in sorted_idxs],
                           min_intensity=min_intensity)

    def __getitem__(self, item):
         return self.channel_dict[item]
     
        
class massTag():
    
    def __init__(self,rules,base_mass,delta,channel_names, name, compositions=None, emp_isotope_file = None):
        
        self.rules = rules
        
        self.mass = base_mass
        
        self.delta = delta
        
        self.n_channels = len(channel_names)
        
        self.channel_names = channel_names
        
        if type(delta)!= list and len(delta)<2:
            self.channel_masses =(np.arange(self.n_channels)*delta)+base_mass
        else:
            assert len(delta)==len(self.channel_names), "Channel names and deltas do not match"
            self.channel_masses =(np.ones(self.n_channels)*delta)+base_mass
        self.name = name
        
        self.mass_dict = {self.name+"-"+str(i):j for i,j in zip(self.channel_names,self.channel_masses)}
    
    
        if compositions is not None:
            self.channel_comp = {i:compositions[i] for i in self.channel_names}
        else: 
            self.channel_comp=None
            
        if emp_isotope_file is not None:
            self.emp_vals = emp_tag_isotopes(channel_names=self.channel_names,
                                             file=emp_isotope_file,
                                             sep=",",
                                             min_intensity=0.01)
        else: 
            self.emp_vals=None
            
    def __repr__(self):
        return("\n".join([
                           "Mass Tag",
                          # f"{self.n_channels} Channels",
                          F"TagName: {self.name}",
                          f"Base Mass: {self.mass}",
                          f"MassDelta(s): {self.delta}",
                          f"ChannelNames: {self.channel_names}",
                          f"ChannelMasses: {self.channel_masses}"]))
    
    def __getitem__(self,item):
        return getattr(self,item)
        
mTRAQ = massTag(rules = "nK",
            base_mass=140.0949630177,
            # delta = 4.0070994,
            # delta = [4.0070994],
            delta = [0.0,4.0070994,8.0141988132],
            channel_names = ["0","4","8"],
            name = "mTRAQ")


### 
mTRAQ_678 = massTag(rules = "nK",
                    base_mass=140.0949630177,
                    # delta = 4.0070994,
                    # delta = [4.0070994],
                    delta = [6.0074891,7.0108440,8.0141988132],
                    channel_names = ["6","7","8"],
                    name = "mTRAQ_678")

 
mTRAQ_02468 =   massTag(rules = "nK",
                        base_mass=140.0949630177,
                        # delta = 4.0070994,
                        # delta = [4.0070994],
                        delta = [0.0,2.0003897,4.0070994,6.0074891,8.0141988132],
                        channel_names = ["0","2","4","6","8"],
                        name = "mTRAQ_02468")


diethyl_6plex =       massTag(rules = "nK",
                        base_mass=56.06260026,
                        delta = [0.0,2.01255348,4.013419349,6.025972839,8.05021396,10.062767459],#,12.06363332,14.07618681],
                        channel_names = ["0","2","4","6","8","10"],#,"12","14"],
                        name = "diethyl_6plex")


diethyl_3plex =       massTag(rules = "nK",
                        base_mass=56.06260026,
                        delta = [0.0,4.013419349,8.05021396],#,12.06363332,14.07618681],
                        channel_names = ["0","4","8"],#,"12","14"],
                        name = "diethyl_3plex")

tag6_compositions = {"0":mass.Composition({"C":18,"H":16,"N":2,"O":3}),
                     "1":mass.Composition({"C":17,"H":16,"N":2,"O":3,"C[13]":1,"O[18]":0}),
                    "2":mass.Composition({"C":16,"H":16,"N":2,"O":3,"C[13]":2,"O[18]":0}),
                    "3":mass.Composition({"C":17,"H":16,"N":2,"O":2,"C[13]":1,"O[18]":1}),
                    "4":mass.Composition({"C":16,"H":16,"N":2,"O":2,"C[13]":2,"O[18]":1}),
                    "6":mass.Composition({"C":12,"H":16,"N":2,"O":3,"C[13]":6,"O[18]":0}),
                    "8":mass.Composition({"C":10,"H":16,"N":2,"O":3,"C[13]":8,"O[18]":0}),
                    "10":mass.Composition({"C":10,"H":16,"N":2,"O":2,"C[13]":8,"O[18]":1}),
                    "12":mass.Composition({"C":7,"H":16,"N":1,"O":3,"C[13]":11,"O[18]":0,"N[15]":1}),
                    "14":mass.Composition({"C":5,"H":16,"N":1,"O":3,"C[13]":13,"O[18]":0,"N[15]":1}),
                    "16":mass.Composition({"C":6,"H":16,"N":0,"O":2,"C[13]":12,"O[18]":1,"N[15]":2}),
                                         }

     
     
tag6 = massTag(rules = "nK",
            base_mass=308.1160923903,
            # delta = 4.0070994,
            # delta = [4.0070994],
            delta = [0.0],#,4.01095605604],#,8.02683870239997],
            channel_names = ["0"],#["0","4"],#,"8"],
            name = "tag6",
            compositions=tag6_compositions,
            emp_isotope_file="/Volumes/Lab/KMD/tag6_emp_isotopes.csv"
            )

tag6_5plex = massTag(rules = "nK",
            base_mass=308.1160923903,
            # delta = 4.0070994,
            # delta = [4.0070994],
            delta = [0.0, 4.01095605604,8.0268387024,
                     12.0339381092,16.03857422084],
            channel_names = ["0","4","8","12","16"],
            name = "tag6_5plex",
            compositions=tag6_compositions,
            # emp_isotope_file="/Volumes/Lab/KMD/tag6_emp_isotopes.csv"
            # emp_isotope_file="/Users/kevinmcdonnell/Programming/Data/tag6/MethodOptimization/tag6_fitted_isotopes.csv"
            )

tag6_9plex = massTag(rules = "nK",
            base_mass=308.1160923903,
            # delta = 4.0070994,
            # delta = [4.0070994],
            delta = [0.0, 2.0067096756, 4.01095605604, 6.0201290268,8.0268387024,10.03108508284,
                     12.0339381092,14.0406477848,16.03857422084],
            channel_names = ["0","2","4","6","8","10","12","14","16"],
            name = "tag6_9plex",
            compositions=tag6_compositions,
            # emp_isotope_file="/Volumes/Lab/KMD/tag6_emp_isotopes.csv"
            # emp_isotope_file="/Users/kevinmcdonnell/Programming/Data/tag6/MethodOptimization/tag6_fitted_isotopes.csv"
            ) 

tag6_d0d2 = massTag(rules = "nK",
            base_mass=308.1160923903,
            # delta = 4.0070994,
            # delta = [4.0070994],
            delta = [0.0, 2.0067096756],
            channel_names = ["0","2"],
            name = "tag6_d0d2",
            compositions=tag6_compositions) 

tag6_d0d4 = massTag(rules = "nK",
            base_mass=308.1160923903,
            # delta = 4.0070994,
            # delta = [4.0070994],
            delta = [0.0, 4.01095605604],
            channel_names = ["0","4"],
            name = "tag6_d0d4",
            compositions=tag6_compositions) 
     

tag6_d0d2d4 = massTag(rules = "nK",
                    base_mass=308.1160923903,
                    # delta = 4.0070994,
                    # delta = [4.0070994],
                    delta = [0.0, 2.0067096756, 4.01095605604],
                    channel_names = ["0","2","4"],
                    name = "tag6_d0d2d4",
                    compositions=tag6_compositions) 
     

tag6lys = massTag(rules = "nK",
            base_mass=464.24235,
            # delta = 4.0070994,
            # delta = [4.0070994],
            delta = [0.0],#,4.01095605604],#,8.02683870239997],
            channel_names = ["0"],#["0","4"],#,"8"],
            name = "tag6lys")
 
     
tag6arg = massTag(rules = "nK",
            base_mass=464.2172,
            # delta = 4.0070994,
            # delta = [4.0070994],
            delta = [0.0],#,4.01095605604],#,8.02683870239997],
            channel_names = ["0"],#["0","4"],#,"8"],
            name = "tag6arg")
 
     
tag6pip = massTag(rules = "nK",
            base_mass=434.1954,
            # delta = 4.0070994,
            # delta = [4.0070994],
            delta = [0.0],#,4.01095605604],#,8.02683870239997],
            channel_names = ["0"],#["0","4"],#,"8"],
            name = "tag6pip")

ProtSci_light_plex = massTag(rules = "nK",
            base_mass=0.0,
            delta = [328.231941371, 332.232721371, 336.246141371, 340.246860336,344.260280336],
            channel_names = ["4","8","12","16", "20"],
            name = "ProtSci_light_plex") 

ProtSci_heavy_plex = massTag(rules = "nK",
            base_mass=0.0,
            delta = [328.231941371, 332.232721371, 336.246141371, 340.246860336,344.260280336],
            channel_names = ["4","8","12","16", "20"],
            name = "ProtSci_light_plex") 

SILAC = massTag(rules="R", 
                base_mass=0, 
                delta=[3.98814], 
                channel_names=[4], 
                name="SILAC")

## split up the fragment name (b/y)(-loss)(frag index)_charge
def split_frag_name(ion_type):
    frag_name,frag_z = ion_type.split("_")
    loss_check = frag_name.split("-")
    loss = ""
    if len(loss_check)>1:
        frag_name,loss = loss_check
    frag_type = frag_name[0]
    frag_idx = int(frag_name[1:])
    
    return frag_type,frag_idx,loss,frag_z 




def get_tag_pos(AA_seq,rules):
    """
    

    Parameters
    ----------
    AA_seq : list
        Separated sequence of AAs, AA assumed to always be first followed by mod if present
    rules : str
        AAs that take a tag
        or
        n denotes n-terminus

    Returns
    -------
    Positions in sequence that take a tag

    """
    additional_tag_masses = np.zeros(len(AA_seq))
    
    all_tag_pos = []
    for rule in rules:
        # break
        # print(rule)
        if re.match("[A-Z]",rule):
            tag_pos = list(np.where([rule==i[0] for i in AA_seq])[0])
            
        elif rule=="n":
            tag_pos = [0]
            
        else:
            raise(ValueError("Unknown Tag Rule"))
        all_tag_pos += tag_pos
        additional_tag_masses[tag_pos]+=1

    return all_tag_pos, additional_tag_masses


## potentially add this as module to Tag class
def tag_library(library,tag=mTRAQ):
    """
    

    Parameters
    ----------
    library : dict
        Spectral library
    tag : Tag
        Curerntly works for mTRAQ, defined above.

    Returns
    -------
    New dictionary with copy of each precursor for each channel.

    """
    print(f"Generating tagged library with tag: {tag.name}")
    
    new_lib = {}
    
    

    for key in tqdm.tqdm(library):
        
        peptide = key[0]
        peptide = "".join(peptide)
        # split_peptide = re.findall("([A-Z](?:\(.*?\))?)",peptide)
        split_peptide = parse_peptide(peptide)
        
        
        
        
        ### Qs: 
        ## Can we have multiple tags on the same AA?
        ## How do mods effect abilty to tag?
        ## DIANN puts the n terminus tag before the 1st AA; What is the appropriate nomenclature?
        
        all_tag_pos, additional_tag_masses = get_tag_pos(split_peptide, tag.rules)
        
        ## use to get number of tags per frag
        
        num_tags_n = np.cumsum(additional_tag_masses,dtype=int)
        num_tags_c = np.cumsum(additional_tag_masses[::-1],dtype=int)
          
        for pos in all_tag_pos:
            split_peptide[pos]+="("+tag.name+")"
        
        
        blank_tags = []
        frags = library[key]["frags"]
        for frag in frags:

            
            ### capture anything in brakets as a modification
            mods = re.finditer("\((.*?)\)",peptide)
        
            stripped_peptide = re.sub("\(.*?\)","",peptide)
            
            frag_type,frag_idx,loss,frag_z = split_frag_name(frag)
            
            assert int(frag_idx)<len(stripped_peptide)
            if frag_type in 'abc':
                seq = split_peptide[:int(frag_idx)]
                num_tags = num_tags_n[frag_idx-1]
            elif frag_type in 'xyz':
                seq = split_peptide[-int(frag_idx):]
                num_tags = num_tags_c[frag_idx-1]
            else:
                raise(ValueError("Invalid ion type"))
                    
            # print(library[key]["frags"][frag],seq,num_tags)
            blank_tags.append([frag,library[key]["frags"][frag],num_tags,frag_z])
            
        
        
        for tag_idx,tag_n in enumerate(tag.channel_names):
            lib_entry  = copy.deepcopy(library[key])
            
            tag_mass = tag.channel_masses[tag_idx]
            new_seq = re.sub(tag.name,tag.name+"-"+str(tag_n),"".join(split_peptide))
            lib_entry["mod_seq"] = new_seq
            lib_entry["prec_mz"]+= (tag_mass*len(all_tag_pos))/lib_entry["prec_z"]
            
            for frag,[mz,I],n_tags,frag_z in blank_tags:
                
                lib_entry["frags"][frag] = [mz+(tag_mass*n_tags/int(frag_z)),I]
                
            lib_entry["spectrum"],lib_entry["ordered_frags"] = frag_to_peak(lib_entry["frags"],return_frags=True)
            if "spec_frags" in library[key]:
                lib_entry["spec_frags"] = specific_frags(lib_entry["frags"])
            new_lib[new_seq,key[1]] = lib_entry
            
        
    return new_lib

# mTRAQ_lib = tag_library(library, tag=mTRAQ)

available_tags = {"mTRAQ":mTRAQ,
                  "mTRAQ678":mTRAQ_678,
                  "mTRAQ02468":mTRAQ_02468,
                  "diethyl_6plex":diethyl_6plex,
                  "diethyl_3plex":diethyl_3plex,
                  "tag6":tag6,
                  "tag6_5plex":tag6_5plex,
                  "tag6_9plex":tag6_9plex,
                  "tag6_d0d2":tag6_d0d2,
                  "tag6_d0d4":tag6_d0d4,
                  "tag6_d0d2d4":tag6_d0d2d4,
                  "tag6pip":tag6pip,
                  "tag6lys":tag6lys,
                  "tag6arg":tag6arg,
                  "ProtSci_light_plex":ProtSci_light_plex,
                  "SILAC":SILAC}

# if config.args.mTRAQ:
#     config.tag = mTRAQ

# else: 
#     config.tag = None

if config.args.tag in available_tags:
    config.tag = available_tags[config.args.tag]
elif config.args.tag in "":
    config.tag = None
else:
    raise Exception("Incompatible Tag")