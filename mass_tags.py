
import re
from pyteomics import mass
import tqdm
import os
import config
import numpy as np
import copy

from iso_functions import split_frag_name, fragment_seq

from miscFunctions import frag_to_peak, specific_frags

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

class massTag():
    
    def __init__(self,rules,base_mass,delta,channel_names, name):
        
        self.rules = rules
        
        self.mass = base_mass
        
        self.delta = delta
        
        self.n_channels = len(channel_names)
        
        self.channel_names = channel_names
        
        if type(delta)!= list or len(delta)<2:
            self.channel_masses =(np.arange(self.n_channels)*delta)+base_mass
        else:
            assert len(delta)==len(self.channel_names), "Channel names and deltas do not match"
            self.channel_masses =(np.ones(self.n_channels)*delta)+base_mass
        self.name = name
        
        self.mass_dict = {self.name+"-"+str(i):j for i,j in zip(self.channel_names,self.channel_masses)}
    
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

# mTRAQ
# print(mTRAQ)

### path to diann searches: /Volumes/Lab/MY/2024-03-25_Sciex_9-plex_combinations/
### 
# 140.0949630177, nK  --channels mTRAQ,6,nK,6.0074891:6.0074891;mTRAQ,7,nK,7.0108440:7.0108440;mTRAQ,8,nK,8.0141988132:8.0141988132 
mTRAQ_678 = massTag(rules = "nK",
                    base_mass=140.0949630177,
                    # delta = 4.0070994,
                    # delta = [4.0070994],
                    delta = [6.0074891,7.0108440,8.0141988132],
                    channel_names = ["6","7","8"],
                    name = "mTRAQ_678")

# print(mTRAQ)
 
### mTRAQ, 140.0949630177, nK  --channels mTRAQ,0,nK,0:0;mTRAQ,2,nK,2.0003897:2.0003897;mTRAQ,4,nK,4.0070994:4.0070994;mTRAQ,6,nK,6.0074891:6.0074891;mTRAQ,8,nK,8.0141988132:8.0141988132   
 
mTRAQ_02468 =   massTag(rules = "nK",
                        base_mass=140.0949630177,
                        # delta = 4.0070994,
                        # delta = [4.0070994],
                        delta = [0.0,2.0003897,4.0070994,6.0074891,8.0141988132],
                        channel_names = ["0","2","4","6","8"],
                        name = "mTRAQ_02468")

# print(mTRAQ)
 

diethyl_6plex =       massTag(rules = "nK",
                        base_mass=56.06260026,
                        delta = [0.0,2.01255348,4.013419349,6.025972839,8.05021396,10.062767459],#,12.06363332,14.07618681],
                        channel_names = ["0","2","4","6","8","10"],#,"12","14"],
                        name = "diethyl_6plex")
 


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
        split_peptide = re.findall("([A-Z](?:\(.*?\))?)",peptide)
        
        
        
        
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
                
            lib_entry["spectrum"] = frag_to_peak(lib_entry["frags"])
            lib_entry["spec_frags"] = specific_frags(lib_entry["frags"])
            new_lib[new_seq,key[1]] = lib_entry
            
        
    return new_lib

# mTRAQ_lib = tag_library(library, tag=mTRAQ)

available_tags = {"mTRAQ":mTRAQ,
                  "mTRAQ678":mTRAQ_678,
                  "mTRAQ02468":mTRAQ_02468,
                  "diethyl_6plex":diethyl_6plex}

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