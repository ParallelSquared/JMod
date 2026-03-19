"""
This Source Code Form is subject to the terms of the Oxford Nanopore
Technologies, Ltd. Public License, v. 1.0.  Full licence can be found
at https://github.com/ParallelSquared/JMod/blob/main/LICENSE.txt
"""

import re
from matplotlib.path import Path
from pyteomics import mass
import tqdm
import os
import src.config as config
import numpy as np
import copy
import json
from pathlib import Path

from src.iso_functions import fragment_seq
from src.utils.parse_peptides import split_frag_name

from src.utils.misc_functions import frag_to_peak, specific_frags
from src.utils.parse_peptides import parse_peptide, split_frag_name

import logging
from src.logger import logger
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
    
    def __init__(self,rules,base_mass,delta,channel_names, name, compositions=None):
        
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
    

def read_json_to_massTag(mass_tags_dir,filename):
    mass_tag_JSON = os.path.join(mass_tags_dir,filename)
    if mass_tag_JSON:
        with open(mass_tag_JSON, 'r') as f:
            try:
                mass_tag_data = json.load(f)
            except json.JSONDecodeError:
                return mass_tag_JSON
        try:
            compositions = {channel: mass.Composition(comp_dict) for channel, comp_dict in mass_tag_data['compositions'].items()}
            masses = [mass.calculate_mass(compositions[channel_name]) for channel_name in compositions]
            delta_calcs = [current_mass - masses[0] for current_mass in masses]
            if round(masses[0], 3) != round(mass_tag_data['base_mass'], 3):
                logging.getLogger("GUI").warning(f"Calculated Base Mass: {round(masses[0], 3)}")
                logging.getLogger("GUI").warning(f"JSON Base Mass: {round(mass_tag_data['base_mass'], 3)}")
                logging.getLogger("GUI").warning("First mass composition mass does not match base mass in tag JSON")
                return mass_tag_data['name']
            if [round(x, 3) for x in mass_tag_data['delta']] != [round(x, 3) for x in delta_calcs]:
                logging.getLogger("GUI").warning([round(x, 3) for x in mass_tag_data['delta']])
                logging.getLogger("GUI").warning([round(x, 3) for x in delta_calcs])
                logging.getLogger("GUI").warning("Deltas do not match mass compositions in tag JSON")
                return mass_tag_data['name']
            if mass_tag_data['name'] != os.path.splitext(filename)[0]:
                logging.getLogger("GUI").warning(f"Tag name:  {mass_tag_data['name']}")
                logging.getLogger("GUI").warning(f"File name: {os.path.splitext(filename)[0]}")
                logging.getLogger("GUI").warning("Tag name does not match filename")
                return(mass_tag_data['name'])
            mass_tag = massTag(rules=mass_tag_data['rules'],
                        base_mass=mass_tag_data['base_mass'],
                        delta=mass_tag_data['delta'],
                        channel_names=mass_tag_data['channel_names'],
                        name=mass_tag_data['name'],
                        compositions=compositions)
            return mass_tag
        except AttributeError:
            logging.getLogger("GUI").warning("Compositions are not defined for tag")
            return mass_tag_data['name']
        except Exception as e:
            logging.getLogger("GUI").warning({e}) 
            return mass_tag_data['name']
    else:
        return None
                    



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
        # logger.info(rule)
        if re.match("[A-Z]",rule):
            tag_pos = list(np.where([rule==i[0] for i in AA_seq])[0])
            
        elif rule=="n":
            tag_pos = [0]
            
        else:
            from src.utils.gui_utils import send_raise_to_TK
            send_raise_to_TK("ValueError - Unknown Tag Rule")
            raise(ValueError("Unknown Tag Rule"))
        all_tag_pos += tag_pos
        additional_tag_masses[tag_pos]+=1

    return all_tag_pos, additional_tag_masses

mTRAQ = massTag(rules = "nK",
                base_mass=140.0949630177,
                # delta = 4.0070994,
                # delta = [4.0070994],
                delta = [0.0,4.0070994,8.0141988132], 
                channel_names = ["0","4","8"],
                name = "mTRAQ")
##TODO what is going on with mTRAQ and tag_library here

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
    logger.info(f"Generating tagged library with tag: {tag.name}")
    
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
                from src.utils.gui_utils import send_raise_to_TK
                send_raise_to_TK("ValueError - Invalid Ion Type")
                raise(ValueError("Invalid ion type"))
                    
            # logger.info(library[key]["frags"][frag],seq,num_tags)
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


    from src.models.spec_lib.library_store import SpectrumLibraryStore
    if isinstance(library, SpectrumLibraryStore):
        return SpectrumLibraryStore.from_dict(new_lib)
    return new_lib

# mTRAQ_lib = tag_library(library, tag=mTRAQ)

def refresh_tags(mass_tags_dir=None): #set mass tags dir to none for testing purposes
    available_tags = {}
    if mass_tags_dir is None:
        mass_tags_dir = Path(__file__).parent / "MassTags"
    subset_dir = mass_tags_dir / "Subsets"
    for directory in (mass_tags_dir, subset_dir):
        for filename in os.listdir(directory):
            if os.path.splitext(filename)[1].lower() == ".json":
                mass_tag = read_json_to_massTag(directory, filename)
                if type(mass_tag) == str:
                    logging.getLogger("GUI").warning(f"Unable to load mass tag from {filename}\n")  
                elif mass_tag:
                    available_tags[mass_tag.name] = mass_tag
    return available_tags

available_tags = refresh_tags()



def set_config_tag(available_tags, config_args_tag):
    if config_args_tag in available_tags:
        return available_tags[config_args_tag]
    elif config_args_tag == "None":
        return None
    else:
        from src.utils.gui_utils import send_raise_to_TK
        send_raise_to_TK(f"Exception - Tag {config_args_tag} not in available tags")
        raise Exception(f"Exception - Tag '{config_args_tag}' not in available tags:\n{list(available_tags.keys())}")


config.tag = set_config_tag(available_tags, config.args.tag)