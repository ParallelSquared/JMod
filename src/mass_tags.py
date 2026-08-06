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
def tag_library(library,tag=None,source_channel=None):
    """


    Parameters
    ----------
    library : dict or SpectrumLibraryStore
        Spectral library
    tag : Tag
    source_channel : The channel (in "PSMtag_5plex-d0" format) that is currently present on library peptides 
                        (None if library had been computationally detagged)

    Returns
    -------
    New SpectrumLibraryStore with copy of each precursor for each channel.

    """
    from src.models.spec_lib.library_store import SpectrumLibraryStore
    if isinstance(library, SpectrumLibraryStore):
        return SpectrumLibraryStore.from_tagged(library, tag, source_channel=source_channel)


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
