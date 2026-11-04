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
from src.utils.errors import JModError
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

        # The name of the channel that is left empty in the sample, searched as a
        # null draw for the channel-level FDR.  None unless with_decoy_channel or
        # resolve_decoy_channel set it; every other channel is a real one.
        self.decoy_channel = None

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

    def select_channels(self, channels):
        """This tag searching only *channels* (names), under the same name.
        mass_dict keeps every channel's mass, so a library tagged in a channel
        not searched is still recognised and retagged."""
        unknown = [c for c in channels if c not in self.channel_names]
        if unknown or not channels:
            raise JModError(f"Channels {channels} are not channels of {self.name}: {self.channel_names}")
        keep = [i for i, c in enumerate(self.channel_names) if c in channels]
        tag = massTag(self.rules, self.mass, [float(self.channel_masses[i] - self.mass) for i in keep],
                      [self.channel_names[i] for i in keep], self.name, compositions=self.channel_comp)
        tag.mass_dict = self.mass_dict
        return tag

    def with_decoy_channel(self):
        """This tag with one more channel appended, to be searched as a decoy.

        The new channel is the step from the first to the second channel applied
        on top of the last channel.  The step is taken as a *composition*, not a
        mass, so the decoy channel carries the isotopes a real channel one step
        further along would carry, and its isotope envelope looks like a real
        channel's rather than like the monoisotopic tag shifted by a mass.

        The step is always extrapolated, even when the tag itself defines a channel
        of that name -- searching 0, 4, 8, 12 of a tag that also has 16 gives a
        decoy 16 of the extrapolation's own chemistry, 6.3 ppm from the reagent's
        on PSMtag_5plex (a different isotope route to the same nominal mass).  The
        decoy channel is a null draw, so what matters is that it sits one real step
        out, not that it is a reagent anyone could buy.  mass_dict's entry for that
        name is the extrapolated channel's from here on.

        Extrapolating can take an element's count negative: the step swaps light
        atoms for heavy ones, and the tag has only so many light ones to give.
        That is fine while every element with a negative count is one the peptide
        itself supplies, which is the usual case (C, H, N, O).  A tag built from an
        element amino acids do not contain would break there, and nothing checks
        for it.
        """
        if self.n_channels < 2:
            raise JModError(f"A decoy channel needs at least two channels to take its step from; "
                            f"{self.name} is searching {self.channel_names}")
        try:
            numbers = [int(c) for c in self.channel_names]
        except ValueError:
            raise JModError(f"A decoy channel can only be named for a tag whose channel names are "
                            f"numbers; {self.name}'s are {self.channel_names}")
        decoy_name = str(numbers[-1] + (numbers[1] - numbers[0]))
        if decoy_name in self.channel_names:
            raise JModError(f"The decoy channel {self.name}-{decoy_name} is already being searched: "
                            f"channels {self.channel_names}")

        if self.channel_comp is not None:
            step = self.channel_comp[self.channel_names[1]] - self.channel_comp[self.channel_names[0]]
            decoy_comp = self.channel_comp[self.channel_names[-1]] + step
            decoy_delta = mass.calculate_mass(decoy_comp) - self.mass
        else:
            decoy_comp = None
            decoy_delta = self.delta[-1] + (self.delta[1] - self.delta[0])
        compositions = None if self.channel_comp is None else {**self.channel_comp, decoy_name: decoy_comp}

        tag = massTag(self.rules, self.mass, list(self.delta) + [decoy_delta],
                      list(self.channel_names) + [decoy_name], self.name,
                      compositions=compositions)
        # As in select_channels, mass_dict keeps every channel the tag defines,
        # now including the decoy one
        tag.mass_dict = {**self.mass_dict, f"{self.name}-{decoy_name}": self.mass + decoy_delta}
        tag.decoy_channel = decoy_name
        return tag


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
            rules_error = tag_rules_error(mass_tag_data['rules'])
            if rules_error:
                logging.getLogger("GUI").warning(rules_error)
                return mass_tag_data['name']
            # The search reads channels back out of sequences with f"{tag.name}-(\d+)"
            if not all(channel.isdigit() for channel in mass_tag_data['channel_names']):
                logging.getLogger("GUI").warning(f"Channel names must be numbers: {mass_tag_data['channel_names']}")
                return mass_tag_data['name']
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
            raise JModError("Unknown Tag Rule")
        all_tag_pos += tag_pos
        additional_tag_masses[tag_pos]+=1

    return all_tag_pos, additional_tag_masses

def tag_rules_error(rules):
    """An error message if *rules* is not "n" (N-terminus) and/or one-letter
    amino acid codes, each once (e.g. "nK", "STY"), else None"""
    if not rules or any(r not in "nACDEFGHIKLMNPQRSTVWY" for r in rules) or len(set(rules)) != len(rules):
        return f"Invalid tag rules '{rules}': use n for the N-terminus and/or one-letter amino acid codes, each once (e.g. nK, R, STY)"
    return None

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
    for filename in os.listdir(mass_tags_dir):
        if os.path.splitext(filename)[1].lower() == ".json":
            mass_tag = read_json_to_massTag(mass_tags_dir, filename)
            if type(mass_tag) == str:
                logging.getLogger("GUI").warning(f"Unable to load mass tag from {filename}\n")
            elif mass_tag:
                available_tags[mass_tag.name] = mass_tag
    return available_tags

available_tags = refresh_tags()


def set_config_tag(available_tags, config_args_tag, channels=None):
    """The tag named *config_args_tag*, searching only *channels* ("0,4" or a list; None for all)"""
    if not config_args_tag or config_args_tag == "None":
        return None
    if config_args_tag not in available_tags:
        raise JModError(f"Tag '{config_args_tag}' not in available tags:\n{list(available_tags.keys())}")
    if channels is None:
        return available_tags[config_args_tag]
    return available_tags[config_args_tag].select_channels(channels.split(",") if isinstance(channels, str) else [str(c) for c in channels])


def resolve_decoy_channel(mass_tag, setting):
    """*mass_tag* with its decoy channel resolved from *setting*, the
    --decoy_channel option.

    "none" leaves the tag alone.  "auto" appends a channel one step past the last
    one being searched (massTag.with_decoy_channel).  Any other value names a
    channel that is already being searched, which is then treated as the empty
    one; a channel the search is not covering is an error, since the decoy
    channel has to be searched like any other.

    Called from build_library, after a pre-tagged library's own channel has been
    matched against the tag: the decoy channel is one JMod invented, so it must
    not be in mass_dict while that match is looking for the library's channel.
    """
    if mass_tag is None or not setting or str(setting).lower() in ("none", "off"):
        return mass_tag
    if mass_tag.decoy_channel is not None:
        return mass_tag  # already resolved; the MBR pass builds a second library from the same tag
    if str(setting).lower() == "auto":
        # "auto" is the default, so it has to mean "if there is one to make": a tag
        # searching a single channel has no step to take one from, and no plex to
        # score channels within either
        if mass_tag.n_channels < 2:
            logger.info(f"No decoy channel: {mass_tag.name} is searching one channel "
                        f"({mass_tag.channel_names[0]})")
            return mass_tag
        mass_tag = mass_tag.with_decoy_channel()
    elif str(setting) in mass_tag.channel_names:
        mass_tag = copy.copy(mass_tag)
        mass_tag.decoy_channel = str(setting)
    else:
        raise JModError(f"Decoy channel '{setting}' is not one of the channels being searched "
                        f"({', '.join(mass_tag.channel_names)}). Use --decoy_channel auto to add a "
                        f"channel one step past the last one, 'none' for no decoy channel, or add "
                        f"the channel to --channels.")
    decoy_mass = mass_tag.mass_dict[f"{mass_tag.name}-{mass_tag.decoy_channel}"]
    logger.info(f"Decoy channel: {mass_tag.name}-{mass_tag.decoy_channel} at "
                f"{decoy_mass:.5f} (delta {decoy_mass - mass_tag.mass:+.5f})")
    return mass_tag


config.tag = set_config_tag(available_tags, config.args.tag, config.args.channels)
