#  Copyright (c) 2026. Parallel Squared Technology Institute
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

import numpy as np
import os
import csv
import pandas as pd
import sqlite3
import struct
import zlib
import pickle
from src.utils.errors import JModError
from src.utils.misc_functions import  frag_to_peak
from src.utils.parse_peptides import change_seq, convert_frags
import copy
import src.config as config
from src.logger import logger
import re
import logging
from dataclasses import dataclass, replace
import polars as pl
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
                raise JModError("Unknown retention time column")
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

def library_mod_names(modified_peptides):
    """The distinct parenthetical modification names in *modified_peptides*,
    e.g. {"UniMod:4", "PSMtag-0"}.  Vectorised: under a second for a
    library of a few million precursors."""
    names = (pl.Series(modified_peptides, dtype=pl.String)
             .str.extract_all(_MOD_PATTERN.pattern)
             .explode().drop_nulls().unique())
    return {name[1:-1].strip() for name in names}


def has_mass_tag(modified_peptides, prec_mzs, prec_zs, known_mods=None):
    """
    Returns True if any ModifiedPeptide string contains a non-UniMod
    parenthetical modification (a mass tag),
    along with the back-calculated mass of that tag.

    The library may hold one modification that config.diann_mods does not
    know, which is its tag.  More than one raises a JModError: several
    channels of one tag (multi-channel libraries are not supported), or
    modifications JMod has no mass for.

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
        Back-calculated mass of the tag channel (the median over the first
        10 precursors carrying it), and whether any tag was found at
        all. Returns (0, False, None) if no tag is present.
    """
    # *known_mods*: the mod table to use, config.diann_mods unless given
    diann_mods = config.diann_mods if known_mods is None else known_mods
    unknown = sorted(library_mod_names(modified_peptides) - set(diann_mods))
    if not unknown:
        return 0, False, None
    if len(unknown) > 1:
        # Channels of one tag: the tag's name, a dash and the channel number
        channels = [re.fullmatch(r"(.+)-\d+", name) for name in unknown]
        if all(channels) and len({channel.group(1) for channel in channels}) == 1:
            raise JModError(f"The spectral library contains several channels of one tag "
                            f"({', '.join(unknown)}). Multi-channel libraries are not supported")
        raise JModError(f"The spectral library contains {len(unknown)} modifications JMod has no "
                        f"mass for: {', '.join(unknown)}. JMod takes one unknown modification "
                        f"to be the library's tag; more than one is not supported. Give their "
                        f"masses with --add_fixed_mod or --strip_mod")

    # The tag's mass from the precursors carrying it: the median over up to
    # 10 of them, so one badly written precursor cannot decide it
    tag_name = unknown[0]
    # The first 10 found: in a tagged library that is almost at once
    annotation = f"({tag_name})"
    tagged = []
    for i, pep in enumerate(modified_peptides):
        if annotation in pep:
            tagged.append(i)
            if len(tagged) == 10:
                break
    prec_mzs, prec_zs = np.asarray(prec_mzs, dtype=np.float64), np.asarray(prec_zs, dtype=np.float64)
    estimates = []
    for i in tagged:
        pep, prec_mz, prec_z = modified_peptides[i], prec_mzs[i], prec_zs[i]
        all_mods = [m.strip() for m in _MOD_PATTERN.findall(pep)]
        tag_mods = [m for m in all_mods if m not in diann_mods]
        known_mods_attached = [m for m in all_mods if m in diann_mods]
        stripped_peptide = re.sub(r"\([^)]*\)", "", pep)
        stripped_mass = mass.fast_mass(stripped_peptide)
        known_mods_mass = sum(diann_mods[m] for m in known_mods_attached)

        untagged_mz = (stripped_mass + known_mods_mass + prec_z * 1.00727647) / prec_z
        mz_delta = prec_mz - untagged_mz
        estimates.append((mz_delta * prec_z) / len(tag_mods))

    tag_mass = float(np.median(estimates)) if estimates else np.nan
    if np.isnan(tag_mass):
        raise JModError(f"The mass of the library's tag ({tag_name}) could not be worked out from its "
                        f"precursors' m/z")
    return tag_mass, True, tag_name


# A library's tag mass this far from the tag's closest channel: a warning
# (e.g. one atom of another isotope, 13C for 15N: 0.0063 Da), and an error
# (another label, e.g. SILAC +8 for dimethyl +8: 0.030 Da; an average mass;
# a wrong tag or channel).  Rounding in a library's precursor m/z is at most
# about 0.0002 Da; channels are at least 1 Da apart.
TAG_MASS_WARNING = 0.002
TAG_MASS_ERROR = 0.01


def match_tag_channel(mass_tag, tag_name, tag_mass):
    """The channel of *mass_tag* closest to the library's tag (*tag_name*, of
    back-calculated mass *tag_mass*), as (channel name, its mass minus
    *tag_mass*, warning or None).  Raises a JModError when the closest
    channel is more than TAG_MASS_ERROR away."""
    index = int(np.argmin(np.abs(mass_tag.channel_masses - tag_mass)))
    channel = f"{mass_tag.name}-{mass_tag.channel_names[index]}"
    channel_mass = float(mass_tag.channel_masses[index])
    difference = channel_mass - tag_mass
    if abs(difference) > TAG_MASS_ERROR:
        raise JModError(f"The library's tag ({tag_name}, {tag_mass:.4f} Da) is {abs(difference):.4f} Da "
                        f"from the closest channel of {mass_tag.name} ({channel}, {channel_mass:.4f} Da): "
                        f"it is not this tag. Choose the library's tag, or strip it with --strip_mod")
    warning = None
    if abs(difference) > TAG_MASS_WARNING:
        warning = (f"The library's tag ({tag_name}, {tag_mass:.4f} Da) is {abs(difference):.4f} Da from "
                   f"{channel} ({channel_mass:.4f} Da). Every channel's precursor and fragment masses "
                   f"will be off by about that much")
    return channel, difference, warning

def check_nterm_tags(modified_peptides, tag):
    """Raise a JModError if the library writes the N-terminal *tag* behind
    the first residue, as older JMod versions did ("P(tag-0)EPTIDEK(tag-0)"),
    instead of in front of it ("(tag-0)PEPTIDEK(tag-0)").  Decoys would
    otherwise move that tag away from the N-terminus.

    A tag copy on the first residue beyond what the tag's residue rules put
    there is a misplaced N-terminal one.  Tags without an "n" rule pass.
    """
    if "n" not in tag.rules:
        return
    seqs = pl.Series(modified_peptides, dtype=pl.String)
    n_misplaced, example = _misplaced_nterm_tags(seqs, rf"\({re.escape(tag.name)}-[^()]*\)", tag.rules)
    if n_misplaced:
        raise JModError(f"{n_misplaced:,} spectral library precursors have their N-terminal tag "
                        f"behind the first residue (e.g. {example}), as older JMod versions wrote "
                        f"them. Write N-terminal tags in front of the first residue: "
                        f"({tag.name}-0)PEPTIDEK({tag.name}-0)")


def _misplaced_nterm_tags(seqs, tag_pattern, rules):
    """(how many, an example) of *seqs* whose first residue carries more copies
    of the tag (the regex *tag_pattern*) than the tag's residue *rules* put
    there: an N-terminal tag written behind the first residue."""
    first_residue = seqs.str.extract(r"^([A-Z](?:\([^()]*\))*)", 1)
    tag_copies = first_residue.str.count_matches(tag_pattern)
    residue_tags = first_residue.str.slice(0, 1).is_in(list(rules.replace("n", ""))).cast(pl.UInt32)
    misplaced = (tag_copies > residue_tags).fill_null(False)
    n_misplaced = int(misplaced.sum())
    return n_misplaced, (seqs.filter(misplaced)[0] if n_misplaced else None)


# ---------------------------------------------------------------------------
# --add_fixed_mod and --strip_mod: fixed modifications added to, or removed
# from, every library entry when the library is loaded (loadSpecLib, through
# SpectrumLibraryStore.edit_mods).  A modification is NAME,MASS,SITES, or
# NAME,SITES when JMod knows its mass (UniMod:N, or one of config.diann_mods);
# a known modification always keeps the mass JMod knows.  SITES is n for the
# peptide N-terminus plus any residues, e.g. "nK"; --strip_mod NAME (or
# NAME,MASS) strips it from every site.  --add_fixed_mod skips a
# site that already carries another modification (with a warning), unless
# the spec ends in ",stack": --add_fixed_mod UniMod:121,K,stack.
# ---------------------------------------------------------------------------

# The residues the library parser accepts (library_store._STANDARD_RESIDUES),
# so "every site" includes U, O and J too
_RESIDUES = "ACDEFGHIJKLMNOPQRSTUVWY"
# Every site: --strip_mod NAME (or NAME,MASS) strips it wherever it is
_ALL_SITES = "n" + _RESIDUES


@dataclass(frozen=True)
class ModSpec:
    """One modification: its name as the library writes it, e.g. "UniMod:4",
    its mass, and its sites ("n" for the N-terminus, plus residues).  *stack*:
    added to a site even when it already carries another modification
    (--add_fixed_mod NAME,...,stack); otherwise such sites are skipped."""
    name: str
    mass: float
    sites: str
    stack: bool = False

    @property
    def annotation(self):
        return f"({self.name})"

    @property
    def nterm(self):
        return "n" in self.sites

    @property
    def residues(self):
        return self.sites.replace("n", "")

    def __str__(self):
        stacked = ", stacked on other modifications" if self.stack else ""
        where = "every site" if self.sites == _ALL_SITES else self.sites
        return f"{self.name} ({self.mass:+.6f} Da) at {where}{stacked}"


@dataclass(frozen=True)
class ModEdits:
    """The modifications to strip from the library, then the fixed ones to
    add, then the variable ones: each precursor also gets a copy for every
    combination of 1 to *max_variable* of their sites."""
    strip: tuple = ()
    add: tuple = ()
    variable: tuple = ()
    max_variable: int = 2

    def __bool__(self):
        return bool(self.strip or self.add or self.variable)


def resolve_mod_edits(add_fixed_mod, strip_mod, add_variable_mod=None, max_variable_mods=2):
    """parse_mod_edits, with the modifications' masses added to
    config.diann_mods, so that every modification in the edited library has a
    mass, and logged."""
    edits = parse_mod_edits(add_fixed_mod, strip_mod, add_variable_mod, max_variable_mods)
    for spec in edits.strip + edits.add + edits.variable:
        config.diann_mods[spec.name] = spec.mass
    for spec in edits.strip:
        logger.info(f"Stripping from the library: {spec}")
    for spec in edits.add:
        logger.info(f"Adding to the library (fixed): {spec}")
    for spec in edits.variable:
        logger.info(f"Adding to the library (variable, up to {edits.max_variable} per precursor): {spec}")
    return edits


def parse_mod_edits(add_fixed_mod, strip_mod, add_variable_mod=None, max_variable_mods=2):
    """The --add_fixed_mod, --strip_mod and --add_variable_mod modifications
    (each a string, a list of strings, or None), checked, with
    --max_variable_mods, as a ModEdits.  Changes nothing: resolve_mod_edits
    also registers their masses."""
    try:
        max_variable_mods = int(max_variable_mods)
    except (TypeError, ValueError):
        raise JModError(f"--max_variable_mods must be a whole number, not {max_variable_mods}")
    if max_variable_mods < 1:
        raise JModError(f"--max_variable_mods must be at least 1, not {max_variable_mods}")
    given = {flag: [(text, parse_mod_spec(text, flag)) for text in _as_list(texts)]
             for flag, texts in (("--strip_mod", strip_mod), ("--add_fixed_mod", add_fixed_mod),
                                 ("--add_variable_mod", add_variable_mod))}
    _check_one_mass_per_name(given)
    return ModEdits(
        strip=tuple(spec for _, spec in given["--strip_mod"]),
        add=tuple(spec for _, spec in given["--add_fixed_mod"]),
        variable=tuple(spec for _, spec in given["--add_variable_mod"]),
        max_variable=max_variable_mods,
    )


def _check_one_mass_per_name(given):
    """Raise a JModError if two modifications in *given* ({flag: [(text,
    ModSpec)]}) share a name but not a mass.  A name has one mass for the
    whole experiment: decoys and the search look masses up by name."""
    first = {}
    for flag, specs in given.items():
        for text, spec in specs:
            if spec.name.lower() not in first:
                first[spec.name.lower()] = (flag, text, spec)
                continue
            other_flag, other_text, other = first[spec.name.lower()]
            if abs(other.mass - spec.mass) > 1e-4:
                raise JModError(f"{spec.name} is given two masses: {other.mass} in {other_flag} {other_text}, "
                                f"and {spec.mass} in {flag} {text}. A modification name has one mass for "
                                f"the whole experiment; give one of them another name, e.g. {spec.name}_2")


def _canonical_name(name, names):
    """*name* as spelled in *names* (a library's modifications, or the mod
    table): an exact match, else the one name that differs only in case, else
    *name* itself.  Two names that differ only in case raise a JModError."""
    if name in names:
        return name
    matches = sorted(other for other in names if other.lower() == name.lower())
    if len(matches) > 1:
        raise JModError(f"{name} could be any of {', '.join(matches)}, which differ only in case; "
                        f"give the name exactly as it is spelled")
    return matches[0] if matches else name


def match_library_spelling(edits, library_names, register=True):
    """*edits* (a ModEdits) with each modification named as the library spells
    it, among *library_names*: --strip_mod dimethyl-lys is the library's
    Dimethyl-Lys.  With *register*, their masses are added to
    config.diann_mods under that spelling."""
    def respell(spec):
        name = _canonical_name(spec.name, library_names)
        if name == spec.name:
            return spec
        if register:
            config.diann_mods[name] = spec.mass
        return replace(spec, name=name)
    return replace(edits, strip=tuple(map(respell, edits.strip)), add=tuple(map(respell, edits.add)),
                   variable=tuple(map(respell, edits.variable)))


def _as_list(value):
    if value is None:
        return []
    return [value] if isinstance(value, str) else list(value)


def parse_mod_spec(text, flag):
    """A ModSpec from NAME,MASS,SITES or NAME,SITES, either followed by
    ",stack" for --add_fixed_mod or --add_variable_mod.  --strip_mod also
    takes NAME or NAME,MASS: every site.  Raises JModError.

    A modification JMod knows (known_mod_mass) gets the known mass, whatever
    MASS says: one name has one mass for the whole experiment."""
    parts = [part.strip() for part in str(text).split(",")]
    stack = len(parts) > 1 and parts[-1].lower() == "stack"
    if stack:
        if flag not in ("--add_fixed_mod", "--add_variable_mod"):
            raise JModError(f"{flag} {text}: stack is only for --add_fixed_mod and --add_variable_mod")
        parts = parts[:-1]
    # Libraries spell UniMod names "UniMod:N"
    parts[0] = re.sub(r"^unimod:", "UniMod:", parts[0], flags=re.IGNORECASE)
    # Names are not case-sensitive: a known name gets the mod table's spelling
    # (and loadSpecLib matches the rest to the library's spelling)
    parts[0] = _canonical_name(parts[0], config.diann_mods)
    # --strip_mod NAME or NAME,MASS: every site
    if flag == "--strip_mod" and (len(parts) == 1 or (len(parts) == 2 and _is_number(parts[1]))):
        parts.append(_ALL_SITES)
    if len(parts) == 3:
        name, mass_text, sites = parts
        try:
            mass = float(mass_text)
        except ValueError:
            raise JModError(f"{flag} {text}: the mass '{mass_text}' is not a number")
        known = known_mod_mass(name)
        if known is not None:
            # A mass that only rounds the known one (57.0215 for 57.021464) is not worth a warning
            if abs(known - mass) > 1e-4:
                logger.warning(f"{flag} {text}: {name} is {known:.6f} Da; the {mass_text} given is ignored")
            mass = known
    elif len(parts) == 2:
        name, sites = parts
        mass = known_mod_mass(name)
        if mass is None:
            spelled = f"{name},MASS" if sites == _ALL_SITES else f"{name},MASS,{sites}"
            raise JModError(f"{flag} {text}: JMod has no mass for {name}; give it as {spelled}")
    else:
        raise JModError(f"{flag} {text}: expected NAME,MASS,SITES or NAME,SITES (then ,stack "
                        f"to add to sites that have another modification), e.g. UniMod:4,C or "
                        f"Dimethyl,28.0313,nK")

    if not name or re.search(r"[()\[\]\s]", name):
        raise JModError(f"{flag} {text}: '{name}' is not a modification name (no spaces or brackets)")
    # Sites are not case-sensitive either, except n: lowercase n is the
    # N-terminus, uppercase N asparagine
    sites = "".join(site if site == "n" else site.upper() for site in sites)
    unknown_sites = sorted(set(sites) - set("n" + _RESIDUES))
    if not sites or unknown_sites:
        raise JModError(f"{flag} {text}: sites must be n (the N-terminus) and/or residues "
                        f"({_RESIDUES}), e.g. nK; got '{sites}'")
    return ModSpec(name=name, mass=mass, sites="".join(dict.fromkeys(sites)), stack=stack)


def _is_number(text):
    try:
        float(text)
        return True
    except ValueError:
        return False


def known_mod_mass(name):
    """The mass of a modification JMod knows: one of config.diann_mods, or
    UniMod:N.  None for any other name; an unknown UniMod number raises."""
    if name in config.diann_mods:
        return config.diann_mods[name]
    unimod = re.fullmatch(r"UniMod:(\d+)", name, flags=re.IGNORECASE)
    if unimod is None:
        return None
    from src.iso_functions import unimods
    try:
        return float(unimods.by_id(int(unimod.group(1)))["mono_mass"])
    except KeyError:
        raise JModError(f"{name} is not in UniMod")


# ---------------------------------------------------------------------------
# --inspect_library (and the GUI's library "i" button): which modifications a
# library has and where, what --strip_mod / --add_fixed_mod / --add_variable_mod
# would do to them, and whether JMod could then load it.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LibraryModRow:
    """One modification at one kind of site: N-term, or a residue letter."""
    name: str
    site: str
    n_precursors: int
    mass: float       # None when JMod has no mass for it
    status: str


@dataclass(frozen=True)
class LibraryReport:
    path: str
    source: str           # read from the "binary cache" or the "file"
    n_precursors: int
    charges: tuple        # (lowest, highest)
    mz_range: tuple       # (lowest, highest)
    has_ion_mobility: bool
    mods: tuple           # LibraryModRow, in the library, most precursors first
    # The edits' effect, worked out on a random sample of the precursors
    # (sample_size of them; every precursor when sample_size == n_precursors)
    sample_size: int
    edit_lines: tuple     # (logging level, message): what the edits do; () without edits
    edited_mods: tuple    # LibraryModRow after the edits, in the sample; () without edits
    n_after: int          # precursors after the edits (estimated from the sample)
    # Exact, from the whole library
    leftovers: tuple      # modifications a --strip_mod leaves at sites it does not cover
    warnings: tuple       # what loading it would warn about
    problems: tuple       # what would stop JMod loading it (after the edits)


def inspect_library(lib_file, mass_tag=None, edits=None, sample_size=20_000, sample_above=200_000):
    """A LibraryReport for *lib_file*: its precursors' modifications, what the
    --strip_mod / --add_fixed_mod / --add_variable_mod *edits* (a ModEdits,
    or None) would do to them, and what would stop JMod loading it with
    *mass_tag* (None for no tag).  The edits follow the same rules as
    SpectrumLibraryStore.edit_mods.  Reads only the precursor columns, from
    the binary cache when it is current, otherwise from the file; changes
    nothing, not even the mod table.

    What would stop JMod loading the library, and what a --strip_mod would
    leave behind, are exact: they depend only on which modifications are
    where.  The edits' counts, and the table after them, come from every
    precursor, or for a library of more than *sample_above* precursors from a
    random sample of *sample_size* of them, which keeps the report quick."""
    if not os.path.exists(lib_file):
        raise JModError(f"Spectral library not found: {lib_file}")
    precursors, source = _library_precursors(lib_file)
    # The masses JMod would know: its table, plus the edits' modifications,
    # named as the library spells them
    masses = dict(config.diann_mods)
    if edits:
        edits = match_library_spelling(edits, library_mod_names(precursors["mod_seq"]), register=False)
        masses.update({spec.name: spec.mass for spec in edits.strip + edits.add + edits.variable})

    seqs, mzs, zs = precursors["mod_seq"], precursors["prec_mz"], precursors["prec_z"]
    mods = _mod_rows(seqs, masses)
    # Every name an edit gives has a mass, and the edits never move the tag:
    # whether JMod can load the library follows from the library as it is
    problems, warnings = _loading_problems(mods, seqs, mzs, zs, masses, mass_tag)

    n = len(precursors)
    n_sample = min(n, sample_size) if n > sample_above else n
    edit_lines, edited_mods, n_after, leftovers = (), (), n, ()
    if edits:
        sample = precursors.sample(n_sample, seed=0) if n_sample < n else precursors
        s_seqs, _, _, edit_lines = _edited_precursors(sample["mod_seq"], sample["prec_mz"],
                                                       sample["prec_z"], edits)
        edited_mods = _mod_rows(s_seqs, masses)
        # The edits leave the tag as it is: it keeps its status from the library's table
        tag_status = {row[0]: row[4] for row in mods if row[4].startswith("tag")}
        for row in edited_mods:
            row[4] = tag_status.get(row[0], row[4])
        n_after = round(n * len(s_seqs) / n_sample) if n_sample else n
        leftovers = _strip_leftovers(mods, edits)

    return LibraryReport(
        path=lib_file, source=source, n_precursors=n,
        charges=(int(precursors["prec_z"].min()), int(precursors["prec_z"].max())) if n else (0, 0),
        mz_range=(float(precursors["prec_mz"].min()), float(precursors["prec_mz"].max())) if n else (0.0, 0.0),
        has_ion_mobility=bool(precursors["ion_mob"].is_not_nan().any()) if n else False,
        mods=tuple(LibraryModRow(*row) for row in mods),
        sample_size=n_sample,
        edit_lines=tuple(edit_lines),
        edited_mods=tuple(LibraryModRow(*row) for row in edited_mods),
        n_after=n_after,
        leftovers=tuple(leftovers),
        warnings=tuple(warnings),
        problems=tuple(problems),
    )


def _strip_leftovers(rows, edits):
    """For each --strip_mod, where its modification stays: the library's
    (modification, site) *rows* at sites the spec does not list."""
    leftovers = []
    for spec in edits.strip:
        missed = [(site, n) for name, site, n, _, _ in rows
                  if name == spec.name and not (spec.nterm if site == "N-term" else site in spec.residues)]
        if not missed:
            continue
        shown = ", ".join(f"{site} ({n:,})" for site, n in missed)
        everywhere = spec.name if known_mod_mass(spec.name) is not None else f"{spec.name},{spec.mass}"
        leftovers.append(f"--strip_mod {spec.name} ({spec.sites}) leaves {spec.name} on "
                         f"{len(missed)} other sites (precursors): {shown}. To strip it everywhere, "
                         f"give no sites: --strip_mod {everywhere}")
    return leftovers


def _mod_rows(seqs, masses):
    """[name, site, precursors, mass, status] per (modification, site) in *seqs*,
    with the masses in *masses*, the mod table loading would use."""
    rows = []
    for name, site, n in _mod_site_counts(seqs).iter_rows():
        if name in masses:
            rows.append([name, site, n, masses[name], "known"])
            continue
        try:
            unimod_mass = known_mod_mass(name)
            status = ("unknown: no mass" if unimod_mass is None else
                      f"unknown: not in JMod's mod table (UniMod lists it as {unimod_mass:+.6f} Da)")
        except JModError as e:
            status = f"unknown: {e}"
        rows.append([name, site, n, None, status])
    return rows


def _edited_precursors(seqs, mzs, zs, edits):
    """The precursors (mod_seq, prec_mz, prec_z) after *edits*, by the rules
    edit_mods uses, and its summary of what they did."""
    from src.models.spec_lib.library_store import (
        _EditCounts, _edit_sequence, _variable_copies, _edit_summary,
    )
    counts = _EditCounts(edits)
    seq_list = seqs.to_list()
    mz = mzs.to_numpy().astype(np.float64)
    z = zs.to_numpy().astype(np.float64)
    if edits.strip or edits.add:
        for i, seq in enumerate(seq_list):
            edited = _edit_sequence(seq, edits, counts)
            if edited is not None:
                seq_list[i], delta = edited
                mz[i] += delta.sum() / z[i]
    copy_seqs, copy_mz, copy_z = [], [], []
    if edits.variable:
        for i, seq in enumerate(seq_list):
            for copy_seq, delta in _variable_copies(seq, edits.variable, edits.max_variable, counts):
                copy_seqs.append(copy_seq)
                copy_mz.append(mz[i] + delta.sum() / z[i])
                copy_z.append(z[i])
    frame = pl.DataFrame({"mod_seq": pl.Series(seq_list + copy_seqs, dtype=pl.String),
                          "prec_mz": np.concatenate([mz, copy_mz]),
                          "prec_z": np.concatenate([z, copy_z])})
    keep = frame.select(pl.struct("mod_seq", "prec_z").is_first_distinct()).to_series()
    n_duplicates = int((~keep).sum())
    frame = frame.filter(keep)
    lines = _edit_summary(edits, counts, len(seq_list), n_duplicates)
    return frame["mod_seq"], frame["prec_mz"], frame["prec_z"], lines


def _loading_problems(rows, seqs, mzs, zs, masses, mass_tag):
    """(What would stop JMod loading precursors *seqs* with *mass_tag*, the
    warnings loading would give), in plain words.  Marks the tag's *rows* with
    its back-calculated mass, and the channel it would be relabelled to."""
    unknown = sorted({row[0] for row in rows if row[3] is None})
    problems, warnings = [], []
    if len(unknown) == 1:
        tag_name = unknown[0]
        try:
            tag_mass, _, _ = has_mass_tag(seqs, mzs, zs, masses)
        except JModError as e:
            return [str(e)], warnings
        if mass_tag is None:
            status = f"tag? ({tag_mass:.4f} Da)"
            problems.append(f"{tag_name} has no mass in JMod, and no tag is selected. If it is the "
                            f"library's tag, choose that tag; otherwise give its mass with "
                            f"--add_fixed_mod {tag_name},MASS,SITES, or remove it with "
                            f"--strip_mod {tag_name},MASS")
        else:
            try:
                channel, difference, warning = match_tag_channel(mass_tag, tag_name, tag_mass)
                status = f"tag ({tag_mass:.4f} Da) -> {channel} (difference {difference:+.6f} Da)"
                if warning:
                    warnings.append(warning)
            except JModError as e:
                status = f"tag ({tag_mass:.4f} Da): not {mass_tag.name}"
                problems.append(str(e))
            if "n" in mass_tag.rules:
                n_old, example = _misplaced_nterm_tags(seqs, rf"\({re.escape(tag_name)}\)", mass_tag.rules)
                if n_old:
                    problems.append(f"{n_old:,} precursors have the N-terminal tag written behind the "
                                    f"first residue (e.g. {example}); it must be in front of it")
        for row in rows:
            if row[0] == tag_name:
                row[4] = status
    elif len(unknown) > 1:
        try:
            has_mass_tag(seqs, mzs, zs, masses)
        except JModError as e:
            problems.append(str(e))
    return problems, warnings


def _library_precursors(lib_file):
    """The library's target precursors (mod_seq, prec_z, prec_mz, ion_mob) and
    where they were read from.  The binary cache's arrays when it is current
    (np.load reads only the members asked for); otherwise the file's precursor
    columns, read the way the parser reads them."""
    from src.models.spec_lib.library_store import (
        STORE_VERSION, _LIBRARY_COLUMN_ALIASES, _LIBRARY_READERS, _STANDARD_RESIDUES,
    )
    store_file = lib_file + "_store.npz"
    if os.path.exists(store_file):
        cache = np.load(store_file, allow_pickle=True)
        if "store_version" in cache and int(cache["store_version"]) == STORE_VERSION:
            frame = pl.DataFrame({
                "mod_seq": pl.Series(cache["mod_seq"], dtype=pl.String),
                "prec_z": cache["prec_z"].astype(np.float64),
                "prec_mz": cache["prec_mz"].astype(np.float64),
                "ion_mob": cache["ion_mob"].astype(np.float64),
            })
            return frame, "binary cache"

    reader = _LIBRARY_READERS.get(lib_file.rsplit(".")[-1].lower())
    if reader is None:
        raise JModError(f"Cannot inspect a .{lib_file.rsplit('.')[-1]} library; use .tsv or .parquet")
    scan = reader(lib_file)
    present = scan.collect_schema().names()
    columns = {}
    for canonical in ("ModifiedPeptide", "PrecursorCharge", "PrecursorMz", "IonMobility", "Decoy"):
        name = next((a for a in _LIBRARY_COLUMN_ALIASES[canonical] if a in present), None)
        if name is not None:
            columns[canonical] = pl.col(name)
    if "ModifiedPeptide" not in columns or "PrecursorCharge" not in columns:
        raise JModError(f"{lib_file} has no modified-sequence or precursor-charge column")
    frame = scan.select(
        columns["ModifiedPeptide"].cast(pl.String).str.strip_chars("_")
        .str.replace_all("[", "(", literal=True).str.replace_all("]", ")", literal=True).alias("mod_seq"),
        columns["PrecursorCharge"].cast(pl.Float64).alias("prec_z"),
        (columns["PrecursorMz"].cast(pl.Float64) if "PrecursorMz" in columns
         else pl.lit(np.nan)).alias("prec_mz"),
        _ion_mobility(columns.get("IonMobility")).alias("ion_mob"),
        (columns["Decoy"].cast(pl.String).str.strip_chars().is_in(["1", "1.0", "True", "true"])
         if "Decoy" in columns else pl.lit(False)).alias("decoy"),
    ).filter(
        ~pl.col("decoy")
        # The parser leaves out precursors with residues that have no mass (X, Z, ...)
        & pl.col("mod_seq").str.replace_all(r"\([^()]*\)", "")
          .str.contains("^[" + "".join(sorted(_STANDARD_RESIDUES)) + "]+$")
    ).unique(["mod_seq", "prec_z"], keep="first").drop("decoy").collect()
    return frame, "file"


def _ion_mobility(column):
    """1/K0 as a float, NaN where missing: empty, or 0.0 as some libraries write it."""
    if column is None:
        return pl.lit(np.nan, dtype=pl.Float64)
    text = column.cast(pl.String).str.strip_chars()
    im = pl.when(text == "").then(None).otherwise(text).cast(pl.Float64)
    return pl.when(im == 0.0).then(None).otherwise(im).fill_null(np.nan)


def _mod_site_counts(seqs):
    """Per (modification, site): how many of *seqs* carry it.  The site is
    "N-term" for a modification in front of the first residue, else the
    residue it follows."""
    groups = pl.DataFrame({"i": np.arange(len(seqs)),
                           "group": seqs.str.extract_all(r"(?:^|[A-Z])(?:\([^()]*\))+")})
    groups = groups.explode("group").drop_nulls("group")
    parsed = []
    for group in groups["group"].unique().to_list():
        site = "N-term" if group.startswith("(") else group[0]
        for name in re.findall(r"\(([^()]*)\)", group):
            parsed.append((group, name.strip(), site))
    parsed = pl.DataFrame(parsed, schema={"group": pl.String, "name": pl.String, "site": pl.String},
                          orient="row")
    return (groups.join(parsed, on="group")
            .group_by("name", "site").agg(pl.col("i").n_unique().alias("n"))
            .sort(["n", "name", "site"], descending=[True, False, False])
            .select("name", "site", "n"))


def format_library_report(report):
    """*report* (a LibraryReport) as lines of text, for the log or a window."""
    low_z, high_z = report.charges
    low_mz, high_mz = report.mz_range
    lines = [report.path,
             f"{report.n_precursors:,} precursors, charge {low_z}-{high_z}, "
             f"m/z {low_mz:.1f}-{high_mz:.1f}, ion mobility: {'yes' if report.has_ion_mobility else 'no'}",
             ""]
    if report.edit_lines:
        sampled = report.sample_size < report.n_precursors
        lines.append("Modifications in the library:")
        lines += _mod_table(report.mods)
        lines += ["", f"What the edits do, in a random sample of {report.sample_size:,} precursors:"
                  if sampled else "What the edits do:"]
        lines += [f"  {'Warning: ' if level >= logging.WARNING else ''}{message.strip()}"
                  for level, message in report.edit_lines]
        lines += [f"  Warning: {leftover}" for leftover in report.leftovers]
        lines += ["", "Modifications after the edits, in the sample:" if sampled else
                      f"Modifications after the edits ({report.n_after:,} precursors):"]
        lines += _mod_table(report.edited_mods)
    else:
        lines += _mod_table(report.mods)
    lines.append("")
    if report.warnings:
        lines += [f"Warning: {warning}" for warning in report.warnings] + [""]
    if report.problems:
        lines.append("JMod will stop with an error when it loads this library:")
        lines += [f"  - {problem}" for problem in report.problems]
    else:
        lines.append("JMod can load this library" + (" with these edits" if report.edit_lines else ""))
    return lines


def _mod_table(rows):
    """Lines of a table of LibraryModRow *rows*."""
    if not rows:
        return ["No modifications"]
    table = [("Modification", "Where", "Precursors", "Mass (Da)", "Status")]
    for row in rows:
        table.append((row.name, row.site, f"{row.n_precursors:,}",
                      f"{row.mass:+.6f}" if row.mass is not None else "-", row.status))
    widths = [max(len(r[c]) for r in table) for c in range(4)]
    return ["  ".join([r[0].ljust(widths[0]), r[1].ljust(widths[1]), r[2].rjust(widths[2]),
                       r[3].rjust(widths[3]), r[4]]).rstrip() for r in table]


def in_windows(mz, ms2scans, margin_ppm=50.0):
    """Boolean mask of the m/z values that at least one isolation window of
    *ms2scans* covers.

    The windows are merged into disjoint intervals, each widened by a ppm
    margin for m/z alignment drift.
    """
    windows = sorted({(float(sc.ms1window[0]), float(sc.ms1window[1]))
                      for sc in ms2scans})
    merged = []
    for lo, hi in windows:
        lo -= lo * margin_ppm * 1e-6
        hi += hi * margin_ppm * 1e-6
        if merged and lo <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    lows = np.array([m[0] for m in merged])
    highs = np.array([m[1] for m in merged])

    mz = np.asarray(mz, dtype=np.float64)
    idx = np.searchsorted(lows, mz, side="right") - 1
    ok = idx >= 0
    ok[ok] = mz[ok] <= highs[idx[ok]]
    return ok


def loadSpecLib(lib_file, mod_edits=None):
    """Load a spectral library (from its binary cache when there is one), with
    the --strip_mod / --add_fixed_mod edits in *mod_edits* applied.  The cache
    holds the library as the file has it."""
    from src.models.spec_lib.library_store import SpectrumLibraryStore, StaleStoreCacheError

    lib_ext = lib_file.rsplit(".")[-1]

    logger.info("Loading Library...")
    store_file = lib_file + "_store.npz"

    spec_lib = None
    if os.path.exists(store_file):
        try:
            logger.info("Loading Library... from binary cache")
            spec_lib = SpectrumLibraryStore.load(store_file)
        except StaleStoreCacheError as e:
            logger.info(f"Binary cache is stale, re-parsing library ({e})")

    if spec_lib is None:
        logger.info("Loading Library... from file")
        if lib_ext == "blib":
            spec_lib = SpectrumLibraryStore.from_blib(lib_file)
        elif lib_ext == "tsv":
            spec_lib = SpectrumLibraryStore.from_tsv(lib_file)
        elif lib_ext == "parquet":
            spec_lib = SpectrumLibraryStore.from_parquet(lib_file)
        else:
            raise ValueError(f"Unsupported spectral library format: .{lib_ext}")
        spec_lib.save(store_file)

    if mod_edits:
        mod_edits = match_library_spelling(mod_edits, library_mod_names(spec_lib.mod_seq))
        spec_lib = spec_lib.edit_mods(mod_edits)

    source_channel_mass, library_tag_bool, library_tag_name = has_mass_tag(spec_lib.mod_seq, spec_lib.prec_mz, spec_lib.prec_z)
    if library_tag_bool:
        logger.info("Spectral Library Contains Tagged Peptides")

    # Ion mobility: a column that is entirely empty means the library simply
    # has no IM, which is normal; a partially populated one is corrupt and would
    # make the IM gates silently inconsistent between precursors, so it raises.
    # With that ruled out, library.has_ion_mobility answers the question for
    # every downstream IM gate.
    _ion_mob = np.asarray(spec_lib.ion_mob, dtype=float)
    _finite = np.isfinite(_ion_mob)
    if _finite.any() and not _finite.all():
        n_missing = int((~_finite).sum())
        raise ValueError(
            f"Library ion mobility is only partially populated: {n_missing} of "
            f"{_finite.size} precursors have no IM value. Expected either all or none."
        )
    if spec_lib.has_ion_mobility:
        logger.info("Spectral library contains ion mobility")

    logger.info(f"Loaded {len(spec_lib)} library precursors")
    return spec_lib, library_tag_bool, source_channel_mass, library_tag_name


def create_decoy_lib(library, rules, tag=None, n_iso=0):
    """Generate decoys and return a combined target+decoy SpectrumLibraryStore.

    Decoys are generated without tagging or isotope expansion — those
    operations should be applied to the combined store afterward.

    Returns a combined store with targets at [0, N) and decoys at [N, N+M).
    """
    from src.models.spec_lib.library_store import SpectrumLibraryStore
    return SpectrumLibraryStore.from_target_with_decoys(library, rules, tag=tag)
            
            
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
        
    