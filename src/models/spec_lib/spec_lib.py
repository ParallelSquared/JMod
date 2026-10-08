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
from dataclasses import dataclass
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


def has_mass_tag(modified_peptides, prec_mzs, prec_zs):
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
        Back-calculated mass of the tag channel from the first tagged
        peptide found, and whether any tag was found at all. Returns
        (0, False, None) if no tag is present.
    """
    diann_mods = config.diann_mods
    unknown = sorted(library_mod_names(modified_peptides) - set(diann_mods))
    if not unknown:
        return 0, False, None
    if len(unknown) > 1:
        tag_names = {name.rsplit("-", 1)[0] for name in unknown}
        if len(tag_names) == 1:
            raise JModError(f"The spectral library contains several channels of one tag "
                            f"({', '.join(unknown)}). Multi-channel libraries are not supported")
        raise JModError(f"The spectral library contains {len(unknown)} modifications JMod has no "
                        f"mass for: {', '.join(unknown)}. JMod takes one unknown modification "
                        f"to be the library's tag; more than one is not supported")

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
# peptide N-terminus plus any residues, e.g. "nK".  --add_fixed_mod skips a
# site that already carries another modification (with a warning), unless
# the spec ends in ",stack": --add_fixed_mod UniMod:121,K,stack.
# ---------------------------------------------------------------------------

_RESIDUES = "ACDEFGHIKLMNPQRSTVWY"


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
        return f"{self.name} ({self.mass:+.6f} Da) at {self.sites}{stacked}"


@dataclass(frozen=True)
class ModEdits:
    """The modifications to strip from the library, then the ones to add."""
    strip: tuple = ()
    add: tuple = ()

    def __bool__(self):
        return bool(self.strip or self.add)


def resolve_mod_edits(add_fixed_mod, strip_mod):
    """The --add_fixed_mod and --strip_mod modifications (each a string, a list
    of strings, or None), checked.  Their masses are added to
    config.diann_mods, so that every modification in the edited library has
    a mass."""
    edits = ModEdits(
        strip=tuple(parse_mod_spec(text, "--strip_mod") for text in _as_list(strip_mod)),
        add=tuple(parse_mod_spec(text, "--add_fixed_mod") for text in _as_list(add_fixed_mod)),
    )
    for spec in edits.strip + edits.add:
        config.diann_mods[spec.name] = spec.mass
    for spec in edits.strip:
        logger.info(f"Stripping from the library: {spec}")
    for spec in edits.add:
        logger.info(f"Adding to the library (fixed): {spec}")
    return edits


def _as_list(value):
    if value is None:
        return []
    return [value] if isinstance(value, str) else list(value)


def parse_mod_spec(text, flag):
    """A ModSpec from NAME,MASS,SITES or NAME,SITES, either followed by
    ",stack" for --add_fixed_mod.  Raises JModError.

    A modification JMod knows (known_mod_mass) gets the known mass, whatever
    MASS says: one name has one mass for the whole experiment."""
    parts = [part.strip() for part in str(text).split(",")]
    stack = len(parts) > 1 and parts[-1].lower() == "stack"
    if stack:
        if flag != "--add_fixed_mod":
            raise JModError(f"{flag} {text}: stack is only for --add_fixed_mod")
        parts = parts[:-1]
    # Libraries spell UniMod names "UniMod:N"
    parts[0] = re.sub(r"^unimod:", "UniMod:", parts[0], flags=re.IGNORECASE)
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
            raise JModError(f"{flag} {text}: JMod has no mass for {name}; give it as "
                            f"{name},MASS,{sites}")
    else:
        raise JModError(f"{flag} {text}: expected NAME,MASS,SITES or NAME,SITES (then ,stack "
                        f"to add to sites that have another modification), e.g. UniMod:4,C or "
                        f"Dimethyl,28.0313,nK")

    if not name or re.search(r"[()\[\]\s]", name):
        raise JModError(f"{flag} {text}: '{name}' is not a modification name (no spaces or brackets)")
    unknown_sites = sorted(set(sites) - set("n" + _RESIDUES))
    if not sites or unknown_sites:
        raise JModError(f"{flag} {text}: sites must be n (the N-terminus) and/or residues "
                        f"({_RESIDUES}), e.g. nK; got '{sites}'")
    return ModSpec(name=name, mass=mass, sites="".join(dict.fromkeys(sites)), stack=stack)


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
# library has, where, and what JMod would make of them when loading it.
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
    source: str           # "binary cache" or "file"
    n_precursors: int
    charges: tuple        # (lowest, highest)
    mz_range: tuple       # (lowest, highest)
    has_ion_mobility: bool
    mods: tuple           # LibraryModRow, most precursors first
    problems: tuple       # what loading the library would raise


def inspect_library(lib_file, mass_tag=None):
    """A LibraryReport for *lib_file*: its precursors' modifications, and the
    problems JMod would raise when loading it with *mass_tag* (None for no
    tag).  Reads only the precursor columns, from the binary cache when it is
    current, otherwise from the file."""
    if not os.path.exists(lib_file):
        raise JModError(f"Spectral library not found: {lib_file}")
    precursors, source = _library_precursors(lib_file)
    seqs = precursors["mod_seq"]
    counts = _mod_site_counts(seqs)

    rows, unknown, problems = [], [], []
    for name, site, n in counts.iter_rows():
        try:
            mass = known_mod_mass(name)
            status = "known" if mass is not None else "unknown: no mass"
        except JModError as e:
            mass, status = None, str(e)
        if mass is None and name not in unknown:
            unknown.append(name)
        rows.append([name, site, n, mass, status])

    if len(unknown) == 1:
        tag_name = unknown[0]
        tag_mass, _, _ = has_mass_tag(seqs, precursors["prec_mz"], precursors["prec_z"])
        if mass_tag is None:
            status = f"tag ({tag_mass:.4f} Da)"
            problems.append(f"{tag_name} is the library's tag (or a modification JMod has no mass "
                            f"for), and no tag is selected: loading raises")
        else:
            channel = int(np.argmin(np.abs(mass_tag.channel_masses - tag_mass)))
            delta = mass_tag.channel_masses[channel] - tag_mass
            status = (f"tag ({tag_mass:.4f} Da) -> {mass_tag.name}-{mass_tag.channel_names[channel]} "
                      f"(difference {delta:+.6f} Da)")
            if "n" in mass_tag.rules:
                n_old, example = _misplaced_nterm_tags(seqs, rf"\({re.escape(tag_name)}\)", mass_tag.rules)
                if n_old:
                    problems.append(f"{n_old:,} precursors have the N-terminal tag behind the first "
                                    f"residue (e.g. {example}): loading raises")
        for row in rows:
            if row[0] == tag_name:
                row[4] = status
    elif len(unknown) > 1:
        try:
            has_mass_tag(seqs, precursors["prec_mz"], precursors["prec_z"])
        except JModError as e:
            problems.append(f"{e}: loading raises")

    charges = precursors["prec_z"]
    mz = precursors["prec_mz"]
    return LibraryReport(
        path=lib_file, source=source, n_precursors=len(precursors),
        charges=(int(charges.min()), int(charges.max())) if len(precursors) else (0, 0),
        mz_range=(float(mz.min()), float(mz.max())) if len(precursors) else (0.0, 0.0),
        has_ion_mobility=bool(precursors["ion_mob"].is_not_nan().any()) if len(precursors) else False,
        mods=tuple(LibraryModRow(*row) for row in rows),
        problems=tuple(problems),
    )


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
             f"m/z {low_mz:.1f}-{high_mz:.1f}, ion mobility: {'yes' if report.has_ion_mobility else 'no'} "
             f"(read from the {report.source})",
             ""]
    if report.mods:
        table = [("Modification", "Where", "Precursors", "Mass (Da)", "Status")]
        for row in report.mods:
            table.append((row.name, row.site, f"{row.n_precursors:,}",
                          f"{row.mass:+.6f}" if row.mass is not None else "-", row.status))
        widths = [max(len(r[c]) for r in table) for c in range(4)]
        for r in table:
            lines.append("  ".join([r[0].ljust(widths[0]), r[1].ljust(widths[1]),
                                    r[2].rjust(widths[2]), r[3].rjust(widths[3]), r[4]]).rstrip())
    else:
        lines.append("No modifications")
    lines.append("")
    if report.problems:
        lines.append("Problems when loading:")
        lines += [f"  - {p}" for p in report.problems]
    else:
        lines.append("No problems found when loading")
    return lines


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
        
    