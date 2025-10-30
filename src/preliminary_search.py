import numpy as np
import peppy_sage as ps
import pandas as pd

import sys
from tqdm.auto import tqdm

from src.logger import logger

"""
Output we need:
    Retention time
    Retention time error
    mz observed/library/error

    e value

ignore fit to lib
"""


def fit_with_features(dia_spectra, library_spectra, mass_tag):
    # Construct modification dict to convert names to masses
    # Start with mass tag
    if mass_tag is not None:
        mod_dict = {'-'.join([mass_tag.name, mass_tag.channel_names[i]]): # construct channel names
                        mass_tag.mass + mass_tag.delta[i] # construct channel masses
                    for i in range(len(mass_tag.channel_names))}
    else:
        mod_dict = {}
    # Add all of the other supported modifications
    # TODO

    pep_seqs = [(v['seq'], v['mod_seq']) for v in library_spectra.values()]

    peps = [ps.Peptide(seq, peptide_to_mod_array(mod_seq, mod_dict)) for seq, mod_seq in pep_seqs]

    # Create indexed database
    db = ps.IndexedDatabase.from_peptides( # TODO pass parameters as parameters
        peptides=peps,
        bucket_size=128,
        ion_kinds=["b", "y"],
        min_ion_index=0,
        generate_decoys=True,
        decoy_tag="rev_",
        peptide_min_mass=0.0,
        peptide_max_mass=5000.0,
    )

    # Create scorer
    scorer = ps.Scorer(
        precursor_tol_da=(-1,1), # TODO placeholder
        fragment_tol_ppm=(-10,10),
        min_isotope_err=0,
        max_isotope_err=0,
        wide_window=True,
        chimera=False,
        annotate_matches=True,
        report_psms=10
    )

    # Convert spectra into a Rust-friendly format
    logger.info("Converting spectra")
    rust_specs = []
    for spec in tqdm(dia_spectra.ms2scans):
        rust_specs += [spec.to_rust_spectrum()]

    rust_specs = rust_specs#[15000:16000]

    # Process spectra in chunks of 1000
    # Smaller chunks increases the amount of time spent passing things back and forth between Python and Rust
    # Multi-threading happens spectrum-by-spectrum at the Rust level
    #   (not sure how Rust does concurrency, processing in groups of spectra may reduce overhead incurred by spinning
    #   up new threads)
    logger.info("Searching spectra")
    chunk_size = 1000
    hits = []
    for i in tqdm(range(0, len(rust_specs), chunk_size)):
        chunk = rust_specs[i:i + chunk_size]
        batch_hits = scorer.score_many(db, chunk)
        hits.extend(batch_hits)


    """
    for prelim_hits in hits:
        for rank_k in prelim_hits:
            print(rank_k)
            print(rank_k.fragments.to_dict())
            print()
            #print(rank_k.to_dict())
        print("****")
    """

    #rows = [feat.to_dict() for group in hits[16000:17000] for feat in group]
    rows = [feat.to_dict() for group in hits for feat in group]

    # turn into a DataFrame (fragments stays as a nested dict column)
    df = pd.DataFrame(rows)

    # save (pick your format)
    #df.to_parquet("features.parquet", index=False)
    df.to_csv("all_matches.tsv", index=False, sep='\t')

    sys.exit(0)


import re
import numpy as np


def peptide_to_mod_array(peptide_str, mod_dict):
    """
    Convert peptide string with modifications to a float array.

    Parameters:
        peptide_str: str, e.g. "K(PSMtag_9plex-6)(PSMtag_9plex-6)VPQVSTPTLVEVSR"
        mod_dict: dict, e.g. {"PSMtag_9plex-6": 6.02}

    Returns:
        numpy array of length len(sequence)+2 with modification masses
    """

    # Extract all modifications with their positions
    # We'll parse N-term modifications first
    n_term_mod = 0.0
    c_term_mod = 0.0

    # Pattern for modifications: something like (MODNAME)
    # Also support modifications before first AA
    mod_pattern = re.compile(r'\(([^\)]+)\)')

    # Remove mods for now to get sequence
    clean_seq = mod_pattern.sub('', peptide_str)

    # Initialize array: length = sequence + 2
    mod_array = np.zeros(len(clean_seq) + 2, dtype=float)

    # Handle N-term modifications
    # If string starts with "X(MOD)" pattern, consider it N-term if first char is modified
    n_term_match = re.match(r'^([A-Z])?\(([^\)]+)\)', peptide_str)
    seq_index = 0  # index for clean_seq
    mod_index = 0  # index in peptide_str

    # Track positions as we parse
    i = 0
    while i < len(peptide_str):
        char = peptide_str[i]
        if char.isalpha():
            # Regular amino acid
            seq_index += 1
            i += 1
        elif char == '(':
            # Start of modification
            j = peptide_str.index(')', i)
            mod_name = peptide_str[i + 1:j]
            mod_mass = mod_dict.get(mod_name, 0.0) #TODO does this silently fail if mod is not in the list?
            # Decide if this is N-term, C-term, or AA mod
            if seq_index == 0:
                # N-term
                mod_array[0] += mod_mass
            elif seq_index == len(clean_seq):
                # C-term
                mod_array[-1] += mod_mass
            else:
                # Modification of previous amino acid
                mod_array[seq_index] += mod_mass
            i = j + 1
        else:
            i += 1

    return mod_array.tolist() # TODO can we keep this as numpy for speed?