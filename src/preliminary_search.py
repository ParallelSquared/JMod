import sys
import numpy as np
import peppy_sage as ps
import pandas as pd
import re

from config import diann_mods
from tqdm.auto import tqdm
from src.logger import logger

"""
Output we need:
    Retention time - incorrect because not standardized?
    Retention time error - to calculate
    mz observed
    mz library? should be calculated
    mz error

    e value? poisson okay?

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
    mod_dict.update(diann_mods)

    # Convert to rust-compatible peptide objects
    pep_seqs = [(v['seq'], v['mod_seq']) for v in library_spectra.values()]
    rust_peps = [ps.Peptide(seq, peptide_to_mod_array(mod_seq, mod_dict)) for seq, mod_seq in pep_seqs]

    # Create indexed database
    db = ps.IndexedDatabase.from_peptides( # TODO pass parameters as parameters
        peptides=rust_peps,
        bucket_size=128,
        ion_kinds=["b", "y"],
        min_ion_index=0,
        generate_decoys=True,
        decoy_tag="rev_",
        peptide_min_mass=0.0,
        peptide_max_mass=5000.0,
    )

    # Create scorer
    # I don't think min_isotope_error needs to be touched for DIA data since we don't care what peaks are
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

    #rust_specs = rust_specs[15000:16000]

    # Process spectra in chunks of 1000
    # Smaller chunks increases the amount of time spent passing things back and forth between Python and Rust
    # Multi-threading happens spectrum-by-spectrum at the Rust level
    #   (not sure how Rust does concurrency, processing in groups of spectra should reduce overhead incurred in spinning
    #   up new threads)
    logger.info("Searching spectra in chunks")
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

    rows = [feat.to_dict() for group in hits for feat in group]

    # turn into a DataFrame (fragments stays as a nested dict column)
    df = pd.DataFrame(rows)


    for spec in dia_spectra.ms2scans:
        ms1spec = dia_spectra.get_nearest_ms1_for_scan(spec.id)
        print(spec.RT, ms1spec.RT)

    # save (pick your format)
    #df.to_parquet("features.parquet", index=False)
    df.to_csv("all_matches.tsv", index=False, sep='\t')

    sys.exit(0)


def peptide_to_mod_array(peptide_str, mod_dict):
    """
    Convert peptide string with modifications to a float array.
    """
    mod_pattern = re.compile(r'\(([^\)]+)\)')
    clean_seq = mod_pattern.sub('', peptide_str)
    mod_array = np.zeros(len(clean_seq) + 2, dtype=float)

    seq_index = 0
    i = 0
    mods_after_first_aa = 0
    mods_after_last_aa = 0

    seq_len = len(clean_seq)

    while i < len(peptide_str):
        char = peptide_str[i]
        if char.isalpha():
            seq_index += 1
            i += 1
        elif char == '(':
            j = peptide_str.index(')', i)
            mod_name = peptide_str[i + 1:j]
            if mod_name not in mod_dict:
                raise ValueError(f"Unknown modification: {mod_name}")
            mod_mass = mod_dict[mod_name]

            # --- N-terminal logic ---
            if seq_index == 1 and mods_after_first_aa == 0:
                # First mod after the first residue → N-term
                mod_array[0] += mod_mass
                mods_after_first_aa += 1

            # --- C-terminal logic ---
            elif seq_index == seq_len:
                # Check if next char(s) are also '(' (another mod at the C-term)
                next_is_mod = j + 1 < len(peptide_str) and peptide_str[j + 1] == '('
                if next_is_mod and mods_after_last_aa == 0:
                    # first of the final two mods → last residue
                    mod_array[seq_index] += mod_mass
                    mods_after_last_aa += 1
                elif mods_after_last_aa >= 1:
                    # second (or more) final mod → C-term
                    mod_array[-1] += mod_mass
                else:
                    # single mod at end → last residue
                    mod_array[seq_index] += mod_mass

            # --- Internal residue mods ---
            else:
                mod_array[seq_index] += mod_mass

            i = j + 1
        else:
            i += 1

    return mod_array.tolist() #TODO will a numpy array work for performance reasons?


if __name__ == '__main__':
    mod_dict = {"PSMtag_9plex-6": 6.02}
    peptide = "K(PSMtag_9plex-6)(PSMtag_9plex-6)VPQVSTPTLVEVSR"
    print(peptide_to_mod_array(peptide, mod_dict))

    peptide = "K(PSMtag_9plex-6)VPQVSTPTLVEVSR"
    print(peptide_to_mod_array(peptide, mod_dict))

    peptide = "K(PSMtag_9plex-6)(PSMtag_9plex-6)VPQVSTPTLVEVSR(PSMtag_9plex-6)"
    print(peptide_to_mod_array(peptide, mod_dict))

    peptide = "K(PSMtag_9plex-6)(PSMtag_9plex-6)VPQVSTPTLVEVSR(PSMtag_9plex-6)(PSMtag_9plex-6)"
    print(peptide_to_mod_array(peptide, mod_dict))

    peptide = "K(PSMtag_9plex-6)VPQVSTPTLVEVS(PSMtag_9plex-6)R"
    print(peptide_to_mod_array(peptide, mod_dict))

    mod_dict = diann_mods
    peptide = "K(UniMod:4)VPQVSTPTLVEVSR"
    print(peptide_to_mod_array(peptide, mod_dict))