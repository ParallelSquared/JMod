import numpy as np
import peppy_sage as ps
import pandas as pd

import sys
from tqdm.auto import tqdm

from src.logger import logger

"""
Retneiton time
Retention time error

mz observed/library/error

e value

ignroe fit to lib

"""


def fit_with_features(dia_spectra, library_spectra):
    pep_seqs = [v['seq'] for v in library_spectra.values()] #TODO implement modifications

    peps = [ps.Peptide(seq) for seq in pep_seqs] #TODO implement modifications, proteins?

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


    for prelim_hits in hits:
        for rank_k in prelim_hits:
            print(rank_k)
            print(rank_k.fragments.to_dict())
            print()
            #print(rank_k.to_dict())
        print("****")

    #rows = [feat.to_dict() for group in hits[16000:17000] for feat in group]
    rows = [feat.to_dict() for group in hits for feat in group]

    # turn into a DataFrame (fragments stays as a nested dict column)
    df = pd.DataFrame(rows)

    # save (pick your format)
    #df.to_parquet("features.parquet", index=False)
    df.to_csv("all_matches.tsv", index=False, sep='\t')

    sys.exit(0)
