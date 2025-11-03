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

    print(mod_dict)

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
        report_psms=1
    )

    # Convert spectra into a Rust-friendly format
    logger.info("Converting spectra")
    rust_specs = []
    # TODO this should probably be done in chunks
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

    # Flatten hits and convert to dict
    #rows = [feat.to_dict() for group in hits for feat in group]

    ppm_errors = []
    errors = []

    for group in hits:
        for feat in group:
            d = feat.to_dict()  # your existing dict

            # Compute theoretical m/z if peptide exists
            pep = ps.Peptide.from_rust(feat.peptide)
            theo_mz = ps.Peptide.calculate_theoretical_mz(feat.sequence, feat.modifications, feat.charge)
            d["theoretical_mz"] = theo_mz
            """
            print("*********")
            print(feat.sequence, feat.modifications, feat.charge)
            print(d["theoretical_mz"], d["calcmass"], d["charge"])
            print("***")
            """

            # Get nearest MS1 scan from cached mapping
            ms1_scan = dia_spectra.get_nearest_ms1_for_scan(feat.spec_id)

            # Find closest peak
            closest_idx, closest_mz, intensity = ms1_scan.closest_peak(theo_mz)

            print("********")
            print(ms1_scan.peak_list())
            print(theo_mz, closest_mz)



            """
            d["closest_peak_mz"] = closest_mz
            d["closest_peak_intensity"] = intensity
            ppm_error = (d["closest_peak_mz"] - d["theoretical_mz"]) / d["theoretical_mz"] * 1e6
            ppm_errors.append(ppm_error)
            print("ppm_error", ppm_error)
            error = (d["closest_peak_mz"] - d["theoretical_mz"])
            errors.append(error)
            print("error", (d["closest_peak_mz"] - d["theoretical_mz"]))

            # Debug print — this is the key part
            print("------------------------------------------------")
            print(f"spec_id: {feat.spec_id}")
            print(f"sequence: {feat.sequence}")
            print(f"modifications: {feat.modifications}")
            print(f"charge: {feat.charge}")
            print(f"calcmass (from Rust feat): {d.get('calcmass')}")
            print(f"theoretical_mz (Python calc): {theo_mz}")
            print(f"closest_peak_mz (from MS1): {closest_mz}")
            print(f"intensity: {intensity}")
            print(f"mass_diff (Da): {closest_mz - theo_mz}")
            print(f"ppm_error (calc): {(closest_mz - theo_mz) / theo_mz * 1e6}")  # correct factor
            print("------------------------------------------------")

            #print(d["closest_peak_mz"], d["theoretical_mz"], ((d["closest_peak_mz"] - d["theoretical_mz"]) / d["theoretical_mz"]) * 10e6)
            """

    # turn into a DataFrame (fragments stays as a nested dict column)
    with open ("ppm_errors.tsv", "w") as fout:
        fout.write('ppm_errors\n')
        fout.write('\n'.join(map(str, ppm_errors)))
    with open ("errors.tsv", "w") as fout:
        fout.write('errors\n')
        fout.write('\n'.join(map(str, errors)))

    df = pd.DataFrame(rows)

    # TODO test this function

    # save (pick your format)
    #df.to_parquet("features.parquet", index=False)
    df.to_csv("all_matches.tsv", index=False, sep='\t')

    sys.exit(0)


def compare_ms1_mappings(spectrum_file):
    """
    Compare the original closest MS1 scan method with the new build_ms2_to_ms1_map method.
    Prints out discrepancies.
    """
    print(f"Comparing MS1 mappings for {len(spectrum_file.ms2scans)} MS2 spectra...")

    mismatches = 0

    for ms2_idx, ms2_scan in enumerate(spectrum_file.ms2scans):
        # Old method: brute-force search (closest_ms1spec style)
        ms1_rts = np.array([s.RT for s in spectrum_file.ms1scans])
        old_idx = np.argmin(np.abs(ms1_rts - ms2_scan.RT))

        # New method: cached map
        new_idx = spectrum_file.ms2_to_ms1_map[ms2_idx]

        if old_idx != new_idx:
            mismatches += 1
            print(f"MS2 scan {ms2_scan.scan_num}: old_idx={old_idx}, new_idx={new_idx}, ms2_rt={ms2_scan.RT:.4f}, old_ms1_rt={spectrum_file.ms1scans[old_idx].RT:.4f}, new_ms1_rt={spectrum_file.ms1scans[new_idx].RT:.4f}")

    print(f"Total mismatches: {mismatches} / {len(spectrum_file.ms2scans)}")



def plot_error_histograms(ppm_file="ppm_errors.tsv", error_file="errors.tsv", ppm_clip=500, err_clip=5):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    ppm_vals = pd.read_csv(ppm_file, sep="\t")["ppm_errors"].to_numpy(dtype=float)
    err_vals = pd.read_csv(error_file, sep="\t")["ppm_errors"].to_numpy(dtype=float)

    print(f"Loaded {len(ppm_vals)} PPM errors, {len(err_vals)} mass errors.")

    # Clip extreme outliers
    ppm_vals_clip = ppm_vals[(ppm_vals > -ppm_clip) & (ppm_vals < ppm_clip)]
    err_vals_clip = err_vals[(err_vals > -err_clip) & (err_vals < err_clip)]

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.hist(ppm_vals_clip, bins=100, edgecolor='k', alpha=0.7)
    plt.title(f"PPM Error Distribution (clipped ±{ppm_clip})")
    plt.xlabel("PPM Error")
    plt.ylabel("Count")

    plt.subplot(1, 2, 2)
    plt.hist(err_vals_clip, bins=100, edgecolor='k', alpha=0.7)
    plt.title(f"Mass Error Distribution (clipped ±{err_clip} Da)")
    plt.xlabel("Mass Error (Da)")
    plt.ylabel("Count")

    plt.tight_layout()
    plt.show()


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
    plot_error_histograms(ppm_file="/Users/danielgeiszler/Documents/jmod_tests/fragment_ion_indexing/ppm_errors.tsv",
                          error_file="/Users/danielgeiszler/Documents/jmod_tests/fragment_ion_indexing/errors.tsv")

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