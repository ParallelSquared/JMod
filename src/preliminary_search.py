import sys
import numpy as np
import peppy_sage as ps
import pandas as pd
import re
from tqdm.auto import tqdm

from .config import diann_mods
from .logger import logger
from .utils.io.load_files import Spectrum

"""
Output we need:
    Retention time - incorrect because not standardized?
    Retention time error - to calculate
    mz observed
    mz library? should be calculated
    mz error

    e value? poisson okay?


    column normalization
        spec_id is a number
        Ms1_spec_id is a number
        seq has mods in it, renormalize
        stripped_sg
        z is charge
        window_mz is window center?
        rt is normal
        lib_rt
"""


def fit_with_features(dia_spectra, library_spectra, mass_tag, ms1_ppm_error=20, ms2_ppm_error=10):

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

    rust_peps = []
    observed_mods: set[str] = set() # Allows us to backtrack original mod names from Sage results

    for seq, mod_seq in pep_seqs:
        rust_peps.append(ps.Peptide(seq, peptide_to_mod_array(mod_seq, mod_dict)))
        observed_mods.update(extract_mod_names(mod_seq))

    # Create indexed database
    db = ps.IndexedDatabase.from_peptides( # TODO pass parameters as parameters
        peptides=rust_peps,
        bucket_size=128,
        ion_kinds=["b", "y"],
        min_ion_index=1,
        generate_decoys=False,
        decoy_tag="rev_",
        peptide_min_mass=0.0,
        peptide_max_mass=5000.0,
    )

    # Create scorer
    # I don't think min_isotope_error needs to be touched for DIA data since we don't care what peaks are annotated as
    scorer = ps.Scorer(
        precursor_tol_da=(-1,1), # Unused in WWA -> defaults to window tol
        fragment_tol_ppm=(-1*ms2_ppm_error,ms2_ppm_error), # TODO pass as parameter
        min_isotope_err=0, # Changing these to look at other isotopes will require increasing report_PSMs due to
        max_isotope_err=0, # degenerate matches
        wide_window=True, # Uses window values instead of precursor masses
        chimera=False, # False, do not iteratively remove peaks
        annotate_matches=True, # Add fragment annotation
        report_psms=5
    )

    # Convert spectra into a Rust-friendly format
    logger.info("Converting spectra")
    rust_specs = []
    # TODO this should probably be done in chunks if it becomes a bottleneck
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
    # rows = [feat.to_dict() for group in hits for feat in group]
    rows = []

    # TODO clean up results
    for group in hits:
        for feat in group:
            d = feat.to_dict()

            # Compute theoretical m/z
            theo_mz = ps.Peptide.calculate_theoretical_mz(feat.sequence, feat.modifications, feat.charge)
            d["theoretical_mz"] = theo_mz

            # Get nearest MS1 scan from cached mapping
            ms1_scan = dia_spectra.get_nearest_ms1_for_scan(feat.spec_id)
            d['Ms1_spec_id'] = ms1_scan.scan_num

            closest_idx, closest_mz, intensity = ms1_scan.closest_peak(theo_mz)

            d["closest_peak_mz_ms1"] = closest_mz
            d["closest_peak_intensity_ms1"] = intensity
            relative_error = (d["closest_peak_mz_ms1"] - d["theoretical_mz"]) / d["theoretical_mz"]
            ppm_error = (d["closest_peak_mz_ms1"] - d["theoretical_mz"]) / d["theoretical_mz"] * 1000000
            d["ppm_error_ms1"] = ppm_error
            d["relative_error_ms1"] = relative_error

            rows.append(d)

    # Convert results to the downstream-compatible df
    df = pd.DataFrame(rows)
    rev_map = {round(mod_dict[m], 4): m for m in observed_mods} # Allows us to lookup mods by mass despite float error

    # Get RTs for alignment
    lib_rts = {v['mod_seq'] : v['iRT'] for v in library_spectra.values()}

    # Format for downstream
    df = adapt_output_df(df, lib_rts, rev_map)
    print(len(df))
    # Filter out large MS1 errors
    df = df[df['ppm_error_ms1'].abs() < ms1_ppm_error]
    print(len(df))

    ##########
    # This is a temporary solution to limit peptides to those inside the library
    # Build valid (seq, charge) keys from the library once
    valid_keys = {
        (v["mod_seq"], v["prec_z"])
        for v in library_spectra.values()
    }

    # Create tuple key per row and keep only those present in the library
    df["__key"] = list(zip(df["seq"], df["z"].astype(int)))
    mask = df["__key"].map(valid_keys.__contains__)

    df = df[mask].drop(columns="__key")
    ##########

    return df


def adapt_output_df(df, lib_rts, rev_map):
    """
    column normalization
        -spec_id is a number
        -Ms1_spec_id is a number
        -seq has mods in it, renormalize
        -stripped_seq
        -z is charge
        window_mz is window center?
        -rt is normal
        -lib_rt
        -mz is theoretical mz

        Index(['file_id', 'spec_id', 'psm_id', 'rank', 'sequence', 'modifications',
       'label', 'hyperscore', 'delta_mass', 'matched_peaks', 'peptide_len',
       'expmass', 'calcmass', 'charge', 'rt', 'aligned_rt', 'predicted_rt',
       'delta_rt_model', 'ims', 'predicted_ims', 'delta_ims_model',
       'isotope_error', 'average_ppm', 'delta_next', 'delta_best', 'longest_b',
       'longest_y', 'longest_y_pct', 'missed_cleavages',
       'matched_intensity_pct', 'scored_candidates', 'poisson',
       'discriminant_score', 'posterior_error', 'spectrum_q', 'peptide_q',
       'protein_q', 'ms2_intensity', 'fragments', 'theoretical_mz',
       'closest_peak_mz_ms1', 'closest_peak_intensity_ms1', 'ppm_error_ms1'],
      dtype='object')
    """

    # Rename columns
    change_columns_dict = {
        'spec_id': 'spec_name',
        'charge' : 'z',
        'sequence' : 'stripped_seq',
        'theoretical_mz' : 'mz'
    }
    df.rename(columns=change_columns_dict, inplace=True)

    # Make matching spec_id column
    df['spec_id'] = df['spec_name'].apply(lambda s: Spectrum.extract_scannum(s))

    # Reconstruct modified peptide sequence string
    def map_mod_list(masses, rev_map):
        out = []
        for m in masses:
            if m != 0.0:
                name = rev_map.get(round(m, 4))
                if name is not None:
                    out.append(name)
            else:
                out.append("")
        return out

    df["modification_names"] = df["modifications"].apply(lambda lst: map_mod_list(lst, rev_map))
    df["seq"] = df.apply(lambda r: mod_array_to_peptide(r["stripped_seq"], r["modification_names"]), axis=1)

    # Grab library rts based on reconstructed seq
    df["lib_rt"] = df["seq"].map(lib_rts)

    return df


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
    err_vals = pd.read_csv(error_file, sep="\t")["errors"].to_numpy(dtype=float)

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

def mod_array_to_peptide(peptide_str, mod_array):
    """
    Convert peptide and mod array to peptide string.
    """

    pep_len = len(peptide_str)
    mod_pep = peptide_str

    for i in range(len(mod_array) - 1, -1, -1):
        mod = mod_array[i]
        if mod != "":
            if i == pep_len + 1 or i == pep_len:
                mod_pep = mod_pep[:pep_len] + f"({mod})" + mod_pep[pep_len:]
            elif i < pep_len and i > 0:
                mod_pep = mod_pep[:i] + f"({mod})" + mod_pep[i:]
            elif i == 0:
                mod_pep = mod_pep[:1] + f"({mod})" + mod_pep[1:]

    return mod_pep


def extract_mod_names(mod_seq: str):
    """
    From a mod_seq like 'ACD(Phospho)EFG(ox)', return ['Phospho', 'ox'].
    Duplicates removed automatically.
    """
    mod_pattern = re.compile(r'\(([^\)]+)\)')
    return list(set(mod_pattern.findall(mod_seq)))


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

    mods = ["(UniMod:4)", "", "", "", "", "", "", "", "", ""]
    peptide = "APTLVVEK"
    print(mod_array_to_peptide(peptide, mods))

    mods = ["(UniMod:4)", "(UniMod:4)", "", "", "", "", "", "", "", ""]
    peptide = "APTLVVEK"
    print(mod_array_to_peptide(peptide, mods))

    mods = ["", "", "", "", "(UniMod:4)", "", "", "", "", ""]
    peptide = "APTLVVEK"
    print(mod_array_to_peptide(peptide, mods))

    mods = ["(UniMod:4)", "", "", "", "", "", "", "", "", "(UniMod:4)"]
    peptide = "APTLVVEK"
    print(mod_array_to_peptide(peptide, mods))

    mods = ["(UniMod:4)", "", "", "", "", "", "", "", "(UniMod:4)", "(UniMod:4)"]
    peptide = "APTLVVEK"
    print(mod_array_to_peptide(peptide, mods))

    mods = ["(UniMod:4)", "", "", "", "", "(UniMod:4)", "", "", "(UniMod:4)", "(UniMod:4)"]
    peptide = "APTLVVEK"
    print(mod_array_to_peptide(peptide, mods))

    mods = ["", "", "UniMod:4", "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""]
    peptide = "TCQLYPNAIASTLVHK"
    print(mod_array_to_peptide(peptide, mods))

