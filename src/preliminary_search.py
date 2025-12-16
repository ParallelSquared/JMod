import sys
import numpy as np
import peppy_sage as ps
import pandas as pd
import polars as pl
import re
from tqdm.auto import tqdm
import ast
from numpy import linalg as la

from src.config import diann_mods
from src.logger import logger
from src.models.spec_lib.spec_lib import python_lib_to_diann_df
from src.utils.io.load_files import Spectrum


def fit_with_features(dia_spectra, library_spectra, mass_tag, ms1_ppm_error=20, ms2_ppm_error=10):
    # Get tag plex
    if mass_tag is not None:
        if mass_tag.channel_names is not None:
            plex = len(mass_tag.channel_names)
    else:
        plex = 1

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

    # Construct polars df from python lib
    pl_lib = python_lib_to_diann_df(library_spectra)
    pl_lib.write_csv('temp.tsv', separator='\t')

    # Add modification array to the polars dataframe
    pl_lib = pl_lib.with_columns(
        pl.struct(["ModifiedPeptide"])
        .map_elements(lambda r: peptide_to_mod_array(r["ModifiedPeptide"], mod_dict),
                      return_dtype=pl.List(pl.Float32))
        .alias("Modifications")
    )

    print(pl_lib)

    # Convert to rust-compatible peptide objects
    pep_seqs = [(v['seq'], v['mod_seq']) for v in library_spectra.values()]

    rust_peps = []
    observed_mods: set[str] = set() # Allows us to backtrack original mod names from Sage results

    for seq, mod_seq in pep_seqs:
        rust_peps.append(ps.Peptide(seq, peptide_to_mod_array(mod_seq, mod_dict)))
        observed_mods.update(extract_mod_names(mod_seq))

    rev_map = {round(mod_dict[m], 4): m for m in observed_mods}  # Allows us to lookup mods by mass despite float error

    # Create indexed database
    db = ps.IndexedDatabase.from_peptides(  # TODO pass parameters as parameters
        peptides=rust_peps,
        bucket_size=4096,
        ion_kinds=["b", "y"],
        min_ion_index=2,
        generate_decoys=False,
        decoy_tag="rev_",
        peptide_min_mass=0.0,
        peptide_max_mass=5000.0,
    )

    print(db.debug_fragment_summary())

    db = ps.IndexedDatabase.from_library(  # TODO pass parameters as parameters
        library=pl_lib,
        bucket_size=4096,
        ion_kinds=["b", "y"],
        min_ion_index=2,
        generate_decoys=False,
        decoy_tag="rev_",
        peptide_min_mass=0.0,
        peptide_max_mass=5000.0,
    )

    print(db.debug_fragment_summary())

    # Create scorer
    # I don't think min_isotope_error needs to be touched for DIA data since we don't care what peaks are annotated as
    scorer = ps.Scorer(
        precursor_tol_da=(-1,1), # Unused in WWA -> defaults to window tol
        fragment_tol_ppm=(-1*ms2_ppm_error,ms2_ppm_error),
        min_isotope_err=0, # Changing these to look at other isotopes will require increasing report_PSMs due to
        max_isotope_err=0, # degenerate matches
        wide_window=True, # Uses window values instead of precursor masses
        chimera=False, # False, do not iteratively remove peaks
        annotate_matches=True, # Add fragment annotation
        min_matched_peaks=3,
        max_fragment_charge=2,
        report_psms=5*plex
    )

    # Convert spectra into a Rust-friendly format
    logger.info("Converting spectra")
    rust_specs = []
    # TODO this should probably be done in chunks if it becomes a bottleneck
    for spec in tqdm(dia_spectra.ms2scans):
        print(spec.id, spec.prec_mz, spec.ms1window)
        rust_specs += [spec.to_rust_spectrum()]

    #rust_specs = rust_specs[15000:16000]

    # Process spectra in chunks of 1000
    # Smaller chunks increases the amount of time spent passing things back and forth between Python and Rust
    # Multi-threading happens spectrum-by-spectrum at the Rust level
    #   (not sure how Rust does concurrency, processing in groups of spectra should reduce overhead incurred in spinning
    #   up new threads)
    logger.info("Searching spectra in chunks")
    chunk_size = 1000
    hits = None # Rust object holding arrays of hits
    for i in tqdm(range(0, len(rust_specs), chunk_size)):
        chunk = rust_specs[i:i + chunk_size]
        batch_hits = scorer.score_many(db, chunk)
        if hits is None:
            hits = batch_hits
        else:
            hits.extend(batch_hits)
            print(batch_hits)

    # Make Polars dataframe
    logger.info("Building results dataframe")
    col_names = hits.get_column_names()
    df = pl.DataFrame({name: getattr(hits, name) for name in col_names})
    del hits # Free the memory of the Rust result container

    # 2. Vectorized Calculation of Theoretical m/z (using Polars UDF)
    # The UDF will run the Python function but is integrated into Polars' execution engine.
    df = df.with_columns(
        pl.struct(['sequence', 'modifications', 'charge'])
        .map_elements(calculate_theo_mz_udf, return_dtype=pl.Float64)
        .alias("theoretical_mz")
    )

    # 3. MS1 Lookup (The non-vectorized bottleneck, handled in a dedicated function)
    # Convert Polars DF to Python lists, run the loop, and convert result back to Polars.
    ms1_data_list = lookup_ms1_data_list(df, dia_spectra)
    ms1_df = pl.DataFrame(ms1_data_list)

    # Add the MS1 data columns back to the main DataFrame
    # Since the data was iterated in order, we can just horizontally stack (hstack)
    df = df.hstack(ms1_df)

    # 4. Vectorized Error Calculations (Polars Expressions)
    df = df.with_columns([
        # Calculate difference
        (pl.col("closest_peak_mz_ms1") - pl.col("theoretical_mz")).alias("mz_diff")
    ]).with_columns([
        # Calculate relative error
        (pl.col("mz_diff") / pl.col("theoretical_mz")).alias("relative_error_ms1")
    ]).with_columns([
        # Calculate PPM error
        (pl.col("relative_error_ms1") * 1_000_000).alias("ppm_error_ms1")
    ])
    # Clean up the intermediate column
    df = df.drop("mz_diff")

    # 5. Filter out large MS1 errors (Polars syntax)
    df = df.filter(pl.col('ppm_error_ms1').abs() < ms1_ppm_error)

    # Get RTs for alignment
    lib_rts = {v['mod_seq'] : v['iRT'] for v in library_spectra.values()}

    # Adapt the dataframe to the format expected downstream
    df = adapt_output_df(df, lib_rts, rev_map)

    # Calculate spectral angle
    fragment_library_map = create_fragment_library_map(library_spectra)

    logger.info("Computing spectral angle")
    # Define the Polars UDF closure to capture the fragment_library_map
    def spectral_angle_polars_udf(r: dict) -> float:
        """
        Polars UDF to calculate the Normalized Spectral Contrast Angle (P=0.5).
        r is a dictionary-like view of the row data (struct of columns).

        OPTIMIZED: Receives native Polars List types (which are Python lists)
        """
        # The lookup key must be the tuple (modified_sequence, charge)
        seq = r['seq']  # Modified sequence string
        z = r['z']  # Charge (renamed from 'charge')
        ms2_intensity = r['ms2_intensity']

        if seq is None or ms2_intensity is None or ms2_intensity == 0.0:
            return 0.0

        # 1. Get the library fragment vector for the peptide (using TUPLE KEY)
        library_key = (seq, z)
        library_vec = fragment_library_map.get(library_key)
        if not library_vec:
            return 0.0

        # 2. Access observed fragment data directly (no string parsing needed!)
        charges = r['frag_charges']  # List(Int64) -> list[int]
        kinds_raw = r['frag_kinds']  # List(String) -> list[str]
        ordinals = r['frag_fragment_ordinals']  # List(Int64) -> list[int]
        intensities = r['frag_intensities']  # List(Float64) -> list[float]

        # 3. Check for consistent array lengths
        if not (len(charges) == len(kinds_raw) == len(ordinals) == len(intensities)):
            logger.error(
                f"Fragment array length mismatch for seq {seq}. Lengths: {len(charges)}, {len(kinds_raw)}, {len(ordinals)}, {len(intensities)}")
            return 0.0

        # 4. Create raw observed vector (A_raw)
        observed_vec = {}

        # Iterate over lists to build observed_vec dictionary
        for c, k, o, i in zip(charges, kinds_raw, ordinals, intensities):
            # Alignment key: (ion_kind, ordinal, charge)
            key = (k.upper(), o, c)
            # Normalize intensity by total MS2 intensity
            observed_vec[key] = i / ms2_intensity

        # 5. Align vectors and create raw intensity vectors (NumPy conversion happens here)
        all_keys = set(observed_vec.keys()) | set(library_vec.keys())

        # Cast to NumPy arrays for fast dot product calculation
        A_raw = np.array([observed_vec.get(k, 0.0) for k in all_keys], dtype=np.float32)
        B_raw = np.array([library_vec.get(k, 0.0) for k in all_keys], dtype=np.float32)

        # 6. Apply Power Transformation (P=0.5)
        A_pow = np.sqrt(A_raw)
        B_pow = np.sqrt(B_raw)

        # 7. Spectral Contrast Angle Calculation
        dot_product = np.dot(A_pow, B_pow)
        norm_A = la.norm(A_pow)
        norm_B = la.norm(B_pow)

        if norm_A == 0.0 or norm_B == 0.0:
            return 0.0

        return dot_product / (norm_A * norm_B)

    # Apply the Polars UDF
    # Note: We must include the fragment columns by their new names
    df = df.with_columns(
        pl.struct([
            'frag_charges',
            'frag_kinds',
            'frag_fragment_ordinals',
            'frag_intensities',
            'ms2_intensity',
            'seq',
            'z'
        ])
        .map_elements(spectral_angle_polars_udf, return_dtype=pl.Float64)
        .alias('spectral_contrast_angle')
    )

    def scribe_score_polars_udf(r: dict) -> float:
        """
        Polars UDF to calculate the primary Scribe Score:
        Score = -ln(sum((A_sqrt - L_sqrt)^2)), where A and L are normalized to sum to 1.
        """
        seq = r['seq']  # Modified sequence string
        z = r['z']  # Charge
        ms2_intensity = r['ms2_intensity']

        if seq is None or z is None or ms2_intensity is None or ms2_intensity == 0.0:
            # Return a definite low score if key data is missing or spectrum is empty
            return -999.0

            # 1. Get the library fragment vector for the peptide (using TUPLE KEY)
        library_key = (seq, z)
        library_vec = fragment_library_map.get(library_key)
        if not library_vec:
            return -999.0

        # 2. Access observed fragment data directly (no string parsing needed!)
        charges = r['frag_charges']  # List(Int64) -> list[int]
        kinds_raw = r['frag_kinds']  # List(String) -> list[str]
        ordinals = r['frag_fragment_ordinals']  # List(Int64) -> list[int]
        intensities = r['frag_intensities']  # List(Float64) -> list[float]

        # 3. Check for consistent array lengths
        if not (len(charges) == len(kinds_raw) == len(ordinals) == len(intensities)):
            logger.error(
                f"Fragment array length mismatch for seq {seq}. Lengths: {len(charges)}, {len(kinds_raw)}, {len(ordinals)}, {len(intensities)}")
            return -999.0

        # 4. Create raw observed vector (A_raw)
        observed_vec = {}

        # Iterate over lists to build observed_vec dictionary
        for c, k, o, i in zip(charges, kinds_raw, ordinals, intensities):
            # Alignment key: (ion_kind, ordinal, charge)
            key = (k.upper(), o, c)
            # Note: The initial intensity normalization by ms2_intensity is NOT used here,
            # as the Scribe formula requires normalization AFTER alignment.
            observed_vec[key] = i

            # 5. Align vectors and create raw intensity vectors (NumPy conversion happens here)

        # KEY CHANGE: Only include ions that are present in the library.
        all_keys = set(library_vec.keys())

        # Cast to NumPy arrays for fast calculation
        # A_raw will be 0.0 if the ion was observed but is NOT in the library keys.
        A_raw = np.array([observed_vec.get(k, 0.0) for k in all_keys], dtype=np.float64)
        B_raw = np.array([library_vec.get(k, 0.0) for k in all_keys], dtype=np.float64)

        # Check for zero vectors AFTER alignment
        sum_A_raw = np.sum(A_raw)
        sum_B_raw = np.sum(B_raw)

        if sum_A_raw == 0.0 or sum_B_raw == 0.0:
            return -999.0

        # 6. Normalization (to sum to 1) and Square Root Transformation
        A_norm = A_raw / sum_A_raw
        B_norm = B_raw / sum_B_raw

        A_sqrt = np.sqrt(A_norm)
        B_sqrt = np.sqrt(B_norm)

        # 7. Sum of Squared Errors
        sq_error_sum = np.sum((A_sqrt - B_sqrt) ** 2)

        # Guard against zero error (perfect match) to avoid log(0)
        if sq_error_sum <= 1e-12:
            return 25.0  # Return a large, stable score

        # 8. Final Negative Log Transformation
        return -np.log(sq_error_sum)

    # Apply the Polars UDF
    df = df.with_columns(
        pl.struct([
            'frag_charges',
            'frag_kinds',
            'frag_fragment_ordinals',
            'frag_intensities',
            'ms2_intensity',
            'seq',  # Modified sequence column
            'z',  # Charge column (renamed from 'charge')
        ])
        .map_elements(scribe_score_polars_udf, return_dtype=pl.Float64)
        .alias('scribe_score')  # Renamed column
    )

    # Back to pandas
    df = df.to_pandas()

    has_rt_df = df[~df['lib_rt'].isna()]
    print("has_rt " + str(len(has_rt_df)))

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


def create_fragment_library_map(library_spectra):
    """
    Converts the library spectra dictionary into a nested, fast-lookup map
    for spectral comparison.

    Keys: (Modified Sequence, Precursor Charge) [TUPLE KEY REQUIRED FOR ACCURACY]
    Values: Dict mapping (ion_kind, ordinal, charge) -> RelativeIntensity

    This function is updated to correctly handle the tuple key.
    """
    frag_map = {}
    for data in library_spectra.values():
        mod_seq = data.get('mod_seq')
        prec_z = data.get('prec_z')  # New: Get precursor charge
        library_frags = data.get('frags', {})

        if not mod_seq or prec_z is None:
            continue

        peptide_map = {}
        # The key for the map is now (modified sequence, charge)
        map_key = (mod_seq, prec_z)

        for frag_str, mz_int_list in library_frags.items():
            # Example frag_str: "b3_1" (type + nr + optional_loss + "_" + z)

            parts = frag_str.split('_')
            if len(parts) != 2:
                # Skip if format is unexpected
                continue

            frag_name = parts[0].upper()  # e.g., "B3" or "Y7-H2O"
            frag_z = int(parts[1])  # e.g., 1

            # Extract type and ordinal, ignoring loss notation for this key
            match = re.match(r'([BY])(\d+)', frag_name)
            if not match:
                continue

            frag_type = match.group(1)  # B or Y
            frag_ordinal = int(match.group(2))  # 3, 7, etc.

            intensity = mz_int_list[1]  # Relative Intensity is the second element

            # Key for alignment: (IonType, Ordinal, Charge)
            key = (frag_type, frag_ordinal, frag_z)
            peptide_map[key] = intensity

        if peptide_map:
            frag_map[map_key] = peptide_map  # Store map with (mod_seq, prec_z) key

    return frag_map


def calculate_spectral_contrast_angle(row, fragment_library_map):
    """
    Pandas UDF to calculate the Normalized Spectral Contrast Angle (P=0.5)
    between observed and library fragments, using (Type, Ordinal, Charge) as key.

    This is mathematically equivalent to Cosine Similarity on square-rooted intensities.
    """

    if pd.isna(row['seq']) or pd.isna(row['fragments']):
        return 0.0

    # 1. Parse observed fragment data
    try:
        observed_data = ast.literal_eval(row['fragments'])
    except:
        return 0.0

    # 2. Get the library fragment vector for the peptide
    library_vec = fragment_library_map.get(row['seq'])
    if not library_vec:
        return 0.0

    # 3. Create raw observed vector (A_raw)
    observed_vec = {}
    ms2_intensity = row['ms2_intensity']

    # Iterate over charges, kinds, ordinals, and intensities simultaneously
    for c, k, o, i in zip(
            observed_data['charges'],
            observed_data['kinds'],
            observed_data['fragment_ordinals'],
            observed_data['intensities']
    ):
        # Alignment key: (ion_kind, ordinal, charge)
        key = (k.upper(), o, c)
        # Normalize intensity by total MS2 intensity
        observed_vec[key] = i / ms2_intensity if ms2_intensity > 0 else 0.0

    # 4. Limit ions to just those in the library
    all_keys = set(library_vec.keys())

    A_raw = np.array([observed_vec.get(k, 0.0) for k in all_keys], dtype=np.float64)
    B_raw = np.array([library_vec.get(k, 0.0) for k in all_keys], dtype=np.float64)

    # 5. Apply Power Transformation (P=0.5)
    A_pow = np.sqrt(A_raw)
    B_pow = np.sqrt(B_raw)

    # 6. Spectral Contrast Angle Calculation: (A_pow . B_pow) / (||A_pow|| * ||B_pow||)
    dot_product = np.dot(A_pow, B_pow)
    norm_A = la.norm(A_pow)
    norm_B = la.norm(B_pow)

    # Return 0.0 if either vector has zero magnitude
    if norm_A == 0.0 or norm_B == 0.0:
        return 0.0

    return dot_product / (norm_A * norm_B)


def calculate_theo_mz_udf(row):
    """Polars UDF to calculate theoretical m/z."""
    # The Polars struct will yield a dictionary-like object per row
    return ps.Peptide.calculate_theoretical_mz(
        row['sequence'], row['modifications'], row['charge']
    )


def lookup_ms1_data_list(df: pl.DataFrame, dia_spectra):
    """
    Performs the non-vectorized MS1 scan lookup and closest peak finding
    in a dedicated Python loop.

    Returns a list of dictionaries with the MS1 data to be converted back
    to a Polars DataFrame.
    """
    ms1_data = []

    # We must extract the columns and iterate in Python because the underlying
    # dia_spectra methods are not vectorized.
    spec_ids = df['spec_id'].to_list()
    theo_mzs = df['theoretical_mz'].to_list()

    for spec_id, theo_mz in tqdm(
            zip(spec_ids, theo_mzs),
            desc="MS1 Peak Finding",
            total=len(spec_ids),
            leave=False
    ):
        ms1_scan = dia_spectra.get_nearest_ms1_for_scan(spec_id)

        if ms1_scan:
            # Assumes ms1_scan.closest_peak(theo_mz) returns (idx, mz, intensity)
            closest_idx, closest_mz, intensity = ms1_scan.closest_peak(theo_mz)
            ms1_data.append({
                'Ms1_spec_id': ms1_scan.scan_num,
                "closest_peak_mz_ms1": closest_mz,
                "closest_peak_intensity_ms1": intensity,
            })
        else:
            # Use 0.0 or None/NaN placeholders for missing data
            ms1_data.append({
                'Ms1_spec_id': None,
                "closest_peak_mz_ms1": 0.0,
                "closest_peak_intensity_ms1": 0.0,
            })

    return ms1_data


def adapt_output_df(df: pl.DataFrame, lib_rts: dict, rev_map: dict) -> pl.DataFrame:
    """
    Performs column normalization and peptide sequence reconstruction using Polars.
    """

    # --- UDF Wrappers to handle non-Polars logic ---

    # 1. Wrapper for map_mod_list (requires rev_map lookup)
    def map_mod_list_wrapper(mod_list: list[float]) -> list[str]:
        out = []
        for m in mod_list:
            if m != 0.0:
                name = rev_map.get(round(m, 4))
                out.append(name if name is not None else "")
            else:
                out.append("")
        return out

    # 2. Wrapper for mod_array_to_peptide (requires data from two columns)
    def mod_array_to_peptide_wrapper(r: dict) -> str:
        # r is a dictionary containing the fields in the struct
        return mod_array_to_peptide(r["stripped_seq"], r["modification_names"])

    # 3. Wrapper for library RT lookup (Polars < 0.19.0 compatibility)
    def map_rt_wrapper(seq: str) -> float:
        # Returns the iRT value, or np.nan if the sequence is not found
        return lib_rts.get(seq, np.nan)

    # 1. Rename columns (Polars version)
    df = df.rename({
        'spec_id': 'spec_name',
        'charge': 'z',
        'sequence': 'stripped_seq',
        'theoretical_mz': 'mz'
    })

    # 2. Make matching spec_id column
    df = df.with_columns(
        pl.col('spec_name')
        .map_elements(lambda s: Spectrum.extract_scannum(s), return_dtype=pl.UInt32)
        .alias('spec_id')
    )

    # 3. Reconstruct modification names (Polars UDF)
    df = df.with_columns(
        pl.col("modifications")
        .map_elements(map_mod_list_wrapper, return_dtype=pl.List(pl.Utf8))
        .alias("modification_names")
    )

    # 4. Reconstruct modified peptide sequence string (Polars struct UDF)
    df = df.with_columns(
        pl.struct(["stripped_seq", "modification_names"])
        .map_elements(mod_array_to_peptide_wrapper, return_dtype=pl.Utf8)
        .alias("seq")
    )

    # 5. Grab library rts based on reconstructed seq (Polars map_dict)
    df = df.with_columns(
        pl.col("seq")
          .map_elements(map_rt_wrapper, return_dtype=pl.Float64) # Use Float32 for RTs
          .alias("lib_rt")
    )

    return df


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

