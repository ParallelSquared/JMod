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

import sys
import numpy as np
import peppy_sage as ps
import pandas as pd
import polars as pl
import re
from functools import partial
from tqdm.auto import tqdm
import ast
from numpy import linalg as la

from src.config import diann_mods
from src.logger import logger
from src.utils.io.load_files import Spectrum



def _dtype_is_null(dt) -> bool:
    """True for Null, and for List(...) nested down to a Null inner type."""
    if dt == pl.Null:
        return True
    inner = getattr(dt, "inner", None)
    return _dtype_is_null(inner) if inner is not None else False


def _reconcile_chunk_schemas(frames: list) -> list:
    """Cast Null-typed columns so a list of per-chunk frames can be concatenated.

    Polars infers a column's type per frame, so a chunk with no data for a column
    gets Null / List(Null) while another gets Int32 / List(Float32). Concat then
    refuses to vstack them. For each column, adopt the first non-Null dtype any
    chunk produced and cast the Null-typed chunks to it. Columns that are Null
    everywhere are left alone -- there is no evidence of what they should be, and
    concat handles the all-Null case fine.
    """
    if len(frames) < 2:
        return frames
    target = {}
    for f in frames:
        for name, dt in f.schema.items():
            if name not in target and not _dtype_is_null(dt):
                target[name] = dt
    out = []
    for f in frames:
        fixes = {n: target[n] for n, dt in f.schema.items()
                 if _dtype_is_null(dt) and n in target}
        out.append(f.cast(fixes) if fixes else f)
    return out


def fit_with_features(dia_spectra, library_spectra, mass_tag, SILAC, ms1_ppm_error=20, ms2_ppm_error=10):
    # Get tag plex
    if mass_tag is not None:
        if mass_tag.channel_names is not None:
            plex = len(mass_tag.channel_names)
    else:
        plex = 1

    # Construct modification dict to convert names to masses
    # Start with SILAC
    if mass_tag is not None:
        mod_dict = {'-'.join([mass_tag.name, mass_tag.channel_names[i]]): # construct channel names
                        mass_tag.mass + mass_tag.delta[i] # construct channel masses
                    for i in range(len(mass_tag.channel_names))}
    else:
        mod_dict = {}
    
    if SILAC is not None:
        mod_dict.update({'-'.join([SILAC.name, SILAC.channel_names[i]]): # construct channel names
                        SILAC.mass + SILAC.delta[i] # construct channel masses
                    for i in range(len(SILAC.channel_names))})

    # Add all of the other supported modifications
    mod_dict.update(diann_mods)

    # Construct polars df from python lib
    pl_lib = library_spectra.to_diann_df()

    # Add modification array to the polars dataframe
    # Deduplicate: compute on unique ModifiedPeptide values, then join back
    unique_peps = pl_lib.select("ModifiedPeptide").unique()
    unique_peps = unique_peps.with_columns(
        pl.col("ModifiedPeptide")
        .map_elements(lambda p: peptide_to_mod_array(p, mod_dict),
                      return_dtype=pl.List(pl.Float32))
        .alias("Modifications")
    )
    pl_lib = pl_lib.join(unique_peps, on="ModifiedPeptide", how="left")

    # Build mod_array lookup from the unique peptides we already computed
    _mod_array_map = dict(zip(
        unique_peps["ModifiedPeptide"].to_list(),
        unique_peps["Modifications"].to_list(),
    ))

    # observed_mods only needs unique mod_seqs
    observed_mods: set[str] = set()
    for ms in _mod_array_map:
        observed_mods.update(extract_mod_names(ms))
    rev_map = {}
    for m in observed_mods:
        if m.startswith('[') and m.endswith(']'):
            inner = m[1:-1]
            mass = mod_dict[inner] if inner in mod_dict else float(inner)
        else:
            mass = mod_dict[m]
        rev_map[round(mass, 4)] = m

    db = ps.IndexedDatabase.from_library(
        library=pl_lib,
        bucket_size=4096,
        ion_kinds=["b", "y"],
        min_ion_index=2,
        generate_decoys=False,
        decoy_tag="rev_",
        peptide_min_mass=0.0,
        peptide_max_mass=5000.0,
    )

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
        report_psms=int(5*plex)
    )


    # Convert and search spectra in chunks of 1000.
    # Smaller chunks increases the amount of time spent passing things back and forth between Python and Rust
    # Multi-threading happens spectrum-by-spectrum at the Rust level
    #   (not sure how Rust does concurrency, processing in groups of spectra should reduce overhead incurred in spinning
    #   up new threads)
    #
    # Conversion is fused into the search loop, and each chunk's hits are turned into
    # Polars immediately, so three whole-run copies never pile up at once. Converting
    # every spectrum up front held a second copy of every MS2 peak on the Rust side for
    # the whole search (the `del rust_specs` below the loop was commented out, so it was
    # still live at `to_polars`), and accumulating one Rust hits object meant that object
    # and the full DataFrame built from it coexisted at peak. On a large timsTOF .d
    # (~9e5 MS2 band spectra) either one is enough to get the process OOM-killed.
    logger.info("Converting and searching spectra in chunks")
    chunk_size = 1000
    ms2scans = dia_spectra.ms2scans
    hit_frames = []
    for i in tqdm(range(0, len(ms2scans), chunk_size)):
        chunk = [spec.to_rust_spectrum() for spec in ms2scans[i:i + chunk_size]]
        batch_hits = scorer.score_many(db, chunk)
        # to_polars() per chunk: the Rust hits for this chunk are released here rather
        # than at the end of the whole search.
        hit_frames.append(batch_hits.to_polars())
        del batch_hits, chunk

    # Reconcile per-chunk dtypes before concatenating
    hit_frames = _reconcile_chunk_schemas(hit_frames)

    # rechunk=False keeps this a cheap multi-chunk view over the per-chunk buffers
    # instead of allocating a second full copy of the concatenated result.
    df = pl.concat(hit_frames, rechunk=False)
    del hit_frames



    # 2. Vectorized Calculation of Theoretical m/z (using Polars UDF)
    # The UDF will run the Python function but is integrated into Polars' execution engine.
    logger.info("Calculating theoretical m/z")
    df = df.with_columns(
        pl.struct(['sequence', 'modifications', 'charge'])
        .map_elements(calculate_theo_mz_udf, return_dtype=pl.Float64)
        .alias("theoretical_mz")
    )

    # 3. MS1 Lookup (The non-vectorized bottleneck, handled in a dedicated function)
    # Convert Polars DF to Python lists, run the loop, and convert result back to Polars.
    logger.info("Looking up MS1 data")
    ms1_data_list = lookup_ms1_data_list(df, dia_spectra)
    ms1_df = pl.DataFrame(ms1_data_list)

    # Add the MS1 data columns back to the main DataFrame
    # Since the data was iterated in order, we can just horizontally stack (hstack)
    df = df.hstack(ms1_df)

    # Per-fragment ion mobility: join each matched fragment's experimental m/z
    # back to its MS2 band spectrum to read the observed peak's 1/K0.
    logger.info("Looking up per-fragment ion mobilities")
    df = df.with_columns(
        pl.Series("frag_ion_mobility", lookup_fragment_mobilities(df, dia_spectra))
    )

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
    # Library ion mobilities, for the IM alignment (NaN when the library has no IM column)
    lib_ims = {v['mod_seq'] : v['IonMob'] for v in library_spectra.values()}

    # Adapt the dataframe to the format expected downstream
    logger.info("Adapting output dataframe")
    df = adapt_output_df(df, lib_rts, rev_map, lib_ims=lib_ims)

    # Calculate spectral angle
    logger.info("Creating fragment library map")
    fragment_library_map = create_fragment_library_map(library_spectra)

    # Filter to only peptides present in the library (before computing expensive scores)
    valid_keys_df = pl.DataFrame({
        "seq": [v["mod_seq"] for v in library_spectra.values()],
        "z": [int(v["prec_z"]) for v in library_spectra.values()]
    }).unique()

    # Semi-join to keep only rows where (seq, z) exists in the library
    df = df.join(valid_keys_df, on=["seq", "z"], how="semi")

    logger.info("Computing spectral angle")

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
        .map_elements(
            partial(spectral_angle_polars_udf, fragment_library_map=fragment_library_map),
            return_dtype=pl.Float64
        )
        .alias('spectral_contrast_angle')
    )

    logger.info("Computing Scribe score")
    df = df.with_columns(
        pl.struct([
            'frag_charges',
            'frag_kinds',
            'frag_fragment_ordinals',
            'frag_intensities',
            'seq',
            'z',
        ])
        .map_elements(
            partial(scribe_score_polars_udf, fragment_library_map=fragment_library_map),
            return_dtype=pl.Float64
        )
        .alias('scribe_score')
    )

    logger.info("Computing matched library intensity percentage")
    df = df.with_columns(
        pl.struct([
            'frag_charges',
            'frag_kinds',
            'frag_fragment_ordinals',
            'frag_intensities',
            'seq',
            'z',
        ])
        .map_elements(
            partial(matched_lib_intensity_pct_udf, fragment_library_map=fragment_library_map),
            return_dtype=pl.Float64
        )
        .alias('matched_lib_pct')
    )


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

# Define the Polars UDF closure to capture the fragment_library_map
def spectral_angle_polars_udf(r: dict, fragment_library_map) -> float:
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


def scribe_score_polars_udf(r: dict, fragment_library_map: dict) -> float:
    """
    Polars UDF to calculate the primary Scribe Score:
    Score = -ln(sum((A_sqrt - L_sqrt)^2)), where A and L are normalized to sum to 1.
    """
    seq = r['seq']  # Modified sequence string
    z = r['z']  # Charge

    if seq is None or z is None:
        # Return a definite low score if key data is missing
        logger.debug(f"Scribe error: Missing data - seq={seq}, z={z}")
        return -999.0

    # 1. Get the library fragment vector for the peptide (using TUPLE KEY)
    library_key = (seq, z)
    library_vec = fragment_library_map.get(library_key)
    if not library_vec:
        logger.debug(f"Scribe error: Library key not found - {library_key}")
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


def matched_lib_intensity_pct_udf(r: dict, fragment_library_map: dict) -> float:
    """
    Calculate the percentage of library intensity covered by matched fragments.

    Returns: (sum of library intensities for matched fragments) / (sum of all library intensities) * 100
    """
    seq = r['seq']
    z = r['z']

    if seq is None or z is None:
        return -999.0

    library_key = (seq, z)
    library_vec = fragment_library_map.get(library_key)
    if not library_vec:
        return -999.0

    # Get observed fragments
    charges = r['frag_charges']
    kinds_raw = r['frag_kinds']
    ordinals = r['frag_fragment_ordinals']
    intensities = r['frag_intensities']

    if not (len(charges) == len(kinds_raw) == len(ordinals) == len(intensities)):
        return -999.0

    # Build observed fragment keys (only those with non-zero intensity)
    observed_keys = set()
    for c, k, o, i in zip(charges, kinds_raw, ordinals, intensities):
        if i > 0:
            key = (k.upper(), o, c)
            observed_keys.add(key)

    # Calculate matched library intensity
    matched_lib_intensity = sum(library_vec[k] for k in library_vec.keys() if k in observed_keys)
    total_lib_intensity = sum(library_vec.values())

    if total_lib_intensity <= 0:
        return -999.0

    return (matched_lib_intensity / total_lib_intensity) * 100


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
            mob = float(ms1_scan.mobility[closest_idx]) if ms1_scan.mobility is not None else None
            ms1_data.append({
                'Ms1_spec_id': ms1_scan.scan_num,
                "closest_peak_mz_ms1": closest_mz,
                "closest_peak_intensity_ms1": intensity,
                "precursor_mobility": mob,
            })
        else:
            # Use 0.0 or None/NaN placeholders for missing data
            ms1_data.append({
                'Ms1_spec_id': None,
                "closest_peak_mz_ms1": 0.0,
                "closest_peak_intensity_ms1": 0.0,
                "precursor_mobility": None,
            })

    return ms1_data


def lookup_fragment_mobilities(df: pl.DataFrame, dia_spectra):
    """Attach the observed ion mobility (1/K0) of each matched fragment.

    For every hit, each matched fragment's experimental m/z is joined back to
    its searched MS2 band spectrum (same spec_id) to read the parallel per-peak
    mobility. Fragments with no observed peak (experimental m/z <= 0) get None.
    A matched fragment whose experimental m/z has no corresponding peak in the
    spectrum is a genuine error and raises.
    """
    spec_ids = df['spec_id'].to_list()
    frag_exp_mzs = df['frag_mz_experimental'].to_list()

    out = []
    for spec_id, exp_mzs in zip(spec_ids, frag_exp_mzs):
        scan_num = spec_id if isinstance(spec_id, int) else Spectrum.extract_scannum(spec_id)
        spec = dia_spectra.get_by_idx(scan_num)
        mz = spec.mz
        mob = spec.mobility
        if mob is None:
            # Non-IM data (e.g. mzML): no per-fragment mobility exists.
            out.append([None] * len(exp_mzs))
            continue
        row = []
        for fmz in exp_mzs:
            if fmz is None or fmz <= 0.0:
                row.append(None)  # library fragment with no observed peak
                continue
            idx = int(np.searchsorted(mz, fmz))
            j = min((i for i in (idx - 1, idx) if 0 <= i < len(mz)),
                    key=lambda i: abs(mz[i] - fmz))
            if abs(mz[j] - fmz) > 1e-4:
                raise ValueError(
                    f"Spectrum {spec_id}: matched fragment m/z {fmz} has no peak "
                    f"(closest {mz[j]}, |diff|={abs(mz[j] - fmz):.6f})")
            row.append(float(mob[j]))
        out.append(row)
    return out


def adapt_output_df(df: pl.DataFrame, lib_rts: dict, rev_map: dict,
                    lib_ims: dict = None) -> pl.DataFrame:
    """
    Performs column normalization and peptide sequence reconstruction using Polars.
    """

    # Wrapper for map_mod_list (requires rev_map lookup)
    def map_mod_list_wrapper(mod_list: list[float]) -> list[str]:
        out = []
        for m in mod_list:
            if m != 0.0:
                name = rev_map.get(round(m, 4))
                out.append(name if name is not None else "")
            else:
                out.append("")
        return out

    # Wrapper for library RT lookup
    def map_rt_wrapper(seq: str) -> float:
        return lib_rts.get(seq, np.nan)

    # Rename columns
    df = df.rename({
        'spec_id': 'spec_name',
        'charge': 'z',
        'sequence': 'stripped_seq',
        'theoretical_mz': 'mz'
    })

    # Make matching spec_id column
    df = df.with_columns(
        pl.col('spec_name')
        .map_elements(lambda s: Spectrum.extract_scannum(s), return_dtype=pl.UInt32)
        .alias('spec_id')
    )

    # Reconstruct modification names
    df = df.with_columns(
        pl.col("modifications")
        .map_elements(map_mod_list_wrapper, return_dtype=pl.List(pl.Utf8))
        .alias("modification_names")
    )

    # Rename modified_peptide to seq
    df = df.rename({"modified_peptide": "seq"})

    # Grab library rts based on seq
    df = df.with_columns(
        pl.col("seq")
          .map_elements(map_rt_wrapper, return_dtype=pl.Float64)
          .alias("lib_rt")
    )

    # Grab library ion mobilities based on seq.  Mirrors lib_rt above; stays all-NaN
    # when the library carries no IM column, which the alignment gate treats as
    # "no library IM" rather than an error.
    if lib_ims is not None:
        df = df.with_columns(
            pl.col("seq")
              .map_elements(lambda s: lib_ims.get(s, np.nan), return_dtype=pl.Float64)
              .alias("lib_im")
        )

    return df


def peptide_to_mod_array(peptide_str, mod_dict):
    """
    Convert peptide string with modifications to a float array.

    Supports two modification notations:
      - Parenthesized names, looked up in ``mod_dict``:  ``K(UniMod:4)PEPTIDE``
      - Bracketed numeric masses (sign required):        ``K[+57.0]PEPTIDE``
    """
    mod_pattern = re.compile(r'\([^\)]+\)|\[[^\]]+\]')
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
        elif char == '(' or char == '[':
            close = ')' if char == '(' else ']'
            j = peptide_str.index(close, i)
            mod_token = peptide_str[i + 1:j]
            # Prefer the dict's exact monoisotopic mass when the token is a
            # known key (e.g. "+57.0" → 57.021464, "UniMod:4" → 57.021464);
            # fall back to parsing as a raw numeric mass.
            if mod_token in mod_dict:
                mod_mass = mod_dict[mod_token]
            else:
                try:
                    mod_mass = float(mod_token)
                except ValueError:
                    raise ValueError(f"Unknown modification: {mod_token}")

            next_char = peptide_str[j + 1] if j + 1 < len(peptide_str) else ''
            next_is_mod = next_char == '(' or next_char == '['

            # --- N-terminal logic ---
            if seq_index == 1 and mods_after_first_aa == 0:
                # First mod after the first residue → N-term
                mod_array[0] += mod_mass
                mods_after_first_aa += 1

            # --- C-terminal logic ---
            elif seq_index == seq_len:
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


def extract_mod_names(mod_seq: str):
    """
    From a mod_seq like 'ACD(Phospho)EFG(ox)K[+57.0]', return
    ['Phospho', 'ox', '[+57.0]']. Paren-mods are returned as bare names;
    bracket-mods are returned with their brackets so the caller can tell
    the two notations apart. Duplicates removed automatically.
    """
    paren = re.compile(r'\(([^\)]+)\)')
    bracket = re.compile(r'(\[[^\]]+\])')
    return list(set(paren.findall(mod_seq)) | set(bracket.findall(mod_seq)))


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

