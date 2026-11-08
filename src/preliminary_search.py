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

import numpy as np
import peppy_sage as ps
import polars as pl
import re
from functools import partial
from tqdm.auto import tqdm
from numpy import linalg as la
from numba import njit

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

    # The store is already columnar, and peppy_sage reads it in that form. Going
    # via to_diann_df would repeat every precursor field across all of that
    # precursor's fragments, and that long-form intermediate -- not the ingest --
    # is the peak-memory event on a large library.
    n = library_spectra.n_targets
    mod_seqs = library_spectra.mod_seq[:n]

    # Modifications are a function of the modified peptide, so compute once per
    # unique one. Doubles as the unique-peptide set `observed_mods` needs.
    _mods_by_pep: dict[str, np.ndarray] = {}
    for ms in mod_seqs:
        if ms not in _mods_by_pep:
            _mods_by_pep[ms] = np.asarray(peptide_to_mod_array(ms, mod_dict),
                                          dtype=np.float32)

    # CSR: entry i's mod masses are mods_values[mods_offsets[i]:mods_offsets[i+1]]
    mods_offsets = np.zeros(n + 1, dtype=np.int64)
    np.cumsum([len(_mods_by_pep[ms]) for ms in mod_seqs], out=mods_offsets[1:])
    mods_values = np.empty(int(mods_offsets[-1]), dtype=np.float32)
    for i, ms in enumerate(mod_seqs):
        mods_values[mods_offsets[i]:mods_offsets[i + 1]] = _mods_by_pep[ms]

    # observed_mods only needs unique mod_seqs
    observed_mods: set[str] = set()
    for ms in _mods_by_pep:
        observed_mods.update(extract_mod_names(ms))
    rev_map = {}
    for m in observed_mods:
        if m.startswith('[') and m.endswith(']'):
            inner = m[1:-1]
            mass = mod_dict[inner] if inner in mod_dict else float(inner)
        else:
            mass = mod_dict[m]
        rev_map[round(mass, 4)] = m

    # frag_* are passed whole; frag_offsets/frag_lengths are sliced to targets, so
    # only target fragments are read and the big arrays are never copied.
    db = ps.IndexedDatabase.from_columnar(
        seq=library_spectra.seq[:n],
        mod_seq=mod_seqs,
        mods_values=mods_values,
        mods_offsets=mods_offsets,
        frag_mz=library_spectra.frag_mz,
        frag_intensity=library_spectra.frag_int,
        frag_codes=library_spectra.frag_keys_data,
        frag_offsets=library_spectra.frag_offsets[:n],
        frag_lengths=library_spectra.frag_lengths[:n],
        bucket_size=4096,
        ion_kinds=["b", "y"],
        min_ion_index=2,
        generate_decoys=False,
        decoy_tag="rev_",
        peptide_min_mass=0.0,
        peptide_max_mass=5000.0,
    )
    del _mods_by_pep, mods_values, mods_offsets

    # Create scorer
    # I don't think min_isotope_error needs to be touched for DIA data since we don't care what peaks are annotated as
    scorer = ps.Scorer(
        precursor_tol_da=(-1,1), # Unused in WWA -> defaults to window tol
        # TODO: optimize the fragment tolerance for the preliminary search.  It is
        # a fixed 10 ppm (ms2_ppm_error, from MZRTfit), chosen before anything
        # about the run's fragment mass accuracy is known; uncalibrated raw
        # timsTOF ions sit on a 3-5 ppm TOF grid, so 10 ppm may be too wide.
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



    # 2. Theoretical m/z from calcmass, the neutral mass Sage already computed.
    # m/z = (M + z*proton) / z, and the z cancels to M/z + proton.
    df = df.with_columns(
        (pl.col("calcmass").cast(pl.Float64) / pl.col("charge") + 1.00727647)
        .alias("mz")
    )

    # 3. MS1 Lookup: each PSM's precursor peak in its nearest MS1 scan
    logger.info("Looking up MS1 data")
    ms1_df = lookup_ms1_data(df, dia_spectra, ms1_ppm_error)

    # Add the MS1 data columns back to the main DataFrame
    # Since the data was iterated in order, we can just horizontally stack (hstack)
    df = df.hstack(ms1_df)

    # 4. Vectorized Error Calculations (Polars Expressions)
    df = df.with_columns([
        # Calculate difference
        (pl.col("closest_peak_mz_ms1") - pl.col("mz")).alias("mz_diff")
    ]).with_columns([
        # Calculate relative error
        (pl.col("mz_diff") / pl.col("mz")).alias("relative_error_ms1")
    ]).with_columns([
        # Calculate PPM error
        (pl.col("relative_error_ms1") * 1_000_000).alias("ppm_error_ms1")
    ])
    # Clean up the intermediate column
    df = df.drop("mz_diff")

    # 5. Filter out large MS1 errors (Polars syntax)
    df = df.filter(pl.col('ppm_error_ms1').abs() < ms1_ppm_error)

    # Everything below looks the library up only for the peptides the search hit,
    # so only their entries are read, not the whole library
    hit_entries = library_entries_for(library_spectra, df["modified_peptide"])

    # Get RTs for alignment
    lib_rts = {v['mod_seq'] : v['iRT'] for v in hit_entries}
    # Library ion mobilities, for the IM alignment (NaN when the library has no IM column)
    lib_ims = {v['mod_seq'] : v['IonMob'] for v in hit_entries}

    # Adapt the dataframe to the format expected downstream
    logger.info("Adapting output dataframe")
    df = adapt_output_df(df, lib_rts, rev_map, lib_ims=lib_ims)

    # Calculate spectral angle
    logger.info("Creating fragment library map")
    fragment_library_map = create_fragment_library_map(hit_entries)

    # Filter to only peptides present in the library (before computing expensive scores)
    valid_keys_df = pl.DataFrame({
        "seq": [v["mod_seq"] for v in hit_entries],
        "z": [int(v["prec_z"]) for v in hit_entries]
    }).unique()
    del hit_entries

    # Semi-join to keep only rows where (seq, z) exists in the library
    df = df.join(valid_keys_df, on=["seq", "z"], how="semi")

    logger.info("Computing spectral angle, Hellinger, Scribe and matched library intensity")
    df = df.with_columns(score_psms(df, fragment_library_map))

    # Drop rows the score UDFs could not score. Those three return -999 when
    # _align_to_library fails or the intensities sum to zero -- a sentinel meaning
    # "no value", not "a bad value".
    _sentinel_cols = ['hellinger_score', 'scribe_score', 'matched_lib_pct']
    _n_before = df.height
    df = df.filter(
        pl.all_horizontal([pl.col(c) != -999.0 for c in _sentinel_cols])
    )
    if df.height < _n_before:
        logger.info(f"Dropped {_n_before - df.height} unscorable PSMs "
                    f"(no library alignment or zero matched intensity)")

    return df


def library_entries_for(library_spectra, mod_seqs):
    """The entries of *library_spectra* whose mod_seq is one of *mod_seqs*.

    Every charge of each peptide, in library order, as library_spectra.values()
    gives them, so a dict built from these holds what one built from the whole
    library holds for those peptides.  The match is vectorised; only the
    matched entries are read.
    """
    n = len(library_spectra)
    library_mod_seq = library_spectra.mod_seq
    is_hit = pl.Series(library_mod_seq[:n], dtype=pl.String).is_in(
        pl.Series(mod_seqs, dtype=pl.String).unique()).to_numpy()
    prec_z = library_spectra.prec_z
    return [library_spectra[(library_mod_seq[i], float(prec_z[i]))] for i in np.flatnonzero(is_hit)]


# Neutral loss name -> code, matching peppy_sage's `loss_code` and the packed
# encoding in src/utils/frag_encoding.py. `None` is the regex's no-loss group.
LOSS_NAME_TO_CODE = {None: 0, '': 0, 'H2O': 1, 'NH3': 2, 'H3PO4': 3}


def create_fragment_library_map(library_entries):
    """
    Converts the library entries (dict-like, e.g. library_spectra.values()) into a
    nested, fast-lookup map for spectral comparison.

    Keys: (Modified Sequence, Precursor Charge) [TUPLE KEY REQUIRED FOR ACCURACY]
    Values: Dict mapping (ion_kind, ordinal, charge) -> RelativeIntensity

    This function is updated to correctly handle the tuple key.
    """
    frag_map = {}
    for data in library_entries:
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

            match = re.match(r'([BY])(\d+)(?:-(H2O|NH3|H3PO4))?$', frag_name)
            if not match:
                continue

            frag_type = match.group(1)  # B or Y
            frag_ordinal = int(match.group(2))  # 3, 7, etc.
            frag_loss = LOSS_NAME_TO_CODE[match.group(3)]

            intensity = mz_int_list[1]  # Relative Intensity is the second element

            # Key for alignment: (IonType, Ordinal, Charge, Loss). Loss is part of
            # a fragment's identity -- "Y7" and "Y7-H2O" are ~18 Da apart and are
            # separate entries in peppy_sage's library index, so collapsing them
            # here would align an observed ion against the wrong intensity.
            key = (frag_type, frag_ordinal, frag_z, frag_loss)
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
    aligned = _align_to_library(r, fragment_library_map, "Spectral angle")
    if aligned is None:
        return 0.0
    observed_vec, library_vec = aligned

    # Union, not library-only: an observed ion absent from the library adds to
    # ||A|| but nothing to the dot product, so extra ions are penalized.
    # sorted: set iteration order varies with PYTHONHASHSEED, and a different
    # float summation order perturbs the score at the last bit.
    all_keys = sorted(set(observed_vec) | set(library_vec))
    A_raw = np.array([observed_vec.get(k, 0.0) for k in all_keys], dtype=np.float32)
    B_raw = np.array([library_vec.get(k, 0.0) for k in all_keys], dtype=np.float32)

    # Power transformation (P=0.5)
    A_pow = np.sqrt(A_raw)
    B_pow = np.sqrt(B_raw)

    dot_product = np.dot(A_pow, B_pow)
    norm_A = la.norm(A_pow)
    norm_B = la.norm(B_pow)

    if norm_A == 0.0 or norm_B == 0.0:
        return 0.0

    cosine = float(np.clip(dot_product / (norm_A * norm_B), -1.0, 1.0))

    return 1.0 - 2.0 * np.arccos(cosine) / np.pi


def _align_to_library(r: dict, fragment_library_map: dict, label: str):
    """
    Parse a PSM row's matched fragments and fetch its library vector.

    Returns (observed_vec, library_vec) keyed on (ion_kind, ordinal, charge, loss),
    or None when the row cannot be scored. Shared by the two spectral-difference
    scores below.
    """
    seq, z = r['seq'], r['z']
    if seq is None or z is None:
        logger.debug(f"{label} error: Missing data - seq={seq}, z={z}")
        return None

    library_vec = fragment_library_map.get((seq, z))
    if not library_vec:
        logger.debug(f"{label} error: Library key not found - {(seq, z)}")
        return None

    charges, kinds = r['frag_charges'], r['frag_kinds']
    ordinals, intensities = r['frag_fragment_ordinals'], r['frag_intensities']
    # Libraries with no loss ions still report a zero per fragment; tolerate the
    # column being absent so an older search result stays scorable.
    losses = r.get('frag_losses') or [0] * len(charges)
    if not (len(charges) == len(kinds) == len(ordinals) == len(intensities)
            == len(losses)):
        logger.error(f"Fragment array length mismatch for seq {seq}. Lengths: "
                     f"{len(charges)}, {len(kinds)}, {len(ordinals)}, "
                     f"{len(intensities)}, {len(losses)}")
        return None

    observed_vec = {(k.upper(), o, c, l): i
                    for c, k, o, i, l in zip(charges, kinds, ordinals, intensities, losses)}
    return observed_vec, library_vec


def _neg_log_sse(A: np.ndarray, B: np.ndarray) -> float:
    """-ln of the sum of squared differences, capped where the match is exact."""
    sse = float(np.sum((A - B) ** 2))
    return 25.0 if sse <= 1e-12 else -np.log(sse)


def hellinger_score_polars_udf(r: dict, fragment_library_map: dict) -> float:
    """
    Squared Hellinger distance over the FULL library support, normalizing each vector
    to sum to 1 BEFORE the square root. Rooting second leaves unit L2 norm, which is
    what makes this Hellinger and not the Scribe score. Library ions with no observed
    peak contribute (0 - sqrt(L))^2, so this penalizes missing ions.
    """
    aligned = _align_to_library(r, fragment_library_map, "Hellinger score")
    if aligned is None:
        return -999.0
    observed_vec, library_vec = aligned

    keys = list(library_vec)
    A = np.array([observed_vec.get(k, 0.0) for k in keys], dtype=np.float64)
    B = np.array([library_vec[k] for k in keys], dtype=np.float64)
    if A.sum() == 0.0 or B.sum() == 0.0:
        return -999.0

    return _neg_log_sse(np.sqrt(A / A.sum()), np.sqrt(B / B.sum()))


def scribe_score_polars_udf(r: dict, fragment_library_map: dict) -> float:
    """
    Scribe score as published (Searle, Shannon & Wilburn, JPR 2023, PMID 36695531):

        Equation 1:  Scribe Score = -ln(sum_i (A_i - L_i)^2)

    The Methods specify square roots BEFORE scoring, normalization to sum 1 AFTER, and
    a sum over MATCHED ions -- all three differ from hellinger_score above, which
    normalizes first and sums over the full library. Both are computed; both feed the
    PCA in src/quality_pca.py.

    Restricting to matched ions means missing library ions cost nothing, so a sparse
    but clean match scores at the cap. That is deliberate: coverage is separately
    available as matched_lib_pct and matched_peaks.
    """
    aligned = _align_to_library(r, fragment_library_map, "Scribe")
    if aligned is None:
        return -999.0
    observed_vec, library_vec = aligned

    # sorted: set iteration order varies with PYTHONHASHSEED, and a different
    # float summation order perturbs the score at the last bit.
    keys = sorted(set(observed_vec) & set(library_vec))
    if not keys:
        return -999.0

    # Clipped because rooting before normalizing lets a negative intensity produce NaN,
    # which would pass silently through the percentile filter in empirical_fit.
    A = np.sqrt(np.clip([observed_vec[k] for k in keys], 0.0, None))
    B = np.sqrt(np.clip([library_vec[k] for k in keys], 0.0, None))
    if A.sum() == 0.0 or B.sum() == 0.0:
        return -999.0

    return _neg_log_sse(A / A.sum(), B / B.sum())


def matched_lib_intensity_pct_udf(r: dict, fragment_library_map: dict) -> float:
    """
    Calculate the percentage of library intensity covered by matched fragments.

    Returns: (sum of library intensities for matched fragments) / (sum of all library intensities) * 100
    """
    aligned = _align_to_library(r, fragment_library_map, "Matched lib pct")
    if aligned is None:
        return -999.0
    observed_vec, library_vec = aligned

    observed_keys = {k for k, i in observed_vec.items() if i > 0}
    matched_lib_intensity = sum(v for k, v in library_vec.items() if k in observed_keys)
    total_lib_intensity = sum(library_vec.values())

    if total_lib_intensity <= 0:
        return -999.0

    return (matched_lib_intensity / total_lib_intensity) * 100



# ---- first-search PSM scores, compiled -----------------------------------
# The four *_polars_udf scores above, computed for every PSM by one compiled
# kernel instead of four map_elements passes (~136 s for 2.4M PSMs in a run;
# Python per row, one thread).  Same rules: the last of duplicate observed ions
# wins, spectral angle over the sorted union in float32, Hellinger over the
# library ions in library order, Scribe over the sorted intersection, matched %
# with sequential sums.  Not bit-identical to the Python versions: numpy's
# pairwise sums and BLAS float32 dot products add in a different order (seen
# on 2.4M PSMs: matched % identical, Hellinger/Scribe within 4e-15, spectral
# angle within 2.1e-6).  matched % uses CPython's compensated sum() and is exact.  Fragment keys (kind, ordinal, charge, loss) are encoded
# as one int64 that sorts like the tuple.

@njit(nogil=True, cache=False)
def _psm_scores_jit(lo, hi, obs_off, obs_code, obs_int, lib_idx, lib_off, lib_code, lib_int, out):
    for h in range(lo, hi):
        li = lib_idx[h]
        if li < 0:
            out[h, 0] = 0.0; out[h, 1] = -999.0; out[h, 2] = -999.0; out[h, 3] = -999.0
            continue
        # observed: unique codes, the last occurrence of a duplicate wins (dict comprehension)
        a, b = obs_off[h], obs_off[h + 1]
        oc = obs_code[a:b]; oi = obs_int[a:b]
        order = np.argsort(oc, kind="mergesort")
        n = 0
        uc = np.empty(b - a, np.int64); ui = np.empty(b - a, np.float32)
        for k in range(order.size):
            c = oc[order[k]]
            if n > 0 and uc[n - 1] == c:
                ui[n - 1] = oi[order[k]]          # stable sort: later index overrides
            else:
                uc[n] = c; ui[n] = oi[order[k]]; n += 1
        uc = uc[:n]; ui = ui[:n]
        la, lb = lib_off[li], lib_off[li + 1]
        lc = lib_code[la:lb]; lv = lib_int[la:lb]
        lorder = np.argsort(lc, kind="mergesort")
        # --- spectral angle over the sorted union, float32 ---
        dot = np.float32(0.0); na = np.float32(0.0); nb = np.float32(0.0)
        i = 0; j = 0
        while i < n or j < lc.size:
            if j >= lc.size or (i < n and uc[i] < lc[lorder[j]]):
                av = np.sqrt(np.float32(ui[i])); bv = np.float32(0.0); i += 1
            elif i >= n or lc[lorder[j]] < uc[i]:
                av = np.float32(0.0); bv = np.sqrt(np.float32(lv[lorder[j]])); j += 1
            else:
                av = np.sqrt(np.float32(ui[i])); bv = np.sqrt(np.float32(lv[lorder[j]])); i += 1; j += 1
            dot += av * bv; na += av * av; nb += bv * bv
        nA = np.sqrt(na); nB = np.sqrt(nb)
        if nA == 0.0 or nB == 0.0:
            out[h, 0] = 0.0
        else:
            cosv = dot / (nA * nB)
            cosv = min(max(cosv, np.float32(-1.0)), np.float32(1.0))
            out[h, 0] = 1.0 - 2.0 * np.arccos(np.float64(cosv)) / np.pi
        # observed value at each library key (library order)
        A = np.zeros(lc.size); B = np.empty(lc.size)
        for t in range(lc.size):
            B[t] = lv[t]
            p = np.searchsorted(uc, lc[t])
            if p < n and uc[p] == lc[t]:
                A[t] = np.float64(ui[p])
        # --- Hellinger over the library keys ---
        sA = A.sum(); sB = B.sum()
        if sA == 0.0 or sB == 0.0:
            out[h, 1] = -999.0
        else:
            sse = 0.0
            for t in range(lc.size):
                d = np.sqrt(A[t] / sA) - np.sqrt(B[t] / sB); sse += d * d
            out[h, 1] = 25.0 if sse <= 1e-12 else -np.log(sse)
        # --- Scribe over the sorted intersection ---
        m = 0
        xa = np.empty(lc.size); xb = np.empty(lc.size)
        for jj in range(lc.size):
            t = lorder[jj]
            p = np.searchsorted(uc, lc[t])
            if p < n and uc[p] == lc[t]:
                xa[m] = np.sqrt(max(np.float64(ui[p]), 0.0)); xb[m] = np.sqrt(max(lv[t], 0.0)); m += 1
        if m == 0:
            out[h, 2] = -999.0
        else:
            sa = xa[:m].sum(); sb = xb[:m].sum()
            if sa == 0.0 or sb == 0.0:
                out[h, 2] = -999.0
            else:
                sse = 0.0
                for t in range(m):
                    d = xa[t] / sa - xb[t] / sb; sse += d * d
                out[h, 2] = 25.0 if sse <= 1e-12 else -np.log(sse)
        # --- matched library intensity %, library order; Python's sum() is
        # compensated (Neumaier, CPython >= 3.12), so the sums are too ---
        matched = 0.0; c_m = 0.0; total = 0.0; c_t = 0.0
        for t in range(lc.size):
            x = lv[t]
            s_ = total + x
            c_t += (total - s_) + x if abs(total) >= abs(x) else (x - s_) + total
            total = s_
            if A[t] > 0.0:
                s_ = matched + x
                c_m += (matched - s_) + x if abs(matched) >= abs(x) else (x - s_) + matched
                matched = s_
        if c_t != 0.0 and np.isfinite(c_t):
            total += c_t
        if c_m != 0.0 and np.isfinite(c_m):
            matched += c_m
        out[h, 3] = -999.0 if total <= 0.0 else (matched / total) * 100.0



def _flatten_for_scoring(df, fmap):
    kinds_u = df["frag_kinds"].explode().str.to_uppercase()
    names = sorted(set(kinds_u.drop_nulls().unique().to_list()) | {"B", "Y"})
    kcode = {k: i for i, k in enumerate(names)}
    ch = df["frag_charges"].explode().to_numpy().astype(np.int64)
    od = df["frag_fragment_ordinals"].explode().to_numpy().astype(np.int64)
    # Libraries with no loss ions report a zero per fragment; tolerate the column
    # being absent, as _align_to_library does
    ls = (df["frag_losses"].explode().fill_null(0).to_numpy().astype(np.int64) if "frag_losses" in df.columns
          else np.zeros(int(df["frag_charges"].list.len().fill_null(0).clip(1, None).sum()), np.int64))
    kc = kinds_u.replace_strict(list(kcode), list(kcode.values()), return_dtype=pl.Int64).to_numpy()
    lens = df["frag_charges"].list.len().fill_null(0).to_numpy()
    good = lens > 0
    # explode gives one null row for an empty list; drop those
    keep = np.repeat(good, np.where(good, lens, 1))
    ch, od, ls, kc = ch[keep], od[keep], ls[keep], kc[keep]
    oi = df["frag_intensities"].explode().to_numpy()[keep].astype(np.float32)
    obs_off = np.zeros(df.height + 1, np.int64); obs_off[1:] = np.cumsum(lens)
    # library: insertion order kept
    keys = list(fmap.keys())
    lib_off = np.zeros(len(keys) + 1, np.int64)
    lcode, lint = [], []
    for e, k in enumerate(keys):
        v = fmap[k]
        for (t, o, c, l), x in v.items():
            lcode.append((kcode[t], o, c, l)); lint.append(x)
        lib_off[e + 1] = len(lint)
    lcode = np.array(lcode, np.int64); lint = np.array(lint, np.float64)
    K1 = int(max(od.max(initial=0), lcode[:, 1].max()) + 1); K2 = int(max(ch.max(initial=0), lcode[:, 2].max()) + 1)
    K3 = int(max(ls.max(initial=0), lcode[:, 3].max()) + 1)
    enc = lambda k, o, c, l: ((k * K1 + o) * K2 + c) * K3 + l
    obs_code = enc(kc, od, ch, ls)
    lib_code = enc(lcode[:, 0], lcode[:, 1], lcode[:, 2], lcode[:, 3])
    # charges compared as numbers, as the dict lookup does (2 == 2.0)
    kf = pl.DataFrame({"seq": [k[0] for k in keys], "z": [float(k[1]) for k in keys], "lib_idx": np.arange(len(keys))},
                      schema={"seq": pl.String, "z": pl.Float64, "lib_idx": pl.Int64})
    lib_idx = (df.select("seq", pl.col("z").cast(pl.Float64)).join(kf, on=["seq", "z"], how="left", maintain_order="left")
               ["lib_idx"].fill_null(-1).to_numpy())
    return obs_off, obs_code, oi, lib_idx, lib_off, lib_code, lint



def score_psms(df: pl.DataFrame, fragment_library_map: dict, n_threads: int = None) -> list:
    """spectral_contrast_angle, hellinger_score, scribe_score and matched_lib_pct
    of every PSM in *df*, as polars Series (see the block comment above)."""
    from concurrent.futures import ThreadPoolExecutor
    from src import config
    arrs = _flatten_for_scoring(df, fragment_library_map)
    n = df.height
    out = np.empty((n, 4), dtype=np.float64)
    n_threads = n_threads or max(1, int(getattr(config.args, "threads", 1) or 1))
    step = max(1, -(-n // (n_threads * 8)))
    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        for f in [pool.submit(_psm_scores_jit, lo, min(lo + step, n), *arrs, out)
                  for lo in range(0, n, step)]:
            f.result()
    return [pl.Series(name, out[:, j], dtype=pl.Float64) for j, name in
            enumerate(("spectral_contrast_angle", "hellinger_score", "scribe_score", "matched_lib_pct"))]


def _most_intense_ms1_peak(ms1_scan, theo_mz, ppm, im_lo=None, im_hi=None):
    """Index of the most intense MS1 peak within *ppm* of *theo_mz*, or None.

    With a mobility band (*im_lo*, *im_hi*) and per-peak mobility, only peaks
    inside the band count: an MS1 frame spans the whole mobility range, so
    the band the PSM's MS2 spectrum was cut from keeps the choice to ions
    that can be its precursor.  Ties go to the lowest m/z.
    """
    mz = ms1_scan.mz
    tol = theo_mz * ppm * 1e-6
    lo = np.searchsorted(mz, theo_mz - tol, side="left")
    hi = np.searchsorted(mz, theo_mz + tol, side="right")
    if hi <= lo:
        return None
    idx = np.arange(lo, hi)
    mob = ms1_scan.mobility
    if im_lo is not None and mob is not None:
        m = mob[lo:hi]
        idx = idx[(m >= im_lo) & (m <= im_hi)]
        if idx.size == 0:
            return None
    return int(idx[np.argmax(ms1_scan.intens[idx])])


@njit(nogil=True)
def _most_intense_ms1_peaks_jit(mz, intens, mob, theo, ppm, gate, im_lo, im_hi, out):
    """_most_intense_ms1_peak for every query *theo[k]* against one MS1 scan:
    *out[k]* is the peak index, or -1.  *gate[k]* applies the mobility band
    (*im_lo*, *im_hi*, already in *mob*'s dtype, as numpy compares them)."""
    for k in range(theo.size):
        t = theo[k]
        tol = t * ppm * 1e-6
        lo = np.searchsorted(mz, t - tol, side="left")
        hi = np.searchsorted(mz, t + tol, side="right")
        best = -1
        for j in range(lo, hi):
            if gate[k] and not (mob[j] >= im_lo[k] and mob[j] <= im_hi[k]):
                continue
            if best == -1 or intens[j] > intens[best]:      # ties: lowest m/z
                best = j
        out[k] = best


def lookup_ms1_data(df: pl.DataFrame, dia_spectra, ms1_ppm_error) -> pl.DataFrame:
    """
    The precursor peak of every PSM: Ms1_spec_id, closest_peak_mz_ms1,
    closest_peak_intensity_ms1 and precursor_mobility, in *df*'s row order.

    The precursor peak is the most intense peak of the nearest MS1 scan within
    *ms1_ppm_error* of the theoretical m/z and, on ion mobility data, within
    the mobility band of the PSM's MS2 spectrum (see _most_intense_ms1_peak).
    PSMs with no such peak get the same placeholder as PSMs with no MS1 scan
    (m/z and intensity 0, no mobility), and fit_with_features then drops them
    with the other PSMs outside *ms1_ppm_error*.

    Each MS2 spectrum's MS1 scan and band are looked up once, and the PSMs are
    then matched one MS1 scan at a time.
    """
    n = df.height
    ids = df["spec_id"]
    unique_ids = ids.unique().sort()
    psm_spec = ids.rank("dense").to_numpy().astype(np.int64) - 1   # row -> unique_ids position
    theo_all = df["mz"].to_numpy().astype(np.float64)

    # Per MS2 spectrum: its MS1 scan (position in ms1_list, -1 for none) and band
    ms1_list, ms1_pos = [], {}
    spec_ms1 = np.full(len(unique_ids), -1, dtype=np.int64)
    spec_lo = np.full(len(unique_ids), np.nan)
    spec_hi = np.full(len(unique_ids), np.nan)
    spec_band = np.zeros(len(unique_ids), dtype=np.bool_)
    for u, spec_id in enumerate(unique_ids.to_list()):
        ms1_scan = dia_spectra.get_nearest_ms1_for_scan(spec_id)
        if not ms1_scan:
            continue
        spec_ms1[u] = ms1_pos.setdefault(id(ms1_scan), len(ms1_list))
        if spec_ms1[u] == len(ms1_list):
            ms1_list.append(ms1_scan)
        # spec_id is a scan number (.d) or a native ID string (mzML), as
        # get_nearest_ms1_for_scan accepts it
        scan_num = (int(re.search(r"scan=(\d+)", spec_id).group(1))
                    if isinstance(spec_id, str) else spec_id)
        ms2_scan = dia_spectra.ms2scans[dia_spectra.ms2_by_id[scan_num]]
        im_lo = getattr(ms2_scan, "im_lo", None)
        if im_lo is not None and ms1_scan.mobility is not None:
            spec_band[u] = True
            spec_lo[u] = im_lo
            spec_hi[u] = getattr(ms2_scan, "im_hi", None)

    psm_ms1 = spec_ms1[psm_spec]
    ms1_id = np.zeros(n, dtype=np.int64)
    has_ms1 = psm_ms1 >= 0
    peak_mz = np.zeros(n, dtype=np.float64)
    peak_int = np.zeros(n, dtype=np.float64)
    peak_mob = np.full(n, np.nan)
    has_mob = np.zeros(n, dtype=np.bool_)

    order = np.argsort(psm_ms1, kind="stable")
    bounds = np.searchsorted(psm_ms1[order], np.arange(len(ms1_list) + 1))
    for m, ms1_scan in enumerate(tqdm(ms1_list, desc="MS1 Peak Finding", leave=False)):
        rows = order[bounds[m]:bounds[m + 1]]
        if rows.size == 0:
            continue
        ms1_id[rows] = ms1_scan.scan_num
        mz, intens = np.asarray(ms1_scan.mz), np.asarray(ms1_scan.intens)
        mob = ms1_scan.mobility
        mob_arr = np.asarray(mob) if mob is not None else np.zeros(1, dtype=np.float32)
        spec = psm_spec[rows]
        gate = spec_band[spec]
        lo = spec_lo[spec].astype(mob_arr.dtype)
        hi = spec_hi[spec].astype(mob_arr.dtype)
        idx = np.empty(rows.size, dtype=np.int64)
        _most_intense_ms1_peaks_jit(mz, intens, mob_arr, theo_all[rows], float(ms1_ppm_error),
                                    gate, lo, hi, idx)
        hit = idx >= 0
        r, i = rows[hit], idx[hit]
        peak_mz[r] = mz[i]
        peak_int[r] = intens[i]
        if mob is not None:
            peak_mob[r] = mob_arr[i]
            has_mob[r] = True

    # Explicit schema: rows without an MS1 peak carry None, and when the first
    # rows are all None (centroided .d input) polars would infer a Null column.
    return pl.DataFrame({
        "Ms1_spec_id": pl.Series(ms1_id, dtype=pl.Int64).scatter(np.flatnonzero(~has_ms1), None),
        "closest_peak_mz_ms1": pl.Series(peak_mz, dtype=pl.Float64),
        "closest_peak_intensity_ms1": pl.Series(peak_int, dtype=pl.Float64),
        "precursor_mobility": pl.Series(peak_mob, dtype=pl.Float64).scatter(np.flatnonzero(~has_mob), None),
    })


def _by_distinct_value(expr: pl.Expr, values: pl.Series, fn, dtype) -> pl.Expr:
    """*expr* with each value replaced by fn(value).

    fn is called once per distinct value in *values* (the values *expr* can
    take) rather than once per row: map_elements on 3M first-search rows cost
    ~3 min, nearly all of it polars building and iterating a Series per row,
    while these columns hold a few thousand distinct values at most.
    """
    distinct = values.drop_nulls().unique()
    return expr.replace_strict(distinct, [fn(v) for v in distinct.to_list()], return_dtype=dtype)


def adapt_output_df(df: pl.DataFrame, lib_rts: dict, rev_map: dict,
                    lib_ims: dict = None) -> pl.DataFrame:
    """
    Performs column normalization and peptide sequence reconstruction using Polars.
    """

    # One modification mass -> its name ("" for no modification or an unknown mass)
    def mod_name(m: float) -> str:
        if m != 0.0:
            name = rev_map.get(round(m, 4))
            return name if name is not None else ""
        return ""

    # Rename columns
    df = df.rename({
        'spec_id': 'spec_name',
        'charge': 'z',
        'sequence': 'stripped_seq',
    })

    # Make matching spec_id column
    df = df.with_columns(
        _by_distinct_value(pl.col('spec_name'), df['spec_name'],
                           Spectrum.extract_scannum, pl.UInt32)
        .alias('spec_id')
    )

    # Reconstruct modification names
    df = df.with_columns(
        pl.col("modifications")
        .list.eval(_by_distinct_value(pl.element(), df["modifications"].explode(), mod_name, pl.Utf8))
        .alias("modification_names")
    )

    # Rename modified_peptide to seq
    df = df.rename({"modified_peptide": "seq"})

    # Grab library rts based on seq
    df = df.with_columns(
        _by_distinct_value(pl.col("seq"), df["seq"], lambda s: lib_rts.get(s, np.nan), pl.Float64)
        .alias("lib_rt")
    )

    # Grab library ion mobilities based on seq.  Mirrors lib_rt above; stays all-NaN
    # when the library carries no IM column, which the alignment gate treats as
    # "no library IM" rather than an error.
    if lib_ims is not None:
        df = df.with_columns(
            _by_distinct_value(pl.col("seq"), df["seq"], lambda s: lib_ims.get(s, np.nan), pl.Float64)
            .alias("lib_im")
        )

    return df


def peptide_to_mod_array(peptide_str, mod_dict):
    """
    Convert peptide string with modifications to a float array.

    Supports two modification notations:
      - Parenthesized names, looked up in ``mod_dict``:  ``K(UniMod:4)PEPTIDE``
      - Bracketed numeric masses (sign required):        ``K[+57.0]PEPTIDE``

    Slot 0 is the N-terminus: the modifications written in front of the first
    residue, e.g. ``(UniMod:1)PEPTIDE``.  Slots 1..n are the residues, and the
    last slot the C-terminus.
    """
    mod_pattern = re.compile(r'\([^\)]+\)|\[[^\]]+\]')
    clean_seq = mod_pattern.sub('', peptide_str)
    mod_array = np.zeros(len(clean_seq) + 2, dtype=float)

    seq_index = 0
    i = 0
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

            # --- N-terminal mods: in front of the first residue ---
            if seq_index == 0:
                mod_array[0] += mod_mass

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

