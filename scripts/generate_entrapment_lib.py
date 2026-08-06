#!/usr/bin/env python
"""Generate an entrapment spectral library by shuffling peptide sequences.

For each precursor in the input library, the amino acid sequence is randomly
shuffled (keeping the C-terminal residue fixed and preserving modifications on
their new positions). Fragment ion m/z values are recomputed for the shuffled
sequence while intensities are preserved.

Output contains BOTH original target precursors (protein names suffixed with
``_0``) and shuffled entrapment precursors (suffixed with ``_1``).

Usage::

    python scripts/generate_entrapment_lib.py data/filtered_library.tsv -o entrapment.tsv
    python scripts/generate_entrapment_lib.py data/filtered_library.tsv -o entrapment.tsv --seed 42
    python scripts/generate_entrapment_lib.py data/filtered_library.tsv -o entrapment.tsv -j 8
"""

import argparse
import csv
import hashlib
import multiprocessing
import os
import random
import re
import sys

import numpy as np
import tqdm
from pyteomics import mass

# ── Amino acid parsing ────────────────────────────────────────────────────────

_TOKEN_RE = re.compile(r"[A-Z](?:\(.*?\)|\[.*?\])*")


def parse_peptide(seq: str) -> list[str]:
    """Split a modified peptide string into tokens, e.g.
    'PEP(+79.97)TIDE' -> ['P', 'E', 'P(+79.97)', 'T', 'I', 'D', 'E']
    """
    return _TOKEN_RE.findall(seq)


def strip_mods(token: str) -> str:
    """Return bare AA letter from a token like 'C(+57.02)'."""
    return token[0]


def extract_mods(token: str) -> list[str]:
    """Return list of modification strings from a token."""
    return re.findall(r"[\(\[].*?[\)\]]", token)


# ── Modification mass lookup ─────────────────────────────────────────────────

KNOWN_MOD_MASSES = {
    "UniMod:4":  57.02146,   # Carbamidomethyl
    "UniMod:1":  42.01057,   # Acetyl
    "UniMod:35": 15.99491,   # Oxidation
    "UniMod:21": 79.96633,   # Phospho
    "UniMod:7":  0.98402,    # Deamidated
}


def mod_mass(mod_str: str) -> float:
    """Parse a modification string and return its mass."""
    inner = mod_str.strip("()[]")
    if inner.startswith("+") or inner.startswith("-"):
        return float(inner)
    if inner in KNOWN_MOD_MASSES:
        return KNOWN_MOD_MASSES[inner]
    try:
        return mass.calculate_mass(formula=inner)
    except Exception:
        print(f"Warning: unknown mod '{inner}', treating as 0 mass", file=sys.stderr)
        return 0.0


# ── Sequence shuffling ────────────────────────────────────────────────────────

def shuffle_seq(tokens: list[str], rng: random.Random) -> list[str]:
    """Shuffle AA tokens, keeping the last residue (C-term) fixed.
    Modifications stay attached to their amino acid during the shuffle.
    Single shuffle — caller is responsible for retries.
    """
    if len(tokens) <= 2:
        return list(tokens)

    body = list(tokens[:-1])
    rng.shuffle(body)
    return body + [tokens[-1]]


# ── Fragment m/z computation ──────────────────────────────────────────────────

def compute_frag_mz(unmod_seq: list[str], mod_masses: list[float],
                     frag_name: str) -> float:
    """Compute m/z for a single fragment given the (shuffled) sequence."""
    ion_part, charge_str = frag_name.split("_")
    charge = int(charge_str)
    ion_type = ion_part[0]
    rest = ion_part[1:]

    parts = rest.split("-")
    ion_nmr = int(parts[0])
    loss = 0.0
    if len(parts) > 1:
        try:
            loss = mass.calculate_mass(formula=parts[1])
        except Exception:
            loss = 0.0

    if ion_type == "b":
        mz = (mass.fast_mass(unmod_seq[:ion_nmr], ion_type, charge)
              - loss / charge
              + sum(mod_masses[:ion_nmr]) / charge)
    elif ion_type == "y":
        mz = (mass.fast_mass(unmod_seq[-ion_nmr:], ion_type, charge)
              - loss / charge
              + sum(mod_masses[-ion_nmr:]) / charge)
    else:
        raise ValueError(f"Unknown ion type '{ion_type}' in fragment '{frag_name}'")

    return mz


# ── Multiprocessing worker ────────────────────────────────────────────────────

_MAX_COLLISION_RETRIES = 50
_worker_existing_mod_seqs = None


def _init_worker(existing_mod_seqs):
    global _worker_existing_mod_seqs
    _worker_existing_mod_seqs = existing_mod_seqs


def _entrapment_worker(args):
    """Worker function for parallel entrapment generation.

    Uses a per-peptide deterministic seed: hash(mod_seq) + user_seed.
    Retries with incremented seed (up to _MAX_COLLISION_RETRIES) if the
    shuffled sequence collides with an existing target modified sequence.
    """
    entry, user_seed = args
    existing_mod_seqs = _worker_existing_mod_seqs
    mod_seq = entry["mod_seq"]
    tokens = parse_peptide(mod_seq)

    # Check if distinct shuffles are even possible (e.g., AAAAAAR cannot shuffle)
    # Only the body (excluding C-term) is shuffled
    shuffleable = len(tokens) > 2 and len(set(strip_mods(t) for t in tokens[:-1])) > 1

    # Per-peptide deterministic seed (hashlib for cross-run reproducibility)
    base_seed = int(hashlib.md5(mod_seq.encode()).hexdigest(), 16) + user_seed

    if not shuffleable:
        rng = random.Random(base_seed)
        shuffled_tokens = shuffle_seq(tokens, rng)
        new_mod_seq = "".join(shuffled_tokens)
    else:
        for attempt in range(_MAX_COLLISION_RETRIES):
            rng = random.Random(base_seed + attempt)
            shuffled_tokens = shuffle_seq(tokens, rng)
            new_mod_seq = "".join(shuffled_tokens)
            if new_mod_seq != mod_seq and new_mod_seq not in existing_mod_seqs:
                break
        else:
            print(f"Warning: could not resolve collision for {mod_seq} "
                  f"after {_MAX_COLLISION_RETRIES} attempts", file=sys.stderr)

    unmod_seq = [strip_mods(t) for t in shuffled_tokens]
    per_residue_mod_mass = [
        sum(mod_mass(m) for m in extract_mods(t)) for t in shuffled_tokens
    ]

    # Recompute precursor m/z
    z = int(entry["prec_z"])
    new_prec_mz = mass.fast_mass(unmod_seq, charge=z) + sum(per_residue_mod_mass) / z

    # Recompute fragment m/z for every fragment — no silent drops
    new_frags = {}
    for frag_key, (old_mz, intensity) in entry["frags"].items():
        new_mz = compute_frag_mz(unmod_seq, per_residue_mod_mass, frag_key)
        new_frags[frag_key] = [new_mz, intensity]

    stripped = "".join(unmod_seq)
    return {
        "mod_seq": new_mod_seq,
        "seq": stripped,
        "prec_mz": new_prec_mz,
        "prec_z": entry["prec_z"],
        "iRT": entry["iRT"],
        "protein_group": entry["protein_group"].removesuffix("_0") + "_1",
        "protein_name": entry["protein_name"].removesuffix("_0") + "_1",
        "genes": entry["genes"].removesuffix("_0") + "_1",
        "IonMob": entry.get("IonMob", ""),
        "frags": new_frags,
    }


# ── Retention time prediction ─────────────────────────────────────────────────

_AA_STR = "ACDEFGHIKLMNPQRSTVWY"
_MAX_SEQ_LEN = 30
_NUM_AA = 20
_AA_TO_IDX = {aa: i for i, aa in enumerate(_AA_STR)}

# Model base paths relative to rt_models/ dir, keyed by tag name fragment
_MODEL_PATHS = {
    "LF":      "iRT_CNN_model_LF_09182024_",
    "mTRAQ":   "iRT_CNN_model_mTRAQ_09182024_",
    "diethyl": "iRT_CNN_model_DiEthyl_11052024_",
    "PSMtag":  "iRT_TransferLearning_Tag6_updated_05072025_",
}


def _one_hot_encode(sequence: str) -> np.ndarray:
    """One-hot encode a stripped peptide sequence to (30, 20) array."""
    enc = np.zeros((_MAX_SEQ_LEN, _NUM_AA), dtype=np.float32)
    for i, aa in enumerate(sequence[:_MAX_SEQ_LEN]):
        idx = _AA_TO_IDX.get(aa.upper())
        if idx is not None:
            enc[i, idx] = 1.0
    return enc


def predict_irts(sequences: list[str], tag: str = "LF") -> np.ndarray:
    """Predict iRT values for a list of stripped peptide sequences.

    Uses the pre-trained CNN ensemble (3 models, averaged).
    """
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")

    rt_models_dir = os.path.join(os.path.dirname(__file__), "..", "rt_models")
    base = _MODEL_PATHS.get(tag)
    if base is None:
        raise ValueError(f"Unknown tag '{tag}', choose from: {list(_MODEL_PATHS)}")
    model_path = os.path.join(rt_models_dir, base)

    models = []
    for i in range(3):
        model = tf.keras.models.load_model(model_path + str(i), compile=False)
        models.append(model)

    X = np.array([_one_hot_encode(seq) for seq in sequences])
    predictions = np.mean([m.predict(X, batch_size=4096, verbose=0) for m in models], axis=0).flatten()
    return predictions


# ── Library I/O ──────────────────────────────────────────────────────────────

def load_tsv_library(path: str) -> list[dict]:
    """Load a DIA-NN format TSV library.

    Returns a list of precursor dicts (one per unique precursor row-group),
    preserving input order and allowing duplicates.
    """
    # Use an ordered dict to group fragments by precursor while preserving
    # insertion order, then convert to list at the end.
    from collections import OrderedDict
    seen = OrderedDict()  # key -> index into entries list
    entries = []

    with open(path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if "ModifiedPeptide" in row:
                mod_seq = row["ModifiedPeptide"].strip("_")
            elif "ModifiedSequence" in row:
                mod_seq = row["ModifiedSequence"].strip("_")
            else:
                raise ValueError("No ModifiedPeptide or ModifiedSequence column found")

            z = float(row["PrecursorCharge"])

            frag_type = row.get("FragmentType", "y")
            frag_nmr = row.get("FragmentSeriesNumber", "1")
            frag_z = row.get("FragmentCharge", "1")
            loss_type = row.get("FragmentLossType", "noloss")
            frag_key = f"{frag_type}{frag_nmr}"
            if loss_type and loss_type != "noloss":
                frag_key += f"-{loss_type}"
            frag_key += f"_{frag_z}"

            frag_mz = float(row.get("ProductMz") or row["FragmentMz"])
            frag_int = float(row.get("LibraryIntensity") or row["RelativeIntensity"])

            # Use (mod_seq, z, frag_key) to detect true precursor boundaries:
            # a new precursor starts when (mod_seq, z) differs from the previous.
            # But within the same precursor, different frag_keys are just more
            # fragments. We track by (mod_seq, z) and accumulate fragments.
            prec_key = (mod_seq, z)
            if prec_key in seen:
                idx = seen[prec_key]
                entries[idx]["frags"][frag_key] = [frag_mz, frag_int]
            else:
                entry = {
                    "mod_seq": mod_seq,
                    "seq": row.get("StrippedPeptide") or row.get("PeptideSequence", ""),
                    "prec_mz": float(row["PrecursorMz"]),
                    "prec_z": z,
                    "iRT": float(row.get("Tr_recalibrated") or row.get("RT") or row.get("iRT") or 0),
                    "protein_group": row.get("ProteinGroup", ""),
                    "protein_name": row.get("ProteinName", ""),
                    "genes": row.get("Genes", ""),
                    "IonMob": row.get("IonMobility", ""),
                    "frags": {frag_key: [frag_mz, frag_int]},
                }
                seen[prec_key] = len(entries)
                entries.append(entry)

    return entries


DIANN_NAMES = [
    "PrecursorMz", "ModifiedPeptide", "PrecursorCharge",
    "Tr_recalibrated", "PeptideSequence", "IonMobility",
    "ProductMz", "LibraryIntensity", "ProteinGroup",
    "ProteinName", "Genes", "FragmentType", "FragmentCharge",
    "FragmentSeriesNumber", "FragmentLossType",
    "EntrapmentGroupId", "PrecursorIdx",
]

JMOD_TO_DIANN = {
    "prec_mz": "PrecursorMz",
    "mod_seq": "ModifiedPeptide",
    "prec_z": "PrecursorCharge",
    "iRT": "Tr_recalibrated",
    "seq": "PeptideSequence",
    "IonMob": "IonMobility",
    "protein_group": "ProteinGroup",
    "protein_name": "ProteinName",
    "genes": "Genes",
    "entrapment_group_id": "EntrapmentGroupId",
    "precursor_idx": "PrecursorIdx",
}


def _entry_to_rows(entry: dict) -> list[list]:
    """Convert a single library entry to a list of TSV rows."""
    precursor = {JMOD_TO_DIANN[k]: entry[k] for k in JMOD_TO_DIANN if k in entry}
    precursor["PrecursorCharge"] = int(entry["prec_z"])

    rows = []
    for frag_key, (mz, intensity) in entry["frags"].items():
        ion_part, frag_z = frag_key.split("_")
        parts = ion_part[1:].split("-")
        frag_type = ion_part[0]
        frag_nmr = int(parts[0])
        loss = parts[1] if len(parts) > 1 else "noloss"

        row_dict = dict(precursor)
        row_dict["ProductMz"] = mz
        row_dict["LibraryIntensity"] = intensity
        row_dict["FragmentType"] = frag_type
        row_dict["FragmentCharge"] = int(frag_z)
        row_dict["FragmentSeriesNumber"] = frag_nmr
        row_dict["FragmentLossType"] = loss

        rows.append([row_dict.get(col, "") for col in DIANN_NAMES])
    return rows


def write_tsv_library(target_lib: list[dict], entrapment_lib: list[dict], path: str):
    """Write combined target + entrapment library to DIA-NN format TSV."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t", quotechar="|", quoting=csv.QUOTE_MINIMAL)
        writer.writerow(DIANN_NAMES)

        for entry in target_lib:
            writer.writerows(_entry_to_rows(entry))
        for entry in entrapment_lib:
            writer.writerows(_entry_to_rows(entry))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate entrapment spectral library by shuffling peptide sequences"
    )
    parser.add_argument("input", help="Input spectral library (DIA-NN TSV format)")
    parser.add_argument("-o", "--output", required=True, help="Output TSV path")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    parser.add_argument("-j", "--jobs", type=int, default=None,
                        help="Number of parallel workers (default: all CPUs)")
    parser.add_argument("--tag", choices=list(_MODEL_PATHS), default="LF",
                        help="RT model tag (default: LF)")
    parser.add_argument("--keep-rt", action="store_true",
                        help="Keep original iRT values instead of predicting new ones")
    args = parser.parse_args()

    print(f"Loading library: {args.input}")
    library = load_tsv_library(args.input)
    print(f"  {len(library)} precursors loaded")

    # Tag target proteins with _0 suffix
    for entry in library:
        entry["protein_group"] = entry["protein_group"] + "_0"
        entry["protein_name"] = entry["protein_name"] + "_0"
        entry["genes"] = entry["genes"] + "_0"

    # Build set of existing modified sequences for target collision detection in workers
    existing_mod_seqs = frozenset(entry["mod_seq"] for entry in library)

    # Build worker args — plain dicts for pickling
    worker_args = [(entry, args.seed) for entry in library]

    print(f"Generating entrapment library (seed={args.seed}, jobs={args.jobs or 'all'})...")
    entrapment = []
    n_workers = args.jobs or multiprocessing.cpu_count()
    with multiprocessing.Pool(n_workers, initializer=_init_worker, initargs=(existing_mod_seqs,)) as pool:
        for result in tqdm.tqdm(pool.imap(_entrapment_worker, worker_args, chunksize=256),
                                total=len(worker_args), desc="Generating entrapments"):
            entrapment.append(result)

    print(f"  {len(entrapment)} entrapment precursors generated")

    # Post-hoc deduplication of entrapment-entrapment collisions
    # Check (mod_seq, prec_z) tuples — same shuffled sequence at different
    # charge states is expected and not a collision.
    seen_precursors = {(entry["mod_seq"], entry["prec_z"]) for entry in library}
    duplicates = []
    for i, entry in enumerate(entrapment):
        key = (entry["mod_seq"], entry["prec_z"])
        if key in seen_precursors:
            duplicates.append(i)
        else:
            seen_precursors.add(key)

    if duplicates:
        print(f"  {len(duplicates)} entrapment-entrapment collisions detected, resolving...")
        _init_worker(existing_mod_seqs)  # set global for main-process retries
        for dup_idx in duplicates:
            entry = library[dup_idx]
            resolved = False
            for seed_offset in range(_MAX_COLLISION_RETRIES, _MAX_COLLISION_RETRIES * 2):
                result = _entrapment_worker((entry, args.seed + seed_offset))
                key = (result["mod_seq"], result["prec_z"])
                if key not in seen_precursors:
                    seen_precursors.add(key)
                    entrapment[dup_idx] = result
                    resolved = True
                    break
            if not resolved:
                print(f"Warning: dropping entrapment for {entry['mod_seq']} — "
                      f"could not resolve collision", file=sys.stderr)
                entrapment[dup_idx] = None

        dropped = {i for i, e in enumerate(entrapment) if e is None}
        if dropped:
            print(f"  {len(dropped)} entrapments dropped due to unresolvable collisions")
            # Also drop the paired targets so indices stay aligned
            library = [e for i, e in enumerate(library) if i not in dropped]
            entrapment = [e for e in entrapment if e is not None]

    # Assign iRT values
    if args.keep_rt:
        print("Keeping original iRT values")
        # Entrapment entries already have the target's iRT from the worker
    else:
        print(f"Predicting iRTs (model={args.tag})...")
        unique_seqs = list({entry["seq"] for entry in library} |
                           {entry["seq"] for entry in entrapment})
        predicted = predict_irts(unique_seqs, tag=args.tag)
        seq_to_irt = dict(zip(unique_seqs, predicted))
        for entry in library:
            entry["iRT"] = float(seq_to_irt[entry["seq"]])
        for entry in entrapment:
            entry["iRT"] = float(seq_to_irt[entry["seq"]])

    # Assign EntrapmentGroupId and PrecursorIdx
    N = len(library)
    for i, entry in enumerate(library):
        entry["entrapment_group_id"] = 0
        entry["precursor_idx"] = i
    for i, entry in enumerate(entrapment):
        entry["entrapment_group_id"] = 1
        entry["precursor_idx"] = i

    n_target_frags = sum(len(e["frags"]) for e in library)
    n_entrap_frags = sum(len(e["frags"]) for e in entrapment)
    print(f"  {len(library)} target + {len(entrapment)} entrapment precursors")
    print(f"  {n_target_frags} target + {n_entrap_frags} entrapment fragments")

    print(f"Writing: {args.output}")
    write_tsv_library(library, entrapment, args.output)
    print("Done.")


if __name__ == "__main__":
    main()
