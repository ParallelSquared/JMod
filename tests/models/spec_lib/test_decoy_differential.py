"""Differential snapshots for decoy library generation.

create_decoy_lib output is snapshotted per (library, rules, tag) case so the
columnar rewrite can be verified against the current worker-pool
implementation. Comparison is exact for every array except fragment/spectrum
m/z, which may differ by <=1 ULP once y-ion masses come from prefix sums
(spectra are canonically re-sorted first, since a ULP shift can swap two
near-equal peaks). Regenerate deliberately with:

    JMOD_REGEN_SNAPSHOTS=1 .venv/bin/pytest tests/models/spec_lib/test_decoy_differential.py
"""
import os
import pickle

import numpy as np
import pytest
from pyteomics import mass

from src.models.spec_lib.library_store import SpectrumLibraryStore
from src.models.spec_lib.spec_lib import create_decoy_lib
from src.mass_tags import read_json_to_massTag
from src.utils.parse_peptides import change_seq

from tests.models.spec_lib.test_library_snapshots import (
    FIXTURE_DIR, SNAPSHOT_DIR, ARRAY_FIELDS, snapshot,
)

# m/z arrays may drift by a few ULP when y-ion masses come from suffix sums
# (reordered float accumulation; measured max 4 ULP / 5e-16 relative on a real
# library — 8+ orders of magnitude below ppm-scale matching tolerances).
# b-ion masses are bit-identical.
ULP_FIELDS = {"spectrum_mz", "frag_data"}
ULP_RTOL = 1e-14


def make_edgecases_store():
    return SpectrumLibraryStore.from_tsv(os.path.join(FIXTURE_DIR, "library_edgecases.tsv"))


def _entry(mod_seq, seq, prec_mz, prec_z, frags):
    names = list(frags)
    peaks = np.array([frags[n] for n in names], dtype=np.float64)
    order = np.argsort(peaks[:, 0], kind="stable")
    return {
        'mod_seq': mod_seq, 'seq': seq, 'prec_mz': prec_mz, 'prec_z': prec_z,
        'iRT': 10.0, 'protein_group': 'PG', 'protein_name': 'Prot',
        'genes': 'GENE', 'frags': frags,
        'spectrum': peaks[order],
        'ordered_frags': np.array(names, dtype=object)[order],
    }


def make_tagged_store():
    """Small pre-tagged library, incl. a double-tagged N-terminal K token."""
    b = lambda s, z=1: mass.fast_mass(sequence=s, ion_type='b', charge=z)
    y = lambda s, z=1: mass.fast_mass(sequence=s, ion_type='y', charge=z)
    d = {
        ("P(test_one_channel-0)EPTIDEK(test_one_channel-0)", 2.0): _entry(
            "P(test_one_channel-0)EPTIDEK(test_one_channel-0)", "PEPTIDEK", 464.75, 2.0,
            {'b3_1': [b("PEP"), 1.0], 'b4_1': [b("PEPT"), 0.85],
             'y3_1': [y("DEK"), 0.5], 'y4_1': [y("IDEK"), 0.4]}),
        ("K(test_one_channel-0)(test_one_channel-0)LIONELR", 2.0): _entry(
            "K(test_one_channel-0)(test_one_channel-0)LIONELR", "KLIONELR", 500.3, 2.0,
            {'b2_1': [b("KL"), 0.6], 'y3_1': [y("ELR"), 1.0],
             'y5-H2O_1': [y("ONELR") - 18.01, 0.2]}),
    }
    return SpectrumLibraryStore.from_dict(d)


def make_collision_store():
    """AAK's shuffle/rev decoy is AAK itself -> removed as a target collision."""
    b = lambda s: mass.fast_mass(sequence=s, ion_type='b', charge=1)
    y = lambda s: mass.fast_mass(sequence=s, ion_type='y', charge=1)
    d = {
        ("AAK", 1.0): _entry("AAK", "AAK", 300.2, 1.0,
                             {'b2_1': [b("AA"), 0.7], 'y2_1': [y("AK"), 1.0]}),
        ("PEPTIDEK", 2.0): _entry("PEPTIDEK", "PEPTIDEK", 464.75, 2.0,
                                  {'b3_1': [b("PEP"), 1.0], 'y3_1': [y("DEK"), 0.5]}),
    }
    return SpectrumLibraryStore.from_dict(d)


def one_channel_tag():
    return read_json_to_massTag("tests/MassTags", "test_one_channel.json")


CASES = {
    "decoy_edgecases_rev": (make_edgecases_store, "rev", None),
    "decoy_edgecases_rev_nc": (make_edgecases_store, "rev_nc", None),
    "decoy_edgecases_shuffle": (make_edgecases_store, "shuffle", None),
    "decoy_tagged_shuffle": (make_tagged_store, "shuffle", one_channel_tag),
    "decoy_tagged_rev": (make_tagged_store, "rev", one_channel_tag),
    "decoy_collision_shuffle": (make_collision_store, "shuffle", None),
}


def spectrum_canonical_order(snap):
    """Global re-sort of spectrum arrays by (precursor, frag code).

    Codes are unique within a precursor and identical across both sides, so
    this order is side-independent — unlike sorting by m/z, where a ULP shift
    can swap near-equal peaks between the two implementations."""
    pidx = np.repeat(np.arange(len(snap["spectrum_lengths"])), snap["spectrum_lengths"])
    return np.lexsort((snap["frag_names_data"], pidx))


def assert_decoy_snapshots_match(actual, expected, label=""):
    a_perm, e_perm = spectrum_canonical_order(actual), spectrum_canonical_order(expected)
    for field in ARRAY_FIELDS:
        a, e = np.asarray(actual[field]), np.asarray(expected[field])
        assert a.shape == e.shape, f"{label}{field}: shape {a.shape} != {e.shape}"
        assert a.dtype == e.dtype, f"{label}{field}: dtype {a.dtype} != {e.dtype}"
        if field in ("spectrum_mz", "spectrum_int", "frag_names_data"):
            a, e = a[a_perm], e[e_perm]
        if field in ULP_FIELDS:
            assert np.allclose(a, e, rtol=ULP_RTOL, atol=0.0, equal_nan=True), \
                f"{label}{field}: beyond float-noise tolerance"
        elif a.dtype == object:
            assert a.tolist() == e.tolist(), f"{label}{field}: values differ"
        else:
            assert a.tobytes() == e.tobytes(), f"{label}{field}: bytes differ"
    for scalar in ("key_to_idx", "n_targets", "n_decoys"):
        assert actual[scalar] == expected[scalar], f"{label}{scalar} differs"


class TestDecoyDifferential:
    @pytest.mark.parametrize("case", sorted(CASES))
    def test_matches_snapshot(self, case):
        make_store, rules, tag_factory = CASES[case]
        tag = tag_factory() if tag_factory else None
        combined = create_decoy_lib(make_store(), rules=rules, tag=tag)
        snap = snapshot(combined)
        path = os.path.join(SNAPSHOT_DIR, f"{case}.snapshot.pkl")
        if os.environ.get("JMOD_REGEN_SNAPSHOTS"):
            os.makedirs(SNAPSHOT_DIR, exist_ok=True)
            with open(path, "wb") as f:
                pickle.dump(snap, f)
            pytest.skip(f"regenerated {path}")
        assert os.path.exists(path), f"Missing snapshot {path}; run with JMOD_REGEN_SNAPSHOTS=1"
        with open(path, "rb") as f:
            expected = pickle.load(f)
        assert_decoy_snapshots_match(snap, expected, label=f"{case}: ")

    def test_consumes_input_store(self):
        # from_target_with_decoys consumes its input: large arrays released
        # during assembly so peak memory stays near the combined-store size
        store = make_collision_store()
        create_decoy_lib(store, rules="rev", tag=None)
        for field in ("frag_keys_data", "frag_data", "spectrum_mz",
                      "spectrum_int", "frag_names_data"):
            assert getattr(store, field) is None, field

    def test_collision_removed(self):
        combined = create_decoy_lib(make_collision_store(), rules="shuffle", tag=None)
        # AAK's decoy collides with the AAK target and is discarded
        assert combined.n_targets == 2
        assert combined.n_decoys == 1
        assert combined.mod_seq[2] == "PPEETDIK"


class TestCombinedStoreDtypes:
    """Silent-recast audit on the combined target+decoy store."""

    def test_dtypes(self):
        from tests.models.spec_lib.test_library_snapshots import TestArrayDtypes
        combined = create_decoy_lib(make_edgecases_store(), rules="shuffle", tag=None)
        for field, expected in TestArrayDtypes.EXPECTED_DTYPES.items():
            actual = np.asarray(getattr(combined, field)).dtype
            assert actual == np.dtype(expected), f"{field}: {actual} != {np.dtype(expected)}"
        for mod_pep, charge in combined.key_to_idx:
            assert type(mod_pep) is str and type(charge) is float


class TestShuffleSeedContract:
    """Literal pins: the md5-seeded shuffle must produce exactly these decoys.
    Any drift here silently changes every decoy library."""

    def test_plain(self):
        assert change_seq("PEPTIDEK", "shuffle") == "PPEETDIK"
        assert change_seq("LIONELK", "shuffle") == "ENILOLK"

    def test_modified(self):
        assert change_seq("C(UniMod:4)SQAPVYGR", "shuffle") == "C(UniMod:4)SVYQAGPR"

    def test_tagged(self):
        tag = one_channel_tag()
        assert change_seq("P(test_one_channel-0)EPTIDEK(test_one_channel-0)",
                          "shuffle", tag=tag) == "P(test_one_channel-0)PEETDIK(test_one_channel-0)"

    def test_low_diversity_falls_back_to_reverse(self):
        assert change_seq("AAK", "shuffle") == "AAK"

    def test_rev_variants(self):
        assert change_seq("PEPTIDEK", "rev") == "EDITPEPK"
        assert change_seq("PEPTIDEK", "rev_nc") == "PEDITPEK"
