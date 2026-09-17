"""Snapshot tests for spectral library parsing.

Each fixture library is parsed and every array on the resulting store is
compared byte-for-byte against a committed reference snapshot
(tests/fixtures/snapshots/). Any parser refactor must leave these
bit-identical. To regenerate after a deliberate behavior change:

    JMOD_REGEN_SNAPSHOTS=1 .venv/bin/pytest tests/models/spec_lib/test_library_snapshots.py
"""
import os
import pickle

import numpy as np
import pytest

from src.models.spec_lib.library_store import (
    SpectrumLibraryStore,
    StaleStoreCacheError,
    STORE_VERSION,
)

FIXTURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "fixtures")
SNAPSHOT_DIR = os.path.join(FIXTURE_DIR, "snapshots")

ARRAY_FIELDS = [
    "mod_seq", "seq", "prec_mz", "prec_z", "iRT", "ion_mob",
    "protein_group", "protein_name", "genes", "uniprot_id",
    "spectrum_mz", "spectrum_int", "spectrum_offsets", "spectrum_lengths",
    "frag_names_data", "spectrum_perm", "frag_mz", "frag_int", "frag_keys_data",
    "frag_offsets", "frag_lengths",
    "top_n_data", "top_n_offsets", "top_n_lengths",
    "parent_idx", "is_decoy",
]

# Directory of Phase-3 (pre-single-copy) finalized snapshots: the reference
# that finalize_spectra() must reproduce bit-for-bit
FINALIZED_REF_DIR = os.path.join(SNAPSHOT_DIR, "finalized_phase3")

FIXTURES = [
    "library_edgecases",
    "library_diann_aliases",
    "library_proteinid_only",
]
FORMATS = ["tsv", "parquet"]


def parse_fixture(name, fmt):
    path = os.path.join(FIXTURE_DIR, f"{name}.{fmt}")
    if fmt == "tsv":
        return SpectrumLibraryStore.from_tsv(path)
    return SpectrumLibraryStore.from_parquet(path)


def snapshot(store):
    snap = {field: getattr(store, field) for field in ARRAY_FIELDS}
    snap["key_to_idx"] = dict(store.key_to_idx)
    snap["n_targets"] = int(store.n_targets)
    snap["n_decoys"] = int(store.n_decoys)
    return snap


def assert_snapshots_identical(actual, expected, label=""):
    for field in ARRAY_FIELDS:
        av, ev = actual.get(field), expected.get(field)
        if av is None or ev is None:
            assert av is None and ev is None, f"{label}{field}: state mismatch"
            continue
        a, e = np.asarray(av), np.asarray(ev)
        assert a.shape == e.shape, f"{label}{field}: shape {a.shape} != {e.shape}"
        assert a.dtype == e.dtype, f"{label}{field}: dtype {a.dtype} != {e.dtype}"
        if a.dtype == object:
            assert a.tolist() == e.tolist(), f"{label}{field}: values differ"
        else:
            assert a.tobytes() == e.tobytes(), f"{label}{field}: bytes differ"
    assert actual["key_to_idx"] == expected["key_to_idx"], f"{label}key_to_idx differs"
    assert actual["n_targets"] == expected["n_targets"], f"{label}n_targets differs"
    assert actual["n_decoys"] == expected["n_decoys"], f"{label}n_decoys differs"


class TestSnapshots:
    @pytest.mark.parametrize("fmt", FORMATS)
    @pytest.mark.parametrize("name", FIXTURES)
    def test_matches_snapshot(self, name, fmt):
        snap = snapshot(parse_fixture(name, fmt))
        snapshot_path = os.path.join(SNAPSHOT_DIR, f"{name}.{fmt}.snapshot.pkl")
        if os.environ.get("JMOD_REGEN_SNAPSHOTS"):
            os.makedirs(SNAPSHOT_DIR, exist_ok=True)
            with open(snapshot_path, "wb") as f:
                pickle.dump(snap, f)
            pytest.skip(f"regenerated {snapshot_path}")
        assert os.path.exists(snapshot_path), (
            f"Missing snapshot {snapshot_path}; run with JMOD_REGEN_SNAPSHOTS=1"
        )
        with open(snapshot_path, "rb") as f:
            expected = pickle.load(f)
        assert_snapshots_identical(snap, expected, label=f"{name}.{fmt}: ")

    @pytest.mark.parametrize("name", FIXTURES)
    def test_tsv_and_parquet_agree(self, name):
        tsv_snap = snapshot(parse_fixture(name, "tsv"))
        pq_snap = snapshot(parse_fixture(name, "parquet"))
        assert_snapshots_identical(pq_snap, tsv_snap, label=f"{name} tsv vs parquet: ")


class TestParsingSemantics:
    """Explicit assertions documenting the quirks the snapshots freeze."""

    @pytest.fixture(scope="class", params=FORMATS)
    def edgecases(self, request):
        return parse_fixture("library_edgecases", request.param)

    def test_decoys_and_invalid_residues_removed(self, edgecases):
        assert sorted(edgecases.mod_seq) == [
            "C(UniMod:4)(tag)SQAPVYGR", "LIONELK", "PEPTIDEK", "SEVENPEPK",
        ]

    def test_underscores_stripped(self, edgecases):
        assert ("LIONELK", 1.0) in edgecases.key_to_idx

    def test_ion_mobility_zero_and_empty_are_nan(self, edgecases):
        idx = edgecases.key_to_idx
        assert np.isnan(edgecases.ion_mob[idx[("C(UniMod:4)(tag)SQAPVYGR", 2.0)]])
        assert np.isnan(edgecases.ion_mob[idx[("LIONELK", 1.0)]])
        assert np.isnan(edgecases.ion_mob[idx[("SEVENPEPK", 2.0)]])
        assert edgecases.ion_mob[idx[("PEPTIDEK", 2.0)]] == 0.95

    def test_empty_genes_becomes_quoted_empty(self, edgecases):
        idx = edgecases.key_to_idx[("C(UniMod:4)(tag)SQAPVYGR", 2.0)]
        assert edgecases.genes[idx] == '""'

    def test_duplicate_fragment_key_last_wins(self, edgecases):
        i = edgecases.key_to_idx[("PEPTIDEK", 2.0)]
        start = edgecases.frag_offsets[i]
        length = edgecases.frag_lengths[i]
        frag_mzs = edgecases.frag_mz[start:start + length]
        # y4_1 appears twice in the fixture (502.29 then 502.30); five unique keys survive
        assert length == 5
        assert np.float32(502.30) in frag_mzs and np.float32(502.29) not in frag_mzs

    def test_equal_mz_tie_is_stable(self, edgecases):
        i = edgecases.key_to_idx[("PEPTIDEK", 2.0)]
        spec = edgecases.get_spectrum(i)  # perm-gathered pre-finalize
        mzs, ints = spec[:, 0], spec[:, 1]
        tied = np.where(mzs == np.float32(227.10))[0]
        # b2_1 (0.3) was inserted before y2_1 (0.4); stable sort keeps that order
        assert list(ints[tied]) == [np.float32(0.3), np.float32(0.4)]

    @pytest.mark.parametrize("fmt", FORMATS)
    def test_proteinid_fills_name_and_uniprot(self, fmt):
        store = parse_fixture("library_proteinid_only", fmt)
        assert store.protein_name[0] == "P00X"
        assert store.uniprot_id[0] == "P00X"

    def test_missing_rt_column_raises(self):
        with pytest.raises(ValueError, match="retention time"):
            parse_fixture("library_missing_rt", "tsv")


class TestArrayDtypes:
    """Guard against silent recasts: every store array must land in its
    documented dtype, with no implicit up/down-casts introduced by the
    columnar parsing path."""

    EXPECTED_DTYPES = {
        "mod_seq": np.object_, "seq": np.object_,
        "protein_group": np.object_, "protein_name": np.object_,
        "genes": np.object_, "uniprot_id": np.object_,
        "prec_mz": np.float64, "prec_z": np.float64,
        "iRT": np.float64, "ion_mob": np.float64,
        # fragment-level values are float32 by design (~30 ppb quantization,
        # far below ppm tolerances); precursor-level scalars stay float64
        "frag_mz": np.float32, "frag_int": np.float32,
        "frag_offsets": np.int64,
        "top_n_offsets": np.int64, "parent_idx": np.int64,
        "frag_lengths": np.int32,
        "top_n_lengths": np.int32,
        "frag_keys_data": np.int32,
        "top_n_data": np.int32,
        "is_decoy": np.bool_,
        "spectrum_perm": np.uint16,
    }
    # Only present once finalized (parser output is pre-finalize)
    FINALIZED_DTYPES = {
        "spectrum_mz": np.float32, "spectrum_int": np.float32,
        "spectrum_offsets": np.int64, "spectrum_lengths": np.int32,
        "frag_names_data": np.int32,
    }

    @pytest.mark.parametrize("fmt", FORMATS)
    @pytest.mark.parametrize("name", FIXTURES)
    def test_store_array_dtypes(self, name, fmt):
        store = parse_fixture(name, fmt)
        assert not store.is_finalized
        finalized = parse_fixture(name, fmt).finalize_spectra()
        for field, expected in self.FINALIZED_DTYPES.items():
            actual = np.asarray(getattr(finalized, field)).dtype
            assert actual == np.dtype(expected), (
                f"{name}.{fmt}: {field} is {actual}, expected {np.dtype(expected)}")
        for field, expected in self.EXPECTED_DTYPES.items():
            actual = np.asarray(getattr(store, field)).dtype
            assert actual == np.dtype(expected), (
                f"{name}.{fmt}: {field} is {actual}, expected {np.dtype(expected)}"
            )
        # key_to_idx keys must stay (str, float) — not numpy scalars
        for mod_pep, charge in store.key_to_idx:
            assert type(mod_pep) is str and type(charge) is float
            break


class TestFinalizeBitIdentity:
    """finalize_spectra() must reproduce the Phase-3 double-stored spectrum
    arrays bit-for-bit from frag arrays + spectrum_perm."""

    @pytest.mark.parametrize("fmt", FORMATS)
    @pytest.mark.parametrize("name", FIXTURES)
    def test_finalize_matches_phase3(self, name, fmt):
        ref_path = os.path.join(FINALIZED_REF_DIR, f"{name}.{fmt}.snapshot.pkl")
        with open(ref_path, "rb") as f:
            ref = pickle.load(f)
        store = parse_fixture(name, fmt)
        assert store.spectrum_perm is not None
        store.finalize_spectra()
        assert store.spectrum_perm is None
        for field in ("spectrum_mz", "spectrum_int", "frag_names_data",
                      "spectrum_offsets", "spectrum_lengths"):
            a, e = np.asarray(getattr(store, field)), np.asarray(ref[field])
            assert a.dtype == e.dtype and a.tobytes() == e.tobytes(), field
        # idempotent
        store.finalize_spectra()


class TestCacheVersioning:
    def test_round_trip_current_version(self, tmp_path):
        store = parse_fixture("library_edgecases", "tsv")
        path = str(tmp_path / "lib_store.npz")
        store.save(path)
        loaded = SpectrumLibraryStore.load(path)
        assert_snapshots_identical(snapshot(loaded), snapshot(store))

    def test_unstamped_cache_rejected(self, tmp_path):
        store = parse_fixture("library_edgecases", "tsv")
        path = str(tmp_path / "lib_store.npz")
        store.save(path)
        data = dict(np.load(path, allow_pickle=True))
        del data["store_version"]
        np.savez(path, **data)
        with pytest.raises(StaleStoreCacheError):
            SpectrumLibraryStore.load(path)

    def test_old_version_cache_rejected(self, tmp_path):
        store = parse_fixture("library_edgecases", "tsv")
        path = str(tmp_path / "lib_store.npz")
        store.save(path)
        data = dict(np.load(path, allow_pickle=True))
        data["store_version"] = np.array(STORE_VERSION - 1)
        np.savez(path, **data)
        with pytest.raises(StaleStoreCacheError):
            SpectrumLibraryStore.load(path)
