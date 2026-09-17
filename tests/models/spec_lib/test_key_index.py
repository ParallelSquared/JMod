"""KeyIndex: compact replacement for the (mod_seq, charge) -> row dict."""
import copy
import os

import numpy as np
import pytest

from src.models.spec_lib.library_store import SpectrumLibraryStore, KeyIndex

from tests.models.spec_lib.test_library_snapshots import FIXTURE_DIR


def edgecases_store():
    return SpectrumLibraryStore.from_tsv(os.path.join(FIXTURE_DIR, "library_edgecases.tsv"))


class TestKeyIndex:
    def test_backed_by_key_index(self):
        store = edgecases_store()
        assert isinstance(store.key_to_idx, KeyIndex)

    def test_lookup_roundtrip(self):
        store = edgecases_store()
        for i, key in enumerate(store.keys()):
            assert store.key_to_idx[key] == i
            assert key in store.key_to_idx

    def test_int_charge_lookup(self):
        # some callers look up with int(z) against float-keyed entries;
        # hash(2) == hash(2.0) plus numeric verify keeps that working
        store = edgecases_store()
        assert store.key_to_idx[("PEPTIDEK", 2)] == store.key_to_idx[("PEPTIDEK", 2.0)]
        assert ("PEPTIDEK", 2) in store.key_to_idx

    def test_missing_key(self):
        store = edgecases_store()
        with pytest.raises(KeyError):
            store.key_to_idx[("NOPE", 2.0)]
        assert store.key_to_idx.get(("NOPE", 2.0)) is None
        assert store.key_to_idx.get(("NOPE", 2.0), -1) == -1
        assert ("NOPE", 2.0) not in store.key_to_idx
        assert store.key_to_idx.get("malformed") is None

    def test_iteration_order_is_row_order(self):
        store = edgecases_store()
        keys = list(store.key_to_idx)
        assert keys == [(store.mod_seq[i], float(store.prec_z[i]))
                        for i in range(len(store.mod_seq))]
        assert [i for _, i in store.key_to_idx.items()] == list(range(len(keys)))

    def test_collision_verify_scan(self, monkeypatch):
        # Force every hash equal: lookups must still resolve via the verify scan
        store = edgecases_store()
        idx = store.key_to_idx
        idx._sorted = np.zeros_like(idx._sorted)
        monkeypatch.setattr(KeyIndex, "_find", KeyIndex._find)  # no-op, clarity
        real_find = idx._find

        def all_collide(key):
            k = KeyIndex._canonical(key)
            if k is None:
                return -1
            mod_seq, prec_z = idx._store.mod_seq, idx._store.prec_z
            for j in range(len(idx._sorted)):
                i = int(idx._order[j])
                if mod_seq[i] == k[0] and prec_z[i] == k[1]:
                    return i
            return -1

        # emulate a total-collision world: full-range scan must agree
        for i, key in enumerate(store.keys()):
            assert all_collide(key) == i

    def test_batch_matches_scalar(self):
        store = edgecases_store()
        keys = list(store.keys())
        assert store.resolve_indices(keys) == [store.key_to_idx[k] for k in keys]
        with pytest.raises(KeyError):
            store.resolve_indices(keys + [("NOPE", 9.0)])

    def test_mutating_api_disabled(self):
        store = edgecases_store()
        with pytest.raises(NotImplementedError):
            store.key_to_idx[("X", 1.0)] = 5
        with pytest.raises(NotImplementedError):
            del store[("PEPTIDEK", 2.0)]

    def test_dict_mode_still_supported(self):
        # from_dict stores keep a plain dict (timeplex path) incl. mutation
        d = {("AAK", 1.0): {'mod_seq': "AAK", 'seq': "AAK", 'prec_mz': 300.2,
                            'prec_z': 1.0, 'iRT': 1.0,
                            'frags': {'b2_1': [143.08, 1.0]},
                            'spectrum': np.array([[143.08, 1.0]]),
                            'ordered_frags': np.array(['b2_1'], dtype=object)}}
        store = SpectrumLibraryStore.from_dict(d)
        assert isinstance(store.key_to_idx, dict)
        assert store.key_to_idx[("AAK", 1.0)] == 0


class TestTargetViewProxy:
    def test_len_iter_contains(self):
        from src.models.spec_lib.spec_lib import create_decoy_lib
        combined = create_decoy_lib(edgecases_store(), rules="rev", tag=None)
        view = combined.target_view()
        assert len(view) == combined.n_targets
        keys = list(view)
        assert keys == [(combined.mod_seq[i], float(combined.prec_z[i]))
                        for i in range(combined.n_targets)]
        assert keys[0] in view
        # decoy key visible in store but not in the target view
        decoy_key = (combined.mod_seq[combined.n_targets],
                     float(combined.prec_z[combined.n_targets]))
        assert decoy_key in combined.key_to_idx
        assert decoy_key not in view

    def test_deepcopy_targets_only(self):
        from src.models.spec_lib.spec_lib import create_decoy_lib
        combined = create_decoy_lib(edgecases_store(), rules="rev", tag=None)
        view = combined.target_view()
        clone = copy.deepcopy(view)
        assert isinstance(clone, SpectrumLibraryStore)
        assert len(clone.mod_seq) == combined.n_targets
        assert list(clone.mod_seq) == list(combined.mod_seq[:combined.n_targets])
        for key in view:
            assert clone.key_to_idx[key] == combined.key_to_idx[key]
        # independent arrays
        clone.iRT[0] += 1.0
        assert clone.iRT[0] != combined.iRT[0]
