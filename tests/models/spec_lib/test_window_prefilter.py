"""Window prefilter: unrecoverable precursors dropped before decoy/tag work."""
import os
import types

import numpy as np
import pytest

from src.models.spec_lib.library_store import SpectrumLibraryStore
from src.models.spec_lib.spec_lib import window_recoverable_mask
from src.mass_tags import read_json_to_massTag

from tests.models.spec_lib.test_library_snapshots import FIXTURE_DIR


def edgecases_store():
    return SpectrumLibraryStore.from_tsv(os.path.join(FIXTURE_DIR, "library_edgecases.tsv"))


def scans(*windows):
    return [types.SimpleNamespace(ms1window=np.array(w, dtype=float)) for w in windows]


class TestMask:
    def test_untagged_membership(self):
        store = edgecases_store()
        # windows covering only PEPTIDEK (464.75) and SEVENPEPK (520.77)
        mask = window_recoverable_mask(store, scans([460, 470], [515, 525]))
        by_key = dict(zip(store.keys(), mask))
        assert by_key[("PEPTIDEK", 2.0)]
        assert by_key[("SEVENPEPK", 2.0)]
        assert not by_key[("LIONELK", 1.0)]         # 830.51
        assert not by_key[("C(UniMod:4)(tag)SQAPVYGR", 2.0)]  # 540.26

    def test_margin(self):
        store = edgecases_store()
        # 464.75 sits ~32 ppm above 464.735; default 50 ppm margin recovers it
        mask = window_recoverable_mask(store, scans([460.0, 464.735]))
        by_key = dict(zip(store.keys(), mask))
        assert by_key[("PEPTIDEK", 2.0)]
        mask_tight = window_recoverable_mask(store, scans([460.0, 464.735]), margin_ppm=1.0)
        assert not dict(zip(store.keys(), mask_tight))[("PEPTIDEK", 2.0)]

    def test_tagged_any_channel_recovers(self):
        # test_one_channel tag: K + N-term sites. PEPTIDEK z=2 has 2 sites;
        # channel shift moves it into a window that misses it untagged.
        store = edgecases_store()
        tag = read_json_to_massTag("tests/MassTags", "test_one_channel.json")
        shift = 2 * tag.channel_masses[0] / 2.0
        win = [464.75 + shift - 0.5, 464.75 + shift + 0.5]
        untagged = window_recoverable_mask(store, scans(win))
        tagged = window_recoverable_mask(store, scans(win), tag=tag)
        by_key_u = dict(zip(store.keys(), untagged))
        by_key_t = dict(zip(store.keys(), tagged))
        assert not by_key_u[("PEPTIDEK", 2.0)]
        assert by_key_t[("PEPTIDEK", 2.0)]


class TestSubsetEntries:
    def test_subset_matches_per_entry_data(self):
        store = edgecases_store()
        keys = list(store.keys())
        mask = np.array([k[0] != "LIONELK" for k in keys])
        sub = store.subset_entries(mask)
        kept = [k for k, m in zip(keys, mask) if m]
        assert list(sub.keys()) == kept
        for k in kept:
            a = sub[k]
            b = store[k]
            assert a["frags"] == b["frags"]
            assert np.array_equal(sub.get_spectrum(sub.key_to_idx[k]),
                                  store.get_spectrum(store.key_to_idx[k]))
            assert a["prec_mz"] == b["prec_mz"]
            assert a["genes"] == b["genes"]

    def test_all_true_is_identity(self):
        store = edgecases_store()
        assert store.subset_entries(np.ones(len(store.mod_seq), dtype=bool)) is store

    def test_finalize_after_subset(self):
        store = edgecases_store()
        mask = np.ones(len(store.mod_seq), dtype=bool)
        mask[0] = False
        sub = store.subset_entries(mask)
        sub.finalize_spectra()
        i = sub.key_to_idx[("SEVENPEPK", 2.0)]
        spec = sub.get_spectrum(i)
        assert np.all(np.diff(spec[:, 0]) >= 0)
