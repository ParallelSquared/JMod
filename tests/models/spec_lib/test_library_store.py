import pytest
import numpy as np
import tempfile
import csv
import copy
import os

from src.models.spec_lib.library_store import SpectrumLibraryStore, _EntryView
from src.mass_tags import massTag, tag_library


def _make_sample_dict():
    """Create a small dict-of-dicts library for testing."""
    from src.utils.misc_functions import frag_to_peak
    frags1 = {'y5_1': [600.1, 0.8], 'b3_1': [300.5, 0.2]}
    spectrum1, ordered_frags1 = frag_to_peak(frags1, return_frags=True)

    frags2 = {'y3_1': [350.2, 1.0], 'b2_1': [200.1, 0.5], 'y4_1': [500.3, 0.7]}
    spectrum2, ordered_frags2 = frag_to_peak(frags2, return_frags=True)

    return {
        ('ACD', 2.0): {
            'mod_seq': 'ACD', 'seq': 'ACD',
            'prec_mz': 450.2, 'prec_z': 2.0, 'iRT': 32.5,
            'frags': frags1, 'spectrum': spectrum1, 'ordered_frags': ordered_frags1,
            'protein_group': 'PG1', 'protein_name': 'Prot1', 'genes': 'GENE1',
        },
        ('EFGH', 3.0): {
            'mod_seq': 'EFGH', 'seq': 'EFGH',
            'prec_mz': 550.3, 'prec_z': 3.0, 'iRT': None,
            'frags': frags2, 'spectrum': spectrum2, 'ordered_frags': ordered_frags2,
            'IonMob': 1.23, 'UniprotID': 'P12345',
        },
    }


class TestFromDict:
    def test_basic_construction(self):
        d = _make_sample_dict()
        store = SpectrumLibraryStore.from_dict(d)
        assert len(store) == 2
        assert ('ACD', 2.0) in store
        assert ('EFGH', 3.0) in store
        assert ('ZZZ', 1.0) not in store

    def test_scalar_fields(self):
        d = _make_sample_dict()
        store = SpectrumLibraryStore.from_dict(d)
        entry = store[('ACD', 2.0)]
        assert entry['mod_seq'] == 'ACD'
        assert entry['seq'] == 'ACD'
        assert entry['prec_mz'] == 450.2
        assert entry['prec_z'] == 2.0
        assert entry['iRT'] == 32.5
        assert entry['protein_group'] == 'PG1'
        assert entry['protein_name'] == 'Prot1'
        assert entry['genes'] == 'GENE1'

    def test_none_iRT(self):
        d = _make_sample_dict()
        store = SpectrumLibraryStore.from_dict(d)
        entry = store[('EFGH', 3.0)]
        assert entry['iRT'] is None

    def test_ion_mobility(self):
        d = _make_sample_dict()
        store = SpectrumLibraryStore.from_dict(d)
        entry1 = store[('ACD', 2.0)]
        assert entry1.get('IonMob') is None
        assert 'IonMob' not in entry1

        entry2 = store[('EFGH', 3.0)]
        assert entry2['IonMob'] == pytest.approx(1.23)
        assert 'IonMob' in entry2

    def test_spectrum(self):
        d = _make_sample_dict()
        store = SpectrumLibraryStore.from_dict(d)
        entry = store[('ACD', 2.0)]
        spec = entry['spectrum']
        assert spec.shape == (2, 2)
        # Should be sorted by m/z
        assert spec[0, 0] < spec[1, 0]

    def test_ordered_frags(self):
        d = _make_sample_dict()
        store = SpectrumLibraryStore.from_dict(d)
        entry = store[('ACD', 2.0)]
        of = entry['ordered_frags']
        assert len(of) == 2
        # Should match spectrum sort order
        spec = entry['spectrum']
        frags = entry['frags']
        for i, name in enumerate(of):
            assert frags[name][0] == pytest.approx(spec[i, 0])

    def test_frags_reconstruction(self):
        d = _make_sample_dict()
        store = SpectrumLibraryStore.from_dict(d)
        entry = store[('ACD', 2.0)]
        frags = entry['frags']
        assert isinstance(frags, dict)
        assert 'y5_1' in frags
        assert 'b3_1' in frags
        assert frags['y5_1'] == [pytest.approx(600.1), pytest.approx(0.8)]


class TestDictInterface:
    def test_iter(self):
        d = _make_sample_dict()
        store = SpectrumLibraryStore.from_dict(d)
        keys = list(store)
        assert len(keys) == 2
        assert ('ACD', 2.0) in keys

    def test_items(self):
        d = _make_sample_dict()
        store = SpectrumLibraryStore.from_dict(d)
        items = list(store.items())
        assert len(items) == 2
        for key, entry in items:
            assert isinstance(entry, _EntryView)

    def test_values(self):
        d = _make_sample_dict()
        store = SpectrumLibraryStore.from_dict(d)
        vals = list(store.values())
        assert len(vals) == 2

    def test_keys(self):
        d = _make_sample_dict()
        store = SpectrumLibraryStore.from_dict(d)
        assert set(store.keys()) == {('ACD', 2.0), ('EFGH', 3.0)}

    def test_get(self):
        d = _make_sample_dict()
        store = SpectrumLibraryStore.from_dict(d)
        assert store.get(('ACD', 2.0)) is not None
        assert store.get(('ZZZ', 1.0)) is None
        assert store.get(('ZZZ', 1.0), 'default') == 'default'


class TestEntryView:
    def test_dict_conversion(self):
        """dict(_EntryView) should produce a plain dict with all fields."""
        d = _make_sample_dict()
        store = SpectrumLibraryStore.from_dict(d)
        entry = store[('ACD', 2.0)]
        plain = dict(entry)
        assert isinstance(plain, dict)
        assert plain['mod_seq'] == 'ACD'
        assert 'spectrum' in plain
        assert 'frags' in plain

    def test_contains(self):
        d = _make_sample_dict()
        store = SpectrumLibraryStore.from_dict(d)
        entry = store[('ACD', 2.0)]
        assert 'mod_seq' in entry
        assert 'spectrum' in entry
        assert 'frags' in entry
        assert 'IonMob' not in entry  # NaN for this entry

    def test_keys_method(self):
        d = _make_sample_dict()
        store = SpectrumLibraryStore.from_dict(d)
        entry = store[('ACD', 2.0)]
        k = entry.keys()
        assert 'mod_seq' in k
        assert 'spectrum' in k

    def test_setitem_iRT(self):
        d = _make_sample_dict()
        store = SpectrumLibraryStore.from_dict(d)
        store[('ACD', 2.0)]['iRT'] = 99.9
        assert store[('ACD', 2.0)]['iRT'] == pytest.approx(99.9)

    def test_setitem_parent_key(self):
        d = _make_sample_dict()
        store = SpectrumLibraryStore.from_dict(d)
        key = ('ACD', 2.0)
        store[key]['parent_key'] = key
        assert store[key]['parent_key'] == key

    def test_setitem_top_n(self):
        d = _make_sample_dict()
        store = SpectrumLibraryStore.from_dict(d)
        key = ('ACD', 2.0)
        entry = store[key]
        spec = entry['spectrum']
        top_n = np.argsort(-spec[:, 1])[:2]
        entry['top_n'] = top_n
        result = store[key]['top_n']
        np.testing.assert_array_equal(result, top_n)

    def test_setitem_seq(self):
        d = _make_sample_dict()
        store = SpectrumLibraryStore.from_dict(d)
        store[('ACD', 2.0)]['seq'] = 'DCA'
        assert store[('ACD', 2.0)]['seq'] == 'DCA'

    def test_setitem_frags(self):
        d = _make_sample_dict()
        store = SpectrumLibraryStore.from_dict(d)
        new_frags = {'y3_1': [400.0, 1.0], 'b2_1': [250.0, 0.5]}
        store[('ACD', 2.0)]['frags'] = new_frags
        result = store[('ACD', 2.0)]['frags']
        assert 'y3_1' in result
        assert result['y3_1'][0] == pytest.approx(400.0)

    def test_deepcopy(self):
        d = _make_sample_dict()
        store = SpectrumLibraryStore.from_dict(d)
        entry = store[('ACD', 2.0)]
        plain = copy.deepcopy(entry)
        assert isinstance(plain, dict)
        assert plain['mod_seq'] == 'ACD'
        # Mutation of the copy shouldn't affect the store
        plain['iRT'] = 999.0
        assert store[('ACD', 2.0)]['iRT'] == pytest.approx(32.5)


class TestStoreMutation:
    def test_setitem_new_key(self):
        d = _make_sample_dict()
        store = SpectrumLibraryStore.from_dict(d)
        new_entry = {
            'mod_seq': 'XYZ', 'seq': 'XYZ',
            'prec_mz': 700.0, 'prec_z': 1.0, 'iRT': 10.0,
            'frags': {'y2_1': [250.0, 1.0]},
        }
        store[('XYZ', 1.0)] = new_entry
        assert len(store) == 3
        assert store[('XYZ', 1.0)]['mod_seq'] == 'XYZ'

    def test_delete_key(self):
        d = _make_sample_dict()
        store = SpectrumLibraryStore.from_dict(d)
        del store[('ACD', 2.0)]
        assert len(store) == 1
        assert ('ACD', 2.0) not in store


class TestSerialization:
    def test_save_load_roundtrip(self, tmp_path):
        d = _make_sample_dict()
        store = SpectrumLibraryStore.from_dict(d)
        # Set some top_n
        for key in store:
            entry = store[key]
            spec = entry['spectrum']
            entry['top_n'] = np.argsort(-spec[:, 1])
        # Set parent_key
        for key in store:
            store[key]['parent_key'] = key

        path = str(tmp_path / "test_store.npz")
        store.save(path)

        loaded = SpectrumLibraryStore.load(path)
        assert len(loaded) == len(store)
        for key in store:
            orig = store[key]
            load = loaded[key]
            assert orig['mod_seq'] == load['mod_seq']
            assert orig['prec_mz'] == pytest.approx(load['prec_mz'])
            np.testing.assert_array_almost_equal(
                orig['spectrum'], load['spectrum']
            )


class TestFromTSV:
    def test_from_tsv_minimal(self):
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".tsv") as tmp:
            writer = csv.writer(tmp, delimiter="\t")
            writer.writerow([
                "ModifiedPeptide", "PrecursorCharge", "PrecursorMz",
                "StrippedPeptide", "FragmentType", "FragmentNumber",
                "FragmentCharge", "FragmentMz", "RelativeIntensity",
                "RT"
            ])
            writer.writerow(["_ACD_", 2, 450.2, "ACD", "y", 5, 1, 600.1, 0.8, 32.5])
            writer.writerow(["_ACD_", 2, 450.2, "ACD", "b", 3, 1, 300.5, 0.2, 32.5])
            filename = tmp.name

        try:
            store = SpectrumLibraryStore.from_tsv(filename)
            assert len(store) == 1
            key = ('ACD', 2.0)
            assert key in store
            entry = store[key]
            assert entry['mod_seq'] == 'ACD'
            assert entry['prec_mz'] == pytest.approx(450.2)
            assert entry['iRT'] == pytest.approx(32.5)
            frags = entry['frags']
            assert 'y5_1' in frags
            assert 'b3_1' in frags
        finally:
            os.unlink(filename)


class TestDeepCopyStore:
    def test_deepcopy_store(self):
        d = _make_sample_dict()
        store = SpectrumLibraryStore.from_dict(d)
        store2 = copy.deepcopy(store)
        # Mutation of copy shouldn't affect original
        store2[('ACD', 2.0)]['iRT'] = 999.0
        assert store[('ACD', 2.0)]['iRT'] == pytest.approx(32.5)
        assert store2[('ACD', 2.0)]['iRT'] == pytest.approx(999.0)

    def test_shallow_copy(self):
        d = _make_sample_dict()
        store = SpectrumLibraryStore.from_dict(d)
        store2 = store.shallow_copy()
        # iRT should be independent
        store2[('ACD', 2.0)]['iRT'] = 999.0
        assert store[('ACD', 2.0)]['iRT'] == pytest.approx(32.5)
        # But spectrum data is shared
        assert store.spectrum_data is store2.spectrum_data
