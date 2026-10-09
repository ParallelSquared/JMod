import pytest
import numpy as np
import tempfile
import csv
import copy
import os

from src.utils.errors import JModError
from src.models.spec_lib.library_store import SpectrumLibraryStore, _EntryView, _TargetView, KeyIndex
from src.models.spec_lib import spec_lib
from src.iso_functions import iso_library
from src.mass_tags import massTag, tag_library

from tests.models.spec_lib.test_library_snapshots import FIXTURE_DIR


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

    def test_setitem_parent_idx(self):
        d = _make_sample_dict()
        store = SpectrumLibraryStore.from_dict(d)
        key = ('ACD', 2.0)
        store[key]['parent_idx'] = 0
        assert store[key]['parent_idx'] == 0

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


def _peptide_store(*mod_seqs):
    """A store of charge-2 precursors, each with b3 and y3 at their true m/z."""
    from pyteomics import mass
    from src.utils.misc_functions import frag_to_peak
    from src.utils.parse_peptides import parse_peptide
    entries = {}
    for mod_seq in mod_seqs:
        seq = "".join(token[0] for token in parse_peptide(mod_seq))
        frags = {'b3_1': [mass.fast_mass(seq[:3], ion_type='b', charge=1), 1.0],
                 'y3_1': [mass.fast_mass(seq[-3:], ion_type='y', charge=1), 0.5]}
        spectrum, ordered_frags = frag_to_peak(frags, return_frags=True)
        entries[(mod_seq, 2.0)] = {
            'mod_seq': mod_seq, 'seq': seq, 'prec_mz': mass.fast_mass(seq, charge=2),
            'prec_z': 2.0, 'iRT': 10.0, 'frags': frags,
            'spectrum': spectrum, 'ordered_frags': ordered_frags}
    return SpectrumLibraryStore.from_dict(entries)


class TestEditMods:
    from src.models.spec_lib.spec_lib import ModEdits, ModSpec
    label = ModSpec("Label", 8.0, "nK")

    def test_added_mod_shifts_the_precursor_and_the_fragments_that_hold_it(self):
        store = _peptide_store("PEPTIDEK")
        entry = store[("PEPTIDEK", 2.0)]   # a view: read the values before the edit
        prec_mz, b3, y3 = entry['prec_mz'], entry['frags']['b3_1'][0], entry['frags']['y3_1'][0]
        store = store.edit_mods(self.ModEdits(add=(self.label,)))

        after = store[("(Label)PEPTIDEK(Label)", 2.0)]
        assert after['prec_mz'] == pytest.approx(prec_mz + 2 * 8.0 / 2)
        assert after['frags']['b3_1'][0] == pytest.approx(b3 + 8.0, abs=1e-3)   # holds the N-terminus
        assert after['frags']['y3_1'][0] == pytest.approx(y3 + 8.0, abs=1e-3)   # holds the K

    def test_adding_a_mod_already_there_changes_nothing(self):
        once = _peptide_store("PEPTIDEK").edit_mods(self.ModEdits(add=(self.label,)))
        prec_mz = once.prec_mz.copy()
        twice = once.edit_mods(self.ModEdits(add=(self.label,)))
        assert list(twice.mod_seq) == ["(Label)PEPTIDEK(Label)"]
        assert np.array_equal(twice.prec_mz, prec_mz)

    def test_stripping_undoes_adding(self):
        original = _peptide_store("PEPTIDEK")
        prec_mz, frag_mz = original.prec_mz.copy(), original.frag_mz.copy()
        store = original.edit_mods(self.ModEdits(add=(self.label,)))
        store = store.edit_mods(self.ModEdits(strip=(self.label,)))
        assert list(store.mod_seq) == ["PEPTIDEK"]
        assert store.prec_mz == pytest.approx(prec_mz)
        assert store.frag_mz == pytest.approx(frag_mz, abs=1e-3)

    def test_stripping_and_adding_the_same_mod_are_counted_apart(self, app_log):
        store = _peptide_store("PEPTIDEK")
        store.edit_mods(self.ModEdits(strip=(self.label,), add=(self.label,)))
        messages = [r.getMessage() for r in app_log]
        assert "Stripped Label from 0 sites" in messages
        assert "Added Label to 2 sites" in messages

    def test_sites_with_another_mod_are_skipped_with_a_warning(self, app_log):
        store = _peptide_store("(UniMod:1)PEPTIDEK", "AAAAK")
        store = store.edit_mods(self.ModEdits(add=(self.label,)))
        assert list(store.mod_seq) == ["(UniMod:1)PEPTIDEK(Label)", "(Label)AAAAK(Label)"]
        warnings = [r.getMessage() for r in app_log if r.levelname == "WARNING"]
        assert len(warnings) == 1
        assert "skipped 1 sites on 1 precursors" in warnings[0] and "UniMod:1 (n)" in warnings[0]
        assert "--strip_mod UniMod:1,n" in warnings[0] and "Label,8.0,nK,stack" in warnings[0]

    def test_stack_adds_alongside_another_mod(self, app_log):
        store = _peptide_store("(UniMod:1)PEPTIDEK")
        store = store.edit_mods(self.ModEdits(add=(self.ModSpec("Label", 8.0, "nK", stack=True),)))
        assert list(store.mod_seq) == ["(UniMod:1)(Label)PEPTIDEK(Label)"]
        assert not [r for r in app_log if r.levelname == "WARNING"]

    def test_strip_by_name_removes_it_from_every_site(self):
        from src.models.spec_lib.spec_lib import parse_mod_spec
        spec = parse_mod_spec("Label,8.0", "--strip_mod")
        store = _peptide_store("PEPTIDEK").edit_mods(self.ModEdits(add=(self.label,)))
        store = store.edit_mods(self.ModEdits(strip=(spec,)))
        assert list(store.mod_seq) == ["PEPTIDEK"]

    def test_entries_made_identical_are_reduced_to_the_first(self):
        store = _peptide_store("PEPM(UniMod:35)K", "PEPMK")
        store = store.edit_mods(self.ModEdits(strip=(self.ModSpec("UniMod:35", 15.994915, "M"),)))
        assert list(store.mod_seq) == ["PEPMK"]
        assert len(store) == 1

    def test_a_parsed_library_is_edited(self):
        store = _edgecases_store()
        i = store.key_to_idx[("(tag)C(UniMod:4)SQAPVYGR", 2.0)]
        prec_mz = store.prec_mz[i]
        store = store.edit_mods(self.ModEdits(strip=(self.ModSpec("UniMod:4", 57.021464, "C"),)))
        i = store.key_to_idx[("(tag)CSQAPVYGR", 2.0)]
        assert store.prec_mz[i] == pytest.approx(prec_mz - 57.021464 / 2)

    def test_spectra_stay_sorted_by_mz(self):
        # +500 at K moves y3 above b3 in the m/z order
        store = _peptide_store("PEPTIDEK").edit_mods(
            self.ModEdits(add=(self.ModSpec("Heavy", 500.0, "K"),)))
        store.finalize_spectra()
        spectrum = store[("PEPTIDEK(Heavy)", 2.0)]['spectrum']
        assert np.all(np.diff(spectrum[:, 0]) >= 0)


class TestVariableMods:
    from src.models.spec_lib.spec_lib import ModEdits, ModSpec
    ox = ModSpec("UniMod:35", 15.994915, "M")
    phospho = ModSpec("UniMod:21", 79.966331, "ST")

    def _variable(self, store, *specs, max_mods=2, **fixed):
        return store.edit_mods(self.ModEdits(variable=specs, max_variable=max_mods, **fixed))

    def test_every_combination_up_to_the_maximum(self):
        store = self._variable(_peptide_store("PEMSTK"), self.ox, self.phospho)
        assert set(store.mod_seq) == {
            "PEMSTK", "PEM(UniMod:35)STK", "PEMS(UniMod:21)TK", "PEMST(UniMod:21)K",
            "PEM(UniMod:35)S(UniMod:21)TK", "PEM(UniMod:35)ST(UniMod:21)K", "PEMS(UniMod:21)T(UniMod:21)K"}

    def test_the_maximum_counts_all_variable_mods_together(self):
        store = self._variable(_peptide_store("PEMSTK"), self.ox, self.phospho, max_mods=1)
        assert len(store) == 4   # the original and one copy per site

    def test_a_copy_has_its_parents_spectrum_shifted(self):
        store = _peptide_store("PEMSTK")
        prec_mz, b3, y3 = (store.prec_mz[0], store["PEMSTK", 2.0]['frags']['b3_1'][0],
                           store["PEMSTK", 2.0]['frags']['y3_1'][0])
        copy = self._variable(store, self.ox)["PEM(UniMod:35)STK", 2.0]
        assert copy['prec_mz'] == pytest.approx(prec_mz + 15.994915 / 2)
        assert copy['frags']['b3_1'][0] == pytest.approx(b3 + 15.994915, abs=1e-3)   # PEM
        assert copy['frags']['y3_1'][0] == pytest.approx(y3, abs=1e-3)              # STK
        assert copy['iRT'] == store["PEMSTK", 2.0]['iRT']

    def test_a_copy_the_library_already_has_is_dropped(self):
        store = self._variable(_peptide_store("PEMK", "PEM(UniMod:35)K"), self.ox)
        assert list(store.mod_seq) == ["PEMK", "PEM(UniMod:35)K"]

    def test_an_occupied_site_is_skipped_unless_stacked(self, app_log):
        assert len(self._variable(_peptide_store("PEM(UniMod:4)K"), self.ox)) == 1
        assert any(r.levelname == "WARNING" and "--add_variable_mod UniMod:35: skipped 1 sites" in r.getMessage()
                   for r in app_log)
        stacked = self.ModSpec("UniMod:35", 15.994915, "M", stack=True)
        assert "PEM(UniMod:4)(UniMod:35)K" in set(self._variable(_peptide_store("PEM(UniMod:4)K"), stacked).mod_seq)

    def test_copies_are_made_after_the_fixed_mods(self):
        label = self.ModSpec("Label", 8.0, "n")
        store = self._variable(_peptide_store("PEMK"), self.ox, add=(label,))
        assert list(store.mod_seq) == ["(Label)PEMK", "(Label)PEM(UniMod:35)K"]


class TestTakeEntries:
    def test_entries_can_repeat(self):
        store = SpectrumLibraryStore.from_dict(_make_sample_dict())
        taken = store.take_entries([1, 0, 1])
        assert list(taken.mod_seq) == [store.mod_seq[1], store.mod_seq[0], store.mod_seq[1]]
        assert np.array_equal(taken.frag_mz[:taken.frag_lengths[0]],
                              store.frag_mz[store.frag_offsets[1]:store.frag_offsets[1] + store.frag_lengths[1]])


class TestSerialization:
    def test_save_load_roundtrip(self, tmp_path):
        d = _make_sample_dict()
        store = SpectrumLibraryStore.from_dict(d)
        # Set some top_n
        for key in store:
            entry = store[key]
            spec = entry['spectrum']
            entry['top_n'] = np.argsort(-spec[:, 1])
        # Set parent_idx
        for i, key in enumerate(store):
            store[key]['parent_idx'] = i

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
        with tempfile.NamedTemporaryFile("w", newline="", delete=False, suffix=".tsv") as tmp:
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

    def test_phospho_spectronaut(self):
        with tempfile.NamedTemporaryFile("w", newline="", delete=False, suffix=".tsv") as tmp:
            writer = csv.writer(tmp, delimiter="\t")
            writer.writerow([
                "ModifiedPeptide", "PrecursorCharge", "PrecursorMz",
                "StrippedPeptide", "FragmentType", "FragmentNumber",
                "FragmentCharge", "FragmentMz", "RelativeIntensity",
                "RT"
            ])
            writer.writerow(["_AC[PHOSPHO (STY)]D_", 2, 450.2, "ACD", "y", 5, 1, 600.1, 0.8, 32.5])
            writer.writerow(["_ACD_", 2, 450.2, "ACD", "b", 3, 1, 300.5, 0.2, 32.5])
            filename = tmp.name

        with pytest.raises(JModError) as exc_info:
            SpectrumLibraryStore.from_tsv(filename)
        assert "Nested modification parentheses are not supported" in str(exc_info.value)
        assert "_AC[PHOSPHO (STY)]D_" in str(exc_info.value)
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
        # But underlying spectrum arrays are shared
        assert store.spectrum_mz is store2.spectrum_mz
        assert store.spectrum_int is store2.spectrum_int

    def test_deepcopy_key_index_store(self):
        # A KeyIndex points back at its store, so the copy needs its own index
        # rather than a deep copy of the original's (which used to recurse)
        store = _edgecases_store()
        assert isinstance(store.key_to_idx, KeyIndex)
        copied = copy.deepcopy(store)
        assert list(copied.keys()) == list(store.keys())
        assert copied.key_to_idx[("PEPTIDEK", 2.0)] == store.key_to_idx[("PEPTIDEK", 2.0)]
        assert copied.key_to_idx._store is copied


def _edgecases_store():
    return SpectrumLibraryStore.from_tsv(os.path.join(FIXTURE_DIR, "library_edgecases.tsv"))


class TestSubsetEntries:
    def test_subset_matches_per_entry_data(self):
        store = _edgecases_store()
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
        store = _edgecases_store()
        assert store.subset_entries(np.ones(len(store.mod_seq), dtype=bool)) is store

    def test_finalize_after_subset(self):
        store = _edgecases_store()
        mask = np.ones(len(store.mod_seq), dtype=bool)
        mask[0] = False
        sub = store.subset_entries(mask)
        sub.finalize_spectra()
        i = sub.key_to_idx[("SEVENPEPK", 2.0)]
        spec = sub.get_spectrum(i)
        assert np.all(np.diff(spec[:, 0]) >= 0)


def _with_decoys():
    return spec_lib.create_decoy_lib(_edgecases_store(), rules="rev")


class TestMonoisotopicTargets:
    def test_without_isotopes_returns_target_view(self):
        store = _with_decoys()
        assert isinstance(store.monoisotopic_targets(), _TargetView)

    def test_spectra_match_the_unexpanded_library(self):
        reference = _with_decoys()
        reference.finalize_spectra()
        mono = iso_library(_with_decoys(), tag=None, n_iso=3).monoisotopic_targets()
        for key in mono.keys():
            np.testing.assert_array_equal(mono.get_spectrum(mono.key_to_idx[key]),
                                          reference.get_spectrum(reference.key_to_idx[key]))

    def test_contains_only_targets(self):
        store = iso_library(_with_decoys(), tag=None, n_iso=3)
        mono = store.monoisotopic_targets()
        assert list(mono.keys()) == list(store.keys())[:store.n_targets]


class TestFreeze:
    def test_writes_to_a_frozen_store_raise(self):
        store = _edgecases_store().freeze()
        with pytest.raises(ValueError):
            store.iRT[0] = 1.0
        with pytest.raises(ValueError):
            store.frag_mz[0] = 1.0

    def test_deepcopy_of_a_frozen_store_is_writable(self):
        store = _edgecases_store().freeze()
        copied = copy.deepcopy(store)
        copied.iRT[0] = 1.0
        assert store.iRT[0] != 1.0
