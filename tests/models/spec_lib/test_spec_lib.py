import pytest
import numpy as np
import pandas as pd
import polars as pl
import os
import tempfile
import csv
import types

from src.models.spec_lib.spec_lib import create_python_lib, LibrarySpectrum, load_tsv_speclib, has_mass_tag, in_windows, check_nterm_tags
from src.models.spec_lib.spec_lib import ModSpec, parse_mod_spec, resolve_mod_edits, parse_mod_edits, match_library_spelling, loadSpecLib
from src.models.spec_lib.spec_lib import inspect_library, format_library_report, _mod_site_counts
from src.models.spec_lib.library_store import SpectrumLibraryStore
from tests.models.spec_lib.test_library_snapshots import FIXTURE_DIR
import src.config as config
from src.utils.errors import JModError
from src.mass_tags import massTag


def test_create_python_lib_basic():
    data = {
        "modification_sequence": ["ACD", "ACD"],
        "stripped_sequence": ["ACD", "ACD"],
        "Q1": [500.2, 500.2],
        "prec_z": [2, 2],
        "iRT": [25.0, 25.0],
        "frg_type": ["b", "y"],
        "frg_nr": [2, 3],
        "frg_z": [1, 1],
        "Q3": [200.1, 300.2],
        "relative_intensity": [0.8, 0.6]
    }

    df = pd.DataFrame(data)
    lib = create_python_lib(df)

    assert len(lib) == 1
    unique_id = ("ACD", 2)
    assert unique_id in lib

    expected_keys = ["mod_seq", "seq", "prec_mz", "prec_z", "iRT", "frags"]
    for key in expected_keys:
        assert key in lib[unique_id]

    assert lib[unique_id]["mod_seq"] == "ACD"
    assert lib[unique_id]["seq"] == "ACD"
    assert lib[unique_id]["prec_mz"] == 500.2
    assert lib[unique_id]["prec_z"] == 2
    assert lib[unique_id]["iRT"] == 25.0

    frags = lib[unique_id]["frags"]
    assert "b2_1" in frags
    assert "y3_1" in frags
    assert frags["b2_1"] == [200.1, 0.8]
    assert frags["y3_1"] == [300.2, 0.6]

def test_library_spectrum_read_entry():
    row = {
        "ModifiedPeptide": "ACD[Oxidation]EFG",
        "PeptideSequence": "ACDEFG",
        "PrecursorMz": "500.25",
        "PrecursorCharge": "2",
        "Tr_recalibrated": "32.5",
        "FragmentLossType": "noloss",
        "FragmentType": "y",
        "FragmentSeriesNumber": "5",
        "FragmentCharge": "1",
        "ProductMz": "350.12",
        "LibraryIntensity": "15000",
        "ProteinGroup": "PG1",
        "ProteinName": "SomeProtein",
        "Genes": "GENE1",
        "IonMobility": "1.23"
    }

    row_series = pd.Series(row)
    spec = LibrarySpectrum(seq="ACDEFG", z=2)

    spec.read_entry(row_series)

    assert spec.__data__["mod_seq"] == "ACD[Oxidation]EFG"
    assert spec.__data__["seq"] == "ACDEFG"
    frags = spec.__data__["frags"]
    assert "y5_1" in frags
    assert frags["y5_1"] == [350.12, 15000.0]

def test_load_tsv_speclib_minimal():
    # Create a temporary minimal DIA-NN style TSV file
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".tsv") as tmp:
        writer = csv.writer(tmp, delimiter="\t")
        
        # Header
        writer.writerow([
            "ModifiedPeptide", "PrecursorCharge", "PrecursorMz",
            "StrippedPeptide", "FragmentType", "FragmentNumber",
            "FragmentCharge", "FragmentMz", "RelativeIntensity",
            "RT"
        ])

        # One precursor, two fragments
        writer.writerow([
            "_ACD_", 2, 450.2,
            "ACD", "y", 5, 1, 600.1, 0.8, 32.5
        ])
        writer.writerow([
            "_ACD_", 2, 450.2,
            "ACD", "b", 3, 1, 300.5, 0.2, 32.5
        ])

        filename = tmp.name

    lib = load_tsv_speclib(filename)
    key = list(lib.keys())[0]
    entry = lib[key]

    assert entry["mod_seq"] == "ACD"    
    assert entry["seq"] == "ACD"
    assert entry["prec_z"] == 2
    assert entry["prec_mz"] == 450.2
    assert entry["iRT"] == 32.5
    frags = entry["frags"]
    assert "y5_1" in frags
    assert "b3_1" in frags
    assert frags["y5_1"] == [600.1, 0.8]


class Test_has_mass_tag():

    def test_no_tags(self):
        peptides = ["PEPTIDEK", "AAAEQAISVR"]
        prec_mzs = [500.0, 600.0]
        prec_zs = [2, 2]
        source_channel_mass, found, name = has_mass_tag(peptides, prec_mzs, prec_zs)
        assert found is False
        assert source_channel_mass == 0
        assert name == None

    def test_one_tag_no_mod(self):
        peptides = ["P(PSMtag-0)EPTIDEK"]
        prec_mzs = [464.734740 + (150/2)]
        prec_zs = [2]
        source_channel_mass, found, name = has_mass_tag(peptides, prec_mzs, prec_zs)
        assert found is True
        assert np.isclose(source_channel_mass, 150)
        assert name == "PSMtag-0"

    def test_two_tags_no_mod(self):
        peptides = ["P(PSMtag-0)EPTIDEK(PSMtag-0)"]
        prec_mzs = [464.734740 + (300/2)]
        prec_zs = [2]
        source_channel_mass, found, name = has_mass_tag(peptides, prec_mzs, prec_zs)
        assert found is True
        assert np.isclose(source_channel_mass, 150)
        assert name == "PSMtag-0"

    def test_mod(self):
        peptides = ["PEP(UniMod:21)TIDEK"]
        prec_mzs = [464.734740 + (79.966331/2)]
        prec_zs = [2]
        source_channel_mass, found, name = has_mass_tag(peptides, prec_mzs, prec_zs)
        assert found is False
        assert source_channel_mass == 0
        assert name == None

    def test_mod_and_tag(self):
        peptides = ["P(PSMtag-0)EP(UniMod:21)TIDEK"]
        prec_mzs = [464.734740 + (79.966331/2) + (150/2)]
        prec_zs = [2]
        source_channel_mass, found, name = has_mass_tag(peptides, prec_mzs, prec_zs)
        assert found is True
        assert np.isclose(source_channel_mass, 150)
        assert name == "PSMtag-0"

    def test_mod_and_tag_same_residue(self):
        peptides = ["PEP(UniMod:21)(PSMtag-0)TIDEK"]
        prec_mzs = [464.734740 + (79.966331/2) + (150/2)]
        prec_zs = [2]
        source_channel_mass, found, name = has_mass_tag(peptides, prec_mzs, prec_zs)
        assert found is True
        assert np.isclose(source_channel_mass, 150)
        assert name == "PSMtag-0"


    def test_several_channels_of_one_tag_raise(self):
        peptides = ["(PSMtag-0)PEPTIDEK(PSMtag-0)", "(PSMtag-4)ELVISK(PSMtag-4)"]
        with pytest.raises(JModError, match="Multi-channel libraries are not supported"):
            has_mass_tag(peptides, [600.0, 500.0], [2, 2])

    def test_names_with_a_dash_but_no_channel_number_are_not_channels(self):
        peptides = ["(Dimethyl-Nter)PEPTIDEK(Dimethyl-Lys)"]
        with pytest.raises(JModError, match="2 modifications JMod has no mass for: Dimethyl-Lys, Dimethyl-Nter"):
            has_mass_tag(peptides, [600.0], [2])

    def test_several_unknown_modifications_raise(self):
        peptides = ["(PSMtag-0)PEPTIDEK(PSMtag-0)", "(DimethylNter)ELVISK"]
        with pytest.raises(JModError, match="DimethylNter, PSMtag-0"):
            has_mass_tag(peptides, [600.0, 500.0], [2, 2])


class TestMatchTagChannel:
    from src.models.spec_lib.spec_lib import match_tag_channel
    match = staticmethod(match_tag_channel)
    tag = massTag(rules="nK", base_mass=100.0, delta=[0.0, 4.0], channel_names=["0", "4"], name="T")

    def test_a_close_mass_matches_without_a_warning(self):
        assert self.match(self.tag, "tag", 104.001) == ("T-4", pytest.approx(-0.001), None)

    def test_a_few_mDa_off_is_a_warning(self):
        channel, _, warning = self.match(self.tag, "tag", 100.005)
        assert channel == "T-0" and "is 0.0050 Da from T-0" in warning

    def test_more_than_10_mDa_off_is_an_error(self):
        # SILAC +8 against dimethyl +8 is 0.030 Da
        with pytest.raises(JModError, match="it is not this tag"):
            self.match(self.tag, "tag", 100.03)

    def test_the_tag_mass_is_the_median_of_its_precursors(self):
        from pyteomics import mass as pyteomics_mass
        mz = lambda seq, tag_mass: (pyteomics_mass.fast_mass(seq) + tag_mass + 2 * 1.00727647) / 2
        peptides = ["(t)PEPTIDEK", "(t)ELVISK", "(t)AAAAK"]
        prec_mz = [mz("PEPTIDEK", 100.0), mz("ELVISK", 100.0), mz("AAAAK", 150.0)]   # one badly written
        tag_mass, found, name = has_mass_tag(peptides, prec_mz, [2, 2, 2])
        assert found and name == "t" and tag_mass == pytest.approx(100.0, abs=1e-6)


class TestTagMassNotWorkedOut:
    def test_a_tag_mass_that_cannot_be_worked_out_raises(self):
        # No precursor m/z to work the mass out from
        with pytest.raises(JModError, match="could not be worked out"):
            has_mass_tag(["(t)PEPTIDEK"], [float("nan")], [2])


class Test_check_nterm_tags():
    tag = massTag(rules="nK", base_mass=140.0949630177, delta=[0.0],
                  channel_names=["0"], name="mTRAQ")

    def test_tags_in_front_of_the_first_residue_pass(self):
        check_nterm_tags(["(mTRAQ-0)PEPTIDEK(mTRAQ-0)", "(mTRAQ-0)K(mTRAQ-0)EPR"], self.tag)

    @pytest.mark.parametrize("peptide", ["P(mTRAQ-0)EPTIDEK(mTRAQ-0)", "K(mTRAQ-0)(mTRAQ-0)EPR"])
    def test_tags_behind_the_first_residue_raise(self, peptide):
        with pytest.raises(JModError, match="behind the first residue"):
            check_nterm_tags(["(mTRAQ-0)ELVISK(mTRAQ-0)", peptide], self.tag)

    def test_tags_without_an_n_rule_pass(self):
        k_only = massTag(rules="K", base_mass=0, delta=[0.0], channel_names=["0"], name="mTRAQ")
        check_nterm_tags(["K(mTRAQ-0)EPK(mTRAQ-0)"], k_only)


def _scans(*windows):
    """Stand-in MS2 scans carrying only their isolation window."""
    return [types.SimpleNamespace(ms1window=np.array(w, dtype=float)) for w in windows]


class Test_in_windows():

    def test_membership(self):
        # windows covering only 464.75 and 520.77
        mz = np.array([464.75, 520.77, 830.51, 540.26])
        mask = in_windows(mz, _scans([460, 470], [515, 525]))
        assert mask.tolist() == [True, True, False, False]

    def test_margin(self):
        # 464.75 sits ~32 ppm above 464.735; the default 50 ppm margin recovers it
        mz = np.array([464.75])
        assert in_windows(mz, _scans([460.0, 464.735]))[0]
        assert not in_windows(mz, _scans([460.0, 464.735]), margin_ppm=1.0)[0]

    def test_overlapping_windows_merge(self):
        mz = np.array([465.0, 472.0, 480.0])
        mask = in_windows(mz, _scans([460, 470], [468, 475]))
        assert mask.tolist() == [True, True, False]

    def test_nan_is_outside(self):
        assert not in_windows(np.array([np.nan]), _scans([460, 470]))[0]


@pytest.fixture
def restore_mods():
    """config.diann_mods as it was before the test (resolve_mod_edits adds to it)."""
    saved = dict(config.diann_mods)
    yield
    config.diann_mods.clear()
    config.diann_mods.update(saved)


class TestParseModSpec:
    def test_name_mass_and_sites(self):
        assert parse_mod_spec("Dimethyl,28.0313,nK", "--add_fixed_mod") == ModSpec("Dimethyl", 28.0313, "nK")

    def test_a_UniMod_mass_is_looked_up(self):
        assert parse_mod_spec("UniMod:4,C", "--strip_mod") == ModSpec("UniMod:4", 57.021464, "C")

    def test_UniMod_is_spelled_as_libraries_spell_it(self):
        assert parse_mod_spec("unimod:4,C", "--strip_mod").name == "UniMod:4"

    def test_strip_by_name_alone_is_every_site(self):
        spec = parse_mod_spec("UniMod:36", "--strip_mod")
        assert spec.name == "UniMod:36" and spec.mass == pytest.approx(28.0313, abs=1e-4)
        assert spec.nterm and spec.residues == "ACDEFGHIJKLMNOPQRSTUVWY"   # every residue the parser accepts
        assert parse_mod_spec("DimethylNter,28.0313", "--strip_mod").sites == spec.sites

    def test_strip_by_an_unknown_name_alone_asks_for_its_mass(self):
        with pytest.raises(JModError, match="give it as DimethylNter,MASS$"):
            parse_mod_spec("DimethylNter", "--strip_mod")

    def test_adding_still_needs_sites(self):
        with pytest.raises(JModError, match="expected NAME,MASS,SITES or NAME,SITES"):
            parse_mod_spec("UniMod:36", "--add_fixed_mod")

    def test_names_and_sites_are_not_case_sensitive(self):
        assert parse_mod_spec("lys8,k", "--add_fixed_mod") == ModSpec("Lys8", 8.014199, "K")
        assert parse_mod_spec("X,1.0,nk", "--add_fixed_mod").sites == "nK"
        assert parse_mod_spec("X,1.0,N", "--add_fixed_mod").sites == "N"   # uppercase N is asparagine

    def test_stack_is_read_from_the_end(self):
        assert parse_mod_spec("UniMod:121,K,stack", "--add_fixed_mod") == ModSpec("UniMod:121", 114.042927, "K", True)
        assert parse_mod_spec("Label,8.0,n,stack", "--add_fixed_mod") == ModSpec("Label", 8.0, "n", True)

    def test_stack_is_only_for_adding(self):
        with pytest.raises(JModError, match="stack is only for --add_fixed_mod and --add_variable_mod"):
            parse_mod_spec("UniMod:4,C,stack", "--strip_mod")

    def test_a_known_modification_keeps_its_known_mass(self, app_log):
        assert parse_mod_spec("UniMod:4,57.0215,C", "--add_fixed_mod").mass == 57.021464
        assert not [r for r in app_log if r.levelname == "WARNING"]   # only rounded

    def test_a_different_mass_for_a_known_modification_is_warned_about(self, app_log):
        assert parse_mod_spec("UniMod:4,333,C", "--add_fixed_mod").mass == 57.021464
        assert any(r.levelname == "WARNING" and "the 333 given is ignored" in r.getMessage()
                   for r in app_log)

    def test_an_unknown_modification_needs_a_mass(self):
        with pytest.raises(JModError, match="give it as Dimethyl,MASS,nK"):
            parse_mod_spec("Dimethyl,nK", "--add_fixed_mod")

    @pytest.mark.parametrize("text", ["Dimethyl", "Dimethyl,28.0313,nK,x", "Dimethyl,heavy,nK",
                                      "Dimethyl,28.0313,", "Dimethyl,28.0313,nB", "Di methyl,28.0313,n"])
    def test_malformed_specs_raise(self, text):
        with pytest.raises(JModError, match="--add_fixed_mod"):
            parse_mod_spec(text, "--add_fixed_mod")


class TestResolveModEdits:
    def test_strips_and_adds_are_parsed_and_their_masses_known(self, restore_mods):
        edits = resolve_mod_edits(["Label,8.0120,n"], "DimethylNter,28.0313,n")
        assert edits.add == (ModSpec("Label", 8.0120, "n"),)
        assert edits.strip == (ModSpec("DimethylNter", 28.0313, "n"),)
        assert config.diann_mods["Label"] == 8.0120
        assert config.diann_mods["DimethylNter"] == 28.0313

    def test_variable_mods_and_their_maximum(self, restore_mods):
        edits = resolve_mod_edits(None, None, ["UniMod:35,M"], 3)
        assert edits.variable == (ModSpec("UniMod:35", 15.994915, "M"),)
        assert edits.max_variable == 3 and edits

    @pytest.mark.parametrize("bad", [0, "two"])
    def test_max_variable_mods_must_be_a_positive_whole_number(self, restore_mods, bad):
        with pytest.raises(JModError, match="--max_variable_mods"):
            resolve_mod_edits(None, None, ["UniMod:35,M"], bad)

    def test_one_name_with_two_masses_raises(self, restore_mods):
        with pytest.raises(JModError, match="SILAC is given two masses: 8.01419867 in --add_fixed_mod "
                                            "SILAC,8.01419867,K,stack, and 3.9881396 in --add_fixed_mod"):
            resolve_mod_edits(["SILAC,8.01419867,K,stack", "SILAC,3.9881396,R"], None)

    def test_one_name_with_one_mass_at_several_sites_is_fine(self, restore_mods):
        edits = resolve_mod_edits(["Label,8.0,K", "Label,8.0,R"], "Label,8.0")
        assert [spec.sites for spec in edits.add] == ["K", "R"]

    def test_one_name_whatever_the_case_has_one_mass(self, restore_mods):
        with pytest.raises(JModError, match="SILAC is given two masses"):
            resolve_mod_edits(["silac,8.0,K", "SILAC,4.0,R"], None)

    def test_names_take_the_librarys_spelling(self, restore_mods):
        edits = match_library_spelling(parse_mod_edits(None, ["dimethyl-lys,28.0313"]), {"Dimethyl-Lys"})
        assert edits.strip[0].name == "Dimethyl-Lys"
        assert config.diann_mods["Dimethyl-Lys"] == 28.0313

    def test_a_name_two_library_spellings_could_mean_raises(self, restore_mods):
        with pytest.raises(JModError, match="differ only in case"):
            match_library_spelling(parse_mod_edits(None, ["dimethyl-lys,28.0313"]),
                                   {"Dimethyl-Lys", "DIMETHYL-LYS"})

    def test_loading_strips_a_name_in_any_case(self, tmp_path, restore_mods):
        path = str(tmp_path / "lib.parquet")
        pl.DataFrame({"ModifiedPeptide": ["PEPC(Dimethyl-Lys)K"], "StrippedPeptide": ["PEPCK"],
                      "PrecursorCharge": [2], "PrecursorMz": [300.0], "Tr_recalibrated": [10.0],
                      "FragmentType": ["y"], "FragmentNumber": [2], "FragmentCharge": [1],
                      "FragmentLossType": ["noloss"], "FragmentMz": [250.0],
                      "RelativeIntensity": [1.0]}).write_parquet(path)
        store, *_ = loadSpecLib(path, parse_mod_edits(None, ["dimethyl-lys,28.0313"]))
        assert list(store.mod_seq) == ["PEPCK"]

    def test_none_is_no_edits(self, restore_mods):
        assert not resolve_mod_edits(None, None)


class TestInspectLibrary:
    """--inspect_library and the GUI's "i" button: a library's modifications, and the
    problems loading it would raise."""
    tag = massTag(rules="nK", base_mass=0.0, delta=[0.0], channel_names=["0"], name="mTRAQ")

    @pytest.fixture
    def edgecases(self, tmp_path):
        """The edge-case library, copied where it has no binary cache."""
        import shutil
        path = tmp_path / "library_edgecases.tsv"
        shutil.copy(os.path.join(FIXTURE_DIR, "library_edgecases.tsv"), path)
        return str(path)

    def test_mods_are_counted_per_site(self):
        counts = _mod_site_counts(pl.Series(["(a)K(b)PEK(b)", "PEPC(c)K", "PEPK"]))
        assert sorted(counts.iter_rows()) == [("a", "N-term", 1), ("b", "K", 1), ("c", "C", 1)]

    def test_targets_and_their_mods_are_reported(self, edgecases):
        report = inspect_library(edgecases)
        assert report.n_precursors == 4   # the decoys and invalid residues are left out
        assert report.source == "file"
        rows = {(r.name, r.site): r for r in report.mods}
        assert rows[("UniMod:4", "C")].mass == pytest.approx(57.021464)
        assert rows[("tag", "N-term")].status.startswith("tag")

    def test_the_binary_cache_gives_the_same_report(self, edgecases):
        SpectrumLibraryStore.from_tsv(edgecases).save(edgecases + "_store.npz")
        cached = inspect_library(edgecases)
        assert cached.source == "binary cache"
        assert cached.mods == inspect_library_from_file(edgecases).mods

    def test_no_tag_selected_is_a_problem(self, edgecases):
        assert any("no tag is selected" in p for p in inspect_library(edgecases).problems)

    @staticmethod
    def _tag_for(path):
        """An mTRAQ-named tag whose one channel is the library's own tag mass."""
        from src.models.spec_lib.spec_lib import _library_precursors
        precursors, _ = _library_precursors(path)
        tag_mass, _, _ = has_mass_tag(precursors["mod_seq"], precursors["prec_mz"], precursors["prec_z"])
        return massTag(rules="nK", base_mass=tag_mass, delta=[0.0], channel_names=["0"], name="mTRAQ")

    def test_with_a_tag_the_channel_is_named(self, edgecases):
        report = inspect_library(edgecases, self._tag_for(edgecases))
        assert not report.problems
        assert "-> mTRAQ-0" in {(r.name, r.site): r for r in report.mods}[("tag", "N-term")].status

    def test_several_unknown_mods_are_a_problem(self, tmp_path):
        path = str(tmp_path / "lib.parquet")
        pl.DataFrame({"ModifiedPeptide": ["(foo)PEPTIDEK", "PEPTIDEK(bar)"],
                      "PrecursorCharge": [2, 2], "PrecursorMz": [500.0, 500.0]}).write_parquet(path)
        report = inspect_library(path)
        assert any("bar, foo" in p for p in report.problems)
        assert "JMod will stop with an error when it loads this library:" in format_library_report(report)

    def test_edits_are_applied_as_a_run_would(self, tmp_path, restore_mods):
        path = str(tmp_path / "lib.parquet")
        pl.DataFrame({"ModifiedPeptide": ["P(Dimethyl-Nter)EPTIDEK(Dimethyl-Lys)", "AAAK(Dimethyl-Lys)"],
                      "PrecursorCharge": [2, 2], "PrecursorMz": [500.0, 300.0]}).write_parquet(path)
        edits = parse_mod_edits(["UniMod:36,nK"], ["Dimethyl-Nter,28.0313", "Dimethyl-Lys,28.0313"])
        report = inspect_library(path, None, edits)
        assert not report.problems   # every mod now has a mass
        assert {(r.name, r.site, r.n_precursors) for r in report.edited_mods} == {
            ("UniMod:36", "N-term", 2), ("UniMod:36", "K", 2)}
        assert ("Stripped Dimethyl-Nter from 1 sites" in
                [message for _, message in report.edit_lines])
        text = format_library_report(report)
        assert "What the edits do:" in text and "JMod can load this library with these edits" in text

    def test_the_tag_keeps_its_status_after_the_edits(self, tmp_path, restore_mods):
        path = str(tmp_path / "lib.parquet")
        pl.DataFrame({"ModifiedPeptide": ["(mTRAQ-0)PEPTIDEK(mTRAQ-0)"], "PrecursorCharge": [2],
                      "PrecursorMz": [500.0]}).write_parquet(path)
        report = inspect_library(path, self._tag_for(path), parse_mod_edits(["Label,8.0,R"], None))
        before = {r.name: r.status for r in report.mods}
        after = {r.name: r.status for r in report.edited_mods}
        assert after["mTRAQ-0"] == before["mTRAQ-0"] and "-> mTRAQ-0" in after["mTRAQ-0"]

    def test_a_strip_that_misses_sites_is_reported(self, tmp_path, restore_mods):
        path = str(tmp_path / "lib.parquet")
        pl.DataFrame({"ModifiedPeptide": ["P(Dimethyl-Nter)EPTIDEK", "A(Dimethyl-Nter)AAK"],
                      "PrecursorCharge": [2, 2], "PrecursorMz": [500.0, 300.0]}).write_parquet(path)
        report = inspect_library(path, None, parse_mod_edits(None, ["Dimethyl-Nter,28.0313,n"]))
        assert len(report.leftovers) == 1
        assert "leaves Dimethyl-Nter on 2 other sites (precursors): A (1), P (1)" in report.leftovers[0]
        assert report.leftovers[0].endswith("--strip_mod Dimethyl-Nter,28.0313")

    def test_a_large_library_is_edited_in_a_sample(self, tmp_path, restore_mods):
        path = str(tmp_path / "lib.parquet")
        pl.DataFrame({"ModifiedPeptide": ["PEPTIDEK", "AAAK", "ELVISK"], "PrecursorCharge": [2, 2, 2],
                      "PrecursorMz": [500.0, 300.0, 400.0]}).write_parquet(path)
        edits = parse_mod_edits(["Label,8.0,K"], None)
        report = inspect_library(path, None, edits, sample_size=2, sample_above=2)
        assert report.sample_size == 2 and report.n_after == 3
        assert "in a random sample of 2 precursors" in "\n".join(format_library_report(report))

    def test_a_library_up_to_the_threshold_is_edited_in_full(self, tmp_path, restore_mods):
        path = str(tmp_path / "lib.parquet")
        pl.DataFrame({"ModifiedPeptide": ["PEPTIDEK", "AAAK", "ELVISK"], "PrecursorCharge": [2, 2, 2],
                      "PrecursorMz": [500.0, 300.0, 400.0]}).write_parquet(path)
        edits = parse_mod_edits(["Label,8.0,K"], None)
        assert inspect_library(path, None, edits, sample_size=2, sample_above=3).sample_size == 3

    def test_inspecting_with_edits_changes_no_masses(self, tmp_path, restore_mods):
        path = str(tmp_path / "lib.parquet")
        pl.DataFrame({"ModifiedPeptide": ["PEPTIDEK"], "PrecursorCharge": [2],
                      "PrecursorMz": [500.0]}).write_parquet(path)
        inspect_library(path, None, parse_mod_edits(["Label,8.0,K"], None))
        assert "Label" not in config.diann_mods


def inspect_library_from_file(path):
    """inspect_library reading the file even when a binary cache is beside it."""
    import shutil, tempfile
    copy = os.path.join(tempfile.mkdtemp(), os.path.basename(path))
    shutil.copy(path, copy)
    return inspect_library(copy)
