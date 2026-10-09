import pytest
import numpy as np
import pandas as pd
import polars as pl
import os
import tempfile
import csv
import types

from src.models.spec_lib.spec_lib import create_python_lib, LibrarySpectrum, load_tsv_speclib, has_mass_tag, in_windows, check_nterm_tags
from src.models.spec_lib.spec_lib import ModSpec, parse_mod_spec, resolve_mod_edits
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

    def test_several_unknown_modifications_raise(self):
        peptides = ["(PSMtag-0)PEPTIDEK(PSMtag-0)", "(DimethylNter)ELVISK"]
        with pytest.raises(JModError, match="DimethylNter, PSMtag-0"):
            has_mass_tag(peptides, [600.0, 500.0], [2, 2])


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

    def test_with_a_tag_the_channel_is_named(self, edgecases):
        report = inspect_library(edgecases, self.tag)
        assert not report.problems
        assert "-> mTRAQ-0" in {(r.name, r.site): r for r in report.mods}[("tag", "N-term")].status

    def test_several_unknown_mods_are_a_problem(self, tmp_path):
        path = str(tmp_path / "lib.parquet")
        pl.DataFrame({"ModifiedPeptide": ["(foo)PEPTIDEK", "PEPTIDEK(bar)"],
                      "PrecursorCharge": [2, 2], "PrecursorMz": [500.0, 500.0]}).write_parquet(path)
        report = inspect_library(path)
        assert any("bar, foo" in p for p in report.problems)
        assert "Problems when loading:" in format_library_report(report)


def inspect_library_from_file(path):
    """inspect_library reading the file even when a binary cache is beside it."""
    import shutil, tempfile
    copy = os.path.join(tempfile.mkdtemp(), os.path.basename(path))
    shutil.copy(path, copy)
    return inspect_library(copy)
