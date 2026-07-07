import pytest
import numpy as np
import pandas as pd
import os
import tempfile
import csv

from src.models.spec_lib.spec_lib import create_python_lib, LibrarySpectrum, load_tsv_speclib, has_mass_tag


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
    # Path to your small test file included in the repo
    test_file = os.path.join("data", "filtered_library.tsv_pythonlib")

    # Load only first ~5 rows by patching csv reader or reading partially
    # Easiest: read full file but assert only minimal behavior
    lib = load_tsv_speclib(test_file)

    # Basic sanity checks
    assert isinstance(lib, dict)
    assert len(lib) > 0  # should parse at least one entry

    # Grab any random entry
    first_key = next(iter(lib.keys()))
    entry = lib[first_key]

    # Check minimal fields
    assert "mod_seq" in entry
    assert "seq" in entry
    assert "prec_mz" in entry
    assert "prec_z" in entry
    assert "frags" in entry
    assert isinstance(entry["frags"], dict)

    # Check fragments structurally look right
    for frag_type, frag_list in entry["frags"].items():
        assert isinstance(frag_type, str)
        assert isinstance(frag_list, list)
        assert len(frag_list) == 2  # mz, intensity

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
        assert name == "(PSMtag-0)"

    def test_two_tags_no_mod(self):
        peptides = ["P(PSMtag-0)EPTIDEK(PSMtag-0)"]
        prec_mzs = [464.734740 + (300/2)]
        prec_zs = [2]
        source_channel_mass, found, name = has_mass_tag(peptides, prec_mzs, prec_zs)
        assert found is True
        assert np.isclose(source_channel_mass, 150)
        assert name == "(PSMtag-0)"

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
        assert name == "(PSMtag-0)"

    def test_mod_and_tag_same_residue(self):
        peptides = ["PEP(UniMod:21)(PSMtag-0)TIDEK"]
        prec_mzs = [464.734740 + (79.966331/2) + (150/2)]
        prec_zs = [2]
        source_channel_mass, found, name = has_mass_tag(peptides, prec_mzs, prec_zs)
        assert found is True
        assert np.isclose(source_channel_mass, 150)
        assert name == "(PSMtag-0)"

