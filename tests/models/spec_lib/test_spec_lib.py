import pytest
import numpy as np
import pandas as pd
import os
import tempfile
import csv
import types

from src.models.spec_lib.spec_lib import create_python_lib, LibrarySpectrum, load_tsv_speclib, has_mass_tag, in_windows


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

