"""
This Source Code Form is subject to the terms of the Oxford Nanopore
Technologies, Ltd. Public License, v. 1.0.  Full licence can be found
at https://github.com/ParallelSquared/JMod/blob/main/LICENSE.txt
"""

"""
Tests for functions in mass_tags.py
"""

import pytest
from unittest.mock import Mock, patch
from pyteomics import mass
import numpy as np

# Import the functions we want to test
from src.iso_functions import (
    fragment_seq, split_peptide, get_seq_comp, frag_isotope, gen_isotopes, calculate_mz, precursor_isotopes, iso_distr, my_iso_distr
)
from tests.fixtures.test_data import SAMPLE_LIBRARY_ENTRY


class TestFragmentSeq:
    """Tests for the fragment_seq function"""

    def test_b_fragment_basic(self):
        peptide = "PEPTIDE"
        ion_type = "b3_1"
        seq, frag_info = fragment_seq(peptide, ion_type)
        
        assert seq == ["P", "E", "P"]
        assert frag_info == ["b", 3, "", "1"]

    def test_y_fragment_basic(self):
        peptide = "PEPTIDE"
        ion_type = "y2_2"
        seq, frag_info = fragment_seq(peptide, ion_type)
        
        assert seq == ["D", "E"]
        assert frag_info == ["y", 2, "", "2"]

    def test_fragment_with_modifications(self):
        peptide = "PEP(Tide)(+15.99)IDE"
        ion_type = "b4_1"
        seq, frag_info = fragment_seq(peptide, ion_type)
        
        assert len(seq) == 4
        assert seq[2].startswith("P") or "(" in seq[2]  # mod may be on 3rd AA
        assert frag_info[0] == "b"
        assert frag_info[1] == 4

    def test_invalid_ion_type(self):
        peptide = "PEPTIDE"
        ion_type = "q3_1"  # invalid fragment type
        with pytest.raises(ValueError, match="Invalid ion type"):
            fragment_seq(peptide, ion_type)

    def test_frag_index_exceeds_length(self):
        peptide = "PEP"
        ion_type = "b5_1"  # frag index exceeds peptide length
        with pytest.raises(AssertionError):
            fragment_seq(peptide, ion_type)

class TestSplitPeptide:
    def test_simple_sequence(self):
        peptide = "ACDE"
        result = split_peptide(peptide)
        assert result == ["A", "C", "D", "E"]

    def test_sequence_with_mods(self):
        peptide = "ACD(+15.99)E"
        result = split_peptide(peptide)
        assert result == ["A", "C", "D(+15.99)", "E"]

    def test_empty_sequence(self):
        peptide = ""
        result = split_peptide(peptide)
        assert result == []

    def test_multiple_mods(self):
        peptide = "A(+15.99)C(+57.02)DE"
        result = split_peptide(peptide)
        assert result == ["A(+15.99)", "C(+57.02)", "D", "E"]

class TestGetSeqComp:
    def test_sequence_no_mods(self):
        split_seq = ["A", "C", "D", "E"]
        comp = get_seq_comp(split_seq, ion_type="b")
        assert comp is not None
        assert sum(comp.values()) > 0

    def test_sequence_with_mods(self):
        split_seq = ["A", "C", "D(UniMod:35)", "E"] 
        comp = get_seq_comp(split_seq, ion_type="y")
        assert comp is not None
        assert sum(comp.values()) > 0


class TestFragIsotope:
    def test_basic_fragment_isotope_generation(self):
        seq = SAMPLE_LIBRARY_ENTRY["mod_seq"]
        frag = "b3_1"
        isotopes = frag_isotope(frag, seq)

        assert isotopes[0].mz < isotopes[-1].mz 

    def test_isotope_intensity_decreases(self):
        seq = SAMPLE_LIBRARY_ENTRY["mod_seq"]
        frag = "y3_1"
        isotopes = frag_isotope(frag, seq)
        intensities = [p.intensity for p in isotopes]
        assert intensities[0] == max(intensities)


class TestGenIsotopes:
    def test_generate_isotopes_from_library(self):
        seq = SAMPLE_LIBRARY_ENTRY["mod_seq"]
        frags = SAMPLE_LIBRARY_ENTRY["frags"]

        new_frags = gen_isotopes(seq, frags)

        assert isinstance(new_frags, np.ndarray)
        assert new_frags.shape[1] == 2
        assert np.isclose(np.max(new_frags[:, 1]), 1.0, atol=1e-6)

    def test_intensity_scaling_relative_to_mono_peak(self):
        seq = SAMPLE_LIBRARY_ENTRY["mod_seq"]
        frags = SAMPLE_LIBRARY_ENTRY["frags"]
        result = gen_isotopes(seq, frags)

        assert np.all(result[:, 1] <= 1)

    def test_output_is_reproducible(self):
        seq = SAMPLE_LIBRARY_ENTRY["mod_seq"]
        frags = SAMPLE_LIBRARY_ENTRY["frags"]
        first = gen_isotopes(seq, frags)
        second = gen_isotopes(seq, frags)

        np.testing.assert_allclose(first, second, rtol=1e-6, atol=1e-6)


class TestCalculateMz:
    def test_basic_mz_calculation(self):
        seq = "PEPTIDE"
        charge = 2
        mz = calculate_mz(seq, charge)
        assert mz > 0
        assert isinstance(mz, float)

    def test_mz_differs_with_charge(self):
        seq = "PEPTIDE"
        mz_2 = calculate_mz(seq, 2)
        mz_3 = calculate_mz(seq, 3)
        assert mz_2 != mz_3
        assert mz_2 > mz_3 


class TestPrecursorIsotopes:
    def test_isotopes_generated(self):
        seq = "PEPTIDE"
        charge = 2
        isotopes = precursor_isotopes(seq, charge, tag=None)
        assert len(isotopes) > 0
        assert hasattr(isotopes[0], "mz")
        assert hasattr(isotopes[0], "intensity")

    def test_isotope_order_and_intensity(self):
        seq = "PEPTIDE"
        charge = 2
        isotopes = precursor_isotopes(seq, charge, tag=None, n_isotopes=3)
        mzs = [p.mz for p in isotopes]
        intensities = [p.intensity for p in isotopes]
        assert mzs[0] < mzs[-1]
        assert intensities[0] == max(intensities)


class TestIsoDistr:
    def test_iso_distr_output_shape_and_type(self):
        temp = [10, 20, 2, 5, 1]  # [C, H, N, O, S]
        iso = iso_distr(temp)
        assert isinstance(iso, np.ndarray)
        assert iso.ndim == 1
        assert np.isclose(np.max(iso), 1.0, atol=1e-6)

    def test_iso_distr_is_normalized(self):
        temp = [5, 10, 2, 3, 0]
        iso = iso_distr(temp)
        assert np.isclose(np.max(iso), 1.0)
        assert np.all(iso >= 0)


class TestMyIsoDistr:
    def test_my_iso_distr_output_shape_and_type(self):
        comp = {"C": 10, "H": 20, "N": 2, "O": 5, "S": 1}
        iso = my_iso_distr(comp)
        assert isinstance(iso, np.ndarray)
        assert iso.ndim == 1
        assert np.isclose(np.max(iso), 1.0, atol=1e-6)

    def test_my_iso_distr_equivalence_with_iso_distr(self):
        temp = [6, 12, 1, 6, 0]
        comp = {"C": 6, "H": 12, "N": 1, "O": 6, "S": 0}
        iso1 = iso_distr(temp)
        iso2 = my_iso_distr(comp)
        np.testing.assert_allclose(iso1, iso2, rtol=1e-6, atol=1e-6)
 