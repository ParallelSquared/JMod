"""
This Source Code Form is subject to the terms of the Oxford Nanopore
Technologies, Ltd. Public License, v. 1.0.  Full licence can be found
at https://github.com/ParallelSquared/JMod/blob/main/LICENSE.txt
"""

"""
Tests for functions in iso_functions.py
Real functional tests without mocking framework
"""
import pytest
import numpy as np
import re
from src.mass_tags import massTag, tag6
import src.mass_tags as mt

# Import the functions we want to test
# Assuming these are imported from the correct module path
try:
    from src.utils.iso_functions import (
        split_frag_name, parse_peptide, fragment_seq,
        bits1, cut, iso_distr, my_iso_distr, get_seq_comp, gen_isotopes_dict 
    )
except ImportError:
    # Skip tests if imports fail
    pytest.skip("iso_functions module not available", allow_module_level=True)


class TestSplitFragName:
    """Test cases for the split_frag_name function"""
    
    def test_split_simple_fragments(self):
        """Test splitting simple fragment names"""
        # Basic b-ion
        frag_type, frag_idx, loss, frag_z = split_frag_name("b5_2")
        assert frag_type == "b"
        assert frag_idx == 5
        assert loss == ""
        assert frag_z == "2"
        
        # Basic y-ion
        frag_type, frag_idx, loss, frag_z = split_frag_name("y10_1")
        assert frag_type == "y"
        assert frag_idx == 10
        assert loss == ""
        assert frag_z == "1"
    
    def test_split_fragments_with_losses(self):
        """Test splitting fragment names with neutral losses"""
        frag_type, frag_idx, loss, frag_z = split_frag_name("y10-H2O_1")
        assert frag_type == "y"
        assert frag_idx == 10
        assert loss == "H2O"
        assert frag_z == "1"
        
        frag_type, frag_idx, loss, frag_z = split_frag_name("b3-NH3_2")
        assert frag_type == "b"
        assert frag_idx == 3
        assert loss == "NH3"
        assert frag_z == "2"
    
    def test_split_various_ion_types(self):
        """Test various fragment ion types"""
        test_cases = [
            ("a3_1", ("a", 3, "", "1")),
            ("c4_2", ("c", 4, "", "2")),
            ("x5_1", ("x", 5, "", "1")),
            ("z6_3", ("z", 6, "", "3")),
        ]
        
        for ion_name, expected in test_cases:
            result = split_frag_name(ion_name)
            assert result == expected
    
    def test_split_error_cases(self):
        """Test error cases for fragment name splitting"""
        with pytest.raises(ValueError):
            split_frag_name("b5")  # Missing underscore and charge
        
        with pytest.raises(ValueError):
            split_frag_name("y10-H2O")  # Missing charge


class TestParsePeptide:
    """Test cases for parse_peptide function"""
    
    def test_parse_simple_sequences(self):
        """Test parsing simple peptide sequences"""
        assert parse_peptide("PEPTIDE") == ['P', 'E', 'P', 'T', 'I', 'D', 'E']
        assert parse_peptide("A") == ['A']
        assert parse_peptide("") == []
    
    def test_parse_with_parentheses_modifications(self):
        """Test parsing with modifications in parentheses"""
        result = parse_peptide("PEP(+10.0)TIDE")
        expected = ['P', 'E', 'P(+10.0)', 'T', 'I', 'D', 'E']
        assert result == expected
        
        result = parse_peptide("K(mTRAQ)PEPTIDE(+15.99)R")
        expected = ['K(mTRAQ)', 'P', 'E', 'P', 'T', 'I', 'D', 'E(+15.99)', 'R']
        assert result == expected
    
    def test_parse_with_square_brackets(self):
        """Test parsing with square bracket modifications"""
        result = parse_peptide("PEP[+80]TIDE")
        expected = ['P', 'E', 'P[+80]', 'T', 'I', 'D', 'E']
        assert result == expected
        
        result = parse_peptide("C[+57.02]PEPTIDE[+15.99]")
        expected = ['C[+57.02]', 'P', 'E', 'P', 'T', 'I', 'D', 'E[+15.99]']
        assert result == expected
    
    def test_parse_mixed_brackets(self):
        """Test parsing with both types of brackets"""
        result = parse_peptide("K(mTRAQ)PEP[+80]TIDE")
        expected = ['K(mTRAQ)', 'P', 'E', 'P[+80]', 'T', 'I', 'D', 'E']
        assert result == expected
    
    def test_parse_nested_brackets(self):
        """Test parsing with nested brackets"""
        # The function should handle basic nesting
        result = parse_peptide("K(nested(content))PEPTIDE")
        expected = ['K(nested(content))', 'P', 'E', 'P', 'T', 'I', 'D', 'E']
        assert result == expected
    
    def test_parse_edge_cases(self):
        """Test edge cases"""
        # Multiple modifications on same residue
        result = parse_peptide("K(+42)(+15)PEPTIDE")
        expected = ['K(+42)(+15)', 'P', 'E', 'P', 'T', 'I', 'D', 'E']
        assert result == expected
        
        # Modification at start (edge case)
        result = parse_peptide("(+42)PEPTIDE")
        expected = ['(+42)', 'P', 'E', 'P', 'T', 'I', 'D', 'E']
        assert result == expected


class TestFragmentSeq:
    """Test cases for fragment_seq function"""
    
    def test_fragment_seq_b_ions(self):
        """Test fragment sequence for b-ions"""
        peptide = "PEPTIDE"
        
        # b3 should give first 3 amino acids
        seq, frag_info = fragment_seq(peptide, "b3_1")
        assert seq == ['P', 'E', 'P']
        assert frag_info == ['b', 3, '', '1']
        
        # b5 should give first 5 amino acids
        seq, frag_info = fragment_seq(peptide, "b5_2")
        assert seq == ['P', 'E', 'P', 'T', 'I']
        assert frag_info == ['b', 5, '', '2']
    
    def test_fragment_seq_y_ions(self):
        """Test fragment sequence for y-ions"""
        peptide = "PEPTIDE"
        
        # y3 should give last 3 amino acids
        seq, frag_info = fragment_seq(peptide, "y3_1")
        assert seq == ['I', 'D', 'E']
        assert frag_info == ['y', 3, '', '1']
        
        # y5 should give last 5 amino acids
        seq, frag_info = fragment_seq(peptide, "y5_2")
        assert seq == ['P', 'T', 'I', 'D', 'E']
        assert frag_info == ['y', 5, '', '2']
    
    def test_fragment_seq_with_modifications(self):
        """Test fragment sequence with modifications"""
        peptide = "PEP(+10.0)TIDE"
        
        # b4 should include the modification
        seq, frag_info = fragment_seq(peptide, "b4_1")
        assert seq == ['P', 'E', 'P(+10.0)', 'T']
        assert frag_info == ['b', 4, '', '1']
        
        # y5 should include the modification
        seq, frag_info = fragment_seq(peptide, "y5_2")
        assert seq == ['P(+10.0)', 'T', 'I', 'D', 'E']
        assert frag_info == ['y', 5, '', '2']
    
    def test_fragment_seq_with_losses(self):
        """Test fragment sequence with neutral losses"""
        peptide = "PEPTIDE"
        
        # Test with water loss
        seq, frag_info = fragment_seq(peptide, "b3-H2O_1")
        assert seq == ['P', 'E', 'P']
        assert frag_info == ['b', 3, 'H2O', '1']
        
        # Test with ammonia loss
        seq, frag_info = fragment_seq(peptide, "y4-NH3_2")
        assert seq == ['T', 'I', 'D', 'E']
        assert frag_info == ['y', 4, 'NH3', '2']
    
    def test_fragment_seq_invalid_ion_type(self):
        """Test invalid ion types raise ValueError"""
        peptide = "PEPTIDE"
        
        with pytest.raises(ValueError, match="Invalid ion type"):
            fragment_seq(peptide, "q3_1")  # z not in 'abc' or 'xyz'
        
        with pytest.raises(ValueError, match="Invalid ion type"):
            fragment_seq(peptide, "m3_1")  # m not valid
    
    def test_fragment_seq_index_out_of_range(self):
        """Test fragment index larger than peptide raises AssertionError"""
        peptide = "PEP"  # Only 3 amino acids
        
        with pytest.raises(AssertionError):
            fragment_seq(peptide, "b5_1")  # Asking for 5th amino acid
        
        with pytest.raises(AssertionError):
            fragment_seq(peptide, "y4_1")  # Asking for 4th from end

class TestBits1:
    """Test cases for bits1 function (binary conversion)"""
    
    def test_bits1_basic_numbers(self):
        """Test binary conversion for basic numbers"""
        assert bits1(0) == [0]
        assert bits1(1) == [1]
        assert bits1(2) == [1, 0]
        assert bits1(3) == [1, 1]
        assert bits1(4) == [1, 0, 0]
        assert bits1(5) == [1, 0, 1]
        assert bits1(7) == [1, 1, 1]
        assert bits1(8) == [1, 0, 0, 0]
    
    def test_bits1_powers_of_two(self):
        """Test powers of two specifically"""
        assert bits1(16) == [1, 0, 0, 0, 0]
        assert bits1(32) == [1, 0, 0, 0, 0, 0]
        assert bits1(64) == [1, 0, 0, 0, 0, 0, 0]
    
    def test_bits1_larger_numbers(self):
        """Test larger numbers"""
        assert bits1(15) == [1, 1, 1, 1]  # 2^4 - 1
        assert bits1(255) == [1, 1, 1, 1, 1, 1, 1, 1]  # 2^8 - 1
    
    @pytest.mark.parametrize("number,expected", [
        (0, [0]),
        (1, [1]),
        (10, [1, 0, 1, 0]),
        (31, [1, 1, 1, 1, 1]),
        (100, [1, 1, 0, 0, 1, 0, 0]),
    ])
    def test_bits1_parametrized(self, number, expected):
        """Parametrized tests for various numbers"""
        assert bits1(number) == expected

class TestCut:
    """Test cases for cut function (array trimming)"""
    
    def test_cut_basic_functionality(self):
        """Test basic array cutting functionality"""
        # Array where values drop below threshold
        array = np.array([0.1, 0.05, 0.02, 0.001, 0.0001])
        result = cut(array, tr=0.01)
        
        # Should cut after 0.02 (last value > 0.01)
        expected = array[:3]  # [0.1, 0.05, 0.02]
        np.testing.assert_array_equal(result, expected)
    
    def test_cut_default_threshold(self):
        """Test cutting with default threshold"""
        array = np.array([1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001])
        result = cut(array)  # Default threshold is 0.00001
        
        # Should include up to and including 0.00001
        expected = array[:6]
        np.testing.assert_array_equal(result, expected)
    
    def test_cut_all_above_threshold(self):
        """Test when all values are above threshold"""
        array = np.array([0.5, 0.3, 0.1, 0.05])
        result = cut(array, tr=0.01)
        
        # Should return entire array
        np.testing.assert_array_equal(result, array)
    
    def test_cut_all_below_threshold(self):
        """Test when no values are above threshold"""
        array = np.array([0.001, 0.0005, 0.0001])
        
        with pytest.raises(IndexError):
            cut(array, tr=0.01)  # No values > 0.01
    
    def test_cut_single_element(self):
        """Test cutting single element array"""
        array = np.array([0.1])
        result = cut(array, tr=0.05)
        np.testing.assert_array_equal(result, array)
        
        # Test when single element is below threshold
        with pytest.raises(IndexError):
            cut(array, tr=0.2)

class TestIsoDistr:
    """Test cases for iso_distr function (isotope distribution)"""
    
    def test_iso_distr_simple_molecule(self):
        """Test isotope distribution for simple molecules"""
        # Test with methane-like composition: [C, H, N, O, S] = [1, 4, 0, 0, 0]
        temp = [1, 4, 0, 0, 0]
        result = iso_distr(temp)
        
        # Should return a valid distribution
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64
        assert len(result) >= 1
        assert np.all(result >= 0)
        assert np.max(result) == 1.0  # Should be normalized to max = 1
    
    def test_iso_distr_no_atoms(self):
        """Test with no atoms"""
        temp = [0, 0, 0, 0, 0]
        result = iso_distr(temp)
        
        # Should still return valid result (probably [1.0])
        assert isinstance(result, np.ndarray)
        assert len(result) >= 1
    
    def test_iso_distr_realistic_peptide(self):
        """Test with realistic peptide-like composition"""
        # Approximate composition for a small peptide
        temp = [10, 15, 3, 5, 1]  # [C, H, N, O, S]
        result = iso_distr(temp)
        
        assert isinstance(result, np.ndarray)
        assert len(result) >= 2  # Should have multiple isotopic peaks
        assert np.max(result) == 1.0
        assert result[0] > 0  # Monoisotopic peak should have intensity
    
    def test_iso_distr_carbon_only(self):
        """Test with only carbon atoms"""
        temp = [5, 0, 0, 0, 0]  # 5 carbons
        result = iso_distr(temp)
        
        # Should show clear 13C isotope pattern
        assert len(result) >= 2
        assert result[0] > result[1]  # Monoisotopic > +1 isotope

class TestMyIsoDistr:
    """Test cases for my_iso_distr function"""
    
    def test_my_iso_distr_basic(self):
        """Test basic functionality"""
        comp = {"H": 10, "C": 5, "N": 1, "O": 2, "S": 0}
        result = my_iso_distr(comp)
        
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64
        assert len(result) >= 1
        assert np.all(result >= 0)
        assert np.max(result) == 1.0
    
    def test_my_iso_distr_vs_iso_distr(self):
        """Test that my_iso_distr gives same result as iso_distr"""
        # Same composition in both formats
        temp = [5, 10, 1, 2, 0]  # [C, H, N, O, S]
        comp = {"H": 10, "C": 5, "N": 1, "O": 2, "S": 0}
        
        result1 = iso_distr(temp)
        result2 = my_iso_distr(comp)
        
        # Should give same results (within numerical precision)
        np.testing.assert_array_almost_equal(result1, result2, decimal=10)
    
    def test_my_iso_distr_missing_elements(self):
        """Test behavior with missing elements"""
        # Missing sulfur - should raise KeyError
        comp = {"H": 5, "C": 3, "N": 0, "O": 1}
        
        with pytest.raises(KeyError):
            my_iso_distr(comp)
    
    def test_my_iso_distr_zero_composition(self):
        """Test with all zeros"""
        comp = {"H": 0, "C": 0, "N": 0, "O": 0, "S": 0}
        result = my_iso_distr(comp)
        
        assert isinstance(result, np.ndarray)
        # Should probably return [1.0] for empty composition

class TestGetSeqComp:
    """Test cases for get_seq_comp function"""
    
    #@pytest.mark.skip(reason="Requires pyteomics library")
    def test_get_seq_comp_simple_sequence(self):
        """Test getting composition for simple amino acid sequence"""
        # Simple sequence without modifications
        split_seq = ['E', 'Q', 'A', 'I', 'S', 'V', 'R']
        ion_type = 'y'
        
        result = get_seq_comp(split_seq, ion_type)
        
        # Expected format: should be a mass.Composition object
        # From the docstring example: Composition({'H': 59, 'C': 33, 'O': 12, 'N': 11})
        assert hasattr(result, '__getitem__')  # Composition acts like a dict
        assert 'H' in result
        assert 'C' in result
        assert 'O' in result
        assert 'N' in result
        
        # Rough expected values (user can correct these)
        assert result['H'] > 0
        assert result['C'] > 0
        assert result['O'] > 0
        assert result['N'] > 0
    
    
    #@pytest.mark.skip(reason="Requires pyteomics library")
    def test_get_seq_comp_with_unimod(self):
        """Test composition with Unimod modifications"""
        # Sequence with Unimod modification
        split_seq = ['E', 'Q(UniMod:35)', 'A', 'I']  # 35 = oxidation
        ion_type = 'b'
        
        result = get_seq_comp(split_seq, ion_type)
        
        # Should include the modification in the composition
        assert hasattr(result, '__getitem__')
        assert 'H' in result
        assert 'C' in result
        assert 'O' in result
        # Oxidation should add oxygen
        assert result['O'] > 4  # Should be higher due to modification
    
    #@pytest.mark.skip(reason="Requires pyteomics library")
    def test_get_seq_comp_different_ion_types(self):
        """Test different ion types affect composition"""
        split_seq = ['P', 'E', 'P', 'T']
        
        result_b = get_seq_comp(split_seq, 'b')
        result_y = get_seq_comp(split_seq, 'y')
        
        # Different ion types should give different compositions
        # (due to different N/C-terminal modifications)
        assert result_b != result_y
    
    #@pytest.mark.skip(reason="Requires pyteomics library")
    def test_get_seq_comp_no_modifications(self):
        """Test sequence with no modifications - verify exact composition"""
        split_seq = ['P', 'E', 'P', 'T', 'I', 'D', 'E']
        ion_type = 'b'
        
        result = get_seq_comp(split_seq, ion_type)
        
        # Verify exact elemental composition for PEPTIDE as b-ion
        # (User can correct these values with actual expected output)
        #verify with https://web.expasy.org/peptide_mass/
        expected_composition = {
            'C': 34,  # Total carbons in PEPTIDE b-ion
            'H': 51,  # Total hydrogens in PEPTIDE b-ion  
            'N': 7,   # Total nitrogens in PEPTIDE b-ion
            'O': 14   # Total oxygens in PEPTIDE b-ion
        }
        
        for element, expected_count in expected_composition.items():
            assert result[element] == expected_count, f"Expected {element}: {expected_count}, got {result[element]}"


class TestGenIsotopesDict:
    """Test cases for gen_isotopes_dict function"""
    
    #@pytest.mark.skip(reason="Requires pyteomics, brainpy, and config modules")
    def test_gen_isotopes_dict_simple(self):
        """Test isotope generation for simple fragment dictionary"""
        seq = 'ELYAQFLR'
        frags = {
            'b3_1': [406.19726206442, 0.12509464],
            'y3_1': [435.27143006419, 0.17212126],
        }

        peaks, frag_names = gen_isotopes_dict(seq, frags)
        
        # Check return types and shapes
        assert isinstance(peaks, np.ndarray)
        assert isinstance(frag_names, np.ndarray)
        assert peaks.shape[1] == 2  # Should have m/z and intensity columns
        assert peaks.shape[0] == frag_names.shape[0]  # Same number of rows
        
        # Should have more peaks than input (due to isotopes)
        assert len(peaks) > len(frags)
        
        # Fragment names should include isotopic variants
        assert any('_iso1' in name for name in frag_names)
        
        # Peaks should be sorted by m/z (first column)
        assert np.all(peaks[:-1, 0] <= peaks[1:, 0])
        
        # Expected rough output based on docstring:
        # Should have both monoisotopic and +1 isotope peaks
        expected_frags = [
            'b3_1', 'b3_1_iso1', 
            'y3_1', 'y3_1_iso1'
        ]
        assert set(frag_names) == set(expected_frags)
    
    #@pytest.mark.skip(reason="Requires external libraries")
    def test_gen_isotopes_dict_with_losses(self):
        """Test isotope generation with neutral losses"""
        seq = 'PEPTIDE'
        frags = {
            'b3_1': [300.1, 0.5],
            'y4-H2O_1': [400.2, 0.3]  # Fragment with water loss
        }
        
        peaks, frag_names = gen_isotopes_dict(seq, frags)
        
        # Should handle neutral losses correctly
        assert isinstance(peaks, np.ndarray)
        assert isinstance(frag_names, np.ndarray)
        assert len(peaks) > len(frags)
        
        # Should have isotopic variants of loss fragments
        loss_variants = [name for name in frag_names if 'H2O' in str(name)]
        assert len(loss_variants) >= 1
        expected_frags = [
            'b3_1', 'b3_1_iso1', 
            'y4-H2O_1', 'y4-H2O_1_iso1', 
        ]
        assert set(frag_names) == set(expected_frags)
    #@pytest.mark.skip(reason="Requires external libraries")
    def test_gen_isotopes_dict_empty_frags(self):
        """Test with empty fragment dictionary"""
        seq = 'PEPTIDE'
        frags = {}
        
        peaks, frag_names = gen_isotopes_dict(seq, frags)
        
        # Should return empty arrays
        assert isinstance(peaks, np.ndarray)
        assert isinstance(frag_names, np.ndarray)
        assert len(peaks) == 0
        assert len(frag_names) == 0
    
    #@pytest.mark.skip(reason="Requires external libraries")
    def test_gen_isotopes_dict_with_tag(self):
        """Test isotope generation with tags"""
        seq = 'K(tag6-0)PEPTIDE'
        frags = {'b3_1': [400.1, 0.5]}
        
        peaks, frag_names = gen_isotopes_dict(seq, frags, tag=mt.tag6)
        
        # Should handle tagged sequences
        assert isinstance(peaks, np.ndarray)
        assert isinstance(frag_names, np.ndarray)
        assert len(peaks) >= len(frags)
        print("peaks: ", peaks)
        print("frag_names: ", frag_names)

class TestIsoLibraryMulti:
    """Test cases for iso_library_multi function"""
    
    @pytest.mark.skip(reason="Requires external libraries and multiprocessing")
    def test_iso_library_multi_single_entry(self):
        """Test multiprocessing isotope addition for single library entry"""
        library = {
            ('AAAEQAISVR', 2.0): {
                'mod_seq': 'AAAEQAISVR',
                'seq': 'AAAEQAISVR',
                'prec_mz': 508.28018033366,
                'prec_z': 2.0,
                'iRT': 0.296130418777466,
                'frags': {
                    'y7_1': [802.44174284642, 1.0],
                    'b3_1': [214.1186178209, 0.48869178],
                    'y5_1': [545.34057225317, 0.35331511]
                },
                'protein_group': 'Q01780'
            }
        }
        
        result = iso_library_multi(library)
        
        # Should return same structure
        assert isinstance(result, dict)
        assert len(result) == len(library)
        
        # Should have same keys
        assert list(result.keys()) == list(library.keys())
        
        # Each entry should have new spectrum and ordered_frags
        key = ('AAAEQAISVR', 2.0)
        entry = result[key]
        
        assert 'spectrum' in entry
        assert 'ordered_frags' in entry
        
        # Spectrum should be numpy array with m/z and intensity
        spectrum = entry['spectrum']
        assert isinstance(spectrum, np.ndarray)
        assert spectrum.shape[1] == 2  # m/z and intensity columns
        
        # Should have more peaks than original fragments (isotopes added)
        assert len(spectrum) > len(library[key]['frags'])
        
        # Ordered_frags should be numpy array of strings
        ordered_frags = entry['ordered_frags']
        assert isinstance(ordered_frags, np.ndarray)
        assert len(ordered_frags) == len(spectrum)
        
        # Should include isotopic variants
        frag_names = [str(name) for name in ordered_frags]
        assert any('_iso1' in name for name in frag_names)
        
        # Original data should be preserved
        assert entry['mod_seq'] == library[key]['mod_seq']
        assert entry['prec_mz'] == library[key]['prec_mz']
        assert entry['frags'] == library[key]['frags']  # Original frags unchanged
    
    @pytest.mark.skip(reason="Requires external libraries and multiprocessing")
    def test_iso_library_multi_multiple_entries(self):
        """Test multiprocessing with multiple library entries"""
        library = {
            ('PEPTIDE', 2.0): {
                'seq': 'PEPTIDE',
                'frags': {'b3_1': [300.1, 0.5], 'y3_1': [400.2, 0.3]}
            },
            ('PROTEIN', 3.0): {
                'seq': 'PROTEIN', 
                'frags': {'b4_1': [500.3, 0.8], 'y5_2': [600.4, 0.6]}
            }
        }
        
        result = iso_library_multi(library)
        
        # Should process all entries
        assert len(result) == 2
        
        # Each entry should have isotopic peaks added
        for key in library.keys():
            assert 'spectrum' in result[key]
            assert 'ordered_frags' in result[key]
            assert len(result[key]['spectrum']) > len(library[key]['frags'])
    
    @pytest.mark.skip(reason="Requires external libraries and multiprocessing")
    def test_iso_library_multi_empty_library(self):
        """Test with empty library"""
        library = {}
        
        result = iso_library_multi(library)
        
        # Should return empty dict
        assert isinstance(result, dict)
        assert len(result) == 0
    
    @pytest.mark.skip(reason="Requires external libraries and multiprocessing")
    def test_iso_library_multi_preserves_structure(self):
        """Test that all original library data is preserved"""
        library = {
            ('TESTPEPTIDE', 1.0): {
                'mod_seq': 'TESTPEPTIDE',
                'seq': 'TESTPEPTIDE',
                'prec_mz': 123.456,
                'prec_z': 1.0,
                'iRT': 0.789,
                'frags': {'b5_1': [555.5, 0.9]},
                'protein_group': 'TEST_PROTEIN',
                'protein_name': 'Test Protein',
                'genes': 'TEST_GENE',
                'custom_field': 'custom_value'
            }
        }
        
        result = iso_library_multi(library)
        
        key = ('TESTPEPTIDE', 1.0)
        original = library[key]
        processed = result[key]
        
        # All original fields should be preserved
        for field in ['mod_seq', 'seq', 'prec_mz', 'prec_z', 'iRT', 
                     'protein_group', 'protein_name', 'genes', 'custom_field']:
            assert processed[field] == original[field]
        
        # Original frags should be unchanged
        assert processed['frags'] == original['frags']
        
        # New fields should be added
        assert 'spectrum' in processed
        assert 'ordered_frags' in processed
        assert 'spectrum' not in original
        assert 'ordered_frags' not in original


# Integration test placeholder for functions requiring external dependencies
class TestIntegrationFunctions:
    """Placeholder for functions that require external libraries"""
    
    @pytest.mark.skip(reason="Requires pyteomics and brainpy libraries")
    def test_get_seq_comp_integration(self):
        """Real integration test for get_seq_comp when libraries are available"""
        # This would test with real pyteomics calls
        pass
    
    @pytest.mark.skip(reason="Requires external libraries")
    def test_gen_isotopes_dict_integration(self):
        """Real integration test for gen_isotopes_dict"""
        # This would test with real data and verify actual isotope calculations
        pass
    
    @pytest.mark.skip(reason="Requires multiprocessing testing")
    def test_iso_library_multi_integration(self):
        """Real integration test for multiprocessing function"""
        # This would test with small real library data
        pass

if __name__ == "__main__":
    pytest.main([__file__])