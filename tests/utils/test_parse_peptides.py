"""
This Source Code Form is subject to the terms of the Oxford Nanopore
Technologies, Ltd. Public License, v. 1.0.  Full licence can be found
at https://github.com/ParallelSquared/JMod/blob/main/LICENSE.txt
"""

"""
Tests for functions in misc_functions.py
"""
import pytest
from unittest.mock import Mock, patch
import numpy as np
import math

import src.config as config


# Import the functions we want to test
from src.utils.parse_peptides import (
    change_seq, convert_prec_mz, convert_frags, parse_peptide, extract_mod, split_frag_name
)

class TestChangeSeq:
    """Test cases for the change_seq function"""
    
    def test_change_seq_diann_simple(self):
        """Test change_seq with DIANN rules on simple sequences"""
        # Test simple sequence
        result = change_seq("PEPTIDE", "diann")
        assert result == "LDLSVED"
        
        # Test with single amino acid
        result = change_seq("A", "diann")
        assert result == "L"
        
        # Test with all unique conversions
        result = change_seq("GAVLIFMPWSTCYHKRQEND", "diann")
        assert result == "LLLVVLLLLTSSSSLLNDQE"
    
    def test_change_seq_reverse_simple(self):
        """Test change_seq with reverse rules"""
        # Test simple reversal - reverses all but last AA
        result = change_seq("PEPTIDE", "rev")
        assert result == "DITPEPE"  # Reverses PEPTID, keeps E at end
        
        # Test with two amino acids
        result = change_seq("KR", "rev")
        assert result == "KR"  # K reversed is K, R stays at end
        
        # Test with single amino acid
        result = change_seq("A", "rev")
        assert result == "A"
    
    def test_convert_frags_shuffle_masses(self):
        """Test that fragment m/z values are correctly recomputed after shuffle."""
        from pyteomics import mass as pmass

        seq = "AAAEQAISVR"
        frags = {
            'b3_1': [214.1186178209, 0.5],
            'y3_1': [361.21939449133, 0.5],
            'y7_1': [802.44174284642, 1.0],
        }
        result = convert_frags(seq, frags, "shuffle")

        # Get the shuffled sequence to verify masses independently
        shuffled = change_seq(seq, "shuffle")
        shuffled_tokens = parse_peptide(shuffled)
        unmod = [t[0] for t in shuffled_tokens]

        # Verify each fragment mass against direct computation
        for frag_key, (mz, intensity) in result.items():
            frag_type, frag_idx, loss, frag_z = split_frag_name(frag_key)
            if frag_type == "b":
                expected = pmass.fast_mass(unmod[:frag_idx], "b", int(frag_z))
            else:
                expected = pmass.fast_mass(unmod[-frag_idx:], "y", int(frag_z))
            assert abs(mz - expected) < 1e-6, f"{frag_key}: {mz} != {expected}"
            # Intensities must be preserved
            assert intensity == frags[frag_key][1]

    def test_convert_frags_shuffle_tagged_channel_delta(self):
        """Test that tagged shuffle produces correct mass deltas between channels.

        For each fragment, the m/z difference between channels should equal
        (n_tags_in_fragment * channel_delta) / charge.
        """
        from src.mass_tags import massTag

        tag = massTag(
            rules="nK",
            base_mass=140.0949630177,
            delta=[0.0, 4.0070994, 8.0141988132],
            channel_names=["0", "4", "8"],
            name="mTRAQ",
        )

        base_seq = "KPEPTIKR"
        frags = {
            'b3_1': [1.0, 0.5],  # dummy m/z — will be recomputed
            'y3_1': [1.0, 0.5],
            'y6_1': [1.0, 1.0],
        }

        # Build tagged sequences for channels 0 and 4
        tagged_0 = "K(mTRAQ-0)PEPTIK(mTRAQ-0)R"
        tagged_4 = "K(mTRAQ-4)PEPTIK(mTRAQ-4)R"

        result_0 = convert_frags(tagged_0, frags, "shuffle", tag=tag)
        result_4 = convert_frags(tagged_4, frags, "shuffle", tag=tag)

        # Tags stay at their original positions (0 and 6) after shuffle,
        # NOT recomputed from nK rules on the shuffled sequence.
        # Original: K(mTRAQ)PEPTIK(mTRAQ)R → tags at positions 0 and 6
        tag_positions = np.zeros(len(base_seq))
        tag_positions[0] = 1  # n-term K
        tag_positions[6] = 1  # second K
        n_tags_n = np.cumsum(tag_positions, dtype=int)
        n_tags_c = np.cumsum(tag_positions[::-1], dtype=int)

        channel_delta = 4.0070994  # delta between channel 0 and 4

        for frag_key in frags:
            frag_type, frag_idx, loss, frag_z = split_frag_name(frag_key)
            if frag_type in "abc":
                n_tags = n_tags_n[frag_idx - 1]
            else:
                n_tags = n_tags_c[frag_idx - 1]

            expected_delta = (n_tags * channel_delta) / int(frag_z)
            actual_delta = result_4[frag_key][0] - result_0[frag_key][0]
            assert abs(actual_delta - expected_delta) < 1e-6, (
                f"{frag_key}: delta {actual_delta} != expected {expected_delta} "
                f"(n_tags={n_tags})"
            )

    def test_change_seq_shuffle_preserves_mods(self):
        """Test that non-tag modifications stay attached to their AA after shuffle."""
        seq = "PEC(+57.02)PTIM(+15.99)R"
        result = change_seq(seq, "shuffle")
        tokens = parse_peptide(result)

        # C-term R stays fixed
        assert tokens[-1] == "R"

        # Every C must still carry (+57.02), every M must still carry (+15.99)
        for token in tokens:
            if token[0] == "C":
                assert "(+57.02)" in token, f"C lost its mod: {token}"
            if token[0] == "M":
                assert "(+15.99)" in token, f"M lost its mod: {token}"

        # Composition of bare AAs must be preserved
        orig_tokens = parse_peptide(seq)
        assert sorted(t[0] for t in tokens) == sorted(t[0] for t in orig_tokens)

    def test_change_seq_shuffle_deterministic(self):
        """Test that shuffle produces the same result on repeated calls."""
        r1 = change_seq("PEPTIDE", "shuffle")
        r2 = change_seq("PEPTIDE", "shuffle")
        assert r1 == r2
        assert r1 != "PEPTIDE"

    def test_change_seq_invalid_rules(self):
        """Test change_seq with invalid rules raises ValueError"""
        with pytest.raises(ValueError, match="Unavailable rules selected"):
            change_seq("PEPTIDE", "invalid_rule")
        
        with pytest.raises(ValueError, match="Unavailable rules selected"):
            change_seq("PEPTIDE", None)
    
    def test_change_seq_with_tags(self):
        """Test change_seq with tagged sequences"""

        mock_tag = Mock()
        mock_tag.name = "mTRAQ"

        # Test with tagged sequence
        result = change_seq("K(mTRAQ)PEPTIDE", "diann", tag=mock_tag)
        assert result == "L(mTRAQ)LDLSVED"

        # Test reverse with tags
        result = change_seq("K(mTRAQ)PEPTIDE", "rev", tag=mock_tag)
        assert result == "D(mTRAQ)ITPEPKE"

    def test_change_seq_invalid_aa_keyerror(self):
        """Test that change_seq raises KeyError for unknown amino acids with diann rules"""
        with pytest.raises(KeyError, match="'X'"):
            change_seq("PQRSTVWXY", "diann")

    def test_change_seq_list_input(self):
       """Test change_seq with list input instead of string"""
       # Create a list that mimics parsed peptide
       seq_list = ['P', 'E', 'P', 'T', 'I', 'D', 'E']
       result = change_seq(seq_list, "diann")
       assert result == "LDLSVED"


class TestPrecMz:
    """Test cases for calculating precursor m/z ratios function"""
    def test_convert_prec_mz(self):
        """Test converting precursor m/z ratios"""
        # Test with a simple m/z value
        #Get precursor m/zs from Skyline for testing purposes 
        result = convert_prec_mz("PEPTIDE", 2)
        assert abs(result - 400.68725848012497) < 1e-6

        #Since no mods dict is specified the mod is ignored 
        result = convert_prec_mz("PEPTIDE(mTRAQ-0)", 2)
        assert abs(result - 400.68725848012497) < 1e-6

        #Now includes the mod mass 
        result = convert_prec_mz("PEPTIDE(mTRAQ-0)", 2, {"mTRAQ-0": 144.102063})
        assert abs(result - 400.68725848012497 - 144.102063/2) < 1e-6
        seq = 'AAAEQAISVR'
        frags = {
                        'b3_1': [214.1186178209, 0.48869178],
                        'b4_1': [343.16121090887, 0.29596102],
                        'b5_1': [471.21978841415, 0.13230422],
                        'b6_1': [542.25690219886, 0.1957216],
                        'y3_1': [361.21939449133, 0.51684165],
                        'y4_1': [474.30345846846, 0.22480424],
                        'y5_1': [545.34057225317, 0.35331511],
                        'y6_1': [673.39914975845, 0.5639957],
                        'y7_1': [802.44174284642, 1.0],
                        'y8_1': [873.47885663113, 0.8503218],
                        'y9_1': [944.51597041584, 0.49269193]
                    }
        expected_new_frags = {
            'b3_1': [300.19178276116, 0.48869178],
            'b4_1': [371.22889654587004, 0.29596102],
            'b5_1': [499.28747405115, 0.13230422],
            'b6_1': [628.3300671391199, 0.1957216],
            'y3_1': [317.19317974349, 0.51684165],
            'y4_1': [388.2302935282, 0.22480424],
            'y5_1': [517.2728866161699, 0.35331511],
            'y6_1': [645.3314641214499, 0.5639957],
            'y7_1': [716.3685779061599, 1.0],
            'y8_1': [829.4526418832899, 0.8503218],
            'y9_1': [916.4846702875599, 0.49269193]
        }
        
        result = convert_frags(seq, frags, "rev")
        expected = expected_new_frags

        for key in result:
            assert math.isclose(result[key][0], expected[key][0], rel_tol=1e-5, abs_tol=1e-8)
            assert math.isclose(result[key][1], expected[key][1], rel_tol=1e-5, abs_tol=1e-8)


class TestParsePeptide:
    """Test cases for the parse_peptide function"""
    
    def test_parse_simple_sequence(self):
        """Test parsing simple peptide sequences"""
        result = parse_peptide("PEPTIDE")
        assert result == ['P', 'E', 'P', 'T', 'I', 'D', 'E']
    
    def test_parse_with_modifications(self):
        """Test parsing peptides with modifications"""
        result = parse_peptide("PEP(+79.97)TIDE")
        assert result == ['P', 'E', 'P(+79.97)', 'T', 'I', 'D', 'E']
        
        result = parse_peptide("K(mTRAQ)PEPTIDE(+15.99)R")
        assert result == ['K(mTRAQ)', 'P', 'E', 'P', 'T', 'I', 'D', 'E(+15.99)', 'R']
    
    def test_parse_with_square_brackets(self):
        """Test parsing peptides with square bracket modifications"""
        result = parse_peptide("PEP[+80]TIDE")
        assert result == ['P', 'E', 'P[+80]', 'T', 'I', 'D', 'E']
    
    def test_parse_empty_string(self):
        """Test parsing empty string"""
        result = parse_peptide("")
        assert result == []
    
    def test_parse_single_amino_acid(self):
        """Test parsing single amino acid"""
        result = parse_peptide("A")
        assert result == ['A']
        
        result = parse_peptide("A(+15.99)")
        assert result == ['A(+15.99)']

class TestExtractMod:
    """Test cases for the extract_mod function"""
    
    def test_extract_mod_simple(self):
        """Test extracting modifications from amino acids"""
        result = extract_mod("K(mTRAQ)")
        assert result == ["(mTRAQ)"]
        
        result = extract_mod("E(+15.99)")
        assert result == ["(+15.99)"]

        result = extract_mod("K(mTRAQ)(+42)")
        assert result == ['(mTRAQ)', '(+42)']
    
    def test_extract_mod_no_modification(self):
        """Test extracting from unmodified amino acid"""
        result = extract_mod("K")
        assert result == []
    
    def test_extract_mod_square_brackets(self):
        """Test extracting square bracket modifications"""
        result = extract_mod("C[+57.02]")
        assert result == ["[+57.02]"]

class TestSplitFragName:
    """Test cases for the split_frag_name function"""
    
    def test_split_simple_fragment(self):
        """Test splitting simple fragment names without neutral losses"""
        # b-ion examples
        frag_type, frag_idx, loss, frag_z = split_frag_name("b5_2")
        assert frag_type == "b"
        assert frag_idx == 5
        assert loss == ""
        assert frag_z == "2"
        
        # y-ion examples
        frag_type, frag_idx, loss, frag_z = split_frag_name("y10_1")
        assert frag_type == "y"
        assert frag_idx == 10
        assert loss == ""
        assert frag_z == "1"
    
    def test_split_fragment_with_loss(self):
        """Test splitting fragment names with neutral losses"""
        # Water loss
        frag_type, frag_idx, loss, frag_z = split_frag_name("y10-H2O_1")
        assert frag_type == "y"
        assert frag_idx == 10
        assert loss == "H2O"
        assert frag_z == "1"
        
        # Ammonia loss
        frag_type, frag_idx, loss, frag_z = split_frag_name("b3-NH3_2")
        assert frag_type == "b"
        assert frag_idx == 3
        assert loss == "NH3"
        assert frag_z == "2"
        
    def test_split_fragment_various_ion_types(self):
        """Test various fragment ion types (a, b, c, x, y, z)"""
        # a-ion
        frag_type, frag_idx, loss, frag_z = split_frag_name("a3_1")
        assert frag_type == "a" and frag_idx == 3
        
        # b-ion
        frag_type, frag_idx, loss, frag_z = split_frag_name("b3_1")
        assert frag_type == "b" and frag_idx == 3
        
        # c-ion
        frag_type, frag_idx, loss, frag_z = split_frag_name("c3_1")
        assert frag_type == "c" and frag_idx == 3
        
        # x-ion
        frag_type, frag_idx, loss, frag_z = split_frag_name("x3_1")
        assert frag_type == "x" and frag_idx == 3
        
        # y-ion
        frag_type, frag_idx, loss, frag_z = split_frag_name("y7_2")
        assert frag_type == "y" and frag_idx == 7
        
        # z-ion
        frag_type, frag_idx, loss, frag_z = split_frag_name("z5_1")
        assert frag_type == "z" and frag_idx == 5
    
    def test_split_fragment_various_charges(self):
        """Test fragments with different charge states"""
        # Charge 1
        frag_type, frag_idx, loss, frag_z = split_frag_name("b5_1")
        assert frag_z == "1"
        
        # Charge 2
        frag_type, frag_idx, loss, frag_z = split_frag_name("y8_2")
        assert frag_z == "2"
        
        # Charge 3
        frag_type, frag_idx, loss, frag_z = split_frag_name("b12_3")
        assert frag_z == "3"
        
        # Higher charges
        frag_type, frag_idx, loss, frag_z = split_frag_name("y15_4")
        assert frag_z == "4"
    
    def test_split_fragment_edge_cases(self):
        """Test edge cases and potential problem inputs"""
        # Minimum index
        frag_type, frag_idx, loss, frag_z = split_frag_name("b1_1")
        assert frag_idx == 1
        
        # Loss with special characters (if supported)
        frag_type, frag_idx, loss, frag_z = split_frag_name("y5-98_1")
        assert loss == "98"  # Numeric loss (like -98 Da)

