import unittest
import re
from miscFunctions import change_seq

class TestChangeSeq(unittest.TestCase):
    def test_string_input_reverse(self):
        """Test reverse rule with a string input"""
        seq = "PEPTIDE"
        result = change_seq(seq, "rev")
        self.assertEqual(result, "DITPEPE")  # Reverse all but last letter

    def test_list_input_reverse(self):
        """Test reverse rule with a list input"""
        seq = ["P", "E", "P", "T", "I", "D", "E"]
        result = change_seq(seq, "rev")
        self.assertEqual(result, "DITPEPE")

    def test_modified_peptide_reverse(self):
        """Test reverse rule with modifications"""
        seq = "PEP(mod)TIDE"
        result = change_seq(seq, "rev")
        # Should strip modifications when reversing
        self.assertEqual(result, "DITP(mod)EPE")
    
    def test_modified_peptide_reverse(self):
        """Test reverse rule with modifications"""
        seq = "P(mod)EPREM(mod)TIDE"
        result = change_seq(seq, "rev")
        # Should strip modifications when reversing
        self.assertEqual(result, "DITM(mod)ERPEP(mod)E")

    def test_diann_rules(self):
        """Test diann rules conversion"""
        seq = "PEPTIDE"
        result = change_seq(seq, "diann")
        self.assertTrue(len(result) == len(seq))

    def test_invalid_rules(self):
        """Test that invalid rules raise ValueError"""
        seq = "PEPTIDE"
        with self.assertRaises(ValueError):
            change_seq(seq, "invalid_rule")

if __name__ == '__main__':
    unittest.main()