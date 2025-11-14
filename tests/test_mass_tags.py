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
import numpy as np

# Import the functions we want to test
from src.mass_tags import (
    get_tag_pos
)

class TestGetTagPos:
    """Tests for get_tag_pos function"""

    def test_single_aa_rule(self):
        """Single AA rule in sequence"""
        seq = ["A", "C", "D"]
        rules = "A"
        positions, masses = get_tag_pos(seq, rules)
        assert positions == [0]
        assert np.array_equal(masses, np.array([1, 0, 0]))

    def test_multiple_aa_rule(self):
        """Rule matches multiple amino acids"""
        seq = ["A", "C", "A", "D", "C"]
        rules = "AC"
        positions, masses = get_tag_pos(seq, rules)
        # 'A' at 0 and 2, 'C' at 1 and 4
        assert sorted(positions) == [0, 1, 2, 4]
        assert np.array_equal(masses, np.array([1, 1, 1, 0, 1]))

    def test_n_terminal_rule(self):
        """N-terminal tagging"""
        seq = ["A", "C", "D"]
        rules = "n"
        positions, masses = get_tag_pos(seq, rules)
        assert positions == [0]
        assert np.array_equal(masses, np.array([1, 0, 0]))

    def test_combined_rules(self):
        """Combination of AA and n-terminal rules"""
        seq = ["A", "C", "D", "A"]
        rules = "nA"
        positions, masses = get_tag_pos(seq, rules)
        # n=0, A at 0 and 3 => positions 0,0,3 => duplicates counted in masses
        assert sorted(positions) == [0, 0, 3]
        assert np.array_equal(masses, np.array([2, 0, 0, 1]))

    def test_sequence_with_modifications(self):
        """AA with modifications should still match the first character"""
        seq = ["A(+15.99)", "C", "D"]
        rules = "A"
        positions, masses = get_tag_pos(seq, rules)
        assert positions == [0]
        assert np.array_equal(masses, np.array([1, 0, 0]))

    def test_empty_sequence(self):
        """Empty sequence should return empty results"""
        seq = []
        rules = "A"
        positions, masses = get_tag_pos(seq, rules)
        assert positions == []
        assert np.array_equal(masses, np.array([]))

    def test_rule_not_in_sequence(self):
        """Rule not present in sequence"""
        seq = ["A", "C", "D"]
        rules = "X"
        positions, masses = get_tag_pos(seq, rules)
        assert positions == []
        assert np.array_equal(masses, np.array([0, 0, 0]))

    def test_invalid_rule(self):
        """Invalid rule should raise ValueError"""
        seq = ["A", "C", "D"]
        rules = "1"  # not A-Z or 'n'
        with pytest.raises(ValueError, match="Unknown Tag Rule"):
            get_tag_pos(seq, rules)

    def test_multiple_occurrences_and_n_terminal(self):
        """Multiple occurrences of the same AA and n-terminal together"""
        seq = ["A", "C", "A", "D"]
        rules = "nA"
        positions, masses = get_tag_pos(seq, rules)
        # positions: n=0, A at 0 and 2 => positions [0,0,2]
        # masses: positions 0 counted twice, 2 counted once
        assert sorted(positions) == [0, 0, 2]
        assert np.array_equal(masses, np.array([2, 0, 1, 0]))
