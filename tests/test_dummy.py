"""
This Source Code Form is subject to the terms of the Oxford Nanopore
Technologies, Ltd. Public License, v. 1.0.  Full licence can be found
at https://github.com/ParallelSquared/JMod/blob/main/LICENSE.txt
"""

"""
Dummy tests to verify the test suite is working correctly
"""
import pytest


def test_dummy_always_passes():
    """A simple test that always passes to verify pytest is working"""
    assert True


def test_dummy_basic_math():
    """Test basic math operations"""
    assert 2 + 2 == 4
    assert 10 - 5 == 5
    assert 3 * 4 == 12
    assert 10 / 2 == 5


def test_dummy_string_operations():
    """Test basic string operations"""
    assert "hello" + " " + "world" == "hello world"
    assert "PEPTIDE".lower() == "peptide"
    assert len("JMOD") == 4


def test_dummy_list_operations():
    """Test basic list operations"""
    test_list = [1, 2, 3]
    assert len(test_list) == 3
    assert test_list[0] == 1
    assert test_list[-1] == 3
    test_list.append(4)
    assert len(test_list) == 4


class TestDummyClass:
    """A dummy test class to verify class-based tests work"""
    
    def test_class_method(self):
        """Test method in a test class"""
        assert isinstance(self, TestDummyClass)
    
    def test_with_fixture(self, sample_sequences):
        """Test that uses a fixture from conftest.py"""
        assert 'simple' in sample_sequences
        assert sample_sequences['simple'] == 'PEPTIDE'


def test_dummy_exception():
    """Test that exceptions are raised correctly"""
    with pytest.raises(ValueError):
        raise ValueError("This is a test exception")


@pytest.mark.parametrize("input_val,expected", [
    (1, 2),
    (2, 4),
    (3, 6),
    (4, 8)
])
def test_dummy_parametrized(input_val, expected):
    """Test parametrization feature of pytest"""
    assert input_val * 2 == expected