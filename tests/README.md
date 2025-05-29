# JMod Test Suite

This directory contains the test suite for JMod

## Setup

First, install the test dependencies:

```bash
pip install -r tests/requirements.txt
```

## Running Tests

There are several ways to run the tests:

### 1. Using the test runner script

```bash
# Run all tests
python run_tests.py

# Run with verbose output
python run_tests.py -v

# Run with coverage report (currently broken)
python run_tests.py -c

# Run only dummy tests
python run_tests.py --dummy

# Run only miscFunctions tests
python run_tests.py --misc

# Run a specific test file
python run_tests.py tests/test_dummy.py

# Run a specific test
python run_tests.py tests/test_miscFunctions.py::TestChangeSeq::test_change_seq_diann_simple
```

### 2. Using JMod in test mode

```bash
# Run all tests
python run_jmod.py --test

# Run with options
python run_jmod.py --test -v
python run_jmod.py --test --coverage
```

### 3. Using pytest directly

```bash
# Run all tests
pytest tests/

# Run with verbose output
pytest -v tests/

# Run with coverage
pytest --cov=. --cov-report=html tests/

# Run a specific test class
pytest tests/test_miscFunctions.py::TestChangeSeq

# Run tests matching a pattern
pytest -k "change_seq" tests/
```

## Test Structure

```
tests/
├── __init__.py              # Makes tests a Python package
├── conftest.py              # Pytest configuration and shared fixtures
├── test_dummy.py            # Simple tests to verify setup
├── test_miscFunctions.py   # Tests for miscFunctions module
├── requirements.txt         # Test dependencies
└── README.md               # This file
```

## Writing New Tests

1. Create a new test file named `test_<module_name>.py`
2. Import the functions you want to test
3. Create test functions starting with `test_`
4. Use pytest fixtures from `conftest.py` as needed

Example:

```python
import pytest
from my_module import my_function

def test_my_function():
    result = my_function("input")
    assert result == "expected_output"

def test_my_function_with_fixture(sample_sequences):
    # sample_sequences is defined in conftest.py
    result = my_function(sample_sequences['simple'])
    assert result is not None
```

## Test Coverage

To generate a coverage report:

```bash
python run_tests.py -c
```

This will:
- Display coverage in the terminal
- Generate an HTML report in `htmlcov/index.html`

## Continuous Integration

You can integrate these tests with CI/CD systems like GitHub Actions, GitLab CI, or Jenkins by running:

```bash
python run_jmod.py --test
```

## Troubleshooting

### Import Errors

If you get import errors when running tests, ensure:
1. You're running tests from the project root directory
2. The project root is in your PYTHONPATH
3. All dependencies are installed

### Mock Config Issues

The test suite automatically mocks the `config` module to avoid dependency issues. If you need to test with specific config values, modify the `mock_config` fixture in `conftest.py`.

## Current Test Coverage

- ✅ `change_seq` function (comprehensive tests)
- ✅ `parse_peptide` function
- ✅ `extract_mod` function
- ✅ `split_frag_name` function
- ✅ `closest_peak_diff` function
- ✅ `frag_to_peak` function
- ✅ `within_tol` function

## TODO

- [ ] Add tests for `convert_frags` function
- [ ] Add tests for `RTAlignment` module
- [ ] Add tests for `SpecLib` module
- [ ] Add integration tests
- [ ] Add performance benchmarks