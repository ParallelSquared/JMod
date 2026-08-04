# JMod Test Suite

All JMod tests can be run with the ```pytest``` Python package. This should already be installed with the initial UV set up.


```bash
# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run a specific test class
pytest tests/test_misc_functions.py::TestChangeSeq

# Run a specific test function in a class
pytest tests/utils/test_misc_functions.py::TestWithinTol::test_within_tol_exact_match

# Run with coverage report
pytest --cov=. --cov-report=html tests/
```

## Test Structure

```
tests/
├── MassTags/                # Houses dummy mass tags used in the tests
├── fixtures/                # Houses dummy test data used in the tests
├── modesl/speclib/          # Tests functions inside src/models/spec_lib/
├── utils/                   # Tests functions inside src/utils/
├── conftest.py              # Pytest configuration and shared fixtures
├── test_[file_to_test].py   # Tests for misc_functions module
└── README.md                # This file
```

## Troubleshooting

### Import Errors

If you get import errors when running tests, ensure:
1. You're running tests from the project root directory
2. The project root is in your PYTHONPATH
3. The UV environment is installed correctly 


