# Load Files Test Suite

This directory contains comprehensive tests for the `src/utils/io/load_files.py` module and its spectrum classes.

## Test Structure

### Core Classes Tested
- `BaseSpectrum` - Abstract base class
- `BaseSpectrumFile` - Abstract base class  
- `MzMLSpectrum` - Concrete mzML spectrum implementation
- `MzMLSpectrumFile` - Concrete mzML file handler
- `loadSpectra()` - Utility function

### Test Categories

#### 1. Unit Tests
- **TestBaseSpectrum**: Abstract class validation
- **TestBaseSpectrumFile**: Abstract class validation
- **TestMzMLSpectrum**: Individual spectrum functionality
- **TestMzMLSpectrumFile**: File handling functionality
- **TestLoadSpectraFunction**: Utility function testing

#### 2. Integration Tests
- **TestIntegration**: End-to-end workflow testing
- Data consistency validation
- Real file processing

#### 3. Performance Tests
- **TestPerformance**: Loading speed and memory usage

## Test Data

The tests use a minimal test mzML file located at:
```
/Users/nathanwamsley/Projects/JMod-1/data/test_sample.mzml
```

This file contains:
- 3 MS1 scans
- 5 MS2 scans
- ~5,568 total peaks
- Size: 153KB (reduced from 339MB original)

## Running Tests

### Basic Usage
```bash
# Run all tests
python -m pytest tests/io/test_load_files.py -v

# Run specific test class
python -m pytest tests/io/test_load_files.py::TestMzMLSpectrum -v

# Run with coverage
python -m pytest tests/io/test_load_files.py --cov=src.utils.io.load_files --cov-report=html
```

### Using Test Runner
```bash
# Run all tests with verbose output
python run_io_tests.py --verbose

# Run specific test class
python run_io_tests.py --pattern TestMzMLSpectrum

# Run with coverage report
python run_io_tests.py --coverage

# List available test classes
python run_io_tests.py --list-tests
```

## Test Coverage

The test suite covers:

### ✅ **Core Functionality**
- Spectrum creation and initialization
- File loading and parsing
- Data access methods (`get_by_idx`, `peak_list`)
- Error handling for malformed data

### ✅ **Data Validation**
- MS1 and MS2 spectrum properties
- Required attribute presence
- Data type consistency
- Array shape validation

### ✅ **Error Handling**
- Missing required fields
- Corrupted scan data
- File not found errors
- Legacy pickle file compatibility

### ✅ **Integration Testing**
- Real mzML file loading
- End-to-end data access
- Pickle file creation and loading
- Cross-component compatibility

### ✅ **Performance Testing**
- Loading speed benchmarks
- Memory usage validation
- File size handling

## Key Test Features

1. **Real Data Testing**: Uses actual mzML data from mass spectrometry experiments
2. **Backwards Compatibility**: Tests legacy pickle file handling
3. **Error Resilience**: Validates graceful error handling
4. **Performance Monitoring**: Ensures reasonable loading times
5. **Mock Testing**: Uses mocks for isolated unit testing

## Test Requirements

```python
pytest>=7.0.0
pytest-cov>=4.0.0  # For coverage reports
```

## Adding New Tests

When adding new functionality to `load_files.py`:

1. **Add unit tests** for new methods/classes
2. **Update integration tests** if workflow changes
3. **Add error handling tests** for new error conditions
4. **Consider performance impact** and add benchmarks if needed

### Example Test Structure
```python
class TestNewFeature:
    """Test cases for new feature"""
    
    def test_basic_functionality(self):
        """Test basic operation"""
        # Test implementation here
        pass
    
    def test_error_handling(self):
        """Test error conditions"""
        # Test error cases here
        pass
    
    @pytest.mark.skipif(not os.path.exists(TEST_MZML_FILE), 
                        reason="Test mzML file not found")
    def test_with_real_data(self):
        """Test with real mzML file"""
        # Integration test here
        pass
```

## Continuous Integration

These tests are designed to run in CI environments and will:
- Skip tests requiring the test mzML file if not present
- Handle temporary file creation/cleanup
- Provide clear error messages for debugging
- Run efficiently (complete suite < 5 seconds)

## Troubleshooting

### Common Issues

1. **Test file not found**: Ensure `data/test_sample.mzml` exists
2. **Import errors**: Run tests from project root directory
3. **Pickle errors**: Clear old pickle files: `rm data/*_pythonspec`
4. **Permission errors**: Ensure write permissions for temp files

### Debug Mode
```bash
# Run with detailed output
python -m pytest tests/io/test_load_files.py -v -s

# Run single test with debugging
python -m pytest tests/io/test_load_files.py::TestMzMLSpectrum::test_empty_spectrum_creation -v -s
```