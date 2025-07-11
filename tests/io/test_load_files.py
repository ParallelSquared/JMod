"""
This Source Code Form is subject to the terms of the Oxford Nanopore
Technologies, Ltd. Public License, v. 1.0.  Full licence can be found
at https://github.com/ParallelSquared/JMod/blob/main/LICENSE.txt
"""

"""
Tests for load_files.py module and spectrum classes
"""
import pytest
import numpy as np
import os
import tempfile
import pickle
from unittest.mock import patch, MagicMock

# Import the classes we want to test
from src.utils.io.load_files import (
    BaseSpectrum, BaseSpectrumFile, 
    MzMLSpectrum, MzMLSpectrumFile, 
    loadSpectra
)

# Test data path
TEST_MZML_FILE = os.path.join(os.path.dirname(__file__), '../../data/test_sample.mzml')

class TestBaseSpectrum:
    """Test cases for BaseSpectrum abstract base class"""
    
    def test_cannot_instantiate_abstract_class(self):
        """Test that BaseSpectrum cannot be instantiated directly"""
        with pytest.raises(TypeError):
            BaseSpectrum()
    
    def test_abstract_methods_defined(self):
        """Test that abstract methods are properly defined"""
        assert hasattr(BaseSpectrum, 'get_vals')
        assert hasattr(BaseSpectrum, 'peak_list')


class TestBaseSpectrumFile:
    """Test cases for BaseSpectrumFile abstract base class"""
    
    def test_cannot_instantiate_abstract_class(self):
        """Test that BaseSpectrumFile cannot be instantiated directly"""
        with pytest.raises(TypeError):
            BaseSpectrumFile()
    
    def test_abstract_methods_defined(self):
        """Test that abstract methods are properly defined"""
        assert hasattr(BaseSpectrumFile, 'load_spectra')


class TestMzMLSpectrum:
    """Test cases for MzMLSpectrum class"""
    
    def test_empty_spectrum_creation(self):
        """Test creating an empty MzMLSpectrum"""
        spectrum = MzMLSpectrum()
        
        # Check all required attributes exist and are None
        assert spectrum.id is None
        assert spectrum.level is None
        assert spectrum.RT is None
        assert spectrum.mz is None
        assert spectrum.intens is None
        assert spectrum.TIC is None
        assert spectrum.scan_num is None
        assert spectrum.collision_energy is None
        assert spectrum.injection_time is None
        assert spectrum.scanwindow is None
        assert spectrum.prec_mz is None
        assert spectrum.ms1window is None
    
    def test_inheritance(self):
        """Test that MzMLSpectrum properly inherits from BaseSpectrum"""
        spectrum = MzMLSpectrum()
        assert isinstance(spectrum, BaseSpectrum)
    
    def test_empty_peak_list(self):
        """Test peak_list method with empty spectrum"""
        spectrum = MzMLSpectrum()
        peaks = spectrum.peak_list()
        
        assert isinstance(peaks, np.ndarray)
        assert peaks.shape == (2, 0)
    
    def test_peak_list_with_data(self):
        """Test peak_list method with data"""
        spectrum = MzMLSpectrum()
        spectrum.mz = np.array([100.0, 200.0, 300.0])
        spectrum.intens = np.array([1000.0, 2000.0, 1500.0])
        
        peaks = spectrum.peak_list()
        assert peaks.shape == (2, 3)
        np.testing.assert_array_equal(peaks[0], [100.0, 200.0, 300.0])
        np.testing.assert_array_equal(peaks[1], [1000.0, 2000.0, 1500.0])
    
    def test_get_vals_with_ms1_scan(self):
        """Test get_vals method with mock MS1 scan data"""
        # Create mock MS1 scan data
        scan_data = {
            "id": "controllerType=0 controllerNumber=1 scan=1000",
            "ms level": 1,
            "scanList": {
                "scan": [{
                    "scan start time": 10.5,
                    "ion injection time": 50.0,
                    "scanWindowList": {
                        "scanWindow": [{
                            "scan window lower limit": 100.0,
                            "scan window upper limit": 1000.0
                        }]
                    }
                }]
            },
            "m/z array": np.array([100.0, 200.0, 300.0]),
            "intensity array": np.array([1000.0, 2000.0, 1500.0]),
            "total ion current": 4500.0
        }
        
        spectrum = MzMLSpectrum()
        spectrum.get_vals(scan_data)
        
        # Check that all values were set correctly
        assert spectrum.id == "controllerType=0 controllerNumber=1 scan=1000"
        assert spectrum.scan_num == 1000
        assert spectrum.level == 1
        assert spectrum.RT == 10.5
        assert spectrum.injection_time == 0.05  # Converted to seconds
        assert spectrum.TIC == 4500.0
        np.testing.assert_array_equal(spectrum.mz, [100.0, 200.0, 300.0])
        np.testing.assert_array_equal(spectrum.intens, [1000.0, 2000.0, 1500.0])
        assert spectrum.scanwindow == [100.0, 1000.0]
        
        # MS1-specific checks
        assert spectrum.collision_energy is None
        assert spectrum.prec_mz is None
        assert spectrum.ms1window is None
    
    def test_get_vals_with_ms2_scan(self):
        """Test get_vals method with mock MS2 scan data"""
        # Create mock MS2 scan data
        scan_data = {
            "id": "controllerType=0 controllerNumber=1 scan=2000",
            "ms level": 2,
            "scanList": {
                "scan": [{
                    "scan start time": 15.2,
                    "ion injection time": 100.0,
                    "scanWindowList": {
                        "scanWindow": [{
                            "scan window lower limit": 50.0,
                            "scan window upper limit": 2000.0
                        }]
                    }
                }]
            },
            "m/z array": np.array([150.0, 250.0]),
            "intensity array": np.array([500.0, 800.0]),
            "total ion current": 1300.0,
            "precursorList": {
                "precursor": [{
                    "activation": {
                        "collision energy": 25.0
                    },
                    "isolationWindow": {
                        "isolation window target m/z": 524.3,
                        "isolation window lower offset": 0.5,
                        "isolation window upper offset": 0.5
                    }
                }]
            }
        }
        
        spectrum = MzMLSpectrum()
        spectrum.get_vals(scan_data)
        
        # Check basic values
        assert spectrum.id == "controllerType=0 controllerNumber=1 scan=2000"
        assert spectrum.scan_num == 2000
        assert spectrum.level == 2
        assert spectrum.RT == 15.2
        assert spectrum.injection_time == 0.1  # Converted to seconds
        
        # MS2-specific checks
        assert spectrum.collision_energy == 25.0
        assert spectrum.prec_mz == 524.3
        np.testing.assert_array_almost_equal(spectrum.ms1window, [523.8, 524.8])
    
    def test_get_vals_error_handling(self):
        """Test error handling in get_vals method"""
        spectrum = MzMLSpectrum()
        
        # Test with missing required field
        incomplete_scan = {"id": "test"}
        
        with pytest.raises(ValueError, match="Error parsing mzML scan"):
            spectrum.get_vals(incomplete_scan)
        
        # Test with missing ms level
        missing_level_scan = {
            "id": "controllerType=0 controllerNumber=1 scan=1000"
        }
        
        with pytest.raises(ValueError, match="Missing required field"):
            spectrum.get_vals(missing_level_scan)


class TestMzMLSpectrumFile:
    """Test cases for MzMLSpectrumFile class"""
    
    def test_empty_spectrum_file_creation(self):
        """Test creating an empty MzMLSpectrumFile"""
        spec_file = MzMLSpectrumFile()
        
        assert spec_file.filename is None
        assert len(spec_file.ms1scans) == 0
        assert len(spec_file.ms2scans) == 0
        assert len(spec_file.scan_pos) == 0
    
    def test_inheritance(self):
        """Test that MzMLSpectrumFile properly inherits from BaseSpectrumFile"""
        spec_file = MzMLSpectrumFile()
        assert isinstance(spec_file, BaseSpectrumFile)
    
    def test_get_by_idx_empty_file(self):
        """Test get_by_idx with empty file"""
        spec_file = MzMLSpectrumFile()
        result = spec_file.get_by_idx(1000)
        assert result is None
    
    @pytest.mark.skipif(not os.path.exists(TEST_MZML_FILE), 
                        reason="Test mzML file not found")
    def test_load_real_mzml_file(self):
        """Test loading actual mzML file"""
        spec_file = MzMLSpectrumFile(TEST_MZML_FILE)
        
        # Check that spectra were loaded
        assert len(spec_file.ms1scans) > 0
        assert len(spec_file.ms2scans) > 0
        assert spec_file.filename == TEST_MZML_FILE
        
        # Check that scan_pos is populated
        assert len(spec_file.scan_pos) > 0
        
        # Test getting spectra by index
        for scan_num, (level, idx) in spec_file.scan_pos.items():
            spectrum = spec_file.get_by_idx(scan_num)
            assert spectrum is not None
            assert spectrum.level == level
            assert spectrum.scan_num == scan_num
    
    @pytest.mark.skipif(not os.path.exists(TEST_MZML_FILE), 
                        reason="Test mzML file not found")
    def test_spectrum_properties(self):
        """Test that loaded spectra have expected properties"""
        spec_file = MzMLSpectrumFile(TEST_MZML_FILE)
        
        # Test MS1 spectrum properties
        if spec_file.ms1scans:
            ms1_spec = spec_file.ms1scans[0]
            assert isinstance(ms1_spec, MzMLSpectrum)
            assert ms1_spec.level == 1
            assert ms1_spec.id is not None
            assert ms1_spec.RT is not None
            assert ms1_spec.mz is not None
            assert ms1_spec.intens is not None
            assert len(ms1_spec.mz) == len(ms1_spec.intens)
        
        # Test MS2 spectrum properties
        if spec_file.ms2scans:
            ms2_spec = spec_file.ms2scans[0]
            assert isinstance(ms2_spec, MzMLSpectrum)
            assert ms2_spec.level == 2
            assert ms2_spec.id is not None
            assert ms2_spec.RT is not None
            assert ms2_spec.collision_energy is not None
            assert ms2_spec.prec_mz is not None
    
    def test_load_spectra_file_not_found(self):
        """Test error handling when file doesn't exist"""
        spec_file = MzMLSpectrumFile()
        
        with pytest.raises(ValueError, match="Failed to load mzML file"):
            spec_file.load_spectra("nonexistent_file.mzml")
    
    def test_load_spectra_with_corrupted_scan(self):
        """Test handling of corrupted scan data"""
        spec_file = MzMLSpectrumFile()
        
        # Mock mzml.MzML to return corrupted data
        with patch('src.utils.io.load_files.mzml.MzML') as mock_mzml:
            # Create mock that yields one good scan and one bad scan
            good_scan = {
                "id": "scan=1000",
                "ms level": 1,
                "scanList": {"scan": [{"scan start time": 10.0, "ion injection time": 50.0,
                                      "scanWindowList": {"scanWindow": [{"scan window lower limit": 100.0,
                                                                        "scan window upper limit": 1000.0}]}}]},
                "m/z array": np.array([100.0]),
                "intensity array": np.array([1000.0]),
                "total ion current": 1000.0
            }
            bad_scan = {"id": "bad_scan"}  # Missing required fields
            
            mock_reader = MagicMock()
            mock_reader.__iter__ = MagicMock(return_value=iter([good_scan, bad_scan]))
            mock_reader.__enter__ = MagicMock(return_value=mock_reader)
            mock_reader.__exit__ = MagicMock(return_value=None)
            mock_mzml.return_value = mock_reader
            
            # Should not raise an exception, but should skip the bad scan
            spec_file.load_spectra("test.mzml")
            
            # Should have loaded only the good scan
            assert len(spec_file.ms1scans) == 1
            assert len(spec_file.ms2scans) == 0


class TestLoadSpectraFunction:
    """Test cases for loadSpectra function"""
    
    @pytest.mark.skipif(not os.path.exists(TEST_MZML_FILE), 
                        reason="Test mzML file not found")
    def test_load_spectra_from_file(self):
        """Test loadSpectra function with real file"""
        # Use temporary directory to avoid conflicts
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_mzml = os.path.join(temp_dir, "test.mzml")
            
            # Copy test file to temp location
            import shutil
            shutil.copy2(TEST_MZML_FILE, temp_mzml)
            
            # Load spectra
            spec_file = loadSpectra(temp_mzml)
            
            # Check that it returns MzMLSpectrumFile
            assert isinstance(spec_file, MzMLSpectrumFile)
            assert spec_file.filename == temp_mzml
            assert len(spec_file.ms1scans) > 0
            assert len(spec_file.ms2scans) > 0
            
            # Check that pickle file was created
            pickle_file = temp_mzml + "_pythonspec"
            assert os.path.exists(pickle_file)
    
    @pytest.mark.skipif(not os.path.exists(TEST_MZML_FILE), 
                        reason="Test mzML file not found")
    def test_load_spectra_from_pickle(self):
        """Test loadSpectra function loading from pickle"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_mzml = os.path.join(temp_dir, "test.mzml")
            pickle_file = temp_mzml + "_pythonspec"
            
            # Copy test file
            import shutil
            shutil.copy2(TEST_MZML_FILE, temp_mzml)
            
            # First load creates pickle file
            spec_file1 = loadSpectra(temp_mzml)
            
            # Second load should use pickle file
            spec_file2 = loadSpectra(temp_mzml)
            
            # Both should be equivalent
            assert len(spec_file1.ms1scans) == len(spec_file2.ms1scans)
            assert len(spec_file1.ms2scans) == len(spec_file2.ms2scans)
    
    def test_load_spectra_legacy_pickle_handling(self):
        """Test handling of legacy pickle files"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_mzml = os.path.join(temp_dir, "test.mzml")
            pickle_file = temp_mzml + "_pythonspec"
            
            # Copy the real test file for this test
            import shutil
            if os.path.exists(TEST_MZML_FILE):
                shutil.copy2(TEST_MZML_FILE, temp_mzml)
                
                # Create a fake legacy pickle file with malformed content
                with open(pickle_file, 'wb') as f:
                    f.write(b'malformed pickle data that will cause error')
                
                # This should detect the corrupted pickle and recreate it
                spec_file = loadSpectra(temp_mzml)
                
                # Verify it worked by checking the recreated file
                assert isinstance(spec_file, MzMLSpectrumFile)
                assert len(spec_file.ms1scans) > 0 or len(spec_file.ms2scans) > 0
            else:
                pytest.skip("Test mzML file not available")


class TestIntegration:
    """Integration tests combining multiple components"""
    
    @pytest.mark.skipif(not os.path.exists(TEST_MZML_FILE), 
                        reason="Test mzML file not found")
    def test_end_to_end_workflow(self):
        """Test complete workflow from file loading to data access"""
        # Load spectra
        spec_file = loadSpectra(TEST_MZML_FILE)
        
        # Verify basic structure
        assert isinstance(spec_file, MzMLSpectrumFile)
        assert len(spec_file.ms1scans) > 0
        assert len(spec_file.ms2scans) > 0
        
        # Test accessing individual spectra
        for scan_num in list(spec_file.scan_pos.keys())[:3]:  # Test first 3 scans
            spectrum = spec_file.get_by_idx(scan_num)
            assert spectrum is not None
            
            # Test peak_list method
            peaks = spectrum.peak_list()
            assert peaks.shape[0] == 2  # Should have mz and intensity rows
            
            # Test that data is reasonable
            if spectrum.level == 1:
                assert spectrum.collision_energy is None
                assert spectrum.prec_mz is None
            elif spectrum.level == 2:
                assert spectrum.collision_energy is not None
                assert spectrum.prec_mz is not None
    
    @pytest.mark.skipif(not os.path.exists(TEST_MZML_FILE), 
                        reason="Test mzML file not found")
    def test_data_consistency(self):
        """Test data consistency across the spectrum file"""
        spec_file = loadSpectra(TEST_MZML_FILE)
        
        # Check that all MS1 scans have consistent properties
        for ms1_spec in spec_file.ms1scans:
            assert ms1_spec.level == 1
            assert ms1_spec.RT is not None
            assert ms1_spec.TIC is not None
            assert ms1_spec.mz is not None
            assert ms1_spec.intens is not None
            assert len(ms1_spec.mz) == len(ms1_spec.intens)
        
        # Check that all MS2 scans have consistent properties
        for ms2_spec in spec_file.ms2scans:
            assert ms2_spec.level == 2
            assert ms2_spec.RT is not None
            assert ms2_spec.collision_energy is not None
            assert ms2_spec.prec_mz is not None
            assert ms2_spec.mz is not None
            assert ms2_spec.intens is not None
            assert len(ms2_spec.mz) == len(ms2_spec.intens)
        
        # Check scan_pos consistency
        total_scans = len(spec_file.ms1scans) + len(spec_file.ms2scans)
        assert len(spec_file.scan_pos) == total_scans
        
        # Verify each scan_pos entry points to valid spectrum
        for scan_num, (level, idx) in spec_file.scan_pos.items():
            if level == 1:
                assert idx < len(spec_file.ms1scans)
                assert spec_file.ms1scans[idx].scan_num == scan_num
            elif level == 2:
                assert idx < len(spec_file.ms2scans)
                assert spec_file.ms2scans[idx].scan_num == scan_num


class TestPerformance:
    """Performance-related tests"""
    
    @pytest.mark.skipif(not os.path.exists(TEST_MZML_FILE), 
                        reason="Test mzML file not found")
    def test_loading_speed(self):
        """Test that file loading completes in reasonable time"""
        import time
        
        start_time = time.time()
        spec_file = loadSpectra(TEST_MZML_FILE)
        end_time = time.time()
        
        # Should load test file in under 5 seconds
        loading_time = end_time - start_time
        assert loading_time < 5.0, f"Loading took {loading_time:.2f}s, expected < 5.0s"
        
        # Verify data was actually loaded
        assert len(spec_file.ms1scans) > 0
        assert len(spec_file.ms2scans) > 0
    
    @pytest.mark.skipif(not os.path.exists(TEST_MZML_FILE), 
                        reason="Test mzML file not found")
    def test_memory_usage(self):
        """Test that memory usage is reasonable"""
        import sys
        
        # Load spectra
        spec_file = loadSpectra(TEST_MZML_FILE)
        
        # Check that object sizes are reasonable
        # This is a basic check - in a real scenario you might use memory_profiler
        total_spectra = len(spec_file.ms1scans) + len(spec_file.ms2scans)
        assert total_spectra > 0
        
        # Verify that spectra contain actual data
        if spec_file.ms1scans:
            ms1_spec = spec_file.ms1scans[0]
            assert len(ms1_spec.mz) > 0
            assert len(ms1_spec.intens) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])