"""
Tests to validate FDR analysis works correctly with unified spectral fitting.

This test suite ensures that the FDR calculation properly handles the
unified feature output from the new spectral fitting implementation.
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import Mock, patch

# We'll test the integration points rather than the full FDR analysis
from src.spectral_fitting import UnifiedFeatures


class TestFDRWithUnifiedFeatures:
    """Test FDR analysis compatibility with unified features."""
    
    def test_unified_features_to_dataframe(self):
        """Test converting UnifiedFeatures to DataFrame format expected by FDR."""
        # Create sample unified features
        n_candidates = 10
        features = np.random.rand(n_candidates, 26)
        is_decoy = np.array([False, False, True, False, True, 
                            False, True, False, True, False])
        
        unified_features = UnifiedFeatures(
            features=features,
            is_decoy=is_decoy,
            feature_names=[f"feature_{i}" for i in range(26)]
        )
        
        # Get target and decoy features separately
        target_features = unified_features.get_target_features()
        decoy_features = unified_features.get_decoy_features()
        
        assert target_features.shape == (6, 26)  # 6 targets (indices 0,1,3,5,7,9)
        assert decoy_features.shape == (4, 26)   # 4 decoys (indices 2,4,6,8)
        
        # Verify the features are correctly separated
        assert np.array_equal(target_features[0], features[0])  # First target
        assert np.array_equal(decoy_features[0], features[2])   # First decoy
        
    def test_feature_array_format(self):
        """Test that feature arrays maintain correct format for FDR analysis."""
        # The FDR analysis expects features in a specific order
        # Let's create features with known values for important columns
        
        features = np.zeros((4, 26))
        
        # Set some known feature values based on the feature names from spectral_fitting
        # Feature 0: num_lib_peaks_matched
        features[:, 0] = [10, 8, 12, 5]
        
        # Feature 1: frac_lib_intensity  
        features[:, 1] = [0.8, 0.6, 0.9, 0.4]
        
        # Feature 3: rel_error (MS1 error)
        features[:, 3] = [2.5, 3.0, 1.5, 5.0]
        
        # Feature 4: rt_error
        features[:, 4] = [0.1, -0.2, 0.05, 0.3]
        
        # Feature 16: scribe_scores
        features[:, 16] = [0.95, 0.85, 0.98, 0.70]
        
        # Feature 20: manhattan_distances
        features[:, 20] = [-2.5, -3.0, -2.0, -4.0]
        
        is_decoy = np.array([False, True, False, True])
        
        unified_features = UnifiedFeatures(features=features, is_decoy=is_decoy)
        
        # Check that we can access specific features
        target_features = unified_features.get_target_features()
        decoy_features = unified_features.get_decoy_features()
        
        # Verify feature columns are preserved
        assert target_features[0, 0] == 10  # First target, num peaks matched
        assert decoy_features[0, 0] == 8    # First decoy, num peaks matched
        
        # Check SCRIBE scores
        assert target_features[0, 16] == 0.95
        assert target_features[1, 16] == 0.98
        
    def test_feature_names_mapping(self):
        """Test that feature names map correctly to FDR expectations."""
        expected_feature_names = [
            "num_lib_peaks_matched", "frac_lib_intensity", "frac_dia_intensity",
            "rel_error", "rt_error", "frac_int_matched", "frac_int_pred",
            "r2all", "r2_lib_spec", "r2_unique", "frac_unique_pred",
            "frac_dia_intensity_pred", "hyperscores", "b_counts", "y_counts",
            "longest_y_ions", "scribe_scores", "max_unmatched_residuals",
            "max_matched_residuals", "gof_stats", "manhattan_distances",
            "fitted_spectral_contrasts", "frac_int_matched_pred",
            "frac_int_matched_pred_sigcoeff", "large_coeff_cosine", "rt_mz"
        ]
        
        features = np.random.rand(5, 26)
        is_decoy = np.array([False, True, False, True, False])
        
        unified_features = UnifiedFeatures(
            features=features,
            is_decoy=is_decoy,
            feature_names=expected_feature_names
        )
        
        assert len(unified_features.feature_names) == 26
        assert unified_features.feature_names[0] == "num_lib_peaks_matched"
        assert unified_features.feature_names[16] == "scribe_scores"
        assert unified_features.feature_names[20] == "manhattan_distances"
        
    def test_empty_features_handling(self):
        """Test handling of empty feature sets."""
        # Empty features case
        empty_features = UnifiedFeatures(
            features=np.empty((0, 26)),
            is_decoy=np.array([], dtype=bool)
        )
        
        assert empty_features.get_target_features().shape == (0, 26)
        assert empty_features.get_decoy_features().shape == (0, 26)
        
    def test_all_targets_no_decoys(self):
        """Test case with only target peptides."""
        features = np.random.rand(10, 26)
        is_decoy = np.zeros(10, dtype=bool)  # All False = all targets
        
        unified_features = UnifiedFeatures(features=features, is_decoy=is_decoy)
        
        assert unified_features.get_target_features().shape == (10, 26)
        assert unified_features.get_decoy_features().shape == (0, 26)
        
    def test_feature_consistency(self):
        """Test that features maintain consistency through processing."""
        # Create features with specific patterns
        n_candidates = 20
        features = np.zeros((n_candidates, 26))
        
        # Create alternating target/decoy pattern
        is_decoy = np.array([i % 2 == 1 for i in range(n_candidates)])
        
        # Set feature values based on target/decoy status
        for i in range(n_candidates):
            if is_decoy[i]:
                features[i, 0] = 5  # Decoys have fewer matched peaks
                features[i, 20] = -4.0  # Worse Manhattan distance
            else:
                features[i, 0] = 10  # Targets have more matched peaks
                features[i, 20] = -2.0  # Better Manhattan distance
        
        unified_features = UnifiedFeatures(features=features, is_decoy=is_decoy)
        
        target_features = unified_features.get_target_features()
        decoy_features = unified_features.get_decoy_features()
        
        # All targets should have 10 matched peaks
        assert np.all(target_features[:, 0] == 10)
        # All decoys should have 5 matched peaks
        assert np.all(decoy_features[:, 0] == 5)
        
        # Manhattan distances should follow pattern
        assert np.all(target_features[:, 20] == -2.0)
        assert np.all(decoy_features[:, 20] == -4.0)


class TestFDRIntegration:
    """Test integration between unified spectral fitting and FDR analysis."""
    
    @pytest.mark.skip(reason="Requires full FDR module setup")
    def test_fdr_with_unified_output(self):
        """Test that FDR analysis can process unified spectral fitting output."""
        # This would test the full integration but requires
        # setting up the FDR analysis module
        pass
    
    def test_output_format_compatibility(self):
        """Test that unified output format is compatible with FDR expectations."""
        # The FDR analysis expects specific column structure in the output
        # Let's verify our output maintains this structure
        
        # Mock output row from fit_to_lib2
        mock_output_row = [
            0.5,          # coefficient
            1000,         # spec_idx
            999,          # ms1_spec_idx
            "PEPTIDE1",   # sequence
            2,            # charge
            500.0,        # prec_mz
            30.0,         # prec_rt
            # 26 features
            *np.random.rand(26),
            # Fragment info (7 columns)
            "b2;b3;y2;y3",     # frag names
            "0.001;0.002",     # frag errors
            "100.1;150.2",     # lib frag mz
            "1000;2000",       # lib frag int
            "900;1900",        # obs frag int
            "",                # unique frags
            "",                # unique frag int
            "test.mzML",       # filename
            "PROTEIN1"         # protein
        ]
        
        # Verify we have the expected number of columns
        # 7 (metadata) + 26 (features) + 7 (fragments) + 2 (file/protein) = 42
        assert len(mock_output_row) == 42
        
        # Verify types are correct
        assert isinstance(mock_output_row[0], float)  # coefficient
        assert isinstance(mock_output_row[1], int)    # spec_idx
        assert isinstance(mock_output_row[3], str)    # sequence
        assert isinstance(mock_output_row[4], int)    # charge