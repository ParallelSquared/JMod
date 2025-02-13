import unittest
import re
from miscFunctions import change_seq
from sklearn.model_selection import GroupKFold
import numpy as np
import pandas as pd
from fdr_analysis import score_model
#from score_model import score_model  # Import your score_model class

#Testing sequence permutations for building decoys
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

#Tests for building frags for decoy sequences 
class TestConvertFrags(unittest.TestCase):
    def dummy_test(self):
        self.assertEqual(True, True)  # Reverse all but last letter

#Testing train-test splitting. Ensure each protein only represented in one CV fold. See fdr_analysis.py 
class TestProteinBasedSplits(unittest.TestCase):
    def setUp(self):
        """Create sample data for testing"""
        # Create synthetic dataset with 100 rows, 3 features, across 10 proteins
        np.random.seed(42)
        n_proteins = 10
        rows_per_protein = 10
        n_rows = n_proteins * rows_per_protein
        
        # Generate feature data
        self.X = pd.DataFrame({
            'feature1': np.random.randn(n_rows),
            'feature2': np.random.randn(n_rows),
            'feature3': np.random.randn(n_rows),
            'protein': [f'protein_{i}' for i in range(n_proteins) for _ in range(rows_per_protein)]
        })
        
        # Generate binary labels
        self.y = np.random.randint(0, 2, n_rows)
        
        # Initialize model
        self.model = score_model(model_type='rf', n_splits=5)

    def test_proteins_stay_together(self):
        """Test that all rows for a protein stay in the same fold"""
        gkf = GroupKFold(n_splits=5)
        groups = self.X['protein']
        X_features = self.X.drop('protein', axis=1)
        
        for train_idx, test_idx in gkf.split(X_features, self.y, groups=groups):
            # Get proteins in training and test sets
            train_proteins = set(groups.iloc[train_idx])
            test_proteins = set(groups.iloc[test_idx])
            
            # Check no overlap between train and test proteins
            self.assertEqual(len(train_proteins.intersection(test_proteins)), 0,
                            "Found same protein in both train and test sets")

    def test_fold_sizes(self):
        """Test that folds are approximately equal size"""
        gkf = GroupKFold(n_splits=5)
        groups = self.X['protein']
        X_features = self.X.drop('protein', axis=1)
        
        fold_sizes = []
        for _, test_idx in gkf.split(X_features, self.y, groups=groups):
            fold_sizes.append(len(test_idx))
        
        # Check that no fold is more than 50% larger than the smallest fold
        min_size = min(fold_sizes)
        max_size = max(fold_sizes)
        self.assertLess(max_size / min_size, 1.5,
                       "Fold sizes are too unbalanced")

    def test_all_proteins_used(self):
        """Test that all proteins appear in exactly one fold"""
        gkf = GroupKFold(n_splits=5)
        groups = self.X['protein']
        X_features = self.X.drop('protein', axis=1)
        
        # Track which proteins have been seen in test sets
        seen_proteins = set()
        
        for _, test_idx in gkf.split(X_features, self.y, groups=groups):
            test_proteins = set(groups.iloc[test_idx])
            
            # Check no protein appears twice
            self.assertEqual(len(seen_proteins.intersection(test_proteins)), 0,
                           "Same protein found in multiple test sets")
            
            seen_proteins.update(test_proteins)
        
        # Check all proteins were used
        all_proteins = set(groups.unique())
        self.assertEqual(seen_proteins, all_proteins,
                        "Not all proteins appeared in test sets")

    def test_model_integration(self):
        """Test the full model with protein-based splitting"""
        # Run the model
        protein_groups = self.X['protein']
        self.X.drop('protein', axis = 1, inplace = True)
        predictions = self.model.run_model(self.X, self.y, protein_groups)
        
        # Basic checks on predictions
        self.assertEqual(len(predictions), len(self.y),
                        "Predictions length doesn't match input length")
        self.assertTrue(all(isinstance(p, (int, float)) for p in predictions),
                       "Predictions contain non-numeric values")
        self.assertTrue(all(0 <= p <= 1 for p in predictions),
                       "Predictions outside [0,1] range")

if __name__ == '__main__':
    unittest.main()