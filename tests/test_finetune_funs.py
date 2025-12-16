import pytest
import numpy as np
import pandas as pd
import sys, os
from src.finetune_funs import one_hot_encode_sequence, create_model_data, scale_rt, fine_tune_rt


max_sequence_length = 5
amino_acids = "ACDEFGHIKLMNPQRSTVWY"
num_amino_acids = len(amino_acids)
amino_acid_to_index = {aa: i for i, aa in enumerate(amino_acids)}

def test_one_hot_encode_sequence_unknown_aa():
    seq = "AXZ"
    encoded = one_hot_encode_sequence(seq)
    
    # Only 'A' should be encoded, 'X' and 'Z' ignored
    assert encoded[0, amino_acid_to_index['A']] == 1.0
    assert np.all(encoded[1:] == 0.0)

def test_create_model_data_shapes():
    data = {
        "PeptideSequence": ["ACD", "FGH", "IKL", "MNP", "QRS", "TVW", "YAC", "DEF", "GHI", "KLM"],
        "RT": np.arange(10.0)
    }
    df = pd.DataFrame(data)
    
    X_train, X_test, Y_train, Y_test = create_model_data(df)
    
    # Check shapes
    assert X_train.shape[0] == 9  # 90% of 10
    assert X_test.shape[0] == 1   # 10% of 10
    assert X_train.shape[1:] == (30, num_amino_acids)
    
    # Check Y_train/Y_test lengths
    assert len(Y_train) == X_train.shape[0]
    assert len(Y_test) == X_test.shape[0]


def test_scale_rt_basic():
    rt = [10, 20, 30]
    min_max = [0, 1]
    scaled = scale_rt(rt, min_max)
    
    assert np.isclose(np.min(scaled), 0.0)
    assert np.isclose(np.max(scaled), 1.0)
    assert np.all(np.diff(scaled) > 0)

def test_fine_tune_rt_runs():
    df = pd.DataFrame({
        "Stripped.Sequence": ["ACD", "FGH", "IKL"],
        "RT": [0.1, 0.2, 0.3]
    })

    class Tag:
        name = "mTRAQ"
    
    # Run fine_tune_rt
    result = fine_tune_rt(df, qc_plots=False, tag=Tag())

    assert isinstance(result, tuple)
    assert len(result) == 3

    data_split, models, model_to_obs = result
    assert isinstance(data_split, tuple)
    assert len(data_split) == 4
    sample_input = np.array([0.1, 0.2])
    output = model_to_obs(sample_input)
    assert isinstance(output, np.ndarray)
    assert output.shape == sample_input.shape