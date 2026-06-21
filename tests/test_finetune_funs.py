import pytest
import numpy as np
import pandas as pd
import sys, os
import inspect
from src.finetune_funs import (
    one_hot_encode_sequence, create_model_data, scale_rt, fine_tune_rt,
    _two_half_gauss_pep, load_existing_models,
)


class TestTwoHalfGaussPep:
    def test_separates_narrow_core_from_wide_tail(self):
        rng = np.random.default_rng(0)
        core = rng.normal(0, 0.7, 4000)        # true positives
        tail = rng.uniform(-12, 12, 800)       # false positives
        r = np.concatenate([core, tail])
        pep, params = _two_half_gauss_pep(np.abs(r))
        # narrow component much tighter than wide
        assert params["s_narrow"] < params["s_wide"]
        # core points get low PEP, tail points get high PEP on average
        assert pep[:4000].mean() < 0.25
        assert pep[4000:].mean() > 0.5

    def test_pep_in_unit_interval(self):
        rng = np.random.default_rng(1)
        pep, _ = _two_half_gauss_pep(np.abs(rng.normal(0, 1, 1000)))
        assert pep.min() >= 0.0 and pep.max() <= 1.0


class TestLoadExistingModelsSignature:
    def test_n_models_default_is_three(self):
        sig = inspect.signature(load_existing_models)
        assert sig.parameters["n_models"].default == 3


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

class TestFineTuneRT:
    """Tests for fine_tune_rt function with different tag types."""

    @pytest.fixture
    def sample_df(self):
        """Minimal DataFrame for testing."""
        return pd.DataFrame({
            "Stripped.Sequence": ["ACD", "FGH", "IKL"],
            "RT": [0.1, 0.2, 0.3]
        })

    def _verify_result(self, result):
        """Common assertions for fine_tune_rt results."""
        assert isinstance(result, tuple)
        assert len(result) == 3

        data_split, models, model_to_obs = result
        assert isinstance(data_split, tuple)
        assert len(data_split) == 4
        sample_input = np.array([0.1, 0.2])
        output = model_to_obs(sample_input)
        assert isinstance(output, np.ndarray)
        assert output.shape == sample_input.shape

    def test_fine_tune_rt_mTRAQ_tag(self, sample_df):
        """Test fine_tune_rt with mTRAQ tag."""
        class Tag:
            name = "mTRAQ"

        result = fine_tune_rt(sample_df, qc_plots=False, tag=Tag())
        self._verify_result(result)

    def test_fine_tune_rt_diethyl_tag(self, sample_df):
        """Test fine_tune_rt with diethyl tag."""
        class Tag:
            name = "diethyl_light"

        result = fine_tune_rt(sample_df, qc_plots=False, tag=Tag())
        self._verify_result(result)

    def test_fine_tune_rt_PSMtag(self, sample_df):
        """Test fine_tune_rt with PSMtag."""
        class Tag:
            name = "PSMtag_5plex"

        result = fine_tune_rt(sample_df, qc_plots=False, tag=Tag())
        self._verify_result(result)

    def test_fine_tune_rt_no_tag(self, sample_df):
        """Test fine_tune_rt with tag=None (label-free)."""
        result = fine_tune_rt(sample_df, qc_plots=False, tag=None)
        self._verify_result(result)

    def test_fine_tune_rt_unknown_tag_raises(self, sample_df):
        """Test fine_tune_rt raises ValueError for unknown tag."""
        class Tag:
            name = "unknown_tag_xyz"

        with pytest.raises(ValueError, match="Unknown label"):
            fine_tune_rt(sample_df, qc_plots=False, tag=Tag())