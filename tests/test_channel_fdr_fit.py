import numpy as np
import polars as pl
import pytest

from src.channel_fdr_fit import (MIN_DECOYS, fit_channel_fdr, model_matrix, per_channel_pep, per_channel_qvalues,
                                 qvalues)
from src.fdr_analysis import estimate_pep


class TestQvalues:
    def test_the_score_precursors_estimator(self):
        # ranked: T T D T D T -> (1 + decoys) / targets, then the minimum from below
        score = np.array([6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
        is_decoy = np.array([False, False, True, False, True, False])
        raw = np.array([1 / 1, 1 / 2, 2 / 2, 2 / 3, 3 / 3, 3 / 4])
        assert qvalues(score, is_decoy) == pytest.approx(np.minimum.accumulate(raw[::-1])[::-1])

    def test_ratio_and_order(self):
        score = np.array([1.0, 3.0, 2.0])
        is_decoy = np.array([False, False, True])
        # by score: T(3) D(2) T(1)
        assert qvalues(score, is_decoy, ratio=2.0) == pytest.approx([2.0, 2.0, 2.0])

    def test_each_channel_against_the_whole_decoy_channel(self):
        channel = np.array([0.0, 0.0, 4.0, 4.0, 20.0, 20.0])
        is_decoy = channel == 20
        score = np.array([9.0, 1.0, 8.0, 7.0, 5.0, 2.0])
        q = per_channel_qvalues(channel, score, is_decoy)
        assert np.isnan(q[is_decoy]).all()
        # channel 0 alone with the decoys: 9 T, 5 D, 2 D, 1 T
        assert q[:2] == pytest.approx(qvalues(np.array([9.0, 1.0, 5.0, 2.0]),
                                              np.array([False, False, True, True]))[:2])
        # channel 4 is not ranked against channel 0: 8 T, 7 T, 5 D, 2 D
        assert q[2:4] == pytest.approx([0.5, 0.5])


class TestPerChannelPep:
    def test_each_channel_against_the_whole_decoy_channel(self):
        rng = np.random.default_rng(1)
        channel = np.repeat([0.0, 4.0, 20.0], 50)
        is_decoy = channel == 20
        score = rng.normal(size=channel.size) + np.where(channel == 0, 1.0, 0.0)
        pep = per_channel_pep(channel, score, is_decoy)
        assert np.isnan(pep[is_decoy]).all()
        alone = (channel == 0) | is_decoy
        assert pep[channel == 0] == pytest.approx(estimate_pep(score[alone], is_decoy[alone])[:50])
        # channel 4 does not change channel 0's
        moved = score.copy()
        moved[channel == 4] += 10
        assert per_channel_pep(channel, moved, is_decoy)[channel == 0] == pytest.approx(pep[channel == 0])

    def test_not_pooled(self):
        # two real channels, both pure null: each is as decoy-like as the decoy
        # channel itself, which pooling them would halve
        rng = np.random.default_rng(2)
        channel = np.repeat([0.0, 4.0, 20.0], 400)
        is_decoy = channel == 20
        score = rng.normal(size=channel.size)
        assert np.median(per_channel_pep(channel, score, is_decoy)[~is_decoy]) >             1.5 * np.median(estimate_pep(score, is_decoy)[~is_decoy])


class TestModelMatrix:
    def test_columns_with_no_information_are_left_out(self):
        features = pl.DataFrame({"a": [1.0, 2.0, None], "constant": [3.0, 3.0, 3.0],
                                 "empty": [None, None, None], "n": [1, 2, 3]},
                                schema_overrides={"empty": pl.Float64})
        X, dead = model_matrix(features, ["a", "constant", "empty", "n"])
        assert list(X.columns) == ["a", "n"] and dead == ["constant", "empty"]
        assert np.isnan(X["a"].iloc[2])                       # a missing value stays missing

    def test_infinite_values_are_an_error(self):
        with pytest.raises(ValueError, match="infinite"):
            model_matrix(pl.DataFrame({"a": [1.0, np.inf, 2.0]}), ["a"])


def _run(n_peptides=150, seed=0):
    """A synthetic run: each peptide in real channels 0 and 4 and the decoy
    channel 20.  A real channel is there for half the peptides, and then its
    "signal" is high; an empty one (and the decoy) draws from the null."""
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_peptides):
        for channel in (0.0, 4.0, 20.0):
            present = channel != 20 and p % 2 == 0
            rows.append({"untag_seq": f"PEP{p}", "channel": channel, "is_decoy_channel": channel == 20,
                         "present": present, "signal": rng.normal(4.0 if present else 0.0, 1.0),
                         "noise": rng.normal(), "constant": 1.0})
    return pl.DataFrame(rows)


@pytest.fixture(scope="module")
def fitted():
    features = _run()
    return features, fit_channel_fdr(features, ["signal", "noise", "constant"])


class TestFit:
    def test_channels_that_are_there_pass_and_empty_ones_do_not(self, fitted):
        features, (score, q, _) = fitted
        present = features["present"].to_numpy()
        real = ~features["is_decoy_channel"].to_numpy()
        assert np.median(q[present]) < 0.05
        assert np.median(q[real & ~present]) > 0.3
        assert score.shape == q.shape == (features.height,)

    def test_the_decoy_channel_has_no_qvalue(self, fitted):
        features, (_, q, _) = fitted
        assert np.isnan(q[features["is_decoy_channel"].to_numpy()]).all()
        assert np.isfinite(q[~features["is_decoy_channel"].to_numpy()]).all()

    def test_importance_is_the_evidence(self, fitted):
        _, (_, _, importance) = fitted
        assert importance.index[0] == "signal"
        assert "constant" not in importance.index                 # left out, no information
        assert importance["share_of_gain"].sum() == pytest.approx(1.0)
        assert (importance["share_of_gain_sd"] >= 0).all()           # over the folds

    def test_too_few_decoys_is_no_fit(self):
        features = _run(n_peptides=MIN_DECOYS - 1)
        assert fit_channel_fdr(features, ["signal", "noise"]) is None
