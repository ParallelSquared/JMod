import os

import numpy as np
import pandas as pd
import polars as pl
import pytest

from src import plex_scoring
from src.plex_scoring import (ECDF_PLOT, FEATURE_FILE, FIRST_MODEL_OUTPUTS, FOLDER, HIST_PLOT, MIN_DECOYS,
                              fit_plex_scoring, model_columns, plot_feature_distributions, protein_qvalues,
                              run_plex_scoring, set_features)
from src.utils.frag_encoding import encode_frag_names


def _rows(**columns):
    """Precursor-channel rows of one run, with the columns set_features reads."""
    n = len(columns["untag_prec"])
    base = {"file_name": ["f"] * n, "is_decoy": [False] * n, "stripped_seq": columns["untag_prec"],
            "untag_prec_Qvalue": [0.0] * n, "Qvalue": [0.0] * n, "rt": [10.0] * n,
            "rt_error": [0.0] * n, "mz_error": [0.0] * n, "hyperscore": [1.0] * n}
    return pd.DataFrame({**base, **columns})


@pytest.fixture
def rank_arrays(monkeypatch):
    """Each measurement channel by channel too (off by default)."""
    monkeypatch.setattr(plex_scoring, "RANK_ARRAYS", True)


class TestSetFeatures:
    """One row per plex set (run, untagged precursor, target or decoy)."""

    def test_one_row_per_set_targets_and_decoys_apart(self):
        rows = _rows(untag_prec=["A_2", "A_2", "A_2", "B_2"], is_decoy=[False, False, True, False],
                     PredVal=[0.9, 0.2, 0.5, 0.7])
        features = set_features(rows)
        assert features.height == 3
        assert features["n_channels"].to_list() == [2, 1, 1]

    def test_the_decoy_channel_is_left_out(self):
        # B is seen only in the decoy channel
        rows = _rows(untag_prec=["A_2", "A_2", "B_2"], is_decoy_channel=[False, True, True],
                     hyperscore=[4.0, 9.0, 8.0], PredVal=[0.4, 0.99, 0.95])
        features = set_features(rows)
        assert features["untag_prec"].to_list() == ["A_2"]
        assert features["n_channels"].to_list() == [1]
        assert features["hyperscore_max"].to_list() == [4.0]

    def test_the_decoy_channel_compared_with_the_real_ones(self):
        rows = _rows(untag_prec=["A_2"] * 3 + ["B_2"], hyperscore=[9.0, 5.0, 7.0, 4.0],
                     is_decoy_channel=[False, False, True, False])
        features = set_features(rows)
        assert features["n_channels"].to_list() == [2, 1]
        assert features["decoy_channel_seen"].to_list() == [1, 0]
        # it beats one of A's two real channels; B has none
        assert features["decoy_channel_outscores"].to_list() == [0.5, None]
        assert features["decoy_channel_hyperscore_from_best"].to_list() == [-2.0, None]
        assert features["decoy_channel_hyperscore_from_median"].to_list() == [0.0, None]

    def test_summaries_over_the_channels_and_no_rank_arrays_by_default(self):
        rows = _rows(untag_prec=["A_2"] * 3, hyperscore=[2.0, 9.0, 5.0], gof_stats=[0.2, 0.9, 0.4],
                     rt=[10.4, 10.0, 10.1])
        features = set_features(rows)
        assert features["gof_stats_mean"].to_list() == pytest.approx([0.5])
        assert features["gof_stats_median"].to_list() == pytest.approx([0.4])
        assert features["gof_stats_min"].to_list() == pytest.approx([0.2])
        assert features["gof_stats_max"].to_list() == pytest.approx([0.9])
        # the other channels' elution against the first-ranked one's (hyperscore 9)
        assert features["rt_from_best_max"].to_list() == pytest.approx([0.4])
        assert features["rt_from_best_min"].to_list() == pytest.approx([0.1])
        assert features["rt_range"].to_list() == pytest.approx([0.4])
        assert not [c for c in features.columns if "_rank" in c]

    @pytest.mark.usefixtures("rank_arrays")
    def test_rank_arrays_side_by_side_padded_with_null(self):
        rows = _rows(untag_prec=["A_2"] * 3 + ["B_2"], hyperscore=[2.0, 9.0, 5.0, 8.0],
                     gof_stats=[0.2, 0.9, 0.5, 0.8], rt=[10.4, 10.0, 10.1, 20.0])
        features = set_features(rows)
        assert features["hyperscore_rank1"].to_list() == [9.0, 8.0]
        assert features["gof_stats_rank2"].to_list() == [0.5, None]
        assert features["gof_stats_rank3"].to_list() == [0.2, None]
        assert features["rt_from_best_rank2"].to_list()[0] == pytest.approx(0.1)
        assert features["rt_from_best_rank3"].to_list()[0] == pytest.approx(0.4)
        assert features["rt_range"].to_list() == pytest.approx([0.4, None], nan_ok=True)

    def test_placeholders_are_not_measurements(self):
        # mz_error -1: no MS1 peak.  Not an error of 1 from the channels that have one
        rows = _rows(untag_prec=["A_2"] * 3, hyperscore=[9.0, 5.0, 2.0], mz_error=[2e-6, -1.0, 3e-6],
                     sq_mz_error=[4e-12, 1.0, 9e-12])
        features = set_features(rows)
        assert features["n_channels_ms1"].to_list() == [2]
        assert features["mz_error_from_best_max"].to_list() == pytest.approx([1e-6])
        assert features["mz_error_range"].to_list() == pytest.approx([1e-6])
        assert features["mz_error_mean"].to_list() == pytest.approx([2.5e-6])
        assert features["sq_mz_error_mean"].to_list() == pytest.approx([6.5e-12])

    def test_nothing_from_the_precursor_scoring_is_a_feature(self):
        rows = _rows(untag_prec=["A_2", "A_2"], PredVal=[0.9, 0.5], Qvalue=[0.001, 0.2],
                     BestChannel_Qvalue=[0.001, 0.001])
        columns = model_columns(set_features(rows))
        assert not [c for c in columns if any(c.startswith(o) for o in FIRST_MODEL_OUTPUTS)]
        # nor the set's identity
        for column in ("__set", "untag_prec", "is_decoy", "stripped_seq"):
            assert column not in columns

    def test_the_precursor_scorings_inputs_are_features(self):
        rows = _rows(untag_prec=["A_2", "A_2"], PredVal=[0.9, 0.5], scribe_scores=[1.0, 2.0],
                     channels_matched=[2, 2], spec_id=[5, 5], __fdc_idx=[0, 1])
        columns = model_columns(set_features(rows))
        assert {"scribe_scores_mean", "scribe_scores_min", "predicted_rt_mean"} <= set(columns)
        assert not [c for c in columns if c.startswith(("channels_matched", "rt_mean", "spec_id", "__"))]


def _scans(*channels):
    """Per-scan coefficients: for each (seq, [(spec_id, coeff), ...])."""
    rows = [(seq, 2.0, float(c), spec) for seq, points in channels for spec, c in points]
    return pl.DataFrame(rows, schema={"seq": pl.String, "z": pl.Float32, "coeff": pl.Float32,
                                      "spec_id": pl.Int32}, orient="row")


def _cycles(n, windows=1):
    """ms2_cycles' table for *n* cycles of *windows* windows: spec_id
    cycle * windows + window."""
    return pl.DataFrame({"spec_id": np.arange(n * windows), "cycle": np.repeat(np.arange(n), windows)})


def _elution(channels, scans, cycles, apexes=None):
    """set_features with elution for one target set A_2 of *channels*
    {seq: hyperscore}, each channel's own apex scan from *apexes*."""
    extra = {} if apexes is None else {"spec_id": apexes}
    rows = _rows(untag_prec=["A_2"] * len(channels), seq=list(channels), z=[2] * len(channels),
                 hyperscore=list(channels.values()), **extra)
    return set_features(rows, scans, cycles)


@pytest.mark.usefixtures("rank_arrays")
class TestElutionFeatures:
    """Each channel's MS2 coefficient trace over the DIA cycles, against the others'."""

    def test_co_eluting_channels_correlate(self):
        peak = [1, 4, 9, 4, 1]
        scans = _scans(("a", list(enumerate(peak))), ("b", [(i, 2 * c) for i, c in enumerate(peak)]))
        features = _elution({"a": 9.0, "b": 5.0}, scans, _cycles(5))
        assert features["elution_corr_mean"].to_list() == pytest.approx([1.0])
        assert features["elution_corr_to_best_rank2"].to_list() == pytest.approx([1.0])
        assert features["elution_overlap_to_best_rank2"].to_list() == [1.0]
        assert features["elution_cycles"].to_list() == [5]

    def test_channels_in_different_cycles_do_not(self):
        scans = _scans(("a", [(0, 5), (1, 9), (2, 5)]), ("b", [(5, 5), (6, 9), (7, 5)]))
        features = _elution({"a": 9.0, "b": 5.0}, scans, _cycles(8))
        assert features["elution_overlap_to_best_rank2"].to_list() == [0.0]
        assert features["elution_corr_to_best_rank2"].to_list()[0] < 0
        assert features["elution_shared_cycles"].to_list() == [0.0]

    def test_aligned_by_cycle_not_by_scan(self):
        # two windows a cycle: the channels' scans differ, their cycles do not
        peak = [1, 4, 9, 4]
        scans = _scans(("a", [(2 * i, c) for i, c in enumerate(peak)]),
                       ("b", [(2 * i + 1, c) for i, c in enumerate(peak)]))
        features = _elution({"a": 9.0, "b": 5.0}, scans, _cycles(4, windows=2))
        assert features["elution_corr_to_best_rank2"].to_list() == pytest.approx([1.0])

    def test_too_short_to_correlate_still_overlaps(self):
        scans = _scans(("a", [(0, 5), (1, 9)]), ("b", [(1, 3)]))
        features = _elution({"a": 9.0, "b": 5.0}, scans, _cycles(2))
        assert features["elution_corr_mean"].to_list() == [None]
        assert features["elution_corr_to_best_rank2"].to_list() == [None]
        assert features["elution_overlap_to_best_rank2"].to_list() == [0.5]

    def test_ranked_by_hyperscore_and_each_against_the_rest(self):
        peak = [1, 4, 9, 4, 1]
        scans = _scans(("a", list(enumerate(peak))), ("b", list(enumerate(peak))),
                       ("c", [(0, 9), (4, 9)]))
        features = _elution({"a": 2.0, "b": 9.0, "c": 5.0}, scans, _cycles(5))
        # b ranks first; c (rank 2) is out of step with it, a (rank 3) is in step
        assert features["elution_corr_to_best_rank3"].to_list() == pytest.approx([1.0])
        assert features["elution_corr_to_best_rank2"].to_list()[0] < 0
        assert features["elution_corr_to_rest_rank2"].to_list()[0] < 0
        # and over the channels: the worst agreement, and the best
        assert features["elution_corr_to_best_min"].to_list()[0] < 0
        assert features["elution_corr_to_best_max"].to_list() == pytest.approx([1.0])

    def test_ratio_stability(self):
        # b holds half of a's height throughout; c rises as a falls
        scans = _scans(("a", list(enumerate([1, 4, 9, 4, 1]))), ("b", list(enumerate([0.5, 2, 4.5, 2, 0.5]))),
                       ("c", list(enumerate([9, 4, 1, 4, 9]))))
        features = _elution({"a": 9.0, "b": 5.0, "c": 2.0}, scans, _cycles(5))
        assert features["elution_ratio_sd_to_best_rank2"].to_list() == pytest.approx([0.0])
        assert features["elution_ratio_sd_to_best_rank3"].to_list()[0] > 1.0

    def test_the_decoy_channel_elutes_apart_without_moving_the_real_ones(self):
        peak = [1, 4, 9, 4, 1]
        scans = _scans(("a", list(enumerate(peak))), ("b", list(enumerate(peak))),
                       ("d", [(6, 5), (7, 9), (8, 5)]))
        rows = _rows(untag_prec=["A_2"] * 3, seq=["a", "b", "d"], z=[2] * 3, hyperscore=[9.0, 5.0, 7.0],
                     is_decoy_channel=[False, False, True], spec_id=[2, 2, 7])
        features = set_features(rows, scans, _cycles(9))
        alone = _elution({"a": 9.0, "b": 5.0}, scans.filter(pl.col("seq") != "d"), _cycles(9), apexes=[2, 2])
        for column in ("n_channels", "elution_cycles", "elution_corr_mean", "elution_overlap_to_best_rank2"):
            assert features[column].to_list() == alone[column].to_list()
        assert features["decoy_channel_elution_overlap_to_best"].to_list() == [0.0]
        assert features["decoy_channel_elution_corr_to_rest"].to_list()[0] < 0
        assert features["decoy_channel_elution_height_at_set_apex"].to_list() == [0.0]
        assert features["decoy_channel_elution_apex_from_set"].to_list() == [5.0]

    def test_apex_agreement(self):
        # a and b peak at cycle 2, as the set does; c peaks at cycle 4, and its
        # own apex is there
        scans = _scans(("a", list(enumerate([1, 4, 9, 4, 1]))), ("b", list(enumerate([1, 4, 9, 4, 1]))),
                       ("c", list(enumerate([0.5, 1, 2, 3, 9]))))
        features = _elution({"a": 9.0, "b": 5.0, "c": 2.0}, scans, _cycles(5), apexes=[2, 2, 4])
        assert features["elution_apex_from_best_rank2"].to_list() == [0.0]
        assert features["elution_apex_from_best_rank3"].to_list() == [2.0]
        assert features["elution_apex_from_set_max"].to_list() == [2.0]
        assert features["elution_apex_agree"].to_list() == pytest.approx([2 / 3])
        # c is at 2/9 of its own peak where the set peaks
        assert features["elution_height_at_set_apex_min"].to_list() == pytest.approx([2 / 9])
        assert features["elution_height_at_set_apex_max"].to_list() == pytest.approx([1.0])


def _fragments(channels, library=None, errors=None, decoy_channel=None):
    """One set A_2, N-terminally tagged ("(TAG-<n>)PEPTIDER": every b ion
    carries the tag, no y ion does), and its apex fragments: {seq: (hyperscore,
    {fragment name: observed intensity})}, the library's intensity of each
    from *library* ({fragment name: intensity}; 1 by default), every
    fragment's mass error from *errors* ({seq: error}; 0 by default), and
    *decoy_channel* the seq of the decoy channel's row."""
    library, errors = library or {}, errors or {}
    seqs = list(channels)
    rows = _rows(untag_prec=["A_2"] * len(seqs), seq=seqs, z=[2] * len(seqs),
                 hyperscore=[h for h, _ in channels.values()], __fdc_idx=list(range(len(seqs))),
                 is_decoy_channel=[seq == decoy_channel for seq in seqs])
    list_cols = pl.DataFrame({
        "__fdc_idx": list(range(len(seqs))),
        "frag_names": [list(encode_frag_names(list(f))) for _, f in channels.values()],
        "obs_int": [[float(v) for v in f.values()] for _, f in channels.values()],
        "frag_int": [[float(library.get(name, 1.0)) for name in f] for _, f in channels.values()],
        "frag_errors": [[float(errors.get(seq, 0.0))] * len(f) for seq, (_, f) in channels.items()],
    }, schema={"__fdc_idx": pl.Int64, "frag_names": pl.List(pl.Int32), "obs_int": pl.List(pl.Float64),
               "frag_int": pl.List(pl.Float64), "frag_errors": pl.List(pl.Float64)})
    return set_features(rows, list_cols=list_cols, label_name="TAG")


@pytest.mark.usefixtures("rank_arrays")
class TestFragmentFeatures:
    """Each channel's channel-specific fragments at its apex, against the others'."""

    def test_channels_that_fragment_alike_correlate(self):
        # the b ions scale together; the y ions (shared: the same peak in every
        # channel) disagree, and are not counted
        features = _fragments({
            "(TAG-0)PEPTIDER": (9.0, {"b2_1": 10, "b3_1": 100, "b4_1": 1000, "y3_1": 5, "y4_1": 5}),
            "(TAG-4)PEPTIDER": (5.0, {"b2_1": 20, "b3_1": 200, "b4_1": 2000, "y3_1": 50, "y4_1": 1}),
        })
        assert features["frag_corr_mean"].to_list() == pytest.approx([1.0])
        assert features["frag_corr_to_best_rank2"].to_list() == pytest.approx([1.0])
        assert features["frag_overlap_mean"].to_list() == [1.0]
        assert features["frag_n_specific_min"].to_list() == [3.0]

    def test_the_library_pattern_alone_is_not_agreement(self):
        # both follow the library's 10x steps, so the raw intensities correlate;
        # their departures from it are opposite
        library = {"b2_1": 10, "b3_1": 100, "b4_1": 1000, "b5_1": 10000}
        features = _fragments({
            "(TAG-0)PEPTIDER": (9.0, {n: v * f for (n, v), f in zip(library.items(), [1, 2, 1, 2])}),
            "(TAG-4)PEPTIDER": (5.0, {n: v * f for (n, v), f in zip(library.items(), [2, 1, 2, 1])}),
        }, library)
        assert features["frag_corr_mean"].to_list()[0] > 0.95
        assert features["frag_resid_corr_mean"].to_list() == pytest.approx([-1.0])
        assert features["frag_resid_corr_to_best_rank2"].to_list() == pytest.approx([-1.0])

    def test_mass_errors_agree_across_channels(self):
        fragments = {"b2_1": 10, "b3_1": 100, "b4_1": 1000, "y3_1": 5}
        features = _fragments({"(TAG-0)PEPTIDER": (9.0, fragments), "(TAG-4)PEPTIDER": (5.0, fragments),
                               "(TAG-8)PEPTIDER": (2.0, fragments)},
                              errors={"(TAG-0)PEPTIDER": 2e-6, "(TAG-4)PEPTIDER": 5e-6, "(TAG-8)PEPTIDER": 3e-6})
        assert features["frag_error_range"].to_list() == pytest.approx([3e-6])
        assert features["frag_error_from_best_max"].to_list() == pytest.approx([3e-6])
        assert features["frag_error_from_best_min"].to_list() == pytest.approx([1e-6])

    def test_the_decoy_channel_against_the_best_real_one(self):
        fragments = {"b2_1": 10, "b3_1": 100, "b4_1": 1000}
        features = _fragments({"(TAG-0)PEPTIDER": (9.0, fragments), "(TAG-4)PEPTIDER": (5.0, fragments),
                               "(TAG-8)PEPTIDER": (7.0, {"b2_1": 1000, "b3_1": 100, "b4_1": 10})},
                              errors={"(TAG-8)PEPTIDER": 4e-6}, decoy_channel="(TAG-8)PEPTIDER")
        # the real channels agree, untouched by it
        assert features["frag_corr_mean"].to_list() == pytest.approx([1.0])
        assert features["frag_n_specific_max"].to_list() == [3.0]
        assert features["decoy_channel_frag_corr_to_best"].to_list() == pytest.approx([-1.0])
        assert features["decoy_channel_frag_overlap_to_best"].to_list() == [1.0]
        assert features["decoy_channel_frag_error_from_best"].to_list() == pytest.approx([4e-6])

    def test_too_few_shared_fragments_to_correlate_still_overlap(self):
        features = _fragments({
            "(TAG-0)PEPTIDER": (9.0, {"b2_1": 10, "b3_1": 100, "b4_1": 1000}),
            "(TAG-4)PEPTIDER": (5.0, {"b2_1": 20, "b3_1": 200, "b5_1": 7}),
        })
        assert features["frag_corr_mean"].to_list() == [None]
        assert features["frag_overlap_to_best_rank2"].to_list() == pytest.approx([0.5])

def _separable_sets(n_targets=300, n_decoys=200, seed=0):
    """Three-channel sets: targets whose channels agree (one RT) and match
    better, decoys whose channels are scattered.  PredVal is noise."""
    rng = np.random.default_rng(seed)
    frames = []
    for i in range(n_targets + n_decoys):
        decoy = i >= n_targets
        rt = rng.uniform(5, 60, 3) if decoy else np.full(3, rng.uniform(5, 60)) + rng.normal(0, 0.01, 3)
        hyperscore = rng.uniform(0.0, 6.0, 3) if decoy else rng.uniform(4.0, 10.0, 3)
        frames.append(_rows(untag_prec=[f"P{i}_2"] * 3, is_decoy=[decoy] * 3, PredVal=rng.uniform(0, 1, 3),
                            hyperscore=hyperscore, rt=rt, stripped_seq=[f"P{i}"] * 3))
    return pd.concat(frames, ignore_index=True)


class TestFitPlexScoring:

    def test_separates_targets_from_decoys(self):
        features = set_features(_separable_sets())
        score, q, importance = fit_plex_scoring(features, model_columns(features), 1.0, model_type="xg",
                                                n_iterations=1)
        is_decoy = features["is_decoy"].to_numpy()
        assert score[~is_decoy].mean() > score[is_decoy].mean()
        assert (q[~is_decoy] <= 0.01).mean() > 0.5
        assert len(importance) > 0

    def test_fitted_with_its_own_xgboost_parameters(self, monkeypatch):
        # depth 6, over the precursor scoring's defaults
        from src import fdr_analysis
        made = []
        real = fdr_analysis.score_model

        def recording(model_type, **kwargs):
            made.append(kwargs)
            return real(model_type, **kwargs)

        monkeypatch.setattr(fdr_analysis, "score_model", recording)
        features = set_features(_separable_sets())
        fit_plex_scoring(features, model_columns(features), 1.0, model_type="xg", n_iterations=2)
        assert len(made) == 2
        assert all(kwargs.get("xgb_params", {}).get("max_depth") == 6 for kwargs in made)

    def test_too_few_decoy_sets_fits_nothing(self, app_log):
        features = set_features(_separable_sets(n_targets=50, n_decoys=MIN_DECOYS - 1))
        assert fit_plex_scoring(features, model_columns(features), 1.0) is None
        assert any("No PlexSet_Qvalue" in r.getMessage() for r in app_log)


class TestRunPlexScoring:

    def test_every_row_gets_its_sets_values(self, monkeypatch, tmp_path):
        rows = _rows(untag_prec=["A_2", "A_2", "B_2", "C_2"], is_decoy=[False, False, True, False],
                     is_decoy_channel=[False, True, False, True], PredVal=[0.9, 0.99, 0.5, 0.95])

        def fitted(features, columns, ratio):
            n = features.height
            return np.arange(n, dtype=float), np.full(n, 0.5), pd.DataFrame(
                {"share_of_gain": [1.0], "share_of_gain_sd": [0.0]}, index=pd.Index(["x"], name="feature"))

        monkeypatch.setattr(plex_scoring, "fit_plex_scoring", fitted)
        score, q = run_plex_scoring(rows, 1.0, str(tmp_path), 0.01)
        # A's decoy channel row gets A's values; C, seen only in the decoy channel, has none
        assert score.tolist()[:3] == [0.0, 0.0, 1.0]
        assert np.isnan(score.iloc[3]) and np.isnan(q.iloc[3])
        assert q.tolist()[:3] == [0.5, 0.5, 0.5]
        written = pl.read_parquet(os.path.join(tmp_path, FOLDER, FEATURE_FILE))
        assert written["PlexSet_Qvalue"].to_list() == [0.5, 0.5]

    def test_nothing_fitted_leaves_nan_and_still_writes_the_table(self, tmp_path):
        rows = _rows(untag_prec=["A_2", "B_2"], is_decoy=[False, True], PredVal=[0.9, 0.5])
        score, q = run_plex_scoring(rows, 1.0, str(tmp_path), 0.01)
        assert score.isna().all() and q.isna().all()
        assert os.path.exists(os.path.join(tmp_path, FOLDER, FEATURE_FILE))


class TestPlotFeatureDistributions:
    """The top features' distributions, IDs vs every target vs the decoys."""

    @staticmethod
    def _features(n=40):
        rng = np.random.default_rng(0)
        is_decoy = np.arange(n) % 2 == 1
        return pl.DataFrame({"is_decoy": is_decoy,
                             "PlexSet_Qvalue": np.where(is_decoy, np.nan, np.linspace(0, 0.05, n)),
                             "good": np.where(is_decoy, rng.normal(0, 1, n), rng.normal(3, 1, n)),
                             # a placeholder made null: missing in most sets
                             "sparse": [1.0 if i % 5 == 0 else None for i in range(n)]})

    @staticmethod
    def _importance(*features):
        return pd.DataFrame({"share_of_gain": np.linspace(1, 0.1, len(features)), "share_of_gain_sd": 0.0},
                            index=pd.Index(list(features), name="feature"))

    def test_both_plots_written(self, tmp_path):
        plot_feature_distributions(self._features(), self._importance("good", "sparse"), str(tmp_path))
        assert os.path.exists(tmp_path / ECDF_PLOT) and os.path.exists(tmp_path / HIST_PLOT)

    def test_features_not_in_the_table_or_never_valued_are_left_out(self, tmp_path):
        features = self._features().with_columns(empty=pl.lit(None, dtype=pl.Float64))
        plot_feature_distributions(features, self._importance("not_there", "empty"), str(tmp_path))
        assert not os.listdir(tmp_path)

    def test_nothing_without_a_plex_set_qvalue(self, tmp_path):
        plot_feature_distributions(self._features().drop("PlexSet_Qvalue"), self._importance("good"),
                                   str(tmp_path))
        assert not os.listdir(tmp_path)


class TestProteinQvalues:
    """Protein_PlexSet_Qvalue: each protein by its best plex set."""

    def _ids(self, **columns):
        n = len(columns["protein"])
        return pd.DataFrame({"is_decoy": [False] * n, "is_decoy_channel": [False] * n, **columns})

    def test_each_protein_by_its_best_set(self):
        # best scores: P1 0.9, P2 0.8, decoy P1 0.7, P3 0.6.  Counted down that
        # order: 1/1, 1/2, 2/2, 2/3, monotonized
        ids = self._ids(protein=["P1", "P1", "P2", "P1", "P3"], is_decoy=[False, False, False, True, False],
                        PlexSetScore=[0.9, 0.2, 0.8, 0.7, 0.6])
        q = protein_qvalues(ids, 1.0)
        assert q.iloc[[0, 1, 2, 4]].tolist() == pytest.approx([0.5, 0.5, 0.5, 2 / 3])
        # a library decoy has none of its own (it shares its target's name)
        assert np.isnan(q.iloc[3])

    def test_the_decoy_channel_is_left_out(self):
        # P2's decoy-channel row would make it the best protein
        ids = self._ids(protein=["P1", "P2", "P2", "D"], is_decoy=[False, False, False, True],
                        is_decoy_channel=[False, False, True, False], PlexSetScore=[0.9, 0.5, 0.99, 0.7])
        q = protein_qvalues(ids, 1.0)
        # P1 0.9, decoy D 0.7, P2 0.5: 1/1, 2/1, 2/2
        assert q.iloc[[0, 1]].tolist() == pytest.approx([1.0, 1.0])
        assert np.isnan(q.iloc[2])

    def test_the_run_ratio_scales_them(self):
        ids = self._ids(protein=["P1", "P2", "D"], is_decoy=[False, False, True], PlexSetScore=[0.9, 0.8, 0.1])
        assert protein_qvalues(ids, 0.5).iloc[:2].tolist() == pytest.approx([0.25, 0.25])

    def test_nothing_without_a_plex_set_score(self):
        assert protein_qvalues(self._ids(protein=["P1"]), 1.0).isna().all()
