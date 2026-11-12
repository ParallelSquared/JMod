import numpy as np
import pandas as pd
import polars as pl

from src.channel_fdr import _AMINO_ACIDS
from src.channel_fdr_plots import (BEST_CHANNEL_PLOT, ECDF_FOLDER, IDS_PLOT, IMPORTANCE_PLOT, RETAINED_PLOT,
                                   best_channel_counts, feature_columns, ks_distance, panels, plot_channel_fdr,
                                   plot_feature_ecdfs, plot_feature_importance, retained_counts)


class TestFeatureEcdfs:
    def test_an_ecdf_of_every_feature(self, tmp_path):
        features = pl.DataFrame({
            "seq": ["A", "A", "B", "B"], "channel": [0.0, 20.0, 0.0, 20.0],
            "is_decoy_channel": [False, True, False, True],
            "cosine_chan": [0.9, 0.2, 0.8, None],            # continuous, a null
            "n_chan_frags": [5, 1, 4, 0],                    # integer
            "im_error": [None, None, None, None]})           # nothing to plot
        assert plot_feature_ecdfs(features, str(tmp_path)) == 2
        assert sorted(p.name for p in tmp_path.iterdir()) == [ECDF_FOLDER]
        assert sorted(p.name for p in (tmp_path / ECDF_FOLDER).iterdir()) == ["cosine_chan.png",
                                                                          "n_chan_frags.png"]

    def test_in_worker_processes(self, tmp_path):
        k, r = (_AMINO_ACIDS.index(aa) for aa in "KR")
        features = pl.DataFrame({
            "channel": [0.0, 20.0, 0.0, 20.0], "is_decoy_channel": [False, True, False, True],
            "last_aa_code": [k, k, r, r], "cosine_chan": [0.9, 0.2, 0.8, 0.1],
            "shared_frag_error": [None, None, 1e-6, 2e-6]})          # nothing in the K panel
        assert plot_feature_ecdfs(features, str(tmp_path), split="KR", workers=2) == 3
        assert sorted(p.name for p in (tmp_path / ECDF_FOLDER).iterdir()) == [
            "cosine_chan.png", "last_aa_code.png", "shared_frag_error.png"]

    def test_split_by_the_c_terminal_residue(self, tmp_path):
        k, r, a = (_AMINO_ACIDS.index(aa) for aa in "KRA")
        features = pl.DataFrame({
            "seq": list("KKRRA"), "channel": [0.0, 20.0, 0.0, 20.0, 0.0],
            "is_decoy_channel": [False, True, False, True, False],
            "last_aa_code": [k, k, r, r, a], "cosine_chan": [0.9, 0.2, 0.8, 0.1, 0.5]})
        (k_title, k_rows), (r_title, r_rows) = panels(features, "KR")
        assert (k_title, k_rows["seq"].to_list()) == ("C-terminal K", ["K", "K"])
        assert (r_title, r_rows["seq"].to_list()) == ("C-terminal R", ["R", "R"])     # not A
        assert panels(features, "all")[0][1].height == 5
        assert plot_feature_ecdfs(features, str(tmp_path), split="kr") == 2          # any case
        assert sorted(p.name for p in (tmp_path / ECDF_FOLDER).iterdir()) == ["cosine_chan.png",
                                                                          "last_aa_code.png"]

    def test_an_unknown_split_plots_nothing(self, tmp_path):
        features = pl.DataFrame({"channel": [0.0], "is_decoy_channel": [False], "cosine_chan": [0.9]})
        assert plot_feature_ecdfs(features, str(tmp_path), split="KQ") == 0
        assert not (tmp_path / ECDF_FOLDER).exists()

    def test_only_the_identity_columns_are_not_features(self):
        columns = ["seq", "z", "protein", "channel", "is_decoy_channel", "BestChannel_Qvalue", "PredVal",
                   "cosine_chan"]
        assert feature_columns(pl.DataFrame({c: [1.0] for c in columns})) == ["z", "PredVal", "cosine_chan"]

    def test_ks_distance(self):
        assert ks_distance(np.array([1.0, 2.0]), np.array([1.0, 2.0])) == 0.0
        assert ks_distance(np.array([1.0, 2.0]), np.array([3.0, 4.0])) == 1.0
        assert np.isnan(ks_distance(np.array([1.0]), np.zeros(0)))
    def test_the_flag_alone_is_all(self):
        import argparse
        from src.config import _add_arguments
        parser = argparse.ArgumentParser()
        _add_arguments(parser)
        assert parser.parse_args([]).channel_fdr_plots == "off"
        assert parser.parse_args(["--channel_fdr_plots"]).channel_fdr_plots == "all"
        assert parser.parse_args(["--channel_fdr_plots", "KR"]).channel_fdr_plots == "KR"


class TestChannelFdrPlots:
    def _scored(self):
        # two sets; in B channels 0 and 20 tie for the highest PredVal
        return pl.DataFrame({
            "untag_prec": ["A", "A", "A", "B", "B", "B"], "seq": list("abcdef"),
            "channel": [0.0, 4.0, 20.0, 0.0, 4.0, 20.0],
            "is_decoy_channel": [False, False, True, False, False, True],
            "PredVal": [0.9, 0.99, 0.5, 1.0, 0.7, 1.0],
            "Channel_Qvalue": [0.02, 0.001, None, 0.005, 0.3, None]})

    def test_best_channel_by_predval_shares_a_tie(self):
        assert best_channel_counts(self._scored()) == {0.0: 0.5, 4.0: 1.0, 20.0: 0.5}

    def test_retained_counts_leave_out_the_decoy_channel(self):
        assert retained_counts(self._scored(), 0.01) == {0.0: (1, 2), 4.0: (1, 2)}
        assert retained_counts(self._scored(), 0.5) == {0.0: (2, 2), 4.0: (2, 2)}

    def test_every_channel_fdr_plot(self, tmp_path):
        plot_channel_fdr(self._scored(), str(tmp_path), importance=self._importance())
        for name in (BEST_CHANNEL_PLOT, IDS_PLOT, RETAINED_PLOT, IMPORTANCE_PLOT):
            assert (tmp_path / name).exists()
        assert (tmp_path / ECDF_FOLDER / "PredVal.png").exists()

    def test_without_ecdfs(self, tmp_path):
        plot_channel_fdr(self._scored(), str(tmp_path), ecdfs=False)
        assert (tmp_path / BEST_CHANNEL_PLOT).exists() and not (tmp_path / ECDF_FOLDER).exists()

    def test_no_ids_plot_without_channel_qvalues(self, tmp_path):
        plot_channel_fdr(self._scored().drop("Channel_Qvalue"), str(tmp_path))
        assert (tmp_path / BEST_CHANNEL_PLOT).exists() and not (tmp_path / IDS_PLOT).exists()
        assert not (tmp_path / RETAINED_PLOT).exists() and not (tmp_path / IMPORTANCE_PLOT).exists()

    def test_an_unknown_split_makes_no_plots(self, tmp_path):
        plot_channel_fdr(self._scored(), str(tmp_path), split="KQ")
        assert not list(tmp_path.iterdir())

    @staticmethod
    def _importance():
        return pd.DataFrame({"share_of_gain": [0.6, 0.3, 0.1], "share_of_gain_sd": [0.05, 0.02, 0.01]},
                            index=pd.Index(["ms1_cor", "cosine_chan", "z"], name="feature"))

    def test_feature_importance(self, tmp_path):
        plot_feature_importance(self._importance(), str(tmp_path), top=2)
        assert (tmp_path / IMPORTANCE_PLOT).exists()
