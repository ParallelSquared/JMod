import polars as pl
import pytest

from src.multi_run import collect_results, combine_runs


def _write_run(tmp_path, run_name, rows):
    """A run's results folder, with an outputs/all_IDs_filtered.parquet of *rows*.

    Each row is (sequence, is_decoy, PredVal, BestChannel_Qvalue); every
    precursor is charge 2, label-free.
    """
    folder = tmp_path / run_name
    (folder / "outputs").mkdir(parents=True)
    seqs = [seq for seq, *_ in rows]
    n = len(rows)
    pl.DataFrame({
        "stripped_seq": seqs, "z": [2.0] * n, "untag_prec": [s + "_2" for s in seqs],
        "file_name": [run_name] * n, "channel": [0] * n,
        "is_decoy": [decoy for _, decoy, _, _ in rows],
        "Qvalue": [q for *_, q in rows], "Protein_Qvalue": [0.001] * n,
        "PredVal": [score for _, _, score, _ in rows], "protein": ["P1"] * n,
        "BestChannel_Qvalue": [q for *_, q in rows], "plex_Area": [1.0] * n,
        "seq": seqs, "silac_channel": [float("nan")] * n, "untag_seq": seqs,
        "rt": [1.0] * n, "mz": [500.0] * n, "coeff": [1.0] * n,
    }).write_parquet(folder / "outputs" / "all_IDs_filtered.parquet")
    return str(folder)


def _global_q(precursor_table):
    return dict(precursor_table.select("untag_prec", "untag_prec_Global_Qvalue").iter_rows())


class TestGlobalQvalue:
    def test_matches_the_per_run_formula(self, tmp_path):
        # (1 + decoys) / targets * ratio down the score order, then monotonized
        run = _write_run(tmp_path, "a", [("AAA", False, 0.9, 0.001), ("BBB", False, 0.8, 0.001),
                                         ("CCC", True, 0.7, 0.5), ("EEE", False, 0.6, 0.001)])
        _, precursors = collect_results({1: run}, target_decoy_ratio=0.5, fdr_threshold=0.01)
        assert _global_q(precursors) == pytest.approx(
            {"AAA_2": 0.25, "BBB_2": 0.25, "CCC_2": 1 / 3, "EEE_2": 1 / 3})

    def test_best_score_across_runs_is_used(self, tmp_path):
        runs = {1: _write_run(tmp_path, "a", [("AAA", False, 0.1, 0.001)]),
                2: _write_run(tmp_path, "b", [("AAA", False, 0.9, 0.001)])}
        _, precursors = collect_results(runs, target_decoy_ratio=1.0, fdr_threshold=0.01)
        assert precursors.select("MaxPredVal", "BestRun").row(0) == (pytest.approx(0.9), 2)

    @pytest.mark.parametrize("rows", [
        [("AAA", False, 0.9, 0.001), ("DDD", True, 0.9, 0.5)],
        [("DDD", True, 0.9, 0.5), ("AAA", False, 0.9, 0.001)],
    ], ids=["target_first", "decoy_first"])
    def test_tied_scores_share_a_q_value(self, tmp_path, rows):
        # One target and one decoy at the same score: (1 + 1) / 1, whichever comes first
        _, precursors = collect_results({1: _write_run(tmp_path, "a", rows)},
                                        target_decoy_ratio=1.0, fdr_threshold=0.01)
        assert _global_q(precursors) == {"AAA_2": 2.0, "DDD_2": 2.0}


class TestGroupedResults:
    def test_run_level_targets_are_kept_sorted_by_precursor(self, tmp_path):
        runs = {1: _write_run(tmp_path, "a", [("BBB", False, 0.9, 0.001), ("AAA", False, 0.8, 0.001),
                                              ("DDD", True, 0.7, 0.001), ("CCC", False, 0.1, 0.2)]),
                2: _write_run(tmp_path, "b", [("AAA", False, 0.9, 0.001)])}
        grouped, _ = collect_results(runs, target_decoy_ratio=1.0, fdr_threshold=0.01)
        # No decoy, nothing failing the run-level FDR, and no filter on the global q-value
        assert grouped.select("untag_prec", "run_idx").rows() == [("AAA_2", 1), ("AAA_2", 2), ("BBB_2", 1)]
        assert "untag_prec_Global_Qvalue" in grouped.columns

    def test_table_and_plots_are_written(self, tmp_path):
        runs = {1: _write_run(tmp_path, "a", [("AAA", False, 0.9, 0.001), ("DDD", True, 0.1, 0.9)]),
                2: _write_run(tmp_path, "b", [("AAA", False, 0.8, 0.001), ("DDD", True, 0.2, 0.9)])}
        experiment_dir = tmp_path / "experiment"
        experiment_dir.mkdir()
        combine_runs(runs, str(experiment_dir), target_decoy_ratio=1.0, fdr_threshold=0.01)
        assert sorted(p.name for p in experiment_dir.iterdir()) == [
            "grouped_best_score_run.png", "grouped_lost_to_global_q.png", "grouped_results.parquet"]
