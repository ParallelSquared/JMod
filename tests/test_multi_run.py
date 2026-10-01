import os

import polars as pl
import pytest

from src.mass_tags import available_tags
from src.models.spec_lib.library_store import SpectrumLibraryStore
from src.multi_run import (_Units, collect_results, combine_runs, mbr_library_rts, mbr_targets,
                           precursors_per_run)
from tests.models.spec_lib.test_decoy_differential import _entry


def _write_run(tmp_path, run_name, rows, proteins=None, channels=None):
    """A run's results folder, with an outputs/all_IDs_filtered.parquet of *rows*.

    Each row is (sequence, is_decoy, PredVal, BestChannel_Qvalue); every
    precursor is charge 2, of protein P1 unless *proteins* names one per row,
    and label-free (channel 0) unless *channels* names one per row.
    """
    folder = tmp_path / run_name
    (folder / "outputs").mkdir(parents=True)
    seqs = [seq for seq, *_ in rows]
    n = len(rows)
    pl.DataFrame({
        "stripped_seq": seqs, "z": [2.0] * n, "untag_prec": [s + "_2" for s in seqs],
        "file_name": [run_name] * n, "channel": channels or [0] * n,
        "is_decoy": [decoy for _, decoy, _, _ in rows],
        "Qvalue": [q for *_, q in rows], "Protein_Qvalue": [0.001] * n,
        "PredVal": [score for _, _, score, _ in rows], "protein": proteins or ["P1"] * n,
        "BestChannel_Qvalue": [q for *_, q in rows], "plex_Area": [1.0] * n,
        "seq": seqs, "silac_channel": [float("nan")] * n, "untag_seq": seqs,
        "rt": [1.0] * n, "mz": [500.0] * n, "prec_im": [float("nan")] * n, "coeff": [1.0] * n,
    }).write_parquet(folder / "outputs" / "all_IDs_filtered.parquet")
    return str(folder)


def _global_q(precursor_table):
    return dict(precursor_table.select("untag_prec", "untag_prec_Global_Qvalue").iter_rows())


class TestGlobalQvalue:
    def test_matches_the_per_run_formula(self, tmp_path):
        # (1 + decoys) / targets * ratio down the score order, then monotonized
        run = _write_run(tmp_path, "a", [("AAA", False, 0.9, 0.001), ("BBB", False, 0.8, 0.001),
                                         ("CCC", True, 0.7, 0.5), ("EEE", False, 0.6, 0.001)])
        _, precursors, _ = collect_results({1: run}, target_decoy_ratio=0.5, fdr_threshold=0.01)
        assert _global_q(precursors) == pytest.approx(
            {"AAA_2": 0.25, "BBB_2": 0.25, "CCC_2": 1 / 3, "EEE_2": 1 / 3})

    def test_best_score_across_runs_is_used(self, tmp_path):
        runs = {1: _write_run(tmp_path, "a", [("AAA", False, 0.1, 0.001)]),
                2: _write_run(tmp_path, "b", [("AAA", False, 0.9, 0.001)])}
        _, precursors, _ = collect_results(runs, target_decoy_ratio=1.0, fdr_threshold=0.01)
        assert precursors.select("MaxPredVal", "BestRun").row(0) == (pytest.approx(0.9), 2)

    @pytest.mark.parametrize("rows", [
        [("AAA", False, 0.9, 0.001), ("DDD", True, 0.9, 0.5)],
        [("DDD", True, 0.9, 0.5), ("AAA", False, 0.9, 0.001)],
    ], ids=["target_first", "decoy_first"])
    def test_tied_scores_share_a_q_value(self, tmp_path, rows):
        # One target and one decoy at the same score: (1 + 1) / 1, whichever comes first
        _, precursors, _ = collect_results({1: _write_run(tmp_path, "a", rows)},
                                        target_decoy_ratio=1.0, fdr_threshold=0.01)
        assert _global_q(precursors) == {"AAA_2": 2.0, "DDD_2": 2.0}


class TestGlobalProteinQvalue:
    def test_a_protein_is_ranked_by_its_best_precursor_in_any_run(self, tmp_path):
        # P2's best precursor (0.8, run 2) outranks P2's decoy (0.5); its other
        # precursor (0.3) would not.  The decoy shares P2's name and counts apart
        runs = {1: _write_run(tmp_path, "a", [("AAA", False, 0.9, 0.001), ("BBB", False, 0.3, 0.001),
                                              ("DDD", True, 0.5, 0.5)], proteins=["P1", "P2", "P2"]),
                2: _write_run(tmp_path, "b", [("CCC", False, 0.8, 0.001)], proteins=["P2"])}
        ids, _, _ = collect_results(runs, target_decoy_ratio=1.0, fdr_threshold=0.01)
        # P1 (1 target), P2 (2 targets), decoy (2 targets, 1 decoy), monotonized
        assert dict(ids.select("protein", "Protein_Global_Qvalue").unique().iter_rows()) == \
            pytest.approx({"P1": 0.5, "P2": 0.5})


class TestPrecursorsPerRun:
    def test_missing_area_is_left_out_of_the_log(self):
        df = pl.DataFrame({"run_idx": [1, 1, 2], "channel": [0, 0, 0], "untag_prec": ["AAA_2", "BBB_2", "AAA_2"],
                           "plex_Area": [8.0, float("nan"), 4.0], "Qvalue": [0.001] * 3,
                           "untag_prec_Global_Qvalue": [0.001, 0.5, 0.001]})
        rows = precursors_per_run(df, _Units(df, [1, 2]), fdr_threshold=0.01).sort("run_idx", "untag_prec")
        assert rows.select("run_idx", "untag_prec", "log2_area", "retained").rows() == [
            (1, "AAA_2", 3.0, True), (1, "BBB_2", None, False), (2, "AAA_2", 2.0, True)]

    def test_each_channel_of_a_run_is_its_own_unit(self):
        # AAA in two channels of run 1: two rows, areas not summed, and only
        # channel 4 passing on its own Qvalue (channel 0 through its best channel)
        df = pl.DataFrame({"run_idx": [1, 1, 2], "channel": [0, 4, 4], "untag_prec": ["AAA_2"] * 3,
                           "plex_Area": [8.0, 4.0, 2.0], "Qvalue": [0.5, 0.001, 0.001],
                           "untag_prec_Global_Qvalue": [0.001] * 3})
        units = _Units(df, [1, 2])
        assert units.labels == ["1·0", "1·4", "2·0", "2·4"]
        rows = precursors_per_run(df, units, fdr_threshold=0.01).sort("unit")
        assert rows.select("unit", "log2_area", "qvalue_pass").rows() == [
            (0, 3.0, False), (1, 2.0, True), (3, 1.0, True)]


def _first_pass_ids(rows):
    """A combined first-pass table of (run_idx, untag_prec, rt, PredVal) rows, all
    target IDs passing the run-level FDR."""
    run_idx, untag_prec, rt, pred_val = zip(*rows)
    n = len(rows)
    return pl.DataFrame({"run_idx": run_idx, "untag_prec": untag_prec, "rt": rt, "PredVal": pred_val,
                         "BestChannel_Qvalue": [0.001] * n, "untag_prec_Global_Qvalue": [0.001] * n,
                         "is_decoy": [False] * n})


class TestMbrLibraryRts:
    def test_runs_are_aligned_to_the_run_with_most_ids(self, tmp_path):
        # Run 2 has the most IDs; run 1 elutes everything 2 minutes later, and
        # also has one precursor run 2 lacks
        shared = [(f"P{i}_2", 10.0 + i) for i in range(40)]
        rows = ([(2, prec, rt, 0.9) for prec, rt in shared + [("REF_2", 60.0), ("REF2_2", 61.0)]]
                + [(1, prec, rt + 2.0, 0.9) for prec, rt in shared] + [(1, "ONLY_2", 72.0, 0.9)])
        rts = dict(mbr_library_rts(_first_pass_ids(rows), 0.01, str(tmp_path)).iter_rows())
        assert rts["P5_2"] == pytest.approx(15.0)      # the reference run's own RT
        assert rts["ONLY_2"] == pytest.approx(70.0)    # aligned onto the reference's time scale
        assert (tmp_path / "library_size_by_run.png").is_file()
        assert (tmp_path / "lowess" / "run_1.png").is_file()

    def test_best_scoring_channel_gives_the_rt(self, tmp_path):
        rows = [(1, "AAA_2", 10.0, 0.9), (1, "AAA_2", 12.0, 0.5)]  # e.g. two channels of one run
        assert mbr_library_rts(_first_pass_ids(rows), 0.01, str(tmp_path)).rows() == [("AAA_2", 10.0)]


class TestMbrTargets:
    def test_tagged_and_detagged_entries_match_on_untag_prec(self):
        frags = {'b2_1': [200.0, 1.0], 'y2_1': [300.0, 0.5]}
        targets = SpectrumLibraryStore.from_dict({
            ("PEPTIDEK", 2.0): _entry("PEPTIDEK", "PEPTIDEK", 464.7, 2.0, frags),
            ("(mTRAQ-0)ELVISK", 2.0): _entry("(mTRAQ-0)ELVISK", "ELVISK", 400.2, 2.0, frags),
            ("NOTFOUNDK", 2.0): _entry("NOTFOUNDK", "NOTFOUNDK", 500.3, 2.0, frags),
        })
        library_rts = pl.DataFrame({"untag_prec": ["ELVISK_2", "PEPTIDEK_2"], "rt": [40.0, 30.0]})
        mbr = mbr_targets(targets, library_rts, available_tags["mTRAQ"], None)
        assert list(zip(mbr.mod_seq, mbr.iRT)) == [("PEPTIDEK", 30.0), ("(mTRAQ-0)ELVISK", 40.0)]


class TestGroupedResults:
    def test_run_level_targets_are_kept_sorted_by_precursor(self, tmp_path):
        runs = {1: _write_run(tmp_path, "a", [("BBB", False, 0.9, 0.001), ("AAA", False, 0.8, 0.001),
                                              ("DDD", True, 0.7, 0.001), ("CCC", False, 0.1, 0.2)]),
                2: _write_run(tmp_path, "b", [("AAA", False, 0.9, 0.001)])}
        grouped, _, _ = collect_results(runs, target_decoy_ratio=1.0, fdr_threshold=0.01)
        # No decoy, nothing failing the run-level FDR, and no filter on the global q-value
        assert grouped.select("untag_prec", "run_idx").rows() == [("AAA_2", 1), ("AAA_2", 2), ("BBB_2", 1)]
        assert "untag_prec_Global_Qvalue" in grouped.columns

    def test_coeff_and_prec_im_are_kept_only_when_the_runs_have_them(self, tmp_path):
        with_them = _write_run(tmp_path, "a", [("AAA", False, 0.9, 0.001)])
        grouped, _, _ = collect_results({1: with_them}, target_decoy_ratio=1.0, fdr_threshold=0.01)
        assert {"coeff", "prec_im"} <= set(grouped.columns)

        without = _write_run(tmp_path, "b", [("AAA", False, 0.9, 0.001)])
        parquet = os.path.join(without, "outputs", "all_IDs_filtered.parquet")
        pl.read_parquet(parquet).drop("coeff", "prec_im").write_parquet(parquet)
        grouped, _, _ = collect_results({1: without}, target_decoy_ratio=1.0, fdr_threshold=0.01)
        assert not {"coeff", "prec_im"} & set(grouped.columns)

    def test_table_beside_every_plot_in_its_folder(self, tmp_path):
        rows = [("AAA", False, 0.9, 0.001), ("DDD", True, 0.1, 0.9)]
        runs = {i: _write_run(tmp_path, name, rows) for i, name in enumerate("abc", start=1)}
        experiment_dir = tmp_path / "experiment"
        experiment_dir.mkdir()
        combine_runs(runs, str(experiment_dir), target_decoy_ratio=1.0, fdr_threshold=0.01)
        assert sorted(p.name for p in (experiment_dir / "experiment_results").iterdir()) == [
            "01_precursors_per_run.png", "02_proteins_per_run.png", "03_precursor_data_completeness.png",
            "04_protein_data_completeness.png", "05_intensity_per_run.png", "06_run_correlation.png",
            "07_precursors_lost_to_global_q.png", "08_proteins_lost_to_global_q.png",
            "09_proteins_lost_by_precursor_count.png", "run_index.txt"]
        assert (experiment_dir / "combined_filtered_IDs.parquet").is_file()

    def test_channels_make_one_run_comparable(self, tmp_path):
        rows = [("AAA", False, 0.9, 0.001), ("AAA", False, 0.8, 0.001), ("DDD", True, 0.1, 0.9)]
        runs = {1: _write_run(tmp_path, "a", rows, channels=[0, 4, 0])}
        experiment_dir = tmp_path / "experiment"
        experiment_dir.mkdir()
        combine_runs(runs, str(experiment_dir), target_decoy_ratio=1.0, fdr_threshold=0.01)
        # One run of two channels is two (run, channel) units, enough to compare
        names = {p.name for p in (experiment_dir / "experiment_results").iterdir()}
        assert {"00_channel_colors.png", "03_precursor_data_completeness.png"} <= names

    def test_single_run_gets_only_the_per_run_plots(self, tmp_path):
        runs = {1: _write_run(tmp_path, "a", [("AAA", False, 0.9, 0.001), ("DDD", True, 0.1, 0.9)])}
        experiment_dir = tmp_path / "experiment"
        experiment_dir.mkdir()
        combine_runs(runs, str(experiment_dir), target_decoy_ratio=1.0, fdr_threshold=0.01)
        assert sorted(p.name for p in (experiment_dir / "experiment_results").iterdir()) == [
            "01_precursors_per_run.png", "02_proteins_per_run.png", "05_intensity_per_run.png",
            "run_index.txt"]

    def test_earlier_results_are_kept(self, tmp_path):
        runs = {1: _write_run(tmp_path, "a", [("AAA", False, 0.9, 0.001)])}
        experiment_dir = tmp_path / "experiment"
        (experiment_dir / "experiment_results").mkdir(parents=True)
        (experiment_dir / "combined_filtered_IDs.parquet").write_text("earlier experiment")
        combine_runs(runs, str(experiment_dir), target_decoy_ratio=1.0, fdr_threshold=0.01)
        assert (experiment_dir / "combined_filtered_IDs.parquet").read_text() == "earlier experiment"
        # Both datestamped
        assert len(list(experiment_dir.glob("combined_filtered_IDs_*.parquet"))) == 1
        assert len(list(experiment_dir.glob("experiment_results_*"))) == 1
