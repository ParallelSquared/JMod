import json
import logging
import os
from types import SimpleNamespace

import pytest

import src.config as config
import src.run_jmod as run_jmod
from src.utils.errors import JModError


@pytest.fixture
def experiment(monkeypatch, tmp_path, combined):
    """main() over two data files, with the library build stubbed out.

    Each test supplies its own process_run.
    """
    monkeypatch.setattr(config.args, "mzml", ["a.mzML", "b.mzML"])
    monkeypatch.setattr(config.args, "speclib", "lib.tsv")
    monkeypatch.setattr(config.args, "output_folder", str(tmp_path))
    monkeypatch.setattr(config.args, "dummy_value", None)
    monkeypatch.setattr(config.args, "config_json", None)
    monkeypatch.setattr(config.args, "plexDIA", config.args.plexDIA)  # main may set it
    monkeypatch.setattr(run_jmod, "set_log_filepath", lambda path: None)
    monkeypatch.setattr(run_jmod, "resolve_tags", lambda: (None, None))
    monkeypatch.setattr(run_jmod, "build_library",
                        lambda *args: SimpleNamespace(target_decoy_ratio=1.0))
    return run_jmod


@pytest.fixture
def combined(monkeypatch):
    """The calls main() makes to combine_runs, recorded instead of run: the
    stubbed runs write no outputs to combine."""
    calls = []

    def combine_runs(*args):
        calls.append(args)
        return "combined IDs"  # stands in for the combined table

    monkeypatch.setattr(run_jmod, "combine_runs", combine_runs)
    return calls


def _finish(runState):
    """What a process_run that completes leaves on runState."""
    runState.results_folder = runState.file_name + "_results"


def _errors(records):
    return [r for r in records if r.levelno >= logging.ERROR]


class TestErrorHandling:
    def test_all_runs_succeeding_logs_no_errors(self, experiment, monkeypatch, app_log):
        ran = []

        def process_run(runState, *rest):
            ran.append(runState.file_name)
            _finish(runState)

        monkeypatch.setattr(experiment, "process_run", process_run)
        experiment.main()
        assert ran == ["a.mzML", "b.mzML"]
        assert _errors(app_log) == []

    def test_failed_run_does_not_stop_the_next_one(self, experiment, monkeypatch):
        ran = []

        def process_run(runState, *rest):
            ran.append(runState.file_name)
            if runState.file_name == "a.mzML":
                raise JModError("bad file")
            _finish(runState)

        monkeypatch.setattr(experiment, "process_run", process_run)
        experiment.main()
        assert ran == ["a.mzML", "b.mzML"]

    def test_failed_run_folder_is_renamed(self, experiment, monkeypatch, tmp_path):
        def process_run(runState, *rest):
            folder = tmp_path / (runState.file_name + "_results")
            folder.mkdir()
            runState.results_folder = str(folder)
            if runState.file_name == "a.mzML":
                raise JModError("bad file")

        monkeypatch.setattr(experiment, "process_run", process_run)
        experiment.main()
        assert (tmp_path / "run_failed_a.mzML_results").is_dir()
        assert not (tmp_path / "a.mzML_results").exists()
        assert (tmp_path / "b.mzML_results").is_dir()  # the run that succeeded keeps its name

    def test_failed_run_is_logged_with_its_file(self, experiment, monkeypatch, app_log):
        def process_run(runState, *rest):
            if runState.file_name == "a.mzML":
                raise JModError("bad file")
            _finish(runState)

        monkeypatch.setattr(experiment, "process_run", process_run)
        experiment.main()
        assert any("Run 1 of 2 failed (a.mzML): bad file" in r.getMessage()
                   for r in _errors(app_log))

    def test_missing_data_file_is_a_user_error(self, tmp_path):
        runState = run_jmod.RunState()
        runState.file_name = str(tmp_path / "missing.mzML")
        with pytest.raises(JModError, match="Data file not found"):
            run_jmod.process_run(runState, None, None, None, None, str(tmp_path / "experiment"))
        assert list(tmp_path.iterdir()) == []  # fails before any results folder is made

    def test_error_outside_the_runs_stops_the_experiment(self, experiment, monkeypatch, app_log):
        def build_library(*args):
            raise JModError("bad library")

        ran = []
        monkeypatch.setattr(experiment, "build_library", build_library)
        monkeypatch.setattr(experiment, "process_run", lambda runState, *rest: ran.append(runState))
        experiment.main()
        assert ran == []
        assert any("JMod stopped: bad library" in r.getMessage() for r in _errors(app_log))


class TestExperimentFolder:
    def test_everything_goes_in_a_new_jmod_results_folder(self, experiment, monkeypatch, tmp_path):
        parents = []

        def process_run(runState, *rest):
            parents.append(rest[-1])
            _finish(runState)

        monkeypatch.setattr(experiment, "process_run", process_run)
        experiment.main()
        assert parents == [str(tmp_path / "JMod_Results")] * 2  # the runs' results folders go in it
        assert (tmp_path / "JMod_Results" / "JMod_config.json").is_file()

    def test_dummy_value_names_the_folder(self, experiment, monkeypatch, tmp_path):
        monkeypatch.setattr(config.args, "dummy_value", "batch2")
        monkeypatch.setattr(experiment, "process_run", lambda runState, *rest: _finish(runState))
        experiment.main()
        assert (tmp_path / "JMod_Results_batch2" / "JMod_config.json").is_file()

    def test_an_earlier_experiments_folder_is_kept(self, experiment, monkeypatch, tmp_path):
        (tmp_path / "JMod_Results").mkdir()
        (tmp_path / "JMod_Results" / "JMod_config.json").write_text("earlier experiment")
        monkeypatch.setattr(experiment, "process_run", lambda runState, *rest: _finish(runState))
        experiment.main()
        assert (tmp_path / "JMod_Results" / "JMod_config.json").read_text() == "earlier experiment"
        assert len(list(tmp_path.glob("JMod_Results_*"))) == 1  # datestamped


class TestExperimentConfig:
    def test_config_lists_every_data_file(self, experiment, monkeypatch, tmp_path):
        monkeypatch.setattr(experiment, "process_run", lambda runState, *rest: _finish(runState))
        experiment.main()
        written = json.loads((tmp_path / "JMod_Results" / "JMod_config.json").read_text())
        assert written["mzml"] == ["a.mzML", "b.mzML"]

    def test_folder_is_recorded_as_its_files(self, experiment, monkeypatch, tmp_path):
        folder = tmp_path / "data"
        folder.mkdir()
        (folder / "c.mzML").write_text("")
        monkeypatch.setattr(config.args, "mzml_folder", str(folder))
        monkeypatch.setattr(experiment, "process_run", lambda runState, *rest: _finish(runState))
        experiment.main()
        written = json.loads((tmp_path / "JMod_Results" / "JMod_config.json").read_text())
        assert written["mzml"] == ["a.mzML", "b.mzML", f"{folder}/c.mzML"]
        assert written["mzml_folder"] is None


class TestCombinedResults:
    def test_completed_runs_are_combined(self, experiment, monkeypatch, combined):
        def process_run(runState, *rest):
            if runState.file_name == "a.mzML":
                raise JModError("bad file")
            _finish(runState)

        monkeypatch.setattr(experiment, "process_run", process_run)
        experiment.main()
        ((run_folders, _, target_decoy_ratio, _),) = combined
        assert run_folders == {2: "b.mzML_results"}  # the failed run 1 is left out
        assert target_decoy_ratio == 1.0

    def test_nothing_is_combined_when_every_run_fails(self, experiment, monkeypatch, combined):
        def process_run(runState, *rest):
            raise JModError("bad file")

        monkeypatch.setattr(experiment, "process_run", process_run)
        experiment.main()
        assert combined == []


class TestMatchBetweenRuns:
    @pytest.fixture
    def mbr_experiment(self, experiment, monkeypatch):
        """The two-file experiment with --mbr, recording each run's results
        parent and use_emp_rt, and the input to build_mbr_library."""
        monkeypatch.setattr(config.args, "mbr", True)
        monkeypatch.setattr(config.args, "use_emp_rt", False)  # the final pass sets it
        runs, mbr_inputs = [], []

        def process_run(runState, *rest):
            runs.append((runState.file_name, rest[-1], config.args.use_emp_rt))
            _finish(runState)

        def build_mbr_library(combined_ids, *rest):
            mbr_inputs.append(combined_ids)
            return SimpleNamespace(target_decoy_ratio=1.0)

        monkeypatch.setattr(experiment, "process_run", process_run)
        monkeypatch.setattr(experiment, "build_mbr_library", build_mbr_library)
        return SimpleNamespace(runs=runs, mbr_inputs=mbr_inputs)

    def test_first_pass_goes_in_first_pass_and_the_final_pass_keeps_normal_names(
            self, experiment, mbr_experiment, combined, tmp_path):
        experiment.main()
        experiment_dir = str(tmp_path / "JMod_Results")
        first_pass = os.path.join(experiment_dir, "first_pass")
        assert mbr_experiment.runs == [("a.mzML", first_pass, False), ("b.mzML", first_pass, False),
                                       ("a.mzML", experiment_dir, True), ("b.mzML", experiment_dir, True)]
        assert [args[1] for args in combined] == [first_pass, experiment_dir]
        assert mbr_experiment.mbr_inputs == ["combined IDs"]  # built from the first pass
        assert (tmp_path / "JMod_Results" / "mbr_library").is_dir()

    def test_single_file_skips_mbr(self, experiment, mbr_experiment, monkeypatch, app_log, tmp_path):
        monkeypatch.setattr(config.args, "mzml", ["a.mzML"])
        experiment.main()
        assert mbr_experiment.runs == [("a.mzML", str(tmp_path / "JMod_Results"), False)]
        assert any("skipping match between runs" in r.getMessage() for r in app_log)


class TestMakeLibrary:
    """--make_library builds a finished experiment's MBR library, without searching."""

    @pytest.fixture
    def finished(self, monkeypatch, tmp_path, combined):
        """An experiment folder with runs a and b, the library step recorded
        instead of run.  Tests add the experiment's files."""
        for name in ("config_json", "speclib", "mzml", "make_library", "plexDIA"):
            monkeypatch.setattr(config.args, name, getattr(config.args, name))  # restored after
        monkeypatch.setattr(config, "cli_args", {})
        monkeypatch.setattr(run_jmod, "set_log_filepath", lambda path: None)
        monkeypatch.setattr(run_jmod, "resolve_tags", lambda: (None, None))
        library_inputs = []
        monkeypatch.setattr(run_jmod, "write_mbr_library",
                            lambda ids, mbr_dir, *tags: library_inputs.append((ids, config.args.speclib)))
        experiment_dir = tmp_path / "experiment"
        library = tmp_path / "lib.tsv"
        library.write_text("")  # only its existence is checked; loading it is stubbed
        for run in ("a", "b"):
            outputs = experiment_dir / f"{run}_results" / "outputs"
            outputs.mkdir(parents=True)
            (outputs / "all_IDs_filtered.parquet").write_text("")  # combine_runs is stubbed
            (outputs / "config.json").write_text(json.dumps({"speclib": str(library), "mzml": f"{run}.mzML"}))
            (outputs / "params.txt").write_text("Args\nppm: 10\n\nConfig\ntarget_decoy_ratio: 0.95\n")
        return SimpleNamespace(dir=experiment_dir, library=library, library_inputs=library_inputs,
                               combined=combined)

    def test_the_experiments_combined_ids_are_used(self, finished, monkeypatch):
        import polars as pl
        ids = pl.DataFrame({"untag_prec": ["AAA_2"], "run_idx": [1]})
        ids.write_parquet(finished.dir / "combined_filtered_IDs.parquet")
        monkeypatch.setattr(config.args, "make_library", str(finished.dir))
        run_jmod.main()
        assert finished.combined == []  # nothing recombined
        assert [i.equals(ids) for i, _ in finished.library_inputs] == [True]
        assert (finished.dir / "mbr_library").is_dir()

    def test_an_older_experiment_is_combined_first_from_its_runs(self, finished):
        # No JMod_config.json and no experiment_results: the settings are a run's,
        # the runs go by name, and the ratio is the library's from params.txt
        run_jmod.run_make_library(str(finished.dir))
        (run_folders, experiment_dir, ratio, _), = finished.combined
        assert run_folders == {1: str(finished.dir / "a_results"), 2: str(finished.dir / "b_results")}
        assert (experiment_dir, ratio) == (str(finished.dir), 0.95)
        assert finished.library_inputs == [("combined IDs", str(finished.library).replace("\\", "/"))]

    def test_runs_are_numbered_in_the_order_of_the_data_files(self, finished):
        (finished.dir / "JMod_config.json").write_text(json.dumps({"speclib": str(finished.library),
                                                                  "mzml": ["b.mzML", "a.mzML"]}))
        run_jmod.run_make_library(str(finished.dir))
        (run_folders, *_), = finished.combined
        assert run_folders == {1: str(finished.dir / "b_results"), 2: str(finished.dir / "a_results")}

    def test_a_moved_library_is_given_on_the_command_line(self, finished, monkeypatch, tmp_path):
        moved = tmp_path / "moved.tsv"
        moved.write_text("")
        monkeypatch.setattr(config, "cli_args", {"speclib": str(moved)})
        run_jmod.run_make_library(str(finished.dir))
        assert [speclib for _, speclib in finished.library_inputs] == [str(moved).replace("\\", "/")]

    def test_a_missing_library_stops_before_anything_is_combined(self, finished):
        finished.library.unlink()
        with pytest.raises(JModError, match="Spectral library not found"):
            run_jmod.run_make_library(str(finished.dir))
        assert finished.combined == [] and not (finished.dir / "mbr_library").exists()


class TestCreateResultsFolder:
    def test_unexplained_folder_error_is_raised(self, monkeypatch, tmp_path):
        # A FileNotFoundError that is neither a missing parent nor a long path
        def mkdir(*args):
            raise FileNotFoundError("something else")

        monkeypatch.setattr(run_jmod.os, "mkdir", mkdir)
        with pytest.raises(JModError, match="Error Creating Results Folder"):
            run_jmod._create_results_folder("data/a.mzML", str(tmp_path))

    def test_named_after_the_data_file_only(self, monkeypatch, tmp_path):
        # --dummy_value names the experiment folder, not the runs'
        monkeypatch.setattr(config.args, "dummy_value", "batch2")
        folder = run_jmod._create_results_folder("data/a.mzML", str(tmp_path))
        assert os.path.basename(folder) == "a_results"
