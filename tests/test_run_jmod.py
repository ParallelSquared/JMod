import json
import logging
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
    monkeypatch.setattr(config.args, "config_json", None)
    monkeypatch.setattr(config.args, "plexDIA", config.args.plexDIA)  # main may set it
    monkeypatch.setattr(run_jmod, "set_log_filepath", lambda path: None)
    monkeypatch.setattr(run_jmod, "resolve_tags", lambda: (None, None))
    monkeypatch.setattr(run_jmod, "build_library",
                        lambda *args: SimpleNamespace(n_targets=10, n_decoys=10))
    return run_jmod


@pytest.fixture
def combined(monkeypatch):
    """The calls main() makes to combine_runs, recorded instead of run: the
    stubbed runs write no outputs to combine."""
    calls = []
    monkeypatch.setattr(run_jmod, "combine_runs", lambda *args: calls.append(args))
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
            run_jmod.process_run(runState, None, None, None, None)
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


class TestExperimentConfig:
    def test_config_lists_every_data_file(self, experiment, monkeypatch, tmp_path):
        monkeypatch.setattr(experiment, "process_run", lambda runState, *rest: _finish(runState))
        experiment.main()
        written = json.loads((tmp_path / "JMod_config.json").read_text())
        assert written["mzml"] == ["a.mzML", "b.mzML"]

    def test_earlier_config_is_kept(self, experiment, monkeypatch, tmp_path):
        (tmp_path / "JMod_config.json").write_text("earlier experiment")
        monkeypatch.setattr(experiment, "process_run", lambda runState, *rest: _finish(runState))
        experiment.main()
        assert (tmp_path / "JMod_config.json").read_text() == "earlier experiment"
        assert len(list(tmp_path.glob("JMod_config_*.json"))) == 1  # datestamped


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


class TestCreateResultsFolder:
    def test_unexplained_folder_error_is_raised(self, monkeypatch, tmp_path):
        # A FileNotFoundError that is neither a missing parent nor a long path
        def mkdir(*args):
            raise FileNotFoundError("something else")

        monkeypatch.setattr(config.args, "output_folder", str(tmp_path))
        monkeypatch.setattr(config.args, "dummy_value", None)
        monkeypatch.setattr(run_jmod.os, "mkdir", mkdir)
        with pytest.raises(JModError, match="Error Creating Results Folder"):
            run_jmod._create_results_folder("data/a.mzML")
