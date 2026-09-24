import logging
import sys

import pytest

from src.utils.errors import JModError, report_error, mark_run_failed


def _errors(records):
    return [r for r in records if r.levelno >= logging.ERROR]


class TestReportError:
    def test_user_error_is_logged_as_its_message(self, app_log):
        report_error(JModError("bad file"), "Run 1 of 2 failed (a.mzML)")
        (record,) = _errors(app_log)
        assert record.getMessage() == "Run 1 of 2 failed (a.mzML): bad file"
        assert record.exc_info is None  # no traceback

    def test_unexpected_error_is_logged_with_traceback(self, app_log):
        try:
            raise ValueError("boom")
        except ValueError as e:
            report_error(e, "JMod stopped")
        (record,) = _errors(app_log)
        assert record.getMessage() == "JMod stopped: unexpected error"
        assert record.exc_info is not None


class TestMarkRunFailed:
    def test_folder_is_renamed(self, tmp_path):
        (tmp_path / "a_results").mkdir()
        mark_run_failed(str(tmp_path / "a_results"))
        assert [p.name for p in tmp_path.iterdir()] == ["run_failed_a_results"]

    def test_earlier_failed_folder_is_kept(self, tmp_path):
        (tmp_path / "run_failed_a_results").mkdir()
        (tmp_path / "run_failed_a_results" / "old.txt").write_text("earlier run")
        (tmp_path / "a_results").mkdir()
        mark_run_failed(str(tmp_path / "a_results"))
        assert (tmp_path / "run_failed_a_results" / "old.txt").read_text() == "earlier run"
        (new_folder,) = [p for p in tmp_path.iterdir() if p.name != "run_failed_a_results"]
        assert new_folder.name.startswith("run_failed_a_results_")  # datestamped
        assert not (tmp_path / "a_results").exists()

    @pytest.mark.skipif(sys.platform != "win32", reason="only Windows refuses the rename")
    def test_refused_rename_leaves_the_folder_whole(self, tmp_path):
        (tmp_path / "a_results").mkdir()
        (tmp_path / "a_results" / "result.txt").write_text("result")
        # Windows will not rename a folder with an open file in it
        with open(tmp_path / "a_results" / "open.txt", "w"):
            mark_run_failed(str(tmp_path / "a_results"))
        assert [p.name for p in tmp_path.iterdir()] == ["a_results"]  # no copy made
        assert (tmp_path / "a_results" / "result.txt").read_text() == "result"
