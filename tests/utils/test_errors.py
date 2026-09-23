import logging

from src.utils.errors import JModError, report_error


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
