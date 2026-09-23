import pytest

from src.models.run_state import RunState


class TestRunState:
    def test_reading_an_unset_field_raises(self):
        runState = RunState()
        with pytest.raises(AttributeError, match="read before it was set"):
            runState.opt_rt_tol

    def test_a_set_field_reads_back(self):
        runState = RunState()
        runState.opt_rt_tol = 0.5
        assert runState.opt_rt_tol == 0.5

    def test_assigning_an_unknown_field_raises(self):
        runState = RunState()
        with pytest.raises(AttributeError):
            runState.opt_rt_toll = 0.5  # typo

    def test_as_dict_reports_unset_fields_as_none(self):
        runState = RunState()
        runState.opt_rt_tol = 0.5
        d = runState.as_dict()
        assert d["opt_rt_tol"] == 0.5
        assert d["opt_ms1_tol"] is None
        assert set(d) == set(RunState.__slots__)
