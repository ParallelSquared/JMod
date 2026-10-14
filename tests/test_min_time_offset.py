"""
Tests for the --reqminTimeOffset / --minTimeOffset timePlex channel separation.
"""

import numpy as np
import pandas as pd
import pytest

import src.config as config
from src.rt_alignment import (
    min_time_offset, drop_close_channel_hits, separate_channel_curves,
    drop_off_channel_hits, enforce_min_channel_offset_matrix, MinOffsetChannel,
)


def _line(slope, intercept):
    return lambda x: slope * np.asarray(x, dtype=float) + intercept


class TestMinTimeOffsetFlag:
    def test_off_by_default(self, monkeypatch):
        monkeypatch.setattr(config.args, "reqminTimeOffset", False, raising=False)
        monkeypatch.setattr(config.args, "minTimeOffset", 3.0, raising=False)
        assert min_time_offset() is None

    def test_on_returns_value(self, monkeypatch):
        monkeypatch.setattr(config.args, "reqminTimeOffset", True, raising=False)
        monkeypatch.setattr(config.args, "minTimeOffset", 0.5, raising=False)
        assert min_time_offset() == 0.5

    def test_rejects_non_positive(self, monkeypatch):
        monkeypatch.setattr(config.args, "reqminTimeOffset", True, raising=False)
        monkeypatch.setattr(config.args, "minTimeOffset", 0.0, raising=False)
        with pytest.raises(ValueError):
            min_time_offset()


class TestDropCloseChannelHits:
    def test_tosses_whole_peptide(self):
        rows = []
        for i in range(60):                       # well separated: kept
            rows += [(f"P{i}", 10.0 + i, 0), (f"P{i}", 18.0 + i, 1)]
        rows += [("CLOSE", 20.0, 0), ("CLOSE", 20.3, 1)]
        df = pd.DataFrame(rows, columns=["stripped_seq", "rt", "channel"])
        keep = drop_close_channel_hits(df, 0.5)
        assert not keep[df["stripped_seq"] == "CLOSE"].any()
        assert keep[df["stripped_seq"] != "CLOSE"].all()

    def test_raises_when_offset_exceeds_spacing(self):
        rows = []
        for i in range(60):
            rows += [(f"P{i}", 10.0 + i, 0), (f"P{i}", 13.0 + i, 1)]
        df = pd.DataFrame(rows, columns=["stripped_seq", "rt", "channel"])
        with pytest.raises(RuntimeError, match="channel spacing"):
            drop_close_channel_hits(df, 5.0)


class TestSeparateChannelCurves:
    def test_collapsed_region_is_filled_with_measured_gap(self):
        irt = np.linspace(0, 100, 1001)
        c0 = _line(0.5, 10)
        # channel 1 sits 8 min after channel 0 above iRT 50 but collapses onto it below
        c1 = lambda x: np.where(np.asarray(x) >= 50, c0(x) + 8.0, c0(x))
        sep = separate_channel_curves([c0, c1], irt, 3.0)
        gap = sep[1](irt) - sep[0](irt)
        assert gap.min() >= 3.0
        assert np.allclose(gap[irt >= 50], 8.0)
        assert np.allclose(gap[irt < 50], 8.0)     # filled from the separated region
        assert sep[0] is c0                         # channel 0 is the anchor

    def test_already_separated_is_untouched(self):
        irt = np.linspace(0, 100, 101)
        c0, c1 = _line(0.5, 10), _line(0.5, 18)
        sep = separate_channel_curves([c0, c1], irt, 3.0)
        assert sep[1] is c1

    def test_never_separated_uses_minimum(self):
        irt = np.linspace(0, 100, 101)
        c0 = _line(0.5, 10)
        sep = separate_channel_curves([c0, c0], irt, 2.0)
        assert np.allclose(sep[1](irt) - c0(irt), 2.0)

    def test_scalar_call_returns_float(self):
        m = MinOffsetChannel(_line(1, 0), _line(1, 0), 1.0, None, None)
        assert isinstance(m(3.0), float) and m(3.0) == pytest.approx(4.0)


class TestDropOffChannelHits:
    def test_mislabelled_peptide_dropped(self):
        curves = [_line(1, 0), _line(1, 8)]
        df = pd.DataFrame({
            "stripped_seq": ["A", "A", "B", "B"],
            "lib_rt": [20.0, 20.0, 30.0, 30.0],
            # A: correct; B: its channel-0 peak was labelled channel 1 (RT-rank error)
            "rt": [20.1, 28.0, 12.0, 30.1],
            "channel": [0, 1, 0, 1],
        })
        keep = drop_off_channel_hits(df, curves)
        assert keep.tolist() == [True, True, False, False]


class TestEnforceMatrix:
    def test_matrix_violations_filled(self):
        irt = np.linspace(0, 100, 201)
        m = np.column_stack([irt * 0.5, irt * 0.5 + 8.0])
        m[irt < 30, 1] = m[irt < 30, 0] + 0.1       # collapsed entries
        out, n_fixed = enforce_min_channel_offset_matrix(irt, m, 3.0)
        assert n_fixed[1] == int((irt < 30).sum())
        assert np.allclose(out[:, 1] - out[:, 0], 8.0)
        assert np.array_equal(out[:, 0], m[:, 0])
