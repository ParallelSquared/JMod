#  Copyright (c) 2026 Parallel Squared Technology Institute
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#          http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import pytest
import builtins
import multiprocessing
from unittest.mock import patch, MagicMock
import sys
import types


# Check if tkinter is functional
_tkinter_available = False
try:
    import tkinter as tk
    root = tk.Tk()
    root.destroy()
    _tkinter_available = True
except Exception:
    pass

if _tkinter_available:
    from src.run_jmod_from_GUI import make_GUI, JModGUI

from src.run_jmod_from_GUI import run_main_process


# @pytest.mark.skipif(not _tkinter_available, reason="Tkinter not available")
# class Test_make_GUI():

#     def test_make_GUI(self):
#         gui = make_GUI(show=False)
#         assert isinstance(gui, JModGUI)

#     def test_all_gui_params_have_tk_handle(self):
#         """Every default_dict entry with in_GUI=True must have a non-None
#         tk_handle after GUI initialization. Catches cases where a new
#         parameter is added to default_dict but no widget is created."""
#         from src.default_dict import default_dict
#         gui = make_GUI(show=False)
#         missing = []
#         for key, entry in default_dict.items():
#             if entry.get('in_GUI') and entry.get('tk_handle') is None:
#                 missing.append(key)
#         assert missing == [], f"in_GUI=True but tk_handle is None: {missing}"


class Test_run_main_process():
    @pytest.fixture
    def log_queue(self):
        return multiprocessing.Queue()


    def test_run_main_process_success(self, tmp_path, log_queue):
        tmp_file = tmp_path / "tmp_0.txt"
        tmp_file.write_text("test")

        fake_run_jmod = types.ModuleType("src.run_jmod")
        fake_run_jmod.main = MagicMock(return_value=None)

        with patch("src.run_jmod_from_GUI.os.remove") as mock_remove, \
            patch("src.run_jmod_from_GUI.sys.exit") as mock_exit, \
            patch("src.run_jmod_from_GUI.logging.getLogger") as mock_get_logger, \
            patch("src.run_jmod_from_GUI.QueueHandler") as mock_QH, \
            patch("src.config.ran_from_GUI", False), \
            patch("src.run_jmod.main") as mock_main:


            # simulate run_jmod.main returning success
            mock_main.return_value = None

            # logger mock
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger

            run_main_process(str(tmp_file), log_queue, 1, 2)

            # One process runs one experiment
            mock_main.assert_called_once_with(str(tmp_file))

            # sys.exit should NOT be called
            mock_exit.assert_not_called()


class TestExperimentQueue:
    """The GUI runs each experiment in its own process, one after another."""

    @pytest.fixture
    def gui(self):
        # check_process called on a stand-in for the GUI object: no window is needed
        import src.run_jmod_from_GUI as gui_module
        stand_in = types.SimpleNamespace(after=MagicMock(), start_next_experiment=MagicMock(),
                                         enable_buttons=MagicMock(), drain_log_queue=MagicMock(),
                                         _pending_experiments=[], _n_experiments=2,
                                         _stop_requested=False)
        return lambda p, idx: gui_module.JModGUI.check_process(stand_in, p, idx), stand_in

    @staticmethod
    def _ended_process(exitcode):
        return MagicMock(is_alive=MagicMock(return_value=False), exitcode=exitcode)

    def test_an_ended_experiment_starts_the_next(self, gui):
        check_process, stand_in = gui
        stand_in._pending_experiments = [(2, "second.json")]
        check_process(self._ended_process(0), 1)
        stand_in.start_next_experiment.assert_called_once()
        stand_in.enable_buttons.assert_not_called()

    def test_the_last_experiment_ending_enables_the_buttons(self, gui):
        check_process, stand_in = gui
        check_process(self._ended_process(0), 2)
        stand_in.start_next_experiment.assert_not_called()
        stand_in.enable_buttons.assert_called_once()

    def test_a_process_that_died_is_reported_and_the_queue_goes_on(self, gui):
        check_process, stand_in = gui
        stand_in._pending_experiments = [(2, "second.json")]
        with patch("src.run_jmod_from_GUI.logging.getLogger") as mock_get_logger:
            check_process(self._ended_process(3221225477), 1)
        assert "Experiment 1 of 2 stopped unexpectedly" in mock_get_logger.return_value.error.call_args[0][0]
        stand_in.start_next_experiment.assert_called_once()

    def test_a_stopped_experiment_is_not_reported(self, gui):
        check_process, stand_in = gui
        stand_in._stop_requested = True
        with patch("src.run_jmod_from_GUI.logging.getLogger") as mock_get_logger:
            check_process(self._ended_process(-15), 1)
        mock_get_logger.return_value.error.assert_not_called()
        stand_in.enable_buttons.assert_called_once()


class TestOfferJsonDataFiles:
    """Loading a configuration JSON that lists data files offers to add them."""

    @pytest.fixture
    def gui(self):
        # Called on a stand-in for the GUI object: no window is needed
        import src.run_jmod_from_GUI as gui_module
        stand_in = types.SimpleNamespace(_DATA_FILE_KINDS=gui_module.JModGUI._DATA_FILE_KINDS,
                                         _add_mzml_or_raw=MagicMock(), _add_d_folder=MagicMock())
        return lambda mzml: gui_module.JModGUI._offer_json_data_files(stand_in, mzml), stand_in

    def test_presets_are_not_asked_about(self, gui):
        offer, _ = gui
        with patch("tkinter.messagebox.askyesno") as ask:
            offer(None)
        ask.assert_not_called()

    def test_question_lists_only_the_kinds_present(self, gui):
        offer, stand_in = gui
        with patch("tkinter.messagebox.askyesno", return_value=False) as ask:
            offer(["a.mzML", "b.mzML", "c.d"])
        message = ask.call_args.args[1]
        assert "2 .mzml Files" in message and "1 .d Folders" in message and ".raw" not in message
        stand_in._add_mzml_or_raw.assert_not_called()  # answered No

    def test_yes_adds_the_files_and_warns_about_missing_ones(self, gui, tmp_path):
        offer, stand_in = gui
        for name in ("a.mzML", "r.raw"):
            (tmp_path / name).write_text("")
        (tmp_path / "x.d").mkdir()
        paths = [str(tmp_path / n) for n in ("a.mzML", "r.raw", "x.d", "gone.mzML")]
        with patch("tkinter.messagebox.askyesno", return_value=True),                 patch("tkinter.messagebox.showwarning") as warn:
            offer(paths)
        assert [c.args[0] for c in stand_in._add_mzml_or_raw.call_args_list] == paths[:2]
        stand_in._add_d_folder.assert_called_once_with(paths[2])
        assert "gone.mzML" in warn.call_args.args[1]


class TestCommandLineArgs:
    """Command-line-only options from a loaded JSON, as additional-commands text."""

    def test_a_list_gives_the_flag_once_per_value(self):
        from src.run_jmod_from_GUI import command_line_args
        assert command_line_args("strip_mod", ["UniMod:4,C", "DimethylNter,28.0313,n"]) == [
            "--strip_mod UniMod:4,C", "--strip_mod DimethylNter,28.0313,n"]

    def test_the_text_parses_back_to_the_list(self):
        import shlex
        from src.config import parser
        from src.run_jmod_from_GUI import command_line_args
        text = " ".join(command_line_args("add_fixed_mod", ["Label,8.0120,n", "UniMod:4,C"]))
        assert parser.parse_args(shlex.split(text)).add_fixed_mod == ["Label,8.0120,n", "UniMod:4,C"]
