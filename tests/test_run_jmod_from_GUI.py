import pytest
from src.run_jmod_from_GUI import make_GUI, JModGUI, run_main_process
import builtins
import multiprocessing
import pytest
from unittest.mock import patch, MagicMock

class Test_make_GUI():

    def test_make_GUI(self):
        gui = make_GUI(show=False)
        assert isinstance(gui, JModGUI)


class Test_run_main_process():
    @pytest.fixture
    def mock_queues(self):
        return multiprocessing.Queue(), multiprocessing.Queue()


    def test_run_main_process_success(self, tmp_path, mock_queues):
        tmp_files = []
        for i in range(2):
            p = tmp_path / f"tmp_{i}.txt"
            p.write_text("test")
            tmp_files.append(str(p))

        result_queue, log_queue = mock_queues

        with patch("src.run_jmod_from_GUI.os.remove") as mock_remove, \
            patch("src.run_jmod_from_GUI.sys.exit") as mock_exit, \
            patch("src.run_jmod_from_GUI.logging.getLogger") as mock_get_logger, \
            patch("src.run_jmod_from_GUI.QueueHandler") as mock_QH, \
            patch("src.run_jmod.main") as mock_main:

            # simulate run_jmod.main returning success
            mock_main.return_value = None

            # logger mock
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger

            run_main_process(tmp_files, result_queue, log_queue)

            # main should be called once per file
            assert mock_main.call_count == 2

            # sys.exit should NOT be called
            mock_exit.assert_not_called()
