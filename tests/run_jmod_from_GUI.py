import pytest
from src.run_jmod_from_GUI import make_GUI, JModGUI

class Test_make_GUI():

    def test_make_GUI(self):
        gui = make_GUI(show=False)
        assert isinstance(gui, JModGUI)