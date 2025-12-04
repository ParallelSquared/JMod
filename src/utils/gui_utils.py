import tkinter as tk
from tkinter import messagebox
import logging
import sys
import importlib

def send_raise_to_TK(error_text):
    if "src.config" in sys.modules:   #this weird import somehow solved an error
        config = sys.modules["src.config"]
    else:
        config = importlib.import_module("src.config")

    config.error_already_handled = True

    logging.getLogger("appLogger").error(error_text)
    logging.getLogger("appLogger").error("runERROR: Run Exited\n")

    if config.ran_from_GUI is True:
        config.GUI_result_queue.put(f"errorGUI_{error_text}")




