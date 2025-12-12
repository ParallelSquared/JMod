import tkinter as tk
from tkinter import messagebox
import logging
import sys
import importlib
import os
import shutil

def send_raise_to_TK(error_text):
    if "src.config" in sys.modules:   #this weird import somehow solved an error
        config = sys.modules["src.config"]
    else:
        config = importlib.import_module("src.config")

    config.error_already_handled = True

    logging.getLogger("appLogger").error(error_text)
    logging.getLogger("appLogger").error("runERROR: Run Exited\n")

    rename_results_folder(config)

    if config.ran_from_GUI is True:
        config.GUI_result_queue.put(f"errorGUI_{error_text}")

def rename_results_folder(config):
    # Rename results folder to indicate failure
    if hasattr(config, 'results_folder_path') and os.path.exists(config.results_folder_path):
        try:
            # Close file handlers to release the log file
            logger = logging.getLogger("appLogger")
            for handler in logger.handlers[:]:
                if isinstance(handler, logging.FileHandler):
                    handler.close()
                    logger.removeHandler(handler)
            
            parent_dir = os.path.dirname(config.results_folder_path)
            folder_name = os.path.basename(config.results_folder_path)
            failed_folder_name = "run_failed_" + folder_name
            failed_folder_path = os.path.join(parent_dir, failed_folder_name)
            
            # Handle if the failed folder already exists
            if os.path.exists(failed_folder_path):
                shutil.rmtree(failed_folder_path)
            
            shutil.move(config.results_folder_path, failed_folder_path)
        except Exception as rename_error:
            logging.getLogger("appLogger").warning(f"Could not rename results folder: {rename_error}")




