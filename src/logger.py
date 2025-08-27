import logging
import logging.config
import time
import os

class ElapsedFormatter(logging.Formatter):
    def __init__(self, fmt=None, datefmt=None, style="%"):
        super().__init__(fmt, datefmt, style)
        self.start_time = time.time()

    def formatTime(self, record, datefmt=None):
        elapsed = record.created - self.start_time
        hours, rem = divmod(int(elapsed), 3600)
        minutes, seconds = divmod(rem, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    def reset_start_time(self):
        self.start_time = time.time()
    
def set_log_filepath(logfile_path):
    with open (logfile_path, "w"):
        pass
    logger = logging.getLogger("appLogger")
    for h in logger.handlers[:]:
        if isinstance(h, logging.FileHandler):
            logger.removeHandler(h)
            h.close()
    file_handler = logging.FileHandler(logfile_path, mode="a")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        ElapsedFormatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(file_handler)


# load config
config_file = os.path.join(os.path.dirname(__file__), "logging.conf")
logging.config.fileConfig(config_file, disable_existing_loggers=False)

# patch handlers for elapsed time
for handler in logging.getLogger("appLogger").handlers:
    handler.setFormatter(
        ElapsedFormatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    
logger = logging.getLogger("appLogger")


