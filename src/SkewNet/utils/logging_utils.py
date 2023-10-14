import datetime
import logging
import os
from logging.handlers import RotatingFileHandler


def setup_logging(log_prefix, log_level=logging.INFO, log_dir="logs"):
    """
    Configure the logging for the application.

    Parameters
    ----------
    log_prefix : str
        A string prefix for the log filename.
    log_level : int, optional
        The log level, default is INFO.
    log_dir : str, optional
        The directory where logs will be stored, default is 'logs'.
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_filename = f"{log_dir}/{log_prefix}_{timestamp}.log"

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            RotatingFileHandler(log_filename, maxBytes=5000000, backupCount=5),
            logging.StreamHandler()
        ]
    )
