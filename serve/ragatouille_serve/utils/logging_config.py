import logging
import sys
from logging.handlers import RotatingFileHandler

# Define basic configuration
LOGGING_LEVEL = logging.INFO
LOG_FILE_PATH = "app.log"  # Path to the log file
MAX_LOG_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
BACKUP_COUNT = 5  # Number of backup log files to keep

# Define log format
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

def setup_logging():
    # Create a logger
    logger = logging.getLogger()
    logger.setLevel(LOGGING_LEVEL)

    # Create handlers
    console_handler = logging.StreamHandler(sys.stdout)
    file_handler = RotatingFileHandler(LOG_FILE_PATH, maxBytes=MAX_LOG_FILE_SIZE, backupCount=BACKUP_COUNT)

    # Create formatters and add it to handlers
    formatter = logging.Formatter(LOG_FORMAT)
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # Log a startup message
    logger.info("Logging is set up.")
