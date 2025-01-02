import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import ClassVar

import colorama
from colorama import Fore, Style

# Initialize colorama for cross-platform colored output
colorama.init()

# Get log level from environment variable, default to INFO
LOG_LEVEL = os.getenv("MILVUS_DATASET_LOG_LEVEL", "INFO").upper()

# Define log directory
LOG_DIR = Path.home() / ".milvus-dataset" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for console output"""

    COLORS: ClassVar[dict[str, str]] = {
        "DEBUG": Fore.BLUE,
        "INFO": Fore.GREEN,
        "WARNING": Fore.YELLOW,
        "ERROR": Fore.RED,
        "CRITICAL": Fore.RED + Style.BRIGHT,
    }

    def format(self, record):
        # Save original values
        orig_msg = record.msg
        orig_levelname = record.levelname

        # Add colors if the record is going to the console
        if getattr(record, "is_console", False):
            color = self.COLORS.get(record.levelname, "")
            record.levelname = f"{color}{record.levelname:8}{Style.RESET_ALL}"
            record.msg = f"{color}{record.msg}{Style.RESET_ALL}"

        # Format the message
        result = super().format(record)

        # Restore original values
        record.msg = orig_msg
        record.levelname = orig_levelname

        return result


def configure_logger(level=None, **kwargs):
    """
    Configure the logger with custom settings.

    Args:
        level (str, optional): Log level. If None, uses environment variable or defaults to INFO.
        **kwargs: Additional configuration parameters for logger.
    """
    log_level = level or LOG_LEVEL

    # Create logger
    logger = logging.getLogger("milvus_dataset")
    logger.setLevel(log_level)

    # Remove existing handlers
    logger.handlers.clear()

    # Common format for all handlers
    base_format = "%(asctime)s.%(msecs)03d | %(levelname)-8s | %(process)d:%(threadName)s | %(name)s:%(funcName)s:%(lineno)d | %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    # Console handler
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(log_level)
    console_formatter = ColoredFormatter(base_format, date_format)
    console_handler.setFormatter(console_formatter)
    console_handler.addFilter(lambda record: setattr(record, "is_console", True) or True)
    logger.addHandler(console_handler)

    # File handler for all logs
    file_handler = logging.handlers.RotatingFileHandler(
        LOG_DIR / "milvus_dataset.log",
        maxBytes=100 * 1024 * 1024,  # 100 MB
        backupCount=30,
    )
    file_handler.setLevel(log_level)
    file_formatter = logging.Formatter(base_format, date_format)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # File handler for errors
    error_handler = logging.handlers.RotatingFileHandler(
        LOG_DIR / "milvus_dataset_error.log",
        maxBytes=100 * 1024 * 1024,  # 100 MB
        backupCount=30,
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(file_formatter)
    logger.addHandler(error_handler)

    return logger


# Configure the default logger
logger = configure_logger()
