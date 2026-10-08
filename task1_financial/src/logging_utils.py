"""Logging setup shared by all Task 1 modules."""
# AI-ASSISTED: Cursor Agent (claude-sonnet-5.5), Prompt: 'yes implement the plan @TASK1A_PLAN.md', Date: 2026-10-08 (see CITATIONS.md Entry 4)

import logging
import sys

from . import config

# Parent logger for the package; every module logs to a child ("task1.data", "task1.news", ...).
PACKAGE_LOGGER_NAME = "task1"


def configure_logging(level: str = config.LOG_LEVEL) -> logging.Logger:
    """Configure the package logger once; safe to call repeatedly (e.g. notebook re-runs)."""
    logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    logger.setLevel(level)
    # Only add a handler the first time; re-running the notebook cell would otherwise print every line twice.
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(config.LOG_FORMAT, datefmt=config.LOG_DATE_FORMAT))
        logger.addHandler(handler)
    # Do not pass records up to the root logger (Jupyter has its own handler there -> duplicates).
    logger.propagate = False

    # yfinance prints its own HTTP errors; keep them out of the way unless they matter.
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    return logger


def get_logger(module_name: str) -> logging.Logger:
    """Return a child logger, e.g. get_logger('data') -> 'task1.data'."""
    return logging.getLogger(f"{PACKAGE_LOGGER_NAME}.{module_name}")
