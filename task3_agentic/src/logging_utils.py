"""Logging setup for Task 3 (same pattern as task1_financial/src/logging_utils.py)."""
# AI-ASSISTED: Claude Code (claude-opus-5-5), Prompt: 'Implement the Task 3A plan (the plan approved in Entry 13)', Date: 2026-10-09 (see CITATIONS.md Entry 14)

import logging
import sys

from . import config

PACKAGE_LOGGER_NAME = "task3"

# Libraries that log every HTTP request or retry at INFO. Their own errors still reach the tool envelope.
QUIET_LOGGERS = ("httpx", "openai", "groq", "yfinance", "ddgs", "primp")


def configure_logging(level: str = config.LOG_LEVEL) -> logging.Logger:
    """Configure the task3 logger once; safe to call again on notebook re-runs."""
    logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(config.LOG_FORMAT, datefmt=config.LOG_DATE_FORMAT))
        logger.addHandler(handler)
    logger.propagate = False
    for name in QUIET_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
    return logger


def get_logger(module_name: str) -> logging.Logger:
    return logging.getLogger(f"{PACKAGE_LOGGER_NAME}.{module_name}")
