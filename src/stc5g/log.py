"""Logging helper shared by all modules."""

import logging
import sys


def get_logger(name: str) -> logging.Logger:
    """Return a logger that writes ``[LEVEL] message`` lines to stdout."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logging.addLevelName(logging.WARNING, "WARN")
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(handler)
    return logger
