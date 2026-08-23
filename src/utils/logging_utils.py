"""
logging_utils.py
=================
LBSM — Logging configuration helpers.

Every notebook currently repeats
``logging.basicConfig(level=logging.WARNING)`` inline (see NB05 cell 2).
This module centralises that so notebooks and experiment scripts share one
configuration instead of each hand-rolling it.
"""

from __future__ import annotations

import logging
import sys
from typing import Optional

_CONFIGURED = False

DEFAULT_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
DEFAULT_DATEFMT = "%H:%M:%S"


def configure_root_logger(level: int = logging.WARNING) -> None:
    """Configure the root logger once, idempotently.

    Matches the ``logging.basicConfig(level=logging.WARNING)`` call at the
    top of every notebook, but safe to call more than once (subsequent calls
    only adjust the level) instead of silently no-op'ing like
    ``basicConfig`` does after the first call.
    """
    global _CONFIGURED
    root = logging.getLogger()
    if not _CONFIGURED:
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setFormatter(logging.Formatter(DEFAULT_FORMAT, DEFAULT_DATEFMT))
        root.addHandler(handler)
        _CONFIGURED = True
    root.setLevel(level)


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """Return a module-level logger, configuring the root logger first if needed.

    Parameters
    ----------
    name  : usually ``__name__`` of the calling module.
    level : per-logger level override; if None, inherits the root logger's level.
    """
    if not _CONFIGURED:
        configure_root_logger()
    logger = logging.getLogger(name)
    if level is not None:
        logger.setLevel(level)
    return logger
