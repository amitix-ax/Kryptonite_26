"""
Centralized logging setup. Nothing in this package configures logging
handlers on import (a library should never do that to its caller) --
call `setup_logging()` once from an application entry point (a script,
notebook, or service) to get sensible console output.

Modules obtain their logger the standard way:

    import logging
    logger = logging.getLogger(__name__)

so `setup_logging()` need only be called once, anywhere upstream.
"""

from __future__ import annotations

import logging


def setup_logging(level: int = logging.INFO, fmt: str | None = None) -> None:
    """Configure the root 'iceberg_model' logger with a console handler.
    Safe to call multiple times (idempotent -- won't stack duplicate handlers)."""
    logger = logging.getLogger("iceberg_model")
    logger.setLevel(level)

    if any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        return  # already configured

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        fmt or "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(handler)
    logger.propagate = False
