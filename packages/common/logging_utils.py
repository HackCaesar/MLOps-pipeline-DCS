"""Logging setup: console + per-run file in storage/runs/{run_id}/logs/."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATEFMT = "%Y-%m-%dT%H:%M:%S"


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    logger_name: Optional[str] = None,
) -> logging.Logger:
    """Configure root logger with console handler and optional file handler.

    Calling this twice is safe — existing handlers are not duplicated.
    """
    root = logging.getLogger() if logger_name is None else logging.getLogger(logger_name)
    target_level = getattr(logging, level.upper(), logging.INFO)
    root.setLevel(target_level)

    has_stream = any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in root.handlers)
    if not has_stream:
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
        root.addHandler(sh)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        already = any(
            isinstance(h, logging.FileHandler) and Path(h.baseFilename) == log_file
            for h in root.handlers
        )
        if not already:
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
            root.addHandler(fh)

    return root


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
