"""Smoke tests for packages.common.seed and logging_utils."""
from __future__ import annotations

import logging
import random
from pathlib import Path

from packages.common.logging_utils import get_logger, setup_logging
from packages.common.seed import set_seed


def test_set_seed_is_reproducible() -> None:
    set_seed(42)
    a = [random.random() for _ in range(5)]
    set_seed(42)
    b = [random.random() for _ in range(5)]
    assert a == b


def test_set_seed_none_is_no_op() -> None:
    # Just verify no exception raised.
    set_seed(None)  # type: ignore[arg-type]


def test_setup_logging_creates_file(tmp_path: Path) -> None:
    log_file = tmp_path / "logs" / "test.log"
    logger = setup_logging(level="DEBUG", log_file=log_file)
    logger.info("hello from test")
    for h in list(logger.handlers):
        h.flush()
    assert log_file.exists()
    assert "hello from test" in log_file.read_text()

    # Cleanup so subsequent tests don't accumulate FileHandlers.
    for h in list(logger.handlers):
        if isinstance(h, logging.FileHandler):
            h.close()
            logger.removeHandler(h)


def test_setup_logging_is_idempotent(tmp_path: Path) -> None:
    log_file = tmp_path / "test.log"
    setup_logging(level="INFO", log_file=log_file)
    n_after_first = len(logging.getLogger().handlers)
    setup_logging(level="INFO", log_file=log_file)
    n_after_second = len(logging.getLogger().handlers)
    assert n_after_first == n_after_second

    # Cleanup
    root = logging.getLogger()
    for h in list(root.handlers):
        if isinstance(h, logging.FileHandler):
            h.close()
            root.removeHandler(h)


def test_get_logger_returns_named() -> None:
    log = get_logger("custom.thing")
    assert log.name == "custom.thing"
