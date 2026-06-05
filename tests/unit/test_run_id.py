"""Unit tests for packages.common.run_id."""
from __future__ import annotations

import datetime as _dt
from pathlib import Path

import pytest

from packages.common.run_id import (
    generate_run_id,
    is_valid_run_id,
    parse_run_id,
)


def _fixed_now(year=2026, month=5, day=24):
    return lambda: _dt.datetime(year, month, day, 12, 0, 0)


def test_format_when_runs_dir_missing(tmp_path: Path) -> None:
    rid = generate_run_id(runs_dir=tmp_path / "does_not_exist", now=_fixed_now())
    assert rid == "2026_05_24_001"


def test_starts_at_001_when_empty(tmp_path: Path) -> None:
    assert generate_run_id(runs_dir=tmp_path, now=_fixed_now()) == "2026_05_24_001"


def test_increments_past_existing_today(tmp_path: Path) -> None:
    (tmp_path / "2026_05_24_001").mkdir()
    (tmp_path / "2026_05_24_002").mkdir()
    assert generate_run_id(runs_dir=tmp_path, now=_fixed_now()) == "2026_05_24_003"


def test_fills_gap_in_existing_today(tmp_path: Path) -> None:
    (tmp_path / "2026_05_24_001").mkdir()
    (tmp_path / "2026_05_24_003").mkdir()
    assert generate_run_id(runs_dir=tmp_path, now=_fixed_now()) == "2026_05_24_002"


def test_ignores_other_days(tmp_path: Path) -> None:
    (tmp_path / "2025_12_31_005").mkdir()
    (tmp_path / "random_thing").mkdir()
    assert generate_run_id(runs_dir=tmp_path, now=_fixed_now()) == "2026_05_24_001"


def test_validation_and_parsing() -> None:
    assert is_valid_run_id("2026_05_24_007")
    assert not is_valid_run_id("2026-05-24-007")
    assert not is_valid_run_id("foo")
    assert not is_valid_run_id("2026_05_24_7")

    date, n = parse_run_id("2026_05_24_007")
    assert date == _dt.date(2026, 5, 24)
    assert n == 7
    with pytest.raises(ValueError):
        parse_run_id("bad")


def test_rejects_overflow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Pretend all 999 slots are taken today.
    for i in range(1, 1000):
        (tmp_path / f"2026_05_24_{i:03d}").mkdir()
    with pytest.raises(RuntimeError, match="Too many runs"):
        generate_run_id(runs_dir=tmp_path, now=_fixed_now())
