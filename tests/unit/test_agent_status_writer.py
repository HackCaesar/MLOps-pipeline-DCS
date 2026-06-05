"""Unit tests for apps.windows_dcs_runner_agent.status_writer."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from apps.windows_dcs_runner_agent.status_writer import (
    VALID_STATUSES,
    CaptureStatus,
    StatusStore,
    utc_now_iso,
)


def test_status_store_in_memory_only(tmp_path: Path) -> None:
    """Without status_dir, store works but nothing hits disk."""
    s = StatusStore(status_dir=None)
    s.upsert(CaptureStatus(run_id="r1", status="queued"))
    got = s.get("r1")
    assert got is not None and got.status == "queued"
    assert "r1" in s.list_ids()


def test_status_store_writes_disk_mirror(tmp_path: Path) -> None:
    s = StatusStore(status_dir=tmp_path)
    s.upsert(CaptureStatus(run_id="r1", status="queued"))
    f = tmp_path / "r1.status.json"
    assert f.is_file()
    data = json.loads(f.read_text(encoding="utf-8"))
    assert data["run_id"] == "r1"
    assert data["status"] == "queued"


def test_status_store_update_merges_fields(tmp_path: Path) -> None:
    s = StatusStore(status_dir=tmp_path)
    s.upsert(CaptureStatus(run_id="r1", status="queued"))
    s.update("r1", status="running", phase="launching_dcs",
              started_at="2026-05-25T00:00:00Z")
    cur = s.get("r1")
    assert cur.status == "running"
    assert cur.phase == "launching_dcs"
    assert cur.started_at == "2026-05-25T00:00:00Z"
    # finished_at not set yet
    assert cur.finished_at is None


def test_status_store_auto_sets_finished_at_on_terminal(tmp_path: Path) -> None:
    s = StatusStore(status_dir=tmp_path)
    s.upsert(CaptureStatus(run_id="r1", status="running"))
    s.update("r1", status="success", phase="done")
    cur = s.get("r1")
    assert cur.status == "success"
    assert cur.finished_at is not None  # auto-set


def test_status_store_update_returns_none_for_unknown(tmp_path: Path) -> None:
    s = StatusStore(status_dir=tmp_path)
    assert s.update("nope", status="success") is None


def test_status_store_rejects_invalid_status(tmp_path: Path) -> None:
    s = StatusStore(status_dir=tmp_path)
    s.upsert(CaptureStatus(run_id="r1"))
    with pytest.raises(ValueError, match="Invalid status"):
        s.update("r1", status="bogus")


def test_status_store_get_returns_copy(tmp_path: Path) -> None:
    """Mutating the returned status must not leak back into the store."""
    s = StatusStore(status_dir=tmp_path)
    s.upsert(CaptureStatus(run_id="r1", num_images=5))
    got = s.get("r1")
    got.num_images = 999
    fresh = s.get("r1")
    assert fresh.num_images == 5


def test_atomic_write_no_partial_files(tmp_path: Path) -> None:
    """After upsert, no .tmp file lingers."""
    s = StatusStore(status_dir=tmp_path)
    s.upsert(CaptureStatus(run_id="r1", status="running"))
    leftovers = list(tmp_path.glob("*.tmp"))
    assert leftovers == []


def test_valid_statuses_constant_complete() -> None:
    assert set(VALID_STATUSES) >= {"queued", "running", "success", "failed", "stopped"}


def test_utc_now_iso_format() -> None:
    s = utc_now_iso()
    assert s.endswith("Z")
    assert "T" in s
