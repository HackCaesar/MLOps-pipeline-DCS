"""Atomic status writer used by both HTTP + folder watcher paths.

A capture's status lives in two places at once:
  - the in-memory ``CaptureRunner._status`` dict (served by HTTP endpoints);
  - ``shared/status/{run_id}.status.json`` on disk (read by controllers without HTTP).

This module keeps both writes consistent and atomic on disk.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

VALID_STATUSES = ("queued", "running", "success", "failed", "stopped")


@dataclass
class CaptureStatus:
    """In-memory + on-disk status for one capture run."""

    run_id: str
    status: str = "queued"
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    raw_dataset_dir: Optional[str] = None
    num_images: int = 0
    num_annotations: int = 0
    error: Optional[str] = None
    phase: Optional[str] = None
    progress: Dict[str, Any] = field(default_factory=dict)
    agent_log_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id":          self.run_id,
            "status":          self.status,
            "started_at":      self.started_at,
            "finished_at":     self.finished_at,
            "raw_dataset_dir": self.raw_dataset_dir,
            "num_images":      self.num_images,
            "num_annotations": self.num_annotations,
            "error":           self.error,
            "phase":           self.phase,
            "progress":        self.progress,
            "agent_log_path":  self.agent_log_path,
        }


class StatusStore:
    """Thread-safe in-memory map run_id → CaptureStatus + atomic disk mirror."""

    def __init__(self, status_dir: Optional[Path]) -> None:
        self.status_dir = Path(status_dir) if status_dir else None
        if self.status_dir is not None:
            self.status_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._statuses: Dict[str, CaptureStatus] = {}

    def get(self, run_id: str) -> Optional[CaptureStatus]:
        with self._lock:
            s = self._statuses.get(run_id)
            return CaptureStatus(**s.to_dict()) if s is not None else None

    def upsert(self, status: CaptureStatus) -> None:
        with self._lock:
            self._statuses[status.run_id] = status
            self._write_disk(status)

    def update(self, run_id: str, **fields: Any) -> Optional[CaptureStatus]:
        with self._lock:
            cur = self._statuses.get(run_id)
            if cur is None:
                return None
            for k, v in fields.items():
                if v is not None:
                    setattr(cur, k, v)
            if "status" in fields and fields["status"] not in VALID_STATUSES:
                raise ValueError(f"Invalid status: {fields['status']!r}")
            if cur.status in ("success", "failed", "stopped") and cur.finished_at is None:
                cur.finished_at = utc_now_iso()
            self._write_disk(cur)
            return CaptureStatus(**cur.to_dict())

    def list_ids(self) -> list[str]:
        with self._lock:
            return list(self._statuses.keys())

    # ---- disk mirror --------------------------------------------------

    def _write_disk(self, status: CaptureStatus) -> None:
        if self.status_dir is None:
            return
        target = self.status_dir / f"{status.run_id}.status.json"
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(json.dumps(status.to_dict(), ensure_ascii=False, indent=2),
                       encoding="utf-8")
        os.replace(tmp, target)


def utc_now_iso() -> str:
    return (_dt.datetime.now(_dt.timezone.utc)
            .replace(microsecond=0).isoformat().replace("+00:00", "Z"))
