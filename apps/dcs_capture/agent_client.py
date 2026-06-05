"""HTTP + shared-folder client for the Windows DCS Runner Agent.

The client used by ``apps/dcs_capture/cli.py`` (and later the pipeline
controller in Phase 14). HTTP-first; if ``GET /health`` fails, it switches to
the shared-folder protocol where commands are written as JSON files and the
agent polls them.

All HTTP calls use stdlib ``urllib.request`` to avoid pulling in ``requests``.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from packages.common.logging_utils import get_logger

LOG = get_logger(__name__)


@dataclass(frozen=True)
class CaptureRequest:
    """Same payload as ``CaptureSpec.from_dict`` accepts."""

    run_id: str
    dataset_id: str
    mission_path: str
    dcs_config_path: str
    dcs_source_dataset_dir: str
    target_dataset_dir: str
    num_frames: int = 1
    timeout_sec: int = 7200

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id":                  self.run_id,
            "dataset_id":              self.dataset_id,
            "mission_path":            self.mission_path,
            "dcs_config_path":         self.dcs_config_path,
            "dcs_source_dataset_dir":  self.dcs_source_dataset_dir,
            "target_dataset_dir":      self.target_dataset_dir,
            "num_frames":              self.num_frames,
            "timeout_sec":             self.timeout_sec,
        }


class AgentTimeoutError(TimeoutError):
    """Raised when ``wait_for_status`` reaches the configured timeout."""


class AgentClient:
    """HTTP-first client with shared-folder fallback.

    Construct with both the URL and the shared dir; ``health()`` decides the
    transport. Methods that follow ``request_capture()`` use the same transport
    that ``request_capture()`` used.
    """

    def __init__(
        self,
        agent_url: Optional[str] = None,
        *,
        commands_dir: Optional[Path] = None,
        status_dir: Optional[Path] = None,
        http_timeout_sec: float = 3.0,
        prefer_shared_folder: bool = False,
    ) -> None:
        self.agent_url = (agent_url or "").rstrip("/") or None
        self.commands_dir = Path(commands_dir) if commands_dir else None
        self.status_dir = Path(status_dir) if status_dir else None
        self.http_timeout_sec = float(http_timeout_sec)
        self._transport: Optional[str] = (
            "shared" if prefer_shared_folder else None
        )

    # ---- transport selection ----

    @property
    def transport(self) -> str:
        """Returns "http" or "shared" — call ``health()`` to lock the choice."""
        if self._transport is None:
            self.health()
        assert self._transport is not None
        return self._transport

    def health(self) -> bool:
        """Probe HTTP; on failure, fall back to shared folder if configured."""
        if self.agent_url:
            try:
                payload = self._http_get("/health")
                if payload.get("status") == "ok":
                    self._transport = "http"
                    LOG.info("AgentClient: HTTP transport ready (%s)", self.agent_url)
                    return True
            except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
                LOG.warning("AgentClient: HTTP /health failed (%s); falling back", exc)
        if self.commands_dir is not None and self.status_dir is not None:
            self._transport = "shared"
            LOG.info("AgentClient: shared-folder transport (%s, %s)",
                     self.commands_dir, self.status_dir)
            return True
        LOG.error("AgentClient: no transport available (HTTP failed + no shared dirs)")
        return False

    # ---- public methods ----

    def request_capture(self, req: CaptureRequest) -> Dict[str, Any]:
        """Submit a capture. Returns ``{"accepted": bool, "run_id": ..., "transport": ...}``."""
        transport = self.transport
        if transport == "http":
            resp = self._http_post("/capture/start", req.to_dict())
            resp["transport"] = "http"
            return resp
        return self._shared_submit(req)

    def get_status(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Read current status. Returns None if not found yet."""
        transport = self.transport
        if transport == "http":
            try:
                return self._http_get(f"/capture/status/{run_id}")
            except urllib.error.HTTPError as exc:
                if exc.code == 404:
                    return None
                raise
        # shared
        if self.status_dir is None:
            return None
        path = self.status_dir / f"{run_id}.status.json"
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def stop_capture(self, run_id: str) -> bool:
        transport = self.transport
        if transport == "http":
            try:
                resp = self._http_post(f"/capture/stop/{run_id}", payload={})
                return bool(resp.get("stopped"))
            except urllib.error.HTTPError:
                return False
        # Shared-folder protocol: emit a sentinel ``stop.command.json``.
        if self.commands_dir is None:
            return False
        sentinel = self.commands_dir / f"{run_id}.stop.json"
        sentinel.write_text(json.dumps({"run_id": run_id, "action": "stop"}),
                             encoding="utf-8")
        return True

    def wait_for_status(
        self,
        run_id: str,
        *,
        final_statuses=("success", "failed", "stopped"),
        timeout_sec: float = 7200.0,
        poll_interval_sec: float = 5.0,
    ) -> Dict[str, Any]:
        """Block until status reaches a terminal value or the timeout fires."""
        deadline = time.monotonic() + float(timeout_sec)
        last_status: Optional[Dict[str, Any]] = None
        while time.monotonic() < deadline:
            cur = self.get_status(run_id)
            if cur is not None:
                last_status = cur
                if cur.get("status") in final_statuses:
                    return cur
            time.sleep(poll_interval_sec)
        raise AgentTimeoutError(
            f"wait_for_status timed out after {timeout_sec}s for run_id={run_id} "
            f"(last status: {last_status or 'none'})"
        )

    # ---- HTTP helpers ----

    def _http_get(self, path: str) -> Dict[str, Any]:
        assert self.agent_url
        url = self.agent_url + path
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=self.http_timeout_sec) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _http_post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        assert self.agent_url
        url = self.agent_url + path
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=body, method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.http_timeout_sec) as resp:
            return json.loads(resp.read().decode("utf-8"))

    # ---- shared-folder helpers ----

    def _shared_submit(self, req: CaptureRequest) -> Dict[str, Any]:
        if self.commands_dir is None:
            raise RuntimeError("commands_dir not configured; cannot submit via shared folder")
        self.commands_dir.mkdir(parents=True, exist_ok=True)
        target = self.commands_dir / f"{req.run_id}.command.json"
        target.write_text(json.dumps(req.to_dict(), indent=2), encoding="utf-8")
        LOG.info("AgentClient: wrote shared command %s", target)
        return {"accepted": True, "run_id": req.run_id, "transport": "shared"}
