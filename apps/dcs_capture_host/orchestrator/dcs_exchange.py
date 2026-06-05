"""DCS file-protocol exchange layer.

The pipeline talks to DCS through a set of JSON files in
``Saved Games/DCS/Logs``. ``DCSExchange`` is a thin client that owns the
read/write/poll semantics for that protocol:

- ``cv_capture_request.json``       Python → Lua (capture request, two-stage
                                    ``pause_pending`` → ``pending`` flow)
- ``cv_camera_request.json``        Python → Lua (camera pose request)
- ``cv_camera_ack.json``            Lua → Python (camera applied)
- ``cv_pause_ack.json``             Lua → Python (DCS paused for capture)
- ``cv_screenshot_ack.json``        Lua → Python (screenshot written)
- ``cv_snapshots/snapshot_<t>.json`` Lua → Python (world + bbox snapshot)
- ``ScreenShots/<frame>_<t>.png``   Lua → Python (actual screenshot)

Token-based sync invariants (preserved byte-for-byte from the original
implementation):

- Every request carries a ``frame_token`` or ``request_token``.
- Acks are accepted only when the token matches; ``_read_json_with_token_when_ready``
  keeps polling until either timeout or match.
- Pause-first capture: Python writes ``status=pause_pending`` first, waits for
  ``cv_pause_ack.json`` with the same token, then promotes the same request to
  ``status=pending`` so Lua can take the snapshot inside the paused state.

This module does not start or stop DCS; it only talks to the file protocol
once DCS is running.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from orchestrator.json_io import atomic_write_json, load_json, utc_now_iso, wait_for_stable_file
from orchestrator.runtime_paths import RuntimePaths


class DCSExchange:
    def __init__(
        self,
        paths: RuntimePaths,
        runtime_cfg: Dict[str, Any],
        image_format: str,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.paths = paths
        self.runtime_cfg = runtime_cfg
        self.image_format = image_format
        self.logger = logger or logging.getLogger(__name__)

    # ---- low-level JSON polling ------------------------------------------

    def _poll_interval_s(self) -> float:
        return float(self.runtime_cfg.get("poll_interval_s", 0.2))

    def _default_timeout_s(self) -> float:
        return float(self.runtime_cfg.get("capture_timeout_s", 120.0))

    def read_json_when_ready(self, path: Path, timeout_s: float) -> Dict[str, Any]:
        deadline = time.time() + timeout_s
        poll_interval_s = self._poll_interval_s()
        while time.time() < deadline:
            if path.exists():
                try:
                    return load_json(path)
                except json.JSONDecodeError:
                    time.sleep(poll_interval_s)
                    continue
            time.sleep(poll_interval_s)
        raise TimeoutError(f"Timed out waiting for JSON file: {path}")

    def read_json_with_token_when_ready(
        self,
        path: Path,
        token_field: str,
        expected_token: str,
        timeout_s: float,
    ) -> Dict[str, Any]:
        deadline = time.time() + timeout_s
        poll_interval_s = self._poll_interval_s()
        last_seen_token = None
        while time.time() < deadline:
            if path.exists():
                try:
                    payload = load_json(path)
                except json.JSONDecodeError:
                    time.sleep(poll_interval_s)
                    continue
                last_seen_token = payload.get(token_field)
                if last_seen_token == expected_token:
                    return payload
            time.sleep(poll_interval_s)
        raise TimeoutError(
            f"Timed out waiting for {path} with {token_field}={expected_token}; "
            f"last seen token={last_seen_token}"
        )

    # ---- exchange-file cleanup -------------------------------------------

    def clear_capture_exchange(self, frame_id: str, frame_token: str) -> None:
        """Delete the four exchange files and the source-screenshot PNG for
        this frame, so a fresh capture starts from a known empty state."""
        for path in [
            self.paths.request_file,
            self.paths.screenshot_ack_file,
            self.paths.pause_ack_file,
            self.paths.snapshot_path(frame_token),
            self.paths.source_screenshot_path(frame_id, frame_token),
        ]:
            try:
                if path.exists():
                    path.unlink()
            except OSError:
                self.logger.warning("Could not delete stale capture exchange file: %s", path)

    # ---- capture request (pause-first) -----------------------------------

    def request_capture(
        self,
        frame_id: str,
        frame_token: str,
        capture_screenshot: bool,
    ) -> Tuple[Optional[Dict[str, Any]], float, str]:
        pause_first = bool(self.runtime_cfg.get("pause_first_capture", True))
        requested_epoch_s = time.time()
        payload = {
            "status": "pause_pending" if pause_first else "pending",
            "frame_id": frame_id,
            "frame_token": frame_token,
            "capture_screenshot": capture_screenshot,
            "requested_at_utc": utc_now_iso(),
        }
        atomic_write_json(self.paths.request_file, payload)
        if not pause_first:
            return None, requested_epoch_s, payload["requested_at_utc"]

        pause_ack = self.wait_for_pause_ack(frame_token)
        payload["status"] = "pending"
        payload["pause_acknowledged_at_utc"] = utc_now_iso()
        atomic_write_json(self.paths.request_file, payload)
        return pause_ack, requested_epoch_s, payload["requested_at_utc"]

    def wait_for_snapshot(self, frame_token: str, timeout_s: Optional[float] = None) -> Dict[str, Any]:
        timeout_s = float(timeout_s or self._default_timeout_s())
        snapshot = self.read_json_when_ready(self.paths.snapshot_path(frame_token), timeout_s)
        if snapshot.get("frame_token") != frame_token:
            raise RuntimeError(
                f"Snapshot token mismatch: expected {frame_token}, got {snapshot.get('frame_token')}"
            )
        return snapshot

    def wait_for_pause_ack(self, frame_token: str, timeout_s: Optional[float] = None) -> Dict[str, Any]:
        timeout_s = float(timeout_s or self._default_timeout_s())
        ack = self.read_json_with_token_when_ready(
            self.paths.pause_ack_file, "frame_token", frame_token, timeout_s
        )
        if ack.get("status") != "paused" or not ack.get("paused"):
            raise RuntimeError(f"Pause-first capture failed: {ack}")
        return ack

    # ---- camera request --------------------------------------------------

    def request_camera_pose(self, request_token: str, pose_request: Dict[str, Any]) -> None:
        payload = {
            "status": "pending",
            "request_token": request_token,
            **pose_request,
            "requested_at_utc": utc_now_iso(),
        }
        atomic_write_json(self.paths.camera_request_file, payload)

    def wait_for_camera_ack(self, request_token: str, timeout_s: Optional[float] = None) -> Dict[str, Any]:
        timeout_s = float(timeout_s or self._default_timeout_s())
        return self.read_json_with_token_when_ready(
            self.paths.camera_ack_file, "request_token", request_token, timeout_s
        )

    # ---- screenshot pickup ----------------------------------------------

    def wait_for_screenshot(
        self,
        frame_id: str,
        frame_token: str,
        requested_epoch_s: float,
        timeout_s: Optional[float] = None,
    ) -> Tuple[Path, Dict[str, Any], Dict[str, Any]]:
        timeout_s = float(timeout_s or self._default_timeout_s())
        ack = self.read_json_with_token_when_ready(
            self.paths.screenshot_ack_file, "frame_token", frame_token, timeout_s
        )
        if ack.get("status") != "done":
            raise RuntimeError(f"Screenshot capture failed: {ack}")

        screenshot_name = ack.get("screenshot_name") or self.paths.screenshot_name(frame_id, frame_token)
        source_path = self.paths.screenshots_dir / screenshot_name
        wait_for_stable_file(
            source_path,
            timeout_s=timeout_s,
            poll_interval_s=self._poll_interval_s(),
        )

        file_stat = source_path.stat()
        stale_screenshot_check_passed = (
            file_stat.st_size > 0
            and file_stat.st_mtime >= requested_epoch_s
            and frame_token in screenshot_name
        )
        screenshot_diagnostics = {
            "screenshot_file_mtime": file_stat.st_mtime,
            "screenshot_file_size": file_stat.st_size,
            "screenshot_name_includes_frame_token": frame_token in screenshot_name,
            "stale_screenshot_check_passed": stale_screenshot_check_passed,
        }
        if not stale_screenshot_check_passed:
            self.logger.warning("Stale screenshot check failed: %s", screenshot_diagnostics)

        destination = self.paths.frame_image_path(frame_id, self.image_format)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination)
        if self.runtime_cfg.get("cleanup_source_screenshots", False):
            try:
                source_path.unlink()
            except OSError:
                self.logger.warning("Could not remove DCS screenshot: %s", source_path)

        return destination, ack, screenshot_diagnostics
