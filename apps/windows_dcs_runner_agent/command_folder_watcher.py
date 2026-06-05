"""Poll ``shared/commands/*.command.json`` and submit them to ``CaptureRunner``.

Fallback path when the controller can't reach the agent's HTTP endpoint.
Same payload format as ``POST /capture/start``. Processed commands are renamed
to ``*.command.json.handled`` so they aren't re-submitted.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Optional

from apps.windows_dcs_runner_agent.process_manager import CaptureRunner, CaptureSpec
from packages.common.logging_utils import get_logger

LOG = get_logger(__name__)


class CommandFolderWatcher:
    """Background thread: polls ``commands_dir`` and submits new commands."""

    def __init__(
        self,
        runner: CaptureRunner,
        commands_dir: Path,
        *,
        poll_interval_sec: float = 2.0,
    ) -> None:
        self.runner = runner
        self.commands_dir = Path(commands_dir)
        self.poll_interval_sec = float(poll_interval_sec)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self.commands_dir.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                          name="command-folder-watcher")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.poll_interval_sec * 2 + 1)

    def _loop(self) -> None:
        LOG.info("CommandFolderWatcher polling %s every %.1fs",
                 self.commands_dir, self.poll_interval_sec)
        while not self._stop.is_set():
            try:
                self._scan_once()
            except Exception:  # noqa: BLE001
                LOG.exception("CommandFolderWatcher scan failed")
            self._stop.wait(self.poll_interval_sec)

    def _scan_once(self) -> None:
        for path in sorted(self.commands_dir.glob("*.command.json")):
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                spec = CaptureSpec.from_dict(raw)
            except (json.JSONDecodeError, ValueError) as exc:
                LOG.error("Invalid command file %s: %s — renaming to .invalid", path, exc)
                self._mark_handled(path, ".invalid")
                continue

            accepted = self.runner.submit(spec)
            if accepted:
                LOG.info("Accepted command from %s for run_id=%s", path, spec.run_id)
                self._mark_handled(path, ".handled")
            else:
                # leave the file; we'll retry on the next tick once runner is free.
                LOG.debug("Runner busy; leaving %s for next tick", path)

    @staticmethod
    def _mark_handled(path: Path, suffix: str) -> None:
        target = path.with_suffix(path.suffix + suffix)
        try:
            path.rename(target)
        except OSError as exc:
            LOG.warning("Could not rename %s → %s: %s", path, target, exc)
