"""Atomic JSON/YAML/file helpers used by the orchestrator and exchange layer.

These small utilities are kept in one place because they share the same
retry-on-PermissionError semantics required by the Windows file-replace path
when DCS Lua scripts are concurrently reading/writing the same JSON files.

Nothing here knows about DCS specifics; they operate on plain ``Path``s.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def utc_now_iso() -> str:
    """ISO-8601 UTC timestamp with millisecond precision, ``Z``-suffixed.

    Format chosen for stability across language stacks reading the metadata
    (matches what existing dataset/metadata files already store).
    """
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Write JSON to ``path`` via tmp + replace, retrying on ``PermissionError``.

    The retry loop matters on Windows where another process (DCS Lua reader)
    may briefly hold a file handle. Up to 20 attempts at 50 ms each (~1 s)
    before re-raising.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    data = json.dumps(payload, indent=2)
    for attempt in range(20):
        temp_path.write_text(data, encoding="utf-8")
        try:
            temp_path.replace(path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.05)


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_yaml_config(config_path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required to load config.yaml. Install with: pip install pyyaml"
        ) from exc
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def wait_for_stable_file(path: Path, timeout_s: float, poll_interval_s: float) -> Path:
    """Wait until ``path`` exists and its size stops changing.

    The file is considered ready after two consecutive non-zero size reads
    return the same value. This guards against DCS Lua emitting a partially
    written JSON snapshot.
    """
    deadline = time.time() + timeout_s
    last_size = -1
    stable_ticks = 0
    while time.time() < deadline:
        if path.exists():
            try:
                current_size = path.stat().st_size
            except OSError:
                current_size = -1
            if current_size > 0 and current_size == last_size:
                stable_ticks += 1
                if stable_ticks >= 2:
                    return path
            else:
                stable_ticks = 0
                last_size = current_size
        time.sleep(poll_interval_s)
    raise TimeoutError(f"Timed out waiting for stable file: {path}")
