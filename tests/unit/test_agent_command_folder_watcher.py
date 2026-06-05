"""Unit tests for command_folder_watcher.

Drive ``_scan_once`` synchronously (no thread) so we can assert behavior
deterministically.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from apps.windows_dcs_runner_agent.command_folder_watcher import (
    CommandFolderWatcher,
)
from apps.windows_dcs_runner_agent.config import AgentConfig
from apps.windows_dcs_runner_agent.process_manager import CaptureRunner
from apps.windows_dcs_runner_agent.status_writer import StatusStore


def _make_runner(tmp_path: Path, *, success: bool = True) -> CaptureRunner:
    cfg = AgentConfig(
        host="127.0.0.1", port=0,
        commands_dir=tmp_path / "cmd",
        status_dir=tmp_path / "status",
        log_dir=tmp_path / "logs",
        orchestrator_cwd=tmp_path,
        adapter_cwd=tmp_path,
    )
    store = StatusStore(status_dir=cfg.status_dir)

    def _run(cmd, **kwargs):
        if not success:
            raise subprocess.CalledProcessError(returncode=1, cmd=cmd, stderr="fail")
        return subprocess.CompletedProcess(cmd, 0, "", "")
    return CaptureRunner(cfg, store, process_runner=_run)


def _write_command(commands_dir: Path, run_id: str, tmp_path: Path) -> Path:
    commands_dir.mkdir(parents=True, exist_ok=True)
    cmd = {
        "run_id": run_id, "dataset_id": f"ds_{run_id}",
        "mission_path": str(tmp_path / "m.miz"),
        "dcs_config_path": str(tmp_path / "dcs.yaml"),
        "dcs_source_dataset_dir": str(tmp_path / "src"),
        "target_dataset_dir": str(tmp_path / "dst"),
    }
    p = commands_dir / f"{run_id}.command.json"
    p.write_text(json.dumps(cmd), encoding="utf-8")
    return p


def test_scan_processes_command_and_marks_handled(tmp_path: Path) -> None:
    runner = _make_runner(tmp_path)
    commands_dir = tmp_path / "cmd"
    cmd_path = _write_command(commands_dir, "r1", tmp_path)

    watcher = CommandFolderWatcher(runner, commands_dir, poll_interval_sec=0.01)
    watcher._scan_once()
    runner.wait_for_completion(timeout=5.0)

    # Original .command.json is gone, renamed to .handled
    assert not cmd_path.is_file()
    handled = list(commands_dir.glob("*.command.json.handled"))
    assert len(handled) == 1


def test_scan_marks_invalid_command(tmp_path: Path) -> None:
    runner = _make_runner(tmp_path)
    commands_dir = tmp_path / "cmd"
    commands_dir.mkdir()
    bad = commands_dir / "broken.command.json"
    bad.write_text("not json at all")

    watcher = CommandFolderWatcher(runner, commands_dir, poll_interval_sec=0.01)
    watcher._scan_once()

    assert not bad.is_file()
    invalid = list(commands_dir.glob("*.command.json.invalid"))
    assert len(invalid) == 1


def test_scan_leaves_file_when_runner_busy(tmp_path: Path) -> None:
    """If runner is busy, the file stays for the next tick."""
    import threading
    cfg = AgentConfig(host="127.0.0.1", port=0,
                       commands_dir=tmp_path / "cmd",
                       status_dir=tmp_path / "status",
                       log_dir=tmp_path / "logs")
    store = StatusStore(status_dir=cfg.status_dir)
    release = threading.Event()
    def _hold(cmd, **kwargs):
        release.wait(timeout=5)
        return subprocess.CompletedProcess(cmd, 0, "", "")
    runner = CaptureRunner(cfg, store, process_runner=_hold)

    # Pre-busy the runner with a first capture.
    p1 = _write_command(tmp_path / "cmd", "r1", tmp_path)
    watcher = CommandFolderWatcher(runner, tmp_path / "cmd", poll_interval_sec=0.01)
    watcher._scan_once()
    assert not p1.is_file()                        # accepted + handled

    # Now drop a second command while the first still holds the slot.
    p2 = _write_command(tmp_path / "cmd", "r2", tmp_path)
    watcher._scan_once()
    assert p2.is_file()                            # left for retry

    # Release, retry — should now be accepted.
    release.set()
    runner.wait_for_completion(timeout=5.0)
    watcher._scan_once()
    runner.wait_for_completion(timeout=5.0)
    assert not p2.is_file()
    handled = sorted(p.name for p in (tmp_path / "cmd").glob("*.handled"))
    assert handled == ["r1.command.json.handled", "r2.command.json.handled"]
