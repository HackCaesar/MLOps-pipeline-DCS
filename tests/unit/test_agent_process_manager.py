"""Unit tests for apps.windows_dcs_runner_agent.process_manager.CaptureRunner.

Uses an injected ``process_runner`` mock instead of actually launching DCS or
subprocess. Tests cover: submit/reject when busy, success path, failure path,
stop semantics, and status transitions.
"""
from __future__ import annotations

import subprocess
import threading
from pathlib import Path
from typing import Callable, List

import pytest

from apps.windows_dcs_runner_agent.config import AgentConfig
from apps.windows_dcs_runner_agent.process_manager import (
    CaptureRunner,
    CaptureSpec,
)
from apps.windows_dcs_runner_agent.status_writer import StatusStore


def _agent_config(tmp_path: Path) -> AgentConfig:
    return AgentConfig(
        host="127.0.0.1", port=0,
        commands_dir=tmp_path / "commands",
        status_dir=tmp_path / "status",
        python_executable="python",
        orchestrator_module="orchestrator.main",
        orchestrator_cwd=tmp_path / "dcs",
        adapter_module="apps.dcs_capture.cli",
        adapter_cwd=tmp_path / "pipeline",
        pipeline_config_path=tmp_path / "pipeline.yaml",
        log_dir=tmp_path / "logs",
    )


def _spec(tmp_path: Path, **overrides) -> CaptureSpec:
    defaults = dict(
        run_id="r1", dataset_id="ds1",
        mission_path=tmp_path / "m.miz",
        dcs_config_path=tmp_path / "dcs.yaml",
        dcs_source_dataset_dir=tmp_path / "dcs_dataset",
        target_dataset_dir=tmp_path / "raw_target",
        num_frames=3, timeout_sec=10,
    )
    defaults.update(overrides)
    return CaptureSpec(**defaults)


# ---- CaptureSpec.from_dict --------------------------------------------

def test_capture_spec_from_dict() -> None:
    spec = CaptureSpec.from_dict({
        "run_id": "r1", "dataset_id": "d1",
        "mission_path": "/a/m.miz", "dcs_config_path": "/a/dcs.yaml",
        "dcs_source_dataset_dir": "/a/src", "target_dataset_dir": "/a/dst",
        "num_frames": 5, "timeout_sec": 100,
    })
    assert spec.run_id == "r1" and spec.num_frames == 5


def test_capture_spec_missing_required_raises() -> None:
    with pytest.raises(ValueError, match="missing required fields"):
        CaptureSpec.from_dict({"run_id": "r1"})


def test_capture_spec_defaults_num_frames_and_timeout(tmp_path: Path) -> None:
    spec = CaptureSpec.from_dict({
        "run_id": "r", "dataset_id": "d",
        "mission_path": "/a/m.miz", "dcs_config_path": "/a/dcs.yaml",
        "dcs_source_dataset_dir": "/a/src", "target_dataset_dir": "/a/dst",
    })
    assert spec.num_frames == 1
    assert spec.timeout_sec == 7200


# ---- CaptureRunner happy path -----------------------------------------

def _success_runner_calls(calls: List[List[str]]) -> Callable:
    def _run(cmd, **kwargs):
        calls.append(list(cmd))
        # close any log file handle passed via stdout=
        return subprocess.CompletedProcess(cmd, 0, "", "")
    return _run


def test_runner_happy_path_runs_orchestrator_then_adapter(tmp_path: Path) -> None:
    cfg = _agent_config(tmp_path)
    store = StatusStore(status_dir=cfg.status_dir)
    calls: List[List[str]] = []
    runner = CaptureRunner(cfg, store, process_runner=_success_runner_calls(calls))
    spec = _spec(tmp_path)

    assert runner.submit(spec) is True
    assert runner.wait_for_completion(timeout=5.0) is True

    final = store.get("r1")
    assert final is not None
    assert final.status == "success", f"got: {final.to_dict()}"
    assert final.phase == "done"
    assert final.raw_dataset_dir == str(spec.target_dataset_dir)

    # Two subprocess invocations: orchestrator and adapter.
    assert len(calls) == 2
    assert any("orchestrator.main" in c for c in calls[0])
    assert "adapter-normalize" in calls[1]
    assert "--dataset-id" in calls[1] and "ds1" in calls[1]


def test_runner_rejects_second_submit_while_busy(tmp_path: Path) -> None:
    cfg = _agent_config(tmp_path)
    store = StatusStore(status_dir=cfg.status_dir)

    # Hold the first subprocess until released.
    release = threading.Event()

    def _hold(cmd, **kwargs):
        release.wait(timeout=5)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    runner = CaptureRunner(cfg, store, process_runner=_hold)
    assert runner.submit(_spec(tmp_path, run_id="r1")) is True
    assert runner.submit(_spec(tmp_path, run_id="r2")) is False
    release.set()
    runner.wait_for_completion(timeout=5.0)


def test_runner_failure_path_records_error(tmp_path: Path) -> None:
    cfg = _agent_config(tmp_path)
    store = StatusStore(status_dir=cfg.status_dir)

    def _fail(cmd, **kwargs):
        raise subprocess.CalledProcessError(returncode=2, cmd=cmd,
                                             stderr="boom")

    runner = CaptureRunner(cfg, store, process_runner=_fail)
    runner.submit(_spec(tmp_path))
    runner.wait_for_completion(timeout=5.0)
    cur = store.get("r1")
    assert cur.status == "failed"
    assert "Subprocess exit 2" in cur.error
    assert "boom" in cur.error


def test_runner_timeout_path_records_error(tmp_path: Path) -> None:
    cfg = _agent_config(tmp_path)
    store = StatusStore(status_dir=cfg.status_dir)

    def _timeout(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=kwargs.get("timeout", 0))

    runner = CaptureRunner(cfg, store, process_runner=_timeout)
    spec = _spec(tmp_path, timeout_sec=42)
    runner.submit(spec)
    runner.wait_for_completion(timeout=5.0)
    cur = store.get("r1")
    assert cur.status == "failed"
    assert "Timed out" in cur.error or "timed out" in cur.error


def test_runner_unexpected_exception_records_failed(tmp_path: Path) -> None:
    cfg = _agent_config(tmp_path)
    store = StatusStore(status_dir=cfg.status_dir)
    def _boom(cmd, **kwargs):
        raise RuntimeError("kaboom")
    runner = CaptureRunner(cfg, store, process_runner=_boom)
    runner.submit(_spec(tmp_path))
    runner.wait_for_completion(timeout=5.0)
    cur = store.get("r1")
    assert cur.status == "failed"
    assert "kaboom" in cur.error


def test_runner_is_busy_active_run_id(tmp_path: Path) -> None:
    cfg = _agent_config(tmp_path)
    store = StatusStore(status_dir=cfg.status_dir)
    release = threading.Event()

    def _hold(cmd, **kwargs):
        release.wait(timeout=5)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    runner = CaptureRunner(cfg, store, process_runner=_hold)
    runner.submit(_spec(tmp_path, run_id="r1"))
    assert runner.is_busy() is True
    assert runner.active_run_id() == "r1"
    release.set()
    runner.wait_for_completion(timeout=5.0)
    assert runner.is_busy() is False


def test_runner_stop_returns_false_for_wrong_run_id(tmp_path: Path) -> None:
    cfg = _agent_config(tmp_path)
    store = StatusStore(status_dir=cfg.status_dir)
    runner = CaptureRunner(cfg, store, process_runner=lambda **kw: None)
    assert runner.stop("ghost") is False


def test_runner_writes_status_disk_mirror(tmp_path: Path) -> None:
    cfg = _agent_config(tmp_path)
    store = StatusStore(status_dir=cfg.status_dir)
    calls: List[List[str]] = []
    runner = CaptureRunner(cfg, store, process_runner=_success_runner_calls(calls))
    runner.submit(_spec(tmp_path))
    runner.wait_for_completion(timeout=5.0)
    path = cfg.status_dir / "r1.status.json"
    assert path.is_file()
