"""Integration test: spin up a real HTTP server bound to a free port,
make HTTP requests with AgentClient, validate end-to-end behavior."""
from __future__ import annotations

import subprocess
import threading
import time
from pathlib import Path
from typing import Iterator, List, Tuple

import pytest

from apps.dcs_capture.agent_client import AgentClient, CaptureRequest
from apps.windows_dcs_runner_agent.api import make_server
from apps.windows_dcs_runner_agent.config import AgentConfig
from apps.windows_dcs_runner_agent.process_manager import CaptureRunner
from apps.windows_dcs_runner_agent.status_writer import StatusStore


def _agent_config(tmp_path: Path) -> AgentConfig:
    return AgentConfig(
        host="127.0.0.1", port=0,
        commands_dir=tmp_path / "commands",
        status_dir=tmp_path / "status",
        log_dir=tmp_path / "logs",
        orchestrator_cwd=tmp_path,
        adapter_cwd=tmp_path,
    )


def _success_runner(calls: List[List[str]]):
    def _run(cmd, **kwargs):
        calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0, "", "")
    return _run


@pytest.fixture
def live_agent(tmp_path: Path) -> Iterator[Tuple[AgentClient, CaptureRunner, str]]:
    cfg = _agent_config(tmp_path)
    store = StatusStore(status_dir=cfg.status_dir)
    calls: List[List[str]] = []
    runner = CaptureRunner(cfg, store, process_runner=_success_runner(calls))
    server = make_server(cfg.host, cfg.port, runner, store)
    actual_port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True,
                          name="test-agent-server")
    t.start()
    client = AgentClient(
        agent_url=f"http://127.0.0.1:{actual_port}",
        commands_dir=cfg.commands_dir,
        status_dir=cfg.status_dir,
        http_timeout_sec=2.0,
    )
    try:
        yield client, runner, str(tmp_path)
    finally:
        server.shutdown()
        server.server_close()


def _make_request(tmp_path_str: str, run_id: str = "r1") -> CaptureRequest:
    tp = Path(tmp_path_str)
    return CaptureRequest(
        run_id=run_id, dataset_id=f"ds_{run_id}",
        mission_path=str(tp / "m.miz"),
        dcs_config_path=str(tp / "dcs.yaml"),
        dcs_source_dataset_dir=str(tp / "src"),
        target_dataset_dir=str(tp / "dst"),
        num_frames=2, timeout_sec=10,
    )


def test_health_endpoint_ok(live_agent) -> None:
    client, _, _ = live_agent
    assert client.health() is True


def test_request_capture_via_http_succeeds(live_agent) -> None:
    client, runner, tmp = live_agent
    req = _make_request(tmp, "r1")
    resp = client.request_capture(req)
    assert resp["accepted"] is True
    assert resp["transport"] == "http"
    runner.wait_for_completion(timeout=5.0)


def test_status_via_http_returns_terminal(live_agent) -> None:
    client, runner, tmp = live_agent
    client.request_capture(_make_request(tmp, "r_status"))
    final = client.wait_for_status("r_status", timeout_sec=5.0,
                                     poll_interval_sec=0.1)
    assert final["status"] == "success"


def test_status_for_unknown_run_returns_none(live_agent) -> None:
    client, _, _ = live_agent
    assert client.get_status("ghost") is None


def test_busy_returns_409_via_client(live_agent, tmp_path: Path) -> None:
    """Second request while first is in flight must be rejected."""
    client, runner, tmp = live_agent
    # Trick: replace runner's process_runner with a blocking one.
    release = threading.Event()
    def _hold(cmd, **kwargs):
        release.wait(timeout=5)
        return subprocess.CompletedProcess(cmd, 0, "", "")
    runner._process_runner = _hold

    client.request_capture(_make_request(tmp, "r1"))
    # Second submit while first still running.
    try:
        client.request_capture(_make_request(tmp, "r2"))
    except Exception as exc:
        assert "409" in str(exc) or "busy" in str(exc).lower()
    finally:
        release.set()
        runner.wait_for_completion(timeout=5.0)


def test_stop_capture_via_http(live_agent, tmp_path: Path) -> None:
    client, runner, tmp = live_agent
    release = threading.Event()
    def _hold(cmd, **kwargs):
        release.wait(timeout=5)
        return subprocess.CompletedProcess(cmd, 0, "", "")
    runner._process_runner = _hold

    client.request_capture(_make_request(tmp, "r_stop"))
    # Give the worker thread a moment to mark itself busy.
    time.sleep(0.1)
    stopped = client.stop_capture("r_stop")
    assert stopped is True
    release.set()
    runner.wait_for_completion(timeout=5.0)
    final = client.get_status("r_stop")
    assert final is not None
    assert final["status"] == "stopped"
