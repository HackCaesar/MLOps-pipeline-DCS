"""Unit tests for apps.dcs_capture.agent_client (transport selection + folder fallback)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from apps.dcs_capture.agent_client import (
    AgentClient,
    AgentTimeoutError,
    CaptureRequest,
)


def _make_req(tmp_path: Path) -> CaptureRequest:
    return CaptureRequest(
        run_id="r1", dataset_id="ds1",
        mission_path=str(tmp_path / "m.miz"),
        dcs_config_path=str(tmp_path / "dcs.yaml"),
        dcs_source_dataset_dir=str(tmp_path / "src"),
        target_dataset_dir=str(tmp_path / "dst"),
    )


# ---- transport selection ---------------------------------------------

def test_no_url_falls_back_to_shared_folder(tmp_path: Path) -> None:
    cmd = tmp_path / "cmd"
    st = tmp_path / "status"
    cmd.mkdir(); st.mkdir()
    client = AgentClient(agent_url=None, commands_dir=cmd, status_dir=st)
    assert client.health() is True            # because shared folder is configured
    assert client.transport == "shared"


def test_no_url_no_shared_folder_fails(tmp_path: Path) -> None:
    client = AgentClient(agent_url=None)
    assert client.health() is False


def test_unreachable_url_falls_back_when_shared_available(tmp_path: Path) -> None:
    cmd = tmp_path / "cmd"
    st = tmp_path / "status"
    cmd.mkdir(); st.mkdir()
    client = AgentClient(
        agent_url="http://127.0.0.1:1",       # nothing listening
        commands_dir=cmd, status_dir=st,
        http_timeout_sec=0.5,
    )
    assert client.health() is True            # fallback
    assert client.transport == "shared"


def test_prefer_shared_folder_skips_health(tmp_path: Path) -> None:
    cmd = tmp_path / "cmd"
    cmd.mkdir()
    client = AgentClient(
        agent_url="http://example:9999",
        commands_dir=cmd, status_dir=tmp_path / "st",
        prefer_shared_folder=True,
    )
    assert client.transport == "shared"       # locked without ever probing HTTP


# ---- shared folder submit ----------------------------------------------

def test_request_capture_writes_command_file(tmp_path: Path) -> None:
    cmd = tmp_path / "cmd"
    st = tmp_path / "status"
    cmd.mkdir(); st.mkdir()
    client = AgentClient(agent_url=None, commands_dir=cmd, status_dir=st)
    resp = client.request_capture(_make_req(tmp_path))
    assert resp["accepted"] is True
    assert resp["transport"] == "shared"
    files = list(cmd.glob("*.command.json"))
    assert len(files) == 1
    payload = json.loads(files[0].read_text(encoding="utf-8"))
    assert payload["run_id"] == "r1"
    assert payload["dataset_id"] == "ds1"


def test_request_capture_creates_commands_dir_if_missing(tmp_path: Path) -> None:
    cmd = tmp_path / "new_cmd"            # does NOT exist yet
    st = tmp_path / "new_status"
    client = AgentClient(agent_url=None, commands_dir=cmd, status_dir=st,
                         prefer_shared_folder=True)
    client.request_capture(_make_req(tmp_path))
    assert cmd.is_dir()


# ---- shared folder status polling --------------------------------------

def test_get_status_returns_none_when_no_file(tmp_path: Path) -> None:
    st = tmp_path / "status"
    st.mkdir()
    client = AgentClient(agent_url=None, commands_dir=tmp_path / "cmd",
                         status_dir=st)
    client.health()
    assert client.get_status("r1") is None


def test_get_status_reads_disk_file(tmp_path: Path) -> None:
    st = tmp_path / "status"
    st.mkdir()
    (st / "r1.status.json").write_text(json.dumps({
        "run_id": "r1", "status": "running", "phase": "launching_dcs",
    }), encoding="utf-8")
    client = AgentClient(agent_url=None, commands_dir=tmp_path / "cmd",
                         status_dir=st)
    client.health()
    got = client.get_status("r1")
    assert got["status"] == "running"
    assert got["phase"] == "launching_dcs"


def test_wait_for_status_returns_when_terminal(tmp_path: Path) -> None:
    st = tmp_path / "status"
    st.mkdir()
    # Pre-create a terminal status file.
    (st / "r1.status.json").write_text(json.dumps({
        "run_id": "r1", "status": "success",
    }), encoding="utf-8")
    client = AgentClient(agent_url=None, commands_dir=tmp_path / "cmd",
                         status_dir=st)
    client.health()
    final = client.wait_for_status("r1", timeout_sec=2.0, poll_interval_sec=0.05)
    assert final["status"] == "success"


def test_wait_for_status_timeout_raises(tmp_path: Path) -> None:
    st = tmp_path / "status"
    st.mkdir()
    (st / "r1.status.json").write_text(json.dumps({
        "run_id": "r1", "status": "running",  # never terminal
    }), encoding="utf-8")
    client = AgentClient(agent_url=None, commands_dir=tmp_path / "cmd",
                         status_dir=st)
    client.health()
    with pytest.raises(AgentTimeoutError, match="timed out"):
        client.wait_for_status("r1", timeout_sec=0.3, poll_interval_sec=0.05)


def test_stop_capture_writes_sentinel(tmp_path: Path) -> None:
    cmd = tmp_path / "cmd"
    cmd.mkdir()
    client = AgentClient(agent_url=None, commands_dir=cmd,
                         status_dir=tmp_path / "status",
                         prefer_shared_folder=True)
    assert client.stop_capture("r1") is True
    assert (cmd / "r1.stop.json").is_file()


def test_capture_request_to_dict_round_trip(tmp_path: Path) -> None:
    req = _make_req(tmp_path)
    d = req.to_dict()
    for k in ("run_id", "dataset_id", "mission_path", "dcs_config_path",
              "dcs_source_dataset_dir", "target_dataset_dir", "num_frames",
              "timeout_sec"):
        assert k in d
