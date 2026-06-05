"""Unit tests for apps.windows_dcs_runner_agent.config."""
from __future__ import annotations

from pathlib import Path

import yaml

from apps.windows_dcs_runner_agent.config import (
    AGENT_VERSION,
    AgentConfig,
)


def test_agent_config_defaults() -> None:
    cfg = AgentConfig()
    assert cfg.host == "0.0.0.0"
    assert cfg.port == 8765
    assert cfg.poll_interval_sec == 2.0
    assert cfg.python_executable == "python"
    assert AGENT_VERSION  # non-empty


def test_agent_config_from_dict_full(tmp_path: Path) -> None:
    raw = {
        "agent": {
            "http": {"host": "127.0.0.1", "port": 9999},
            "shared_folder": {
                "commands_dir": str(tmp_path / "cmd"),
                "status_dir": str(tmp_path / "status"),
                "poll_interval_sec": 0.5,
            },
            "subprocess": {
                "python_executable": "/usr/bin/python3",
                "orchestrator_module": "orch.main",
                "orchestrator_cwd": str(tmp_path / "dcs"),
                "adapter_module": "apps.dcs_capture.cli",
                "adapter_cwd": str(tmp_path / "pipe"),
                "pipeline_config_path": str(tmp_path / "pipeline.yaml"),
            },
            "paths": {
                "raw_target_root": str(tmp_path / "raw"),
                "log_dir": str(tmp_path / "logs"),
            },
        }
    }
    cfg = AgentConfig.from_config(raw)
    assert cfg.host == "127.0.0.1"
    assert cfg.port == 9999
    assert cfg.poll_interval_sec == 0.5
    assert cfg.python_executable == "/usr/bin/python3"
    assert cfg.orchestrator_module == "orch.main"
    assert cfg.commands_dir == (tmp_path / "cmd")
    assert cfg.pipeline_config_path == (tmp_path / "pipeline.yaml")
    assert cfg.log_dir == (tmp_path / "logs")


def test_agent_config_from_flat_dict() -> None:
    """``from_config`` should accept either nested under ``agent`` or flat."""
    raw = {"http": {"host": "1.2.3.4", "port": 7777}}
    cfg = AgentConfig.from_config(raw)
    assert cfg.host == "1.2.3.4"
    assert cfg.port == 7777


def test_agent_config_from_file(tmp_path: Path) -> None:
    p = tmp_path / "agent.yaml"
    p.write_text(yaml.safe_dump({
        "agent": {"http": {"port": 8888}},
    }), encoding="utf-8")
    cfg = AgentConfig.from_file(p)
    assert cfg.port == 8888
