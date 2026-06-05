"""Agent-side configuration. Parsed from ``configs/dcs_agent.yaml`` (Phase 11).

The agent uses its own config (not the full pipeline.yaml) because it lives on
the Windows host and only cares about: where DCS lives, where to write raw
datasets, which CLI to invoke, and the HTTP port + shared folder paths.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

from packages.common.config import load_config
from packages.common.paths import resolve_path

AGENT_VERSION = "0.1.0"


@dataclass(frozen=True)
class AgentConfig:
    """Resolved config for the Windows DCS Runner Agent."""

    # HTTP server
    host: str = "0.0.0.0"
    port: int = 8765

    # Shared folder fallback (both HTTP + folder watcher run simultaneously)
    commands_dir: Optional[Path] = None
    status_dir:   Optional[Path] = None
    poll_interval_sec: float = 2.0

    # Subprocess: how to call DCS orchestrator + adapter normalize
    python_executable:   str = "python"
    orchestrator_module: str = "orchestrator.main"   # in DCS_AutoDataset
    orchestrator_cwd:    Optional[Path] = None       # working dir for orchestrator
    adapter_module:      str = "apps.dcs_capture.cli"
    adapter_cwd:         Optional[Path] = None       # working dir for adapter (pipeline root)
    pipeline_config_path: Optional[Path] = None      # to pass to adapter-normalize

    # Default raw dataset target (overridden per-command)
    raw_target_root: Optional[Path] = None

    # Logging
    log_dir: Optional[Path] = None

    @classmethod
    def from_config(cls, raw: Mapping[str, Any]) -> "AgentConfig":
        agent = raw.get("agent") or raw  # accept either {"agent": {...}} or flat
        sub = agent.get("subprocess") or {}
        shared = agent.get("shared_folder") or {}
        http = agent.get("http") or {}
        paths = agent.get("paths") or {}

        return cls(
            host=str(http.get("host", "0.0.0.0")),
            port=int(http.get("port", 8765)),
            commands_dir=resolve_path(shared.get("commands_dir")),
            status_dir=resolve_path(shared.get("status_dir")),
            poll_interval_sec=float(shared.get("poll_interval_sec", 2.0)),
            python_executable=str(sub.get("python_executable", "python")),
            orchestrator_module=str(sub.get("orchestrator_module", "orchestrator.main")),
            orchestrator_cwd=resolve_path(sub.get("orchestrator_cwd")),
            adapter_module=str(sub.get("adapter_module", "apps.dcs_capture.cli")),
            adapter_cwd=resolve_path(sub.get("adapter_cwd")),
            pipeline_config_path=resolve_path(sub.get("pipeline_config_path")),
            raw_target_root=resolve_path(paths.get("raw_target_root")),
            log_dir=resolve_path(paths.get("log_dir")),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "AgentConfig":
        return cls.from_config(load_config(path))
