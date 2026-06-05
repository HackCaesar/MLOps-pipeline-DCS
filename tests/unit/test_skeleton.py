"""Skeleton-level smoke tests: every CLI's --help must work without import errors."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


CLI_MODULES = [
    "apps.pipeline.cli",
    "apps.dcs_capture.cli",
    "apps.dataset_enrichment.cli",
    "apps.yolox_training.cli",
    "apps.evaluation_export.cli",
    "apps.windows_dcs_runner_agent.agent",
]


@pytest.mark.parametrize("module", CLI_MODULES)
def test_cli_help_exits_zero(module: str) -> None:
    proc = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert proc.returncode == 0, (
        f"`python -m {module} --help` returned {proc.returncode}\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    assert "usage" in proc.stdout.lower()


def test_validate_config_runs_on_shipped_yaml() -> None:
    config_path = REPO_ROOT / "configs" / "pipeline.yaml"
    assert config_path.exists()

    proc = subprocess.run(
        [sys.executable, "-m", "apps.pipeline.cli", "validate-config", "--config", str(config_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert proc.returncode == 0, f"validate-config failed: {proc.stderr}"
    assert '"training.max_epoch"' in proc.stdout
    assert '"training.multiscale_range": 0' in proc.stdout, \
        "multiscale_range must be 0 (locked decision §13)"
