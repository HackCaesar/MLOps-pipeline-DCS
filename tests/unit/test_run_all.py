"""Unit tests for apps.pipeline.run_all — uses an injectable mock runner so
no actual subprocesses are launched."""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import List

import pytest
import yaml

from apps.pipeline.run_all import (
    STAGES_ORDER,
    _select_stages,
    run_all,
)

# ---- helpers ----------------------------------------------------------

def _write_cfg(tmp_path: Path) -> Path:
    cfg = {
        "pipeline": {"run_id": None, "seed": 42, "cleanup_tmp_after_run": True},
        "storage": {
            "root_dir":      str(tmp_path / "storage"),
            "datasets_dir":  str(tmp_path / "storage" / "datasets"),
            "tmp_dir":       str(tmp_path / "storage" / "tmp"),
            "artifacts_dir": str(tmp_path / "storage" / "artifacts"),
            "runs_dir":      str(tmp_path / "storage" / "runs"),
            "metadata_dir":  str(tmp_path / "storage" / "metadata"),
        },
        "metadata_store": {"backend": "sqlite",
                            "sqlite_path": str(tmp_path / "p.db")},
        "mlflow": {"enabled": False},
    }
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


class _Recorder:
    def __init__(self):
        self.calls: List[List[str]] = []

    def __call__(self, cmd, **kwargs):
        self.calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0, "ok", "")


# ---- _select_stages ---------------------------------------------------

def test_select_stages_default_is_full_canonical_order() -> None:
    out = _select_stages(only=None, start_at=None, stop_after=None, skip=None)
    assert out == list(STAGES_ORDER)


def test_select_stages_only_filters_in_canonical_order() -> None:
    out = _select_stages(only=["train_yolox", "build_or_reuse_tile_cache"], start_at=None,
                          stop_after=None, skip=None)
    assert out == ["build_or_reuse_tile_cache", "train_yolox"]


def test_select_stages_start_at_and_stop_after() -> None:
    out = _select_stages(only=None, start_at="build_or_reuse_tile_cache",
                          stop_after="evaluate_model", skip=None)
    assert out == ["build_or_reuse_tile_cache", "train_yolox", "evaluate_model"]


def test_select_stages_skip_excludes() -> None:
    out = _select_stages(only=None, start_at=None, stop_after=None,
                          skip=["dcs_capture", "export_trt_int8"])
    assert "dcs_capture" not in out
    assert "export_trt_int8" not in out
    assert "train_yolox" in out


def test_select_stages_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown"):
        _select_stages(only=["does_not_exist"], start_at=None,
                        stop_after=None, skip=None)


# ---- run_all happy path ----------------------------------------------

def test_run_all_success_invokes_every_stage(tmp_path: Path) -> None:
    cfg_path = _write_cfg(tmp_path)
    rec = _Recorder()
    result = run_all(
        config_path=cfg_path, run_id="r1", dataset_id="ds1",
        runner=rec,
    )
    assert result.success is True
    assert len(result.stages) == len(STAGES_ORDER)
    assert all(s.succeeded for s in result.stages)
    # Should have invoked one subprocess per stage.
    assert len(rec.calls) == len(STAGES_ORDER)


def test_run_all_failure_in_middle_marks_run_failed_but_runs_status_update(tmp_path: Path) -> None:
    cfg_path = _write_cfg(tmp_path)
    fail_at = "train_yolox"
    saw_status_update = {"value": False}

    def _runner(cmd, **kwargs):
        if any("apps.yolox_training.cli" in c for c in cmd):
            raise subprocess.CalledProcessError(returncode=2, cmd=cmd,
                                                 stderr="boom", output="")
        if any("pipeline_run_status" in c for c in cmd) and "update" in cmd:
            saw_status_update["value"] = True
        return subprocess.CompletedProcess(cmd, 0, "ok", "")

    result = run_all(
        config_path=cfg_path, run_id="r2", dataset_id="ds2",
        runner=_runner,
    )
    assert result.success is False
    # Status update STILL runs because it is in _CLEANUP_STAGES.
    assert saw_status_update["value"]
    # Failed stage recorded.
    failed = [s for s in result.stages if not s.succeeded]
    assert any(s.name == fail_at for s in failed)


def test_run_all_dry_run_does_not_execute(tmp_path: Path) -> None:
    cfg_path = _write_cfg(tmp_path)
    rec = _Recorder()
    result = run_all(
        config_path=cfg_path, run_id="r3", dataset_id="ds3",
        runner=rec, dry_run=True,
    )
    assert result.success is True
    assert rec.calls == []
    assert all(s.skipped for s in result.stages)


def test_run_all_only_runs_subset(tmp_path: Path) -> None:
    cfg_path = _write_cfg(tmp_path)
    rec = _Recorder()
    result = run_all(
        config_path=cfg_path, run_id="r4", dataset_id="ds4",
        only=["validate_config", "build_or_reuse_tile_cache"], runner=rec,
    )
    names = [s.name for s in result.stages]
    assert names == ["validate_config", "build_or_reuse_tile_cache"]


def test_run_all_autogenerates_run_id_when_missing(tmp_path: Path) -> None:
    cfg_path = _write_cfg(tmp_path)
    rec = _Recorder()
    result = run_all(config_path=cfg_path, runner=rec)
    assert result.run_id is not None
    assert result.dataset_id == f"dcs_{result.run_id}"


def test_run_all_records_stage_output_tails(tmp_path: Path) -> None:
    cfg_path = _write_cfg(tmp_path)

    def _runner(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 0, "stage stdout " + " ".join(cmd[-2:]), "")

    result = run_all(config_path=cfg_path, run_id="r5", dataset_id="ds5",
                      runner=_runner, only=["validate_config"])
    assert "stage stdout" in result.stages[0].stdout_tail
