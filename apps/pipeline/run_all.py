"""Single-process orchestrator for the full DCS → YOLOX pipeline.

Runs the whole pipeline as a plain Python process (driven by
``apps.pipeline.cli run``). Useful for:

- CI smoke tests;
- one-off runs on a dev machine;
- debugging a single stage in isolation (via ``--start-at`` / ``--stop-after``).

Each stage is a thin wrapper over the same CLI commands used in production.
"""
from __future__ import annotations

import datetime as _dt
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from packages.common.config import load_config
from packages.common.logging_utils import get_logger
from packages.common.mlflow_utils import MLflowFacade
from packages.common.paths import resolve_path
from packages.common.run_id import generate_run_id

LOG = get_logger(__name__)


# Canonical ordering of stages, run sequentially in-process.
STAGES_ORDER: Sequence[str] = (
    "validate_config",
    "init_metadata_db",
    "create_pipeline_run",
    "dcs_capture",
    "validate_raw_dataset",
    "register_dataset",
    "build_or_reuse_tile_cache",
    "train_yolox",
    "evaluate_model",
    "export_onnx",
    "export_trt_fp16",
    "export_trt_int8",
    "register_artifacts",
    "update_pipeline_run_status",
)


@dataclass
class StageResult:
    name: str
    skipped: bool = False
    succeeded: bool = True
    duration_sec: float = 0.0
    stdout_tail: str = ""
    stderr_tail: str = ""
    error: Optional[str] = None


@dataclass
class RunResult:
    run_id: str
    dataset_id: Optional[str]
    success: bool
    stages: List[StageResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "dataset_id": self.dataset_id,
            "success": self.success,
            "stages": [vars(s) for s in self.stages],
        }


# ---------------------------------------------------------------------- #
# Public entry
# ---------------------------------------------------------------------- #

def run_all(
    *,
    config_path: str | Path,
    run_id: Optional[str] = None,
    dataset_id: Optional[str] = None,
    dcs_mode: Optional[str] = None,
    start_at: Optional[str] = None,
    stop_after: Optional[str] = None,
    only: Optional[Sequence[str]] = None,
    skip: Optional[Sequence[str]] = None,
    runner: Optional[Callable[..., subprocess.CompletedProcess]] = None,
    dry_run: bool = False,
) -> RunResult:
    """Run the pipeline end-to-end.

    Stage selection (in this priority order):
      - ``only=["enrichment", "train_yolox"]`` runs exactly those stages, in
        canonical order.
      - ``start_at="train_yolox", stop_after="evaluate_model"`` runs the slice.
      - ``skip=["dcs_capture"]`` excludes specific stages from the full pipeline.

    Setting ``dry_run=True`` prints the resolved command for each stage instead
    of executing it. Useful for verifying argv before a long run.
    """
    cfg = load_config(str(config_path))
    config_path = Path(str(config_path)).resolve()
    runner = runner or _real_runner

    run_id = run_id or _generate_run_id_from_cfg(cfg)
    dataset_id = dataset_id or f"dcs_{run_id}"
    selected = _select_stages(only=only, start_at=start_at,
                                stop_after=stop_after, skip=skip)

    LOG.info("run_all: run_id=%s, dataset_id=%s, stages=%s",
             run_id, dataset_id, ",".join(selected))

    # Run tracking (status.json + events.jsonl + SQLite mirror). Strict per
    # tracking.strict; skipped for dry-run. init_run is critical (raises strict).
    tracker = None
    if not dry_run:
        from packages.common.status_tracker import StatusTracker
        tracker = StatusTracker.from_config(cfg, run_id, stage_total=len(selected))
        tracker.init_run(config=cfg, dataset_id=dataset_id, status="running")

    mlflow = MLflowFacade.from_config(cfg)
    with mlflow.start_run(
        run_name=f"{run_id}_pipeline",
        tags={"stage": "pipeline_run_all", "run_id": run_id, "dataset_id": dataset_id},
    ) as parent_mlrun:
        if parent_mlrun.enabled:
            parent_mlrun.log_params({
                "run_id": run_id, "dataset_id": dataset_id,
                "config_path": str(config_path),
                "stages": ",".join(selected),
                "dcs_mode": dcs_mode or "default",
            })

        result = RunResult(run_id=run_id, dataset_id=dataset_id, success=True)
        total = len(selected)
        for i, stage_name in enumerate(selected):
            stage = _STAGE_FUNCS[stage_name]
            if tracker is not None:
                tracker.set_current_stage(stage_name, stage_index=i, stage_total=total)
                tracker.emit_stage(stage_name, "running", current=i, total=total,
                                   message="stage start")
            outcome = _execute_stage(
                stage_name, stage,
                cfg=cfg, config_path=config_path, run_id=run_id,
                dataset_id=dataset_id, dcs_mode=dcs_mode,
                runner=runner, dry_run=dry_run,
            )
            result.stages.append(outcome)
            if tracker is not None:
                if outcome.succeeded:
                    tracker.emit_stage(stage_name, "success", current=i + 1, total=total)
                    if stage_name == "build_or_reuse_tile_cache":
                        _track_tile_cache(tracker, cfg, run_id)
                else:
                    tracker.emit_stage(stage_name, "failed", current=i, total=total,
                                       message=outcome.error or "")
            if not outcome.succeeded and not _is_cleanup_stage(stage_name):
                result.success = False
                LOG.error("Stage %s failed: %s", stage_name, outcome.error)
                # Don't break — let cleanup + status_update still run.

        if tracker is not None:
            tracker.finish(status="success" if result.success else "failed",
                           error=None if result.success else "one or more stages failed")

    return result


def _track_tile_cache(tracker, cfg: dict, run_id: str) -> None:
    """Mirror the Stage-3 breadcrumb into status.json (strict) + SQLite (best-effort)."""
    runs_dir = resolve_path((cfg.get("storage") or {}).get("runs_dir"))
    if runs_dir is None:
        return
    bc = runs_dir / run_id / "tile_cache.json"
    if not bc.is_file():
        return
    try:
        info = json.loads(bc.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return
    tcid = info.get("tile_cache_id")
    if tcid:
        tracker.set_tile_cache(tcid)
    db_path = (cfg.get("metadata_store") or {}).get("sqlite_path")
    if not (db_path and tcid):
        return
    try:  # idempotent (INSERT OR REPLACE) + best-effort: never fail the run on DB issues
        from packages.common.metadata_store import SQLiteMetadataStore
        from packages.common.status_tracker import utc_now_iso
        with SQLiteMetadataStore.open(resolve_path(db_path)) as store:
            store.register_tile_cache(
                tile_cache_id=tcid, dataset_id=info.get("dataset_id") or "",
                dataset_hash=info.get("dataset_hash") or "",
                split_hash=info.get("split_hash") or "",
                tile_config_hash=info.get("tile_config_hash") or "",
                storage_path=info.get("cache_dir") or "",
                num_train_tiles=int(info.get("num_train_tiles") or 0),
                num_val_tiles=int(info.get("num_val_tiles") or 0),
                num_test_tiles=int(info.get("num_test_tiles") or 0),
                created_at=utc_now_iso(), status=info.get("status") or "created",
            )
    except Exception as exc:
        LOG.warning("run_all: tile_cache SQLite registration failed (ignored): %s", exc)


# ---------------------------------------------------------------------- #
# Stage implementations — each returns the argv to invoke.
#
# Convention:
#   - light tasks invoke our CLI in-process (subprocess into same python);
#   - heavy tasks invoke `docker compose run --rm <service> python -m ...`;
#   - the runner picks one based on PIPELINE_DOCKER_COMPOSE env, falling back
#     to in-process when unset.
# ---------------------------------------------------------------------- #

StageFunc = Callable[..., List[str]]


def _stage_validate_config(*, config_path: Path, **_) -> List[str]:
    return [sys.executable, "-m", "apps.pipeline.cli", "validate-config",
            "--config", str(config_path)]


def _stage_init_metadata_db(*, config_path: Path, **_) -> List[str]:
    return [sys.executable, "-m", "scripts.init_metadata_db",
            "--config", str(config_path), "--quiet"]


def _stage_create_pipeline_run(*, config_path: Path, run_id: str, dataset_id: str, **_) -> List[str]:
    return [sys.executable, "-m", "scripts.pipeline_run_status", "create",
            "--config", str(config_path),
            "--run-id", run_id, "--dataset-id", dataset_id,
            "--status", "running", "--replace"]


def _stage_dcs_capture(*, config_path: Path, run_id: str, dataset_id: str,
                       dcs_mode: Optional[str], **_) -> List[str]:
    args = [sys.executable, "-m", "apps.dcs_capture.cli", "request-capture",
            "--config", str(config_path),
            "--run-id", run_id, "--dataset-id", dataset_id]
    if dcs_mode:
        # The CLI reads dcs_capture.mode from config; we override at config-load
        # level via env if user wants. For now log + let config decide.
        LOG.info("dcs_mode override requested: %s (read from config)", dcs_mode)
    return args


def _stage_validate_raw_dataset(*, cfg: dict, dataset_id: str, **_) -> List[str]:
    datasets_dir = resolve_path((cfg.get("storage") or {}).get("datasets_dir"))
    assert datasets_dir is not None
    return [sys.executable, "-m", "scripts.validate_raw_dataset",
            "--dataset-dir", str(datasets_dir / "raw" / dataset_id)]


def _stage_register_dataset(*, config_path: Path, cfg: dict, dataset_id: str, **_) -> List[str]:
    datasets_dir = resolve_path((cfg.get("storage") or {}).get("datasets_dir"))
    assert datasets_dir is not None
    return [sys.executable, "-m", "scripts.register_dataset",
            "--config", str(config_path),
            "--dataset-dir", str(datasets_dir / "raw" / dataset_id),
            "--replace"]


def _stage_build_tile_cache(*, config_path: Path, run_id: str, **_) -> List[str]:
    return [sys.executable, "-m", "apps.dataset_enrichment.cli", "build-cache",
            "--config", str(config_path), "--run-id", run_id]


def _stage_train_yolox(*, config_path: Path, run_id: str, **_) -> List[str]:
    return [sys.executable, "-m", "apps.yolox_training.cli", "train",
            "--config", str(config_path), "--run-id", run_id]


def _stage_evaluate_model(*, config_path: Path, cfg: dict, run_id: str, **_) -> List[str]:
    artifacts = resolve_path((cfg.get("storage") or {}).get("artifacts_dir"))
    assert artifacts is not None
    ckpt = artifacts / "checkpoints" / run_id / "best_ckpt.pth"
    out  = artifacts / "reports" / run_id
    return [sys.executable, "-m", "apps.evaluation_export.cli", "evaluate",
            "--config", str(config_path), "--split", "test",
            "--backend", "pytorch",
            "--checkpoint", str(ckpt),
            "--exp-module", "experiments.yolox.class_obj_at_sea",
            "--device", "cuda", "--fp16",
            "--output-dir", str(out)]


def _stage_export_onnx(*, config_path: Path, cfg: dict, run_id: str, **_) -> List[str]:
    artifacts = resolve_path((cfg.get("storage") or {}).get("artifacts_dir"))
    assert artifacts is not None
    ckpt = artifacts / "checkpoints" / run_id / "best_ckpt.pth"
    out  = artifacts / "exports" / run_id / "model.onnx"
    return [sys.executable, "-m", "apps.evaluation_export.cli", "export-onnx",
            "--config", str(config_path),
            "--checkpoint", str(ckpt),
            "--output", str(out),
            "--exp-module", "experiments.yolox.class_obj_at_sea"]


def _stage_export_trt_fp16(*, config_path: Path, cfg: dict, run_id: str, **_) -> List[str]:
    artifacts = resolve_path((cfg.get("storage") or {}).get("artifacts_dir"))
    assert artifacts is not None
    onnx = artifacts / "exports" / run_id / "model.onnx"
    out  = artifacts / "exports" / run_id / "model_fp16.engine"
    return [sys.executable, "-m", "apps.evaluation_export.cli", "export-trt",
            "--config", str(config_path),
            "--onnx", str(onnx), "--output", str(out),
            "--precision", "fp16"]


def _stage_export_trt_int8(*, config_path: Path, cfg: dict, dataset_id: str, run_id: str, **_) -> List[str]:
    artifacts = resolve_path((cfg.get("storage") or {}).get("artifacts_dir"))
    datasets_dir = resolve_path((cfg.get("storage") or {}).get("datasets_dir"))
    assert artifacts is not None and datasets_dir is not None
    onnx = artifacts / "exports" / run_id / "model.onnx"
    out  = artifacts / "exports" / run_id / "model_int8.engine"
    calib = datasets_dir / "raw" / dataset_id / "images" / "train"
    cache = artifacts / "exports" / run_id / "int8.cache"
    return [sys.executable, "-m", "apps.evaluation_export.cli", "export-trt",
            "--config", str(config_path),
            "--onnx", str(onnx), "--output", str(out),
            "--precision", "int8",
            "--calib-images", str(calib), "--calib-cache", str(cache)]


def _stage_register_artifacts(*, config_path: Path, run_id: str, **_) -> List[str]:
    return [sys.executable, "-m", "scripts.register_artifacts",
            "--config", str(config_path), "--run-id", run_id, "--replace"]


def _stage_update_pipeline_run_status(*, config_path: Path, run_id: str,
                                       cfg: dict, **kwargs) -> List[str]:
    # success/failed status decided by overall run outcome; we always close
    # with "success" here — caller will overwrite to "failed" if needed via
    # the bookkeeping pass after run_all returns.
    status = kwargs.get("final_status", "success")
    return [sys.executable, "-m", "scripts.pipeline_run_status", "update",
            "--config", str(config_path),
            "--run-id", run_id,
            "--status", status, "--auto-finished-at"]


_STAGE_FUNCS: Dict[str, StageFunc] = {
    "validate_config":            _stage_validate_config,
    "init_metadata_db":           _stage_init_metadata_db,
    "create_pipeline_run":        _stage_create_pipeline_run,
    "dcs_capture":                _stage_dcs_capture,
    "validate_raw_dataset":       _stage_validate_raw_dataset,
    "register_dataset":           _stage_register_dataset,
    "build_or_reuse_tile_cache":  _stage_build_tile_cache,
    "train_yolox":                _stage_train_yolox,
    "evaluate_model":             _stage_evaluate_model,
    "export_onnx":                _stage_export_onnx,
    "export_trt_fp16":            _stage_export_trt_fp16,
    "export_trt_int8":            _stage_export_trt_int8,
    "register_artifacts":         _stage_register_artifacts,
    "update_pipeline_run_status": _stage_update_pipeline_run_status,
}

_CLEANUP_STAGES = {"update_pipeline_run_status"}


def _is_cleanup_stage(name: str) -> bool:
    return name in _CLEANUP_STAGES


# ---------------------------------------------------------------------- #
# Stage selection
# ---------------------------------------------------------------------- #

def _select_stages(
    *,
    only: Optional[Sequence[str]],
    start_at: Optional[str],
    stop_after: Optional[str],
    skip: Optional[Sequence[str]],
) -> List[str]:
    if only:
        for s in only:
            if s not in _STAGE_FUNCS:
                raise ValueError(f"Unknown stage: {s!r}")
        return [s for s in STAGES_ORDER if s in set(only)]

    selected = list(STAGES_ORDER)
    if start_at:
        if start_at not in selected:
            raise ValueError(f"Unknown start_at stage: {start_at!r}")
        i = selected.index(start_at)
        selected = selected[i:]
    if stop_after:
        if stop_after not in selected:
            raise ValueError(f"stop_after stage not in remaining selection: {stop_after!r}")
        i = selected.index(stop_after)
        selected = selected[: i + 1]
    if skip:
        skip_set = set(skip)
        selected = [s for s in selected if s not in skip_set]
    return selected


# ---------------------------------------------------------------------- #
# Stage execution
# ---------------------------------------------------------------------- #

def _execute_stage(
    name: str,
    func: StageFunc,
    *,
    cfg: dict,
    config_path: Path,
    run_id: str,
    dataset_id: str,
    dcs_mode: Optional[str],
    runner: Callable[..., subprocess.CompletedProcess],
    dry_run: bool,
) -> StageResult:
    LOG.info("─── stage %s ───", name)
    started = _now()
    cmd = func(
        cfg=cfg, config_path=config_path, run_id=run_id,
        dataset_id=dataset_id, dcs_mode=dcs_mode,
    )
    if dry_run:
        LOG.info("DRY-RUN cmd: %s", " ".join(cmd))
        return StageResult(name=name, skipped=True, succeeded=True,
                            duration_sec=0.0)
    try:
        proc = runner(cmd, capture_output=True, text=True, check=True)
        return StageResult(
            name=name, succeeded=True,
            duration_sec=_now() - started,
            stdout_tail=(proc.stdout or "")[-2000:],
            stderr_tail=(proc.stderr or "")[-2000:],
        )
    except subprocess.CalledProcessError as exc:
        return StageResult(
            name=name, succeeded=False,
            duration_sec=_now() - started,
            stdout_tail=((exc.stdout or "") if isinstance(exc.stdout, str) else "")[-2000:],
            stderr_tail=((exc.stderr or "") if isinstance(exc.stderr, str) else "")[-2000:],
            error=f"exit code {exc.returncode}",
        )
    except FileNotFoundError as exc:
        return StageResult(name=name, succeeded=False,
                            duration_sec=_now() - started,
                            error=f"command not found: {exc}")


def _real_runner(cmd, **kwargs):
    return subprocess.run(cmd, **kwargs)


def _now() -> float:
    return _dt.datetime.now().timestamp()


def _generate_run_id_from_cfg(cfg: dict) -> str:
    runs_dir = resolve_path((cfg.get("storage") or {}).get("runs_dir"))
    return generate_run_id(runs_dir=runs_dir) if runs_dir else generate_run_id()
