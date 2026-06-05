"""Train entry point for YOLOX. Reads pipeline.yaml + raw dataset metadata,
constructs an ``Exp`` and runs ``CustomTrainer``.

Two paths:

- ``run_training(config)``: full training. Errors out if no GPU is available
  unless ``training.allow_cpu_fallback=true`` or ``training.dry_run=true``.
- ``run_dry_run(config, max_iter)``: short run for smoke testing. CPU OK.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from apps.yolox_training.exp_factory import build_exp_from_config
from apps.yolox_training.trainer_subclass import TrainingHooks, make_custom_trainer
from packages.common.logging_utils import get_logger
from packages.common.mlflow_utils import MLflowFacade, flatten_params
from packages.common.paths import resolve_path
from packages.common.seed import set_seed
from packages.common.status_tracker import ModelInfo, StatusTracker

LOG = get_logger(__name__)


@dataclass(frozen=True)
class EnvCheck:
    has_torch: bool
    has_cuda: bool
    cuda_device_count: int
    torch_version: Optional[str] = None


def check_env() -> EnvCheck:
    """Inspect torch availability — no-op-friendly when torch missing."""
    try:
        import torch
        cuda_ok = bool(torch.cuda.is_available())
        n = torch.cuda.device_count() if cuda_ok else 0
        return EnvCheck(True, cuda_ok, n, torch_version=torch.__version__)
    except ImportError:
        return EnvCheck(False, False, 0)


def _resolve_cache_root(storage: Mapping[str, Any]) -> Optional[Path]:
    cache_dir_raw = storage.get("cache_dir")
    if not cache_dir_raw and storage.get("root_dir"):
        cache_dir_raw = f"{storage.get('root_dir')}/cache"
    cache_root = resolve_path(cache_dir_raw) if cache_dir_raw else None
    return (cache_root / "tiles") if cache_root is not None else None


def _resolve_tile_cache_dir(config: Mapping[str, Any], run_id: str,
                            raw_dataset_dir: Optional[Path]) -> Path:
    """data_dir = the tile cache built by `build-cache`.

    Prefer the breadcrumb written by the build-cache stage
    (``runs/{run_id}/tile_cache.json``); otherwise recompute the cache id
    deterministically from the raw dataset + enrichment config (same hashing the
    cache builder uses). No dependence on storage/tmp.
    """
    storage = config.get("storage") or {}
    runs_dir = resolve_path(storage.get("runs_dir"))
    if runs_dir is not None and run_id:
        breadcrumb = runs_dir / run_id / "tile_cache.json"
        if breadcrumb.is_file():
            try:
                cache_dir = json.loads(breadcrumb.read_text(encoding="utf-8")).get("cache_dir")
                if cache_dir:
                    return Path(cache_dir)
            except json.JSONDecodeError:
                pass

    cache_root = _resolve_cache_root(storage)
    if cache_root is None or raw_dataset_dir is None:
        raise SystemExit(
            "Cannot resolve the tile cache: set storage.cache_dir + data.raw_dataset_dir, "
            "or run `apps.dataset_enrichment.cli build-cache` first."
        )
    from apps.dataset_enrichment.tile_cache import resolve_cache_dir
    return resolve_cache_dir(raw_dataset_dir, config.get("enrichment") or {}, cache_root)


def resolve_training_paths(config: Mapping[str, Any], run_id: str) -> dict[str, Path]:
    """Resolve data_dir (tile cache), ckpt_dir, raw_dataset_dir from the config."""
    storage = config.get("storage") or {}
    artifacts = resolve_path(storage.get("artifacts_dir"))
    if artifacts is None:
        raise SystemExit("storage.artifacts_dir is required")

    raw_dataset_dir_raw = (config.get("data") or {}).get("raw_dataset_dir")
    raw_dataset_dir = resolve_path(raw_dataset_dir_raw) if raw_dataset_dir_raw else None

    data_dir = _resolve_tile_cache_dir(config, run_id, raw_dataset_dir)

    # ckpt_dir from training.ckpt_dir if explicit, else artifacts/checkpoints/{run_id}/
    training = config.get("training") or {}
    raw_ckpt_dir = training.get("ckpt_dir") or f"{artifacts}/checkpoints/{run_id}"
    # ${pipeline.run_id} may already be substituted; safe to resolve.
    ckpt_dir = resolve_path(raw_ckpt_dir.replace("${pipeline.run_id}", run_id) if isinstance(raw_ckpt_dir, str) else str(raw_ckpt_dir))
    assert ckpt_dir is not None

    return {
        "data_dir":        data_dir,
        "ckpt_dir":        ckpt_dir,
        "raw_dataset_dir": raw_dataset_dir,
        "artifacts":       artifacts,
    }


def load_num_classes(raw_dataset_dir: Optional[Path]) -> int:
    """Read num_classes from the raw dataset's ``classes.json``.

    Falls back to the canonical class count (3) when metadata is missing — never
    the legacy default of 4 (see packages/common/classes.py).
    """
    from packages.common.classes import NUM_CLASSES as _CANON_NUM_CLASSES
    if raw_dataset_dir is None:
        return _CANON_NUM_CLASSES
    p = raw_dataset_dir / "metadata" / "classes.json"
    if not p.is_file():
        return _CANON_NUM_CLASSES
    try:
        return len(json.loads(p.read_text(encoding="utf-8")).get("categories", []))
    except (json.JSONDecodeError, KeyError):
        return _CANON_NUM_CLASSES


def run_training(
    config: Mapping[str, Any],
    *,
    run_id: str,
    dry_run: bool = False,
    max_iter: Optional[int] = None,
) -> int:
    """Run training (or a dry-run). Returns CLI exit code."""
    seed = (config.get("pipeline") or {}).get("seed", 42)
    set_seed(seed)

    paths = resolve_training_paths(config, run_id)
    if not paths["data_dir"].is_dir():
        LOG.error("Tile cache directory does not exist: %s. "
                  "Run `apps.dataset_enrichment.cli build-cache` first.", paths["data_dir"])
        return 1

    # ---- env check + CPU fallback policy (§15.1) ----
    env = check_env()
    training_cfg = config.get("training") or {}
    allow_cpu = bool(training_cfg.get("allow_cpu_fallback", False)) or dry_run

    if not env.has_torch:
        LOG.error("torch is not importable. Install torch or run inside yolox_trainer container.")
        return 1
    if not env.has_cuda and not allow_cpu:
        LOG.error(
            "No CUDA GPU detected. Full training requires GPU. To enable: install "
            "NVIDIA Container Toolkit + enable Docker Desktop GPU support; or set "
            "training.allow_cpu_fallback=true for a (slow) CPU run."
        )
        return 1
    if not env.has_cuda:
        LOG.warning("Running on CPU (allow_cpu_fallback=true or dry_run=true). "
                    "Expect very slow throughput.")

    # ---- build Exp ----
    num_classes = load_num_classes(paths["raw_dataset_dir"])
    paths["ckpt_dir"].mkdir(parents=True, exist_ok=True)
    LOG.info("Building Exp: data_dir=%s num_classes=%d ckpt_dir=%s",
             paths["data_dir"], num_classes, paths["ckpt_dir"])

    exp = build_exp_from_config(
        config,
        data_dir=paths["data_dir"],
        num_classes=num_classes,
        output_dir=paths["ckpt_dir"].parent,
        exp_name=run_id,
    )

    # ---- run tracking (best-effort from the train subprocess; writes to the
    #      shared runs/{run_id}/status.json + events.jsonl that run_all created) ----
    tracker = StatusTracker.from_config(config, run_id, strict=False)
    tracker.set_model(ModelInfo(
        name="YOLOX-S", num_classes=num_classes,
        input_size=list(getattr(exp, "input_size", []) or []), pretrained=None))

    # ---- args mirror the shape that yolox.tools.train expects ----
    args = _make_yolox_args(config, env, paths, run_id, dry_run=dry_run)

    # ---- mlflow lifecycle wraps the entire training run ----
    mlflow = MLflowFacade.from_config(config)
    with mlflow.start_run(
        run_name=run_id,
        tags={"stage": "training", "dry_run": str(dry_run),
              "device": str(env.has_cuda and "cuda" or "cpu")},
    ) as mlrun:
        # Static params from training section + checkpoint policy
        params: Dict[str, Any] = {}
        params.update(flatten_params("training", training_cfg))
        params["num_classes"] = num_classes
        params["run_id"] = run_id
        params["seed"] = seed
        params["dataset_id"] = paths["raw_dataset_dir"].name if paths["raw_dataset_dir"] else None
        params["torch_version"] = env.torch_version
        params["cuda_available"] = env.has_cuda
        params["cuda_device_count"] = env.cuda_device_count
        mlrun.log_params(params)

        # Hooks: per-epoch metrics + checkpoint artifact logging
        hooks = _build_training_hooks(mlrun, paths, tracker)
        trainer = make_custom_trainer(exp, args, max_iter=max_iter, hooks=hooks)

        LOG.info("Starting %s (max_epoch=%d, no_aug_epochs=%d, eval_interval=%d, ms_range=%d)",
                 "DRY-RUN" if dry_run else "training",
                 exp.max_epoch, exp.no_aug_epochs, exp.eval_interval, exp.multiscale_range)
        trainer.train()

        # Final artifacts — best/latest checkpoints + the snapshot config (if exists)
        for name in ("best_ckpt.pth", "latest_ckpt.pth", "last_epoch_ckpt.pth", "last_mosaic_epoch_ckpt.pth"):
            ckpt_path = paths["ckpt_dir"] / name
            if ckpt_path.is_file():
                mlrun.log_artifact(ckpt_path, artifact_path="checkpoints")
        LOG.info("Training finished. Checkpoints in %s", paths["ckpt_dir"])
        return 0


def _build_training_hooks(mlrun, paths: dict, tracker=None) -> "TrainingHooks":
    """Per-epoch metrics → MLflow; per-iter progress → status tracker (best-effort)."""

    _last_log = [0.0]

    def _on_after_iter(trainer: Any) -> None:
        if tracker is None:
            return
        iter_count = int(getattr(trainer, "pipeline_iter_count", 0))
        now = time.time()
        if not tracker.should_log_iter(iter_count, _last_log[0], now):
            return
        _last_log[0] = now
        epoch = _safe_int(getattr(trainer, "epoch", None), add1=True)
        max_epoch = _safe_int(getattr(trainer, "max_epoch", None))
        it = _safe_int(getattr(trainer, "iter", None), add1=True)
        max_it = _safe_int(getattr(trainer, "max_iter", None))
        # Only report metrics we can read reliably (never fabricate — null otherwise).
        loss = _safe_meter(trainer, "total_loss")
        lr = _safe_meter(trainer, "lr")
        tracker.record_training_progress(epoch=epoch, max_epoch=max_epoch,
                                         iter=it, max_iter=max_it, loss=loss, lr=lr)
        msg = f"Epoch {epoch}/{max_epoch}" if (epoch and max_epoch) else "training"
        tracker.emit_progress("train", "running", current=it, total=max_it,
                              message=msg, payload=_eta_payload(trainer))

    def _on_after_epoch(trainer: Any) -> None:
        epoch = int(getattr(trainer, "epoch", 0)) + 1
        metrics: Dict[str, float] = {}
        if hasattr(trainer, "meter") and trainer.meter is not None:
            try:
                # YOLOX collects loss meters under trainer.meter (MeterBuffer)
                for k, m in trainer.meter.items():
                    try:
                        metrics[f"train/{k}"] = float(m.latest)
                    except (AttributeError, TypeError, ValueError):
                        continue
            except Exception:
                pass
        if hasattr(trainer, "best_ap"):
            try:
                metrics["val/ap50_95_best"] = float(trainer.best_ap)
            except (TypeError, ValueError):
                pass
        if metrics:
            mlrun.log_metrics(metrics, step=epoch)

    def _on_save_checkpoint(trainer: Any, ckpt_name: str) -> None:
        ckpt_path = Path(paths["ckpt_dir"]) / f"{ckpt_name}_ckpt.pth"
        if ckpt_path.is_file():
            mlrun.log_artifact(ckpt_path, artifact_path="checkpoints")

    return TrainingHooks(
        on_after_epoch=_on_after_epoch,
        on_save_checkpoint=_on_save_checkpoint,
        on_after_iter=_on_after_iter,
    )


# ---------- helpers --------------------------------------------------------

def _safe_int(value: Any, *, add1: bool = False) -> Optional[int]:
    try:
        n = int(value)
        return n + 1 if add1 else n
    except (TypeError, ValueError):
        return None


def _safe_meter(trainer: Any, key: str) -> Optional[float]:
    """Read a yolox MeterBuffer value (e.g. total_loss, lr) — None if unavailable."""
    try:
        meter = getattr(trainer, "meter", None)
        if meter is None:
            return None
        return float(meter[key].latest)
    except (KeyError, AttributeError, TypeError, ValueError):
        return None


def _eta_payload(trainer: Any) -> dict:
    """ETA seconds computed like yolox; empty dict if not derivable (never fabricated)."""
    try:
        it_time = float(trainer.meter["iter_time"].global_avg)
        left = int(trainer.max_iter) * int(trainer.max_epoch) - (int(trainer.progress_in_iter) + 1)
        return {"eta_seconds": round(it_time * max(0, left), 1)}
    except Exception:
        return {}

def _make_yolox_args(config, env, paths, run_id, *, dry_run: bool):
    """Build the argparse-like ``args`` namespace that yolox.Trainer expects."""
    training = config.get("training") or {}
    ns = argparse.Namespace()
    ns.experiment_name = run_id
    ns.name = run_id
    ns.dist_backend = "nccl"
    ns.dist_url     = None
    ns.batch_size   = int(training.get("batch_size", 8))
    ns.devices      = 1 if env.has_cuda else 0
    ns.local_rank   = 0
    ns.machine_rank = 0
    ns.num_machines = 1
    ns.resume       = False
    ns.start_epoch  = None
    ns.fp16         = bool(training.get("fp16", True)) and env.has_cuda
    ns.cache        = False
    ns.occupy       = False
    ns.exp_file     = None
    # Fine-tune from pretrained weights (e.g. COCO yolox_s.pth) when configured.
    # YOLOX loads matching-shape tensors and re-inits the 3-class head automatically.
    pretrained = training.get("pretrained_weights")
    ckpt_path = resolve_path(pretrained) if pretrained else None
    if ckpt_path is not None and not ckpt_path.is_file():
        LOG.warning("training.pretrained_weights=%s not found — training from scratch.", ckpt_path)
        ckpt_path = None
    if ckpt_path is not None:
        LOG.info("Fine-tuning from pretrained weights: %s", ckpt_path)
    ns.ckpt         = str(ckpt_path) if ckpt_path else None
    ns.opts         = []
    ns.logger       = "tensorboard"
    return ns
