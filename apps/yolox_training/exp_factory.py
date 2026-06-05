"""Build a YOLOX ``Exp`` instance configured from our pipeline.yaml.

Two-layer design so most of the logic can be unit-tested without torch/yolox:

1. ``ExpConfig`` — pure dataclass aggregated from YAML. No yolox dependency.
2. ``apply_exp_overrides(exp, exp_config)`` — mutates a real yolox Exp instance
   to honour the values in ``ExpConfig``. Yolox is imported only at runtime
   inside ``build_exp_from_config``.

See ``experiments/yolox/class_obj_at_sea.py`` for the actual Exp subclass that
plugs into yolox's machinery (custom dataset names ``images/train`` etc.).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class CheckpointPolicy:
    save_best: bool = True
    save_latest: bool = True
    save_every_n_epochs: int = 20
    save_history_ckpt: bool = False


@dataclass(frozen=True)
class ExpConfig:
    """Snapshot of all training params that come from pipeline.yaml.

    These map 1:1 onto yolox Exp attributes (handled by ``apply_exp_overrides``).
    ``data_dir`` is filled at runtime to point at the reusable tile cache
    (``storage/cache/tiles/{tile_cache_id}``).
    """

    num_classes: int
    input_size: tuple[int, int]
    test_size: tuple[int, int]
    multiscale_range: int
    max_epoch: int
    batch_size: int
    no_aug_epochs: int
    eval_interval: int
    num_workers: int
    fp16: bool
    data_dir: Path
    train_ann: str = "instances_train.json"
    val_ann:   str = "instances_val.json"
    test_ann:  str = "instances_test.json"
    train_image_subdir: str = "images/train"   # passed into COCODataset as `name`
    val_image_subdir:   str = "images/val"
    test_image_subdir:  str = "images/test"
    checkpoint_policy: CheckpointPolicy = field(default_factory=CheckpointPolicy)
    exp_name: str = "class_obj_at_sea"
    output_dir: Optional[Path] = None  # YOLOX puts YOLOX_outputs/<exp_name>/ here


def build_exp_config(
    pipeline_config: Mapping[str, Any],
    *,
    data_dir: Path,
    num_classes: int,
    output_dir: Optional[Path] = None,
    exp_name: Optional[str] = None,
) -> ExpConfig:
    """Build an ``ExpConfig`` from the ``training:`` section of pipeline.yaml.

    The caller passes runtime values that are not in the YAML:
    ``data_dir`` (enriched tmp dir), ``num_classes`` (from raw dataset's
    ``classes.json``), and optional ``output_dir`` (where checkpoints land).
    """
    training = dict(pipeline_config.get("training") or {})
    cp = dict(training.get("checkpoint_policy") or {})

    input_size = _as_size(training.get("input_size", [640, 640]), "input_size")
    test_size  = _as_size(training.get("test_size",  [640, 640]), "test_size")

    return ExpConfig(
        num_classes=int(num_classes),
        input_size=input_size,
        test_size=test_size,
        multiscale_range=int(training.get("multiscale_range", 0)),
        max_epoch=int(training.get("max_epoch", 100)),
        batch_size=int(training.get("batch_size", 8)),
        no_aug_epochs=int(training.get("no_aug_epochs", 15)),
        eval_interval=int(training.get("eval_interval", 10)),
        num_workers=int(training.get("num_workers", 4)),
        fp16=bool(training.get("fp16", True)),
        data_dir=Path(data_dir),
        checkpoint_policy=CheckpointPolicy(
            save_best=bool(cp.get("save_best", True)),
            save_latest=bool(cp.get("save_latest", True)),
            save_every_n_epochs=int(cp.get("save_every_n_epochs", 20)),
            save_history_ckpt=bool(cp.get("save_history_ckpt", False)),
        ),
        exp_name=str(exp_name or "class_obj_at_sea"),
        output_dir=Path(output_dir) if output_dir is not None else None,
    )


def apply_exp_overrides(exp: Any, config: ExpConfig) -> Any:
    """Stamp every override from ``config`` onto a yolox Exp instance.

    ``exp`` must be a yolox Exp (e.g. ``experiments.yolox.class_obj_at_sea.Exp()``).
    We use ``setattr`` so this works without importing yolox here.
    """
    exp.num_classes      = config.num_classes
    exp.input_size       = config.input_size
    exp.test_size        = config.test_size
    exp.multiscale_range = config.multiscale_range
    exp.max_epoch        = config.max_epoch
    exp.no_aug_epochs    = config.no_aug_epochs
    exp.eval_interval    = config.eval_interval
    exp.data_num_workers = config.num_workers
    exp.data_dir         = str(config.data_dir)
    exp.train_ann        = config.train_ann
    exp.val_ann          = config.val_ann
    exp.test_ann         = config.test_ann
    exp.exp_name         = config.exp_name

    # Tell the parent class to skip per-epoch ckpt saves so our CustomTrainer
    # decides ``save_every_n_epochs`` policy itself.
    exp.save_history_ckpt = config.checkpoint_policy.save_history_ckpt

    # Attach the policy + image_subdirs on the Exp object so the Exp subclass
    # (and CustomTrainer) can read them without re-deriving from YAML.
    exp.pipeline_checkpoint_policy = config.checkpoint_policy
    exp.pipeline_train_image_subdir = config.train_image_subdir
    exp.pipeline_val_image_subdir   = config.val_image_subdir
    exp.pipeline_test_image_subdir  = config.test_image_subdir
    return exp


def build_exp_from_config(
    pipeline_config: Mapping[str, Any],
    *,
    data_dir: Path,
    num_classes: int,
    output_dir: Optional[Path] = None,
    exp_name: Optional[str] = None,
) -> Any:
    """End-to-end: read config → build ExpConfig → instantiate our Exp → apply overrides.

    Requires yolox available; raises ``ModuleNotFoundError`` with a clear
    remediation message if not.
    """
    cfg = build_exp_config(
        pipeline_config,
        data_dir=data_dir, num_classes=num_classes,
        output_dir=output_dir, exp_name=exp_name,
    )
    try:
        from experiments.yolox.class_obj_at_sea import Exp
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "YOLOX is not importable. Install via `pip install -v -e /workspace/YOLOX` "
            "or run inside the yolox_trainer Docker container. "
            f"Original error: {exc}"
        ) from exc

    exp = Exp()
    apply_exp_overrides(exp, cfg)
    if cfg.output_dir is not None:
        exp.output_dir = str(cfg.output_dir)
    return exp


# ---------- helpers --------------------------------------------------------

def _as_size(raw: Any, field_name: str) -> tuple[int, int]:
    if not (isinstance(raw, (list, tuple)) and len(raw) == 2):
        raise ValueError(f"training.{field_name} must be [H, W], got {raw!r}")
    return int(raw[0]), int(raw[1])
