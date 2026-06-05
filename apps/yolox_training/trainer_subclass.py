"""``CustomTrainer`` — a thin layer over ``yolox.core.trainer.Trainer``.

What it adds:

- ``save_every_n_epochs`` checkpoint policy (the stock trainer either saves every
  epoch or none at all);
- max-iter cap for ``dry-run``;
- MLflow logging hook (placeholder; full integration in Phase 10).

Imports yolox at runtime — the module is importable without it, but
instantiating ``CustomTrainer`` requires yolox + torch.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass(frozen=True)
class TrainingHooks:
    """Optional callbacks that ``CustomTrainer`` invokes at well-defined points."""

    on_after_epoch:    Optional[Callable[[Any], None]] = None
    on_after_train:    Optional[Callable[[Any], None]] = None
    on_save_checkpoint: Optional[Callable[[Any, str], None]] = None
    on_after_iter:     Optional[Callable[[Any], None]] = None  # best-effort progress


def make_custom_trainer(exp: Any, args: Any, *, max_iter: Optional[int] = None,
                        hooks: Optional[TrainingHooks] = None) -> Any:
    """Construct a ``CustomTrainer`` instance.

    ``exp`` must already have ``pipeline_checkpoint_policy`` set on it (see
    ``apps.yolox_training.exp_factory.apply_exp_overrides``).
    """
    try:
        from yolox.core.trainer import Trainer as YoloxTrainer
    except ImportError as exc:
        raise ModuleNotFoundError(
            "yolox is not importable; cannot construct CustomTrainer. "
            "Install yolox or run inside the yolox_trainer container."
        ) from exc

    policy = getattr(exp, "pipeline_checkpoint_policy", None)
    if policy is None:
        raise RuntimeError(
            "Exp instance is missing pipeline_checkpoint_policy. "
            "Did you forget apply_exp_overrides()?"
        )

    class _Inner(YoloxTrainer):
        def __init__(self, _exp, _args):
            super().__init__(_exp, _args)
            self.pipeline_policy = policy
            self.pipeline_hooks = hooks or TrainingHooks()
            self.pipeline_max_iter = max_iter
            self.pipeline_iter_count = 0

        def after_iter(self):
            super().after_iter()
            self.pipeline_iter_count += 1
            if (self.pipeline_max_iter is not None
                    and self.pipeline_iter_count >= self.pipeline_max_iter):
                # Force the epoch loop to break by setting epoch to max_epoch.
                # YOLOX checks ``self.epoch + 1 == self.max_epoch`` in after_epoch.
                self.epoch = self.max_epoch - 1
                self.iter = self.max_iter - 1
            # Iteration progress is BEST-EFFORT — a failing hook must never break training.
            if self.pipeline_hooks.on_after_iter:
                try:
                    self.pipeline_hooks.on_after_iter(self)
                except Exception:
                    pass

        def after_epoch(self):
            super().after_epoch()
            epoch_n = self.epoch + 1
            every_n = self.pipeline_policy.save_every_n_epochs
            if (every_n and every_n > 0 and epoch_n % every_n == 0
                    and epoch_n != self.max_epoch):
                ckpt_name = f"epoch_{epoch_n:03d}"
                self.save_ckpt(ckpt_name)
                if self.pipeline_hooks.on_save_checkpoint:
                    self.pipeline_hooks.on_save_checkpoint(self, ckpt_name)
            if self.pipeline_hooks.on_after_epoch:
                self.pipeline_hooks.on_after_epoch(self)

        def after_train(self):
            super().after_train()
            if self.pipeline_hooks.on_after_train:
                self.pipeline_hooks.on_after_train(self)

    return _Inner(exp, args)
