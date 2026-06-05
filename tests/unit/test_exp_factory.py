"""Unit tests for apps.yolox_training.exp_factory.

These tests do NOT require yolox — they exercise:
- ``build_exp_config`` (pure dataclass aggregation from YAML);
- ``apply_exp_overrides`` (mutation of a fake yolox Exp object).
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from apps.yolox_training.exp_factory import (
    CheckpointPolicy,
    apply_exp_overrides,
    build_exp_config,
)

_PIPELINE_CFG = {
    "training": {
        "max_epoch": 100,
        "batch_size": 8,
        "input_size": [640, 640],
        "test_size":  [640, 640],
        "multiscale_range": 0,
        "no_aug_epochs": 15,
        "eval_interval": 10,
        "num_workers": 4,
        "fp16": True,
        "allow_cpu_fallback": False,
        "checkpoint_policy": {
            "save_best": True,
            "save_latest": True,
            "save_every_n_epochs": 20,
            "save_history_ckpt": False,
        },
    },
}


def test_build_exp_config_propagates_values(tmp_path: Path) -> None:
    cfg = build_exp_config(
        _PIPELINE_CFG, data_dir=tmp_path / "enriched", num_classes=4,
    )
    assert cfg.num_classes == 4
    assert cfg.input_size == (640, 640)
    assert cfg.test_size  == (640, 640)
    assert cfg.multiscale_range == 0
    assert cfg.max_epoch == 100
    assert cfg.no_aug_epochs == 15
    assert cfg.eval_interval == 10
    assert cfg.fp16 is True
    assert cfg.data_dir == tmp_path / "enriched"
    # locked decision: save_history_ckpt is False
    assert cfg.checkpoint_policy.save_history_ckpt is False
    assert cfg.checkpoint_policy.save_every_n_epochs == 20


def test_build_exp_config_no_aug_epochs_overridable(tmp_path: Path) -> None:
    cfg = build_exp_config(
        {"training": {**_PIPELINE_CFG["training"], "no_aug_epochs": 10}},
        data_dir=tmp_path, num_classes=4,
    )
    assert cfg.no_aug_epochs == 10


def test_build_exp_config_defaults_when_section_missing(tmp_path: Path) -> None:
    cfg = build_exp_config({}, data_dir=tmp_path, num_classes=4)
    assert cfg.max_epoch == 100
    assert cfg.input_size == (640, 640)
    assert cfg.multiscale_range == 0
    assert cfg.no_aug_epochs == 15


def test_build_exp_config_rejects_bad_input_size(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="input_size"):
        build_exp_config(
            {"training": {**_PIPELINE_CFG["training"], "input_size": [640]}},
            data_dir=tmp_path, num_classes=4,
        )


def test_build_exp_config_sets_default_image_subdirs(tmp_path: Path) -> None:
    cfg = build_exp_config(_PIPELINE_CFG, data_dir=tmp_path, num_classes=4)
    assert cfg.train_image_subdir == "images/train"
    assert cfg.val_image_subdir   == "images/val"
    assert cfg.test_image_subdir  == "images/test"


def test_build_exp_config_carries_output_dir(tmp_path: Path) -> None:
    out = tmp_path / "ckpts"
    cfg = build_exp_config(_PIPELINE_CFG, data_dir=tmp_path, num_classes=4,
                            output_dir=out, exp_name="run42")
    assert cfg.output_dir == out
    assert cfg.exp_name == "run42"


def test_apply_exp_overrides_mutates_fake_exp(tmp_path: Path) -> None:
    cfg = build_exp_config(_PIPELINE_CFG, data_dir=tmp_path / "enr", num_classes=4)
    fake_exp = SimpleNamespace(
        # Default values that the override should replace.
        num_classes=80, input_size=(1024, 1024), test_size=(1024, 1024),
        multiscale_range=5, max_epoch=300, no_aug_epochs=999,
        eval_interval=1, save_history_ckpt=True, data_num_workers=0,
        data_dir=None, train_ann="bogus.json", val_ann="bogus.json",
        test_ann="bogus.json", exp_name="bogus",
    )
    apply_exp_overrides(fake_exp, cfg)
    assert fake_exp.num_classes == 4
    assert fake_exp.input_size == (640, 640)
    assert fake_exp.multiscale_range == 0
    assert fake_exp.save_history_ckpt is False
    assert fake_exp.data_dir == str(tmp_path / "enr")
    assert fake_exp.train_ann == "instances_train.json"
    assert fake_exp.val_ann   == "instances_val.json"
    # Subdirs attached on the Exp instance for the subclass / trainer to read.
    assert fake_exp.pipeline_train_image_subdir == "images/train"
    assert fake_exp.pipeline_val_image_subdir   == "images/val"
    # Checkpoint policy attached too.
    assert isinstance(fake_exp.pipeline_checkpoint_policy, CheckpointPolicy)
    assert fake_exp.pipeline_checkpoint_policy.save_every_n_epochs == 20


def test_apply_exp_overrides_returns_same_object(tmp_path: Path) -> None:
    cfg = build_exp_config(_PIPELINE_CFG, data_dir=tmp_path, num_classes=4)
    fake_exp = SimpleNamespace()
    out = apply_exp_overrides(fake_exp, cfg)
    assert out is fake_exp


def test_checkpoint_policy_defaults() -> None:
    cp = CheckpointPolicy()
    assert cp.save_best is True
    assert cp.save_latest is True
    assert cp.save_every_n_epochs == 20
    assert cp.save_history_ckpt is False


def test_exp_config_frozen(tmp_path: Path) -> None:
    cfg = build_exp_config(_PIPELINE_CFG, data_dir=tmp_path, num_classes=4)
    with pytest.raises(Exception):
        cfg.max_epoch = 50  # type: ignore[misc]
