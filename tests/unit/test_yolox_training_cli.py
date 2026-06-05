"""Tests for apps.yolox_training.cli — CLI shape, check-env, graceful failure."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_cli_help_works() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "apps.yolox_training.cli", "--help"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=15,
    )
    assert proc.returncode == 0, proc.stderr
    assert "train" in proc.stdout
    assert "dry-run" in proc.stdout
    assert "check-env" in proc.stdout


def test_cli_check_env_runs() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "apps.yolox_training.cli", "check-env"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=15,
    )
    assert proc.returncode == 0, proc.stderr
    assert "torch installed:" in proc.stdout
    assert "yolox installed:" in proc.stdout


def test_train_without_enrichment_returns_error(tmp_path: Path) -> None:
    """When the enriched dataset directory does not exist, train should fail
    cleanly instead of crashing in YOLOX internals."""
    cfg = {
        "pipeline": {"run_id": "test_run", "seed": 1},
        "storage":  {
            "root_dir":      str(tmp_path),
            "cache_dir":     str(tmp_path / "cache"),
            "artifacts_dir": str(tmp_path / "artifacts"),
        },
        "data":     {"raw_dataset_dir": str(tmp_path / "raw")},
        "enrichment": {},
        "training": {
            "max_epoch": 1, "batch_size": 2, "input_size": [640, 640],
            "test_size": [640, 640], "multiscale_range": 0,
            "no_aug_epochs": 0, "eval_interval": 1, "num_workers": 0,
            "fp16": False, "allow_cpu_fallback": True,
            "checkpoint_policy": {"save_best": True, "save_latest": True,
                                  "save_every_n_epochs": 0, "save_history_ckpt": False},
        },
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    proc = subprocess.run(
        [sys.executable, "-m", "apps.yolox_training.cli", "train",
         "--config", str(cfg_path)],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 1
    assert "Tile cache directory does not exist" in (proc.stdout + proc.stderr)


def test_train_no_gpu_without_fallback_errors(tmp_path: Path) -> None:
    """If torch is installed but CUDA missing AND allow_cpu_fallback is false,
    train should refuse to run with a clear message."""
    # Make sure the tile-cache dir EXISTS so we exercise the GPU-policy branch.
    from packages.common import hashing
    raw = tmp_path / "raw"
    raw.mkdir()
    cache_root = tmp_path / "cache" / "tiles"
    tcid = hashing.compute_hashes(raw, {})["tile_cache_id"]
    (cache_root / tcid).mkdir(parents=True)
    cfg = {
        "pipeline": {"run_id": "test_run", "seed": 1},
        "storage":  {
            "root_dir":      str(tmp_path),
            "cache_dir":     str(tmp_path / "cache"),
            "artifacts_dir": str(tmp_path / "artifacts"),
        },
        "data":     {"raw_dataset_dir": str(raw)},
        "enrichment": {},
        "training": {
            "max_epoch": 1, "batch_size": 2, "input_size": [640, 640],
            "test_size": [640, 640], "multiscale_range": 0,
            "no_aug_epochs": 0, "eval_interval": 1, "num_workers": 0,
            "fp16": False, "allow_cpu_fallback": False,  # ← strict policy
            "checkpoint_policy": {"save_best": True, "save_latest": True,
                                  "save_every_n_epochs": 0, "save_history_ckpt": False},
        },
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    proc = subprocess.run(
        [sys.executable, "-m", "apps.yolox_training.cli", "train",
         "--config", str(cfg_path)],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=30,
    )
    # In this WSL env we have no torch + no CUDA. Either:
    #   - torch missing → "torch is not importable" → exit 1
    #   - torch present + no CUDA → "No CUDA GPU detected" → exit 1
    out = proc.stdout + proc.stderr
    assert proc.returncode == 1
    assert ("No CUDA GPU detected" in out) or ("torch is not importable" in out)


def test_class_obj_at_sea_module_requires_yolox() -> None:
    """The Exp file must import yolox at top-level; without yolox the import errors.

    Confirming this so we know our lazy-import pattern works: callers should
    never accidentally rely on yolox being importable.
    """
    try:
        # yolox is installed; the assertion below would fail. Skip instead.
        import pytest
        import yolox  # noqa: F401
        pytest.skip("yolox is installed; this test only relevant without yolox")
    except ImportError:
        pass

    try:
        from experiments.yolox import class_obj_at_sea  # noqa: F401
    except ModuleNotFoundError as exc:
        assert "yolox" in str(exc).lower()
    else:
        raise AssertionError("Expected ModuleNotFoundError for missing yolox")
