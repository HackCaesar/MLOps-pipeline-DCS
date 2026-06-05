"""Unit tests for scripts.pipeline_run_status CLI."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

from packages.common.metadata_store import SQLiteMetadataStore

REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_cfg(tmp_path: Path, db_path: Path) -> Path:
    cfg = {"metadata_store": {"backend": "sqlite", "sqlite_path": str(db_path)}}
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def _run(*args, cwd=REPO_ROOT):
    return subprocess.run(
        [sys.executable, "-m", "scripts.pipeline_run_status", *args],
        cwd=cwd, capture_output=True, text=True, timeout=15,
    )


def test_create_inserts_new_row(tmp_path: Path) -> None:
    db = tmp_path / "p.db"
    cfg = _write_cfg(tmp_path, db)
    # Pre-register the dataset to satisfy the pipeline_runs.dataset_id FK.
    with SQLiteMetadataStore(db) as store:
        store.init_db()
        store.register_dataset(
            dataset_id="ds1", name="Smoke",
            created_at="2026-05-27T00:00:00Z", source="test",
            storage_path="storage/datasets/raw/ds1",
            annotation_format="coco",
            num_images=1, num_annotations=1, num_classes=4,
        )
    proc = _run("create", "--config", str(cfg), "--run-id", "r1",
                 "--dataset-id", "ds1", "--status", "running")
    assert proc.returncode == 0, proc.stderr
    with SQLiteMetadataStore(db) as store:
        row = store.get_run("r1")
    assert row is not None
    assert row.status == "running"
    assert row.dataset_id == "ds1"


def test_create_soft_nulls_unknown_dataset_id(tmp_path: Path) -> None:
    """If --dataset-id refers to a missing row, the CLI warns and records
    dataset_id=NULL — matches the DAG order (create_pipeline_run runs before
    register_dataset)."""
    db = tmp_path / "p.db"
    cfg = _write_cfg(tmp_path, db)
    proc = _run("create", "--config", str(cfg), "--run-id", "r2",
                 "--dataset-id", "ghost", "--status", "running")
    assert proc.returncode == 0, proc.stderr
    with SQLiteMetadataStore(db) as store:
        row = store.get_run("r2")
    assert row is not None
    assert row.dataset_id is None
    assert "not yet registered" in (proc.stderr + proc.stdout)


def test_create_existing_no_replace_is_no_op(tmp_path: Path) -> None:
    db = tmp_path / "p.db"
    cfg = _write_cfg(tmp_path, db)
    _run("create", "--config", str(cfg), "--run-id", "r1", "--status", "running")
    proc = _run("create", "--config", str(cfg), "--run-id", "r1", "--status", "queued")
    assert proc.returncode == 0
    out = json.loads(proc.stdout.splitlines()[-1] if proc.stdout.count("\n") == 0 else proc.stdout)
    assert out.get("existing") is True


def test_create_with_replace_overwrites(tmp_path: Path) -> None:
    db = tmp_path / "p.db"
    cfg = _write_cfg(tmp_path, db)
    _run("create", "--config", str(cfg), "--run-id", "r1", "--status", "running")
    proc = _run("create", "--config", str(cfg), "--run-id", "r1",
                 "--status", "queued", "--replace")
    assert proc.returncode == 0
    with SQLiteMetadataStore(db) as store:
        assert store.get_run("r1").status == "queued"


def test_update_transitions_status(tmp_path: Path) -> None:
    db = tmp_path / "p.db"
    cfg = _write_cfg(tmp_path, db)
    _run("create", "--config", str(cfg), "--run-id", "r1", "--status", "running")
    proc = _run("update", "--config", str(cfg), "--run-id", "r1",
                 "--status", "success", "--auto-finished-at")
    assert proc.returncode == 0
    with SQLiteMetadataStore(db) as store:
        row = store.get_run("r1")
    assert row.status == "success"
    assert row.finished_at is not None
    assert row.finished_at.endswith("Z")


def test_update_unknown_run_returns_error(tmp_path: Path) -> None:
    db = tmp_path / "p.db"
    cfg = _write_cfg(tmp_path, db)
    with SQLiteMetadataStore(db) as s:
        s.init_db()
    proc = _run("update", "--config", str(cfg), "--run-id", "ghost",
                 "--status", "failed")
    assert proc.returncode == 1
    assert "No pipeline_run" in (proc.stderr + proc.stdout)


def test_get_prints_row_json(tmp_path: Path) -> None:
    db = tmp_path / "p.db"
    cfg = _write_cfg(tmp_path, db)
    _run("create", "--config", str(cfg), "--run-id", "r1", "--status", "running")
    proc = _run("get", "--config", str(cfg), "--run-id", "r1")
    assert proc.returncode == 0
    parsed = json.loads(proc.stdout)
    assert parsed["run_id"] == "r1"
    assert parsed["status"] == "running"


def test_update_records_error_message(tmp_path: Path) -> None:
    db = tmp_path / "p.db"
    cfg = _write_cfg(tmp_path, db)
    _run("create", "--config", str(cfg), "--run-id", "r1", "--status", "running")
    _run("update", "--config", str(cfg), "--run-id", "r1",
          "--status", "failed", "--error", "explosion", "--auto-finished-at")
    with SQLiteMetadataStore(db) as store:
        row = store.get_run("r1")
    assert row.status == "failed"
    assert row.error == "explosion"
