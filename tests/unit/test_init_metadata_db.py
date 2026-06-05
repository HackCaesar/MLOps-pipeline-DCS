"""End-to-end smoke for scripts/init_metadata_db.py over a tmp config."""
from __future__ import annotations

import sqlite3
import subprocess
import sys
from pathlib import Path

import yaml

from packages.common.metadata_store import TABLE_NAMES

REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_tmp_config(tmp_path: Path) -> Path:
    db_path = tmp_path / "metadata" / "pipeline.db"
    cfg = {
        "pipeline":       {"run_id": None, "seed": 1, "cleanup_tmp_after_run": True, "log_level": "INFO"},
        "storage":        {"root_dir": str(tmp_path / "storage")},
        "metadata_store": {"backend": "sqlite", "sqlite_path": str(db_path)},
    }
    p = tmp_path / "pipeline.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def test_init_metadata_db_creates_file_and_tables(tmp_path: Path) -> None:
    config_path = _write_tmp_config(tmp_path)
    proc = subprocess.run(
        [sys.executable, "-m", "scripts.init_metadata_db",
         "--config", str(config_path), "--quiet"],
        cwd=REPO_ROOT,
        capture_output=True, text=True, timeout=20,
    )
    assert proc.returncode == 0, f"stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}"

    # Inspect the resulting DB directly via sqlite3.
    db_path = tmp_path / "metadata" / "pipeline.db"
    assert db_path.exists(), "DB file should exist after init_metadata_db"
    conn = sqlite3.connect(str(db_path))
    try:
        present = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}
    finally:
        conn.close()

    for name in TABLE_NAMES:
        assert name in present, f"missing table after init: {name}"


def test_init_metadata_db_is_idempotent(tmp_path: Path) -> None:
    config_path = _write_tmp_config(tmp_path)
    for _ in range(2):
        proc = subprocess.run(
            [sys.executable, "-m", "scripts.init_metadata_db",
             "--config", str(config_path), "--quiet"],
            cwd=REPO_ROOT,
            capture_output=True, text=True, timeout=20,
        )
        assert proc.returncode == 0, proc.stderr
