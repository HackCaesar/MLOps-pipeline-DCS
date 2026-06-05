"""Unit tests for scripts.register_artifacts."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from packages.common.metadata_store import SQLiteMetadataStore
from scripts.register_artifacts import register_artifacts


def _seed_run_row(db_path: Path, run_id: str) -> None:
    with SQLiteMetadataStore(db_path) as store:
        store.init_db()
        store.create_pipeline_run(
            run_id=run_id, dataset_id=None,
            started_at="2026-05-26T00:00:00Z", status="running",
            config_path="cfg.yaml", config_snapshot_path="snap.yaml",
        )


def _make_artifact_tree(artifacts_dir: Path, run_id: str) -> None:
    (artifacts_dir / "checkpoints" / run_id).mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "checkpoints" / run_id / "best_ckpt.pth").write_bytes(b"best")
    (artifacts_dir / "checkpoints" / run_id / "latest_ckpt.pth").write_bytes(b"latest")
    (artifacts_dir / "checkpoints" / run_id / "epoch_020_ckpt.pth").write_bytes(b"epoch20")

    rdir = artifacts_dir / "reports" / run_id / "metrics"
    rdir.mkdir(parents=True, exist_ok=True)
    summary = {"metrics": {"map50": 0.812, "map5095": 0.741}}
    (rdir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (rdir / "summary.csv").write_text("key,value\nmap50,0.812\n", encoding="utf-8")
    (rdir / "confusion_matrix.png").write_bytes(b"\x89PNG\r\n\x1a\nfake")
    (rdir / "false_positives.jsonl").write_text("{}\n", encoding="utf-8")

    (artifacts_dir / "exports" / run_id).mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "exports" / run_id / "model.onnx").write_bytes(b"onnx")
    (artifacts_dir / "exports" / run_id / "model_fp16.engine").write_bytes(b"trt")


@pytest.fixture
def setup(tmp_path: Path) -> tuple[Path, Path, str]:
    """Returns (artifacts_dir, db_path, run_id)."""
    run_id = "run_test"
    artifacts_dir = tmp_path / "artifacts"
    db_path = tmp_path / "pipeline.db"
    _make_artifact_tree(artifacts_dir, run_id)
    _seed_run_row(db_path, run_id)
    return artifacts_dir, db_path, run_id


# ---- happy path -------------------------------------------------------

def test_registers_all_checkpoints(setup) -> None:
    artifacts_dir, db_path, run_id = setup
    result = register_artifacts(run_id=run_id, artifacts_dir=artifacts_dir,
                                  metadata_db_path=db_path)
    assert result.checkpoints == 3
    with SQLiteMetadataStore(db_path) as store:
        cks = store.get_checkpoints(run_id)
    types = sorted(c.checkpoint_type for c in cks)
    assert types == sorted(["best", "every_n", "latest"])
    by_type = {c.checkpoint_type: c for c in cks}
    assert by_type["every_n"].epoch == 20


def test_registers_evaluation_report_with_primary_metric(setup) -> None:
    artifacts_dir, db_path, run_id = setup
    register_artifacts(run_id=run_id, artifacts_dir=artifacts_dir,
                        metadata_db_path=db_path)
    with SQLiteMetadataStore(db_path) as store:
        reports = store.get_evaluation_reports(run_id)
    assert len(reports) == 1
    assert reports[0].primary_metric_name == "mAP@50"
    assert reports[0].primary_metric_value == pytest.approx(0.812)


def test_registers_each_report_file_as_artifact(setup) -> None:
    artifacts_dir, db_path, run_id = setup
    register_artifacts(run_id=run_id, artifacts_dir=artifacts_dir,
                        metadata_db_path=db_path)
    with SQLiteMetadataStore(db_path) as store:
        arts = store.get_artifacts(run_id)
    types = sorted({a.artifact_type for a in arts})
    assert "report_file" in types
    assert "export" in types


def test_assigns_metric_value_to_checkpoint_when_summary_present(setup) -> None:
    artifacts_dir, db_path, run_id = setup
    register_artifacts(run_id=run_id, artifacts_dir=artifacts_dir,
                        metadata_db_path=db_path)
    with SQLiteMetadataStore(db_path) as store:
        cks = store.get_checkpoints(run_id)
    metric_values = {c.checkpoint_type: c.metric_value for c in cks}
    # All checkpoints get the same map5095 from the shared summary file.
    assert metric_values["best"] == pytest.approx(0.741)


# ---- idempotency ------------------------------------------------------

def test_register_twice_without_replace_skips_duplicates(setup) -> None:
    """Re-running without --replace must not crash. Duplicate inserts are caught
    inside ``_register_*`` helpers (warning + ``skipped`` counter) so the script
    stays idempotent-ish without forcing the caller to clean up first."""
    artifacts_dir, db_path, run_id = setup
    register_artifacts(run_id=run_id, artifacts_dir=artifacts_dir,
                        metadata_db_path=db_path)
    second = register_artifacts(run_id=run_id, artifacts_dir=artifacts_dir,
                                  metadata_db_path=db_path)
    # All rows already existed → every register attempt should have been skipped.
    assert second.skipped >= 3, f"expected ≥3 skipped, got {second.skipped}"
    assert second.checkpoints == 0
    # Counts in SQLite stayed exactly at first-run level.
    with SQLiteMetadataStore(db_path) as store:
        assert len(store.get_checkpoints(run_id)) == 3


def test_replace_allows_re_registration(setup) -> None:
    artifacts_dir, db_path, run_id = setup
    register_artifacts(run_id=run_id, artifacts_dir=artifacts_dir,
                        metadata_db_path=db_path)
    result = register_artifacts(run_id=run_id, artifacts_dir=artifacts_dir,
                                  metadata_db_path=db_path, replace=True)
    assert result.checkpoints == 3
    with SQLiteMetadataStore(db_path) as store:
        assert len(store.get_checkpoints(run_id)) == 3


# ---- edge cases -------------------------------------------------------

def test_missing_artifacts_dir_is_no_op(tmp_path: Path) -> None:
    run_id = "ghost"
    db_path = tmp_path / "pipeline.db"
    _seed_run_row(db_path, run_id)
    result = register_artifacts(
        run_id=run_id,
        artifacts_dir=tmp_path / "nope",
        metadata_db_path=db_path,
    )
    assert result.checkpoints == 0
    assert result.reports == 0


def test_no_summary_json_no_metric(tmp_path: Path) -> None:
    run_id = "no_summary"
    db_path = tmp_path / "pipeline.db"
    artifacts_dir = tmp_path / "artifacts"
    (artifacts_dir / "checkpoints" / run_id).mkdir(parents=True)
    (artifacts_dir / "checkpoints" / run_id / "best_ckpt.pth").write_bytes(b"x")
    _seed_run_row(db_path, run_id)
    register_artifacts(run_id=run_id, artifacts_dir=artifacts_dir,
                        metadata_db_path=db_path)
    with SQLiteMetadataStore(db_path) as store:
        cks = store.get_checkpoints(run_id)
    assert cks[0].metric_value is None


def test_export_files_under_exports_dir(setup) -> None:
    artifacts_dir, db_path, run_id = setup
    register_artifacts(run_id=run_id, artifacts_dir=artifacts_dir,
                        metadata_db_path=db_path)
    with SQLiteMetadataStore(db_path) as store:
        arts = store.get_artifacts(run_id)
    exports = [a for a in arts if a.artifact_type == "export"]
    names = sorted(Path(a.storage_path).name for a in exports)
    assert names == ["model.onnx", "model_fp16.engine"]


def test_content_hash_is_recorded_for_exports(setup) -> None:
    artifacts_dir, db_path, run_id = setup
    register_artifacts(run_id=run_id, artifacts_dir=artifacts_dir,
                        metadata_db_path=db_path)
    with SQLiteMetadataStore(db_path) as store:
        arts = [a for a in store.get_artifacts(run_id) if a.artifact_type == "export"]
    for a in arts:
        assert a.content_hash and len(a.content_hash) == 64
