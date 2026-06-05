"""Unit tests for packages.common.metadata_store.SQLiteMetadataStore."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from packages.common.metadata_store import (
    TABLE_NAMES,
    SQLiteMetadataStore,
)


@pytest.fixture
def store(tmp_path: Path) -> SQLiteMetadataStore:
    s = SQLiteMetadataStore(tmp_path / "pipeline.db")
    s.init_db()
    yield s
    s.close()


# ---- schema -----------------------------------------------------------

def test_init_db_creates_all_tables(store: SQLiteMetadataStore) -> None:
    present = set(store.list_tables())
    for name in TABLE_NAMES:
        assert name in present, f"missing table: {name}"


def test_init_db_is_idempotent(tmp_path: Path) -> None:
    s = SQLiteMetadataStore(tmp_path / "db.sqlite")
    s.init_db()
    s.init_db()  # second call must not raise
    assert set(s.list_tables()) >= set(TABLE_NAMES)
    s.close()


def test_db_path_parent_is_created(tmp_path: Path) -> None:
    nested = tmp_path / "a" / "b" / "c" / "pipeline.db"
    SQLiteMetadataStore(nested).close()
    assert nested.parent.is_dir()


# ---- dataset roundtrip ------------------------------------------------

def _register_min_dataset(store: SQLiteMetadataStore, dataset_id: str = "dcs_1") -> None:
    store.register_dataset(
        dataset_id=dataset_id, name="Smoke",
        created_at="2026-05-24T00:00:00Z", source="test",
        storage_path=f"datasets/raw/{dataset_id}",
        annotation_format="coco",
        num_images=1, num_annotations=2, num_classes=4,
    )


def test_register_dataset_roundtrip(store: SQLiteMetadataStore) -> None:
    _register_min_dataset(store)
    got = store.get_dataset("dcs_1")
    assert got is not None
    assert got.dataset_id == "dcs_1"
    assert got.num_classes == 4
    assert got.status == "registered"


def test_get_dataset_returns_none_when_missing(store: SQLiteMetadataStore) -> None:
    assert store.get_dataset("nope") is None


def test_register_duplicate_dataset_upserts(store: SQLiteMetadataStore) -> None:
    # register_dataset is idempotent: re-registering the same id updates the row in
    # place (ON CONFLICT DO UPDATE) instead of raising. This is required so a
    # pipeline re-run can re-register a dataset that already has tile_cache / run FK
    # children. The "no overwrite without --replace" policy lives in
    # scripts/register_dataset.py (a get_dataset() check), not in the store.
    _register_min_dataset(store)
    _register_min_dataset(store)            # must NOT raise
    assert len(store.list_datasets()) == 1


def test_list_datasets(store: SQLiteMetadataStore) -> None:
    _register_min_dataset(store, "a")
    _register_min_dataset(store, "b")
    assert sorted(d.dataset_id for d in store.list_datasets()) == ["a", "b"]


# ---- images + annotations + FK ----------------------------------------

def test_register_image_and_query_by_split(store: SQLiteMetadataStore) -> None:
    _register_min_dataset(store)
    store.register_image(
        image_id="dcs_1:img1", dataset_id="dcs_1", split="train",
        file_name="img1.png", width=2560, height=1440,
        storage_path="images/train/img1.png", has_objects=True,
        source_frame_id="frame_001",
    )
    store.register_image(
        image_id="dcs_1:img2", dataset_id="dcs_1", split="val",
        file_name="img2.png", width=2560, height=1440,
        storage_path="images/val/img2.png", has_objects=False,
    )
    train = store.get_images_by_split("dcs_1", "train")
    val = store.get_images_by_split("dcs_1", "val")
    assert [i.image_id for i in train] == ["dcs_1:img1"]
    assert [i.image_id for i in val] == ["dcs_1:img2"]
    assert train[0].has_objects is True
    assert val[0].has_objects is False


def test_register_image_fk_to_missing_dataset_raises(store: SQLiteMetadataStore) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        store.register_image(
            image_id="x", dataset_id="ghost", split="train",
            file_name="x.png", width=10, height=10,
            storage_path="x.png", has_objects=False,
        )


def test_register_annotation_and_lookup(store: SQLiteMetadataStore) -> None:
    _register_min_dataset(store)
    store.register_image(
        image_id="dcs_1:img1", dataset_id="dcs_1", split="train",
        file_name="img1.png", width=2560, height=1440,
        storage_path="images/train/img1.png", has_objects=True,
    )
    store.register_annotation(
        annotation_id="dcs_1:ann1", dataset_id="dcs_1",
        image_id="dcs_1:img1", category_id=2,
        bbox_xywh=(10.0, 20.0, 30.5, 40.5), area=1236.75,
    )
    anns = store.get_annotations_by_image("dcs_1:img1")
    assert len(anns) == 1
    assert anns[0].bbox_xywh == (10.0, 20.0, 30.5, 40.5)
    assert anns[0].category_id == 2


def test_register_annotation_bad_bbox_raises(store: SQLiteMetadataStore) -> None:
    _register_min_dataset(store)
    store.register_image(
        image_id="i", dataset_id="dcs_1", split="train",
        file_name="x.png", width=1, height=1,
        storage_path="x", has_objects=True,
    )
    with pytest.raises(ValueError, match="4 floats"):
        store.register_annotation(
            annotation_id="a", dataset_id="dcs_1", image_id="i",
            category_id=0, bbox_xywh=(1.0, 2.0, 3.0), area=0.0,
        )


# ---- pipeline_runs + status update ------------------------------------

def test_create_and_update_pipeline_run(store: SQLiteMetadataStore) -> None:
    _register_min_dataset(store)
    store.create_pipeline_run(
        run_id="2026_05_24_001", dataset_id="dcs_1",
        started_at="2026-05-24T00:00:00Z", status="running",
        config_path="configs/pipeline.yaml",
        config_snapshot_path="storage/runs/2026_05_24_001/config_snapshot.yaml",
    )
    got = store.get_run("2026_05_24_001")
    assert got is not None and got.status == "running"

    store.update_pipeline_run_status(
        "2026_05_24_001", status="success",
        finished_at="2026-05-24T01:00:00Z",
        mlflow_run_id="mlf_abc",
    )
    got2 = store.get_run("2026_05_24_001")
    assert got2.status == "success"
    assert got2.finished_at == "2026-05-24T01:00:00Z"
    assert got2.mlflow_run_id == "mlf_abc"
    assert got2.error is None


def test_update_status_unknown_run_raises(store: SQLiteMetadataStore) -> None:
    with pytest.raises(KeyError, match="No pipeline_run"):
        store.update_pipeline_run_status("ghost", status="failed")


def test_pipeline_run_can_have_null_dataset_id(store: SQLiteMetadataStore) -> None:
    store.create_pipeline_run(
        run_id="r", dataset_id=None,
        started_at="2026-05-24T00:00:00Z", status="running",
        config_path="c", config_snapshot_path="cs",
    )
    assert store.get_run("r").dataset_id is None


def test_list_runs(store: SQLiteMetadataStore) -> None:
    _register_min_dataset(store)
    for rid in ("r1", "r2"):
        store.create_pipeline_run(
            run_id=rid, dataset_id="dcs_1",
            started_at="2026-05-24T00:00:00Z", status="running",
            config_path="c", config_snapshot_path="cs",
        )
    assert sorted(r.run_id for r in store.list_runs()) == ["r1", "r2"]


# ---- artifacts / checkpoints / reports --------------------------------

def _make_run(store: SQLiteMetadataStore, run_id: str = "r1") -> None:
    _register_min_dataset(store)
    store.create_pipeline_run(
        run_id=run_id, dataset_id="dcs_1",
        started_at="2026-05-24T00:00:00Z", status="running",
        config_path="c", config_snapshot_path="cs",
    )


def test_register_artifact_and_query(store: SQLiteMetadataStore) -> None:
    _make_run(store)
    store.register_artifact(
        artifact_id="a1", run_id="r1", artifact_type="manifest",
        storage_path="runs/r1/manifests/manifest.jsonl",
        created_at="2026-05-24T00:30:00Z", content_hash="abc",
    )
    arts = store.get_artifacts("r1")
    assert len(arts) == 1 and arts[0].artifact_type == "manifest"


def test_register_checkpoint_and_query(store: SQLiteMetadataStore) -> None:
    _make_run(store)
    store.register_checkpoint(
        checkpoint_id="ck_best", run_id="r1", checkpoint_type="best",
        storage_path="artifacts/checkpoints/r1/best_ckpt.pth",
        created_at="2026-05-24T01:00:00Z",
        epoch=42, metric_name="ap50_95", metric_value=0.781,
    )
    cks = store.get_checkpoints("r1")
    assert cks[0].epoch == 42 and cks[0].metric_value == pytest.approx(0.781)


def test_register_evaluation_report_and_query(store: SQLiteMetadataStore) -> None:
    _make_run(store)
    store.register_checkpoint(
        checkpoint_id="ck_best", run_id="r1", checkpoint_type="best",
        storage_path="best.pth", created_at="2026-05-24T01:00:00Z",
    )
    store.register_evaluation_report(
        report_id="rep1", run_id="r1", checkpoint_id="ck_best",
        metrics_path="reports/r1/metrics.json",
        report_path="reports/r1/",
        created_at="2026-05-24T01:30:00Z",
        primary_metric_name="ap50", primary_metric_value=0.812,
    )
    reps = store.get_evaluation_reports("r1")
    assert len(reps) == 1
    assert reps[0].checkpoint_id == "ck_best"
    assert reps[0].primary_metric_value == pytest.approx(0.812)


def test_artifact_fk_to_missing_run_raises(store: SQLiteMetadataStore) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        store.register_artifact(
            artifact_id="x", run_id="ghost", artifact_type="manifest",
            storage_path="x", created_at="2026-05-24T00:00:00Z",
        )


def test_context_manager_closes(tmp_path: Path) -> None:
    db = tmp_path / "ctx.db"
    with SQLiteMetadataStore(db) as s:
        s.init_db()
        _register_min_dataset(s)
    # After exit: connection closed, opening fresh handle works.
    s2 = SQLiteMetadataStore(db)
    assert s2.get_dataset("dcs_1") is not None
    s2.close()
