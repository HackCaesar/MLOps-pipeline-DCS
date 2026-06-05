"""tile_caches table + idempotent pipeline_runs migration (needs sqlite3 → Docker/CI)."""
from __future__ import annotations

from pathlib import Path

from packages.common.metadata_store import TABLE_NAMES, SQLiteMetadataStore


def test_tile_caches_in_table_names() -> None:
    assert "tile_caches" in TABLE_NAMES


def test_init_db_idempotent_adds_columns_once(tmp_path: Path) -> None:
    db = tmp_path / "m.db"
    with SQLiteMetadataStore.open(db) as s:
        s.init_db()
        s.init_db()  # re-run must be a safe no-op (PRAGMA-guarded ALTER)
        cols = {r["name"] for r in s._conn.execute("PRAGMA table_info(pipeline_runs)")}
        tables = set(s.list_tables())
    assert {"current_stage", "model_name", "tile_cache_id"} <= cols
    assert "tile_caches" in tables


def test_migration_tolerates_legacy_pipeline_runs(tmp_path: Path) -> None:
    """A pre-existing pipeline_runs WITHOUT the new columns must migrate cleanly."""
    db = tmp_path / "legacy.db"
    with SQLiteMetadataStore.open(db) as s:
        s._conn.execute(
            """CREATE TABLE pipeline_runs (run_id TEXT PRIMARY KEY, dataset_id TEXT,
               started_at TEXT NOT NULL, finished_at TEXT, status TEXT NOT NULL,
               config_path TEXT NOT NULL, config_snapshot_path TEXT NOT NULL,
               mlflow_run_id TEXT, error TEXT)""")
        s.init_db()  # should ADD the 3 columns, not error
        cols = {r["name"] for r in s._conn.execute("PRAGMA table_info(pipeline_runs)")}
    assert {"current_stage", "model_name", "tile_cache_id"} <= cols


def test_register_tile_cache_roundtrip_idempotent(tmp_path: Path) -> None:
    db = tmp_path / "m.db"
    with SQLiteMetadataStore.open(db) as s:
        s.init_db()
        # tile_caches.dataset_id has a FK to datasets — register the parent first
        # (the real flow registers the dataset before build_or_reuse_tile_cache).
        s.register_dataset(
            dataset_id="ds", name="ds", created_at="2026-05-31T00:00:00Z",
            source="test", storage_path="/raw/ds", annotation_format="coco",
            num_images=1, num_annotations=1, num_classes=3)
        s.register_tile_cache(
            tile_cache_id="a__b__c", dataset_id="ds", dataset_hash="x",
            split_hash="y", tile_config_hash="z", storage_path="/cache/a__b__c",
            created_at="2026-05-31T00:00:00Z", status="created",
            num_train_tiles=5, num_val_tiles=2, num_test_tiles=1)
        # Re-register same id (content-addressed reuse) must NOT raise.
        s.register_tile_cache(
            tile_cache_id="a__b__c", dataset_id="ds", dataset_hash="x",
            split_hash="y", tile_config_hash="z", storage_path="/cache/a__b__c",
            created_at="2026-05-31T01:00:00Z", status="reused",
            num_train_tiles=5, num_val_tiles=2, num_test_tiles=1)
        row = s.get_tile_cache("a__b__c")
    assert row is not None
    assert row.status == "reused" and row.num_train_tiles == 5


def test_register_dataset_upsert_preserves_fk_children(tmp_path: Path) -> None:
    """Re-registering a dataset that already has a tile_cache FK child must upsert
    the parent in place — NOT raise FOREIGN KEY constraint failed.

    Regression: scripts.register_dataset --replace used to DELETE the datasets row,
    which tile_caches / pipeline_runs reference (FK, no CASCADE), so any re-run on a
    populated DB failed. The store now does ON CONFLICT(dataset_id) DO UPDATE.
    """
    db = tmp_path / "m.db"
    with SQLiteMetadataStore.open(db) as s:
        s.init_db()
        s.register_dataset(
            dataset_id="ds", name="ds", created_at="2026-06-04T00:00:00Z",
            source="test", storage_path="/raw/ds", annotation_format="coco",
            num_images=1, num_annotations=1, num_classes=3)
        s.register_tile_cache(
            tile_cache_id="a__b__c", dataset_id="ds", dataset_hash="x",
            split_hash="y", tile_config_hash="z", storage_path="/cache/a__b__c",
            created_at="2026-06-04T00:00:00Z", status="created",
            num_train_tiles=5, num_val_tiles=2, num_test_tiles=1)
        # Upsert the parent while a tile_cache references it — must NOT raise.
        s.register_dataset(
            dataset_id="ds", name="ds-renamed", created_at="2026-06-04T00:00:00Z",
            source="test", storage_path="/raw/ds", annotation_format="coco",
            num_images=42, num_annotations=7, num_classes=3)
        ds = s.get_dataset("ds")
        cache = s.get_tile_cache("a__b__c")
    assert ds is not None and ds.num_images == 42 and ds.name == "ds-renamed"
    assert cache is not None      # FK child survived the re-register


def test_update_pipeline_run_new_columns(tmp_path: Path) -> None:
    db = tmp_path / "m.db"
    with SQLiteMetadataStore.open(db) as s:
        s.init_db()
        s.create_pipeline_run(run_id="r1", dataset_id=None,
                              started_at="2026-05-31T00:00:00Z", status="running",
                              config_path="c.yaml", config_snapshot_path="snap.yaml")
        s.update_pipeline_run_status("r1", current_stage="train")           # no status kwarg
        s.update_pipeline_run_status("r1", tile_cache_id="a__b__c", model_name="YOLOX-S")
        run = s.get_run("r1")
    assert run.current_stage == "train"
    assert run.tile_cache_id == "a__b__c" and run.model_name == "YOLOX-S"
