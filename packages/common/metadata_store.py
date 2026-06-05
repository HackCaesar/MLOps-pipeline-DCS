"""SQLite-backed metadata index.

Read the DDL carefully before changing it — pipeline_runs, artifacts,
model_checkpoints and evaluation_reports have explicit foreign keys back to
datasets/runs and the rest of the pipeline assumes them.
"""
from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Protocol, Sequence

# ---------- table schema ----------------------------------------------------

_SCHEMA: list[str] = [
    """CREATE TABLE IF NOT EXISTS datasets (
        dataset_id        TEXT PRIMARY KEY,
        name              TEXT NOT NULL,
        created_at        TEXT NOT NULL,
        source            TEXT NOT NULL,
        storage_path      TEXT NOT NULL,
        annotation_format TEXT NOT NULL,
        num_images        INTEGER NOT NULL,
        num_annotations   INTEGER NOT NULL,
        num_classes       INTEGER NOT NULL,
        status            TEXT NOT NULL,
        content_hash      TEXT,
        notes             TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS images (
        image_id        TEXT PRIMARY KEY,
        dataset_id      TEXT NOT NULL,
        split           TEXT NOT NULL,
        file_name       TEXT NOT NULL,
        width           INTEGER NOT NULL,
        height          INTEGER NOT NULL,
        storage_path    TEXT NOT NULL,
        has_objects     INTEGER NOT NULL DEFAULT 0,
        source_frame_id TEXT,
        FOREIGN KEY(dataset_id) REFERENCES datasets(dataset_id)
    )""",
    """CREATE TABLE IF NOT EXISTS annotations (
        annotation_id        TEXT PRIMARY KEY,
        dataset_id           TEXT NOT NULL,
        image_id             TEXT NOT NULL,
        category_id          INTEGER NOT NULL,
        bbox_xywh            TEXT NOT NULL,
        area                 REAL NOT NULL,
        iscrowd              INTEGER NOT NULL DEFAULT 0,
        source_annotation_id TEXT,
        FOREIGN KEY(dataset_id) REFERENCES datasets(dataset_id),
        FOREIGN KEY(image_id)  REFERENCES images(image_id)
    )""",
    """CREATE TABLE IF NOT EXISTS pipeline_runs (
        run_id               TEXT PRIMARY KEY,
        dataset_id           TEXT,
        started_at           TEXT NOT NULL,
        finished_at          TEXT,
        status               TEXT NOT NULL,
        config_path          TEXT NOT NULL,
        config_snapshot_path TEXT NOT NULL,
        mlflow_run_id        TEXT,
        error                TEXT,
        FOREIGN KEY(dataset_id) REFERENCES datasets(dataset_id)
    )""",
    """CREATE TABLE IF NOT EXISTS tile_caches (
        tile_cache_id        TEXT PRIMARY KEY,
        dataset_id           TEXT NOT NULL,
        dataset_hash         TEXT NOT NULL,
        split_hash           TEXT NOT NULL,
        tile_config_hash     TEXT NOT NULL,
        storage_path         TEXT NOT NULL,
        num_train_tiles      INTEGER NOT NULL DEFAULT 0,
        num_val_tiles        INTEGER NOT NULL DEFAULT 0,
        num_test_tiles       INTEGER NOT NULL DEFAULT 0,
        created_at           TEXT NOT NULL,
        status               TEXT NOT NULL,
        config_snapshot_path TEXT,
        FOREIGN KEY(dataset_id) REFERENCES datasets(dataset_id)
    )""",
    """CREATE TABLE IF NOT EXISTS artifacts (
        artifact_id   TEXT PRIMARY KEY,
        run_id        TEXT NOT NULL,
        artifact_type TEXT NOT NULL,
        storage_path  TEXT NOT NULL,
        created_at    TEXT NOT NULL,
        content_hash  TEXT,
        FOREIGN KEY(run_id) REFERENCES pipeline_runs(run_id)
    )""",
    """CREATE TABLE IF NOT EXISTS model_checkpoints (
        checkpoint_id   TEXT PRIMARY KEY,
        run_id          TEXT NOT NULL,
        checkpoint_type TEXT NOT NULL,
        epoch           INTEGER,
        metric_name     TEXT,
        metric_value    REAL,
        storage_path    TEXT NOT NULL,
        created_at      TEXT NOT NULL,
        FOREIGN KEY(run_id) REFERENCES pipeline_runs(run_id)
    )""",
    """CREATE TABLE IF NOT EXISTS evaluation_reports (
        report_id            TEXT PRIMARY KEY,
        run_id               TEXT NOT NULL,
        checkpoint_id        TEXT,
        metrics_path         TEXT NOT NULL,
        report_path          TEXT NOT NULL,
        created_at           TEXT NOT NULL,
        primary_metric_name  TEXT,
        primary_metric_value REAL,
        FOREIGN KEY(run_id) REFERENCES pipeline_runs(run_id),
        FOREIGN KEY(checkpoint_id) REFERENCES model_checkpoints(checkpoint_id)
    )""",
]

_INDEXES: list[str] = [
    "CREATE INDEX IF NOT EXISTS idx_images_dataset_split ON images(dataset_id, split)",
    "CREATE INDEX IF NOT EXISTS idx_annotations_image    ON annotations(image_id)",
    "CREATE INDEX IF NOT EXISTS idx_annotations_dataset  ON annotations(dataset_id)",
    "CREATE INDEX IF NOT EXISTS idx_artifacts_run        ON artifacts(run_id)",
    "CREATE INDEX IF NOT EXISTS idx_checkpoints_run      ON model_checkpoints(run_id)",
    "CREATE INDEX IF NOT EXISTS idx_reports_run          ON evaluation_reports(run_id)",
    "CREATE INDEX IF NOT EXISTS idx_tile_caches_dataset  ON tile_caches(dataset_id)",
]

TABLE_NAMES: tuple[str, ...] = (
    "datasets", "images", "annotations", "pipeline_runs",
    "tile_caches", "artifacts", "model_checkpoints", "evaluation_reports",
)


# ---------- row dataclasses -------------------------------------------------

@dataclass(frozen=True)
class DatasetRow:
    dataset_id: str
    name: str
    created_at: str
    source: str
    storage_path: str
    annotation_format: str
    num_images: int
    num_annotations: int
    num_classes: int
    status: str
    content_hash: Optional[str]
    notes: Optional[str]


@dataclass(frozen=True)
class ImageRow:
    image_id: str
    dataset_id: str
    split: str
    file_name: str
    width: int
    height: int
    storage_path: str
    has_objects: bool
    source_frame_id: Optional[str]


@dataclass(frozen=True)
class AnnotationRow:
    annotation_id: str
    dataset_id: str
    image_id: str
    category_id: int
    bbox_xywh: tuple[float, float, float, float]
    area: float
    iscrowd: bool
    source_annotation_id: Optional[str]


@dataclass(frozen=True)
class PipelineRunRow:
    run_id: str
    dataset_id: Optional[str]
    started_at: str
    finished_at: Optional[str]
    status: str
    config_path: str
    config_snapshot_path: str
    mlflow_run_id: Optional[str]
    error: Optional[str]
    current_stage: Optional[str] = None
    model_name: Optional[str] = None
    tile_cache_id: Optional[str] = None


@dataclass(frozen=True)
class TileCacheRow:
    tile_cache_id: str
    dataset_id: str
    dataset_hash: str
    split_hash: str
    tile_config_hash: str
    storage_path: str
    num_train_tiles: int
    num_val_tiles: int
    num_test_tiles: int
    created_at: str
    status: str
    config_snapshot_path: Optional[str]


@dataclass(frozen=True)
class ArtifactRow:
    artifact_id: str
    run_id: str
    artifact_type: str
    storage_path: str
    created_at: str
    content_hash: Optional[str]


@dataclass(frozen=True)
class CheckpointRow:
    checkpoint_id: str
    run_id: str
    checkpoint_type: str
    epoch: Optional[int]
    metric_name: Optional[str]
    metric_value: Optional[float]
    storage_path: str
    created_at: str


@dataclass(frozen=True)
class EvaluationReportRow:
    report_id: str
    run_id: str
    checkpoint_id: Optional[str]
    metrics_path: str
    report_path: str
    created_at: str
    primary_metric_name: Optional[str]
    primary_metric_value: Optional[float]


# ---------- store -----------------------------------------------------------

class MetadataStore(Protocol):
    def init_db(self) -> None: ...
    def close(self) -> None: ...

    def register_dataset(self, **fields: Any) -> None: ...
    def register_image(self, **fields: Any) -> None: ...
    def register_annotation(self, **fields: Any) -> None: ...
    def create_pipeline_run(self, **fields: Any) -> None: ...
    def update_pipeline_run_status(self, run_id: str, **fields: Any) -> None: ...
    def register_artifact(self, **fields: Any) -> None: ...
    def register_checkpoint(self, **fields: Any) -> None: ...
    def register_evaluation_report(self, **fields: Any) -> None: ...

    def get_dataset(self, dataset_id: str) -> Optional[DatasetRow]: ...
    def list_datasets(self) -> list[DatasetRow]: ...
    def get_images_by_split(self, dataset_id: str, split: str) -> list[ImageRow]: ...
    def get_run(self, run_id: str) -> Optional[PipelineRunRow]: ...
    def list_runs(self) -> list[PipelineRunRow]: ...


class SQLiteMetadataStore:
    """SQLite-backed implementation. Single connection per instance.

    Use ``SQLiteMetadataStore.open(path)`` or construct directly. Call
    ``close()`` when done; or use as a context manager.
    """

    def __init__(self, db_path: str | os.PathLike[str]) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        # Storage often lives on a drvfs / bind-mounted volume, and the status
        # tracker mirrors columns from a SECOND connection during a run. Wait on
        # locks (up to 30s) instead of failing fast with "database is locked"
        # (the likely cause of transient register_dataset failures on Windows).
        self._conn.execute("PRAGMA busy_timeout = 30000")

    # ---- lifecycle --------------------------------------------------

    @classmethod
    def open(cls, db_path: str | os.PathLike[str]) -> "SQLiteMetadataStore":
        return cls(db_path)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "SQLiteMetadataStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ---- schema -----------------------------------------------------

    def init_db(self) -> None:
        with self._conn:
            for ddl in _SCHEMA:
                self._conn.execute(ddl)
            for idx in _INDEXES:
                self._conn.execute(idx)
            self._migrate_pipeline_runs()

    def _migrate_pipeline_runs(self) -> None:
        """Idempotently add Stage-4 columns to an existing pipeline_runs table.

        SQLite has no ``ADD COLUMN IF NOT EXISTS``; ``PRAGMA table_info`` is the
        canonical existence check, so re-running ``init_db`` is a safe no-op.
        """
        existing = {r["name"] for r in self._conn.execute("PRAGMA table_info(pipeline_runs)")}
        for col, decl in (("current_stage", "TEXT"),
                          ("model_name", "TEXT"),
                          ("tile_cache_id", "TEXT")):
            if col not in existing:
                self._conn.execute(f"ALTER TABLE pipeline_runs ADD COLUMN {col} {decl}")

    def list_tables(self) -> list[str]:
        cur = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        return [r["name"] for r in cur.fetchall()]

    # ---- registrations ----------------------------------------------

    def register_dataset(
        self,
        *,
        dataset_id: str,
        name: str,
        created_at: str,
        source: str,
        storage_path: str,
        annotation_format: str,
        num_images: int,
        num_annotations: int,
        num_classes: int,
        status: str = "registered",
        content_hash: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> None:
        with self._conn:
            self._conn.execute(
                """INSERT INTO datasets
                   (dataset_id, name, created_at, source, storage_path,
                    annotation_format, num_images, num_annotations, num_classes,
                    status, content_hash, notes)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(dataset_id) DO UPDATE SET
                     name=excluded.name, created_at=excluded.created_at,
                     source=excluded.source, storage_path=excluded.storage_path,
                     annotation_format=excluded.annotation_format,
                     num_images=excluded.num_images,
                     num_annotations=excluded.num_annotations,
                     num_classes=excluded.num_classes, status=excluded.status,
                     content_hash=excluded.content_hash, notes=excluded.notes""",
                (dataset_id, name, created_at, source, storage_path,
                 annotation_format, int(num_images), int(num_annotations),
                 int(num_classes), status, content_hash, notes),
            )

    def register_image(
        self,
        *,
        image_id: str,
        dataset_id: str,
        split: str,
        file_name: str,
        width: int,
        height: int,
        storage_path: str,
        has_objects: bool,
        source_frame_id: Optional[str] = None,
    ) -> None:
        with self._conn:
            self._conn.execute(
                """INSERT INTO images
                   (image_id, dataset_id, split, file_name, width, height,
                    storage_path, has_objects, source_frame_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (image_id, dataset_id, split, file_name, int(width), int(height),
                 storage_path, int(bool(has_objects)), source_frame_id),
            )

    def register_annotation(
        self,
        *,
        annotation_id: str,
        dataset_id: str,
        image_id: str,
        category_id: int,
        bbox_xywh: Sequence[float],
        area: float,
        iscrowd: bool = False,
        source_annotation_id: Optional[str] = None,
    ) -> None:
        if len(bbox_xywh) != 4:
            raise ValueError(f"bbox_xywh must have 4 floats, got {len(bbox_xywh)}")
        with self._conn:
            self._conn.execute(
                """INSERT INTO annotations
                   (annotation_id, dataset_id, image_id, category_id, bbox_xywh,
                    area, iscrowd, source_annotation_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (annotation_id, dataset_id, image_id, int(category_id),
                 _serialize_bbox(bbox_xywh), float(area), int(bool(iscrowd)),
                 source_annotation_id),
            )

    def create_pipeline_run(
        self,
        *,
        run_id: str,
        dataset_id: Optional[str],
        started_at: str,
        status: str,
        config_path: str,
        config_snapshot_path: str,
        mlflow_run_id: Optional[str] = None,
    ) -> None:
        with self._conn:
            self._conn.execute(
                """INSERT INTO pipeline_runs
                   (run_id, dataset_id, started_at, finished_at, status,
                    config_path, config_snapshot_path, mlflow_run_id, error)
                   VALUES (?, ?, ?, NULL, ?, ?, ?, ?, NULL)""",
                (run_id, dataset_id, started_at, status,
                 config_path, config_snapshot_path, mlflow_run_id),
            )

    def update_pipeline_run_status(
        self,
        run_id: str,
        *,
        status: Optional[str] = None,
        finished_at: Optional[str] = None,
        error: Optional[str] = None,
        mlflow_run_id: Optional[str] = None,
        current_stage: Optional[str] = None,
        model_name: Optional[str] = None,
        tile_cache_id: Optional[str] = None,
    ) -> None:
        # Build dynamic update so we don't clobber unspecified fields.
        sets: list[str] = []
        params: list[Any] = []
        for col, val in (("status", status), ("finished_at", finished_at),
                         ("error", error), ("mlflow_run_id", mlflow_run_id),
                         ("current_stage", current_stage), ("model_name", model_name),
                         ("tile_cache_id", tile_cache_id)):
            if val is not None:
                sets.append(f"{col} = ?")
                params.append(val)
        if not sets:
            return
        params.append(run_id)
        with self._conn:
            cur = self._conn.execute(
                f"UPDATE pipeline_runs SET {', '.join(sets)} WHERE run_id = ?",
                params,
            )
            if cur.rowcount == 0:
                raise KeyError(f"No pipeline_run with run_id={run_id!r}")

    def register_tile_cache(self, *, tile_cache_id: str, dataset_id: str,
                            dataset_hash: str, split_hash: str, tile_config_hash: str,
                            storage_path: str, created_at: str, status: str,
                            num_train_tiles: int = 0, num_val_tiles: int = 0,
                            num_test_tiles: int = 0,
                            config_snapshot_path: Optional[str] = None) -> None:
        # Content-addressed + reused across runs → re-registration is idempotent.
        with self._conn:
            self._conn.execute(
                """INSERT OR REPLACE INTO tile_caches
                   (tile_cache_id, dataset_id, dataset_hash, split_hash, tile_config_hash,
                    storage_path, num_train_tiles, num_val_tiles, num_test_tiles,
                    created_at, status, config_snapshot_path)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (tile_cache_id, dataset_id, dataset_hash, split_hash, tile_config_hash,
                 storage_path, num_train_tiles, num_val_tiles, num_test_tiles,
                 created_at, status, config_snapshot_path),
            )

    def get_tile_cache(self, tile_cache_id: str) -> Optional[TileCacheRow]:
        cur = self._conn.execute(
            "SELECT * FROM tile_caches WHERE tile_cache_id = ?", (tile_cache_id,))
        row = cur.fetchone()
        return _row_to_tile_cache(row) if row else None

    def list_tile_caches(self) -> list[TileCacheRow]:
        cur = self._conn.execute("SELECT * FROM tile_caches")
        return [_row_to_tile_cache(r) for r in cur.fetchall()]

    def register_artifact(
        self,
        *,
        artifact_id: str,
        run_id: str,
        artifact_type: str,
        storage_path: str,
        created_at: str,
        content_hash: Optional[str] = None,
    ) -> None:
        with self._conn:
            self._conn.execute(
                """INSERT INTO artifacts
                   (artifact_id, run_id, artifact_type, storage_path,
                    created_at, content_hash)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (artifact_id, run_id, artifact_type, storage_path,
                 created_at, content_hash),
            )

    def register_checkpoint(
        self,
        *,
        checkpoint_id: str,
        run_id: str,
        checkpoint_type: str,
        storage_path: str,
        created_at: str,
        epoch: Optional[int] = None,
        metric_name: Optional[str] = None,
        metric_value: Optional[float] = None,
    ) -> None:
        with self._conn:
            self._conn.execute(
                """INSERT INTO model_checkpoints
                   (checkpoint_id, run_id, checkpoint_type, epoch,
                    metric_name, metric_value, storage_path, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (checkpoint_id, run_id, checkpoint_type,
                 int(epoch) if epoch is not None else None,
                 metric_name,
                 float(metric_value) if metric_value is not None else None,
                 storage_path, created_at),
            )

    def register_evaluation_report(
        self,
        *,
        report_id: str,
        run_id: str,
        metrics_path: str,
        report_path: str,
        created_at: str,
        checkpoint_id: Optional[str] = None,
        primary_metric_name: Optional[str] = None,
        primary_metric_value: Optional[float] = None,
    ) -> None:
        with self._conn:
            self._conn.execute(
                """INSERT INTO evaluation_reports
                   (report_id, run_id, checkpoint_id, metrics_path, report_path,
                    created_at, primary_metric_name, primary_metric_value)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (report_id, run_id, checkpoint_id, metrics_path, report_path,
                 created_at, primary_metric_name,
                 float(primary_metric_value) if primary_metric_value is not None else None),
            )

    # ---- queries ----------------------------------------------------

    def get_dataset(self, dataset_id: str) -> Optional[DatasetRow]:
        cur = self._conn.execute(
            "SELECT * FROM datasets WHERE dataset_id = ?", (dataset_id,)
        )
        row = cur.fetchone()
        return _row_to_dataset(row) if row else None

    def list_datasets(self) -> list[DatasetRow]:
        cur = self._conn.execute("SELECT * FROM datasets ORDER BY created_at, dataset_id")
        return [_row_to_dataset(r) for r in cur.fetchall()]

    def get_images_by_split(self, dataset_id: str, split: str) -> list[ImageRow]:
        cur = self._conn.execute(
            "SELECT * FROM images WHERE dataset_id = ? AND split = ? ORDER BY image_id",
            (dataset_id, split),
        )
        return [_row_to_image(r) for r in cur.fetchall()]

    def get_annotations_by_image(self, image_id: str) -> list[AnnotationRow]:
        cur = self._conn.execute(
            "SELECT * FROM annotations WHERE image_id = ? ORDER BY annotation_id",
            (image_id,),
        )
        return [_row_to_annotation(r) for r in cur.fetchall()]

    def get_run(self, run_id: str) -> Optional[PipelineRunRow]:
        cur = self._conn.execute(
            "SELECT * FROM pipeline_runs WHERE run_id = ?", (run_id,)
        )
        row = cur.fetchone()
        return _row_to_run(row) if row else None

    def list_runs(self) -> list[PipelineRunRow]:
        cur = self._conn.execute(
            "SELECT * FROM pipeline_runs ORDER BY started_at, run_id"
        )
        return [_row_to_run(r) for r in cur.fetchall()]

    def get_artifacts(self, run_id: str) -> list[ArtifactRow]:
        cur = self._conn.execute(
            "SELECT * FROM artifacts WHERE run_id = ? ORDER BY created_at, artifact_id",
            (run_id,),
        )
        return [_row_to_artifact(r) for r in cur.fetchall()]

    def get_checkpoints(self, run_id: str) -> list[CheckpointRow]:
        cur = self._conn.execute(
            "SELECT * FROM model_checkpoints WHERE run_id = ? ORDER BY epoch, checkpoint_id",
            (run_id,),
        )
        return [_row_to_checkpoint(r) for r in cur.fetchall()]

    def get_evaluation_reports(self, run_id: str) -> list[EvaluationReportRow]:
        cur = self._conn.execute(
            "SELECT * FROM evaluation_reports WHERE run_id = ? ORDER BY created_at, report_id",
            (run_id,),
        )
        return [_row_to_report(r) for r in cur.fetchall()]


# ---------- helpers ---------------------------------------------------------

def _serialize_bbox(bbox: Sequence[float]) -> str:
    return json.dumps([float(v) for v in bbox])


def _deserialize_bbox(raw: str) -> tuple[float, float, float, float]:
    data = json.loads(raw)
    if len(data) != 4:
        raise ValueError(f"Corrupt bbox in DB: {raw!r}")
    return float(data[0]), float(data[1]), float(data[2]), float(data[3])


def _row_to_dataset(r: sqlite3.Row) -> DatasetRow:
    return DatasetRow(
        dataset_id=r["dataset_id"], name=r["name"], created_at=r["created_at"],
        source=r["source"], storage_path=r["storage_path"],
        annotation_format=r["annotation_format"],
        num_images=int(r["num_images"]), num_annotations=int(r["num_annotations"]),
        num_classes=int(r["num_classes"]), status=r["status"],
        content_hash=r["content_hash"], notes=r["notes"],
    )


def _row_to_image(r: sqlite3.Row) -> ImageRow:
    return ImageRow(
        image_id=r["image_id"], dataset_id=r["dataset_id"], split=r["split"],
        file_name=r["file_name"], width=int(r["width"]), height=int(r["height"]),
        storage_path=r["storage_path"], has_objects=bool(r["has_objects"]),
        source_frame_id=r["source_frame_id"],
    )


def _row_to_annotation(r: sqlite3.Row) -> AnnotationRow:
    return AnnotationRow(
        annotation_id=r["annotation_id"], dataset_id=r["dataset_id"],
        image_id=r["image_id"], category_id=int(r["category_id"]),
        bbox_xywh=_deserialize_bbox(r["bbox_xywh"]),
        area=float(r["area"]), iscrowd=bool(r["iscrowd"]),
        source_annotation_id=r["source_annotation_id"],
    )


def _row_to_run(r: sqlite3.Row) -> PipelineRunRow:
    keys = set(r.keys())
    return PipelineRunRow(
        run_id=r["run_id"], dataset_id=r["dataset_id"],
        started_at=r["started_at"], finished_at=r["finished_at"],
        status=r["status"], config_path=r["config_path"],
        config_snapshot_path=r["config_snapshot_path"],
        mlflow_run_id=r["mlflow_run_id"], error=r["error"],
        current_stage=(r["current_stage"] if "current_stage" in keys else None),
        model_name=(r["model_name"] if "model_name" in keys else None),
        tile_cache_id=(r["tile_cache_id"] if "tile_cache_id" in keys else None),
    )


def _row_to_tile_cache(r: sqlite3.Row) -> TileCacheRow:
    return TileCacheRow(
        tile_cache_id=r["tile_cache_id"], dataset_id=r["dataset_id"],
        dataset_hash=r["dataset_hash"], split_hash=r["split_hash"],
        tile_config_hash=r["tile_config_hash"], storage_path=r["storage_path"],
        num_train_tiles=r["num_train_tiles"], num_val_tiles=r["num_val_tiles"],
        num_test_tiles=r["num_test_tiles"], created_at=r["created_at"],
        status=r["status"], config_snapshot_path=r["config_snapshot_path"],
    )


def _row_to_artifact(r: sqlite3.Row) -> ArtifactRow:
    return ArtifactRow(
        artifact_id=r["artifact_id"], run_id=r["run_id"],
        artifact_type=r["artifact_type"], storage_path=r["storage_path"],
        created_at=r["created_at"], content_hash=r["content_hash"],
    )


def _row_to_checkpoint(r: sqlite3.Row) -> CheckpointRow:
    return CheckpointRow(
        checkpoint_id=r["checkpoint_id"], run_id=r["run_id"],
        checkpoint_type=r["checkpoint_type"],
        epoch=int(r["epoch"]) if r["epoch"] is not None else None,
        metric_name=r["metric_name"],
        metric_value=float(r["metric_value"]) if r["metric_value"] is not None else None,
        storage_path=r["storage_path"], created_at=r["created_at"],
    )


def _row_to_report(r: sqlite3.Row) -> EvaluationReportRow:
    return EvaluationReportRow(
        report_id=r["report_id"], run_id=r["run_id"],
        checkpoint_id=r["checkpoint_id"], metrics_path=r["metrics_path"],
        report_path=r["report_path"], created_at=r["created_at"],
        primary_metric_name=r["primary_metric_name"],
        primary_metric_value=float(r["primary_metric_value"]) if r["primary_metric_value"] is not None else None,
    )
