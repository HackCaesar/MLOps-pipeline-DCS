"""Scan ``storage/artifacts/{checkpoints,reports,exports}/{run_id}/`` and
register every artifact in the SQLite metadata store.

Idempotent: re-runs replace previously-registered rows when ``--replace`` is set.

Used by ``apps.pipeline.run_all`` after each stage that produces artifacts.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from packages.common.config import ConfigError, load_config
from packages.common.logging_utils import get_logger, setup_logging
from packages.common.metadata_store import SQLiteMetadataStore
from packages.common.paths import resolve_path
from packages.common.storage import LocalStorageBackend

LOG = get_logger(__name__)


# Match the YOLOX checkpoint filename convention from Phase 6:
#   best_ckpt.pth, latest_ckpt.pth, epoch_NNN_ckpt.pth, last_epoch_ckpt.pth, last_mosaic_epoch_ckpt.pth
_CHECKPOINT_RE = re.compile(r"^(?P<kind>best|latest|last_epoch|last_mosaic_epoch|epoch_(?P<epoch>\d+))_ckpt\.pth$")


@dataclass(frozen=True)
class RegistrationSummary:
    run_id: str
    checkpoints: int = 0
    reports:     int = 0
    artifacts:   int = 0     # exports + visualizations + misc
    skipped:     int = 0

    def to_dict(self) -> dict:
        return dict(self.__dict__)


# ---------------------------------------------------------------------- #
# core entry
# ---------------------------------------------------------------------- #

def register_artifacts(
    *,
    run_id: str,
    artifacts_dir: Path,
    metadata_db_path: Path,
    storage_root: Optional[Path] = None,
    replace: bool = False,
) -> RegistrationSummary:
    """Walk ``artifacts_dir`` for ``{run_id}`` subfolders and register everything."""
    backend = LocalStorageBackend(root_dir=storage_root or artifacts_dir.parent)
    counters = {"checkpoints": 0, "reports": 0, "artifacts": 0, "skipped": 0}

    with SQLiteMetadataStore(metadata_db_path) as store:
        store.init_db()
        if replace:
            _delete_existing(store, run_id)

        ckpt_dir = artifacts_dir / "checkpoints" / run_id
        for ckpt_path in _iter_files(ckpt_dir, ("*.pth",)):
            try:
                _register_checkpoint(store, backend, run_id, ckpt_path, ckpt_dir)
                counters["checkpoints"] += 1
            except Exception as exc:  # noqa: BLE001
                LOG.warning("Skipping checkpoint %s: %s", ckpt_path, exc)
                counters["skipped"] += 1

        report_dir = artifacts_dir / "reports" / run_id
        if report_dir.is_dir():
            try:
                _register_report(store, backend, run_id, report_dir)
                counters["reports"] += 1
            except Exception as exc:  # noqa: BLE001
                LOG.warning("Skipping report dir %s: %s", report_dir, exc)
                counters["skipped"] += 1

            for path in _iter_files(report_dir, ("*.json", "*.csv", "*.png", "*.jsonl")):
                try:
                    _register_generic_artifact(store, backend, run_id, path,
                                                artifact_type="report_file")
                    counters["artifacts"] += 1
                except Exception as exc:  # noqa: BLE001
                    LOG.warning("Skipping %s: %s", path, exc)
                    counters["skipped"] += 1

        for sub in ("exports", "visualizations"):
            sub_dir = artifacts_dir / sub / run_id
            for path in _iter_files(sub_dir, ("*",)):
                try:
                    _register_generic_artifact(store, backend, run_id, path,
                                                artifact_type=sub.rstrip("s"))
                    counters["artifacts"] += 1
                except Exception as exc:  # noqa: BLE001
                    LOG.warning("Skipping %s: %s", path, exc)
                    counters["skipped"] += 1

    return RegistrationSummary(
        run_id=run_id,
        checkpoints=counters["checkpoints"],
        reports=counters["reports"],
        artifacts=counters["artifacts"],
        skipped=counters["skipped"],
    )


# ---------------------------------------------------------------------- #
# helpers
# ---------------------------------------------------------------------- #

def _iter_files(directory: Path, patterns: Iterable[str]) -> List[Path]:
    if not directory.is_dir():
        return []
    out: List[Path] = []
    for pat in patterns:
        for p in directory.rglob(pat):
            if p.is_file():
                out.append(p)
    return sorted(set(out))


def _register_checkpoint(
    store: SQLiteMetadataStore,
    backend: LocalStorageBackend,
    run_id: str,
    ckpt_path: Path,
    ckpt_dir: Path,
) -> None:
    fname = ckpt_path.name
    m = _CHECKPOINT_RE.match(fname)
    if m:
        kind = m.group("kind")
        epoch = int(m.group("epoch")) if m.group("epoch") else None
        if kind.startswith("epoch_"):
            kind = "every_n"
    else:
        kind = "unknown"
        epoch = None

    # Try to read AP from a sibling metrics JSON if present.
    metric_name: Optional[str] = None
    metric_value: Optional[float] = None
    sibling_metrics = ckpt_dir.parent.parent / "reports" / run_id / "metrics" / "summary.json"
    if sibling_metrics.is_file():
        try:
            data = json.loads(sibling_metrics.read_text(encoding="utf-8"))
            metrics = data.get("metrics") or {}
            metric_name = "ap50_95"
            v = metrics.get("map5095")
            if v is not None:
                metric_value = float(v)
        except (OSError, json.JSONDecodeError, KeyError, TypeError):
            pass

    store.register_checkpoint(
        checkpoint_id=f"{run_id}:{fname}",
        run_id=run_id,
        checkpoint_type=kind,
        storage_path=str(ckpt_path),
        created_at=_mtime_iso(ckpt_path),
        epoch=epoch,
        metric_name=metric_name,
        metric_value=metric_value,
    )


def _register_report(
    store: SQLiteMetadataStore,
    backend: LocalStorageBackend,
    run_id: str,
    report_dir: Path,
) -> None:
    summary = report_dir / "metrics" / "summary.json"
    primary_name: Optional[str] = None
    primary_value: Optional[float] = None
    if summary.is_file():
        try:
            data = json.loads(summary.read_text(encoding="utf-8"))
            metrics = data.get("metrics") or {}
            v = metrics.get("map50")
            if v is not None:
                primary_name = "mAP@50"
                primary_value = float(v)
        except (OSError, json.JSONDecodeError, KeyError, TypeError):
            pass

    store.register_evaluation_report(
        report_id=f"{run_id}:report",
        run_id=run_id,
        metrics_path=str(summary) if summary.is_file() else str(report_dir),
        report_path=str(report_dir),
        created_at=_mtime_iso(report_dir),
        primary_metric_name=primary_name,
        primary_metric_value=primary_value,
    )


def _register_generic_artifact(
    store: SQLiteMetadataStore,
    backend: LocalStorageBackend,
    run_id: str,
    path: Path,
    artifact_type: str,
) -> None:
    h: Optional[str] = None
    try:
        h = backend.compute_file_hash(path)
    except (OSError, ValueError):
        pass
    store.register_artifact(
        artifact_id=f"{run_id}:{artifact_type}:{path.name}",
        run_id=run_id,
        artifact_type=artifact_type,
        storage_path=str(path),
        created_at=_mtime_iso(path),
        content_hash=h,
    )


def _delete_existing(store: SQLiteMetadataStore, run_id: str) -> None:
    """Remove artifact/checkpoint/report rows for run_id so we can re-register."""
    # We hold the connection inside the context manager; bypass into the raw
    # connection via the public attribute. This is the only place that needs
    # write-level access without a wrapper method.
    conn = store._conn  # noqa: SLF001
    with conn:
        conn.execute("DELETE FROM artifacts          WHERE run_id = ?", (run_id,))
        conn.execute("DELETE FROM model_checkpoints  WHERE run_id = ?", (run_id,))
        conn.execute("DELETE FROM evaluation_reports WHERE run_id = ?", (run_id,))


def _mtime_iso(p: Path) -> str:
    ts = p.stat().st_mtime
    return (_dt.datetime.fromtimestamp(ts, tz=_dt.timezone.utc)
            .replace(microsecond=0).isoformat().replace("+00:00", "Z"))


# ---------------------------------------------------------------------- #
# CLI
# ---------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",        default=None, help="pipeline.yaml; supplies sqlite_path + artifacts_dir if --db-path / --artifacts-dir are omitted")
    p.add_argument("--run-id",        required=True)
    p.add_argument("--db-path",       default=None)
    p.add_argument("--artifacts-dir", default=None)
    p.add_argument("--storage-root",  default=None)
    p.add_argument("--replace",       action="store_true",
                   help="delete existing rows for this run_id before inserting")
    args = p.parse_args(argv)

    setup_logging()

    db_path: Optional[Path] = None
    artifacts_dir: Optional[Path] = None
    storage_root: Optional[Path] = None
    if args.config:
        try:
            cfg = load_config(args.config)
        except ConfigError as exc:
            LOG.error("CONFIG ERROR: %s", exc)
            return 2
        db_path = resolve_path((cfg.get("metadata_store") or {}).get("sqlite_path"))
        artifacts_dir = resolve_path((cfg.get("storage") or {}).get("artifacts_dir"))
        storage_root = resolve_path((cfg.get("storage") or {}).get("root_dir"))
    if args.db_path:
        db_path = resolve_path(args.db_path)
    if args.artifacts_dir:
        artifacts_dir = resolve_path(args.artifacts_dir)
    if args.storage_root:
        storage_root = resolve_path(args.storage_root)

    if db_path is None or artifacts_dir is None:
        LOG.error("Need both metadata DB path and artifacts dir "
                  "(via --config or --db-path + --artifacts-dir)")
        return 2

    result = register_artifacts(
        run_id=args.run_id,
        artifacts_dir=artifacts_dir,
        metadata_db_path=db_path,
        storage_root=storage_root,
        replace=args.replace,
    )
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    LOG.info("OK: registered %d checkpoint(s), %d report(s), %d generic artifact(s); %d skipped",
             result.checkpoints, result.reports, result.artifacts, result.skipped)
    return 0


if __name__ == "__main__":
    sys.exit(main())
