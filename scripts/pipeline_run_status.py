"""CLI wrappers for ``MetadataStore.create_pipeline_run`` and
``update_pipeline_run_status``.

Two subcommands:

  create  — insert a new pipeline_runs row at the start of a run.
  update  — update status (running|success|failed|stopped) + finished_at + error.

Keeps the SQLite metadata store in sync with run state. Driven by
``apps.pipeline.run_all`` (and usable from a shell when debugging).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path
from typing import Optional

from packages.common.config import ConfigError, load_config
from packages.common.logging_utils import get_logger, setup_logging
from packages.common.metadata_store import SQLiteMetadataStore
from packages.common.paths import resolve_path

LOG = get_logger(__name__)


def _resolve_db_path(cfg: dict, override: Optional[str]) -> Path:
    if override:
        p = resolve_path(override)
        assert p is not None
        return p
    raw = (cfg.get("metadata_store") or {}).get("sqlite_path")
    if not raw:
        raise SystemExit("metadata_store.sqlite_path missing from config and no --db-path")
    p = resolve_path(raw)
    assert p is not None
    return p


def _utc_now_iso() -> str:
    return (_dt.datetime.now(_dt.timezone.utc)
            .replace(microsecond=0).isoformat().replace("+00:00", "Z"))


def cmd_create(args: argparse.Namespace) -> int:
    cfg = load_config(args.config) if args.config else {}
    db_path = _resolve_db_path(cfg, args.db_path)
    with SQLiteMetadataStore(db_path) as store:
        store.init_db()
        existing = store.get_run(args.run_id)
        if existing is not None and not args.replace:
            LOG.info("pipeline_run %s already exists; pass --replace to overwrite", args.run_id)
            print(json.dumps({"run_id": args.run_id, "existing": True}, indent=2))
            return 0
        if existing is not None and args.replace:
            # Delete via raw conn (no public delete method).
            with store._conn:  # noqa: SLF001
                store._conn.execute("DELETE FROM pipeline_runs WHERE run_id = ?", (args.run_id,))  # noqa: SLF001

        # FK constraint on pipeline_runs.dataset_id → datasets.dataset_id.
        # The DAG calls create_pipeline_run BEFORE register_dataset, so the row
        # may not exist yet; soft-null it with a warning rather than crashing.
        dataset_id = args.dataset_id
        if dataset_id is not None and store.get_dataset(dataset_id) is None:
            LOG.warning("dataset_id=%s not yet registered; recording pipeline_run "
                         "with dataset_id=NULL (will be linked when register_dataset runs)",
                         dataset_id)
            dataset_id = None

        store.create_pipeline_run(
            run_id=args.run_id,
            dataset_id=dataset_id,
            started_at=args.started_at or _utc_now_iso(),
            status=args.status,
            config_path=args.config_path or (args.config or ""),
            config_snapshot_path=args.config_snapshot_path or "",
            mlflow_run_id=args.mlflow_run_id,
        )
        got = store.get_run(args.run_id)
    print(json.dumps({"run_id": args.run_id, "created": True,
                       "status": got.status if got else None}, indent=2))
    return 0


def cmd_update(args: argparse.Namespace) -> int:
    cfg = load_config(args.config) if args.config else {}
    db_path = _resolve_db_path(cfg, args.db_path)
    with SQLiteMetadataStore(db_path) as store:
        try:
            store.update_pipeline_run_status(
                args.run_id,
                status=args.status,
                finished_at=args.finished_at,
                error=args.error,
                mlflow_run_id=args.mlflow_run_id,
            )
        except KeyError as exc:
            LOG.error("%s", exc)
            return 1
        got = store.get_run(args.run_id)
    print(json.dumps({
        "run_id": args.run_id,
        "status": got.status if got else None,
        "finished_at": got.finished_at if got else None,
    }, indent=2))
    return 0


def cmd_get(args: argparse.Namespace) -> int:
    cfg = load_config(args.config) if args.config else {}
    db_path = _resolve_db_path(cfg, args.db_path)
    with SQLiteMetadataStore(db_path) as store:
        got = store.get_run(args.run_id)
    if got is None:
        LOG.error("No pipeline_run with run_id=%s", args.run_id)
        return 1
    from dataclasses import asdict
    print(json.dumps(asdict(got), indent=2, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True, metavar="<subcommand>")

    pc = sub.add_parser("create", help="insert a new pipeline_runs row")
    pc.add_argument("--config",   default=None)
    pc.add_argument("--db-path",  default=None)
    pc.add_argument("--run-id",   required=True)
    pc.add_argument("--dataset-id", default=None)
    pc.add_argument("--status",   default="running",
                    choices=["queued", "running", "success", "failed", "stopped"])
    pc.add_argument("--started-at",          default=None)
    pc.add_argument("--config-path",         default=None)
    pc.add_argument("--config-snapshot-path", default=None)
    pc.add_argument("--mlflow-run-id",       default=None)
    pc.add_argument("--replace", action="store_true")
    pc.set_defaults(func=cmd_create)

    pu = sub.add_parser("update", help="update status/finished_at/error on an existing run")
    pu.add_argument("--config",   default=None)
    pu.add_argument("--db-path",  default=None)
    pu.add_argument("--run-id",   required=True)
    pu.add_argument("--status",   required=True,
                    choices=["queued", "running", "success", "failed", "stopped"])
    pu.add_argument("--finished-at",   default=None,
                    help="ISO timestamp; defaults to now when status is terminal")
    pu.add_argument("--error",        default=None)
    pu.add_argument("--mlflow-run-id", default=None)
    pu.add_argument("--auto-finished-at", action="store_true",
                    help="set finished_at=now if status is terminal and --finished-at omitted")
    pu.set_defaults(func=cmd_update)

    pg = sub.add_parser("get", help="print the pipeline_runs row as JSON")
    pg.add_argument("--config",  default=None)
    pg.add_argument("--db-path", default=None)
    pg.add_argument("--run-id",  required=True)
    pg.set_defaults(func=cmd_get)
    return p


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    args = build_parser().parse_args(argv)
    if args.cmd == "update" and getattr(args, "auto_finished_at", False) \
            and args.finished_at is None \
            and args.status in {"success", "failed", "stopped"}:
        args.finished_at = _utc_now_iso()
    try:
        return args.func(args)
    except ConfigError as exc:
        LOG.error("CONFIG ERROR: %s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
