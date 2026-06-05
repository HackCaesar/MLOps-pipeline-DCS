"""Create (or upgrade) the pipeline SQLite metadata database.

Reads ``metadata_store.sqlite_path`` from the config and runs
``SQLiteMetadataStore.init_db()``. Idempotent: re-running is safe.

Example:
    python -m scripts.init_metadata_db --config configs/pipeline.yaml
"""
from __future__ import annotations

import argparse
import sys

from packages.common.config import ConfigError, load_config
from packages.common.logging_utils import get_logger, setup_logging
from packages.common.metadata_store import TABLE_NAMES, SQLiteMetadataStore
from packages.common.paths import resolve_path

LOG = get_logger(__name__)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, help="path to pipeline.yaml")
    p.add_argument("--db-path", default=None,
                   help="override metadata_store.sqlite_path from config")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    setup_logging(level="WARNING" if args.quiet else "INFO")

    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        LOG.error("CONFIG ERROR: %s", exc)
        return 2

    db_path_raw: str | None = args.db_path or cfg.get("metadata_store", {}).get("sqlite_path")
    if not db_path_raw:
        LOG.error("metadata_store.sqlite_path is missing from config and no --db-path override given")
        return 2

    db_path = resolve_path(db_path_raw)
    assert db_path is not None
    db_path.parent.mkdir(parents=True, exist_ok=True)

    LOG.info("Initializing SQLite metadata DB at %s", db_path)
    with SQLiteMetadataStore(db_path) as store:
        store.init_db()
        present = store.list_tables()

    missing = set(TABLE_NAMES) - set(present)
    if missing:
        LOG.error("Missing tables after init_db: %s", sorted(missing))
        return 1

    LOG.info("OK. Tables present: %s", ", ".join(sorted(TABLE_NAMES)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
