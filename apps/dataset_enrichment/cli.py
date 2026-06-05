"""Dataset enrichment CLI: dry-run + build."""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Optional

from apps.dataset_enrichment.enrich import enrich_dataset
from apps.dataset_enrichment.tile_cache import build_or_reuse_cache
from packages.common.config import ConfigError, load_config
from packages.common.logging_utils import get_logger, setup_logging
from packages.common.mlflow_utils import MLflowFacade, flatten_params
from packages.common.paths import resolve_path
from packages.common.run_id import generate_run_id

LOG = get_logger(__name__)


def _resolve_dataset_dir(cfg: dict) -> Path:
    raw_dir_raw = (cfg.get("data") or {}).get("raw_dataset_dir")
    if not raw_dir_raw:
        # Try assembling from dataset_id + datasets_dir.
        dataset_id = (cfg.get("data") or {}).get("dataset_id")
        datasets_dir = (cfg.get("storage") or {}).get("datasets_dir")
        if dataset_id and datasets_dir:
            raw_dir_raw = f"{datasets_dir}/raw/{dataset_id}"
    if not raw_dir_raw:
        raise SystemExit("data.raw_dataset_dir (or data.dataset_id + storage.datasets_dir) missing in config")
    p = resolve_path(raw_dir_raw)
    assert p is not None
    return p


def _resolve_cache_root(cfg: dict) -> Path:
    cache_dir_raw = (cfg.get("storage") or {}).get("cache_dir")
    if not cache_dir_raw:
        root = (cfg.get("storage") or {}).get("root_dir")
        if not root:
            raise SystemExit("storage.cache_dir (or storage.root_dir) missing in config")
        cache_dir_raw = f"{root}/cache"
    p = resolve_path(cache_dir_raw)
    assert p is not None
    return p / "tiles"


def _resolve_runs_dir(cfg: dict) -> Optional[Path]:
    return resolve_path((cfg.get("storage") or {}).get("runs_dir"))


def cmd_dry_run(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    dataset_dir = _resolve_dataset_dir(cfg)
    run_id = args.run_id or "dryrun"

    # Dry-run never materializes a cache — use a throwaway tempdir, not storage/tmp.
    with tempfile.TemporaryDirectory(prefix="enrich_dryrun_") as scratch:
        LOG.info("DRY-RUN: dataset=%s, num_images=%d", dataset_dir, args.num_images)
        result = enrich_dataset(
            raw_dataset_dir=dataset_dir,
            tmp_root=Path(scratch),
            dataset_id=_dataset_id_from_dir(dataset_dir),
            run_id=run_id,
            enrichment_cfg=cfg.get("enrichment") or {},
            num_images_per_split=args.num_images,
            write_images=False,
            write_coco=False,
        )
        print(json.dumps(result.to_summary(), indent=2, ensure_ascii=False))
    return 0


def cmd_build_cache(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    dataset_dir = _resolve_dataset_dir(cfg)
    dataset_id = _dataset_id_from_dir(dataset_dir)
    enrichment_cfg = cfg.get("enrichment") or {}
    cache_root = _resolve_cache_root(cfg)
    runs_dir = _resolve_runs_dir(cfg)

    run_id = args.run_id or (cfg.get("pipeline") or {}).get("run_id")
    if not run_id:
        run_id = generate_run_id(runs_dir=runs_dir) if runs_dir is not None else generate_run_id()
        LOG.info("Generated run_id=%s", run_id)

    LOG.info("BUILD-CACHE: dataset=%s, cache_root=%s, run_id=%s", dataset_dir, cache_root, run_id)

    mlflow = MLflowFacade.from_config(cfg)
    with mlflow.start_run(
        run_name=f"{run_id}_tile_cache",
        tags={"stage": "tile_cache", "run_id": run_id, "dataset_id": dataset_id},
    ) as mlrun:
        mlrun.log_params(flatten_params("enrichment", enrichment_cfg))
        info = build_or_reuse_cache(
            raw_dataset_dir=dataset_dir,
            dataset_id=dataset_id,
            enrichment_cfg=enrichment_cfg,
            cache_root=cache_root,
            runs_dir=runs_dir,
            run_id=run_id,
            strict=bool(getattr(args, "strict_hash", False)),
        )
        mlrun.log_params({
            "tile_cache_id": info["tile_cache_id"],
            "cache_status": info["status"],
            "dataset_id": dataset_id,
            "run_id": run_id,
        })
        mlrun.log_metrics({
            "num_train_tiles": info["num_train_tiles"],
            "num_val_tiles": info["num_val_tiles"],
            "num_test_tiles": info["num_test_tiles"],
        })
        cache_dir = Path(info["cache_dir"])
        for art in ("tile_manifest.json", "cache_meta.json", "dropped_tiles_manifest.jsonl"):
            p = cache_dir / art
            if p.is_file():
                mlrun.log_artifact(p, artifact_path="tile_cache")

    print(json.dumps(info, indent=2, ensure_ascii=False))
    return 0


def _dataset_id_from_dir(dataset_dir: Path) -> str:
    info_path = dataset_dir / "metadata" / "dataset_info.json"
    if info_path.is_file():
        try:
            return json.loads(info_path.read_text(encoding="utf-8"))["dataset_id"]
        except (KeyError, json.JSONDecodeError):
            pass
    return dataset_dir.name


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="apps.dataset_enrichment.cli",
                                description="Multi-scale tiling enrichment for YOLOX 640×640 training.")
    sub = p.add_subparsers(dest="cmd", required=True, metavar="<subcommand>")

    p1 = sub.add_parser("dry-run", help="show what would be enriched for N images per split (no writes)")
    p1.add_argument("--config", required=True)
    p1.add_argument("--num-images", type=int, default=3)
    p1.add_argument("--run-id", required=False)
    p1.set_defaults(func=cmd_dry_run)

    p2 = sub.add_parser("build-cache",
                        help="build or reuse the content-addressed tile cache under storage/cache/tiles/")
    p2.add_argument("--config", required=True)
    p2.add_argument("--run-id", required=False)
    p2.add_argument("--strict-hash", action="store_true",
                    help="hash image bytes (slower) instead of (relpath, size)")
    p2.set_defaults(func=cmd_build_cache)

    # Back-compat alias: `build` now routes to the cache builder (no more tmp).
    p3 = sub.add_parser("build", help="alias of build-cache (kept for back-compat)")
    p3.add_argument("--config", required=True)
    p3.add_argument("--run-id", required=False)
    p3.add_argument("--strict-hash", action="store_true")
    p3.set_defaults(func=cmd_build_cache)

    return p


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except ConfigError as exc:
        LOG.error("CONFIG ERROR: %s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
