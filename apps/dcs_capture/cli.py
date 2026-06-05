"""DCS capture CLI: ``request-capture``, ``wait-for-dataset``, ``adapter-normalize``."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from apps.dcs_capture.adapter import normalize_dataset
from apps.dcs_capture.agent_client import (
    AgentClient,
    AgentTimeoutError,
    CaptureRequest,
)
from packages.common.config import ConfigError, load_config
from packages.common.logging_utils import get_logger, setup_logging
from packages.common.paths import resolve_path
from packages.common.run_id import generate_run_id

LOG = get_logger(__name__)


def _build_agent_client(cfg: dict) -> AgentClient:
    dcs = cfg.get("dcs_capture") or {}
    return AgentClient(
        agent_url=dcs.get("agent_url"),
        commands_dir=resolve_path(dcs.get("command_folder")),
        status_dir=resolve_path(dcs.get("status_folder")),
        http_timeout_sec=float(dcs.get("http_timeout_sec", 3.0)),
    )


def _resolve_run_id(cfg: dict, override: str | None) -> str:
    if override:
        return override
    rid = (cfg.get("pipeline") or {}).get("run_id")
    if rid:
        return rid
    runs_dir = resolve_path((cfg.get("storage") or {}).get("runs_dir"))
    return generate_run_id(runs_dir=runs_dir) if runs_dir else generate_run_id()


def cmd_request_capture(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    dcs = cfg.get("dcs_capture") or {}
    mode = dcs.get("mode", "request_agent")
    if mode == "disabled":
        LOG.info("dcs_capture.mode=disabled — nothing to request.")
        return 0

    run_id = _resolve_run_id(cfg, args.run_id)
    dataset_id = args.dataset_id or f"dcs_{run_id}"

    raw_root_raw = dcs.get("raw_target_dir") or (cfg.get("storage") or {}).get("datasets_dir")
    if not raw_root_raw:
        LOG.error("Cannot resolve target raw dataset dir")
        return 2
    raw_root = resolve_path(raw_root_raw)
    target_dir = raw_root / dataset_id  # type: ignore[operator]

    mission_path = args.mission or dcs.get("mission_path")
    dcs_config_path = args.dcs_config or dcs.get("dcs_config_path")
    dcs_source = args.dcs_source_dataset_dir or dcs.get("source_output_dir")
    if not (mission_path and dcs_config_path and dcs_source):
        LOG.error("Need mission_path + dcs_config_path + source_output_dir "
                  "(via CLI flag or dcs_capture.* config)")
        return 2

    req = CaptureRequest(
        run_id=run_id,
        dataset_id=dataset_id,
        mission_path=str(mission_path),
        dcs_config_path=str(dcs_config_path),
        dcs_source_dataset_dir=str(dcs_source),
        target_dataset_dir=str(target_dir),
        num_frames=int(args.num_frames or dcs.get("num_frames", 1)),
        timeout_sec=int(args.timeout_sec or dcs.get("timeout_sec", 7200)),
    )

    client = _build_agent_client(cfg)
    if not client.health() and mode == "request_agent":
        LOG.error("Agent unreachable and no shared-folder fallback configured.")
        return 1

    submit = client.request_capture(req)
    LOG.info("Capture submitted via %s transport: run_id=%s, accepted=%s",
             submit.get("transport"), submit.get("run_id"), submit.get("accepted"))
    if not submit.get("accepted"):
        LOG.error("Agent rejected capture: %s", submit.get("reason"))
        return 1
    print(json.dumps(submit, indent=2, ensure_ascii=False))

    if args.no_wait:
        return 0

    try:
        final = client.wait_for_status(
            run_id,
            timeout_sec=req.timeout_sec,
            poll_interval_sec=float(args.poll_interval_sec),
        )
    except AgentTimeoutError as exc:
        LOG.error("%s", exc)
        return 1
    print(json.dumps({"final_status": final}, indent=2, ensure_ascii=False))
    return 0 if final.get("status") == "success" else 1


def cmd_wait_for_dataset(args: argparse.Namespace) -> int:
    """Wait for a raw dataset directory to appear (used in `mode=wait_for_existing_raw_dataset`)."""
    cfg = load_config(args.config)
    dcs = cfg.get("dcs_capture") or {}
    raw_root = resolve_path(dcs.get("raw_target_dir") or (cfg.get("storage") or {}).get("datasets_dir"))
    if raw_root is None:
        LOG.error("Cannot resolve raw target dir")
        return 2
    dataset_id = args.dataset_id or (cfg.get("data") or {}).get("dataset_id")
    if not dataset_id:
        LOG.error("--dataset-id is required (no value in config)")
        return 2

    target = raw_root / dataset_id
    sentinel = target / "metadata" / "dataset_info.json"
    timeout = float(args.timeout_sec or dcs.get("timeout_sec", 7200))
    poll = float(args.poll_interval_sec)

    import time
    deadline = time.monotonic() + timeout
    LOG.info("Waiting for %s (timeout=%.0fs, poll=%.1fs)", sentinel, timeout, poll)
    while time.monotonic() < deadline:
        if sentinel.is_file():
            LOG.info("Dataset appeared: %s", target)
            print(json.dumps({"target_dataset_dir": str(target),
                              "dataset_id": dataset_id}, indent=2))
            return 0
        time.sleep(poll)
    LOG.error("Timed out waiting for %s", sentinel)
    return 1


def cmd_adapter_normalize(args: argparse.Namespace) -> int:
    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        LOG.error("CONFIG ERROR: %s", exc)
        return 2

    source = resolve_path(args.source_dir)
    if source is None or not source.is_dir():
        LOG.error("Source dataset not found: %s", args.source_dir)
        return 1

    raw_root_raw = (cfg.get("dcs_capture") or {}).get("raw_target_dir") \
        or (cfg.get("storage") or {}).get("datasets_dir")
    if not raw_root_raw:
        LOG.error("Cannot resolve raw target directory: set dcs_capture.raw_target_dir or storage.datasets_dir")
        return 2
    raw_root = resolve_path(raw_root_raw)
    assert raw_root is not None

    dataset_id = args.dataset_id or _auto_dataset_id(source)
    target_dir = raw_root / dataset_id

    split_ratios = (cfg.get("data") or {}).get("split_ratios") \
        or {"train": 0.70, "val": 0.15, "test": 0.15}

    dcs_config_path = args.dcs_config or (cfg.get("dcs_capture") or {}).get("dcs_config_path")

    image_placement = getattr(args, "link_mode", None) \
        or (cfg.get("dcs_capture") or {}).get("image_placement") \
        or "copy"   # robust default; hardlink is opt-in (see --link-mode)

    LOG.info("Adapter: %s → %s (dataset_id=%s)", source, target_dir, dataset_id)
    result = normalize_dataset(
        source_dir=source,
        target_dir=target_dir,
        dataset_id=dataset_id,
        name=args.name,
        split_ratios=split_ratios,
        dcs_config_path=dcs_config_path,
        dataset_source=args.dataset_source,
        overwrite=args.overwrite,
        copy_images=not args.no_images,
        image_placement=image_placement,
        allow_copy_fallback=bool(getattr(args, "allow_copy_fallback", False)),
    )

    # Staging is ephemeral — optionally remove the source after a SUCCESSFUL
    # import (normalize_dataset returning == all writes above succeeded), but only
    # when it actually lives under a `_staging/` path. Never on error.
    src_resolved = Path(str(source)).resolve()
    if getattr(args, "cleanup_staging", False) and not getattr(args, "keep_staging", False):
        if not args.no_images and "_staging" in src_resolved.parts:
            shutil.rmtree(src_resolved, ignore_errors=True)
            LOG.info("Removed ephemeral staging source after import: %s", src_resolved)
        else:
            LOG.warning("--cleanup-staging skipped: %s is not under a _staging/ path "
                        "(or --no-images set)", src_resolved)

    summary = {
        "dataset_id":      result.dataset_id,
        "target_dir":      str(result.target_dir),
        "num_images":      result.num_images,
        "num_annotations": result.num_annotations,
        "num_classes":     result.num_classes,
        "splits":          result.splits,
        "warnings":        result.warnings,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if result.warnings:
        LOG.warning("Adapter finished with %d warnings.", len(result.warnings))
    return 0


def _auto_dataset_id(source: Path) -> str:
    """Default dataset_id when not given: "dcs_<source-folder-name>_<YYYYMMDD>"."""
    import datetime as _dt
    today = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%d")
    return f"dcs_{source.name}_{today}"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="apps.dcs_capture.cli",
                                description="DCS capture: request agent / wait / adapter.")
    sub = p.add_subparsers(dest="cmd", required=True, metavar="<subcommand>")

    p1 = sub.add_parser("request-capture",
                        help="ask Windows DCS Runner Agent to capture a fresh dataset")
    p1.add_argument("--config", required=True)
    p1.add_argument("--run-id", default=None,
                    help="explicit run_id (default: from pipeline.run_id or autogenerate)")
    p1.add_argument("--dataset-id", default=None,
                    help="explicit dataset_id (default: derived from run_id)")
    p1.add_argument("--mission", default=None,
                    help="override dcs_capture.mission_path")
    p1.add_argument("--dcs-config", default=None,
                    help="override dcs_capture.dcs_config_path")
    p1.add_argument("--dcs-source-dataset-dir", default=None,
                    help="override dcs_capture.source_output_dir (where DCS writes its plain dataset)")
    p1.add_argument("--num-frames", type=int, default=None,
                    help="override dcs_capture.num_frames")
    p1.add_argument("--timeout-sec", type=int, default=None,
                    help="override dcs_capture.timeout_sec")
    p1.add_argument("--no-wait", action="store_true",
                    help="submit and return immediately (don't block on status)")
    p1.add_argument("--poll-interval-sec", type=float, default=5.0,
                    help="status poll interval in seconds (default 5.0)")
    p1.set_defaults(func=cmd_request_capture)

    p2 = sub.add_parser("wait-for-dataset",
                        help="poll until storage/datasets/raw/{dataset_id}/ appears")
    p2.add_argument("--config", required=True)
    p2.add_argument("--dataset-id", default=None,
                    help="dataset id to wait for (default: data.dataset_id from config)")
    p2.add_argument("--timeout-sec", type=int, default=None)
    p2.add_argument("--poll-interval-sec", type=float, default=5.0)
    p2.set_defaults(func=cmd_wait_for_dataset)

    p3 = sub.add_parser("adapter-normalize",
                        help="convert flat DCS dataset to raw dataset contract layout")
    p3.add_argument("--config", required=True)
    p3.add_argument("--source-dir", required=True,
                    help="path to the DCS_AutoDataset/dataset/ directory")
    p3.add_argument("--dataset-id", required=False,
                    help="target dataset_id (default: derived from source folder + date)")
    p3.add_argument("--name", required=False, help="human-readable dataset name")
    p3.add_argument("--dataset-source", default="DCS_AutoDataset",
                    help="value to write into dataset_info.source")
    p3.add_argument("--dcs-config", required=False,
                    help="path to DCS_AutoDataset config.yaml to snapshot (default from pipeline.yaml)")
    p3.add_argument("--overwrite", action="store_true",
                    help="remove existing target_dir before writing")
    p3.add_argument("--no-images", action="store_true",
                    help="skip image copy (annotations/metadata only)")
    p3.add_argument("--link-mode", choices=["hardlink", "copy", "move"], default=None,
                    help="how to place images into raw (default: config dcs_capture.image_placement, else copy; "
                         "hardlink is opt-in and requires staging+raw on one volume)")
    p3.add_argument("--allow-copy-fallback", action="store_true",
                    help="permit a copy when hardlink fails (cross-filesystem); off by default "
                         "so a hidden full-copy duplicate cannot happen silently")
    p3.add_argument("--cleanup-staging", action="store_true",
                    help="remove the source after a successful import, if it is under a _staging/ path")
    p3.add_argument("--keep-staging", action="store_true",
                    help="never remove the staging source (overrides --cleanup-staging)")
    p3.set_defaults(func=cmd_adapter_normalize)

    return p


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
