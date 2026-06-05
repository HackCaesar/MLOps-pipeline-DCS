"""Top-level pipeline CLI: validate-config / check-gpu / run-all (stubs in Phase 1)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from packages.common.config import ConfigError, load_config
from packages.common.logging_utils import get_logger, setup_logging
from packages.common.paths import resolve_path
from packages.common.run_id import generate_run_id

LOG = get_logger(__name__)


def cmd_validate_config(args: argparse.Namespace) -> int:
    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(f"CONFIG ERROR: {exc}", file=sys.stderr)
        return 2

    # Tolerant lookups: validate-config must surface a resolved summary, not crash
    # on an optional key. host_root, for instance, only exists in the host configs
    # (pipeline.yaml / pipeline_host.yaml), not in pipeline_production.yaml.
    st = cfg.get("storage") or {}
    pl = cfg.get("pipeline") or {}
    tr = cfg.get("training") or {}
    summary = {
        "config_path": str(Path(args.config).resolve()),
        "pipeline.run_id": pl.get("run_id"),
        "pipeline.seed":   pl.get("seed"),
        "storage.root_dir":   st.get("root_dir"),
        "storage.host_root":  st.get("host_root"),
        "storage.datasets_dir": st.get("datasets_dir"),
        "dcs_capture.mode":    (cfg.get("dcs_capture") or {}).get("mode"),
        "training.max_epoch":  tr.get("max_epoch"),
        "training.multiscale_range": tr.get("multiscale_range"),
        "evaluation.full_image_first": (cfg.get("evaluation") or {}).get("full_image_first"),
        "mlflow.tracking_uri": (cfg.get("mlflow") or {}).get("tracking_uri"),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def cmd_check_gpu(args: argparse.Namespace) -> int:
    # Delegates to scripts/check_gpu.py — keeps the heavy stuff in one place.
    from scripts.check_gpu import main as check_main
    return check_main(verbose=args.verbose)


def cmd_run_all(args: argparse.Namespace) -> int:
    from apps.pipeline.run_all import run_all
    result = run_all(
        config_path=args.config,
        run_id=args.run_id,
        dataset_id=args.dataset_id,
        dcs_mode=args.dcs_mode,
        start_at=args.start_at,
        stop_after=args.stop_after,
        only=args.only,
        skip=args.skip,
        dry_run=args.dry_run,
    )
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    return 0 if result.success else 1


def cmd_new_run_id(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    runs_dir = resolve_path(cfg["storage"]["runs_dir"])
    print(generate_run_id(runs_dir=runs_dir))
    return 0


def cmd_gc(args: argparse.Namespace) -> int:
    import datetime as _dt

    import yaml

    from apps.pipeline.gc import RetentionPolicy, apply_plan, collect_state, plan_retention
    from packages.common.metadata_store import SQLiteMetadataStore

    cfg = load_config(args.config)
    storage = cfg.get("storage") or {}

    # Policy: --retention file > configs/retention.yaml (if present) > config 'retention' > defaults.
    pol_cfg = None
    default_retention = Path("configs/retention.yaml")
    if args.retention:
        pol_cfg = yaml.safe_load(Path(args.retention).read_text(encoding="utf-8"))
    elif default_retention.is_file():
        pol_cfg = yaml.safe_load(default_retention.read_text(encoding="utf-8"))
    else:
        pol_cfg = cfg.get("retention")
    policy = RetentionPolicy.from_config(pol_cfg)

    runs_dir = resolve_path(storage.get("runs_dir"))
    cache_dir = resolve_path(storage.get("cache_dir"))
    datasets_dir = resolve_path(storage.get("datasets_dir"))
    db_path = resolve_path((cfg.get("metadata_store") or {}).get("sqlite_path"))
    if db_path is None:
        print("metadata_store.sqlite_path is required for gc", file=sys.stderr)
        return 2

    with SQLiteMetadataStore.open(db_path) as store:
        runs, caches, staging = collect_state(
            store,
            runs_dir=runs_dir,
            cache_tiles_dir=(cache_dir / "tiles") if cache_dir else None,
            staging_dir=(datasets_dir / "_staging") if datasets_dir else None,
        )
    plan = plan_retention(runs, caches, staging, policy, now=_dt.datetime.now(_dt.timezone.utc))
    result = apply_plan(plan, dry_run=not args.apply)
    print(json.dumps({"policy": vars(policy), "plan": plan.to_dict(), "result": result},
                     indent=2, ensure_ascii=False))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from apps.pipeline.run_all import run_all

    if not getattr(args, "watch", False):
        result = run_all(
            config_path=args.config, run_id=args.run_id, dataset_id=args.dataset_id,
            dcs_mode=args.dcs_mode, start_at=args.start_at, stop_after=args.stop_after,
            only=args.only, skip=args.skip, dry_run=args.dry_run,
        )
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
        return 0 if result.success else 1

    import threading

    from apps.pipeline import tui

    cfg = load_config(args.config)
    runs_dir = resolve_path((cfg.get("storage") or {}).get("runs_dir"))
    if args.run_id:
        run_id = args.run_id
    elif runs_dir is not None:
        run_id = generate_run_id(runs_dir=runs_dir)   # resolve up front so the TUI knows the dir
    else:
        run_id = generate_run_id()
    run_dir = (runs_dir / run_id) if runs_dir is not None else Path("runs") / run_id

    holder: dict = {"result": None, "exc": None}
    stop = threading.Event()

    def _worker() -> None:
        try:
            holder["result"] = run_all(
                config_path=args.config, run_id=run_id, dataset_id=args.dataset_id,
                dcs_mode=args.dcs_mode, start_at=args.start_at, stop_after=args.stop_after,
                only=args.only, skip=args.skip, dry_run=args.dry_run,
            )
        except BaseException as exc:   # surface strict-tracking / unexpected failures
            holder["exc"] = exc
        finally:
            stop.set()

    try:
        from apps.pipeline.control_room import ControlRoomApp
    except ImportError:
        ControlRoomApp = None

    worker = threading.Thread(target=_worker, daemon=True)
    worker.start()
    if ControlRoomApp is not None:
        ControlRoomApp(run_dir, interval=getattr(args, "interval", 0.7),
                       exit_on_terminal=True).run()
    else:
        tui.run_live(run_dir, stop_event=stop, ascii=tui.want_ascii())   # main thread only READS files
    worker.join(timeout=10)

    if holder["exc"] is not None:
        print(f"run failed: {holder['exc']}", file=sys.stderr)
        return 1
    result = holder["result"]
    return 0 if (result is not None and result.success) else 1


def _resolve_run_dir(args: argparse.Namespace):
    from apps.pipeline import tui
    cfg = load_config(args.config)
    runs_dir = resolve_path((cfg.get("storage") or {}).get("runs_dir"))
    if runs_dir is None:
        print("storage.runs_dir is not set in config", file=sys.stderr)
        return None
    run_id = args.run_id or tui.latest_run_id(runs_dir)
    if not run_id:
        print(f"no runs found under {runs_dir}; pass --run-id", file=sys.stderr)
        return None
    return runs_dir / run_id


def cmd_tui(args: argparse.Namespace) -> int:
    from apps.pipeline import tui
    run_dir = _resolve_run_dir(args)
    if run_dir is None:
        return 2
    # Default to the Control Room (textual); --ascii or a missing textual install
    # fall back to the basic rich live view / one-shot snapshot.
    if not args.ascii:
        try:
            from apps.pipeline.control_room import run_control_room
        except ImportError:
            print("textual not installed; using the basic view. "
                  "For the Control Room: pip install '.[tui]'")
        else:
            run_control_room(run_dir, interval=args.interval)
            return 0
    tui.run_live(run_dir, interval=args.interval, ascii=args.ascii or tui.want_ascii())
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    from apps.pipeline import tui  # pure helpers only; never imports rich
    run_dir = _resolve_run_dir(args)
    if run_dir is None:
        return 2
    status = tui.read_status(run_dir)
    if getattr(args, "json", False):
        print(json.dumps(status, indent=2, ensure_ascii=False))
        return 0
    vm = tui.build_view(status, tui.tail_events(run_dir), manifest=tui.read_manifest(run_dir))
    tui.print_oneshot(vm, ascii=args.ascii or tui.want_ascii())
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="apps.pipeline.cli",
                                description="Top-level pipeline CLI (DCS → YOLOX).")
    sub = p.add_subparsers(dest="cmd", required=True, metavar="<subcommand>")

    pv = sub.add_parser("validate-config", help="parse pipeline.yaml and print resolved summary")
    pv.add_argument("--config", required=True, type=str)
    pv.set_defaults(func=cmd_validate_config)

    pg = sub.add_parser("check-gpu", help="run nvidia-smi inside a CUDA container; report status")
    pg.add_argument("--verbose", action="store_true")
    pg.set_defaults(func=cmd_check_gpu)

    pr = sub.add_parser("run-all",
                        help="run the full pipeline end-to-end (in-process orchestrator)")
    pr.add_argument("--config",      required=True, type=str)
    pr.add_argument("--run-id",      default=None)
    pr.add_argument("--dataset-id",  default=None)
    pr.add_argument("--dcs-mode",    default=None,
                    choices=["request_agent", "wait_for_existing_raw_dataset", "disabled"])
    pr.add_argument("--start-at",    default=None,
                    help="skip stages before this one")
    pr.add_argument("--stop-after",  default=None,
                    help="stop after running this stage")
    pr.add_argument("--only",        nargs="*", default=None,
                    help="run only the listed stages (in canonical order)")
    pr.add_argument("--skip",        nargs="*", default=None,
                    help="exclude these stages from the run")
    pr.add_argument("--dry-run",     action="store_true",
                    help="print resolved commands instead of executing them")
    pr.set_defaults(func=cmd_run_all)

    pn = sub.add_parser("new-run-id", help="generate the next available YYYY_MM_DD_NNN run id")
    pn.add_argument("--config", required=True, type=str)
    pn.set_defaults(func=cmd_new_run_id)

    pgc = sub.add_parser("gc",
                         help="prune old runs / orphan tile caches / stale staging (dry-run by default)")
    pgc.add_argument("--config", required=True, type=str)
    pgc.add_argument("--retention", default=None,
                     help="retention.yaml path (default: configs/retention.yaml if present)")
    pgc.add_argument("--apply", action="store_true", help="actually delete (default: dry-run)")
    pgc.set_defaults(func=cmd_gc)

    prun = sub.add_parser("run", help="run the pipeline; with --watch, show a live TUI")
    prun.add_argument("--config", required=True, type=str)
    prun.add_argument("--run-id", default=None)
    prun.add_argument("--dataset-id", default=None)
    prun.add_argument("--dcs-mode", default=None,
                      choices=["request_agent", "wait_for_existing_raw_dataset", "disabled"])
    prun.add_argument("--start-at", default=None)
    prun.add_argument("--stop-after", default=None)
    prun.add_argument("--only", nargs="*", default=None)
    prun.add_argument("--skip", nargs="*", default=None)
    prun.add_argument("--dry-run", action="store_true")
    prun.add_argument("--watch", action="store_true",
                      help="render a live Rich TUI while the run executes")
    prun.set_defaults(func=cmd_run)

    ptui = sub.add_parser("tui", help="attach a live TUI to a run (requires rich)")
    ptui.add_argument("--config", required=True, type=str)
    ptui.add_argument("--run-id", default=None, help="default: latest run under storage.runs_dir")
    ptui.add_argument("--interval", type=float, default=0.7)
    ptui.add_argument("--ascii", action="store_true",
                      help="ASCII glyphs (for consoles without UTF-8)")
    ptui.set_defaults(func=cmd_tui)

    pst = sub.add_parser("status", help="one-shot run status snapshot (works without rich)")
    pst.add_argument("--config", required=True, type=str)
    pst.add_argument("--run-id", default=None, help="default: latest run under storage.runs_dir")
    pst.add_argument("--json", action="store_true", help="print the raw status.json")
    pst.add_argument("--ascii", action="store_true")
    pst.set_defaults(func=cmd_status)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(level="INFO")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
