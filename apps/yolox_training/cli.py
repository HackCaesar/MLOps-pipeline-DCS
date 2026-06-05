"""YOLOX training CLI: ``train`` and ``dry-run``."""
from __future__ import annotations

import argparse
import sys

from packages.common.config import ConfigError, load_config
from packages.common.logging_utils import get_logger, setup_logging
from packages.common.paths import resolve_path
from packages.common.run_id import generate_run_id

LOG = get_logger(__name__)


def _resolve_run_id(cfg: dict, override: str | None) -> str:
    if override:
        return override
    run_id = (cfg.get("pipeline") or {}).get("run_id")
    if run_id:
        return run_id
    runs_dir = resolve_path((cfg.get("storage") or {}).get("runs_dir"))
    return generate_run_id(runs_dir=runs_dir) if runs_dir else generate_run_id()


def cmd_train(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    from apps.yolox_training.train import run_training
    run_id = _resolve_run_id(cfg, args.run_id)
    return run_training(cfg, run_id=run_id, dry_run=False, max_iter=None)


def cmd_dry_run(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    from apps.yolox_training.train import run_training
    run_id = _resolve_run_id(cfg, args.run_id)
    LOG.info("DRY-RUN: capping at %d iterations", args.max_iter)
    return run_training(cfg, run_id=run_id, dry_run=True, max_iter=args.max_iter)


def cmd_check_env(args: argparse.Namespace) -> int:
    from apps.yolox_training.train import check_env
    env = check_env()
    print(f"torch installed:    {env.has_torch}")
    if env.has_torch:
        print(f"torch version:      {env.torch_version}")
        print(f"CUDA available:     {env.has_cuda}")
        print(f"CUDA device count:  {env.cuda_device_count}")
    try:
        import yolox  # noqa: F401
        print("yolox installed:    True")
    except ImportError:
        print("yolox installed:    False")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="apps.yolox_training.cli",
        description="YOLOX training (640×640, multiscale_range=0).",
    )
    sub = p.add_subparsers(dest="cmd", required=True, metavar="<subcommand>")

    p1 = sub.add_parser("train", help="full training run (requires GPU unless allow_cpu_fallback)")
    p1.add_argument("--config", required=True)
    p1.add_argument("--run-id", required=False)
    p1.set_defaults(func=cmd_train)

    p2 = sub.add_parser("dry-run", help="short --max-iter N run for smoke testing (CPU OK)")
    p2.add_argument("--config", required=True)
    p2.add_argument("--max-iter", type=int, default=5)
    p2.add_argument("--run-id", required=False)
    p2.set_defaults(func=cmd_dry_run)

    p3 = sub.add_parser("check-env", help="report torch/CUDA/yolox availability without running anything")
    p3.set_defaults(func=cmd_check_env)

    return p


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except ConfigError as exc:
        LOG.error("CONFIG ERROR: %s", exc)
        return 2
    except ModuleNotFoundError as exc:
        LOG.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
