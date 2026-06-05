"""Windows DCS Runner Agent — entrypoint.

Starts the HTTP server + shared-folder watcher in background threads and blocks
on SIGINT. One ``CaptureRunner`` instance is shared between both intake paths
so concurrent submits are serialized.

Usage:
  python -m apps.windows_dcs_runner_agent.agent --config configs/dcs_agent.yaml
"""
from __future__ import annotations

import argparse
import signal
import sys
import threading
from typing import Optional

from apps.windows_dcs_runner_agent.api import make_server
from apps.windows_dcs_runner_agent.command_folder_watcher import CommandFolderWatcher
from apps.windows_dcs_runner_agent.config import AGENT_VERSION, AgentConfig
from apps.windows_dcs_runner_agent.process_manager import CaptureRunner
from apps.windows_dcs_runner_agent.status_writer import StatusStore
from packages.common.logging_utils import get_logger, setup_logging

LOG = get_logger(__name__)


def run_agent(cfg: AgentConfig) -> int:
    """Boot the agent and block until SIGINT. Returns CLI exit code."""
    LOG.info("Windows DCS Runner Agent v%s starting", AGENT_VERSION)
    LOG.info("HTTP %s:%d, commands_dir=%s, status_dir=%s",
             cfg.host, cfg.port, cfg.commands_dir, cfg.status_dir)

    if cfg.log_dir is not None:
        cfg.log_dir.mkdir(parents=True, exist_ok=True)

    status_store = StatusStore(status_dir=cfg.status_dir)
    runner = CaptureRunner(cfg, status_store)

    server = make_server(cfg.host, cfg.port, runner, status_store)
    http_thread = threading.Thread(target=server.serve_forever, daemon=True,
                                     name="agent-http-server")
    http_thread.start()
    LOG.info("HTTP server listening on http://%s:%d", cfg.host, cfg.port)

    watcher: Optional[CommandFolderWatcher] = None
    if cfg.commands_dir is not None:
        watcher = CommandFolderWatcher(
            runner=runner,
            commands_dir=cfg.commands_dir,
            poll_interval_sec=cfg.poll_interval_sec,
        )
        watcher.start()
        LOG.info("Command folder watcher started")

    stop_event = threading.Event()

    def _on_sigint(signum, frame):  # noqa: ANN001  — signal handler signature
        LOG.info("Caught signal %d, shutting down", signum)
        stop_event.set()

    signal.signal(signal.SIGINT, _on_sigint)
    try:
        signal.signal(signal.SIGTERM, _on_sigint)
    except (ValueError, AttributeError):
        # SIGTERM is unavailable on Windows from non-main threads etc.
        pass

    try:
        stop_event.wait()
    finally:
        LOG.info("Shutting down HTTP server")
        server.shutdown()
        if watcher is not None:
            watcher.stop()
        # If a capture is in flight, allow a short grace period to flush status.
        if runner.is_busy():
            active = runner.active_run_id()
            LOG.warning("Capture %s still active during shutdown", active)
            if active:
                runner.stop(active)
            runner.wait_for_completion(timeout=10.0)
    LOG.info("Agent stopped")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, help="path to dcs_agent.yaml")
    p.add_argument("--port", type=int, default=None,
                   help="override agent.http.port from config")
    p.add_argument("--host", default=None,
                   help="override agent.http.host from config")
    args = p.parse_args(argv)

    setup_logging()
    cfg = AgentConfig.from_file(args.config)
    if args.port is not None:
        cfg = _override(cfg, port=int(args.port))
    if args.host is not None:
        cfg = _override(cfg, host=args.host)
    return run_agent(cfg)


def _override(cfg: AgentConfig, **fields) -> AgentConfig:
    from dataclasses import replace
    return replace(cfg, **fields)


if __name__ == "__main__":
    sys.exit(main())
