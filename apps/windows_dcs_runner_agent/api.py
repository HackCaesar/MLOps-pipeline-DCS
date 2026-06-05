"""HTTP API for the Windows DCS Runner Agent — stdlib ``http.server`` based.

Endpoints:

  GET  /health                       → {"status": "ok", "agent_version": ..., "busy": bool}
  POST /capture/start                body = CaptureSpec dict (see process_manager.py)
                                     → {"accepted": bool, "run_id": ..., "reason": ...?}
  GET  /capture/status/{run_id}      → CaptureStatus dict
  POST /capture/stop/{run_id}        → {"stopped": bool}

Errors return JSON with ``{"error": "..."}`` and a non-200 code.

We use ``ThreadingHTTPServer`` so multiple status polls can run concurrently
without blocking each other. The capture runner itself is single-threaded — it
serializes captures regardless of how many HTTP threads call ``submit``.
"""
from __future__ import annotations

import json
import re
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from apps.windows_dcs_runner_agent.config import AGENT_VERSION
from apps.windows_dcs_runner_agent.process_manager import CaptureRunner, CaptureSpec
from apps.windows_dcs_runner_agent.status_writer import StatusStore
from packages.common.logging_utils import get_logger

LOG = get_logger(__name__)


_STATUS_PATH_RE = re.compile(r"^/capture/status/([A-Za-z0-9_.\-]+)/?$")
_STOP_PATH_RE   = re.compile(r"^/capture/stop/([A-Za-z0-9_.\-]+)/?$")


def make_handler_class(runner: CaptureRunner, status_store: StatusStore):
    """Closure-based handler factory. ``BaseHTTPRequestHandler`` is constructed
    per-request by the server, so we bind dependencies via closure."""

    class _Handler(BaseHTTPRequestHandler):

        server_version = f"DCSRunnerAgent/{AGENT_VERSION}"

        # ---- response helpers ----

        def _send_json(self, status: int, payload: Any) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_json_body(self) -> Any:
            length = int(self.headers.get("Content-Length") or "0")
            if length <= 0:
                return None
            raw = self.rfile.read(length)
            return json.loads(raw.decode("utf-8"))

        def log_message(self, fmt: str, *args: Any) -> None:  # type: ignore[override]
            LOG.info("HTTP %s — " + fmt, self.address_string(), *args)

        # ---- routes ----

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                self._send_json(HTTPStatus.OK, _health_payload(runner))
                return
            m = _STATUS_PATH_RE.match(self.path)
            if m:
                run_id = m.group(1)
                status = status_store.get(run_id)
                if status is None:
                    self._send_json(HTTPStatus.NOT_FOUND,
                                     {"error": f"unknown run_id: {run_id}"})
                    return
                self._send_json(HTTPStatus.OK, status.to_dict())
                return
            self._send_json(HTTPStatus.NOT_FOUND,
                             {"error": f"unknown route: {self.path}"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path == "/capture/start":
                try:
                    body = self._read_json_body()
                    spec = CaptureSpec.from_dict(body or {})
                except (ValueError, json.JSONDecodeError) as exc:
                    self._send_json(HTTPStatus.BAD_REQUEST,
                                     {"error": f"invalid CaptureSpec: {exc}"})
                    return

                accepted = runner.submit(spec)
                if not accepted:
                    self._send_json(HTTPStatus.CONFLICT,
                                     {"accepted": False,
                                      "reason": "agent busy",
                                      "active_run_id": runner.active_run_id()})
                    return
                self._send_json(HTTPStatus.ACCEPTED,
                                 {"accepted": True, "run_id": spec.run_id})
                return

            m = _STOP_PATH_RE.match(self.path)
            if m:
                run_id = m.group(1)
                stopped = runner.stop(run_id)
                self._send_json(HTTPStatus.OK if stopped else HTTPStatus.NOT_FOUND,
                                 {"stopped": stopped})
                return

            self._send_json(HTTPStatus.NOT_FOUND,
                             {"error": f"unknown route: {self.path}"})

    return _Handler


def _health_payload(runner: CaptureRunner) -> dict:
    return {
        "status":        "ok",
        "agent_version": AGENT_VERSION,
        "busy":          runner.is_busy(),
        "active_run_id": runner.active_run_id(),
    }


def make_server(host: str, port: int,
                runner: CaptureRunner, status_store: StatusStore) -> ThreadingHTTPServer:
    handler_cls = make_handler_class(runner, status_store)
    return ThreadingHTTPServer((host, port), handler_cls)
