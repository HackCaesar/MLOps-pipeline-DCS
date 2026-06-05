"""Run one DCS capture at a time, then run the adapter-normalize step.

This is the workhorse of the agent. ``CaptureRunner`` accepts one ``CaptureSpec``
at a time (others are rejected with ``False``), spawns the DCS_AutoDataset
orchestrator + the adapter as subprocesses, and updates ``StatusStore`` at every
phase. Subprocess invocation is injected so tests can mock it.
"""
from __future__ import annotations

import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Mapping, Optional, Sequence

from apps.windows_dcs_runner_agent.config import AgentConfig
from apps.windows_dcs_runner_agent.status_writer import (
    CaptureStatus,
    StatusStore,
    utc_now_iso,
)
from packages.common.logging_utils import get_logger

LOG = get_logger(__name__)


@dataclass(frozen=True)
class CaptureSpec:
    """A capture command as received by the agent."""

    run_id: str
    dataset_id: str
    mission_path:           Path     # .miz file given to DCS
    dcs_config_path:        Path     # DCS_AutoDataset config.yaml
    dcs_source_dataset_dir: Path     # where DCS_AutoDataset writes its output
    target_dataset_dir:     Path     # where the adapter normalizes to
    num_frames:             int = 1
    timeout_sec:            int = 7200

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "CaptureSpec":
        required = ("run_id", "dataset_id", "mission_path",
                    "dcs_config_path", "dcs_source_dataset_dir",
                    "target_dataset_dir")
        missing = [f for f in required if f not in raw]
        if missing:
            raise ValueError(f"CaptureSpec missing required fields: {missing}")
        return cls(
            run_id=str(raw["run_id"]),
            dataset_id=str(raw["dataset_id"]),
            mission_path=Path(str(raw["mission_path"])),
            dcs_config_path=Path(str(raw["dcs_config_path"])),
            dcs_source_dataset_dir=Path(str(raw["dcs_source_dataset_dir"])),
            target_dataset_dir=Path(str(raw["target_dataset_dir"])),
            num_frames=int(raw.get("num_frames", 1)),
            timeout_sec=int(raw.get("timeout_sec", 7200)),
        )


# ProcessRunner: a callable that runs a command and returns CompletedProcess.
# Default = subprocess.run with check=True. Tests inject a mock.
ProcessRunner = Callable[..., subprocess.CompletedProcess]


class CaptureRunner:
    """Holds state for ONE active capture; rejects concurrent submits."""

    def __init__(
        self,
        agent_config: AgentConfig,
        status_store: StatusStore,
        *,
        process_runner: Optional[ProcessRunner] = None,
        process_terminator: Optional[Callable[[], None]] = None,
    ) -> None:
        self.cfg = agent_config
        self.status_store = status_store
        self._lock = threading.Lock()
        self._active_run_id: Optional[str] = None
        self._active_thread: Optional[threading.Thread] = None
        self._stop_flag = threading.Event()
        self._process_runner: ProcessRunner = process_runner or subprocess.run
        self._process_terminator = process_terminator

    # ---- public API ----

    def is_busy(self) -> bool:
        with self._lock:
            return self._active_run_id is not None

    def active_run_id(self) -> Optional[str]:
        with self._lock:
            return self._active_run_id

    def submit(self, spec: CaptureSpec) -> bool:
        """Accept a capture. Returns True if accepted, False if agent is busy."""
        with self._lock:
            if self._active_run_id is not None:
                return False
            self._active_run_id = spec.run_id
        self._stop_flag.clear()
        self.status_store.upsert(CaptureStatus(
            run_id=spec.run_id,
            status="queued",
            phase="queued",
            agent_log_path=str(self._log_path_for(spec.run_id))
            if self.cfg.log_dir else None,
        ))
        t = threading.Thread(target=self._run, args=(spec,), daemon=True,
                              name=f"capture-{spec.run_id}")
        with self._lock:
            self._active_thread = t
        t.start()
        return True

    def stop(self, run_id: str) -> bool:
        """Request a graceful stop of a running capture. Returns True on success."""
        with self._lock:
            if self._active_run_id != run_id:
                return False
        self._stop_flag.set()
        if self._process_terminator is not None:
            try:
                self._process_terminator()
            except Exception:  # noqa: BLE001
                LOG.exception("process_terminator raised")
        self.status_store.update(run_id, status="stopped", phase="stopped",
                                  error="Stopped by user")
        return True

    def wait_for_completion(self, timeout: Optional[float] = None) -> bool:
        """Block until the active capture finishes. Returns False if a timeout fires."""
        with self._lock:
            t = self._active_thread
        if t is None:
            return True
        t.join(timeout=timeout)
        return not t.is_alive()

    # ---- internal ----

    def _run(self, spec: CaptureSpec) -> None:
        try:
            self.status_store.update(spec.run_id, status="running",
                                      started_at=utc_now_iso(),
                                      phase="launching_dcs")
            if self._stop_flag.is_set():
                return

            self._run_orchestrator(spec)
            if self._stop_flag.is_set():
                self.status_store.update(spec.run_id, status="stopped",
                                          error="Stopped during DCS capture")
                return

            self.status_store.update(spec.run_id, phase="normalizing")
            self._run_adapter(spec)
            if self._stop_flag.is_set():
                self.status_store.update(spec.run_id, status="stopped",
                                          error="Stopped during normalize")
                return

            self.status_store.update(spec.run_id, status="success",
                                      phase="done",
                                      raw_dataset_dir=str(spec.target_dataset_dir))

        except subprocess.TimeoutExpired:
            self.status_store.update(spec.run_id, status="failed",
                                      error=f"Subprocess timed out after {spec.timeout_sec}s")
        except subprocess.CalledProcessError as exc:
            msg = f"Subprocess exit {exc.returncode}"
            if exc.stderr:
                msg += f": {(exc.stderr if isinstance(exc.stderr, str) else exc.stderr.decode(errors='ignore'))[:300]}"
            self.status_store.update(spec.run_id, status="failed", error=msg)
        except Exception as exc:  # noqa: BLE001
            LOG.exception("CaptureRunner._run failed")
            self.status_store.update(spec.run_id, status="failed",
                                      error=str(exc)[:500])
        finally:
            with self._lock:
                self._active_run_id = None
                self._active_thread = None

    def _run_orchestrator(self, spec: CaptureSpec) -> None:
        cmd: List[str] = [
            self.cfg.python_executable, "-m", self.cfg.orchestrator_module,
            "--config",   str(spec.dcs_config_path),
            "--mission",  str(spec.mission_path),
            "--frames",   str(spec.num_frames),
            "--frame-id", spec.run_id,
        ]
        LOG.info("Running orchestrator: %s", " ".join(cmd))
        self._run_subprocess(
            cmd, cwd=self.cfg.orchestrator_cwd, timeout=spec.timeout_sec,
            log_label=f"orchestrator/{spec.run_id}",
        )

    def _run_adapter(self, spec: CaptureSpec) -> None:
        pipeline_cfg = self.cfg.pipeline_config_path or Path("configs/pipeline.yaml")
        cmd: List[str] = [
            self.cfg.python_executable, "-m", self.cfg.adapter_module,
            "adapter-normalize",
            "--config",     str(pipeline_cfg),
            "--source-dir", str(spec.dcs_source_dataset_dir),
            "--dataset-id", spec.dataset_id,
            "--overwrite",
        ]
        LOG.info("Running adapter: %s", " ".join(cmd))
        self._run_subprocess(
            cmd, cwd=self.cfg.adapter_cwd, timeout=spec.timeout_sec,
            log_label=f"adapter/{spec.run_id}",
        )

    def _run_subprocess(
        self,
        cmd: Sequence[str],
        *,
        cwd: Optional[Path],
        timeout: int,
        log_label: str,
    ) -> None:
        log_path = self._log_path_for(log_label.replace("/", "_"))
        if log_path:
            log_path.parent.mkdir(parents=True, exist_ok=True)
        # Default subprocess.run; the test injects a mock.
        try:
            if log_path:
                with log_path.open("a", encoding="utf-8") as log_file:
                    log_file.write(f"\n=== {log_label} @ {utc_now_iso()} ===\n")
                    log_file.write(f"CMD: {' '.join(cmd)}\n")
                    log_file.flush()
                    self._process_runner(
                        list(cmd),
                        cwd=str(cwd) if cwd else None,
                        stdout=log_file, stderr=subprocess.STDOUT,
                        timeout=timeout, check=True,
                    )
            else:
                self._process_runner(
                    list(cmd),
                    cwd=str(cwd) if cwd else None,
                    capture_output=True, text=True,
                    timeout=timeout, check=True,
                )
        except FileNotFoundError as exc:
            # Re-raise with a clearer message
            raise subprocess.CalledProcessError(
                returncode=127, cmd=cmd,
                stderr=f"Command not found: {exc}",
            ) from exc

    def _log_path_for(self, label: str) -> Optional[Path]:
        if self.cfg.log_dir is None:
            return None
        safe = "".join(c if c.isalnum() or c in "_-." else "_" for c in label)
        return self.cfg.log_dir / f"agent_{safe}.log"
