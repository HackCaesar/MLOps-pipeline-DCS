"""Live run tracking: ``runs/{run_id}/status.json`` + ``events.jsonl`` (+ SQLite).

Reliability policy (configurable via ``tracking:`` — see configs):

* **critical** ops are *strict* (raise when ``strict=True``) so the stage/run
  fails loudly instead of silently losing state: creating ``runs/{run_id}/``,
  writing ``config_snapshot.yaml`` + the initial ``status.json``, appending stage
  start/success/failed events, and the terminal ``finish()``.
* **iteration progress** is always *best-effort* — ``record_training_progress``
  and ``emit_progress`` never raise; a failed intermediate write logs a warning.
* the **SQLite row lifecycle** (create/finalize) is owned by the run_all stages
  (``create_pipeline_run`` / ``update_pipeline_run_status``) — those are strict
  by being stages. The column *mirrors* written here (current_stage/model_name/
  tile_cache_id) are best-effort, and the critical SQLite op (schema init/migration)
  lives in ``metadata_store.init_db`` / the ``init_metadata_db`` stage.

``status.json`` is updated by **merge** (read-modify-write), never rebuilt, so a
writer (e.g. the train subprocess) cannot clobber fields another writer set
(``dataset_id``, ``tile_cache_id``, ``model``, ``progress.stage_*``). Writes are
atomic (tmp + ``os.replace``). ``events.jsonl`` is strictly append-only. All
timestamps are UTC ISO ``YYYY-MM-DDTHH:MM:SSZ``. Pure-stdlib + pyyaml.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from packages.common.logging_utils import get_logger

LOG = get_logger(__name__)

STATUS_SCHEMA_VERSION = 1
EVENT_SCHEMA_VERSION = 1


def utc_now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class ModelInfo:
    name: Optional[str] = None
    num_classes: Optional[int] = None
    input_size: Optional[list] = None
    pretrained: Optional[str] = None


class StatusTracker:
    def __init__(
        self,
        run_id: str,
        runs_dir: str | Path,
        *,
        db_path: Optional[str | Path] = None,
        strict: bool = True,
        stage_total: int = 0,
        train_log_interval_iters: int = 10,
        train_log_interval_seconds: float = 5.0,
    ) -> None:
        self.run_id = run_id
        self.run_dir = Path(runs_dir) / run_id
        self.status_path = self.run_dir / "status.json"
        self.events_path = self.run_dir / "events.jsonl"
        self.db_path = str(db_path) if db_path else None
        self.strict = strict
        self.stage_total = stage_total
        self.train_log_interval_iters = max(1, int(train_log_interval_iters))
        self.train_log_interval_seconds = float(train_log_interval_seconds)
        self._lock = threading.Lock()

    @classmethod
    def from_config(cls, config: Mapping[str, Any], run_id: str, *,
                    strict: Optional[bool] = None, stage_total: int = 0) -> "StatusTracker":
        storage = config.get("storage") or {}
        from packages.common.paths import resolve_path
        runs_dir = resolve_path(storage.get("runs_dir")) or Path("runs")
        db_path = (config.get("metadata_store") or {}).get("sqlite_path")
        tcfg = config.get("tracking") or {}
        return cls(
            run_id, runs_dir, db_path=db_path, stage_total=stage_total,
            strict=tcfg.get("strict", True) if strict is None else strict,
            train_log_interval_iters=int(tcfg.get("train_log_interval_iters", 10)),
            train_log_interval_seconds=float(tcfg.get("train_log_interval_seconds", 5.0)),
        )

    # ---- critical lifecycle (strict) --------------------------------------

    def init_run(self, *, config: Optional[Mapping[str, Any]] = None,
                 dataset_id: Optional[str] = None, status: str = "running",
                 model: Optional[ModelInfo] = None) -> None:
        def _do() -> None:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            if config is not None:
                self._write_config_snapshot(config)
            seed = {
                "schema_version": STATUS_SCHEMA_VERSION,
                "run_id": self.run_id, "status": status, "current_stage": None,
                "dataset_id": dataset_id, "tile_cache_id": None,
                "model": asdict(model) if model else {"name": None, "num_classes": None,
                                                      "input_size": None, "pretrained": None},
                "progress": {"stage_index": 0, "stage_total": self.stage_total,
                             "epoch": None, "max_epoch": None, "remaining_epochs": None,
                             "iter": None, "max_iter": None, "stage_percent": None},
                "last_metrics": {"loss": None, "lr": None},
                "updated_at": utc_now_iso(),
            }
            with self._lock:
                self._atomic_write_json(self.status_path, seed)
            self._append_event(stage="run", status=status, message="run started")
        self._critical("init_run", _do)
        # Provenance manifest is best-effort (a git/torch probe must never fail the run).
        try:
            self._write_run_manifest(config)
        except Exception as exc:
            LOG.warning("status_tracker: run_manifest.json not written (ignored): %s", exc)

    def set_current_stage(self, stage: str, *, stage_index: int, stage_total: int) -> None:
        self.stage_total = stage_total
        self._critical("set_current_stage", lambda: self._merge_status(
            {"current_stage": stage,
             "progress": {"stage_index": stage_index, "stage_total": stage_total}}))
        self._sqlite_best_effort(current_stage=stage)

    def emit_stage(self, stage: str, status: str, *, current: Optional[int] = None,
                   total: Optional[int] = None, message: str = "",
                   payload: Optional[Mapping[str, Any]] = None) -> None:
        def _do() -> None:
            self._append_event(stage=stage, status=status, current=current, total=total,
                               message=message, payload=payload)
            changes: dict[str, Any] = {"status": status, "current_stage": stage}
            if current is not None and total:
                changes["progress"] = {"stage_percent": round(100.0 * current / total, 1)}
            self._merge_status(changes)
        self._critical("emit_stage", _do)

    def set_tile_cache(self, tile_cache_id: str) -> None:
        self._critical("set_tile_cache", lambda: self._merge_status({"tile_cache_id": tile_cache_id}))
        self._sqlite_best_effort(tile_cache_id=tile_cache_id)

    def set_model(self, model: ModelInfo) -> None:
        self._critical("set_model", lambda: self._merge_status({"model": asdict(model)}))
        self._sqlite_best_effort(model_name=model.name)

    def finish(self, *, status: str, error: Optional[str] = None) -> None:
        def _do() -> None:
            self._merge_status({"status": status})
            self._append_event(stage="finalize", status=status,
                               message=error or "", payload=({"error": error} if error else None))
        self._critical("finish", _do)
        self._sqlite_best_effort(status=status, finished_at=utc_now_iso(), error=error)

    # ---- best-effort training progress ------------------------------------

    def should_log_iter(self, iter_count: int, last_log_time: float, now: float) -> bool:
        """Pure throttle decision (tested independently of yolox)."""
        return (iter_count % self.train_log_interval_iters == 0
                or (now - last_log_time) >= self.train_log_interval_seconds)

    def record_training_progress(self, *, epoch: Optional[int] = None,
                                 max_epoch: Optional[int] = None,
                                 iter: Optional[int] = None, max_iter: Optional[int] = None,
                                 loss: Optional[float] = None, lr: Optional[float] = None) -> None:
        """BEST-EFFORT: never raises. Writes only the fields actually provided."""
        try:
            progress: dict[str, Any] = {}
            if epoch is not None:
                progress["epoch"] = epoch
            if max_epoch is not None:
                progress["max_epoch"] = max_epoch
            if epoch is not None and max_epoch is not None:
                progress["remaining_epochs"] = max(0, max_epoch - epoch)
            if iter is not None:
                progress["iter"] = iter
            if max_iter is not None:
                progress["max_iter"] = max_iter
            metrics: dict[str, Any] = {}
            if loss is not None:
                metrics["loss"] = loss
            if lr is not None:
                metrics["lr"] = lr
            changes: dict[str, Any] = {}
            if progress:
                changes["progress"] = progress
            if metrics:
                changes["last_metrics"] = metrics
            if changes:
                self._merge_status(changes)
        except Exception as exc:
            LOG.warning("status_tracker: training progress update failed (ignored): %s", exc)

    def emit_progress(self, stage: str, status: str = "running", *,
                      current: Optional[int] = None, total: Optional[int] = None,
                      message: str = "", payload: Optional[Mapping[str, Any]] = None) -> None:
        """BEST-EFFORT progress event: never raises."""
        try:
            self._append_event(stage=stage, status=status, current=current, total=total,
                               message=message, payload=payload)
        except Exception as exc:
            LOG.warning("status_tracker: progress event append failed (ignored): %s", exc)

    # ---- internals ---------------------------------------------------------

    def _critical(self, what: str, fn: Callable[[], None]) -> None:
        try:
            fn()
        except Exception as exc:
            if self.strict:
                raise
            LOG.warning("status_tracker: critical op %s failed (non-strict, ignored): %s", what, exc)

    def _read_status(self) -> dict[str, Any]:
        if self.status_path.is_file():
            try:
                return json.loads(self.status_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                LOG.warning("status_tracker: status.json unreadable, reseeding")
        return {}

    def _merge_status(self, changes: Mapping[str, Any]) -> None:
        with self._lock:
            merged = _deep_merge(self._read_status(), dict(changes))
            merged["updated_at"] = utc_now_iso()
            self._atomic_write_json(self.status_path, merged)

    def _append_event(self, *, stage: str, status: str, current: Optional[int] = None,
                      total: Optional[int] = None, message: str = "",
                      payload: Optional[Mapping[str, Any]] = None) -> None:
        event = {
            "event_schema_version": EVENT_SCHEMA_VERSION,
            "run_id": self.run_id, "stage": stage, "status": status,
            "current": current, "total": total, "message": message,
            "payload": dict(payload) if payload else {},
            "created_at": utc_now_iso(),
        }
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with open(self.events_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
            f.flush()

    def _write_config_snapshot(self, config: Mapping[str, Any]) -> None:
        import yaml
        snap = self.run_dir / "config_snapshot.yaml"
        tmp = snap.with_suffix(".yaml.tmp")
        tmp.write_text(yaml.safe_dump(dict(config), sort_keys=False, allow_unicode=True),
                       encoding="utf-8")
        os.replace(tmp, snap)

    def _write_run_manifest(self, config: Optional[Mapping[str, Any]]) -> None:
        """Write-once provenance: code/env versions + config hash for reproducibility."""
        import platform
        manifest = {
            "manifest_schema_version": 1,
            "run_id": self.run_id,
            "created_at": utc_now_iso(),
            "package_version": _dist_version("dcs-yolox-pipeline"),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "git_sha": _git_sha(),
            "torch_version": _dist_version("torch"),
            "cuda_version": _torch_cuda(),
            "yolox_version": _dist_version("yolox"),
            "tiling_algorithm_version": _tiling_algorithm_version(),
            "config_sha256": _config_sha(config) if config is not None else None,
        }
        self._atomic_write_json(self.run_dir / "run_manifest.json", manifest)

    def _atomic_write_json(self, path: Path, obj: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.parent / (path.name + ".tmp")
        tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)

    def _sqlite_best_effort(self, **fields: Any) -> None:
        """Mirror columns into pipeline_runs. Best-effort — the row lifecycle is
        owned by the create/update stages, so a missing row/DB just warns."""
        if not self.db_path:
            return
        try:
            from packages.common.metadata_store import SQLiteMetadataStore
            with SQLiteMetadataStore.open(self.db_path) as store:
                store.update_pipeline_run_status(self.run_id, **fields)
        except Exception as exc:
            LOG.warning("status_tracker: SQLite column mirror failed (ignored): %s", exc)


def _deep_merge(dst: dict, src: Mapping[str, Any]) -> dict:
    out = dict(dst)
    for k, v in src.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


# ---- provenance helpers (all graceful — return None rather than raise) ------

def _dist_version(name: str) -> Optional[str]:
    try:
        from importlib.metadata import PackageNotFoundError, version
        try:
            return version(name)
        except PackageNotFoundError:
            return None
    except Exception:
        return None


def _git_sha() -> Optional[str]:
    import subprocess
    repo_root = Path(__file__).resolve().parents[2]
    try:
        out = subprocess.run(["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        return (out.stdout.strip() or None) if out.returncode == 0 else None
    except Exception:
        return None


def _torch_cuda() -> Optional[str]:
    """CUDA version from an ALREADY-imported torch (no heavy import in the orchestrator)."""
    import sys
    m = sys.modules.get("torch")
    try:
        return getattr(getattr(m, "version", None), "cuda", None) if m else None
    except Exception:
        return None


def _tiling_algorithm_version() -> Optional[int]:
    try:
        from packages.common.hashing import TILING_ALGORITHM_VERSION
        return TILING_ALGORITHM_VERSION
    except Exception:
        return None


def _config_sha(config: Mapping[str, Any]) -> str:
    import hashlib
    blob = json.dumps(config, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()
