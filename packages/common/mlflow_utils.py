"""MLflow facade with a no-op fallback.

Design goal: every stage of the pipeline (train, enrich, evaluate, export) can
call ``MLflowFacade.from_config(cfg).start_run(run_name=...)`` unconditionally.
- If ``mlflow.enabled = true`` AND ``mlflow`` is importable → real logging.
- If ``mlflow.enabled = false`` OR ``mlflow`` is missing → all calls are no-ops,
  no exceptions, a warning is emitted once.

This means the pipeline always works the same whether or not MLflow is set up.
"""
from __future__ import annotations

from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from packages.common.logging_utils import get_logger

LOG = get_logger(__name__)


# ---------------------------------------------------------------------- #
# Facade
# ---------------------------------------------------------------------- #

class MLflowFacade:
    """Top-level entry point — built once per stage from the pipeline config."""

    def __init__(
        self,
        *,
        enabled: bool = False,
        tracking_uri: Optional[str] = None,
        experiment_name: Optional[str] = None,
        artifact_root: Optional[str] = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.artifact_root = artifact_root
        self._mlflow: Any = None

        if not self.enabled:
            return
        try:
            import mlflow
        except ImportError:
            LOG.warning(
                "mlflow.enabled=true but the mlflow package is not installed; "
                "logging will be a no-op. Install: pip install mlflow"
            )
            self.enabled = False
            return

        try:
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
            if experiment_name:
                mlflow.set_experiment(experiment_name)
        except Exception as exc:  # MLflow may raise if server isn't reachable
            LOG.warning("Could not configure mlflow (tracking_uri=%s experiment=%s): %s. "
                        "Falling back to no-op.", tracking_uri, experiment_name, exc)
            self.enabled = False
            return

        self._mlflow = mlflow

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "MLflowFacade":
        mlf = config.get("mlflow") or {}
        return cls(
            enabled=bool(mlf.get("enabled", False)),
            tracking_uri=mlf.get("tracking_uri"),
            experiment_name=mlf.get("experiment_name"),
            artifact_root=mlf.get("artifact_root"),
        )

    @classmethod
    def disabled(cls) -> "MLflowFacade":
        """Build a no-op facade explicitly (handy for tests)."""
        return cls(enabled=False)

    def start_run(
        self, *, run_name: Optional[str] = None,
        tags: Optional[Mapping[str, str]] = None,
        nested: bool = False,
    ) -> "MLflowRun":
        """Open a new run context. Use as ``with facade.start_run(...) as run: ...``."""
        if not self.enabled or self._mlflow is None:
            return _NoOpRun()
        return _RealRun(self._mlflow, run_name=run_name, tags=dict(tags or {}),
                        nested=nested)


# ---------------------------------------------------------------------- #
# Run protocol
# ---------------------------------------------------------------------- #

class MLflowRun(AbstractContextManager):
    """Run handle — supports context-manager + explicit log_* calls."""

    enabled: bool = False
    run_id: Optional[str] = None
    experiment_id: Optional[str] = None

    def __enter__(self) -> "MLflowRun":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        pass

    # All log_* methods accept dict/path-like values. No-op base implementations.

    def log_params(self, params: Mapping[str, Any]) -> None: ...
    def log_metrics(self, metrics: Mapping[str, Any], step: Optional[int] = None) -> None: ...
    def log_artifact(self, local_path: str | Path,
                      artifact_path: Optional[str] = None) -> None: ...
    def log_artifacts_dir(self, local_dir: str | Path,
                           artifact_path: Optional[str] = None) -> None: ...
    def log_dict(self, payload: Mapping[str, Any], artifact_file: str) -> None: ...
    def set_tag(self, key: str, value: str) -> None: ...
    def set_tags(self, tags: Mapping[str, str]) -> None: ...


class _NoOpRun(MLflowRun):
    enabled = False

    def log_params(self, params): pass
    def log_metrics(self, metrics, step=None): pass
    def log_artifact(self, local_path, artifact_path=None): pass
    def log_artifacts_dir(self, local_dir, artifact_path=None): pass
    def log_dict(self, payload, artifact_file): pass
    def set_tag(self, key, value): pass
    def set_tags(self, tags): pass


class _RealRun(MLflowRun):
    enabled = True

    def __init__(self, mlflow_mod: Any, *, run_name: Optional[str],
                 tags: Dict[str, str], nested: bool) -> None:
        self._mlflow = mlflow_mod
        self._run_name = run_name
        self._tags = tags
        self._nested = nested

    def __enter__(self) -> "_RealRun":
        active = self._mlflow.start_run(run_name=self._run_name,
                                         tags=self._tags or None,
                                         nested=self._nested)
        self.run_id = active.info.run_id
        self.experiment_id = active.info.experiment_id
        LOG.info("MLflow run started: run_id=%s experiment=%s",
                 self.run_id, self.experiment_id)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        status = "FINISHED"
        if exc_type is not None:
            status = "FAILED"
            try:
                self._mlflow.set_tag("status", "failed")
                self._mlflow.set_tag("error", str(exc)[:500])
            except Exception:  # noqa: BLE001 — best-effort tagging
                pass
        try:
            self._mlflow.end_run(status=status)
        except Exception as e:  # noqa: BLE001
            LOG.warning("mlflow.end_run failed: %s", e)

    # --- logging ---

    def log_params(self, params: Mapping[str, Any]) -> None:
        if not params:
            return
        clean = {str(k): _coerce_param(v) for k, v in params.items()}
        try:
            self._mlflow.log_params(clean)
        except Exception as e:  # noqa: BLE001
            LOG.warning("mlflow.log_params failed: %s", e)

    def log_metrics(self, metrics: Mapping[str, Any], step: Optional[int] = None) -> None:
        if not metrics:
            return
        clean: Dict[str, float] = {}
        for k, v in metrics.items():
            try:
                clean[str(k)] = float(v)
            except (TypeError, ValueError):
                LOG.debug("Skipping non-numeric metric %s=%r", k, v)
        if not clean:
            return
        try:
            self._mlflow.log_metrics(clean, step=step)
        except Exception as e:  # noqa: BLE001
            LOG.warning("mlflow.log_metrics failed: %s", e)

    def log_artifact(self, local_path: str | Path,
                      artifact_path: Optional[str] = None) -> None:
        p = Path(local_path)
        if not p.exists():
            LOG.debug("log_artifact: path missing, skipping: %s", p)
            return
        try:
            self._mlflow.log_artifact(str(p), artifact_path=artifact_path)
        except Exception as e:  # noqa: BLE001
            LOG.warning("mlflow.log_artifact(%s) failed: %s", p, e)

    def log_artifacts_dir(self, local_dir: str | Path,
                           artifact_path: Optional[str] = None) -> None:
        p = Path(local_dir)
        if not p.is_dir():
            LOG.debug("log_artifacts_dir: not a directory, skipping: %s", p)
            return
        try:
            self._mlflow.log_artifacts(str(p), artifact_path=artifact_path)
        except Exception as e:  # noqa: BLE001
            LOG.warning("mlflow.log_artifacts(%s) failed: %s", p, e)

    def log_dict(self, payload: Mapping[str, Any], artifact_file: str) -> None:
        try:
            self._mlflow.log_dict(dict(payload), artifact_file)
        except Exception as e:  # noqa: BLE001
            LOG.warning("mlflow.log_dict(%s) failed: %s", artifact_file, e)

    def set_tag(self, key: str, value: str) -> None:
        try:
            self._mlflow.set_tag(key, str(value))
        except Exception as e:  # noqa: BLE001
            LOG.warning("mlflow.set_tag(%s) failed: %s", key, e)

    def set_tags(self, tags: Mapping[str, str]) -> None:
        if not tags:
            return
        try:
            self._mlflow.set_tags({str(k): str(v) for k, v in tags.items()})
        except Exception as e:  # noqa: BLE001
            LOG.warning("mlflow.set_tags failed: %s", e)


# ---------------------------------------------------------------------- #
# small helpers
# ---------------------------------------------------------------------- #

def flatten_params(prefix: str, payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Flatten nested config into dotted-key dict suitable for MLflow params.

    MLflow rejects nested dicts; we flatten them to ``a.b.c``. Lists become
    strings (joined by ',') to avoid weird shape issues.
    """
    out: Dict[str, Any] = {}
    for k, v in payload.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, Mapping):
            out.update(flatten_params(key, v))
        elif isinstance(v, (list, tuple)):
            out[key] = ",".join(str(x) for x in v)
        else:
            out[key] = v
    return out


def _coerce_param(value: Any) -> str:
    """MLflow stores params as strings; do the conversion here to keep call sites tidy."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return str(value)
    if isinstance(value, (list, tuple)):
        return ",".join(str(x) for x in value)
    if isinstance(value, Mapping):
        return ",".join(f"{k}={v}" for k, v in value.items())
    return str(value)
