"""Unit tests for packages.common.mlflow_utils.

Most tests exercise the no-op path (mlflow not installed / disabled). When
mlflow is installed we run a few additional tests against a local file URI.
"""
from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from packages.common.mlflow_utils import (
    MLflowFacade,
    MLflowRun,
    _NoOpRun,
    flatten_params,
)


def _mlflow_available() -> bool:
    try:
        import mlflow  # noqa: F401
        return True
    except ImportError:
        return False


# ---- facade construction ----------------------------------------------

def test_disabled_facade_is_noop() -> None:
    f = MLflowFacade.disabled()
    assert f.enabled is False
    with f.start_run(run_name="anything") as run:
        assert isinstance(run, _NoOpRun)
        assert run.enabled is False
        # All log_* calls must NOT raise.
        run.log_params({"a": 1})
        run.log_metrics({"x": 0.5}, step=1)
        run.log_artifact("/no/such/path")
        run.log_artifacts_dir("/no/such/dir")
        run.log_dict({"key": "value"}, "test.json")
        run.set_tag("k", "v")
        run.set_tags({"a": "b", "c": "d"})


def test_from_config_enabled_false() -> None:
    f = MLflowFacade.from_config({"mlflow": {"enabled": False}})
    assert f.enabled is False
    with f.start_run() as run:
        assert run.enabled is False


def test_from_config_missing_mlflow_section() -> None:
    """If the whole mlflow section is missing, default to disabled."""
    f = MLflowFacade.from_config({})
    assert f.enabled is False


def test_from_config_enabled_without_mlflow_package(monkeypatch) -> None:
    """If enabled=true but mlflow can't be imported, fall back to no-op."""
    if _mlflow_available():
        pytest.skip("mlflow is installed — can't simulate ImportError")
    f = MLflowFacade.from_config({"mlflow": {"enabled": True,
                                              "tracking_uri": "http://localhost:5000",
                                              "experiment_name": "test"}})
    # __init__ catches ImportError → enabled becomes False
    assert f.enabled is False
    with f.start_run() as run:
        assert isinstance(run, _NoOpRun)


# ---- log_* on real run via mock mlflow --------------------------------

def _install_fake_mlflow(monkeypatch) -> "SimpleNamespace":
    """Install a fake ``mlflow`` module in sys.modules and return the recorder."""
    calls = SimpleNamespace(
        start_run=[], end_run=[], log_params=[], log_metrics=[],
        log_artifact=[], log_artifacts=[], log_dict=[],
        set_tag=[], set_tags=[], set_tracking_uri=[], set_experiment=[],
    )

    class _ActiveRun:
        def __init__(self):
            self.info = SimpleNamespace(run_id="fake-run-id",
                                          experiment_id="fake-exp-id")

    fake = SimpleNamespace()
    fake.set_tracking_uri = lambda uri: calls.set_tracking_uri.append(uri)
    fake.set_experiment   = lambda name: calls.set_experiment.append(name)
    def _start_run(run_name=None, tags=None, nested=False):
        calls.start_run.append({"run_name": run_name, "tags": tags, "nested": nested})
        return _ActiveRun()
    fake.start_run    = _start_run
    fake.end_run      = lambda status=None: calls.end_run.append(status)
    fake.log_params   = lambda p: calls.log_params.append(dict(p))
    fake.log_metrics  = lambda m, step=None: calls.log_metrics.append({"metrics": dict(m), "step": step})
    fake.log_artifact = lambda path, artifact_path=None: calls.log_artifact.append(
        {"path": path, "artifact_path": artifact_path})
    fake.log_artifacts = lambda path, artifact_path=None: calls.log_artifacts.append(
        {"path": path, "artifact_path": artifact_path})
    fake.log_dict     = lambda d, file: calls.log_dict.append({"data": dict(d), "file": file})
    fake.set_tag      = lambda k, v: calls.set_tag.append({"k": k, "v": v})
    fake.set_tags     = lambda t: calls.set_tags.append(dict(t))

    monkeypatch.setitem(sys.modules, "mlflow", fake)
    return calls


def test_real_run_lifecycle_logs(monkeypatch) -> None:
    calls = _install_fake_mlflow(monkeypatch)
    f = MLflowFacade(enabled=True, tracking_uri="http://localhost:5000",
                     experiment_name="exp")
    assert calls.set_tracking_uri == ["http://localhost:5000"]
    assert calls.set_experiment == ["exp"]
    assert f.enabled is True

    with f.start_run(run_name="my_run", tags={"a": "b"}) as run:
        assert run.enabled is True
        assert run.run_id == "fake-run-id"
        run.log_params({"max_epoch": 100, "lr": 0.01})
        run.log_metrics({"loss": 0.5, "ap50": 0.8}, step=1)
        run.set_tag("status", "running")

    assert calls.start_run == [
        {"run_name": "my_run", "tags": {"a": "b"}, "nested": False}
    ]
    assert calls.log_params and "max_epoch" in calls.log_params[0]
    assert calls.log_metrics and calls.log_metrics[0]["step"] == 1
    assert calls.end_run == ["FINISHED"]


def test_real_run_records_failure_on_exception(monkeypatch) -> None:
    calls = _install_fake_mlflow(monkeypatch)
    f = MLflowFacade(enabled=True, tracking_uri="x", experiment_name="y")
    with pytest.raises(RuntimeError):
        with f.start_run(run_name="boom"):
            raise RuntimeError("kaboom")
    assert calls.end_run == ["FAILED"]
    tag_keys = [t["k"] for t in calls.set_tag]
    assert "status" in tag_keys


def test_real_run_skips_non_numeric_metrics(monkeypatch) -> None:
    calls = _install_fake_mlflow(monkeypatch)
    f = MLflowFacade(enabled=True, tracking_uri="x", experiment_name="y")
    with f.start_run() as run:
        run.log_metrics({"good": 0.5, "bad": "not_a_number", "empty": None})
    assert calls.log_metrics
    logged = calls.log_metrics[0]["metrics"]
    assert "good" in logged
    assert "bad" not in logged
    assert "empty" not in logged


def test_real_run_log_artifact_skips_missing(monkeypatch, tmp_path) -> None:
    calls = _install_fake_mlflow(monkeypatch)
    f = MLflowFacade(enabled=True, tracking_uri="x", experiment_name="y")
    with f.start_run() as run:
        run.log_artifact(tmp_path / "no_such_file.pth")
    assert calls.log_artifact == []  # silently skipped, no crash


def test_real_run_log_artifact_real_file(monkeypatch, tmp_path) -> None:
    calls = _install_fake_mlflow(monkeypatch)
    f = MLflowFacade(enabled=True, tracking_uri="x", experiment_name="y")
    p = tmp_path / "thing.txt"
    p.write_text("hello")
    with f.start_run() as run:
        run.log_artifact(p, artifact_path="weights")
    assert calls.log_artifact == [{"path": str(p), "artifact_path": "weights"}]


def test_real_run_log_artifacts_dir(monkeypatch, tmp_path) -> None:
    calls = _install_fake_mlflow(monkeypatch)
    f = MLflowFacade(enabled=True, tracking_uri="x", experiment_name="y")
    d = tmp_path / "reports"
    d.mkdir()
    (d / "metrics.json").write_text("{}")
    with f.start_run() as run:
        run.log_artifacts_dir(d, artifact_path="reports")
    assert calls.log_artifacts == [{"path": str(d), "artifact_path": "reports"}]


def test_set_tracking_uri_failure_falls_back_to_noop(monkeypatch) -> None:
    """If mlflow.set_tracking_uri raises, facade gracefully disables itself."""
    fake = SimpleNamespace(
        set_tracking_uri=lambda uri: (_ for _ in ()).throw(ConnectionError("no server")),
        set_experiment=lambda *a, **kw: None,
    )
    monkeypatch.setitem(sys.modules, "mlflow", fake)
    f = MLflowFacade(enabled=True, tracking_uri="http://nowhere:5000",
                     experiment_name="x")
    assert f.enabled is False


# ---- flatten_params ----------------------------------------------------

def test_flatten_params_flat_dict() -> None:
    out = flatten_params("training", {"max_epoch": 100, "lr": 0.01})
    assert out == {"training.max_epoch": 100, "training.lr": 0.01}


def test_flatten_params_nested() -> None:
    out = flatten_params("enrichment", {
        "crop_size": 640,
        "background_policy": {"resize_to": [640, 640], "method": "letterbox"},
    })
    assert "enrichment.crop_size" in out
    assert "enrichment.background_policy.method" in out
    assert out["enrichment.background_policy.method"] == "letterbox"


def test_flatten_params_list_becomes_string() -> None:
    out = flatten_params("eval", {"iou_thresholds": [0.5, 0.6, 0.7]})
    assert out["eval.iou_thresholds"] == "0.5,0.6,0.7"


def test_flatten_params_empty_prefix() -> None:
    out = flatten_params("", {"a": 1, "b": {"c": 2}})
    assert "a" in out and "b.c" in out


# ---- contract for type Protocol --------------------------------------

def test_noop_run_is_subclass_of_mlflow_run() -> None:
    assert issubclass(_NoOpRun, MLflowRun)


def test_real_run_starts_with_facade_disabled_returns_noop() -> None:
    f = MLflowFacade(enabled=False)
    with f.start_run(run_name="x") as run:
        assert isinstance(run, _NoOpRun)
