"""Structural validation of docker-compose.yml + Dockerfiles (new, Airflow-free stack).

Can't `docker build` in CI/WSL, but we CAN check:
  - compose parses; the 5 expected services exist; NO Airflow;
  - every service references an existing Dockerfile;
  - storage bind mount (host ${STORAGE_ROOT} -> /workspace/storage) on every service;
  - GPU services use the pytorch/pytorch CUDA base, reserve an nvidia device, and
    BAKE vendored YOLOX (no /workspace/YOLOX bind mount any more);
  - pipeline-controller reaches the Windows agent (host.docker.internal:8765);
  - MLFLOW_TRACKING_URI consistent; each Dockerfile has FROM + ENTRYPOINT/CMD.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

PIPELINE_ROOT = Path(__file__).resolve().parents[2]
COMPOSE_PATH = PIPELINE_ROOT / "docker-compose.yml"
DOCKER_DIR = PIPELINE_ROOT / "docker"

SERVICES = {"mlflow", "pipeline-controller", "dataset-processor",
            "yolox-trainer", "evaluator-exporter"}
GPU_SERVICES = {"yolox-trainer", "evaluator-exporter"}
MLFLOW_CLIENTS = {"pipeline-controller", "dataset-processor", "yolox-trainer", "evaluator-exporter"}
DOCKERFILES = ["pipeline_controller.Dockerfile", "dataset_processor.Dockerfile",
               "yolox_trainer.Dockerfile", "evaluator_exporter.Dockerfile", "mlflow.Dockerfile"]


@pytest.fixture(scope="module")
def compose() -> dict:
    assert COMPOSE_PATH.is_file(), f"missing {COMPOSE_PATH}"
    data = yaml.safe_load(COMPOSE_PATH.read_text(encoding="utf-8"))
    assert isinstance(data, dict) and "services" in data
    return data


def _svc(compose: dict, name: str) -> dict:
    return compose["services"][name]


# ---- top-level structure + no Airflow ---------------------------------

def test_exactly_the_expected_services(compose: dict) -> None:
    assert set(compose["services"].keys()) == SERVICES


def test_no_airflow_anywhere(compose: dict) -> None:
    assert not any("airflow" in s for s in compose["services"]), "Airflow service still in compose"
    assert not (DOCKER_DIR / "airflow.Dockerfile").exists(), "airflow.Dockerfile should be gone"
    for svc in compose["services"].values():
        for v in (svc.get("volumes") or []):
            assert "/opt/airflow" not in v, "stray Airflow dags mount"


# ---- Dockerfiles -------------------------------------------------------

@pytest.mark.parametrize("dockerfile_name", DOCKERFILES)
def test_dockerfile_exists_from_and_cmd(dockerfile_name: str) -> None:
    p = DOCKER_DIR / dockerfile_name
    assert p.is_file(), f"missing {p}"
    upper = p.read_text(encoding="utf-8").upper()
    assert "FROM " in upper, f"{dockerfile_name} has no FROM"
    assert ("ENTRYPOINT" in upper) or ("CMD" in upper), f"{dockerfile_name} has no ENTRYPOINT/CMD"


def test_gpu_dockerfiles_use_pytorch_cuda_base() -> None:
    for df in ("yolox_trainer.Dockerfile", "evaluator_exporter.Dockerfile"):
        text = (DOCKER_DIR / df).read_text()
        assert "pytorch/pytorch" in text and "cuda" in text.lower(), f"{df} not on a CUDA pytorch base"


def test_gpu_dockerfiles_bake_vendored_yolox() -> None:
    for df in ("yolox_trainer.Dockerfile", "evaluator_exporter.Dockerfile"):
        text = (DOCKER_DIR / df).read_text()
        assert "COPY vendor/YOLOX" in text, f"{df} should bake vendor/YOLOX (no bind mount)"


def test_cpu_dockerfiles_use_slim_base() -> None:
    for df in ("mlflow.Dockerfile", "pipeline_controller.Dockerfile", "dataset_processor.Dockerfile"):
        assert "python:3.11-slim" in (DOCKER_DIR / df).read_text(), f"{df} not on python:3.11-slim"


def test_dataset_processor_has_image_libs() -> None:
    p = (DOCKER_DIR / "dataset_processor.Dockerfile").read_text()
    for lib in ("numpy", "Pillow", "opencv-python-headless", "pycocotools", "matplotlib"):
        assert lib in p, f"dataset_processor missing dep: {lib}"


def test_evaluator_exporter_has_onnx_libs() -> None:
    p = (DOCKER_DIR / "evaluator_exporter.Dockerfile").read_text()
    for lib in ("onnx", "onnxruntime-gpu"):
        assert lib in p, f"evaluator_exporter missing dep: {lib}"


def test_mlflow_dockerfile_serves_5000() -> None:
    p = (DOCKER_DIR / "mlflow.Dockerfile").read_text()
    assert "EXPOSE 5000" in p
    assert "--backend-store-uri" in p and "--default-artifact-root" in p


def test_yolox_entrypoint_script() -> None:
    p = DOCKER_DIR / "yolox_entrypoint.sh"
    text = p.read_text(encoding="utf-8")
    assert text.startswith("#!") and 'exec "$@"' in text
    assert "pip install" in text and "/workspace/YOLOX" in text   # legacy bind-mount fallback


# ---- per-service compose checks ---------------------------------------

@pytest.mark.parametrize("name", sorted(SERVICES))
def test_service_mounts_storage(compose: dict, name: str) -> None:
    volumes = _svc(compose, name).get("volumes") or []
    assert any(":/workspace/storage" in v for v in volumes), f"{name} missing /workspace/storage mount"


def test_storage_mount_uses_storage_root_env(compose: dict) -> None:
    vols = _svc(compose, "mlflow").get("volumes") or []
    assert any("STORAGE_ROOT" in v and ":/workspace/storage" in v for v in vols), \
        "storage host path should come from ${STORAGE_ROOT}"


@pytest.mark.parametrize("name", sorted(GPU_SERVICES))
def test_gpu_service_reserves_nvidia(compose: dict, name: str) -> None:
    deploy = _svc(compose, name).get("deploy", {})
    devices = (((deploy.get("resources") or {}).get("reservations") or {}).get("devices") or [])
    assert any(d.get("driver") == "nvidia" for d in devices), f"{name} missing nvidia reservation"


@pytest.mark.parametrize("name", sorted(GPU_SERVICES))
def test_gpu_service_sets_shm_size(compose: dict, name: str) -> None:
    # PyTorch DataLoader workers need a real /dev/shm; the 64 MB Docker default
    # crashes them with a bus error at "init prefetcher".
    assert _svc(compose, name).get("shm_size"), f"{name} must set shm_size for DataLoader workers"


@pytest.mark.parametrize("name", sorted(GPU_SERVICES))
def test_gpu_service_does_not_bind_yolox(compose: dict, name: str) -> None:
    # YOLOX is baked into the image now — there must be NO /workspace/YOLOX bind mount.
    volumes = _svc(compose, name).get("volumes") or []
    assert not any(":/workspace/YOLOX" in v for v in volumes), f"{name} should not bind /workspace/YOLOX"


def test_pipeline_controller_points_to_agent(compose: dict) -> None:
    env = _svc(compose, "pipeline-controller").get("environment") or {}
    url = env.get("DCS_AGENT_URL", "")
    assert "host.docker.internal" in url and url.endswith(":8765")
    extra = _svc(compose, "pipeline-controller").get("extra_hosts") or []
    assert any("host.docker.internal:host-gateway" in h for h in extra)


def test_mlflow_published_on_5000(compose: dict) -> None:
    ports = _svc(compose, "mlflow").get("ports") or []
    assert any(str(p).startswith("5000:5000") for p in ports)


def test_mlflow_tracking_uri_consistent(compose: dict) -> None:
    for name in MLFLOW_CLIENTS:
        env = _svc(compose, name).get("environment") or {}
        assert env.get("MLFLOW_TRACKING_URI") == "http://mlflow:5000", f"{name} wrong MLFLOW_TRACKING_URI"


def test_every_service_references_existing_dockerfile(compose: dict) -> None:
    for name, svc in compose["services"].items():
        build = svc.get("build")
        if isinstance(build, dict) and build.get("dockerfile"):
            df_path = PIPELINE_ROOT / build["dockerfile"]
            assert df_path.is_file(), f"{name} references missing Dockerfile {df_path}"


# ---- .dockerignore ----------------------------------------------------

def test_dockerignore_excludes_junk_and_artifacts() -> None:
    text = (PIPELINE_ROOT / ".dockerignore").read_text(encoding="utf-8")
    for pat in ("__pycache__", ".pytest_cache", ".venv/", "vendor/YOLOX/.git/", "*.pth"):
        assert pat in text, f".dockerignore missing pattern: {pat}"
