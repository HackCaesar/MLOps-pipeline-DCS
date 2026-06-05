# Docker

Five services. No Airflow. Detailed reference: [`../docker/README.md`](../docker/README.md).

| Service | GPU | Lifecycle | Purpose |
|---|:---:|---|---|
| `mlflow` | ✗ | long-running (`up -d`) | tracking server, port 5000 |
| `pipeline-controller` | ✗ | `run --rm` | orchestration CLI + TUI |
| `dataset-processor` | ✗ | `run --rm` | adapter / enrichment / validation |
| `yolox-trainer` | ✓ | `run --rm` | training |
| `evaluator-exporter` | ✓ | `run --rm` | evaluation + ONNX export |

## Storage & env

Storage lives outside the repo. Set `STORAGE_ROOT` in `.env` (see
[`../.env.example`](../.env.example)); it is bind-mounted to `/workspace/storage`
in every container.

```
host ${STORAGE_ROOT}  →  container /workspace/storage
  Windows: D:\MLOps_storage        WSL: /mnt/d/MLOps_storage
```

## Build & run

```bash
cp .env.example .env
docker compose build
docker compose up -d mlflow
docker compose run --rm yolox-trainer python -m apps.yolox_training.cli check-env
```

## YOLOX

YOLOX is vendored at `vendor/YOLOX` (no nested `.git`) and baked into the two GPU
images at build time via `COPY vendor/YOLOX /opt/yolox` + editable install — no
bind mount, no per-run install cost.
