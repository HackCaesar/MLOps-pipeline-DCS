# docker/

5 services, 5 Dockerfiles. Source-of-truth = `../docker-compose.yml`. There is
**no Airflow** in this stack — orchestration is done by `apps.pipeline.cli`.

## Quick start

```bash
# Build everything (first time only)
docker compose build

# Long-running service: MLflow tracking UI
docker compose up -d mlflow                # http://localhost:5000

# One-off CLI runs (each stage in its own --rm container)
docker compose run --rm pipeline-controller \
    python -m scripts.init_metadata_db --config configs/pipeline.yaml

docker compose run --rm dataset-processor \
    python -m apps.dataset_enrichment.cli build --config configs/pipeline.yaml

docker compose run --rm yolox-trainer \
    python -m apps.yolox_training.cli train --config configs/pipeline.yaml \
    --run-id 2026_05_25_001

docker compose run --rm evaluator-exporter \
    python -m apps.evaluation_export.cli evaluate \
    --config configs/pipeline.yaml --backend pytorch \
    --checkpoint /workspace/storage/runs/2026_05_25_001/checkpoints/best_ckpt.pth \
    --exp-module experiments.yolox.class_obj_at_sea \
    --device cuda --fp16
```

## Service matrix

| Service               | GPU | Base image                          | Heavy deps                       | Use case |
|-----------------------|:---:|-------------------------------------|----------------------------------|----------|
| `mlflow`              |  ✗  | `python:3.11-slim`                  | mlflow                           | tracking server (port 5000) |
| `pipeline-controller` |  ✗  | `python:3.11-slim`                  | pyyaml + rich                    | top-level CLI orchestration / TUI |
| `dataset-processor`   |  ✗  | `python:3.11-slim`                  | numpy/Pillow/opencv/pycocotools  | adapter, enrichment, validation |
| `yolox-trainer`       |  ✓  | `pytorch/pytorch:2.8.0-cuda12.8`    | torch/CUDA + yolox (baked)       | training |
| `evaluator-exporter`  |  ✓  | `pytorch/pytorch:2.8.0-cuda12.8`    | torch + onnx + onnxruntime-gpu   | evaluation + ONNX export |

The two GPU services share a base image so PyTorch ↔ ONNX see a consistent
CUDA / cuDNN runtime. YOLOX is baked from `vendor/YOLOX` at build time.

## Storage layout (host `${STORAGE_ROOT}` → container `/workspace/storage`)

Storage lives **outside** the repo (default `D:\MLOps_storage`, WSL
`/mnt/d/MLOps_storage`). Set `STORAGE_ROOT` in `.env` (see `../.env.example`).

```
${STORAGE_ROOT}/
├── datasets/raw/{dataset_id}/...                 # single canonical raw dataset
├── cache/tiles/{dataset_hash}__{split_hash}__{tile_config_hash}/   # reusable tile cache
├── runs/{run_id}/{status.json,events.jsonl,logs,checkpoints,metrics,exports,reports}
├── metadata/pipeline.db                          # SQLite registry
├── mlflow_artifacts/                             # MLflow artifact store
└── logs/
```

There is no `storage/tmp/...` and no `storage/shared/dcs_source` any more.

## Windows DCS Runner Agent (NOT in compose)

DCS World + the capture agent run natively on the Windows host
(see `../docs/dcs_capture.md`). Containers reach the agent via
`host.docker.internal:8765`. The `extra_hosts: ["host.docker.internal:host-gateway"]`
in compose makes plain-Linux Docker daemons resolve it too.

## GPU smoke check

```bash
docker compose run --rm yolox-trainer python -m apps.yolox_training.cli check-env
# Expected: torch True / CUDA available True / device count >= 1 / yolox True
```

If `CUDA available: False`, install the NVIDIA Container Toolkit or enable GPU
support in Docker Desktop. See `../scripts/check_gpu.py`.

## Common operations

| Task                            | Command |
|---------------------------------|---------|
| Rebuild after a dep change      | `docker compose build <service>` |
| Tail MLflow logs                | `docker compose logs -f mlflow` |
| Drop into a controller shell    | `docker compose run --rm pipeline-controller bash` |
| Stop everything                 | `docker compose down` |
