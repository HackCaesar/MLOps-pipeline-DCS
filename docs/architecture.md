# Architecture

A single MLOps system, orchestrated by a Python CLI (no Airflow, no web UI).

```
┌─────────────────────────── Windows host ───────────────────────────┐
│  DCS World  ──▶  dcs_capture_host  ──▶  windows_dcs_runner_agent     │
└───────────────────────────────┬─────────────────────────────────────┘
                                 │ HTTP (agent_client)         writes
                                 ▼                               ▼
                        apps/dcs_capture (adapter)   STORAGE_ROOT/datasets/raw/{dataset_id}
                                 │
   ┌─────────────────────────────┼──────────────────────────────────────┐
   │            dockerized pipeline (compose, run --rm)                   │
   │                                                                      │
   │  register_dataset ─▶ build_or_reuse_tile_cache ─▶ train ─▶ evaluate  │
   │       │                      │                      │         │      │
   │   SQLite registry      cache/tiles/...        runs/{run_id}   reports│
   │                                                      │               │
   │                                                  export ONNX         │
   │                                                      │               │
   │                                                   MLflow ◀───────────┘
   └──────────────────────────────────────────────────────────────────────┘
                                 ▲
                          apps/pipeline/cli (orchestrator + Rich TUI, --watch)
```

## Components

| Module | Responsibility |
|---|---|
| `apps/dcs_capture_host` | host-side DCS World capture (missions, projection, export) |
| `apps/windows_dcs_runner_agent` | host HTTP agent that drives capture + reports status |
| `apps/dcs_capture` | ingest boundary: normalize DCS output → COCO raw; HTTP client to the agent |
| `apps/dataset_enrichment` | multi-scale tiling into a reusable, content-addressed tile cache |
| `apps/yolox_training` | YOLOX-S `Exp` build + `CustomTrainer` (progress hooks) |
| `apps/evaluation_export` | inference, metrics, full-image merge, ONNX export, reporting |
| `apps/pipeline` | top-level orchestrator CLI + terminal TUI |
| `packages/common` | shared libs: coco_io, tiling, letterbox, metadata_store (SQLite), mlflow_utils, paths, run_id, hashing*, status_tracker* |
| `experiments/yolox` | `class_obj_at_sea` Exp (3 classes, 640×640, multiscale off) |
| `vendor/YOLOX` | vendored upstream YOLOX, baked into GPU images |

`*` `hashing` and `status_tracker` land in Stages 3–4.

## Identity separation

- `dataset_id` — a canonical raw dataset under `datasets/raw/`.
- `tile_cache_id` = `{dataset_hash}__{split_hash}__{tile_config_hash}` — a derived
  tile dataset, reused across runs with the same data + split + tiling config.
- `run_id` — one pipeline execution; stores only results + references, never a
  copy of the dataset.

## Non-goals

No Airflow. No web dashboard / Streamlit / FastAPI UI for monitoring (the agent's
HTTP API is for capture control only). Monitoring is terminal-based (Rich TUI).
