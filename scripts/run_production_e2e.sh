#!/usr/bin/env bash
# End-to-end production run inside Docker containers.
#
# Stages (in order):
#   1. dataset-processor: adapter-normalize places ephemeral staging capture from
#                          /workspace/storage/datasets/_staging/dcs_caucasus into
#                          /workspace/storage/datasets/raw/dcs_caucasus_100 via hardlink
#                          (fallback copy across filesystems) — no permanent intermediate copy
#   2. dataset-processor: validate_raw_dataset on the split COCO
#   3. dataset-processor: init metadata DB + register_dataset + create pipeline_run
#   4. dataset-processor: build-cache → /workspace/storage/cache/tiles/{tile_cache_id} (reused if present)
#   5. yolox-trainer:     train YOLOX with MLflow autologging
#   6. evaluator-exporter: evaluate PyTorch on val split (best_ckpt.pth)
#   7. evaluator-exporter: export-onnx
#   8. dataset-processor: register_artifacts
#   9. dataset-processor: pipeline_run_status update → success

set -euo pipefail

CONFIG=configs/pipeline_production.yaml
RUN_ID=prod_2026_05_29_001
DATASET_ID=dcs_caucasus_100

cd "$(dirname "$0")/.."

echo "================ STAGE 1/9: adapter ================"
docker compose run --rm dataset-processor python -m apps.dcs_capture.cli adapter-normalize \
    --config $CONFIG \
    --source-dir /workspace/storage/datasets/_staging/dcs_caucasus \
    --dataset-id $DATASET_ID \
    --name "DCS Caucasus 100-frame production capture" \
    --dcs-config /workspace/pipeline/configs/caucasus_static_neutral_capture.yaml \
    --overwrite \
    --cleanup-staging

echo "================ STAGE 2/9: validate raw ================"
docker compose run --rm dataset-processor python -m scripts.validate_raw_dataset \
    --config $CONFIG \
    --dataset-dir /workspace/storage/datasets/raw/$DATASET_ID

echo "================ STAGE 3a/9: init metadata DB ================"
docker compose run --rm dataset-processor python -m scripts.init_metadata_db --config $CONFIG --quiet

echo "================ STAGE 3b/9: register dataset ================"
docker compose run --rm dataset-processor python -m scripts.register_dataset \
    --config $CONFIG \
    --dataset-dir /workspace/storage/datasets/raw/$DATASET_ID \
    --replace

echo "================ STAGE 3c/9: create pipeline_run ================"
docker compose run --rm dataset-processor python -m scripts.pipeline_run_status create \
    --config $CONFIG \
    --run-id $RUN_ID \
    --dataset-id $DATASET_ID \
    --config-path $CONFIG \
    --status running \
    --replace

echo "================ STAGE 4/9: build-or-reuse tile cache ================"
docker compose run --rm dataset-processor python -m apps.dataset_enrichment.cli build-cache \
    --config $CONFIG \
    --run-id $RUN_ID

echo "================ STAGE 5/9: train YOLOX ================"
docker compose run --rm yolox-trainer python -m apps.yolox_training.cli train \
    --config $CONFIG \
    --run-id $RUN_ID

echo "================ STAGE 6/9: evaluate (pytorch) ================"
docker compose run --rm evaluator-exporter python -m apps.evaluation_export.cli evaluate \
    --config $CONFIG \
    --split val \
    --backend pytorch \
    --checkpoint /workspace/storage/artifacts/checkpoints/$RUN_ID/best_ckpt.pth \
    --exp-module experiments.yolox.class_obj_at_sea \
    --device cuda \
    --output-dir /workspace/storage/artifacts/reports/$RUN_ID/val

echo "================ STAGE 7/9: ONNX export ================"
docker compose run --rm evaluator-exporter python -m apps.evaluation_export.cli export-onnx \
    --config $CONFIG \
    --checkpoint /workspace/storage/artifacts/checkpoints/$RUN_ID/best_ckpt.pth \
    --exp-module experiments.yolox.class_obj_at_sea \
    --output /workspace/storage/artifacts/exports/$RUN_ID/model.onnx \
    --opset 13

echo "================ STAGE 8/9: register artifacts ================"
docker compose run --rm dataset-processor python -m scripts.register_artifacts \
    --config $CONFIG --run-id $RUN_ID --replace

echo "================ STAGE 9/9: finalize pipeline_run ================"
docker compose run --rm dataset-processor python -m scripts.pipeline_run_status update \
    --config $CONFIG --run-id $RUN_ID --status success --auto-finished-at

echo ""
echo "==================== ALL STAGES OK ===================="
