# Dataset adapter + enrichment + COCO validation (CPU).
#
# Used to run:
#   python -m apps.dcs_capture.cli adapter-normalize ...
#   python -m apps.dataset_enrichment.cli build ...
#   python -m scripts.validate_raw_dataset ...
#   python -m scripts.register_dataset ...
#   python -m apps.evaluation_export.cli evaluate --backend mock ...   (no model needed)
#
# CPU-only image — the heavy image/array work uses numpy + opencv-headless +
# Pillow. No torch / yolox / TensorRT here.

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/workspace/pipeline

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl tini \
        libglib2.0-0 libgomp1 libgl1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && pip install \
        "pyyaml>=6.0" \
        "numpy>=1.26" \
        "Pillow>=10.0" \
        "opencv-python-headless>=4.9" \
        "pycocotools>=2.0.7" \
        "matplotlib>=3.8"

WORKDIR /workspace/pipeline
VOLUME ["/workspace/storage"]

ENTRYPOINT ["tini", "--", "python"]
CMD ["-m", "apps.dataset_enrichment.cli", "--help"]
