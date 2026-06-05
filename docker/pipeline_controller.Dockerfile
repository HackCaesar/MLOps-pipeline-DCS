# Pipeline orchestration controller (lightweight CPU image).
#
# This container runs the top-level CLI commands and the DCS agent client.
# It does NOT need PyTorch / YOLOX / TensorRT — those live in their own
# specialized containers. Heavy stages (training, evaluation, export) are
# kicked off via `docker compose run --rm <service>` from this container's
# orchestration scripts.

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/workspace/pipeline

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl tini \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && pip install \
        "pyyaml>=6.0"

WORKDIR /workspace/pipeline

# Source code is bind-mounted from the host at compose time, so we don't COPY it
# here. This keeps the image small and the dev loop fast. For an immutable
# production image, switch to COPY apps/ packages/ configs/ scripts/ ./
# and rebuild on every change.

ENTRYPOINT ["tini", "--", "python"]
CMD ["-m", "apps.pipeline.cli", "--help"]
