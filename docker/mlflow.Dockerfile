# MLflow tracking server with SQLite backend store + local artifact dir.
#
# Reachable as http://mlflow:5000 from other compose services and as
# http://localhost:5000 from the host. UI/API both serve on this port.

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl tini libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && pip install \
        "mlflow>=2.16,<3" \
        "pyyaml>=6.0"

WORKDIR /workspace
VOLUME ["/workspace/storage"]

EXPOSE 5000

# Backend + artifact paths point at the shared storage bind mount so MLflow
# state survives container rebuilds. Override via compose `command:` if needed.
ENTRYPOINT ["tini", "--", "mlflow"]
CMD ["server", \
     "--backend-store-uri", "sqlite:////workspace/storage/metadata/mlflow.db", \
     "--default-artifact-root", "/workspace/storage/mlflow_artifacts", \
     "--host", "0.0.0.0", \
     "--port", "5000"]
