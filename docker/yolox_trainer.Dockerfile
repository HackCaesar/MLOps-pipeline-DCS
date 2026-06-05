# YOLOX training (GPU CUDA + PyTorch).
#
# Base image: NVIDIA NGC PyTorch container — bundles a CUDA-matched PyTorch +
# Apex + cuDNN + NCCL. Bumping the year/month tag is the standard upgrade.
#
# yolox is vendored at vendor/YOLOX and baked into the image (editable install)
# at build time — no bind mount needed.

FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/workspace/pipeline \
    MPLBACKEND=Agg \
    DEBIAN_FRONTEND=noninteractive

# YOLOX setup.py pre-compiles a C++ extension (FastCOCOEvalOp) on Linux, so we
# need a C++ toolchain in the image. libgl1 + libglib2.0-0 keep opencv happy.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential libgl1 libglib2.0-0 libgomp1 git \
    && rm -rf /var/lib/apt/lists/*

# YOLOX has runtime deps (opencv, pycocotools, ...) declared in its setup.py;
# we still add the ones our CLI needs directly: pyyaml + mlflow + matplotlib.
# Pillow/numpy come from the NGC image.
RUN pip install --upgrade pip && pip install \
        "pyyaml>=6.0" \
        "mlflow>=2.16" \
        "matplotlib>=3.8" \
        "tabulate>=0.9" \
        "tqdm>=4.66" \
        "loguru>=0.7" \
        "thop>=0.1.1" \
        "ninja>=1.11" \
        "numpy>=1.26,<2" \
        "Pillow>=10.0" \
        "opencv-python-headless>=4.9" \
        "pycocotools>=2.0.7"

# Bake YOLOX into the image so every `compose run --rm` doesn't pay the
# ~5-min editable-install + C++ ext compile cost. The source is COPY'd from
# the build context, NOT the bind mount, so the install survives container
# restarts. Patched requirements.txt drops onnx-simplifier==0.4.10 (Py3.11 incompat).
COPY vendor/YOLOX /opt/yolox
RUN pip install -v -e /opt/yolox --no-build-isolation 2>&1 | tail -10 \
    && python -c "import yolox; print('yolox', yolox.__version__, 'baked in OK')"

WORKDIR /workspace/pipeline
VOLUME ["/workspace/storage"]

# Lightweight entrypoint — no install needed any more.
COPY docker/yolox_entrypoint.sh /usr/local/bin/yolox-entrypoint
RUN chmod +x /usr/local/bin/yolox-entrypoint

ENTRYPOINT ["/usr/local/bin/yolox-entrypoint"]
CMD ["python", "-m", "apps.yolox_training.cli", "check-env"]
