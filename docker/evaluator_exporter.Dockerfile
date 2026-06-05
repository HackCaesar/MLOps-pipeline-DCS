# Evaluation + ONNX/TensorRT export (GPU CUDA + PyTorch + ONNX + TRT).
#
# Reuses the NGC PyTorch base (same CUDA/cuDNN as yolox_trainer) so PyTorch ↔
# ONNX ↔ TensorRT all see a consistent runtime. The NGC image bundles TensorRT
# already; we add onnxruntime-gpu + pycuda + matplotlib.

FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/workspace/pipeline \
    MPLBACKEND=Agg \
    DEBIAN_FRONTEND=noninteractive

# Same C++ toolchain as yolox_trainer (YOLOX FastCOCOEvalOp builds at install).
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential libgl1 libglib2.0-0 libgomp1 git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && pip install \
        "pyyaml>=6.0" \
        "mlflow>=2.16" \
        "matplotlib>=3.8" \
        "tabulate>=0.9" \
        "tqdm>=4.66" \
        "loguru>=0.7" \
        "ninja>=1.11" \
        "numpy>=1.26,<2" \
        "Pillow>=10.0" \
        "opencv-python-headless>=4.9" \
        "pycocotools>=2.0.7" \
        "onnx>=1.16" \
        "onnxruntime-gpu>=1.18"
# Note: tensorrt + pycuda removed for the slim pytorch/pytorch base. If you
# need TRT export, switch back to nvcr.io/nvidia/pytorch:24.xx and re-add
# tensorrt + pycuda.

# Same baked-in YOLOX as yolox_trainer so eval doesn't pay install cost.
COPY vendor/YOLOX /opt/yolox
RUN pip install -v -e /opt/yolox --no-build-isolation 2>&1 | tail -10 \
    && python -c "import yolox; print('yolox', yolox.__version__, 'baked in OK')"

WORKDIR /workspace/pipeline
VOLUME ["/workspace/storage"]

COPY docker/yolox_entrypoint.sh /usr/local/bin/yolox-entrypoint
RUN chmod +x /usr/local/bin/yolox-entrypoint

ENTRYPOINT ["/usr/local/bin/yolox-entrypoint"]
CMD ["python", "-m", "apps.evaluation_export.cli", "--help"]
