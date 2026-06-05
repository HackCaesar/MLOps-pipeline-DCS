#!/usr/bin/env bash
# Install YOLOX editable on first start (if the bind mount has setup.py) and
# then exec the requested command. Subsequent starts skip the install because
# the .egg-link / installed metadata persist across compose runs.
set -euo pipefail

# YOLOX is now baked into /opt/yolox at image build time (see Dockerfile),
# so we don't need to install on container start. Kept as a fallback for the
# case where someone bind-mounts /workspace/YOLOX (legacy dev flow) and the
# baked install is missing.
if ! python -c "import yolox" >/dev/null 2>&1; then
    YOLOX_DIR="${YOLOX_DIR:-/workspace/YOLOX}"
    if [ -f "${YOLOX_DIR}/setup.py" ]; then
        echo "[entrypoint] Fallback: installing yolox from ${YOLOX_DIR}…" >&2
        pip install -v -e "${YOLOX_DIR}" --no-build-isolation 2>&1 | tail -20
    fi
fi

exec "$@"
