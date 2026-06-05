# YOLOX — vendored (partial copy)

This directory is a **partial, vendored copy** of [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX),
the anchor-free YOLO detector by Megvii.

- **Upstream:** https://github.com/Megvii-BaseDetection/YOLOX
- **Version:** `0.3.0` (see [`yolox/__init__.py`](yolox/__init__.py))
- **License:** Apache License 2.0 — Copyright (c) Megvii, Inc. and its affiliates.
  The full license text is preserved verbatim in [`LICENSE`](LICENSE).

## Why it is vendored

The GPU training and evaluation images bake YOLOX in at build time
(`COPY vendor/YOLOX /opt/yolox` + `pip install -e`). Vendoring keeps the Docker
builds **self-contained and reproducible** — they do not depend on GitHub being
reachable, and the exact YOLOX revision is pinned alongside the rest of the code.

## Changes from upstream

Per Apache-2.0 §4(b), the modifications relative to upstream `0.3.0` are:

- The YOLOX **source code is unmodified.** Everything required to
  `pip install -e .` and `import yolox` is byte-for-byte upstream:
  `yolox/`, `tools/`, `exps/`, `setup.py`, `setup.cfg`, `MANIFEST.in`,
  `requirements.txt`, and `LICENSE`.
- **Non-essential files were removed** to keep this repository lean (they are
  not needed to build or import the package): `assets/`, `demo/`, `docs/`,
  `tests/`, `datasets/`, `.github/`, `hubconf.py`, `SECURITY.md`,
  `.pre-commit-config.yaml`, `.readthedocs.yaml`, and `.gitignore`.
- **This `README.md` replaces** the upstream readme (whose images referenced the
  removed `assets/`).

For the complete project, demos, documentation, and assets, see the upstream
repository linked above. Third-party attribution for the whole project is in
[`THIRD_PARTY_NOTICES.md`](../../THIRD_PARTY_NOTICES.md).
