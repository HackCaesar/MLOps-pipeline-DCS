# Third-Party Notices

This project redistributes the third-party components listed below. Each remains
the property of its respective authors and is licensed under its own terms. The
project's own code is licensed under Apache-2.0 (see [`LICENSE`](LICENSE)).

## YOLOX

|              |                                                                    |
| ------------ | ------------------------------------------------------------------ |
| Component    | YOLOX — anchor-free YOLO object detector                           |
| Version      | 0.3.0                                                              |
| Location     | [`vendor/YOLOX/`](vendor/YOLOX/) (partial vendored copy)          |
| Source       | https://github.com/Megvii-BaseDetection/YOLOX                      |
| Copyright    | Copyright (c) 2021-2022 Megvii Inc. and its affiliates.            |
| License      | Apache License 2.0 — [`vendor/YOLOX/LICENSE`](vendor/YOLOX/LICENSE) |

**Modifications:** Non-essential files (`assets/`, `demo/`, `docs/`, `tests/`,
`datasets/`, `.github/`, `hubconf.py`, and CI/lint config) were removed to reduce
size, and the upstream `README.md` was replaced. The YOLOX *source* is otherwise
unmodified. The exact change list is in
[`vendor/YOLOX/README.md`](vendor/YOLOX/README.md).

**Usage:** YOLOX is installed (editable) into the GPU training and evaluation
Docker images — see [`docker/yolox_trainer.Dockerfile`](docker/yolox_trainer.Dockerfile)
and [`docker/evaluator_exporter.Dockerfile`](docker/evaluator_exporter.Dockerfile).
