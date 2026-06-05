"""Evaluation + export CLI: ``evaluate``, ``evaluate-from-predictions``, ``export-onnx``, ``export-trt``."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from packages.common.config import ConfigError, load_config
from packages.common.logging_utils import get_logger, setup_logging
from packages.common.mlflow_utils import MLflowFacade, flatten_params
from packages.common.paths import resolve_path

LOG = get_logger(__name__)


def _resolve_classes(cfg: dict) -> list[str]:
    classes_path = (cfg.get("data") or {}).get("class_names_path") or "configs/classes.yaml"
    classes_path = resolve_path(classes_path)
    if classes_path and classes_path.is_file():
        import yaml
        data = yaml.safe_load(classes_path.read_text(encoding="utf-8")) or {}
        cats = data.get("categories") or []
        if cats:
            return [c["name"] for c in cats]
    from packages.common.classes import CLASS_NAMES
    return list(CLASS_NAMES)


def _resolve_dataset_dir(cfg: dict) -> Path:
    raw = (cfg.get("data") or {}).get("raw_dataset_dir")
    if not raw:
        raise SystemExit("data.raw_dataset_dir is required for evaluation")
    p = resolve_path(raw)
    assert p is not None
    return p


def _resolve_output_dir(cfg: dict, override: str | None) -> Path:
    if override:
        p = resolve_path(override)
        assert p is not None
        return p
    reports_dir = (cfg.get("evaluation") or {}).get("reports_dir")
    if not reports_dir:
        raise SystemExit("evaluation.reports_dir is required (or pass --output-dir)")
    p = resolve_path(reports_dir)
    assert p is not None
    return p


def cmd_evaluate(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    from apps.evaluation_export.evaluate import evaluate
    from apps.evaluation_export.inference import build_predictor

    eval_cfg = cfg.get("evaluation") or {}
    dataset_dir = _resolve_dataset_dir(cfg)
    classes = _resolve_classes(cfg)
    output_dir = _resolve_output_dir(cfg, args.output_dir)

    backend = args.backend or "pytorch"
    predictor_kwargs: dict = {}
    if backend == "mock":
        if not args.predictions:
            LOG.error("--predictions is required for backend=mock")
            return 2
        predictor_kwargs["predictions_path"] = args.predictions
    elif backend == "pytorch":
        if not args.checkpoint:
            LOG.error("--checkpoint is required for backend=pytorch")
            return 2
        if not (args.exp_file or args.exp_module):
            LOG.error("backend=pytorch requires either --exp-file or --exp-module")
            return 2
        predictor_kwargs.update({
            "checkpoint":      args.checkpoint,
            "exp_file":        args.exp_file,
            "exp_module":      args.exp_module,
            "num_classes":     len(classes),
            "input_size":      tuple((eval_cfg.get("crop_size", 640),) * 2),
            "device":          args.device,
            "conf_threshold":  float(args.predictor_conf_threshold),
            "nms_threshold":   float(args.predictor_nms_threshold),
            "fp16":            bool(args.fp16),
            "batch_size":      int(args.batch_size),
        })
    elif backend == "onnx":
        if not args.onnx:
            LOG.error("--onnx is required for backend=onnx")
            return 2
        predictor_kwargs.update({
            "onnx_path":      args.onnx,
            "num_classes":    len(classes),
            "input_size":     tuple((eval_cfg.get("crop_size", 640),) * 2),
            "device":         args.device,
            "conf_threshold": float(args.predictor_conf_threshold),
            "nms_threshold":  float(args.predictor_nms_threshold),
            "batch_size":     int(args.batch_size),
        })
    elif backend == "trt":
        if not args.engine:
            LOG.error("--engine is required for backend=trt")
            return 2
        predictor_kwargs.update({
            "engine_path":    args.engine,
            "num_classes":    len(classes),
            "input_size":     tuple((eval_cfg.get("crop_size", 640),) * 2),
            "conf_threshold": float(args.predictor_conf_threshold),
            "nms_threshold":  float(args.predictor_nms_threshold),
            "batch_size":     int(args.batch_size),
        })

    try:
        predictor = build_predictor(backend, **predictor_kwargs)
    except (ModuleNotFoundError, FileNotFoundError, ValueError) as exc:
        LOG.error("Cannot build predictor: %s", exc)
        return 1

    run_id_for_tag = (cfg.get("pipeline") or {}).get("run_id") or "ad-hoc"
    mlflow = MLflowFacade.from_config(cfg)
    with mlflow.start_run(
        run_name=f"{run_id_for_tag}_evaluate_{args.split}",
        tags={"stage": "evaluation", "backend": backend, "split": args.split,
              "run_id": run_id_for_tag},
    ) as mlrun:
        mlrun.log_params(flatten_params("evaluation", eval_cfg))
        mlrun.log_params({
            "backend":          backend,
            "split":            args.split,
            "num_classes":      len(classes),
            "checkpoint":       args.checkpoint or "",
            "onnx":             args.onnx or "",
            "engine":           args.engine or "",
            "predictions":      args.predictions or "",
        })

        try:
            result = evaluate(
                predictor=predictor,
                raw_dataset_dir=dataset_dir,
                split=args.split,
                classes=classes,
                eval_cfg=eval_cfg,
                output_dir=output_dir,
                backend_name=backend,
                model_path=args.checkpoint,
                predictions_path=args.predictions,
                mlflow_run=mlrun,
            )
        except NotImplementedError as exc:
            LOG.error("%s", exc)
            return 1

    LOG.info("Reports written to: %s", result.output_dir)
    return 0


def cmd_export_onnx(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    classes = _resolve_classes(cfg)
    from apps.evaluation_export.export import export_to_onnx

    run_id = (cfg.get("pipeline") or {}).get("run_id") or "ad-hoc"
    mlflow = MLflowFacade.from_config(cfg)
    with mlflow.start_run(
        run_name=f"{run_id}_export_onnx",
        tags={"stage": "export_onnx", "run_id": run_id},
    ) as mlrun:
        mlrun.log_params({
            "checkpoint":   args.checkpoint,
            "opset":        args.opset,
            "dynamic_axes": args.dynamic_axes,
            "num_classes":  len(classes),
            "exp_module":   args.exp_module or "",
            "exp_file":     args.exp_file or "",
        })
        try:
            result = export_to_onnx(
                checkpoint=args.checkpoint,
                exp_file=args.exp_file,
                exp_module=args.exp_module,
                output=args.output,
                num_classes=len(classes),
                input_size=tuple((cfg.get("evaluation", {}).get("crop_size", 640),) * 2),
                opset=int(args.opset),
                dynamic_axes=bool(args.dynamic_axes),
                device=args.device,
                sanity_atol=float(args.sanity_atol),
            )
        except (ModuleNotFoundError, FileNotFoundError, ValueError) as exc:
            LOG.error("export-onnx failed: %s", exc)
            return 1
        mlrun.log_metrics({
            "export/onnx/size_mb":       result.file_size_bytes / 1e6,
            "export/onnx/max_abs_diff":  result.max_abs_diff if result.max_abs_diff == result.max_abs_diff else 0.0,
            "export/onnx/mean_abs_diff": result.mean_abs_diff if result.mean_abs_diff == result.mean_abs_diff else 0.0,
        })
        mlrun.log_artifact(result.onnx_path, artifact_path="exports/onnx")
        mlrun.log_artifact(result.report_path, artifact_path="exports/onnx")
        summary = {
            "onnx_path": str(result.onnx_path),
            "report_path": str(result.report_path),
            "size_mb": round(result.file_size_bytes / 1e6, 2),
            "sha256_prefix": result.sha256[:16],
            "checker_ok": result.onnx_checker_ok,
            "max_abs_diff": result.max_abs_diff,
            "mean_abs_diff": result.mean_abs_diff,
            "opset": result.opset,
        }
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def cmd_export_trt(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    from apps.evaluation_export.export import build_tensorrt_engine

    run_id = (cfg.get("pipeline") or {}).get("run_id") or "ad-hoc"
    mlflow = MLflowFacade.from_config(cfg)
    with mlflow.start_run(
        run_name=f"{run_id}_export_trt_{args.precision}",
        tags={"stage": "export_trt", "precision": args.precision, "run_id": run_id},
    ) as mlrun:
        mlrun.log_params({
            "onnx":         args.onnx,
            "precision":    args.precision,
            "workspace_mb": args.workspace_mb,
            "calib_images": args.calib_images or "",
            "calib_cache":  args.calib_cache or "",
        })
        try:
            result = build_tensorrt_engine(
                onnx_path=args.onnx,
                output=args.output,
                precision=args.precision,
                calib_images_dir=args.calib_images,
                calib_cache=args.calib_cache,
                workspace_mb=int(args.workspace_mb),
                input_size=tuple((cfg.get("evaluation", {}).get("crop_size", 640),) * 2),
            )
        except (ModuleNotFoundError, FileNotFoundError, ValueError, RuntimeError) as exc:
            LOG.error("export-trt failed: %s", exc)
            return 1
        mlrun.log_metrics({
            "export/trt/size_mb":            result.file_size_bytes / 1e6,
            "export/trt/calibration_images": float(result.calibration_image_count),
        })
        mlrun.log_artifact(result.engine_path, artifact_path=f"exports/trt_{result.precision}")
        mlrun.log_artifact(result.report_path, artifact_path=f"exports/trt_{result.precision}")
        summary = {
            "engine_path": str(result.engine_path),
            "report_path": str(result.report_path),
            "precision":   result.precision,
            "size_mb":     round(result.file_size_bytes / 1e6, 2),
            "sha256_prefix": result.sha256[:16],
            "workspace_mb": result.workspace_mb,
            "calibration": {
                "image_count": result.calibration_image_count,
                "cache":       str(result.calibration_cache_used) if result.calibration_cache_used else None,
            },
        }
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="apps.evaluation_export.cli",
        description="Full-source-first evaluation + ONNX/TRT export (export = Phase 9 stub).",
    )
    sub = p.add_subparsers(dest="cmd", required=True, metavar="<subcommand>")

    pe = sub.add_parser("evaluate", help="evaluate a model (or mock predictions) on a raw-dataset split")
    pe.add_argument("--config", required=True)
    pe.add_argument("--split", default="test", choices=["train", "val", "test"])
    pe.add_argument("--backend", default="mock",
                    choices=["mock", "pytorch", "onnx", "trt"],
                    help="mock = read pre-baked predictions from a JSON file (no model)")
    pe.add_argument("--predictions", default=None,
                    help="for backend=mock: path to raw tile predictions JSON")
    pe.add_argument("--checkpoint", default=None,
                    help="for backend=pytorch: path to .pth weights")
    pe.add_argument("--onnx", default=None,
                    help="for backend=onnx: path to .onnx file")
    pe.add_argument("--engine", default=None,
                    help="for backend=trt: path to .engine file")
    pe.add_argument("--exp-file", default=None,
                    help="for backend=pytorch: path to yolox Exp .py file")
    pe.add_argument("--exp-module", default=None,
                    help="for backend=pytorch: importable Exp module (e.g. experiments.yolox.class_obj_at_sea)")
    pe.add_argument("--device", default=None,
                    help='for backend=pytorch: "cuda", "cuda:0", "cpu" (auto if omitted)')
    pe.add_argument("--fp16", action="store_true",
                    help="for backend=pytorch: half-precision (requires GPU)")
    pe.add_argument("--batch-size", type=int, default=8,
                    help="for backend=pytorch: inference batch size")
    pe.add_argument("--predictor-conf-threshold", type=float, default=0.01,
                    help="conf threshold INSIDE yolox postprocess (different from merge.confidence_threshold)")
    pe.add_argument("--predictor-nms-threshold", type=float, default=0.65,
                    help="NMS IoU threshold INSIDE yolox postprocess")
    pe.add_argument("--output-dir", default=None)
    pe.set_defaults(func=cmd_evaluate)

    po = sub.add_parser("export-onnx", help="export .pth → ONNX (requires torch+onnx)")
    po.add_argument("--config", required=True)
    po.add_argument("--checkpoint", required=True)
    po.add_argument("--output", required=True, help="path to write .onnx file")
    po.add_argument("--exp-file", default=None)
    po.add_argument("--exp-module", default="experiments.yolox.class_obj_at_sea")
    po.add_argument("--opset", type=int, default=11)
    po.add_argument("--dynamic-axes", action="store_true",
                    help="export with dynamic batch dimension")
    po.add_argument("--device", default=None)
    po.add_argument("--sanity-atol", type=float, default=1e-3,
                    help="absolute tolerance for PyTorch vs ONNX numeric sanity")
    po.set_defaults(func=cmd_export_onnx)

    pt = sub.add_parser("export-trt", help="ONNX → TensorRT FP16/INT8 (requires tensorrt+pycuda)")
    pt.add_argument("--config", required=True)
    pt.add_argument("--onnx", required=True)
    pt.add_argument("--output", required=True, help="path to write .engine file")
    pt.add_argument("--precision", choices=["fp16", "int8"], default="fp16")
    pt.add_argument("--calib-images", default=None,
                    help="for precision=int8: dir with calibration images (PNG/JPG)")
    pt.add_argument("--calib-cache", default=None,
                    help="for precision=int8: path to read/write INT8 calibration cache")
    pt.add_argument("--workspace-mb", type=int, default=4096,
                    help="TRT workspace pool size in MB (default 4096)")
    pt.set_defaults(func=cmd_export_trt)

    return p


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except ConfigError as exc:
        LOG.error("CONFIG ERROR: %s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
