#!/usr/bin/env python3
"""Evaluation pipeline for VisionDrive3D YOLO models."""

from __future__ import annotations

import argparse
import json
import os
import time
import re
from pathlib import Path
from typing import Dict, List, Sequence

import cv2
import numpy as np

SCENE_IMAGE_RE = re.compile(r"scene_(\d+)\.png$")


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_dataset_root(dataset_arg: str, root: Path) -> Path:
    candidate = Path(dataset_arg).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    return candidate.resolve()


def scene_index(path: Path) -> int:
    match = SCENE_IMAGE_RE.fullmatch(path.name)
    if not match:
        raise ValueError(f"Unexpected scene filename: {path.name}")
    return int(match.group(1))


def list_test_images(dataset_root: Path) -> List[Path]:
    test_dir = dataset_root / "images" / "test"
    images: List[Path] = []
    for path in test_dir.glob("scene_*.png"):
        if path.is_file() and SCENE_IMAGE_RE.fullmatch(path.name):
            images.append(path)
    images.sort(key=scene_index)
    return images


def count_annotations(label_path: Path) -> int:
    if not label_path.exists():
        return 0
    with label_path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def select_top_annotated_images(dataset_root: Path, test_images: Sequence[Path], top_k: int = 5) -> List[Path]:
    labels_test_dir = dataset_root / "labels" / "yolo" / "test"

    scored: List[tuple[int, int, Path]] = []
    for img_path in test_images:
        ann_count = count_annotations(labels_test_dir / f"{img_path.stem}.txt")
        scored.append((ann_count, scene_index(img_path), img_path))

    scored.sort(key=lambda item: (-item[0], item[1]))
    return [item[2] for item in scored[:top_k]]


def measure_inference_ms(model, test_images: Sequence[Path], max_images: int = 20, warmup: int = 5) -> float:
    sample = list(test_images[:max_images])
    if not sample:
        return 0.0

    times_ms: List[float] = []
    for img_path in sample:
        t0 = time.perf_counter()
        model(str(img_path), verbose=False)
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    if len(times_ms) > warmup:
        measured = times_ms[warmup:]
    else:
        measured = times_ms

    return float(np.mean(measured)) if measured else 0.0


def _extract_metrics_from_namespace(ns) -> Dict[str, float]:
    if ns is None:
        return {}
    out: Dict[str, float] = {}
    if hasattr(ns, "map50"):
        out["mAP50"] = float(ns.map50)
    if hasattr(ns, "map"):
        out["mAP50_95"] = float(ns.map)
    if hasattr(ns, "mp"):
        out["precision"] = float(ns.mp)
    if hasattr(ns, "mr"):
        out["recall"] = float(ns.mr)
    return out


def _extract_metrics_from_results_dict(results_dict: Dict[str, float], prefer_seg: bool) -> Dict[str, float]:
    if not isinstance(results_dict, dict):
        return {}

    suffix = "M" if prefer_seg else "B"
    key_map = {
        f"metrics/mAP50({suffix})": "mAP50",
        f"metrics/mAP50-95({suffix})": "mAP50_95",
        f"metrics/precision({suffix})": "precision",
        f"metrics/recall({suffix})": "recall",
    }

    out: Dict[str, float] = {}
    for raw_key, target_key in key_map.items():
        if raw_key in results_dict:
            out[target_key] = float(results_dict[raw_key])

    return out


def extract_metric_bundle(metrics, prefer_seg: bool) -> Dict[str, float]:
    metric_values: Dict[str, float] = {}

    if prefer_seg and hasattr(metrics, "seg"):
        metric_values.update(_extract_metrics_from_namespace(metrics.seg))

    if not metric_values and hasattr(metrics, "box"):
        metric_values.update(_extract_metrics_from_namespace(metrics.box))

    if not metric_values:
        metric_values.update(_extract_metrics_from_results_dict(getattr(metrics, "results_dict", {}), prefer_seg=prefer_seg))

    # Try fallback namespace to fill missing fields.
    if hasattr(metrics, "box"):
        metric_values = {
            **_extract_metrics_from_namespace(metrics.box),
            **metric_values,
        }

    if prefer_seg and hasattr(metrics, "seg"):
        metric_values = {
            **_extract_metrics_from_namespace(metrics.seg),
            **metric_values,
        }

    defaults = {
        "mAP50": 0.0,
        "mAP50_95": 0.0,
        "precision": 0.0,
        "recall": 0.0,
    }
    defaults.update({k: float(v) for k, v in metric_values.items() if isinstance(v, (int, float))})
    return defaults


def resolve_weight_path(weight_ref: str, root: Path) -> Path:
    weight_path = Path(weight_ref)
    if not weight_path.is_absolute():
        weight_path = root / weight_path
    return weight_path.resolve()


def configure_ultralytics(root: Path) -> None:
    config_dir = root / ".ultralytics"
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(config_dir))


def resolve_pretrained_weight(root: Path, filename: str) -> str:
    local_path = root / "weights" / filename
    return str(local_path) if local_path.exists() else filename


def resolve_runtime_device(requested_device: str) -> str:
    requested = str(requested_device).strip()
    if requested.lower() == "cpu":
        return "cpu"

    wants_cuda = requested.isdigit() or requested.lower().startswith("cuda")
    if not wants_cuda:
        return requested

    try:
        import torch
    except Exception:
        return requested

    if not torch.cuda.is_available():
        print(
            f"[WARN] Requested CUDA device '{requested}' but CUDA is unavailable. "
            "Falling back to CPU."
        )
        return "cpu"

    return requested


def evaluate_single_model(
    model,
    dataset_yaml: Path,
    test_images: Sequence[Path],
    prefer_seg_metrics: bool,
    qualitative_images: Sequence[Path],
    qualitative_dir: Path,
    output_prefix: str,
    device: str,
) -> Dict[str, float]:
    val_metrics = model.val(
        data=str(dataset_yaml),
        split="test",
        device=device,
        verbose=False,
    )

    metric_bundle = extract_metric_bundle(val_metrics, prefer_seg=prefer_seg_metrics)
    metric_bundle["inference_ms"] = measure_inference_ms(model, test_images=test_images)

    for img_path in qualitative_images:
        res = model(str(img_path), verbose=False)[0]
        out_path = qualitative_dir / f"{output_prefix}_{img_path.name}"
        cv2.imwrite(str(out_path), res.plot())

    return metric_bundle


def count_split(dataset_root: Path, split_name: str) -> int:
    split_dir = dataset_root / "images" / split_name
    return len(list(split_dir.glob("scene_*.png"))) if split_dir.exists() else 0


def print_summary_table(summary_rows: Sequence[tuple[str, Dict[str, float]]]) -> None:
    print("Model                  | mAP@50 | mAP@50-95 | Speed(ms)")
    for model_name, metrics in summary_rows:
        print(
            f"{model_name:<22} | "
            f"{metrics['mAP50']:.3f}  | "
            f"{metrics['mAP50_95']:.3f}     | "
            f"{metrics['inference_ms']:.1f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate YOLO models on VisionDrive3D test split")
    parser.add_argument("--dataset", type=str, default="./output_dataset")
    parser.add_argument("--device", type=str, default="0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = project_root()
    dataset_root = resolve_dataset_root(args.dataset, root)
    runtime_device = resolve_runtime_device(args.device)

    dataset_yaml = dataset_root / "dataset.yaml"
    if not dataset_yaml.exists():
        raise FileNotFoundError(f"Missing dataset config: {dataset_yaml}")

    trained_models_path = dataset_root / "trained_models.json"
    if not trained_models_path.exists():
        raise FileNotFoundError(f"Missing trained model paths: {trained_models_path}")

    seg_dataset_yaml = dataset_root / "dataset_seg.yaml"
    if not seg_dataset_yaml.exists():
        raise FileNotFoundError(f"Missing segmentation dataset config: {seg_dataset_yaml}")


    trained_models = json.loads(trained_models_path.read_text(encoding="utf-8"))
    det_ft_path = resolve_weight_path(trained_models["detection"], root)
    seg_ft_path = resolve_weight_path(trained_models["segmentation"], root)

    if not det_ft_path.exists():
        raise FileNotFoundError(f"Fine-tuned detection weights not found: {det_ft_path}")
    if not seg_ft_path.exists():
        raise FileNotFoundError(f"Fine-tuned segmentation weights not found: {seg_ft_path}")

    test_images = list_test_images(dataset_root)
    if not test_images:
        raise RuntimeError(
            f"No test images found in {dataset_root / 'images' / 'test'}. "
            "Run train.py first to prepare split symlinks."
        )

    qualitative_images = select_top_annotated_images(dataset_root, test_images, top_k=5)

    qualitative_dir = dataset_root / "ai_results" / "qualitative"
    qualitative_dir.mkdir(parents=True, exist_ok=True)

    # Save input reference images once.
    for img_path in qualitative_images:
        image_bgr = cv2.imread(str(img_path))
        if image_bgr is not None:
            cv2.imwrite(str(qualitative_dir / f"input_{img_path.name}"), image_bgr)

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "ultralytics is not installed. Install it in the active environment first."
        ) from exc

    configure_ultralytics(root)

    det_baseline_model = YOLO(resolve_pretrained_weight(root, "yolov8n.pt"))
    det_finetuned_model = YOLO(str(det_ft_path))
    seg_baseline_model = YOLO(resolve_pretrained_weight(root, "yolov8n-seg.pt"))
    seg_finetuned_model = YOLO(str(seg_ft_path))

    det_baseline = evaluate_single_model(
        model=det_baseline_model,
        dataset_yaml=dataset_yaml,
        test_images=test_images,
        prefer_seg_metrics=False,
        qualitative_images=qualitative_images,
        qualitative_dir=qualitative_dir,
        output_prefix="detection_baseline",
        device=runtime_device,
    )

    det_finetuned = evaluate_single_model(
        model=det_finetuned_model,
        dataset_yaml=dataset_yaml,
        test_images=test_images,
        prefer_seg_metrics=False,
        qualitative_images=qualitative_images,
        qualitative_dir=qualitative_dir,
        output_prefix="detection_finetuned",
        device=runtime_device,
    )

    seg_baseline = evaluate_single_model(
        model=seg_baseline_model,
        dataset_yaml=seg_dataset_yaml,
        test_images=test_images,
        prefer_seg_metrics=True,
        qualitative_images=qualitative_images,
        qualitative_dir=qualitative_dir,
        output_prefix="seg_baseline",
        device=runtime_device,
    )

    seg_finetuned = evaluate_single_model(
        model=seg_finetuned_model,
        dataset_yaml=seg_dataset_yaml,
        test_images=test_images,
        prefer_seg_metrics=True,
        qualitative_images=qualitative_images,
        qualitative_dir=qualitative_dir,
        output_prefix="seg_finetuned",
        device=runtime_device,
    )

    results_payload = {
        "detection": {
            "baseline": det_baseline,
            "finetuned": det_finetuned,
        },
        "segmentation": {
            "baseline": seg_baseline,
            "finetuned": seg_finetuned,
        },
        "meta": {
            "resolution": "1280x720",
            "classes": ["car"],
            "splits": {
                "train": count_split(dataset_root, "train"),
                "val": count_split(dataset_root, "val"),
                "test": count_split(dataset_root, "test"),
            },
            "qualitative_scenes": [img.stem for img in qualitative_images],
        },
    }

    results_path = dataset_root / "results.json"
    results_path.write_text(json.dumps(results_payload, indent=2), encoding="utf-8")

    summary_rows = [
        ("YOLOv8n (baseline)", det_baseline),
        ("YOLOv8n (fine-tuned)", det_finetuned),
        ("YOLOv8n-seg (baseline)", seg_baseline),
        ("YOLOv8n-seg (finetuned)", seg_finetuned),
    ]
    print_summary_table(summary_rows)
    print(f"Saved results to: {results_path}")
    print(f"Saved qualitative predictions to: {qualitative_dir}")


if __name__ == "__main__":
    main()
