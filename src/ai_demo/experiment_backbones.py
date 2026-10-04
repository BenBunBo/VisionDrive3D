#!/usr/bin/env python3
"""Backbone comparison experiments for VisionDrive3D detection task."""

from __future__ import annotations

import argparse
import csv
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np

SCENE_IMAGE_RE = re.compile(r"scene_(\d+)\.png$")
MODEL_NAMES = ["yolov8n", "yolov8s", "yolov8m", "rtdetr-l"]


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_dataset_root(dataset_arg: str, root: Path) -> Path:
    candidate = Path(dataset_arg).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    return candidate.resolve()


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


def scene_index(path: Path) -> int:
    match = SCENE_IMAGE_RE.fullmatch(path.name)
    if not match:
        raise ValueError(f"Unexpected scene filename: {path.name}")
    return int(match.group(1))


def list_test_images(dataset_root: Path) -> List[Path]:
    test_dir = dataset_root / "images" / "test"
    images = [p for p in test_dir.glob("scene_*.png") if p.is_file() and SCENE_IMAGE_RE.fullmatch(p.name)]
    images.sort(key=scene_index)
    return images


def measure_inference_ms(model, test_images: Sequence[Path], max_images: int = 20, warmup: int = 5) -> float:
    sample = list(test_images[:max_images])
    if not sample:
        return 0.0

    times_ms: List[float] = []
    for img_path in sample:
        t0 = time.perf_counter()
        model(str(img_path), verbose=False)
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    measured = times_ms[warmup:] if len(times_ms) > warmup else times_ms
    return float(np.mean(measured)) if measured else 0.0


def extract_detection_metrics(val_metrics) -> Dict[str, float]:
    out = {
        "mAP50": 0.0,
        "mAP50_95": 0.0,
        "precision": 0.0,
        "recall": 0.0,
    }

    if hasattr(val_metrics, "box") and val_metrics.box is not None:
        box = val_metrics.box
        out["mAP50"] = float(getattr(box, "map50", 0.0))
        out["mAP50_95"] = float(getattr(box, "map", 0.0))
        out["precision"] = float(getattr(box, "mp", 0.0))
        out["recall"] = float(getattr(box, "mr", 0.0))
        return out

    results_dict = getattr(val_metrics, "results_dict", {})
    if isinstance(results_dict, dict):
        out["mAP50"] = float(results_dict.get("metrics/mAP50(B)", out["mAP50"]))
        out["mAP50_95"] = float(results_dict.get("metrics/mAP50-95(B)", out["mAP50_95"]))
        out["precision"] = float(results_dict.get("metrics/precision(B)", out["precision"]))
        out["recall"] = float(results_dict.get("metrics/recall(B)", out["recall"]))

    return out


def params_in_millions(model) -> float:
    module = getattr(model, "model", None)
    if module is None or not hasattr(module, "parameters"):
        return 0.0
    total = 0
    for param in module.parameters():
        total += int(param.numel())
    return float(total / 1_000_000.0)


def is_oom_error(exc: BaseException) -> bool:
    return "out of memory" in str(exc).lower()


def train_with_oom_fallback(model, *, data: str, epochs: int, imgsz: int, batch: int, device: str, project: str, name: str) -> int:
    train_kwargs = dict(
        data=data,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=name,
        exist_ok=True,
        seed=42,
    )

    try:
        model.train(**train_kwargs)
        return batch
    except RuntimeError as exc:
        if not is_oom_error(exc) or batch <= 8:
            raise

        print(f"[WARN] {name}: OOM at batch={batch}, retrying with batch=8")
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

        train_kwargs["batch"] = 8
        model.train(**train_kwargs)
        return 8


def write_csv(csv_path: Path, rows: Sequence[Dict[str, float]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["model", "mAP50", "mAP50_95", "precision", "recall", "inference_ms", "params_M"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_results(rows: Sequence[Dict[str, float]], out_png: Path) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    models = [row["model"] for row in rows]
    map50 = [float(row["mAP50"]) for row in rows]
    speed = [float(row["inference_ms"]) for row in rows]
    params_m = [float(row["params_M"]) for row in rows]

    fig, (ax_bar, ax_scatter) = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    ax_bar.bar(models, map50, color="#4A90D9")
    ax_bar.set_title("mAP@50 by Backbone")
    ax_bar.set_ylabel("mAP@50")
    ax_bar.set_ylim(0.0, max(map50) * 1.15 if map50 else 1.0)
    ax_bar.grid(axis="y", alpha=0.3)

    sizes = [max(30.0, p * 30.0) for p in params_m]
    scatter = ax_scatter.scatter(speed, map50, s=sizes, c="#4A90D9", alpha=0.75, edgecolors="black", linewidths=0.5)
    _ = scatter
    ax_scatter.set_title("Speed vs mAP@50")
    ax_scatter.set_xlabel("Inference time (ms)")
    ax_scatter.set_ylabel("mAP@50")
    ax_scatter.grid(alpha=0.3)

    for x, y, label in zip(speed, map50, models):
        ax_scatter.annotate(label, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)

    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run detection backbone comparison on VisionDrive3D")
    parser.add_argument("--dataset", type=str, default="./output_dataset")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
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

    test_images = list_test_images(dataset_root)
    if not test_images:
        raise RuntimeError(
            f"No test images found in {dataset_root / 'images' / 'test'}. "
            "Run train.py first to prepare split symlinks."
        )

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "ultralytics is not installed. Install it in the active environment first."
        ) from exc

    configure_ultralytics(root)

    rows: List[Dict[str, float]] = []

    for model_name in MODEL_NAMES:
        weights_name = resolve_pretrained_weight(root, f"{model_name}.pt")
        print(f"\n[INFO] Backbone experiment: {model_name}")
        try:
            model = YOLO(weights_name)
        except Exception as exc:
            print(f"[WARN] Skipping {model_name}: failed to load {weights_name}: {exc}")
            continue

        params_m = params_in_millions(model)

        try:
            used_batch = train_with_oom_fallback(
                model,
                data=str(dataset_yaml),
                epochs=args.epochs,
                imgsz=args.imgsz,
                batch=args.batch,
                device=runtime_device,
                project="runs/backbone_exp",
                name=model_name,
            )
        except Exception as exc:
            print(f"[WARN] Skipping {model_name}: training failed: {exc}")
            continue

        try:
            val_metrics = model.val(data=str(dataset_yaml), split="test", device=runtime_device, verbose=False)
            metric_bundle = extract_detection_metrics(val_metrics)
        except Exception as exc:
            print(f"[WARN] Skipping {model_name}: validation failed: {exc}")
            continue

        inference_ms = measure_inference_ms(model, test_images)

        row = {
            "model": model_name,
            "mAP50": float(metric_bundle["mAP50"]),
            "mAP50_95": float(metric_bundle["mAP50_95"]),
            "precision": float(metric_bundle["precision"]),
            "recall": float(metric_bundle["recall"]),
            "inference_ms": float(inference_ms),
            "params_M": float(params_m),
        }
        rows.append(row)

        print(
            f"[INFO] {model_name}: batch={used_batch}, mAP50={row['mAP50']:.3f}, "
            f"mAP50-95={row['mAP50_95']:.3f}, speed={row['inference_ms']:.1f} ms, params={row['params_M']:.2f}M"
        )

    csv_path = dataset_root / "backbone_comparison.csv"
    write_csv(csv_path, rows)

    chart_path = dataset_root / "ai_results" / "backbone_comparison.png"
    if rows:
        plot_results(rows, chart_path)

    print(f"Saved backbone CSV to: {csv_path}")
    if rows:
        print(f"Saved backbone comparison chart to: {chart_path}")
    else:
        print("No successful backbone runs; chart not generated.")


if __name__ == "__main__":
    main()
