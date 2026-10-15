#!/usr/bin/env python3
"""Benchmark zero-shot YOLO26-sem and an optional fine-tuned n checkpoint.

Run: /home/ml4u/anaconda3/envs/vd3d-ai/bin/python src/ai_demo/benchmark_yolo26_semantic.py
Run comparison: /home/ml4u/anaconda3/envs/vd3d-ai/bin/python src/ai_demo/benchmark_yolo26_semantic.py --finetuned .mesh_cache/yolo26_semantic_finetune/runs/yolo26n_sem_50ep_b8/weights/best.pt
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import statistics
import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils.downloads import attempt_download_asset

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / ".mesh_cache/semantic_dataset_v1"
CHECKPOINTS = ROOT / ".mesh_cache/benchmark_yolo26_semantic/checkpoints"
OUTPUT = ROOT / ".mesh_cache/benchmark_yolo26_semantic"
FAMILY = {f"yolo26{x}-sem": f"yolo26{x}-sem.pt" for x in "nsmlx"}
SCENES = ("scene_00025", "scene_00046", "scene_00733")
COLOR = np.array([36, 180, 95], dtype=np.uint8)
IMGSZ = 640
WARMUP = 10
REPEATS = 5


def ensure_weight(filename: str) -> Path:
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    path = CHECKPOINTS / filename
    if not path.exists():
        downloaded = Path(attempt_download_asset(str(path)))
        if downloaded != path and downloaded.exists():
            path = downloaded
    if not path.is_file():
        raise FileNotFoundError(f"Missing pretrained model: {path}")
    return path


def car_class_id(model: YOLO) -> int:
    names = model.names
    items = names.items() if isinstance(names, dict) else enumerate(names)
    for idx, name in items:
        if str(name).strip().casefold() == "car":
            return int(idx)
    raise ValueError(f"Checkpoint has no class named car: {names}")


def semantic_prediction(model: YOLO, image: np.ndarray, class_id: int) -> np.ndarray:
    result = model.predict(image, device=0, imgsz=IMGSZ, rect=False, verbose=False)[0]
    if result.semantic_mask is None:
        raise RuntimeError("Semantic checkpoint returned no class map")
    predicted = result.semantic_mask.data.detach().cpu().numpy()
    if predicted.shape != (900, 1600):
        raise ValueError(f"Expected original-size class map, got {predicted.shape}")
    return (predicted == class_id).astype(np.uint8)


def update_confusion(confusion: np.ndarray, pred_car: np.ndarray, target: np.ndarray) -> None:
    values = np.bincount((target.astype(np.uint8).ravel() * 2 + pred_car.ravel()), minlength=4)
    confusion += values.reshape(2, 2)


def metrics_from_confusion(cm: np.ndarray) -> dict:
    # Rows are ground truth [background, car], columns are predictions.
    intersection = np.diag(cm).astype(np.float64)
    union = cm.sum(axis=0) + cm.sum(axis=1) - intersection
    iou = np.divide(intersection, union, out=np.zeros(2), where=union > 0)
    dice_denom = cm.sum(axis=0) + cm.sum(axis=1)
    dice = np.divide(2 * intersection, dice_denom, out=np.zeros(2), where=dice_denom > 0)
    total = int(cm.sum())
    return {
        "background_iou": float(iou[0]), "car_iou": float(iou[1]),
        "car_dice": float(dice[1]), "miou": float(iou.mean()),
        "pixel_accuracy": float(intersection.sum() / total) if total else 0.0,
        "confusion_matrix_gt_rows_pred_columns": cm.tolist(),
    }


def flops_for_model(model: YOLO) -> float | None:
    try:
        from thop import profile
        # YOLO's high-level predictor runs an AutoBackend copy on the selected
        # device, while model.model itself remains on CPU. Profile the module
        # where it lives so THOP does not mix CPU weights with a CUDA input.
        device = next(model.model.parameters()).device
        dummy = torch.zeros((1, 3, IMGSZ, IMGSZ), device=device)
        macs, _ = profile(model.model, inputs=(dummy,), verbose=False)
        return float(macs * 2) if macs else None
    except Exception as exc:
        print(f"FLOPs unavailable for {model.names}: {exc}", flush=True)
        return None


def load_test() -> list[tuple[Path, np.ndarray, np.ndarray]]:
    ids = [v.strip() for v in (ROOT / "output_dataset/ImageSets/test.txt").read_text().splitlines() if v.strip()]
    if len(ids) != 100 or len(set(ids)) != 100:
        raise ValueError(f"Expected 100 unique test images, found {len(ids)}")
    rows = []
    for sid in ids:
        image = DATA / "images/test" / f"scene_{sid}.png"
        image_data = cv2.imread(str(image), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(DATA / "masks/test" / f"scene_{sid}.png"), cv2.IMREAD_GRAYSCALE)
        if image_data is None or image_data.shape != (900, 1600, 3) or mask is None or mask.shape != (900, 1600):
            raise ValueError(f"Missing or invalid test pair for scene_{sid}")
        if not set(np.unique(mask)).issubset({0, 1}):
            raise ValueError(f"Unexpected labels in semantic mask for scene_{sid}")
        rows.append((image, image_data, mask))
    return rows


def measure(model: YOLO, weight: Path, rows: list[tuple[Path, np.ndarray, np.ndarray]], *, make_qual: bool) -> dict:
    cid = car_class_id(model)
    torch.set_num_threads(4)
    for _, image, _ in rows[:WARMUP]:
        semantic_prediction(model, image, cid)
    timings = []
    final_confusion = np.zeros((2, 2), dtype=np.int64)
    for repeat in range(REPEATS):
        torch.cuda.synchronize()
        start = time.perf_counter()
        confusion = np.zeros((2, 2), dtype=np.int64)
        for _, image, target in rows:
            pred = semantic_prediction(model, image, cid)
            update_confusion(confusion, pred, target)
        torch.cuda.synchronize()
        timings.append(1000.0 * (time.perf_counter() - start) / len(rows))
        final_confusion = confusion
        print(f"{weight.stem} pass {repeat + 1}/{REPEATS}: {timings[-1]:.2f} ms/image", flush=True)
    result = metrics_from_confusion(final_confusion)
    result.update({
        "status": "complete", "task": "semantic", "class": "car",
        "input_resolution": "640 × 640", "n_images": len(rows),
        "time_ms": statistics.mean(timings), "timing_runs_ms": timings,
        "FPS": 1000.0 / statistics.mean(timings),
        "checkpoint": str(weight.relative_to(ROOT)),
        "params": sum(p.numel() for p in model.model.parameters()),
        "FLOPs": flops_for_model(model),
    })
    if make_qual:
        for stem in SCENES:
            image_path = DATA / "images/test" / f"{stem}.png"
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            pred = semantic_prediction(model, image, cid).astype(bool)
            overlay = image.copy()
            overlay[pred] = COLOR
            image[pred] = cv2.addWeighted(overlay, 0.32, image, 0.68, 0)[pred]
            destination = OUTPUT / "qualitative" / f"{weight.stem}_{stem}.png"
            destination.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(destination), image)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--finetuned", type=Path, help="Optional yolo26n-sem best.pt to compare and visualize")
    parser.add_argument("--models", nargs="+", choices=list(FAMILY), default=list(FAMILY))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for YOLO26 semantic benchmarking")
    rows = load_test()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    results = {}
    for key in args.models:
        weight = ensure_weight(FAMILY[key])
        model = YOLO(str(weight))
        results[key] = measure(model, weight, rows, make_qual=key == "yolo26n-sem")
        print(json.dumps({key: results[key]}, indent=2), flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()

    finetuned = None
    if args.finetuned:
        weight = args.finetuned.resolve()
        if not weight.is_file():
            raise FileNotFoundError(weight)
        model = YOLO(str(weight))
        finetuned = measure(model, weight, rows, make_qual=True)
        results["yolo26n-sem-finetuned"] = finetuned
        print(json.dumps({"yolo26n-sem-finetuned": finetuned}, indent=2), flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()

    payload = {
        "models": results,
        "meta": {
            "task": "semantic segmentation", "class": "car", "background_id": 0, "car_id": 1,
            "split": "test", "n_images": len(rows), "imgsz": IMGSZ,
            "rect": False, "warmup_images": WARMUP, "timing_repeats": REPEATS,
            "precision": "FP32", "device": torch.cuda.get_device_name(0),
            "dataset": str((DATA / "dataset.yaml").relative_to(ROOT)),
        },
    }
    out = OUTPUT / "results.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Saved {out}", flush=True)


if __name__ == "__main__":
    main()
