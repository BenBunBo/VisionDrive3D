#!/usr/bin/env python3
"""Zero-shot car instance-segmentation benchmark on the VisionDrive3D test split.

Run: /home/ml4u/anaconda3/envs/vd3d-ai/bin/python src/ai_demo/benchmark_instance_segmentation.py
Requires a working CUDA GPU. Results are saved under .mesh_cache/benchmark_instance_segmentation/.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "YOLO_CONFIG_DIR",
    str(Path(__file__).resolve().parents[2] / ".mesh_cache/benchmark_instance_segmentation/yolo_config"),
)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2
import numpy as np
import torch
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from src.ai_demo.model_zoo import project_root

ROOT = project_root()
MODELS = {
    "yolo26n-seg": {"kind": "ultralytics", "weight": "yolo26n-seg.pt"},
    "yolo26s-seg": {"kind": "ultralytics", "weight": "yolo26s-seg.pt"},
    "yolo26m-seg": {"kind": "ultralytics", "weight": "yolo26m-seg.pt"},
    "yolo26l-seg": {"kind": "ultralytics", "weight": "yolo26l-seg.pt"},
    "yolo26x-seg": {"kind": "ultralytics", "weight": "yolo26x-seg.pt"},
    "mask2former-swin-t": {
        "kind": "transformers", "repo": "facebook/mask2former-swin-tiny-coco-instance",
    },
    "mask2former-swin-s": {
        "kind": "transformers", "repo": "facebook/mask2former-swin-small-coco-instance",
    },
}
CONF = 0.001
MASK_THRESHOLD = 0.5
MAX_DETECTIONS = 100
WARMUP_IMAGES = 10
TIMING_REPEATS = 5


def encode_mask(binary: np.ndarray) -> dict[str, Any]:
    rle = mask_utils.encode(np.asfortranarray(binary.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def decode_car_palette() -> dict[tuple[int, int, int], int]:
    """Renderer palette: Knuth-hashed pool IDs, saved through OpenGL RGB readback."""
    colors = {}
    for instance_id in range(1, 8):
        hashed = instance_id * 2654435761
        rgb = tuple((hashed >> shift) & 255 for shift in (0, 8, 16))
        colors[rgb] = instance_id
    return colors


def load_split_and_ground_truth() -> tuple[list[Path], dict, dict[str, int]]:
    split = ROOT / "output_dataset/ImageSets/test.txt"
    ids = [line.strip() for line in split.read_text().splitlines() if line.strip()]
    if len(ids) != len(set(ids)):
        raise ValueError("test split contains duplicate scene IDs")
    images, annotations, paths = [], [], {}
    annotation_id = 1
    palette = decode_car_palette()
    dataset = ROOT / "output_dataset"
    for image_id, scene_id in enumerate(ids):
        name = f"scene_{scene_id}.png"
        image_path = dataset / "images" / name
        rgb = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask_bgr = cv2.imread(str(dataset / "masks" / name), cv2.IMREAD_COLOR)
        if rgb is None or mask_bgr is None:
            raise FileNotFoundError(f"Missing test RGB image or instance mask for {name}")
        height, width = rgb.shape[:2]
        if mask_bgr.shape[:2] != (height, width):
            raise ValueError(f"Image/mask dimensions differ for {name}")
        paths[name] = image_id
        images.append({"id": image_id, "file_name": name, "width": width, "height": height})

        # The dataset writer stores the OpenGL readback bytes directly; its pixel
        # tuples therefore match the renderer's hashed palette without a channel swap.
        # Match palette colors to class-car boxes to exclude palette-like background.
        label_path = dataset / "labels/yolo" / f"scene_{scene_id}.txt"
        boxes = []
        if label_path.exists():
            for line in label_path.read_text().splitlines():
                if not line.strip():
                    continue
                _, cx, cy, bw, bh = map(float, line.split()[:5])
                boxes.append((max(0, int((cx - bw / 2) * width)),
                              max(0, int((cy - bh / 2) * height)),
                              min(width, int((cx + bw / 2) * width)),
                              min(height, int((cy + bh / 2) * height))))
        instance_masks = []
        for x1, y1, x2, y2 in boxes:
            crop = mask_bgr[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            counts = []
            for color in palette:
                count = int(np.all(crop == color, axis=2).sum())
                counts.append((count, color))
            count, color = max(counts, default=(0, ()))
            if count:
                binary = np.zeros((height, width), dtype=np.uint8)
                binary[y1:y2, x1:x2] = np.all(crop == color, axis=2).astype(np.uint8)
                instance_masks.append(binary)
        for binary in instance_masks:
            ys, xs = np.where(binary)
            if not len(xs):
                continue
            x, y = int(xs.min()), int(ys.min())
            box_w, box_h = int(xs.max()) - x + 1, int(ys.max()) - y + 1
            annotations.append({"id": annotation_id, "image_id": image_id, "category_id": 1,
                                "bbox": [x, y, box_w, box_h], "area": int(binary.sum()),
                                "segmentation": encode_mask(binary), "iscrowd": 0})
            annotation_id += 1
    if not images:
        raise ValueError("test split is empty")
    gt = {"images": images, "annotations": annotations, "categories": [{"id": 1, "name": "car"}]}
    return [dataset / "images" / im["file_name"] for im in images], gt, paths


def load_model(name: str, cache_dir: Path):
    spec = MODELS[name]
    if spec["kind"] == "ultralytics":
        from ultralytics import YOLO
        weight_path = cache_dir / spec["weight"]
        if not weight_path.exists():
            from ultralytics.utils.downloads import attempt_download_asset
            weight_path = Path(attempt_download_asset(str(weight_path)))
        model = YOLO(str(weight_path))
        model.model.to("cuda:0").eval()
        model.model.end2end = True
        # Count/profile the graph used for inference: Conv/BN fused and the
        # auxiliary detection branch removed, as in Ultralytics' model table.
        model.model.fuse(verbose=False, imgsz=640)
        return {"model": model, "path": str(weight_path), "kind": "ultralytics"}

    from huggingface_hub import snapshot_download
    from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
    snapshot = Path(snapshot_download(repo_id=spec["repo"], cache_dir=str(cache_dir / "huggingface")))
    processor = AutoImageProcessor.from_pretrained(str(snapshot))
    model = Mask2FormerForUniversalSegmentation.from_pretrained(str(snapshot)).to("cuda:0").eval()
    return {"model": model, "processor": processor, "path": str(snapshot), "kind": "transformers"}


def make_predictions(bundle: dict, image_bgr: np.ndarray) -> list[dict]:
    height, width = image_bgr.shape[:2]
    if bundle["kind"] == "ultralytics":
        # Ultralytics numpy inputs use OpenCV's BGR channel order.
        # Evaluate every YOLO26-seg model in its native end-to-end NMS-free mode.
        result = bundle["model"].predict(image_bgr, conf=CONF, iou=0.7,
                                         max_det=MAX_DETECTIONS, imgsz=640,
                                         nms=False, verbose=False)[0]
        if result.boxes is None or result.masks is None:
            return []
        boxes = result.boxes.xyxy.detach().cpu().numpy()
        classes = result.boxes.cls.detach().cpu().numpy().astype(int)
        scores = result.boxes.conf.detach().cpu().numpy()
        masks = result.masks.data.detach().cpu().numpy()
        predictions = []
        for index, (box, class_id, score) in enumerate(zip(boxes, classes, scores)):
            if class_id != 2:
                continue
            binary = cv2.resize(masks[index].astype(np.float32), (width, height),
                                interpolation=cv2.INTER_LINEAR) >= MASK_THRESHOLD
            encoded = encode_mask(binary)
            predictions.append({"bbox": [float(box[0]), float(box[1]),
                                            float(box[2] - box[0]), float(box[3] - box[1])],
                                "score": float(score), "segmentation": encoded})
        return predictions

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    from PIL import Image
    inputs = bundle["processor"](images=Image.fromarray(rgb), return_tensors="pt").to("cuda:0")
    with torch.inference_mode():
        outputs = bundle["model"](**inputs)
    target_sizes = [(height, width)]
    results = bundle["processor"].post_process_instance_segmentation(
        outputs, threshold=CONF, mask_threshold=MASK_THRESHOLD,
        target_sizes=target_sizes, return_binary_maps=True,
    )[0]
    binary_maps = results.get("segmentation")
    segments = results.get("segments_info", [])
    if binary_maps is None or not segments:
        return []
    if isinstance(binary_maps, torch.Tensor):
        binary_maps = binary_maps.detach().cpu().numpy()
    predictions = []
    for index, segment in enumerate(segments):
        if int(segment["label_id"]) != 2:
            continue
        binary = np.asarray(binary_maps[index]) >= MASK_THRESHOLD
        if binary.shape != (height, width):
            binary = cv2.resize(binary.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
        ys, xs = np.where(binary)
        if not len(xs):
            continue
        x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
        predictions.append({"bbox": [x1, y1, x2 - x1, y2 - y1],
                            "score": float(segment.get("score", 1.0)),
                            "segmentation": encode_mask(binary)})
    predictions.sort(key=lambda item: item["score"], reverse=True)
    return predictions[:MAX_DETECTIONS]


def evaluate(gt: dict, predictions: list[dict], iou_type: str) -> float:
    coco_gt = COCO()
    coco_gt.dataset = gt
    coco_gt.createIndex()
    if not predictions:
        return 0.0
    coco_dt = coco_gt.loadRes(predictions)
    evaluator = COCOeval(coco_gt, coco_dt, iou_type)
    evaluator.params.maxDets = [1, 10, MAX_DETECTIONS]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    return float(evaluator.stats[0])


def measure_flops(bundle: dict, image_path: Path) -> float | None:
    from thop import profile
    target = bundle["model"].model if bundle["kind"] == "ultralytics" else bundle["model"]
    try:
        if bundle["kind"] == "ultralytics":
            dummy = torch.zeros(1, 3, 640, 640, device="cuda:0")
            macs, _ = profile(target, inputs=(dummy,), verbose=False)
        else:
            from PIL import Image
            bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            tensors = bundle["processor"](images=Image.fromarray(rgb), return_tensors="pt").to("cuda:0")
            forward_inputs = (tensors["pixel_values"],)
            if "pixel_mask" in tensors:
                forward_inputs += (tensors["pixel_mask"],)
            macs, _ = profile(target, inputs=forward_inputs, verbose=False)
    except Exception as exc:
        print(f"FLOPs unavailable: {exc}")
        return None
    if not macs:
        return None
    return float(macs * 2)


def count_parameters(bundle: dict) -> int:
    model = bundle["model"].model if bundle["kind"] == "ultralytics" else bundle["model"]
    return sum(parameter.numel() for parameter in model.parameters())


def run_one_model(name: str, bundle: dict, image_paths: list[Path], gt: dict,
                  name_to_id: dict[str, int], output: Path) -> dict[str, Any]:
    parameter_count = count_parameters(bundle)
    flops = measure_flops(bundle, image_paths[0])
    input_images = [cv2.imread(str(path), cv2.IMREAD_COLOR) for path in image_paths]
    for image in input_images[:min(WARMUP_IMAGES, len(input_images))]:
        make_predictions(bundle, image)
    timing_runs, final_preds = [], []
    for repeat in range(TIMING_REPEATS):
        torch.cuda.synchronize()
        start = time.perf_counter()
        flat = []
        for path, image in zip(image_paths, input_images):
            current = make_predictions(bundle, image)
            image_id = name_to_id[path.name]
            for prediction in current:
                flat.append({"image_id": image_id, "category_id": 1,
                             "bbox": prediction["bbox"], "score": prediction["score"],
                             "segmentation": prediction["segmentation"]})
        torch.cuda.synchronize()
        timing_runs.append(1000 * (time.perf_counter() - start) / len(image_paths))
        final_preds = flat
        print(f"run {repeat + 1}/{TIMING_REPEATS}: {timing_runs[-1]:.2f} ms/image", flush=True)
    box_map = evaluate(gt, [{k: v for k, v in p.items() if k != "segmentation"} for p in final_preds], "bbox")
    mask_map = evaluate(gt, final_preds, "segm")
    (output / f"{name}_predictions.json").write_text(json.dumps(final_preds))
    result = {"status": "complete", "mAPbox_50_95": box_map,
              "mAPmask_50_95": mask_map, "time_ms": statistics.mean(timing_runs),
              "FPS": 1000 / statistics.mean(timing_runs),
              "input_resolution": "640 × 640" if name.startswith("yolo26") else "384 × 384",
              "timing_runs_ms": timing_runs, "params": parameter_count,
              "FLOPs": flops, "n_predictions": len(final_preds), "checkpoint": bundle["path"]}
    print(f"{name}: box AP={box_map:.4f}, mask AP={mask_map:.4f}, "
          f"time={statistics.mean(timing_runs):.2f} ms, params={parameter_count / 1e6:.2f}M, FLOPs={flops}",
          flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    parser.add_argument("--save-dir", default=".mesh_cache/benchmark_instance_segmentation")
    parser.add_argument("--force", action="store_true", help="Re-run models with completed results.")
    args = parser.parse_args()
    output = (ROOT / args.save_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    results_path = output / "results.json"
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; benchmark requires a working GPU (CPU mode is disabled).")
        device_name = torch.cuda.get_device_name(0)
        image_paths, gt, name_to_id = load_split_and_ground_truth()
    except Exception as exc:
        placeholder = {name: {"status": "pending", "reason": str(exc)} for name in MODELS}
        pending_meta = {"split": "test", "n_images": 100, "device": None,
                        "status": "awaiting_gpu", "reason": str(exc)}
        payload = {"models": placeholder, "meta": pending_meta}
        results_path.write_text(json.dumps(payload, indent=2))
        (ROOT / "docs/assets/data/benchmark_instance_segmentation.json").write_text(
            json.dumps(payload, indent=2) + "\n")
        print(f"Preflight failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    cache_dir = output / "checkpoints"
    cache_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    if results_path.exists():
        try:
            results = json.loads(results_path.read_text()).get("models", {})
        except (OSError, json.JSONDecodeError):
            results = {}
    for model_name, result in results.items():
        if model_name in MODELS:
            result.setdefault("input_resolution", "640 × 640" if model_name.startswith("yolo26") else "384 × 384")
            if result.get("time_ms"):
                result["FPS"] = 1000 / result["time_ms"]
    meta = {"split": "test", "n_images": len(image_paths), "n_gt_instances": len(gt["annotations"]),
            "device": device_name, "precision": "FP32", "confidence": CONF,
            "mask_threshold": MASK_THRESHOLD, "max_detections": MAX_DETECTIONS,
            "timing": "preprocess+transfer+forward+postprocess; excludes disk I/O and COCO evaluation",
            "ground_truth_boxes": "bounding boxes derived from car instance masks",
            "warmup_images": WARMUP_IMAGES, "timing_repeats": TIMING_REPEATS}
    results_path.write_text(json.dumps({"models": results, "meta": meta}, indent=2))
    for name in args.models:
        if results.get(name, {}).get("status") == "complete" and not args.force:
            print(f"{name}: using completed result (pass --force to re-run)", flush=True)
            continue
        print(f"\n=== {name} ===", flush=True)
        try:
            bundle = load_model(name, cache_dir)
            results[name] = run_one_model(name, bundle, image_paths, gt, name_to_id, output)
        except Exception as exc:
            results[name] = {"status": "error", "reason": str(exc)}
            print(f"{name} failed: {exc}", file=sys.stderr, flush=True)
        finally:
            if "bundle" in locals():
                del bundle
            gc.collect()
            torch.cuda.empty_cache()
        results_path.write_text(json.dumps({"models": results, "meta": meta}, indent=2))
        (ROOT / "docs/assets/data/benchmark_instance_segmentation.json").write_text(
            json.dumps({"models": results, "meta": meta}, indent=2) + "\n")
    meta["status"] = "complete" if all(results.get(name, {}).get("status") == "complete" for name in MODELS) else "partial"
    results_path.write_text(json.dumps({"models": results, "meta": meta}, indent=2))
    (ROOT / "docs/assets/data/benchmark_instance_segmentation.json").write_text(
        json.dumps({"models": results, "meta": meta}, indent=2) + "\n")
    print(f"Saved benchmark results: {results_path}")


if __name__ == "__main__":
    main()
