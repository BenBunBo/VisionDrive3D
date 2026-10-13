#!/usr/bin/env python3
"""Compare COCO baseline and VisionDrive3D fine-tuned YOLO26n on the car test split."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2
import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics import YOLO

from src.ai_demo.benchmark_instance_segmentation import decode_car_palette, encode_mask
from src.ai_demo.model_zoo import project_root

ROOT = project_root()
CONFIDENCE = 0.001
IMAGE_SIZE = 640
MAX_DETECTIONS = 100
SCALAR_METRICS = ("mAPbox_50", "mAPbox_50_95", "mAPmask_50", "mAPmask_50_95")


def build_ground_truth(dataset: Path):
    ids = [line.strip() for line in (dataset / "ImageSets/test.txt").read_text().splitlines()
           if line.strip()]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Test ImageSets must contain unique scene IDs")
    images, box_annotations, mask_annotations = [], [], []
    name_to_id = {}
    box_ann_id = mask_ann_id = 1
    palette = decode_car_palette()
    for image_id, scene_id in enumerate(ids):
        name = f"scene_{scene_id}.png"
        image = cv2.imread(str(dataset / "images" / name), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(dataset / "masks" / name), cv2.IMREAD_COLOR)
        if image is None or mask is None:
            raise FileNotFoundError(f"Missing test image or instance mask for {name}")
        height, width = image.shape[:2]
        if mask.shape[:2] != (height, width):
            raise ValueError(f"Image and mask shapes differ for {name}")
        images.append({"id": image_id, "file_name": name, "width": width, "height": height})
        name_to_id[name] = image_id
        labels = dataset / "labels/yolo" / f"scene_{scene_id}.txt"
        if not labels.exists():
            continue
        for raw in labels.read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            cls, cx, cy, bw, bh = map(float, raw.split()[:5])
            if int(cls) != 0:
                continue
            abs_w, abs_h = bw * width, bh * height
            x1, y1 = max(0, int((cx - bw / 2) * width)), max(0, int((cy - bh / 2) * height))
            x2, y2 = min(width, int((cx + bw / 2) * width)), min(height, int((cy + bh / 2) * height))
            box_annotations.append({
                "id": box_ann_id, "image_id": image_id, "category_id": 1,
                "bbox": [(cx * width) - abs_w / 2, (cy * height) - abs_h / 2, abs_w, abs_h],
                "area": abs_w * abs_h, "iscrowd": 0,
            })
            box_ann_id += 1

            crop = mask[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            counts = [(int(np.all(crop == color, axis=2).sum()), color) for color in palette]
            pixel_count, color = max(counts, default=(0, ()))
            if not pixel_count:
                continue
            instance = np.zeros((height, width), dtype=np.uint8)
            instance[y1:y2, x1:x2] = np.all(crop == color, axis=2).astype(np.uint8)
            ys, xs = np.where(instance)
            if not len(xs):
                continue
            x, y = int(xs.min()), int(ys.min())
            mask_annotations.append({
                "id": mask_ann_id, "image_id": image_id, "category_id": 1,
                "bbox": [x, y, int(xs.max()) - x + 1, int(ys.max()) - y + 1],
                "area": int(instance.sum()), "segmentation": encode_mask(instance), "iscrowd": 0,
            })
            mask_ann_id += 1
    categories = [{"id": 1, "name": "car"}]
    return (
        {"images": images, "annotations": box_annotations, "categories": categories},
        {"images": images, "annotations": mask_annotations, "categories": categories},
        name_to_id,
    )


def coco_ap(gt: dict, predictions: list[dict], iou_type: str) -> tuple[float, float]:
    if not predictions:
        return 0.0, 0.0
    coco_gt = COCO()
    coco_gt.dataset = gt
    coco_gt.createIndex()
    coco_dt = coco_gt.loadRes(predictions)
    evaluator = COCOeval(coco_gt, coco_dt, iou_type)
    evaluator.params.maxDets = [1, 10, MAX_DETECTIONS]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    return float(evaluator.stats[1]), float(evaluator.stats[0])


def evaluate_model(weight: Path, kind: str, class_id: int, images: list[dict],
                   name_to_id: dict[str, int], box_gt: dict, mask_gt: dict) -> dict:
    model = YOLO(str(weight))
    box_predictions, mask_predictions = [], []
    for meta in images:
        image_path = ROOT / "output_dataset/images" / meta["file_name"]
        result = model.predict(
            str(image_path), imgsz=IMAGE_SIZE, conf=CONFIDENCE,
            max_det=MAX_DETECTIONS, nms=False, verbose=False,
        )[0]
        if result.boxes is None:
            continue
        boxes = result.boxes.xyxy.detach().cpu().numpy()
        classes = result.boxes.cls.detach().cpu().numpy().astype(int)
        scores = result.boxes.conf.detach().cpu().numpy()
        masks = result.masks.data.detach().cpu().numpy() if kind == "segment" and result.masks is not None else None
        height, width = meta["height"], meta["width"]
        for index, (xyxy, cls_id, score) in enumerate(zip(boxes, classes, scores)):
            if int(cls_id) != class_id:
                continue
            xywh = [float(xyxy[0]), float(xyxy[1]),
                    float(xyxy[2] - xyxy[0]), float(xyxy[3] - xyxy[1])]
            box_predictions.append({
                "image_id": name_to_id[meta["file_name"]], "category_id": 1,
                "bbox": xywh, "score": float(score),
            })
            if masks is None or index >= len(masks):
                continue
            binary = cv2.resize(masks[index].astype(np.float32), (width, height),
                                interpolation=cv2.INTER_LINEAR) >= 0.5
            mask_predictions.append({
                "image_id": name_to_id[meta["file_name"]], "category_id": 1,
                "bbox": xywh, "score": float(score), "segmentation": encode_mask(binary),
            })
    box50, box95 = coco_ap(box_gt, box_predictions, "bbox")
    if kind == "segment":
        mask50, mask95 = coco_ap(mask_gt, mask_predictions, "segm")
    else:
        mask50 = mask95 = None
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "mAPbox_50": box50, "mAPbox_50_95": box95,
        "mAPmask_50": mask50, "mAPmask_50_95": mask95,
        "n_box_predictions": len(box_predictions), "n_mask_predictions": len(mask_predictions),
        "checkpoint": str(weight.relative_to(ROOT)),
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to measure this model comparison")
    manifest_path = ROOT / "docs/assets/data/qualitative_yolo26.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset = ROOT / "output_dataset"
    box_gt, mask_gt, name_to_id = build_ground_truth(dataset)
    images = box_gt["images"]
    models = {
        "yolo26n": {
            "baseline": (dataset / "ckpt/yolo26n.pt", "detect", 2),
            "finetuned": (ROOT / manifest["detection_finetuned"], "detect", 0),
        },
        "yolo26n-seg": {
            "baseline": (ROOT / manifest["segmentation_baseline"], "segment", 2),
            "finetuned": (ROOT / manifest["segmentation_finetuned"], "segment", 0),
        },
    }
    results = {}
    for model_name, versions in models.items():
        results[model_name] = {}
        kind = versions["baseline"][1]
        for version, (weight, _, class_id) in versions.items():
            print(f"Evaluating {model_name} {version}: {weight}", flush=True)
            results[model_name][version] = evaluate_model(
                weight, kind, class_id, images, name_to_id, box_gt, mask_gt)
            row = results[model_name][version]
            print(f"  box AP50={row['mAPbox_50']:.4f}, AP50-95={row['mAPbox_50_95']:.4f}; "
                  f"mask AP50={row['mAPmask_50']}, AP50-95={row['mAPmask_50_95']}", flush=True)
    payload = {
        "models": results,
        "meta": {
            "split": "test", "n_images": len(images), "n_gt_boxes": len(box_gt["annotations"]),
            "n_gt_masks": len(mask_gt["annotations"]), "class": "car", "confidence": CONFIDENCE,
            "imgsz": IMAGE_SIZE, "max_detections": MAX_DETECTIONS, "nms": False,
            "device": torch.cuda.get_device_name(0), "precision": "FP32",
        },
    }
    output = ROOT / "docs/assets/data/benchmark_yolo26_finetune.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
