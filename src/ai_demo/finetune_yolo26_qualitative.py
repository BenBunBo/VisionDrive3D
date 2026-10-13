#!/usr/bin/env python3
"""Fine-tune YOLO26n models and refresh the three-scene docs qualitative grid."""
from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.ai_demo.train import (
    SplitData,
    prepare_split_symlinks,
    project_root,
    write_dataset_yaml,
    write_seg_dataset_yaml,
)

SCENES = ("scene_00025", "scene_00046", "scene_00733")
CONFIDENCE = 0.25
IMAGE_SIZE = 640
SEED = 42
COLORS = ((36, 180, 95), (240, 150, 40), (60, 130, 235), (200, 80, 180))


def build_car_polygon_labels(dataset_root: Path) -> Path:
    """Extract only renderer car colors inside GT boxes, matching benchmark mask decoding."""
    palette = {}
    for instance_id in range(1, 8):
        hashed = (instance_id * 2654435761) & 0xFFFFFFFF
        palette[tuple((hashed >> shift) & 255 for shift in (0, 8, 16))] = instance_id

    labels_root = dataset_root / "labels_seg"
    processed = 0
    for split_name in ("train", "val", "test"):
        source = dataset_root / "labels/yolo" / split_name
        target = labels_root / split_name
        target.mkdir(parents=True, exist_ok=True)
        for label_path in sorted(source.glob("scene_*.txt")):
            mask_path = dataset_root / "masks" / f"{label_path.stem}.png"
            mask = cv2.imread(str(mask_path), cv2.IMREAD_COLOR)
            if mask is None:
                raise FileNotFoundError(f"Unable to read instance mask {mask_path}")
            height, width = mask.shape[:2]
            lines = []
            for raw in label_path.read_text(encoding="utf-8").splitlines():
                if not raw.strip():
                    continue
                class_id, cx, cy, bw, bh = map(float, raw.split()[:5])
                if int(class_id) != 0:
                    continue
                x1 = max(0, int((cx - bw / 2) * width))
                y1 = max(0, int((cy - bh / 2) * height))
                x2 = min(width, int((cx + bw / 2) * width))
                y2 = min(height, int((cy + bh / 2) * height))
                crop = mask[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                counts = [(int(np.all(crop == color, axis=2).sum()), color)
                          for color in palette]
                pixel_count, color = max(counts, default=(0, ()))
                if pixel_count == 0:
                    continue
                instance_mask = np.all(crop == color, axis=2).astype(np.uint8)
                contours, _ = cv2.findContours(instance_mask, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    if cv2.contourArea(contour) < 100:
                        continue
                    approx = cv2.approxPolyDP(contour, 0.002 * cv2.arcLength(contour, True), True)
                    if len(approx) < 3:
                        continue
                    points = approx.reshape(-1, 2).astype(np.float32)
                    points[:, 0] = (points[:, 0] + x1) / width
                    points[:, 1] = (points[:, 1] + y1) / height
                    points = np.clip(points, 0.0, 1.0)
                    coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in points)
                    lines.append(f"0 {coords}")
            (target / label_path.name).write_text(
                ("\n".join(lines) + "\n") if lines else "", encoding="utf-8")
            processed += 1
            if processed % 100 == 0:
                print(f"Prepared car instance polygons for {processed} images", flush=True)
    return labels_root


def prepare_datasets(root: Path, dataset: Path, work: Path) -> tuple[Path, Path]:
    ids = {}
    for name in ("train", "val", "test"):
        ids[name] = [line.strip() for line in (dataset / "ImageSets" / f"{name}.txt")
                     .read_text(encoding="utf-8").splitlines() if line.strip()]
    groups = [set(items) for items in ids.values()]
    if any(groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("ImageSets train/val/test contain overlapping scene IDs")
    test_ids = set(ids["test"])
    if any(scene.removeprefix("scene_") not in test_ids for scene in SCENES):
        raise ValueError("All requested qualitative scenes must remain in the existing test split")

    data_root = work / "dataset"
    (data_root / "images").mkdir(parents=True, exist_ok=True)
    local_yolo = data_root / "labels" / "yolo"
    local_yolo.mkdir(parents=True, exist_ok=True)
    for label in (dataset / "labels" / "yolo").glob("scene_*.txt"):
        dst = local_yolo / label.name
        if not dst.exists():
            dst.symlink_to(label.resolve())
    mask_link = data_root / "masks"
    if not mask_link.exists():
        mask_link.symlink_to((dataset / "masks").resolve(), target_is_directory=True)

    split = SplitData(
        train=[dataset / "images" / f"scene_{scene}.png" for scene in ids["train"]],
        val=[dataset / "images" / f"scene_{scene}.png" for scene in ids["val"]],
        test=[dataset / "images" / f"scene_{scene}.png" for scene in ids["test"]],
    )
    prepare_split_symlinks(data_root, split)
    detection_yaml = write_dataset_yaml(data_root)
    segmentation_labels = build_car_polygon_labels(data_root)
    segmentation_yaml = write_seg_dataset_yaml(data_root, segmentation_labels)
    return detection_yaml, segmentation_yaml


def train_one(weight: Path, data_yaml: Path, project: Path, name: str,
              epochs: int, requested_batch: int, device: str) -> Path:
    batch = requested_batch
    while batch >= 1:
        try:
            model = YOLO(str(weight))
            model.train(
                data=str(data_yaml), epochs=epochs, imgsz=IMAGE_SIZE, batch=batch,
                device=device, project=str(project), name=f"{name}_b{batch}",
                exist_ok=True, seed=SEED, val=True, cache=False, workers=4,
                plots=False, verbose=True,
            )
            best = Path(model.trainer.best).resolve()
            if not best.is_file():
                raise FileNotFoundError(f"Training did not produce best.pt: {best}")
            del model
            gc.collect()
            torch.cuda.empty_cache()
            return best
        except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
            if "out of memory" not in str(exc).lower() or batch == 1:
                raise
            print(f"CUDA OOM at batch={batch}; retrying at batch={batch // 2}", flush=True)
            batch //= 2
            gc.collect()
            torch.cuda.empty_cache()
    raise RuntimeError("Could not train with any supported batch size")


def draw_predictions(model: YOLO, image: np.ndarray, class_id: int, output: Path,
                     *, segmentation: bool) -> None:
    result = model.predict(
        source=image, imgsz=IMAGE_SIZE, conf=CONFIDENCE, max_det=100,
        nms=False, verbose=False,
    )[0]
    canvas = image.copy()
    if result.boxes is None:
        cv2.imwrite(str(output), canvas)
        return
    boxes = result.boxes.xyxy.detach().cpu().numpy()
    classes = result.boxes.cls.detach().cpu().numpy().astype(int)
    scores = result.boxes.conf.detach().cpu().numpy()
    polygons = result.masks.xy if segmentation and result.masks is not None else []
    overlay = canvas.copy()
    kept = []
    for i, (box, cls, score) in enumerate(zip(boxes, classes, scores)):
        if int(cls) != class_id:
            continue
        color = COLORS[len(kept) % len(COLORS)]
        polygon = polygons[i] if i < len(polygons) else None
        if segmentation and polygon is not None and len(polygon) >= 3:
            pts = np.round(polygon).astype(np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(overlay, [pts], color)
            cv2.polylines(canvas, [pts], True, color, 2, cv2.LINE_AA)
        kept.append((box, float(score), color))
    if segmentation:
        canvas = cv2.addWeighted(overlay, 0.32, canvas, 0.68, 0)
    for box, score, color in kept:
        x1, y1, x2, y2 = np.round(box).astype(int)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        label = f"car {score:.2f}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        top = max(0, y1 - th - baseline - 7)
        cv2.rectangle(canvas, (x1, top), (x1 + tw + 8, top + th + baseline + 7), color, -1)
        cv2.putText(canvas, label, (x1 + 4, top + th + 1), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 2, cv2.LINE_AA)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), canvas)


def generate_images(root: Path, dataset: Path, work: Path,
                    det_best: Path, seg_best: Path) -> dict:
    output = work / "qualitative"
    output.mkdir(parents=True, exist_ok=True)
    scene_inputs = {scene: dataset / "images" / f"{scene}.png" for scene in SCENES}
    for path in scene_inputs.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    source_weights = {
        "detection_baseline": dataset / "ckpt" / "yolo26n.pt",
        "detection_finetuned": det_best,
        "seg_baseline": root / ".mesh_cache/benchmark_instance_segmentation/checkpoints/yolo26n-seg.pt",
        "seg_finetuned": seg_best,
    }
    if any(not weight.is_file() for weight in source_weights.values()):
        raise FileNotFoundError("A required pretrained or fine-tuned YOLO26 checkpoint is missing")
    class_ids = {
        "detection_baseline": 2, "seg_baseline": 2,
        "detection_finetuned": 0, "seg_finetuned": 0,
    }
    outputs = []
    for prefix, weight in source_weights.items():
        is_seg = prefix.startswith("seg_")
        model = YOLO(str(weight))
        for scene, image_path in scene_inputs.items():
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"Could not read {image_path}")
            name = f"{prefix}_{scene}.png"
            path = output / name
            draw_predictions(model, image, class_ids[prefix], path, segmentation=is_seg)
            outputs.append(path)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    input_outputs = []
    for scene, src in scene_inputs.items():
        path = output / f"input_{scene}.png"
        shutil.copy2(src, path)
        input_outputs.append(path)
    all_outputs = input_outputs + outputs
    if len(all_outputs) != 15 or any(not path.is_file() or path.stat().st_size == 0 for path in all_outputs):
        raise RuntimeError("Expected 15 non-empty qualitative images")
    return {
        "scenes": list(SCENES),
        "confidence": CONFIDENCE,
        "image_size": IMAGE_SIZE,
        "nms": False,
        "detection_baseline": str(source_weights["detection_baseline"].relative_to(root)),
        "detection_finetuned": str(det_best.relative_to(root)),
        "segmentation_baseline": str(source_weights["seg_baseline"].relative_to(root)),
        "segmentation_finetuned": str(seg_best.relative_to(root)),
        "images": [path.name for path in all_outputs],
    }


def publish(root: Path, generated: Path, manifest: dict) -> None:
    destination = root / "docs/assets/images/qualitative"
    files = [generated / name for name in manifest["images"]]
    if len(files) != 15 or any(not path.is_file() for path in files):
        raise RuntimeError("Refusing to replace Qualitative images until all new images exist")
    destination.mkdir(parents=True, exist_ok=True)
    for old in destination.glob("*.png"):
        old.unlink()
    for path in files:
        shutil.copy2(path, destination / path.name)
    manifest_path = root / "docs/assets/data/qualitative_yolo26.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    results_path = root / "docs/assets/data/results.json"
    results = json.loads(results_path.read_text(encoding="utf-8"))
    results.setdefault("meta", {})["qualitative_scenes"] = list(SCENES)
    results["meta"]["qualitative_models"] = {
        key: manifest[key] for key in (
            "detection_baseline", "detection_finetuned", "segmentation_baseline", "segmentation_finetuned"
        )
    }
    results_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for YOLO26 fine-tuning")
    root = project_root()
    dataset = root / "output_dataset"
    work = root / ".mesh_cache/yolo26_qualitative"
    work.mkdir(parents=True, exist_ok=True)
    detection_yaml, segmentation_yaml = prepare_datasets(root, dataset, work)
    det_best = train_one(dataset / "ckpt/yolo26n.pt", detection_yaml,
                         work / "runs/detect", f"yolo26n_{args.epochs}ep",
                         args.epochs, args.batch, args.device)
    seg_best = train_one(root / ".mesh_cache/benchmark_instance_segmentation/checkpoints/yolo26n-seg.pt",
                         segmentation_yaml, work / "runs/segment", f"yolo26n_seg_{args.epochs}ep",
                         args.epochs, args.batch, args.device)
    manifest = generate_images(root, dataset, work, det_best, seg_best)
    manifest["epochs"] = args.epochs
    manifest["seed"] = SEED
    manifest["batch_requested"] = args.batch
    (work / "qualitative_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if args.publish:
        publish(root, work / "qualitative", manifest)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
