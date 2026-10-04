#!/usr/bin/env python3
"""Training pipeline for VisionDrive3D YOLO detection + segmentation."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import cv2
import numpy as np

SEED = 42
SCENE_IMAGE_RE = re.compile(r"scene_(\d+)\.png$")


@dataclass(frozen=True)
class SplitData:
    train: List[Path]
    val: List[Path]
    test: List[Path]


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_dataset_root(dataset_arg: str, root: Path) -> Path:
    candidate = Path(dataset_arg).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    return candidate.resolve()


def scene_index_from_name(name: str) -> int:
    match = SCENE_IMAGE_RE.fullmatch(name)
    if not match:
        raise ValueError(f"Unsupported scene filename format: {name}")
    return int(match.group(1))


def collect_scene_images(images_root: Path) -> List[Path]:
    image_paths: List[Path] = []
    for path in images_root.glob("scene_*.png"):
        if not path.is_file() or path.parent != images_root:
            continue
        if SCENE_IMAGE_RE.fullmatch(path.name) is None:
            continue
        image_paths.append(path)
    image_paths.sort(key=lambda p: scene_index_from_name(p.name))
    return image_paths


def split_images(image_paths: List[Path], seed: int = SEED) -> SplitData:
    indices = list(range(len(image_paths)))
    rng = random.Random(seed)
    rng.shuffle(indices)

    n_total = len(indices)
    n_train = int(n_total * 0.8)
    n_val = int(n_total * 0.1)
    n_test = n_total - n_train - n_val

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val : n_train + n_val + n_test]

    def gather(idxs: Iterable[int]) -> List[Path]:
        subset = [image_paths[i] for i in idxs]
        subset.sort(key=lambda p: scene_index_from_name(p.name))
        return subset

    return SplitData(train=gather(train_idx), val=gather(val_idx), test=gather(test_idx))


def _remove_stale_split_links(split_dir: Path, expected_names: set[str]) -> None:
    if not split_dir.exists():
        return
    for path in split_dir.glob("scene_*"):
        if path.name in expected_names:
            continue
        if path.is_symlink():
            path.unlink()


def _ensure_relative_symlink(link_path: Path, target_path: Path) -> None:
    target_rel = os.path.relpath(target_path, start=link_path.parent)

    if link_path.is_symlink():
        if os.readlink(link_path) == target_rel:
            return
        link_path.unlink()
    elif link_path.exists():
        raise FileExistsError(
            f"Cannot create symlink because non-link path already exists: {link_path}"
        )

    link_path.symlink_to(target_rel)


def prepare_split_symlinks(dataset_root: Path, split_data: SplitData) -> None:
    images_root = dataset_root / "images"
    labels_root = dataset_root / "labels"
    yolo_root = dataset_root / "labels" / "yolo"

    split_map: Dict[str, List[Path]] = {
        "train": split_data.train,
        "val": split_data.val,
        "test": split_data.test,
    }

    for split_name, split_images_list in split_map.items():
        split_img_dir = images_root / split_name
        split_lbl_dir = yolo_root / split_name
        split_lbl_dir_flat = labels_root / split_name
        split_img_dir.mkdir(parents=True, exist_ok=True)
        split_lbl_dir.mkdir(parents=True, exist_ok=True)
        split_lbl_dir_flat.mkdir(parents=True, exist_ok=True)

        expected_img_names = {img_path.name for img_path in split_images_list}
        expected_lbl_names = {f"{img_path.stem}.txt" for img_path in split_images_list}

        _remove_stale_split_links(split_img_dir, expected_img_names)
        _remove_stale_split_links(split_lbl_dir, expected_lbl_names)
        _remove_stale_split_links(split_lbl_dir_flat, expected_lbl_names)

        for img_path in split_images_list:
            lbl_path = yolo_root / f"{img_path.stem}.txt"
            if not lbl_path.exists():
                raise FileNotFoundError(f"Missing YOLO label for {img_path.name}: {lbl_path}")

            _ensure_relative_symlink(split_img_dir / img_path.name, img_path)
            _ensure_relative_symlink(split_lbl_dir / lbl_path.name, lbl_path)
            _ensure_relative_symlink(split_lbl_dir_flat / lbl_path.name, lbl_path)


def write_dataset_yaml(dataset_root: Path) -> Path:
    dataset_yaml = dataset_root / "dataset.yaml"
    yaml_content = (
        f"path: {dataset_root}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "nc: 1\n"
        "names: ['car']\n"
    )
    dataset_yaml.write_text(yaml_content, encoding="utf-8")
    return dataset_yaml


def _write_text_if_changed(path: Path, content: str) -> None:
    if path.exists() and path.read_text(encoding="utf-8") == content:
        return
    path.write_text(content, encoding="utf-8")


def _bbox_line_to_polygon(tokens: list[str]) -> str:
    cls_id = int(float(tokens[0]))
    cx, cy, bw, bh = (float(v) for v in tokens[1:5])

    x1 = cx - bw / 2.0
    y1 = cy - bh / 2.0
    x2 = cx + bw / 2.0
    y2 = cy + bh / 2.0

    return (
        f"{cls_id} "
        f"{x1:.6f} {y1:.6f} "
        f"{x2:.6f} {y1:.6f} "
        f"{x2:.6f} {y2:.6f} "
        f"{x1:.6f} {y2:.6f}"
    )


def _parse_yolo_boxes(label_path: Path, img_w: int, img_h: int) -> list[tuple[int, int, int, int]]:
    boxes: list[tuple[int, int, int, int]] = []
    if not label_path.exists():
        return boxes

    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        tokens = line.split()
        if len(tokens) < 5:
            continue

        cls_id = int(float(tokens[0]))
        if cls_id != 0:
            continue

        cx, cy, bw, bh = (float(v) for v in tokens[1:5])
        x1 = int(round((cx - bw / 2.0) * img_w))
        y1 = int(round((cy - bh / 2.0) * img_h))
        x2 = int(round((cx + bw / 2.0) * img_w))
        y2 = int(round((cy + bh / 2.0) * img_h))

        x1 = max(0, min(img_w - 1, x1))
        y1 = max(0, min(img_h - 1, y1))
        x2 = max(0, min(img_w, x2))
        y2 = max(0, min(img_h, y2))

        if x2 > x1 and y2 > y1:
            boxes.append((x1, y1, x2, y2))

    return boxes


def _contour_inside_box_ratio(
    contour: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    shape_hw: tuple[int, int],
) -> float:
    if not boxes:
        return 0.0

    contour_area = float(cv2.contourArea(contour))
    if contour_area <= 0:
        return 0.0

    mask = np.zeros(shape_hw, dtype=np.uint8)
    cv2.drawContours(mask, [contour], contourIdx=-1, color=1, thickness=-1)

    best_ratio = 0.0
    for x1, y1, x2, y2 in boxes:
        inside = float(mask[y1:y2, x1:x2].sum())
        ratio = inside / contour_area
        if ratio > best_ratio:
            best_ratio = ratio
    return best_ratio


def build_segmentation_labels(dataset_root: Path) -> Path:
    labels_root = dataset_root / "labels"
    yolo_root = labels_root / "yolo"
    masks_root = dataset_root / "masks"
    seg_root = dataset_root / "labels_seg"
    split_names = ("train", "val", "test")
    epsilon_factor = 0.002
    min_area_px = 100
    min_inside_ratio_when_no_black_bg = 0.35

    for split_name in split_names:
        src_dir = yolo_root / split_name
        dst_dir = seg_root / split_name
        dst_dir.mkdir(parents=True, exist_ok=True)

        expected_names: set[str] = set()
        for src_path in sorted(src_dir.glob("scene_*.txt")):
            expected_names.add(src_path.name)
            mask_path = masks_root / f"{src_path.stem}.png"
            if not mask_path.exists() and src_path.is_symlink():
                mask_path = masks_root / f"{src_path.resolve().stem}.png"
            if not mask_path.exists():
                raise FileNotFoundError(f"Missing mask for split label {src_path}: {mask_path}")

            mask = cv2.imread(str(mask_path), cv2.IMREAD_COLOR)
            if mask is None:
                raise RuntimeError(f"Failed to read mask image: {mask_path}")

            img_h, img_w = mask.shape[:2]
            yolo_boxes = _parse_yolo_boxes(src_path, img_w=img_w, img_h=img_h)
            has_black_background = bool(np.any(np.all(mask == 0, axis=2)))
            unique_colors = np.unique(mask.reshape(-1, 3), axis=0)

            polygon_lines: list[str] = []
            for color in unique_colors:
                # Skip canonical black background when present.
                if color.tolist() == [0, 0, 0]:
                    continue

                instance_mask = cv2.inRange(mask, color, color)
                contours, _ = cv2.findContours(
                    instance_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )

                for contour in contours:
                    area = float(cv2.contourArea(contour))
                    if area < min_area_px:
                        continue

                    # If the mask has no black background, keep only contours that
                    # significantly overlap a known car detection box.
                    if not has_black_background and yolo_boxes:
                        inside_ratio = _contour_inside_box_ratio(
                            contour=contour,
                            boxes=yolo_boxes,
                            shape_hw=(img_h, img_w),
                        )
                        if inside_ratio < min_inside_ratio_when_no_black_bg:
                            continue

                    epsilon = epsilon_factor * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    if len(approx) < 3:
                        continue

                    coords = approx.reshape(-1, 2).astype(np.float32)
                    coords[:, 0] /= float(img_w)
                    coords[:, 1] /= float(img_h)
                    coords = np.clip(coords, 0.0, 1.0)
                    coord_str = " ".join(f"{x:.6f} {y:.6f}" for x, y in coords)
                    polygon_lines.append(f"0 {coord_str}")

            content = ("\n".join(polygon_lines) + "\n") if polygon_lines else ""
            _write_text_if_changed(dst_dir / src_path.name, content)

        for stale_path in dst_dir.glob("scene_*.txt"):
            if stale_path.name not in expected_names and stale_path.is_file():
                stale_path.unlink()

    return seg_root


def _ensure_split_symlink_tree(src_dir: Path, dst_dir: Path, pattern: str) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    src_paths = sorted([p for p in src_dir.glob(pattern) if p.is_file()])
    expected_names = {p.name for p in src_paths}
    _remove_stale_split_links(dst_dir, expected_names)
    for src_path in src_paths:
        _ensure_relative_symlink(dst_dir / src_path.name, src_path)


def write_seg_dataset_yaml(dataset_root: Path, seg_labels_root: Path) -> Path:
    seg_dataset_root = dataset_root / "_seg_dataset"
    seg_dataset_root.mkdir(parents=True, exist_ok=True)

    images_root = seg_dataset_root / "images"
    labels_root = seg_dataset_root / "labels"

    # Migrate old symlink-based layout if present.
    if images_root.is_symlink():
        images_root.unlink()
    if labels_root.is_symlink():
        labels_root.unlink()
    images_root.mkdir(parents=True, exist_ok=True)
    labels_root.mkdir(parents=True, exist_ok=True)

    for split_name in ("train", "val", "test"):
        _ensure_split_symlink_tree(
            src_dir=dataset_root / "images" / split_name,
            dst_dir=images_root / split_name,
            pattern="scene_*.png",
        )
        _ensure_split_symlink_tree(
            src_dir=seg_labels_root / split_name,
            dst_dir=labels_root / split_name,
            pattern="scene_*.txt",
        )

    dataset_yaml = dataset_root / "dataset_seg.yaml"
    yaml_content = (
        f"path: {seg_dataset_root}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "nc: 1\n"
        "names: ['car']\n"
    )
    _write_text_if_changed(dataset_yaml, yaml_content)
    return dataset_yaml


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


def run_training(
    root: Path,
    dataset_yaml: Path,
    epochs: int,
    imgsz: int,
    batch: int,
    device: str,
) -> dict:
    configure_ultralytics(root)
    runtime_device = resolve_runtime_device(device)

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "ultralytics is not installed. Install it in the active environment first."
        ) from exc

    det_model = YOLO(resolve_pretrained_weight(root, "yolov8n.pt"))
    det_model.train(
        data=str(dataset_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=runtime_device,
        project="runs/detect",
        name="yolov8n_finetuned",
        exist_ok=True,
        seed=SEED,
        val=True,
    )

    dataset_root = dataset_yaml.parent
    seg_labels_root = build_segmentation_labels(dataset_root)
    seg_dataset_yaml = write_seg_dataset_yaml(dataset_root, seg_labels_root)

    seg_model = YOLO(resolve_pretrained_weight(root, "yolov8n-seg.pt"))
    seg_model.train(
        data=str(seg_dataset_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=runtime_device,
        project="runs/seg",
        name="yolov8n_seg_finetuned",
        exist_ok=True,
        seed=SEED,
    )

    det_best = Path(det_model.trainer.best).resolve()
    seg_best = Path(seg_model.trainer.best).resolve()

    def _relative_or_abs(path: Path, base: Path) -> str:
        try:
            return str(path.relative_to(base))
        except ValueError:
            return str(path)

    trained_models = {
        "detection": _relative_or_abs(det_best, root),
        "segmentation": _relative_or_abs(seg_best, root),
    }
    return trained_models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO models on VisionDrive3D dataset")
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

    images_root = dataset_root / "images"
    if not images_root.exists():
        raise FileNotFoundError(f"Images directory not found: {images_root}")

    image_paths = collect_scene_images(images_root)
    if not image_paths:
        raise RuntimeError(f"No scene_XXXXX.png files found under: {images_root}")

    split_data = split_images(image_paths, seed=SEED)
    prepare_split_symlinks(dataset_root, split_data)

    print(
        f"Dataset split: {len(split_data.train)} train | "
        f"{len(split_data.val)} val | {len(split_data.test)} test"
    )

    dataset_yaml = write_dataset_yaml(dataset_root)
    print(f"Wrote dataset config: {dataset_yaml}")

    trained_models = run_training(
        root=root,
        dataset_yaml=dataset_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
    )

    trained_models_path = dataset_root / "trained_models.json"
    trained_models_path.write_text(json.dumps(trained_models, indent=2), encoding="utf-8")
    print(f"Wrote trained model paths: {trained_models_path}")


if __name__ == "__main__":
    main()
