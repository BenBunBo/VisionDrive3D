"""
Usage examples:
  # Visualize one scene:
  python -m src.vis.dataset_visualizer --dataset ./output_dataset --scene 0

  # Visualize all scenes:
  python -m src.vis.dataset_visualizer --dataset ./output_dataset --all

  # Save grid overview for report:
  python -m src.vis.dataset_visualizer --dataset ./output_dataset --grid
"""

import os
import cv2
import json
import glob
import math
import numpy as np
from pathlib import Path


class DatasetVisualizer:
    """
    Reads an exported dataset and produces annotated preview images.
    Never modifies the dataset — only reads images, labels, metadata.
    """

    def __init__(self, dataset_root: str, output_dir: str = None):
        self.dataset_root = Path(dataset_root)
        self.output_dir = Path(output_dir) if output_dir else self.dataset_root / "previews"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rgb_bbox_dir = self.output_dir / "rgb_bbox"
        self.mask_dir = self.output_dir / "mask"
        self.depth_dir = self.output_dir / "depth"
        self.rgb_bbox_dir.mkdir(parents=True, exist_ok=True)
        self.mask_dir.mkdir(parents=True, exist_ok=True)
        self.depth_dir.mkdir(parents=True, exist_ok=True)

        self.category_colors = {
            0: (0, 255, 0),     # car = green (BGR)
            1: (0, 255, 255),   # traffic_light = yellow
            2: (255, 0, 0),     # traffic_sign = blue
            3: (0, 0, 255),     # pedestrian = red
            99: (128, 128, 128) # entity = gray
        }

        # Fixed color table for masks
        self.mask_colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255),
            (255, 0, 255), (192, 192, 192), (128, 128, 128), (128, 0, 0), (128, 128, 0),
            (0, 128, 0), (128, 0, 128), (0, 128, 128), (0, 0, 128), (255, 128, 0),
            (255, 0, 128), (0, 255, 128), (128, 255, 0), (0, 128, 255), (128, 0, 255)
        ]

    def _get_mask_color(self, unique_val):
        """Deterministically map a unique mask value to a color from the table."""
        idx = (int(unique_val) * 7) % len(self.mask_colors)
        return self.mask_colors[idx]

    def draw_scene(self, scene_id: int) -> np.ndarray:
        """
        Load scene RGB image and overlay:
        - 2D bounding boxes (from COCO JSON) with class label + confidence
        - Instance ID or class name text above each box
        - Depth map as a semi-transparent viridis colormap strip
          (show as inset in bottom-right corner, 25% of image size)
        - Segmentation mask as a semi-transparent color overlay
          (alpha=0.3, each unique mask value gets a distinct color)
        Returns the annotated image as a numpy array (BGR, for cv2).
        """
        stem = f"scene_{scene_id:05d}"

        img_path = self.dataset_root / "images" / f"{stem}.png"
        coco_path = self.dataset_root / "labels" / "coco" / f"{stem}.json"
        depth_path = self.dataset_root / "depth" / f"{stem}.npy"
        mask_path = self.dataset_root / "masks" / f"{stem}.png"

        if not img_path.exists():
            raise FileNotFoundError(f"Image not found at: {img_path}")

        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]

        # 1. Mask overlay
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                color_mask = np.zeros_like(img)
                unique_vals = np.unique(mask)
                for val in unique_vals:
                    if val == 0:
                        continue  # skip background
                    color = self._get_mask_color(val)
                    color_mask[mask == val] = color

                # Blend with alpha=0.3
                alpha = 0.3
                mask_indices = mask > 0
                img[mask_indices] = cv2.addWeighted(img, 1.0 - alpha, color_mask, alpha, 0)[mask_indices]

        # 2. Bounding boxes from COCO
        if coco_path.exists():
            with open(coco_path, "r") as f:
                coco_data = json.load(f)

            categories = {c["id"]: c["name"] for c in coco_data.get("categories", [])}
            annotations = coco_data.get("annotations", [])

            for ann in annotations:
                x_bbox, y_bbox, bw, bh = ann["bbox"]
                cat_id = ann["category_id"]
                class_name = categories.get(cat_id, "unknown")
                color = self.category_colors.get(cat_id, self.category_colors[99])

                x1, y1 = int(x_bbox), int(y_bbox)
                x2, y2 = int(x_bbox + bw), int(y_bbox + bh)

                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

                label = f"{class_name}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 1

                (label_w, label_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
                cv2.rectangle(img, (x1, y1 - label_h - baseline - 2), (x1 + label_w + 2, y1), color, -1)
                cv2.putText(img, label, (x1 + 1, y1 - baseline - 1), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

        # 3. Depth inset
        if depth_path.exists():
            depth = np.load(str(depth_path))

            d_min, d_max = depth.min(), depth.max()
            if d_max > d_min:
                depth_norm = (depth - d_min) / (d_max - d_min)
            else:
                depth_norm = depth - d_min

            depth_8u = (depth_norm * 255).astype(np.uint8)
            depth_color = cv2.applyColorMap(depth_8u, cv2.COLORMAP_VIRIDIS)

            inset_w, inset_h = int(w * 0.25), int(h * 0.25)
            depth_inset = cv2.resize(depth_color, (inset_w, inset_h))

            bd = 2
            bordered_inset = cv2.copyMakeBorder(depth_inset, bd, bd, bd, bd, cv2.BORDER_CONSTANT, value=[255, 255, 255])

            fh, fw = bordered_inset.shape[:2]
            y_offset = h - fh
            x_offset = w - fw

            if y_offset >= 0 and x_offset >= 0:
                img[y_offset:y_offset+fh, x_offset:x_offset+fw] = bordered_inset

        return img

    def draw_rgb_with_bboxes(self, scene_id: int) -> np.ndarray:
        """
        Draw only bounding boxes on RGB image.
        """
        stem = f"scene_{scene_id:05d}"
        img_path = self.dataset_root / "images" / f"{stem}.png"
        coco_path = self.dataset_root / "labels" / "coco" / f"{stem}.json"

        if not img_path.exists():
            raise FileNotFoundError(f"Image not found at: {img_path}")

        img = cv2.imread(str(img_path))
        if coco_path.exists():
            with open(coco_path, "r") as f:
                coco_data = json.load(f)
            categories = {c["id"]: c["name"] for c in coco_data.get("categories", [])}
            annotations = coco_data.get("annotations", [])
            for ann in annotations:
                x_bbox, y_bbox, bw, bh = ann["bbox"]
                cat_id = ann["category_id"]
                class_name = categories.get(cat_id, "unknown")
                color = self.category_colors.get(cat_id, self.category_colors[99])

                x1, y1 = int(x_bbox), int(y_bbox)
                x2, y2 = int(x_bbox + bw), int(y_bbox + bh)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

                label = f"{class_name}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 1
                (label_w, label_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
                cv2.rectangle(img, (x1, y1 - label_h - baseline - 2), (x1 + label_w + 2, y1), color, -1)
                cv2.putText(img, label, (x1 + 1, y1 - baseline - 1), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
        return img

    def draw_mask(self, scene_id: int) -> np.ndarray:
        """
        Draw segmentation mask as a standalone colorized image.
        """
        stem = f"scene_{scene_id:05d}"
        mask_path = self.dataset_root / "masks" / f"{stem}.png"
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask not found at: {mask_path}")
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Failed to read mask at: {mask_path}")

        color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        unique_vals = np.unique(mask)
        for val in unique_vals:
            if val == 0:
                continue
            color_mask[mask == val] = self._get_mask_color(val)
        return color_mask

    def draw_depth(self, scene_id: int) -> np.ndarray:
        """
        Draw depth map as a standalone viridis image.
        """
        stem = f"scene_{scene_id:05d}"
        depth_path = self.dataset_root / "depth" / f"{stem}.npy"
        if not depth_path.exists():
            raise FileNotFoundError(f"Depth not found at: {depth_path}")
        depth = np.load(str(depth_path))
        d_min, d_max = depth.min(), depth.max()
        if d_max > d_min:
            depth_norm = (depth - d_min) / (d_max - d_min)
        else:
            depth_norm = depth - d_min
        depth_8u = (depth_norm * 255).astype(np.uint8)
        return cv2.applyColorMap(depth_8u, cv2.COLORMAP_VIRIDIS)

    def draw_depth_comparison(self, scene_id: int) -> np.ndarray:
        """
        Side-by-side: [RGB image] | [depth map colorized with viridis]
        Add a colorbar on the right showing min/max depth values in meters.
        Returns combined image as numpy array.
        """
        stem = f"scene_{scene_id:05d}"
        img_path = self.dataset_root / "images" / f"{stem}.png"
        depth_path = self.dataset_root / "depth" / f"{stem}.npy"

        if not img_path.exists() or not depth_path.exists():
            raise FileNotFoundError(f"Missing image or depth array for scene {scene_id}")

        img = cv2.imread(str(img_path))
        depth = np.load(str(depth_path))
        h, w = img.shape[:2]

        d_min, d_max = depth.min(), depth.max()
        if d_max > d_min:
            depth_norm = (depth - d_min) / (d_max - d_min)
        else:
            depth_norm = depth - d_min

        depth_8u = (depth_norm * 255).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_8u, cv2.COLORMAP_VIRIDIS)

        if depth_color.shape[:2] != (h, w):
            depth_color = cv2.resize(depth_color, (w, h))

        bar_w = 50
        colorbar = np.zeros((h, bar_w, 3), dtype=np.uint8)
        for y in range(h):
            val = int(255 * (1.0 - y / h))
            colorbar[y, :] = val
        colorbar = cv2.applyColorMap(colorbar, cv2.COLORMAP_VIRIDIS)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(colorbar, f"{d_max:.1f}m", (2, 20), font, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(colorbar, f"{d_min:.1f}m", (2, h - 10), font, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        combined = np.hstack([img, depth_color, colorbar])
        return combined

    def save_scene_preview(self, scene_id: int):
        """
        Saves three standalone outputs:
        - output_dir/rgb_bbox/scene_XXXXX_rgb_bbox.png
        - output_dir/mask/scene_XXXXX_mask.png
        - output_dir/depth/scene_XXXXX_depth.png
        """
        rgb_bbox = self.draw_rgb_with_bboxes(scene_id)
        mask = self.draw_mask(scene_id)
        depth = self.draw_depth(scene_id)

        rgb_path = self.rgb_bbox_dir / f"scene_{scene_id:05d}_rgb_bbox.png"
        mask_path = self.mask_dir / f"scene_{scene_id:05d}_mask.png"
        depth_path = self.depth_dir / f"scene_{scene_id:05d}_depth.png"

        cv2.imwrite(str(rgb_path), rgb_bbox)
        cv2.imwrite(str(mask_path), mask)
        cv2.imwrite(str(depth_path), depth)
        print(f"Saved {rgb_path}")
        print(f"Saved {mask_path}")
        print(f"Saved {depth_path}")

    def save_all_previews(self, max_scenes: int = None):
        """
        Runs save_scene_preview() for every scene in the dataset.
        Prints progress: [1/10] scene_00000_preview.png
        """
        img_paths = sorted(glob.glob(str(self.dataset_root / "images" / "scene_*.png")))
        if not img_paths:
            print("No scenes found.")
            return

        scene_ids = []
        for p in img_paths:
            stem = Path(p).stem
            scene_ids.append(int(stem.split("_")[1]))

        if max_scenes:
            scene_ids = scene_ids[:max_scenes]

        total = len(scene_ids)
        for i, sid in enumerate(scene_ids):
            self.save_scene_preview(sid)
            print(f"[{i+1}/{total}] scene_{sid:05d}_preview.png")

    def save_grid(self, scene_ids: list[int] = None, cols: int = 4):
        """
        Arrange multiple scene previews in a grid image.
        Saves to output_dir/previews/grid_overview.png
        Useful for showing dataset variety in the report.
        """
        if scene_ids is None:
            img_paths = sorted(glob.glob(str(self.dataset_root / "images" / "scene_*.png")))
            scene_ids = [int(Path(p).stem.split("_")[1]) for p in img_paths]

        if not scene_ids:
            print("No scenes available for grid.")
            return

        images = []
        for sid in scene_ids:
            images.append(self.draw_scene(sid))

        if not images:
            return

        h, w, c = images[0].shape
        n = len(images)
        rows = math.ceil(n / cols)

        grid_w = cols * w
        grid_h = rows * h
        grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

        for idx, img in enumerate(images):
            r = idx // cols
            c_idx = idx % cols

            if img.shape[:2] != (h, w):
                img = cv2.resize(img, (w, h))

            grid[r*h:(r+1)*h, c_idx*w:(c_idx+1)*w] = img

        out_path = self.output_dir / "grid_overview.png"
        cv2.imwrite(str(out_path), grid)
        print(f"Saved grid to {out_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--scene", type=int, default=None,
                        help="visualize single scene ID")
    parser.add_argument("--all", action="store_true",
                        help="visualize all scenes")
    parser.add_argument("--grid", action="store_true",
                        help="save grid overview")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    vis = DatasetVisualizer(args.dataset, args.output)
    if args.scene is not None:
        vis.save_scene_preview(args.scene)
    if args.all:
        vis.save_all_previews()
    if args.grid:
        vis.save_grid(scene_ids=None, cols=4)
