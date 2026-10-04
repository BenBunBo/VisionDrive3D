#!/usr/bin/env python3
"""Master script to run training/evaluation/backbone experiments and page asset sync."""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import cv2


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_dataset_root(dataset_arg: str, root: Path) -> Path:
    candidate = Path(dataset_arg).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    return candidate.resolve()


def run_python_script(root: Path, script_rel: str, args: Iterable[str]) -> None:
    cmd = [sys.executable, str(root / script_rel), *args]
    print(f"[RUN] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=root, check=True)


def safe_copy(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def build_lightweight_dataset_grid(
    dataset_root: Path,
    dst_path: Path,
    *,
    max_scenes: int = 12,
    cols: int = 4,
    tile_width: int = 360,
) -> bool:
    """Build a compact grid image for the project page to avoid oversized PNG files."""
    image_paths = sorted((dataset_root / "images").glob("scene_*.png"))
    if not image_paths:
        return False

    if len(image_paths) > max_scenes:
        step = max(1, len(image_paths) // max_scenes)
        image_paths = [image_paths[i] for i in range(0, len(image_paths), step)][:max_scenes]

    tiles = []
    tile_height = None
    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w = img.shape[:2]
        if h <= 0 or w <= 0:
            continue

        target_h = max(1, int(round((tile_width * h) / w)))
        tile = cv2.resize(img, (tile_width, target_h), interpolation=cv2.INTER_AREA)
        tile = cv2.copyMakeBorder(tile, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=(245, 245, 245))

        if tile_height is None:
            tile_height = tile.shape[0]
        elif tile.shape[0] != tile_height:
            tile = cv2.resize(tile, (tile.shape[1], tile_height), interpolation=cv2.INTER_AREA)

        tiles.append(tile)

    if not tiles:
        return False

    rows = math.ceil(len(tiles) / cols)
    blank = tiles[0].copy()
    blank[:] = 245
    while len(tiles) < rows * cols:
        tiles.append(blank.copy())

    row_images = [cv2.hconcat(tiles[r * cols : (r + 1) * cols]) for r in range(rows)]
    grid = cv2.vconcat(row_images)

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(dst_path), grid, [cv2.IMWRITE_PNG_COMPRESSION, 9]))


def ensure_dataset_grid(dataset_root: Path, root: Path) -> Path | None:
    previews_dir = dataset_root / "previews"
    candidate_paths = [
        previews_dir / "dataset_grid.png",
        previews_dir / "grid_overview.png",
    ]

    for candidate in candidate_paths:
        if candidate.exists():
            return candidate

    # Attempt generation via existing visualizer CLI.
    try:
        cmd = [
            sys.executable,
            "-m",
            "src.vis.dataset_visualizer",
            "--dataset",
            str(dataset_root),
            "--grid",
            "--output",
            str(previews_dir),
        ]
        print(f"[RUN] {' '.join(cmd)}")
        subprocess.run(cmd, cwd=root, check=True)
    except Exception as exc:
        print(f"[WARN] Failed to generate dataset grid preview: {exc}")

    for candidate in candidate_paths:
        if candidate.exists():
            return candidate

    return None


def sync_project_page_assets(dataset_root: Path, root: Path) -> None:
    page_root = root / "project_page"
    assets_data = page_root / "assets" / "data"
    assets_images = page_root / "assets" / "images"
    assets_qual = assets_images / "qualitative"
    assets_plots = page_root / "assets" / "plots"

    assets_data.mkdir(parents=True, exist_ok=True)
    assets_images.mkdir(parents=True, exist_ok=True)
    assets_qual.mkdir(parents=True, exist_ok=True)
    assets_plots.mkdir(parents=True, exist_ok=True)

    results_src = dataset_root / "results.json"
    backbone_csv_src = dataset_root / "backbone_comparison.csv"
    backbone_plot_src = dataset_root / "ai_results" / "backbone_comparison.png"

    if not safe_copy(results_src, assets_data / "results.json"):
        print(f"[WARN] Missing results.json at {results_src}")
    if not safe_copy(backbone_csv_src, assets_data / "backbone_comparison.csv"):
        print(f"[WARN] Missing backbone_comparison.csv at {backbone_csv_src}")
    if not safe_copy(backbone_plot_src, assets_plots / "backbone_comparison.png"):
        print(f"[WARN] Missing backbone comparison plot at {backbone_plot_src}")

    grid_dst = assets_images / "dataset_grid.png"
    grid_src = ensure_dataset_grid(dataset_root, root)
    max_grid_size_bytes = 12 * 1024 * 1024

    if grid_src is not None and grid_src.exists() and grid_src.stat().st_size <= max_grid_size_bytes:
        safe_copy(grid_src, grid_dst)
    else:
        built = build_lightweight_dataset_grid(dataset_root, grid_dst)
        if not built:
            if grid_src is not None:
                safe_copy(grid_src, grid_dst)
            print("[WARN] Dataset grid image not found or generated.")
        elif grid_src is not None and grid_src.exists():
            print(
                f"[INFO] Replaced large grid ({grid_src.stat().st_size / (1024 * 1024):.1f} MB) "
                f"with compact grid at {grid_dst}."
            )

    qualitative_src = dataset_root / "ai_results" / "qualitative"
    if qualitative_src.exists():
        for img_path in sorted(qualitative_src.glob("*.png")):
            safe_copy(img_path, assets_qual / img_path.name)
    else:
        print(f"[WARN] Qualitative directory not found: {qualitative_src}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full VisionDrive3D AI pipeline")
    parser.add_argument("--dataset", type=str, default="./output_dataset")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-backbone", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = project_root()
    dataset_root = resolve_dataset_root(args.dataset, root)

    dataset_arg = str(dataset_root)

    if not args.skip_train:
        run_python_script(root, "src/ai_demo/train.py", ["--dataset", dataset_arg, "--batch", "32"])
    else:
        print("[INFO] Skipping training step (--skip-train)")

    run_python_script(root, "src/ai_demo/evaluate.py", ["--dataset", dataset_arg])

    if not args.skip_backbone:
        run_python_script(root, "src/ai_demo/experiment_backbones.py", ["--dataset", dataset_arg])
    else:
        print("[INFO] Skipping backbone experiment step (--skip-backbone)")

    sync_project_page_assets(dataset_root, root)
    print("Done. Open project_page/index.html to view results.")


if __name__ == "__main__":
    main()
