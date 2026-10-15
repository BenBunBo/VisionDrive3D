#!/usr/bin/env python3
"""Repair palette-7 overlay pixels by depth and prepare YOLO semantic masks.

Run preflight: xvfb-run -a ./venv/bin/python src/ai_demo/prepare_semantic_dataset.py --preflight
Prepare after reviewing preflight: xvfb-run -a ./venv/bin/python src/ai_demo/prepare_semantic_dataset.py --prepare
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("PYOPENGL_PLATFORM", "")
sys.path.insert(0, str(ROOT))

import cv2
import glfw
import numpy as np
import OpenGL.GL as GL

from libs.shader import Shader
from src.scene.scene_overlay import SceneOverlay

SOURCE = ROOT / "output_dataset"
WORK = ROOT / ".mesh_cache/semantic_dataset_v1"
WIDTH, HEIGHT = 1600, 900
NEAR, FAR = 0.1, 150.0
INSTANCE_COLORS_BGR = [tuple((i * 2654435761 >> shift) & 255 for shift in (0, 8, 16)) for i in range(1, 8)]


def scene_ids() -> dict[str, list[str]]:
    result = {}
    for split in ("train", "val", "test"):
        ids = [v.strip() for v in (SOURCE / "ImageSets" / f"{split}.txt").read_text().splitlines() if v.strip()]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate IDs in {split} split")
        result[split] = ids
    if any(set(result[a]) & set(result[b]) for i, a in enumerate(result) for b in list(result)[i + 1:]):
        raise ValueError("ImageSets splits overlap")
    return result


def make_projection(k: np.ndarray, w: int, h: int, near: float, far: float) -> np.ndarray:
    fx, fy, cx, cy = float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2])
    p = np.zeros((4, 4), dtype=np.float32)
    p[0, 0], p[1, 1] = 2 * fx / w, 2 * fy / h
    p[0, 2], p[1, 2] = 1 - 2 * cx / w, 2 * cy / h - 1
    p[2, 2] = -(far + near) / (far - near)
    p[2, 3] = -2 * far * near / (far - near)
    p[3, 2] = -1
    return p


def open_renderer():
    if not glfw.init():
        raise RuntimeError("GLFW initialization failed; run under xvfb-run on headless hosts")
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.DEPTH_BITS, 24)
    window = glfw.create_window(WIDTH, HEIGHT, "VisionDrive3D semantic mask repair", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("Could not create OpenGL 3.3 context")
    glfw.make_context_current(window)

    fbo = GL.glGenFramebuffers(1)
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)
    color = GL.glGenTextures(1)
    GL.glBindTexture(GL.GL_TEXTURE_2D, color)
    GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGB8, WIDTH, HEIGHT, 0, GL.GL_RGB, GL.GL_UNSIGNED_BYTE, None)
    GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, color, 0)
    depth = GL.glGenRenderbuffers(1)
    GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, depth)
    GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, GL.GL_DEPTH_COMPONENT24, WIDTH, HEIGHT)
    GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER, GL.GL_DEPTH_ATTACHMENT, GL.GL_RENDERBUFFER, depth)
    status = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
    if status != GL.GL_FRAMEBUFFER_COMPLETE:
        raise RuntimeError(f"Overlay FBO incomplete: {status}")
    GL.glViewport(0, 0, WIDTH, HEIGHT)
    GL.glEnable(GL.GL_DEPTH_TEST)
    GL.glDepthFunc(GL.GL_LESS)
    shader = Shader(str(ROOT / "shaders/mask.vert"), str(ROOT / "shaders/mask.frag"))
    with contextlib.redirect_stdout(io.StringIO()):
        overlay = SceneOverlay()
    return window, fbo, color, depth, shader, overlay


def close_renderer(handles):
    window, fbo, color, depth, shader, _ = handles
    GL.glDeleteTextures([color])
    GL.glDeleteRenderbuffers(1, [depth])
    GL.glDeleteFramebuffers(1, [fbo])
    del shader
    glfw.destroy_window(window)
    glfw.terminate()


def render_overlay_depth(meta: dict, shader, overlay) -> np.ndarray:
    width, height = map(int, meta["render_config"]["resolution"])
    if (width, height) != (WIDTH, HEIGHT):
        raise ValueError(f"Unexpected render resolution {(width, height)}")
    camera = meta["camera"]
    near, far = float(camera["near"]), float(camera["far"])
    k = np.asarray(camera["intrinsic_matrix"], dtype=np.float32)
    view = np.asarray(camera["extrinsic_matrix"], dtype=np.float32)
    projection = make_projection(k, width, height, near, far)

    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, GL.glGetIntegerv(GL.GL_FRAMEBUFFER_BINDING))
    GL.glViewport(0, 0, width, height)
    GL.glClearColor(0, 0, 0, 1)
    GL.glClearDepth(1.0)
    GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
    GL.glEnable(GL.GL_DEPTH_TEST)
    GL.glDepthFunc(GL.GL_LESS)
    overlay.render(shader, projection, view, is_rgb=False)
    raw = GL.glReadPixels(0, 0, width, height, GL.GL_DEPTH_COMPONENT, GL.GL_FLOAT)
    raw = np.frombuffer(raw, dtype=np.float32).reshape(height, width)[::-1].copy()
    denom = far + near - (raw * 2.0 - 1.0) * (far - near)
    metric = (2.0 * near * far) / np.maximum(denom, 1e-12)
    metric[raw >= 1.0] = np.inf
    return metric


def vehicle7_bbox(meta: dict) -> tuple[int, int, int, int] | None:
    cars = [o for o in meta["objects"] if o.get("class_name") == "Car" and o.get("annotate", True)]
    if len(cars) < 7:
        return None
    x1, y1, x2, y2 = map(int, cars[6]["bbox_2d"])
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def process_scene(split: str, sid: str, overlay_depth: np.ndarray | None, *, write_mask: bool) -> dict:
    stem = f"scene_{sid}"
    meta = json.loads((SOURCE / "metadata" / f"{stem}.json").read_text())
    image = cv2.imread(str(SOURCE / "images" / f"{stem}.png"), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(SOURCE / "masks" / f"{stem}.png"), cv2.IMREAD_COLOR)
    depth = np.load(SOURCE / "depth" / f"{stem}.npy", mmap_mode="r")
    if image is None or mask is None or image.shape[:2] != (HEIGHT, WIDTH) or mask.shape != image.shape:
        raise ValueError(f"Image/mask missing or dimensions differ for {stem}")
    if depth.shape != (HEIGHT, WIDTH) or not np.isfinite(depth).all():
        raise ValueError(f"Depth missing, invalid, or different dimensions for {stem}")

    id7 = np.all(mask == INSTANCE_COLORS_BGR[6], axis=2)
    if overlay_depth is None:
        return {"scene": stem, "id7_pixels": int(id7.sum())}

    # One 24-bit depth-buffer ULP converted to metric depth, with two ULPs
    # plus float32 storage tolerance for equality with the saved RGB depth.
    tol = 2.0 * overlay_depth**2 * (FAR - NEAR) / (2.0 * NEAR * FAR * (2**24)) + 1e-6
    rendered_overlay = np.isfinite(overlay_depth)
    same_surface = rendered_overlay & (np.abs(overlay_depth - depth) <= tol)
    # The renderer assigns pool slot i the palette ID i+1, and metadata
    # serializes those same entities in pool order. Its projected 2D box gives
    # the support of car 7; depth separates car pixels from overlay pixels
    # where their regions overlap.
    box = vehicle7_bbox(meta)
    in_car7_box = np.zeros((HEIGHT, WIDTH), dtype=bool)
    if box is not None:
        x1, y1, x2, y2 = box
        in_car7_box[y1:y2, x1:x2] = True
    keep_car7 = id7 & in_car7_box & ~same_surface
    repair = id7 & ~keep_car7
    report = {
        "scene": stem,
        "palette7_pixels": int(id7.sum()),
        "palette7_inside_car7_projected_box": int((id7 & in_car7_box).sum()),
        "overlay_depth_matches": int((id7 & same_surface).sum()),
        "palette7_outside_car7_projected_box_relabelled_background": int((id7 & ~in_car7_box).sum()),
        "palette7_inside_box_without_overlay_depth_match_kept_as_car": int(keep_car7.sum()),
    }
    out = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    # All renderer car palette IDs map to semantic car. Only ID 7 needs depth
    # disambiguation; other palette values are known car-only instance colors.
    for color in INSTANCE_COLORS_BGR[:6]:
        out[np.all(mask == color, axis=2)] = 1
    out[keep_car7] = 1
    out[repair] = 0
    if np.any((out != 0) & (out != 1)):
        raise ValueError(f"Unexpected semantic labels in {stem}")
    report["semantic_car_pixels"] = int((out == 1).sum())
    report["semantic_background_pixels"] = int((out == 0).sum())
    if write_mask:
        target = WORK / "masks" / split / f"{stem}.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(target), out):
            raise OSError(f"Could not write {target}")
    return report


def validate_existing_dataset() -> None:
    splits = scene_ids()
    report = json.loads((WORK / "mask_repair_report.json").read_text())
    if len(report.get("scenes", [])) != 1000:
        raise ValueError("The saved repair report does not cover all 1,000 scenes")
    pixel_counts = {"background": 0, "car": 0}
    checksums = {"images": hashlib.sha256(), "semantic_masks": hashlib.sha256()}
    for split, ids in splits.items():
        for sid in ids:
            name = f"scene_{sid}.png"
            image = WORK / "images" / split / name
            mask_path = WORK / "masks" / split / name
            expected_image = (SOURCE / "images" / name).resolve()
            if not image.is_symlink() or image.resolve() != expected_image:
                raise ValueError(f"RGB link is missing, copied, or points to the wrong scene: {image}")
            label = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if label is None or label.shape != (HEIGHT, WIDTH):
                raise ValueError(f"Semantic mask is missing or has the wrong shape: {mask_path}")
            values, counts = np.unique(label, return_counts=True)
            if not set(values.tolist()).issubset({0, 1}):
                raise ValueError(f"Semantic mask contains classes outside {{0, 1}}: {mask_path}")
            pixel_counts["background"] += int(counts[values == 0].sum())
            pixel_counts["car"] += int(counts[values == 1].sum())
            for key, path in (("images", expected_image), ("semantic_masks", mask_path)):
                checksums[key].update(name.encode())
                with path.open("rb") as stream:
                    while chunk := stream.read(1024 * 1024):
                        checksums[key].update(chunk)
    manifest = json.loads((WORK / "manifest.json").read_text())
    manifest["images_sha256"] = checksums["images"].hexdigest()
    manifest["semantic_masks_sha256"] = checksums["semantic_masks"].hexdigest()
    manifest["pixel_counts"] = pixel_counts
    (WORK / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"split_counts": {k: len(v) for k, v in splits.items()},
                      "pixel_counts": pixel_counts, "images_sha256": manifest["images_sha256"],
                      "semantic_masks_sha256": manifest["semantic_masks_sha256"]}, indent=2), flush=True)


def run(preflight: bool, prepare: bool) -> None:
    splits = scene_ids()
    jobs = [(split, sid) for split, ids in splits.items() for sid in ids]
    if not jobs or len(jobs) != 1000:
        raise ValueError(f"Expected the existing 1,000-scene dataset, found {len(jobs)}")
    if preflight:
        sample_ids = {"00025", "00030", "00046", "00733"}
        for split, ids in splits.items():
            for sid in ids:
                if sid in sample_ids:
                    continue
                meta = json.loads((SOURCE / "metadata" / f"scene_{sid}.json").read_text())
                cars = [o for o in meta["objects"] if o.get("class_name") == "Car" and o.get("annotate", True)]
                if len(cars) >= 7 and cars[6].get("bbox_2d", [0, 0, 0, 0]) != [0, 0, 0, 0]:
                    sample_ids.add(sid)
                    break
            if len(sample_ids) >= 5:
                break
        jobs = [(split, sid) for split, sid in jobs if sid in sample_ids]
    output = []
    handles = open_renderer()
    try:
        _, _, _, _, shader, overlay = handles
        for index, (split, sid) in enumerate(jobs):
            meta = json.loads((SOURCE / "metadata" / f"scene_{sid}.json").read_text())
            overlay_depth = render_overlay_depth(meta, shader, overlay)
            row = process_scene(split, sid, overlay_depth, write_mask=prepare)
            output.append(row)
            if (index + 1) % 25 == 0:
                print(f"Replayed overlay depth for {index + 1}/{len(jobs)} scenes", flush=True)
    finally:
        close_renderer(handles)

    report_path = WORK / "mask_repair_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps({"scene_count": len(output), "preflight": preflight, "scenes": output}, indent=2) + "\n")
    preserved_car7 = sum(row["palette7_inside_box_without_overlay_depth_match_kept_as_car"] for row in output)
    print(json.dumps({
        "scenes": len(output), "preflight": preflight,
        "id7_pixels": sum(row["palette7_pixels"] for row in output),
        "overlay_pixels_repaired": sum(row["overlay_depth_matches"] for row in output),
        "car7_pixels_preserved_using_metadata_and_depth": preserved_car7,
        "report": str(report_path),
    }, indent=2), flush=True)
    if preflight and len(output) < 5:
        raise RuntimeError("Preflight must cover the qualitative scenes and one visible vehicle 7")
    if prepare:
        for split, ids in splits.items():
            (WORK / "images" / split).mkdir(parents=True, exist_ok=True)
            for sid in ids:
                src = (SOURCE / "images" / f"scene_{sid}.png").resolve()
                dst = WORK / "images" / split / src.name
                if dst.is_symlink() and dst.resolve() == src:
                    continue
                if dst.exists() or dst.is_symlink():
                    dst.unlink()
                dst.symlink_to(src)
        (WORK / "dataset.yaml").write_text(
            f"path: {WORK}\ntrain: images/train\nval: images/val\ntest: images/test\n"
            "masks_dir: masks\nnc: 2\nnames:\n  0: background\n  1: car\n",
            encoding="utf-8",
        )
        def checksum_tree(kind: str) -> str:
            digest = hashlib.sha256()
            for split, ids in splits.items():
                for sid in ids:
                    path = WORK / kind / split / f"scene_{sid}.png"
                    digest.update(path.name.encode())
                    with path.open("rb") as stream:
                        while chunk := stream.read(1024 * 1024):
                            digest.update(chunk)
            return digest.hexdigest()

        for split, ids in splits.items():
            for sid in ids:
                mask_path = WORK / "masks" / split / f"scene_{sid}.png"
                label = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if label is None or label.shape != (HEIGHT, WIDTH) or not set(np.unique(label)).issubset({0, 1}):
                    raise ValueError(f"Prepared semantic mask failed final validation: {mask_path}")
                image_link = WORK / "images" / split / f"scene_{sid}.png"
                if not image_link.is_symlink() or image_link.resolve() != (SOURCE / "images" / image_link.name).resolve():
                    raise ValueError(f"Prepared RGB link failed validation: {image_link}")
        manifest = {
            "dataset": str(WORK.relative_to(ROOT)), "source_dataset": "output_dataset",
            "split_counts": {k: len(v) for k, v in splits.items()},
            "images_sha256": checksum_tree("images"), "semantic_masks_sha256": checksum_tree("masks"),
            "mask_encoding": {"background": 0, "car": 1},
            "imgsz": 640, "renderer_replay": "overlay depth compared to saved RGB metric depth",
            "repair_report": str(report_path.relative_to(ROOT)),
        }
        (WORK / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"Prepared semantic dataset at {WORK}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight", action="store_true", help="Render and audit every scene; write masks in cache")
    group.add_argument("--prepare", action="store_true", help="Run full repair, then prepare images, YAML, and manifest")
    group.add_argument("--validate-only", action="store_true", help="Validate an existing prepared dataset and update content hashes")
    args = parser.parse_args()
    if args.validate_only:
        validate_existing_dataset()
    else:
        run(preflight=args.preflight, prepare=args.prepare)


if __name__ == "__main__":
    main()
