# annotations.py
import os
import json
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import OpenGL.GL as GL
from PIL import Image


# ---------------------------------------------------------------------------
# Shared per-entity geometry helpers (used by BOTH the interactive export
# hook and the headless dataset generator).
# ---------------------------------------------------------------------------

def _entity_scale(entity) -> np.ndarray:
    raw_scale = getattr(entity, "scale", [1.0, 1.0, 1.0])
    if np.isscalar(raw_scale):
        return np.array([raw_scale] * 3, dtype=np.float32)
    scale = np.atleast_1d(np.array(raw_scale, dtype=np.float32))
    if scale.shape[0] == 1:
        scale = np.array([scale[0]] * 3, dtype=np.float32)
    return scale


def _entity_yaw_rad(entity) -> float:
    rot = getattr(entity, "rotation_euler", getattr(entity, "rotation", [0, 0, 0]))
    ry_deg = float(np.atleast_1d(rot)[1])
    return float(np.deg2rad(ry_deg))


def _rotation_y(ry_rad: float) -> np.ndarray:
    cr, sr = np.cos(ry_rad), np.sin(ry_rad)
    return np.array([[cr, 0, sr], [0, 1, 0], [-sr, 0, cr]], dtype=np.float32)


def _scaled_local_box(entity) -> Tuple[np.ndarray, np.ndarray]:
    """
    (bmin, bmax) of the entity in *scaled* local space.
    Prefers the mesh local AABB; falls back to mesh_extents / class defaults.
    """
    scale = _entity_scale(entity)

    local_min = getattr(entity, "local_aabb_min", None)
    local_max = getattr(entity, "local_aabb_max", None)
    if local_min is not None and local_max is not None:
        lmin = np.atleast_1d(np.array(local_min, dtype=np.float32))[:3]
        lmax = np.atleast_1d(np.array(local_max, dtype=np.float32))[:3]
        if (
            lmin.shape[0] >= 3 and lmax.shape[0] >= 3
            and np.all(np.isfinite(lmin)) and np.all(np.isfinite(lmax))
            and np.any(np.abs(lmax - lmin) > 1e-6)
        ):
            return lmin * scale[:3], lmax * scale[:3]

    # Fallback: half-extents centered at origin
    extents = getattr(entity, "mesh_extents", None)
    if extents is None or np.allclose(extents, 0):
        key = getattr(entity, "class_name", type(entity).__name__).lower().replace("_", "")
        defaults = {
            "car":          (1.0, 0.75, 2.0),
            "trafficlight": (0.2, 0.6,  0.2),
            "trafficsign":  (0.3, 0.3,  0.1),
            "pedestrian":   (0.3, 0.9,  0.3),
        }
        hx, hy, hz = defaults.get(key, (1.0, 1.0, 1.0))
    else:
        extents = np.atleast_1d(np.array(extents, dtype=np.float32))
        hx, hy, hz = float(extents[0]), float(extents[1]), float(extents[2])

    hx *= float(scale[0])
    hy *= float(scale[1])
    hz *= float(scale[2])
    return np.array([-hx, -hy, -hz], dtype=np.float32), np.array([hx, hy, hz], dtype=np.float32)


def compute_obb_corners(entity) -> np.ndarray:
    """
    8 corners of the entity's oriented bounding box in world space
    (scaled local AABB -> yaw rotation around Y -> translate to position).
    Returns an (8, 3) float32 array.
    """
    bmin, bmax = _scaled_local_box(entity)
    x0, y0, z0 = float(bmin[0]), float(bmin[1]), float(bmin[2])
    x1, y1, z1 = float(bmax[0]), float(bmax[1]), float(bmax[2])
    corners_local = np.array([
        [x0, y0, z0],
        [x0, y0, z1],
        [x0, y1, z0],
        [x0, y1, z1],
        [x1, y0, z0],
        [x1, y0, z1],
        [x1, y1, z0],
        [x1, y1, z1],
    ], dtype=np.float32)

    corners_rot = (_rotation_y(_entity_yaw_rad(entity)) @ corners_local.T).T
    pos = np.array(getattr(entity, "position", [0, 0, 0])[:3], dtype=np.float32)
    return corners_rot + pos


def compute_bbox_3d_kitti(entity) -> Dict[str, Any]:
    """
    KITTI-style 3D bounding box:
        h, w, l      : box height (Y), width (X), length (Z) in world units
        location     : [x, y, z] center of the box BOTTOM face, world coords
        rotation_y   : yaw around the Y axis, radians
        corners_world: bonus (8, 3) world-space OBB corners
    """
    bmin, bmax = _scaled_local_box(entity)
    size = bmax - bmin
    ry = _entity_yaw_rad(entity)

    bottom_center_local = np.array(
        [(bmin[0] + bmax[0]) * 0.5, bmin[1], (bmin[2] + bmax[2]) * 0.5],
        dtype=np.float32,
    )
    pos = np.array(getattr(entity, "position", [0, 0, 0])[:3], dtype=np.float32)
    location = _rotation_y(ry) @ bottom_center_local + pos

    corners_world = compute_obb_corners(entity)
    return {
        "h": float(size[1]),
        "w": float(size[0]),
        "l": float(size[2]),
        "location": [float(v) for v in location],
        "rotation_y": float(ry),
        "corners_world": [[float(c) for c in corner] for corner in corners_world],
    }


def project_entity_bbox_2d(entity, intrinsics, extrinsics, width: int, height: int) -> List[int]:
    """
    Project the entity's OBB (8 corners) to a 2D pixel bbox [x0, y0, x1, y1]
    using the camera intrinsics (pinhole) instead of a full MVP chain.

    Returns [0, 0, 0, 0] when the entity is behind the camera or its
    projection is smaller than 4 px in either dimension.
    """
    width, height = int(width), int(height)
    corners_world = compute_obb_corners(entity)

    ext = np.array(extrinsics, dtype=np.float32)
    ones = np.ones((8, 1), dtype=np.float32)
    corners_h = np.hstack([corners_world, ones])
    corners_cam = (ext @ corners_h.T).T
    cx = corners_cam[:, 0]
    cy = corners_cam[:, 1]
    cz = corners_cam[:, 2]

    near = 0.1
    valid = cz < -near
    if not valid.any():
        return [0, 0, 0, 0]

    depth = -cz[valid]
    K = np.array(intrinsics, dtype=np.float32)
    fx, fy = K[0, 0], K[1, 1]
    ppx, ppy = K[0, 2], K[1, 2]

    px = fx * (cx[valid] / depth) + ppx
    py = fy * (-cy[valid] / depth) + ppy

    px = np.clip(px, 0, width)
    py = np.clip(py, 0, height)

    x0, y0 = int(np.min(px)), int(np.min(py))
    x1, y1 = int(np.max(px)), int(np.max(py))

    if x1 - x0 < 4 or y1 - y0 < 4:
        return [0, 0, 0, 0]

    return [x0, y0, x1, y1]


# ---------------------------------------------------------------------------
# KITTI-format 3D labels
# ---------------------------------------------------------------------------

def _wrap_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi]."""
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def build_kitti_lines(objects, extrinsics, img_w, img_h,
                      min_area: float = 300.0, min_dim: float = 10.0) -> List[str]:
    """
    Build KITTI-format label lines for annotated objects.

    One line per object, 15 fields:
        type  truncated  occluded  alpha
        bbox_x0 bbox_y0 bbox_x1 bbox_y1
        h w l  loc_x loc_y loc_z  rotation_y

    Conventions:
    - location  : bottom-center of the 3D box in CAMERA coordinates,
                  KITTI style (x right, y DOWN, z forward)
    - rotation_y: object yaw relative to the camera heading, radians [-pi, pi]
    - alpha     : observation angle = rotation_y + atan2(loc_x, loc_z)
    - truncated : 1.0 when the 2D box touches the image border, else 0.0
    """
    E = np.array(extrinsics, dtype=np.float32)
    R = E[:3, :3]
    # Camera forward in world space (GL cameras look along -Z in camera space)
    fwd_world = -(R.T @ np.array([0.0, 0.0, 1.0], dtype=np.float32))
    cam_yaw = float(np.arctan2(float(fwd_world[0]), float(fwd_world[2])))

    lines: List[str] = []
    for obj in objects:
        if not obj.get("annotate", True):
            continue
        bbox_3d = obj.get("bbox_3d")
        x0, y0, x1, y1 = obj.get("bbox_2d", [0, 0, 0, 0])
        bw, bh = x1 - x0, y1 - y0
        if bbox_3d is None or bw * bh < min_area or bw < min_dim or bh < min_dim:
            continue

        cls = obj.get("class_name", "Misc")

        # World bottom-center -> GL camera coords -> KITTI coords (y down, z fwd)
        loc_w = np.array(bbox_3d["location"], dtype=np.float32)
        loc_gl = (E @ np.append(loc_w, 1.0).astype(np.float32))[:3]
        lx, ly, lz = float(loc_gl[0]), float(-loc_gl[1]), float(-loc_gl[2])
        if lz <= 0.0:
            continue  # behind the camera

        ry = _wrap_angle(float(bbox_3d["rotation_y"]) - cam_yaw)
        alpha = _wrap_angle(ry + float(np.arctan2(lx, lz)))

        truncated = 1.0 if (x0 <= 0 or y0 <= 0 or x1 >= img_w or y1 >= img_h) else 0.0
        occluded = int(obj.get("occluded", 0))

        lines.append(
            f"{cls} {truncated:.2f} {occluded} {alpha:.2f} "
            f"{x0:.2f} {y0:.2f} {x1:.2f} {y1:.2f} "
            f"{bbox_3d['h']:.2f} {bbox_3d['w']:.2f} {bbox_3d['l']:.2f} "
            f"{lx:.2f} {ly:.2f} {lz:.2f} {ry:.2f}"
        )
    return lines


def write_kitti_labels(path, objects, extrinsics, img_w, img_h) -> None:
    """Write one KITTI .txt label file (one line per annotated object)."""
    lines = build_kitti_lines(objects, extrinsics, img_w, img_h)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


class BBoxCalculator:
    """
    CPU-only 2D bbox extraction:
    local AABB (8 corners) -> Model -> View -> Projection -> NDC -> Pixel bbox

    Projects only the 8 corners of each entity's local-space AABB instead of
    the full vertex buffer: O(8) per entity instead of O(N_vertices). For
    convex-ish objects (cars) the projected AABB is the standard way to get a
    2D bounding box and is orders of magnitude cheaper.
    """

    def __init__(self):
        pass

    @staticmethod
    def _to_homogeneous(vertices: np.ndarray) -> np.ndarray:
        ones = np.ones((vertices.shape[0], 1), dtype=np.float32)
        return np.concatenate([vertices.astype(np.float32), ones], axis=1)

    def calculate(
        self,
        entities,
        projection: np.ndarray,
        view: np.ndarray,
        width: int,
        height: int,
    ) -> List[Dict[str, Any]]:
        bboxes: List[Dict[str, Any]] = []
        width = int(width)
        height = int(height)

        for ent in entities:
            local_min = getattr(ent, "local_aabb_min", None)
            local_max = getattr(ent, "local_aabb_max", None)
            if local_min is None or local_max is None:
                continue

            model = ent.world_matrix().astype(np.float32)
            mvp = (projection @ view @ model).astype(np.float32)

            x0, y0, z0 = (float(v) for v in local_min[:3])
            x1, y1, z1 = (float(v) for v in local_max[:3])
            corners_local = np.array(
                [
                    [x0, y0, z0],
                    [x0, y0, z1],
                    [x0, y1, z0],
                    [x0, y1, z1],
                    [x1, y0, z0],
                    [x1, y0, z1],
                    [x1, y1, z0],
                    [x1, y1, z1],
                ],
                dtype=np.float32,
            )

            v_local_h = self._to_homogeneous(corners_local)  # 8x4
            clip = (mvp @ v_local_h.T).T  # 8x4

            w = clip[:, 3]
            # Only corners in front of the camera plane are projectable.
            valid_w = w > 1e-6
            if not np.any(valid_w):
                continue

            ndc = clip[valid_w, :3] / w[valid_w, None]  # Nx3 in [-inf, inf]

            # Remove points behind far plane or near plane.
            # Requirement explicitly asks to remove objects behind camera with Z > 1.0.
            z_ok = (ndc[:, 2] <= 1.0) & (ndc[:, 2] >= -1.0)
            if not np.any(z_ok):
                continue
            ndc = ndc[z_ok]

            # Entire object outside viewport check
            if np.max(ndc[:, 0]) < -1.0 or np.min(ndc[:, 0]) > 1.0:
                continue
            if np.max(ndc[:, 1]) < -1.0 or np.min(ndc[:, 1]) > 1.0:
                continue

            # Clip against screen edges in NDC
            x_ndc = np.clip(ndc[:, 0], -1.0, 1.0)
            y_ndc = np.clip(ndc[:, 1], -1.0, 1.0)

            # NDC -> pixel
            x_pix = (x_ndc * 0.5 + 0.5) * (width - 1)
            y_pix = (1.0 - (y_ndc * 0.5 + 0.5)) * (height - 1)

            x_min = float(np.min(x_pix))
            y_min = float(np.min(y_pix))
            x_max = float(np.max(x_pix))
            y_max = float(np.max(y_pix))

            if x_max - x_min < 1.0 or y_max - y_min < 1.0:
                continue

            local_center = (ent.local_aabb_min + ent.local_aabb_max) * 0.5
            world_center = model @ np.array([local_center[0], local_center[1], local_center[2], 1.0], dtype=np.float32)
            bboxes.append(
                {
                    "entity_name": ent.name,
                    "class_id": int(ent.class_id),
                    "xyxy": [x_min, y_min, x_max, y_max],
                    "position3d": model[:3, 3].tolist(),
                    "center3d": world_center[:3].tolist()
                }
            )

        return bboxes


class DatasetExporter:
    """
    Write multi-view camera outputs:
    - RGB PNG
    - Mask PNG
    - Depth .npy (float32, metric — linearized from the real depth buffer)
    - YOLO label TXT
    """

    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = output_dir
        self.rgb_dir = os.path.join(output_dir, "rgb")
        self.mask_dir = os.path.join(output_dir, "mask")
        self.depth_dir = os.path.join(output_dir, "depth")
        self.label_dir = os.path.join(output_dir, "labels")
        self.labels_kitti_dir = os.path.join(output_dir, "labels_kitti")

        os.makedirs(self.rgb_dir, exist_ok=True)
        os.makedirs(self.mask_dir, exist_ok=True)
        os.makedirs(self.depth_dir, exist_ok=True)
        os.makedirs(self.label_dir, exist_ok=True)
        os.makedirs(self.labels_kitti_dir, exist_ok=True)

    @staticmethod
    def read_rgb_buffer(width: int, height: int) -> np.ndarray:
        data = GL.glReadPixels(
            0, 0, width, height,
            GL.GL_RGB, GL.GL_UNSIGNED_BYTE
        )
        img = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 3))
        return np.flipud(img).copy()

    @staticmethod
    def read_depth_buffer_as_gray(width: int, height: int) -> np.ndarray:
        # Depth pass shader already writes linear depth into RGB channels.
        data = GL.glReadPixels(
            0, 0, width, height,
            GL.GL_RGB, GL.GL_UNSIGNED_BYTE
        )
        img_rgb = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 3))
        img_rgb = np.flipud(img_rgb)
        return img_rgb[:, :, 0].copy()

    @staticmethod
    def _xyxy_to_yolo(xyxy: List[float], width: int, height: int) -> Tuple[float, float, float, float]:
        x_min, y_min, x_max, y_max = xyxy
        bw = max(0.0, x_max - x_min)
        bh = max(0.0, y_max - y_min)
        cx = x_min + 0.5 * bw
        cy = y_min + 0.5 * bh

        return (
            cx / max(1, width),
            cy / max(1, height),
            bw / max(1, width),
            bh / max(1, height),
        )

    def save_multi_view(
        self,
        frame_idx: int,
        camera_data: Dict[str, Any]
    ):
        """
        camera_data = {
            "CAM_FRONT": {
                "rgb": np.ndarray,          # (h, w, 3) uint8
                "mask": np.ndarray,         # (h, w, 3) uint8
                "depth": np.ndarray,        # (h, w) float32 metric depth
                "bboxes": List[Dict],
                "intrinsics": np.ndarray,
                "extrinsics": np.ndarray,
                "width": int,
                "height": int
            },
            ...
        }
        """
        stem_idx = f"{frame_idx:06d}"
        cameras_meta = {}

        for cam_name, data in camera_data.items():
            rgb = data["rgb"]
            mask = data["mask"]
            depth = data["depth"]
            bboxes = data["bboxes"]
            K = data["intrinsics"]
            E = data["extrinsics"]
            width = data["width"]
            height = data["height"]

            prefix = f"{cam_name}_{stem_idx}"

            Image.fromarray(rgb, mode="RGB").save(os.path.join(self.rgb_dir, f"{prefix}.png"))
            Image.fromarray(mask, mode="RGB").save(os.path.join(self.mask_dir, f"{prefix}.png"))
            np.save(os.path.join(self.depth_dir, f"{prefix}.npy"), depth.astype(np.float32))

            label_path = os.path.join(self.label_dir, f"{prefix}.txt")
            with open(label_path, "w", encoding="utf-8") as f:
                for box in bboxes:
                    cls_id = int(box["class_id"])
                    x, y, w, h = self._xyxy_to_yolo(box["xyxy"], width, height)

                    x = float(np.clip(x, 0.0, 1.0))
                    y = float(np.clip(y, 0.0, 1.0))
                    w = float(np.clip(w, 0.0, 1.0))
                    h = float(np.clip(h, 0.0, 1.0))

                    f.write(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\\n")

            # KITTI 3D labels for this camera view
            kitti_objects = [
                {
                    "class_name": box.get("class_name", "Misc"),
                    "bbox_2d": box["xyxy"],
                    "bbox_3d": box.get("bbox_3d"),
                    "annotate": True,
                }
                for box in bboxes
            ]
            write_kitti_labels(
                os.path.join(self.labels_kitti_dir, f"{prefix}.txt"),
                kitti_objects, E, width, height,
            )
            
            cameras_meta[cam_name] = {
                "intrinsics": K.tolist(),
                "extrinsics": E.tolist(),
                "bboxes": bboxes
            }

        metadata = {
            "frame_idx": frame_idx,
            "cameras": cameras_meta
        }

        meta_path = os.path.join(self.output_dir, f"scene_metadata_{stem_idx}.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)