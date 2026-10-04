# renderers.py
import os
from typing import Tuple, Optional

import OpenGL.GL as GL
import numpy as np

from libs.shader import Shader
from libs.buffer import UManager
from libs.lighting import LightingManager
from libs.pbo import PBOReader
from src.annotation.annotations import (
    BBoxCalculator,
    DatasetExporter,
    compute_bbox_3d_kitti,
    project_entity_bbox_2d,
)


class RGBRenderer:
    def __init__(self, base_dir: str):
        self.shader = Shader(
            os.path.join(base_dir, "shaders", "phong.vert"),
            os.path.join(base_dir, "shaders", "phong.frag"),
        )
        self.dimmed = False

    def set_dimmed(self, dimmed: bool):
        self.dimmed = bool(dimmed)

    def toggle_dimmed(self) -> bool:
        self.dimmed = not self.dimmed
        return self.dimmed

    def render(self, scene, projection: np.ndarray, view: np.ndarray):
        GL.glUseProgram(self.shader.render_idx)
        uma = UManager(self.shader)

        if self.dimmed:
            light_color = np.array([0.5, 0.5, 0.5], dtype=np.float32)
            ambient_strength = 0.10
            specular_strength = 0.08
        else:
            light_color = np.array([1.0, 1.0, 1.0], dtype=np.float32)
            ambient_strength = 0.28
            specular_strength = 0.35

        uma.upload_uniform_vector3fv(light_color, "light_color")
        uma.upload_uniform_vector3fv(np.array([0.0, 8.0, 5.0], dtype=np.float32), "light_pos")
        uma.upload_uniform_scalar1f(64.0, "shininess")
        uma.upload_uniform_scalar1f(ambient_strength, "ambient_strength")
        uma.upload_uniform_scalar1f(specular_strength, "specular_strength")

        for ent in scene.entities:
            ent.mesh.draw(projection, view, ent.world_matrix(), self.shader)


class MaskRenderer:
    def __init__(self, base_dir: str):
        self.shader = Shader(
            os.path.join(base_dir, "shaders", "mask.vert"),
            os.path.join(base_dir, "shaders", "mask.frag"),
        )

    def render(self, scene, projection: np.ndarray, view: np.ndarray):
        # Clear specific to Mask: Sky semantic class = 1 (Bright Green visually)
        prev_clear = GL.glGetFloatv(GL.GL_COLOR_CLEAR_VALUE)
        GL.glClearColor(0.0, 1.0, 0.0, 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        
        GL.glUseProgram(self.shader.render_idx)
        uma = UManager(self.shader)

        for ent in scene.entities:
            uma.upload_uniform_vector3fv(ent.instance_color, "instance_color")
            ent.mesh.draw(projection, view, ent.world_matrix(), self.shader)
            
        GL.glClearColor(*prev_clear)


class DepthRenderer:
    def __init__(self, base_dir: str, near: float = 0.1, far: float = 150.0):
        self.shader = Shader(
            os.path.join(base_dir, "shaders", "depth.vert"),
            os.path.join(base_dir, "shaders", "depth.frag"),
        )
        self.near = float(near)
        self.far = float(far)

    def render(self, scene, projection: np.ndarray, view: np.ndarray):
        GL.glUseProgram(self.shader.render_idx)
        uma = UManager(self.shader)
        uma.upload_uniform_scalar1f(self.near, "near")
        uma.upload_uniform_scalar1f(self.far, "far")

        for ent in scene.entities:
            ent.mesh.draw(projection, view, ent.world_matrix(), self.shader)


class RenderManager:
    """
    Multi-pass manager:
    - interactive draw mode
    - one-call export_current_frame(): RGB + Mask + Depth + YOLO labels
    """

    def __init__(self, scene, base_dir: str, scene_overlay=None, output_dir: str = "outputs", near: float = 0.1, far: float = 150.0):
        self.scene = scene
        self.rgb_renderer = RGBRenderer(base_dir)
        self.mask_renderer = MaskRenderer(base_dir)
        self.depth_renderer = DepthRenderer(base_dir, near=near, far=far)
        self.scene_overlay = scene_overlay
        self.bbox_calculator = BBoxCalculator()
        self.exporter = DatasetExporter(output_dir=output_dir)

        self.mode = "RGB"
        self.frame_idx = 0

        # Offscreen FBO used by export_current_frame (created lazily, resized
        # to match each camera's native resolution).
        self._export_fbo = None
        self._export_color_tex = None
        self._export_depth_rbo = None
        self._export_fbo_size = (0, 0)

    def set_mode(self, mode: str):
        mode = mode.upper()
        if mode in {"RGB", "MASK", "DEPTH"}:
            self.mode = mode

    def toggle_rgb_dimmed(self) -> bool:
        return self.rgb_renderer.toggle_dimmed()

    def is_rgb_dimmed(self) -> bool:
        return self.rgb_renderer.dimmed

    @staticmethod
    def _reset_texture_state(shader=None):
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        if shader is not None:
            GL.glUseProgram(shader.render_idx)
            loc_has_texture = GL.glGetUniformLocation(shader.render_idx, "has_texture")
            if loc_has_texture != -1:
                GL.glUniform1i(loc_has_texture, 0)

    def draw(self, projection: np.ndarray, view: np.ndarray):
        self._reset_texture_state()

        if self.mode == "RGB":
            self.rgb_renderer.render(self.scene, projection, view)
            self._reset_texture_state(self.rgb_renderer.shader)
            if self.scene_overlay:
                self.scene_overlay.render(
                    self.rgb_renderer.shader,
                    projection,
                    view,
                    is_rgb=True,
                    rgb_dimmed=self.is_rgb_dimmed(),
                )
            self._reset_texture_state(self.rgb_renderer.shader)

        elif self.mode == "MASK":
            self._reset_texture_state(self.mask_renderer.shader)
            self.mask_renderer.render(self.scene, projection, view)
            self._reset_texture_state(self.mask_renderer.shader)
            if self.scene_overlay:
                self.scene_overlay.render(self.mask_renderer.shader, projection, view, is_rgb=False)
            self._reset_texture_state(self.mask_renderer.shader)

        elif self.mode == "DEPTH":
            self._reset_texture_state(self.depth_renderer.shader)
            self.depth_renderer.render(self.scene, projection, view)
            self._reset_texture_state(self.depth_renderer.shader)
            if self.scene_overlay:
                self.scene_overlay.render(self.depth_renderer.shader, projection, view, is_rgb=False)
            self._reset_texture_state(self.depth_renderer.shader)

    # ------------------------------------------------------------------
    # Offscreen export FBO
    # ------------------------------------------------------------------
    def _ensure_export_fbo(self, width: int, height: int):
        """Create (or resize) the offscreen FBO used for dataset exports."""
        width, height = int(width), int(height)
        if self._export_fbo is not None and self._export_fbo_size == (width, height):
            return

        if self._export_fbo is None:
            self._export_fbo = int(GL.glGenFramebuffers(1))
            self._export_color_tex = int(GL.glGenTextures(1))
            self._export_depth_rbo = int(GL.glGenRenderbuffers(1))

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._export_fbo)

        GL.glBindTexture(GL.GL_TEXTURE_2D, self._export_color_tex)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGB,
                        width, height, 0,
                        GL.GL_RGB, GL.GL_UNSIGNED_BYTE, None)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER,
                                  GL.GL_COLOR_ATTACHMENT0,
                                  GL.GL_TEXTURE_2D, self._export_color_tex, 0)

        GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, self._export_depth_rbo)
        GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, GL.GL_DEPTH24_STENCIL8,
                                 width, height)
        GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER,
                                     GL.GL_DEPTH_STENCIL_ATTACHMENT,
                                     GL.GL_RENDERBUFFER, self._export_depth_rbo)

        status = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
        if status != GL.GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError(f"Export FBO incomplete, status: {hex(status)}")

        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
        self._export_fbo_size = (width, height)

    def export_current_frame(self, camera_manager, multiview: bool = False,
                             exclude_names=None) -> dict:
        """
        Render dataset views offscreen and return them.

        - multiview=False: only the ACTIVE camera (default; single view).
        - multiview=True : ALL cameras, each rendered into an FBO at its own
          native resolution (no more back-buffer overreads), and the per-view
          files are also written under <output_dir>/ via DatasetExporter.
        - exclude_names  : optional set of entity names excluded from labels
          (e.g. the ego vehicle carrying the cameras).

        Every view is produced as:
            rgb   : (h, w, 3) uint8
            mask  : (h, w, 3) uint8 instance mask
            depth : (h, w) float32 METRIC depth (real depth buffer, linearized)

        Returns: {cam_name: view_dict}
        """
        old_mode = self.mode
        exclude = set(exclude_names or ())

        if multiview:
            cameras = camera_manager.get_all_cameras()
        else:
            cameras = [camera_manager.get_active_camera()]

        dyn = self.scene.get_dynamic_entities()

        # PBO strategy: enqueue an async readback after every pass (returns
        # immediately, GPU keeps rendering), compute CPU labels while the GPU
        # works, then synchronize exactly once for the whole frame.
        reader = PBOReader()
        pending = []  # (cam_name, idx_rgb, idx_depth, idx_mask)
        views = {}

        prev_fbo = int(GL.glGetIntegerv(GL.GL_FRAMEBUFFER_BINDING))

        for cam in cameras:
            w, h = int(cam.resolution[0]), int(cam.resolution[1])
            projection = cam.projection_matrix()
            view = cam.view_matrix()

            self._ensure_export_fbo(w, h)
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._export_fbo)
            GL.glViewport(0, 0, w, h)

            # 1) RGB pass
            GL.glClearColor(0.18, 0.20, 0.24, 1.0) # normal sky clear color
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
            self._reset_texture_state(self.rgb_renderer.shader)
            self.rgb_renderer.render(self.scene, projection, view)
            if self.scene_overlay:
                self.scene_overlay.render(
                    self.rgb_renderer.shader,
                    projection,
                    view,
                    is_rgb=True,
                    rgb_dimmed=self.is_rgb_dimmed(),
                )
            self._reset_texture_state(self.rgb_renderer.shader)
            # Read RGB + real depth buffer BEFORE the mask pass overwrites
            # them (queued on the GPU timeline, so ordering is guaranteed).
            idx_rgb = reader.enqueue_rgb(w, h)
            idx_depth = reader.enqueue_depth_float(w, h)

            # 2) Mask pass (MaskRenderer clears its own color for Sky)
            self._reset_texture_state(self.mask_renderer.shader)
            self.mask_renderer.render(self.scene, projection, view)
            if self.scene_overlay:
                self.scene_overlay.render(self.mask_renderer.shader, projection, view, is_rgb=False)
            self._reset_texture_state(self.mask_renderer.shader)
            idx_mask = reader.enqueue_rgb(w, h)

            pending.append((cam.name, idx_rgb, idx_depth, idx_mask))

            # 3) CPU labels from dynamic entities — runs while the GPU is
            #    still busy with the passes queued above.
            K = cam.get_intrinsics()
            E = cam.get_extrinsics()
            bboxes = []
            for ent in dyn:
                if ent.name in exclude:
                    continue  # e.g. the ego vehicle carrying the cameras
                xyxy = project_entity_bbox_2d(ent, K, E, w, h)
                if xyxy == [0, 0, 0, 0]:
                    continue
                bboxes.append({
                    "entity_name": ent.name,
                    "class_id": int(ent.class_id),
                    "class_name": getattr(ent, "class_name", type(ent).__name__),
                    "xyxy": [float(v) for v in xyxy],
                    "bbox_3d": compute_bbox_3d_kitti(ent),
                })

            views[cam.name] = {
                "bboxes": bboxes,
                "intrinsics": K,
                "extrinsics": E,
                "width": w,
                "height": h,
                "near": float(cam.near),
                "far": float(cam.far),
            }

        # 4) Single CPU<->GPU sync point, then attach the images.
        frames = reader.fetch_all()
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, prev_fbo)

        for cam_name, idx_rgb, idx_depth, idx_mask in pending:
            data = views[cam_name]
            data["rgb"] = frames[idx_rgb]
            data["mask"] = frames[idx_mask]
            raw_depth = frames[idx_depth]
            near, far = data["near"], data["far"]
            data["depth"] = (
                (2.0 * near * far)
                / (far + near - (raw_depth * 2.0 - 1.0) * (far - near))
            ).astype(np.float32)

        # 5) Multiview: write one file set per camera.
        if multiview:
            self.exporter.save_multi_view(
                frame_idx=self.frame_idx,
                camera_data=views,
            )

        # 6) In-memory arrays of the ACTIVE camera for export hooks, so they
        #    don't need to re-read anything from disk.
        active_cam_name = camera_manager.get_active_camera().name
        self.last_exported = views[active_cam_name]

        self.frame_idx += 1
        self.mode = old_mode
        return views
