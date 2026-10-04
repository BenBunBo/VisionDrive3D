from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import OpenGL.GL as GL

from src.scene.api_scene_overlay import build_drawables_from_folder
from src.core.entity import Entity
from libs import transform as T
from libs.buffer import UManager
from src.core.mesh import Mesh


class _SubMeshAdapter:
    """
    Adapter để biến 1 submesh thành API tương thích với Entity.mesh.

    Mục tiêu là tái dùng đúng texture/material pipeline vốn có từ Mesh.setup()
    mà không phát minh luồng texture mới cho phần nhà dân.
    """

    def __init__(self, submesh: dict):
        self.submeshes = [submesh]
        self.vertices = np.asarray(submesh["vertices"], dtype=np.float32)

        self.texture_id = submesh.get("texture_id")
        self.has_texture = bool(submesh.get("has_texture", False))

    def compute_local_aabb(self):
        if self.vertices.size == 0:
            zero = np.zeros(3, dtype=np.float32)
            return zero, zero
        return np.min(self.vertices, axis=0).astype(np.float32), np.max(self.vertices, axis=0).astype(np.float32)

    def draw(self, projection: np.ndarray, view: np.ndarray, model: np.ndarray, shader_program):
        modelview = view @ model
        uma = UManager(shader_program)
        uma.upload_uniform_matrix4fv(projection, "projection", transpose=True)
        uma.upload_uniform_matrix4fv(modelview, "modelview", transpose=True)
        uma.upload_uniform_matrix4fv(model, "model", transpose=True)

        for submesh in self.submeshes:
            texture_id = submesh.get("texture_id")
            if texture_id is not None:
                GL.glActiveTexture(GL.GL_TEXTURE0)
                GL.glBindTexture(GL.GL_TEXTURE_2D, texture_id)
                uma.upload_uniform_scalar1i(0, "diffuse_map")
                uma.upload_uniform_scalar1i(1, "has_texture")
            else:
                GL.glActiveTexture(GL.GL_TEXTURE0)
                GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
                uma.upload_uniform_scalar1i(0, "has_texture")

            uma.upload_uniform_scalar1i(1, "use_uniform_base_color")
            uma.upload_uniform_vector3fv(np.asarray(submesh["base_color"], dtype=np.float32), "base_color")

            submesh["vao"].activate()
            GL.glDrawArrays(GL.GL_TRIANGLES, 0, int(submesh["vertex_count"]))
            submesh["vao"].deactivate()

        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)


class SceneOverlay:
    """
    Controller ghép scene:
    1) Base Junction: assets/road_junction (dùng API api_scene_overlay.py)
    2) Buildings: trích từ assets/scene, lọc bỏ mặt đường cũ, giữ texture pipeline của Mesh/Entity.
    """

    def __init__(
        self,
        road_junction_folder: Path | None = None,
        scene_obj_path: Path | None = None,
        corner_offsets: list[dict] | None = None,
        load_buildings: bool = False,
    ):
        # Repo root (src/scene/scene_overlay.py -> parents[2])
        root = Path(__file__).resolve().parents[2]
        self.road_junction_folder = (road_junction_folder or (root / "assets" / "road_junction")).expanduser().resolve()
        self.scene_obj_path = (scene_obj_path or (root / "assets" / "scene" / "Street environment_V01.obj")).expanduser().resolve()

        if not self.road_junction_folder.exists():
            raise FileNotFoundError(f"Road-junction folder not found: {self.road_junction_folder}")
        if not self.scene_obj_path.exists():
            raise FileNotFoundError(f"Scene OBJ not found: {self.scene_obj_path}")

        # Base junction cố định tại gốc tọa độ.
        self.base_junction_drawables = build_drawables_from_folder(self.road_junction_folder)

        # Cấu hình offset dễ chỉnh tay sau khi test.
        self.corner_offsets = corner_offsets or [
            {"name": "north_west", "offset": np.array([-64.0, 0.0, -64.0], dtype=np.float32)},
            {"name": "north_east", "offset": np.array([64.0, 0.0, -64.0], dtype=np.float32)},
            {"name": "south_west", "offset": np.array([-64.0, 0.0, 64.0], dtype=np.float32)},
            {"name": "south_east", "offset": np.array([64.0, 0.0, 64.0], dtype=np.float32)},
        ]

        # "Buildings" are extracted from the SAME street OBJ that viewer.py
        # already adds to the scene as the "street_environment" entity.
        # Loading them again duplicates the whole mesh in VRAM and draws the
        # identical geometry twice per pass (2x vertex cost + z-fighting), so
        # this is opt-in via load_buildings=True.
        self.building_entities = self._load_building_entities() if load_buildings else []

    # def _is_old_road_submesh(self, submesh: dict, index: int = 0) -> bool:
    #     """
    #     Heuristic lọc phần mặt đường cũ đã được cải tiến.
    #     """
    #     verts = np.asarray(submesh.get("vertices"), dtype=np.float32)
    #     if verts.size == 0:
    #         return True
    #     vmin = np.min(verts, axis=0)
    #     vmax = np.max(verts, axis=0)
    #     extent = vmax - vmin
    #     # Lấy thông số
    #     height_y = float(extent[1])
    #     lowest_y = float(vmin[1])
    #     # ĐIỀU KIỆN MỚI: Chỉ cần mesh mỏng (cao độ thấp) VÀ nằm sát mặt đất
    #     is_flat = height_y <= 3.5       # Nới lỏng: Cho phép vỉa hè/đường dày tới 3.5 đơn vị
    #     is_on_ground = lowest_y <= 1.5  # Nới lỏng: Điểm thấp nhất nằm sát mốc 0
    #     # Nếu muốn lọc theo Tên hoặc Material (Cách chính xác nhất):
    #     # Bạn có thể kiểm tra submesh.get("name") hoặc ID material ở đây.
    #     is_road = is_flat and is_on_ground
    #     # Tắt comment dòng lệnh dưới đây nếu bạn muốn kiểm tra tại sao mặt đường vẫn lọt qua
    #     print(f"Mesh {index:03d} | Height: {height_y:.2f} | Lowest Y: {lowest_y:.2f} | is_road: {is_road}")
    #     return is_road

    def _load_building_entities(self) -> list[Entity]:
        """
        Tải assets/scene theo luồng Mesh hiện tại (đã xử lý map_Kd chuẩn),
        rồi tách từng submesh để dễ lọc và bố trí không gian.
        """
        scene_mesh = Mesh(str(self.scene_obj_path)).setup()

        entities: list[Entity] = []
        kept_count = 0
        # dropped_count = 0

        for idx, submesh in enumerate(scene_mesh.submeshes):
            # if self._is_old_road_submesh(submesh, idx):
            #     dropped_count += 1
            #     continue

            mesh_adapter = _SubMeshAdapter(submesh)
            ent = Entity(
                name=f"building_part_{idx:03d}",
                mesh=mesh_adapter,
                class_id=-1,
                instance_color=(-1.0, 0.0, 0.0),
                is_dynamic=False,
            )
            entities.append(ent)
            kept_count += 1

        print(
            f"[SceneOverlay] scene submeshes: kept={kept_count}"
        )
        # , filtered_old_road={dropped_count}
        return entities

    @staticmethod
    def _extract_eye_pos_from_view(view: np.ndarray) -> np.ndarray:
        try:
            inv_view = np.linalg.inv(np.asarray(view, dtype=np.float32))
            return np.asarray(inv_view[:3, 3], dtype=np.float32)
        except np.linalg.LinAlgError:
            return np.array([0.0, 0.0, 0.0], dtype=np.float32)

    @staticmethod
    def _capture_gl_state() -> dict:
        return {
            "program": int(GL.glGetIntegerv(GL.GL_CURRENT_PROGRAM)),
            "vao": int(GL.glGetIntegerv(GL.GL_VERTEX_ARRAY_BINDING)),
            "active_texture": int(GL.glGetIntegerv(GL.GL_ACTIVE_TEXTURE)),
            "texture_2d": int(GL.glGetIntegerv(GL.GL_TEXTURE_BINDING_2D)),
        }

    @staticmethod
    def _restore_gl_state(state: dict):
        GL.glUseProgram(int(state["program"]))
        GL.glBindVertexArray(int(state["vao"]))
        GL.glActiveTexture(int(state["active_texture"]))
        GL.glBindTexture(GL.GL_TEXTURE_2D, int(state["texture_2d"]))

    @staticmethod
    def _iter_offsets(corner_offsets: Iterable[dict]):
        for item in corner_offsets:
            offset = np.asarray(item.get("offset", [0.0, 0.0, 0.0]), dtype=np.float32)
            yield T.translate(offset).astype(np.float32)

    def render(
        self,
        shader_program,
        projection: np.ndarray,
        view: np.ndarray,
        is_rgb: bool = True,
        rgb_dimmed: bool = False,
    ):
        state_before = self._capture_gl_state()

        # Tính khung cắt (clip_box)
        x_vals = [item["offset"][0] for item in self.corner_offsets]
        z_vals = [item["offset"][2] for item in self.corner_offsets]
        clip_box = {
            "min_x": float(min(x_vals)) + 16.0, 
            "max_x": float(max(x_vals)) - 16.0, 
            "min_z": float(min(z_vals)) + 16.0, 
            "max_z": float(max(z_vals)) - 16.0, 
        }

        # 1) Vẽ ngã tư nền (Truyền cờ is_rgb xuống)
        eye_pos = self._extract_eye_pos_from_view(view)
        for drawable in self.base_junction_drawables:
            drawable.draw(
                projection,
                view,
                shader_program,
                is_rgb=is_rgb,
                eye_pos=eye_pos,
                clip_box=clip_box,
                rgb_dimmed=rgb_dimmed,
            )

        # 2) Vẽ nhà dân bằng shader của viewer.py
        GL.glUseProgram(int(shader_program.render_idx))
        for ent in self.building_entities:
            model = ent.world_matrix()
            ent.mesh.draw(projection, view, model, shader_program)

        GL.glBindVertexArray(0)
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

        self._restore_gl_state(state_before)