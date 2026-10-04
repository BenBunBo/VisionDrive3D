# import os
# os.environ['ASSIMP_LIBRARY_PATH'] = '/opt/homebrew/lib/libassimp.dylib'

# # import pyassimp  
import argparse
import ctypes
import gc
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import trimesh
import glfw
import numpy as np
import OpenGL.GL as GL
from PIL import Image

# Make project root importable when running this file directly
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from libs import transform as T
from libs.buffer import UManager, VAO
from libs.shader import Shader


VERT_SHADER = """
#version 330 core
layout(location = 0) in vec3 position;
layout(location = 1) in vec2 texcoord; // Thêm toạ độ UV
layout(location = 2) in vec3 normal;

uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;

out vec3 v_pos;
out vec2 v_texcoord; // Truyền UV sang frag
out vec3 v_normal;

void main() {
    vec4 world = model * vec4(position, 1.0);
    v_pos = world.xyz;
    v_texcoord = texcoord;
    mat3 normal_mat = mat3(transpose(inverse(model)));
    v_normal = normalize(normal_mat * normal);
    gl_Position = projection * view * world;
}
"""


FRAG_SHADER = """
#version 330 core
in vec3 v_pos;
in vec2 v_texcoord;
in vec3 v_normal;

uniform vec3 light_dir;
uniform vec3 eye_pos;
uniform sampler2D diffuse_map; // Đọc ảnh texture
uniform bool use_texture;      // Cờ kiểm tra xem lưới có texture không
uniform vec3 base_color;
uniform float lighting_scale;

uniform bool use_clip;
uniform vec2 clip_x; // vec2(min_x, max_x)
uniform vec2 clip_z; // vec2(min_z, max_z)

out vec4 fragColor;

void main() {
    
    if (use_clip) {
        if (v_pos.x < clip_x.x || v_pos.x > clip_x.y || 
            v_pos.z < clip_z.x || v_pos.z > clip_z.y) {
            discard; 
        }
    }
    
    vec3 N = normalize(v_normal);
    vec3 L = normalize(-light_dir);
    vec3 V = normalize(eye_pos - v_pos);
    vec3 H = normalize(L + V);

    float diff = max(dot(N, L), 0.0);
    float spec = pow(max(dot(N, H), 0.0), 24.0);

    // Lấy màu từ ảnh thay vì màu trơn
    vec3 albedo = base_color;
    if (use_texture) {
        albedo = texture(diffuse_map, v_texcoord).rgb;
    }

    vec3 ambient = 0.18 * albedo;
    vec3 diffuse = 0.70 * diff * albedo;
    vec3 specular = 0.35 * spec * vec3(1.0);
    
    fragColor = vec4((ambient + diffuse + specular) * lighting_scale, 1.0);
}
"""


@dataclass
class FileEntry:
    path: Path
    rel: str
    suffix: str
    size: int
    modified: str


@dataclass
class MaterialEntry:
    name: str
    ka: tuple | None = None
    kd: tuple | None = None
    ks: tuple | None = None
    ns: float | None = None
    map_kd: str | None = None


def _suffix_color(suffix: str) -> tuple[float, float, float]:
    palette = {
        ".fbx": (0.96, 0.63, 0.21),
        ".blend": (0.95, 0.38, 0.24),
        ".mtl": (0.35, 0.78, 0.63),
        ".obj": (0.42, 0.66, 0.98),
        ".ply": (0.74, 0.59, 0.96),
        ".stl": (0.58, 0.80, 0.44),
        ".glb": (0.31, 0.72, 0.90),
        ".gltf": (0.31, 0.72, 0.90),
    }
    return palette.get(suffix.lower(), (0.75, 0.78, 0.84))


def _make_proxy_box(entry: FileEntry) -> dict:
    size_mb = float(entry.size) / (1024.0 * 1024.0)
    s = float(np.clip(0.35 + 0.15 * math.log10(size_mb + 1.0), 0.30, 0.70))

    p = np.array(
        [
            [-s, -s, -s], [s, -s, -s], [s, s, -s], [-s, s, -s],
            [-s, -s, s], [s, -s, s], [s, s, s], [-s, s, s],
        ],
        dtype=np.float32,
    )
    faces = np.array(
        [
            [0, 1, 2], [0, 2, 3],
            [4, 6, 5], [4, 7, 6],
            [0, 4, 5], [0, 5, 1],
            [1, 5, 6], [1, 6, 2],
            [2, 6, 7], [2, 7, 3],
            [3, 7, 4], [3, 4, 0],
        ],
        dtype=np.int32,
    )

    tri_vertices = p[faces].reshape(-1, 3).astype(np.float32)
    tri_normals = np.zeros_like(tri_vertices, dtype=np.float32)
    for i in range(0, len(tri_vertices), 3):
        p0, p1, p2 = tri_vertices[i], tri_vertices[i + 1], tri_vertices[i + 2]
        n = np.cross(p1 - p0, p2 - p0)
        n_norm = float(np.linalg.norm(n))
        if n_norm > 1e-8:
            n = n / n_norm
        else:
            n = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        tri_normals[i:i + 3] = n

    return {
        "name": f"proxy::{entry.rel}",
        "vertices": tri_vertices,
        "normals": tri_normals,
        "color": _suffix_color(entry.suffix),
    }


def scan_folder(folder: Path) -> list[FileEntry]:
    entries = []
    for path in sorted(folder.rglob("*")):
        if not path.is_file():
            continue
        stat = path.stat()
        entries.append(
            FileEntry(
                path=path,
                rel=str(path.relative_to(folder)),
                suffix=path.suffix.lower(),
                size=stat.st_size,
                modified=datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
            )
        )
    return entries


def parse_mtl(path: Path) -> list[MaterialEntry]:
    mats: list[MaterialEntry] = []
    current: MaterialEntry | None = None

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            key = parts[0]

            if key == "newmtl":
                if current is not None:
                    mats.append(current)
                current = MaterialEntry(name=" ".join(parts[1:]))
                continue
            if current is None:
                continue

            if key in {"Ka", "Kd", "Ks"} and len(parts) >= 4:
                value = (float(parts[1]), float(parts[2]), float(parts[3]))
                if key == "Ka":
                    current.ka = value
                elif key == "Kd":
                    current.kd = value
                else:
                    current.ks = value
            elif key == "Ns" and len(parts) >= 2:
                current.ns = float(parts[1])
            elif key == "map_Kd" and len(parts) >= 2:
                # Normalizes Windows-style double backslashes to POSIX-style forward slashes
                raw_path = " ".join(parts[1:])
                current.map_kd = raw_path.replace("\\\\", "/").replace("\\", "/")

    if current is not None:
        mats.append(current)
    return mats


def print_manifest(folder: Path, entries: list[FileEntry], materials_by_file: dict[str, list[MaterialEntry]]) -> None:
    print("=" * 72)
    print("ROAD JUNCTION ASSET MANIFEST")
    print("Folder:", folder)
    print("Total files:", len(entries))
    print("-" * 72)
    for e in entries:
        print(f"{e.rel:48s} | {e.suffix:6s} | {e.size:9d} bytes | {e.modified}")

    if materials_by_file:
        print("-" * 72)
        print("MTL MATERIALS")
        for mtl_file, mats in materials_by_file.items():
            print(f"[{mtl_file}] materials = {len(mats)}")
            for m in mats:
                print(
                    f"  - {m.name} | Ka={m.ka} Kd={m.kd} Ks={m.ks} Ns={m.ns} map_Kd={m.map_kd}"
                )
    print("=" * 72)


def _mesh_color_from_material(material, fallback=(0.78, 0.80, 0.84)):
    try:
        if material is not None and getattr(material, "main_color", None) is not None:
            c = np.asarray(material.main_color, dtype=np.float32)
            if c.size >= 3:
                rgb = c[:3]
                if np.max(rgb) > 1.0:
                    rgb = rgb / 255.0
                return tuple(float(np.clip(v, 0.0, 1.0)) for v in rgb)
    except Exception:
        pass
    return fallback

def create_texture_from_pil(img: Image.Image) -> int:
    try:
        # Chuyển ảnh về định dạng RGBA để đồng bộ chuẩn OpenGL.
        # np.asarray trực tiếp từ PIL — KHÔNG dùng list(img.getdata()):
        # bản đó tạo hàng triệu Python tuples (vài giây/ảnh 2K).
        img = img.convert("RGBA")
        img_data = np.asarray(img, dtype=np.uint8)
        
        texture_id = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, texture_id)
        
        # Thiết lập các tham số lọc ảnh (làm mượt khi phóng to/thu nhỏ)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_REPEAT)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_REPEAT)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR_MIPMAP_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        
        # Đẩy dữ liệu pixel lên GPU
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, img.width, img.height, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, img_data)
        GL.glGenerateMipmap(GL.GL_TEXTURE_2D)
        
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        return texture_id
    except Exception as e:
        print(f"Lỗi tạo texture: {e}")
        return 0

def load_meshes_with_trimesh(mesh_file: Path):
    try:
        import trimesh
    except Exception as exc:
        # Added explicit notification for pyassimp requirement regarding FBX compatibility
        raise RuntimeError(
            "Missing dependencies 'trimesh' or 'pyassimp'. Install with: pip install trimesh pyassimp"
        ) from exc

    loaded = trimesh.load(str(mesh_file), force="scene", process=False)
    if isinstance(loaded, trimesh.Trimesh):
        geometries = [(mesh_file.stem, loaded)]
    else:
        geometries = list(loaded.geometry.items())

    meshes = []
    for name, mesh in geometries:
        if not isinstance(mesh, trimesh.Trimesh):
            continue
        if mesh.vertices is None or mesh.faces is None or len(mesh.faces) == 0:
            continue

        verts = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.faces, dtype=np.int64)

        tri_vertices = verts[faces].reshape(-1, 3).astype(np.float32)

        # 1. TRÍCH XUẤT NORMALS
        if getattr(mesh, "vertex_normals", None) is not None and len(mesh.vertex_normals) == len(mesh.vertices):
            nrm = np.asarray(mesh.vertex_normals, dtype=np.float32)
            tri_normals = nrm[faces].reshape(-1, 3).astype(np.float32)
        else:
            p0 = verts[faces[:, 0]]
            p1 = verts[faces[:, 1]]
            p2 = verts[faces[:, 2]]
            face_normals = np.cross(p1 - p0, p2 - p0)
            norm = np.linalg.norm(face_normals, axis=1, keepdims=True)
            face_normals = face_normals / np.maximum(norm, 1e-8)
            tri_normals = np.repeat(face_normals, 3, axis=0).astype(np.float32)

        # 2. TRÍCH XUẤT TỌA ĐỘ UV
        if hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None and len(mesh.visual.uv) == len(verts):
            uvs = np.asarray(mesh.visual.uv, dtype=np.float32)
            # Lật trục Y của UV (do OpenGL đọc hệ tọa độ ảnh từ dưới lên)
            uvs[:, 1] = 1.0 - uvs[:, 1]
        else:
            uvs = np.zeros((len(verts), 2), dtype=np.float32)
            
        tri_uvs = uvs[faces].reshape(-1, 2).astype(np.float32)

        # 3. TRÍCH XUẤT ẢNH TEXTURE TỪ VẬT LIỆU
        img = getattr(mesh.visual.material, 'image', None)

        meshes.append(
            {
                "name": f"{mesh_file.name}::{name}",
                "vertices": tri_vertices,
                "normals": tri_normals,
                "uvs": tri_uvs, # Truyền thêm UV
                "image": img,   # Truyền thêm ảnh gốc
                "color": _mesh_color_from_material(getattr(mesh.visual, "material", None)),
            }
        )

    return meshes


class RoadMeshDrawable:
    def __init__(self, vertices: np.ndarray, normals: np.ndarray, uvs: np.ndarray, image, color, model: np.ndarray):
        self.vertices = np.asarray(vertices, dtype=np.float32)
        self.normals = np.asarray(normals, dtype=np.float32)
        self.uvs = np.asarray(uvs, dtype=np.float32)
        self.color = np.asarray(color, dtype=np.float32)
        self.model = np.asarray(model, dtype=np.float32)

        # Tạo Texture nếu có ảnh
        self.texture_id = 0
        if image is not None:
            self.texture_id = create_texture_from_pil(image)

        self.shader = Shader(VERT_SHADER, FRAG_SHADER)
        self.uma = UManager(self.shader)
        self.vao = VAO()

        # Gộp Vertices (3), UVs (2), Normals (3) thành một mảng xen kẽ (Interleaved Array)
        interleaved = np.hstack([self.vertices, self.uvs, self.normals]).astype(np.float32)
        self.indices = np.arange(self.vertices.shape[0], dtype=np.uint32)

        stride = 8 * 4 # Tổng cộng 8 thành phần float (3 + 2 + 3)
        
        # Location 0: Position
        self.vao.add_vbo(0, interleaved, ncomponents=3, dtype=GL.GL_FLOAT, normalized=False, stride=stride, offset=None)
        # Location 1: UV Texcoord
        self.vao.add_vbo(1, interleaved, ncomponents=2, dtype=GL.GL_FLOAT, normalized=False, stride=stride, offset=ctypes.c_void_p(3 * 4))
        # Location 2: Normal
        self.vao.add_vbo(2, interleaved, ncomponents=3, dtype=GL.GL_FLOAT, normalized=False, stride=stride, offset=ctypes.c_void_p(5 * 4))
        
        self.vao.add_ebo(self.indices)

    def draw(
        self,
        projection: np.ndarray,
        view: np.ndarray,
        viewer_shader,
        is_rgb: bool = True,
        eye_pos: np.ndarray = None,
        clip_box: dict = None,
        rgb_dimmed: bool = False,
    ):
        
        if is_rgb:
            # ==========================================
            # CHẾ ĐỘ RGB: Dùng Shader và cấu hình gốc của bạn
            # ==========================================
            active_shader_idx = self.shader.render_idx
            GL.glUseProgram(active_shader_idx)
            
            self.uma.upload_uniform_matrix4fv(projection, "projection", True)
            self.uma.upload_uniform_matrix4fv(view, "view", True)
            self.uma.upload_uniform_matrix4fv(self.model, "model", True)
            self.uma.upload_uniform_vector3fv(self.color, "base_color")
            self.uma.upload_uniform_vector3fv(np.array([0.6, -0.9, -0.35], dtype=np.float32), "light_dir")
            lighting_scale = 0.45 if rgb_dimmed else 1.0
            self.uma.upload_uniform_scalar1f(float(lighting_scale), "lighting_scale")
            
            if eye_pos is not None:
                self.uma.upload_uniform_vector3fv(np.asarray(eye_pos, dtype=np.float32), "eye_pos")

            # Xử lý Texture (RGB Mode)
            loc_use_tex = GL.glGetUniformLocation(active_shader_idx, "use_texture")
            if self.texture_id > 0:
                GL.glActiveTexture(GL.GL_TEXTURE0)
                GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture_id)
                loc_diff_map = GL.glGetUniformLocation(active_shader_idx, "diffuse_map")
                if loc_diff_map != -1: GL.glUniform1i(loc_diff_map, 0)
                if loc_use_tex != -1: GL.glUniform1i(loc_use_tex, 1)
            else:
                if loc_use_tex != -1: GL.glUniform1i(loc_use_tex, 0)

        else:
            # ==========================================
            # CHẾ ĐỘ KHÁC (Depth, Seg...): Dùng Shader của Viewer
            # ==========================================
            active_shader_idx = viewer_shader.render_idx
            GL.glUseProgram(active_shader_idx)
            
            uma_viewer = UManager(viewer_shader)
            modelview = view @ self.model
            
            uma_viewer.upload_uniform_matrix4fv(projection, "projection", transpose=True)
            uma_viewer.upload_uniform_matrix4fv(modelview, "modelview", transpose=True)
            uma_viewer.upload_uniform_matrix4fv(self.model, "model", transpose=True)
            
            # Setup texture cơ bản nếu các mode khác có dùng map alpha/mask
            if self.texture_id > 0:
                GL.glActiveTexture(GL.GL_TEXTURE0)
                GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture_id)
                uma_viewer.upload_uniform_scalar1i(0, "diffuse_map")
                uma_viewer.upload_uniform_scalar1i(1, "has_texture") 
            else:
                GL.glActiveTexture(GL.GL_TEXTURE0)
                GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
                uma_viewer.upload_uniform_scalar1i(0, "has_texture")


        # ==========================================
        # LOGIC CẮT GỌT (CLIPPING) - Áp dụng chung cho mọi Shader
        # ==========================================
        loc_use_clip = GL.glGetUniformLocation(active_shader_idx, "use_clip")
        if loc_use_clip != -1:
            if clip_box is not None:
                GL.glUniform1i(loc_use_clip, 1)
                loc_clip_x = GL.glGetUniformLocation(active_shader_idx, "clip_x")
                loc_clip_z = GL.glGetUniformLocation(active_shader_idx, "clip_z")
                GL.glUniform2f(loc_clip_x, clip_box["min_x"], clip_box["max_x"])
                GL.glUniform2f(loc_clip_z, clip_box["min_z"], clip_box["max_z"])
            else:
                GL.glUniform1i(loc_use_clip, 0)

        # Draw call
        self.vao.activate()
        GL.glDrawElements(GL.GL_TRIANGLES, int(self.indices.size), GL.GL_UNSIGNED_INT, None)
        self.vao.deactivate()


class SamplingViewer:
    def __init__(self, title: str = "Sampling Road Junction Viewer", width: int = 1280, height: int = 800):
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL.GL_TRUE)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

        self.win = glfw.create_window(width, height, title, None, None)
        if not self.win:
            raise RuntimeError("Cannot create GLFW window")
        glfw.make_context_current(self.win)

        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glClearColor(0.06, 0.08, 0.12, 1.0)

        self.drawables: list[RoadMeshDrawable] = []
        
        # Thiết lập Free Camera
        self.camera_pos = np.array([0.0, 5.0, 15.0], dtype=np.float32)
        self.camera_front = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        self.camera_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        
        self.yaw = -90.0
        self.pitch = -20.0
        self.camera_speed = 15.0 # Tốc độ di chuyển
        self.mouse_sensitivity = 0.15

        self.keys = {} # Lưu trạng thái các phím đang được giữ
        self.last_time = glfw.get_time()

        self.dragging = False
        self.last_mouse = None

        glfw.set_key_callback(self.win, self.on_key)
        glfw.set_mouse_button_callback(self.win, self.on_mouse_button)
        glfw.set_cursor_pos_callback(self.win, self.on_mouse_move)
        glfw.set_scroll_callback(self.win, self.on_scroll)

        self._update_camera_vectors()

    def add(self, drawable: RoadMeshDrawable):
        self.drawables.append(drawable)

    def _update_camera_vectors(self):
        # Tính toán lại hướng nhìn của camera dựa trên yaw và pitch
        front = np.zeros(3, dtype=np.float32)
        front[0] = math.cos(math.radians(self.yaw)) * math.cos(math.radians(self.pitch))
        front[1] = math.sin(math.radians(self.pitch))
        front[2] = math.sin(math.radians(self.yaw)) * math.cos(math.radians(self.pitch))
        
        norm = np.linalg.norm(front)
        if norm > 0:
            self.camera_front = front / norm

    def on_key(self, _win, key, _scancode, action, _mods):
        if key in (glfw.KEY_ESCAPE, glfw.KEY_Q) and action == glfw.PRESS:
            glfw.set_window_should_close(self.win, True)
            
        # Ghi nhận trạng thái nhấn/nhả phím
        if action == glfw.PRESS:
            self.keys[key] = True
        elif action == glfw.RELEASE:
            self.keys[key] = False

    def on_mouse_button(self, win, button, action, _mods):
        # Giữ chuột trái để xoay camera
        if button != glfw.MOUSE_BUTTON_LEFT:
            return
        if action == glfw.PRESS:
            self.dragging = True
            self.last_mouse = glfw.get_cursor_pos(win)
        elif action == glfw.RELEASE:
            self.dragging = False
            self.last_mouse = None

    def on_mouse_move(self, _win, x, y):
        if not self.dragging or self.last_mouse is None:
            return
        lx, ly = self.last_mouse
        dx = x - lx
        dy = ly - y # Đảo ngược tọa độ Y
        self.last_mouse = (x, y)

        self.yaw += dx * self.mouse_sensitivity
        self.pitch += dy * self.mouse_sensitivity
        self.pitch = float(np.clip(self.pitch, -89.0, 89.0))
        
        self._update_camera_vectors()

    def on_scroll(self, _win, _dx, dy):
        # Tăng giảm tốc độ bay bằng con lăn chuột
        self.camera_speed = float(np.clip(self.camera_speed + dy * 2.0, 1.0, 100.0))

    def process_input(self, delta_time):
        velocity = self.camera_speed * delta_time
        
        # Di chuyển tới/lui (W/S)
        if self.keys.get(glfw.KEY_W, False):
            self.camera_pos += self.camera_front * velocity
        if self.keys.get(glfw.KEY_S, False):
            self.camera_pos -= self.camera_front * velocity
            
        # Di chuyển trái/phải (A/D)
        if self.keys.get(glfw.KEY_A, False) or self.keys.get(glfw.KEY_D, False):
            right = np.cross(self.camera_front, self.camera_up)
            norm_right = np.linalg.norm(right)
            if norm_right > 0:
                right = right / norm_right
                if self.keys.get(glfw.KEY_A, False):
                    self.camera_pos -= right * velocity
                if self.keys.get(glfw.KEY_D, False):
                    self.camera_pos += right * velocity
                    
        # Bay lên/xuống (E/C)
        if self.keys.get(glfw.KEY_E, False):
            self.camera_pos += self.camera_up * velocity
        if self.keys.get(glfw.KEY_C, False):
            self.camera_pos -= self.camera_up * velocity

    def run(self):
        while not glfw.window_should_close(self.win):
            current_time = glfw.get_time()
            delta_time = current_time - self.last_time
            self.last_time = current_time

            self.process_input(delta_time)

            w, h = glfw.get_framebuffer_size(self.win)
            w = max(1, int(w))
            h = max(1, int(h))
            GL.glViewport(0, 0, w, h)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

            view = T.lookat(self.camera_pos, self.camera_pos + self.camera_front, self.camera_up)
            projection = T.perspective(45.0, float(w) / float(h), 0.1, 1000.0)

            for drawable in self.drawables:
                drawable.draw(projection, view, self.camera_pos)

            glfw.swap_buffers(self.win)
            glfw.poll_events()

    def close(self):
        self.drawables.clear()
        if self.win is not None:
            glfw.destroy_window(self.win)
            self.win = None


def _grid_model(idx: int, count: int, spacing: float = 3.2):
    cols = int(math.ceil(math.sqrt(max(count, 1))))
    row = idx // cols
    col = idx % cols
    x = (col - (cols - 1) * 0.5) * spacing
    z = (row - (cols - 1) * 0.5) * spacing
    return T.translate(x, 0.0, z).astype(np.float32)


def build_drawables_from_folder(folder: Path):
    entries = scan_folder(folder)

    materials_by_file: dict[str, list[MaterialEntry]] = {}
    for e in entries:
        if e.suffix == ".mtl":
            materials_by_file[e.rel] = parse_mtl(e.path)

    print_manifest(folder, entries, materials_by_file)

    mesh_ext = {".obj", ".ply", ".stl", ".glb", ".gltf"}
    
    # Bộ lọc này sẽ chỉ nạp file .obj bạn vừa tạo (và các định dạng an toàn khác).
    mesh_files = [e.path for e in entries if e.suffix in mesh_ext]
    if not mesh_files:
        raise RuntimeError("No native mesh files found (.obj/.ply/.stl/.glb/.gltf)")

    loaded_meshes = []
    errors = []
    for file_path in mesh_files:
        try:
            loaded_meshes.extend(load_meshes_with_trimesh(file_path))
        except Exception as exc:
            errors.append(f"{file_path.name}: {exc}")

    if not loaded_meshes:
        print("No native mesh could be loaded. Falling back to proxy boxes per file.")
        for e in entries:
            loaded_meshes.append(_make_proxy_box(e))

    if errors:
        print("Some files were not loaded natively (using proxy where needed):")
        for e in errors:
            print(" -", e)

    drawables = []
    for i, data in enumerate(loaded_meshes):
        # model = _grid_model(i, len(loaded_meshes))
        identity_model = np.eye(4, dtype=np.float32)
        scale_factor = 1.78
        scale_model = np.eye(4, dtype=np.float32)
        scale_model[0, 0] = scale_factor  # Thu nhỏ trục X
        scale_model[1, 1] = scale_factor  # Thu nhỏ trục Y
        scale_model[2, 2] = scale_factor
        scale_model[0, 3] = -7.0  # Gán giá trị X
        scale_model[1, 3] = 0.2  # fixed
        scale_model[2, 3] = 28.6
        drawables.append(
            RoadMeshDrawable(
                vertices=data["vertices"],
                normals=data["normals"],
                uvs=data["uvs"],      # Thêm dòng này
                image=data["image"],  # Thêm dòng này
                color=data["color"],
                model=scale_model,
            )
        )
    return drawables


def main():
    # 1. Initialize GLFW strictly first to prevent macOS threading and property errors
    if not glfw.init():
        sys.exit("Failed to initialize GLFW")

    parser = argparse.ArgumentParser(
        description="Render all assets/metadata from sampling/road_junction"
    )
    parser.add_argument(
        "--folder",
        default=str(Path(__file__).resolve().parent / "assets" / "road_junction"),
        help="Target folder containing road-junction assets",
    )
    args = parser.parse_args()

    folder = Path(args.folder).expanduser().resolve()
    if not folder.exists() or not folder.is_dir():
        glfw.terminate()
        raise FileNotFoundError(f"Folder not found: {folder}")

    viewer = None
    try:
        viewer = SamplingViewer()
        for drawable in build_drawables_from_folder(folder):
            viewer.add(drawable)

        print("Controls: drag mouse to orbit, wheel to zoom, R to reset, Q/Esc to quit")
        viewer.run()
    finally:
        if viewer is not None:
            viewer.close()
        gc.collect()
        glfw.terminate()


if __name__ == "__main__":
    main()