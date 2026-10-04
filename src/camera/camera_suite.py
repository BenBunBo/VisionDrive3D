# camera_suite.py
import numpy as np
import glfw
import math
from typing import List, Optional, Tuple

from src.core.entity import Node, Scene
from libs import transform as T

class Camera(Node):
    """
    Extends Node to behave as a Camera with Intrinsics.
    The Extrinsic matrix (World to Camera) is the inverse of the world_matrix.
    """
    def __init__(
        self, 
        name: str = "camera", 
        K: Optional[np.ndarray] = None, 
        resolution: Tuple[int, int] = (1280, 720),
        near: float = 0.1,
        far: float = 150.0,
        parent: Optional[Node] = None,
        is_free_cam: bool = False
    ):
        super().__init__(name=name, parent=parent)
        self.resolution = resolution
        self.near = float(near)
        self.far = float(far)
        
        if K is not None:
            self.K = np.array(K, dtype=np.float32)
        else:
            w, h = resolution
            f = w / 2.0
            self.K = np.array([
                [f, 0, w/2.0],
                [0, f, h/2.0],
                [0, 0, 1.0]
            ], dtype=np.float32)

        self.is_free_cam = is_free_cam
        # Free-cam variables
        if self.is_free_cam:
            self.yaw = -90.0
            self.pitch = 0.0
            self.front = np.array([0.0, 0.0, -1.0], dtype=np.float32)
            self.up_ref = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            self.move_speed = 12.0
            self.mouse_sensitivity = 0.10
            self.eye_height = 1.7
            self.enable_ground_follow = False

    def view_matrix(self) -> np.ndarray:
        if self.is_free_cam:
            eye = self.position
            target = self.position + self.front
            return T.lookat(eye, target, self.up_ref)
        else:
            # View matrix is inverse of World matrix
            return np.linalg.inv(self.world_matrix())

    def projection_matrix(self) -> np.ndarray:
        w, h = self.resolution
        if self.is_free_cam:
            aspect = max(1e-6, w / max(1, h))
            return T.perspective(60.0, aspect, self.near, self.far)
            
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        
        P = np.zeros((4, 4), dtype=np.float32)
        P[0, 0] = 2.0 * fx / w
        P[1, 1] = 2.0 * fy / h
        P[0, 2] = 1.0 - 2.0 * cx / w   # OpenGL NDC mapping
        P[1, 2] = 2.0 * cy / h - 1.0
        P[2, 2] = -(self.far + self.near) / (self.far - self.near)
        P[2, 3] = -2.0 * self.far * self.near / (self.far - self.near)
        P[3, 2] = -1.0
        
        return P

    def get_extrinsics(self) -> np.ndarray:
        """ Returns the 4x4 Extrinsic matrix (World to Camera). """
        return self.view_matrix()

    def get_intrinsics(self) -> np.ndarray:
        """ Returns the 3x3 Intrinsic matrix (K). """
        return self.K
        
    def _update_front(self):
        yaw_r = math.radians(self.yaw)
        pitch_r = math.radians(self.pitch)
        front = np.array(
            [
                math.cos(yaw_r) * math.cos(pitch_r),
                math.sin(pitch_r),
                math.sin(yaw_r) * math.cos(pitch_r),
            ],
            dtype=np.float32,
        )
        n = np.linalg.norm(front)
        if n > 1e-8:
            self.front = front / n

    def process_mouse_delta(self, dx: float, dy: float):
        if not self.is_free_cam: return
        self.yaw += float(dx) * self.mouse_sensitivity
        self.pitch -= float(dy) * self.mouse_sensitivity
        self.pitch = float(np.clip(self.pitch, -89.0, 89.0))
        self._update_front()

    def process_keyboard(self, window, dt: float):
        if not self.is_free_cam: return
        v = self.move_speed * float(dt)
        right = np.cross(self.front, self.up_ref)
        rnorm = np.linalg.norm(right)
        if rnorm > 1e-8:
            right = right / rnorm

        # WASD in XZ plane.
        forward_flat = np.array([self.front[0], 0.0, self.front[2]], dtype=np.float32)
        fnorm = np.linalg.norm(forward_flat)
        if fnorm > 1e-8:
            forward_flat /= fnorm

        if glfw.get_key(window, glfw.KEY_W) == glfw.PRESS:
            self.position += forward_flat * v
        if glfw.get_key(window, glfw.KEY_S) == glfw.PRESS:
            self.position -= forward_flat * v
        if glfw.get_key(window, glfw.KEY_A) == glfw.PRESS:
            self.position -= right * v
        if glfw.get_key(window, glfw.KEY_D) == glfw.PRESS:
            self.position += right * v

        if glfw.get_key(window, glfw.KEY_E) == glfw.PRESS:
            self.position[1] += v
        if glfw.get_key(window, glfw.KEY_Q) == glfw.PRESS or glfw.get_key(window, glfw.KEY_C) == glfw.PRESS:
            self.position[1] -= v

    def update_ground_alignment(self, scene: Scene, dt: float):
        if not self.is_free_cam or not self.enable_ground_follow:
            return
        ground_y = scene.raycast_ground_height(
            float(self.position[0]),
            float(self.position[2]),
            default_y=0.0,
        )
        target_y = ground_y + self.eye_height
        alpha = float(np.clip(dt * self.ground_follow_lerp, 0.0, 1.0))
        self.position[1] = float(self.position[1] + alpha * (target_y - self.position[1]))


class CameraPresetFactory:
    @staticmethod
    def create_kitti_stereo(ego_vehicle: Node) -> List[Camera]:
        """ Create Cam 2 and Cam 3 from KITTI (approx). Front facing. """
        w, h = 1242, 375
        f = 721.5
        K = np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]], dtype=np.float32)
        
        cam_left = Camera(name="kitti_cam_2", K=K, resolution=(w, h), parent=ego_vehicle)
        cam_left.position = np.array([-0.27, 1.65, 0.0], dtype=np.float32)
        cam_left.rotation = np.array([0, 0, 0], dtype=np.float32)
        
        cam_right = Camera(name="kitti_cam_3", K=K, resolution=(w, h), parent=ego_vehicle)
        cam_right.position = np.array([0.27, 1.65, 0.0], dtype=np.float32)
        cam_right.rotation = np.array([0, 0, 0], dtype=np.float32)
        return [cam_left, cam_right]

    @staticmethod
    def create_nuscenes_surround(ego_vehicle: Node) -> List[Camera]:
        """ Create 6 surround cameras approx matching nuScenes. """
        w, h = 1600, 900
        f = 1200.0
        K = np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]], dtype=np.float32)
        
        cameras = []
        # Yaws relative to the car. Our car models face LOCAL +Z (verified via
        # wheel offsets: front wheels at +Z), while a GL camera looks along its
        # LOCAL -Z. So a camera with yaw=0 actually looks at the car's REAR:
        # CAM_FRONT must use yaw=180 to see the road ahead.
        # Car space: +Z = forward, +X = left.
        configs = [
            ("CAM_FRONT", 180.0, [0.0, 1.7, 0.5]),
            ("CAM_FRONT_LEFT", -125.0, [0.5, 1.7, 0.0]),
            ("CAM_FRONT_RIGHT", 125.0, [-0.5, 1.7, 0.0]),
            ("CAM_BACK", 0.0, [0.0, 1.7, -0.5]),
            ("CAM_BACK_LEFT", -55.0, [0.5, 1.7, -0.2]),
            ("CAM_BACK_RIGHT", 55.0, [-0.5, 1.7, -0.2]),
        ]
        
        for name, yaw_deg, pos in configs:
            cam = Camera(name=name, K=K, resolution=(w, h), parent=ego_vehicle)
            cam.position = np.array(pos, dtype=np.float32)
            cam.rotation = np.array([0, yaw_deg, 0], dtype=np.float32)
            cameras.append(cam)
        return cameras

class CameraManager:
    def __init__(self):
        self.cameras: List[Camera] = []
        self.active_camera_idx: int = 0
        
    def add_camera(self, camera: Camera):
        self.cameras.append(camera)
        
    def add_cameras(self, cameras: List[Camera]):
        self.cameras.extend(cameras)
        
    def set_active_camera(self, idx: int):
        if 0 <= idx < len(self.cameras):
            self.active_camera_idx = idx
            
    def get_active_camera(self) -> Camera:
        if not self.cameras:
            raise RuntimeError("No cameras available")
        return self.cameras[self.active_camera_idx]
        
    def get_all_cameras(self) -> List[Camera]:
        return self.cameras

    def switch_next(self):
        if self.cameras:
            self.active_camera_idx = (self.active_camera_idx + 1) % len(self.cameras)
