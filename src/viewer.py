import os
import random
from itertools import cycle

import cv2
import glfw
import numpy as np
import OpenGL.GL as GL

from src.annotation.annotations import compute_bbox_3d_kitti, project_entity_bbox_2d
from src.core.mesh import Mesh
from src.core.entity import Scene, Entity
from src.camera.camera_suite import Camera, CameraPresetFactory, CameraManager
from src.render.renderers import RenderManager
from src.scene.car import Car
from src.scene.scene_overlay import SceneOverlay
from src.scene.traffic import TrafficManager
from src.dataset.dataset_manager import DatasetManager


def instance_color_from_id(idx: int) -> tuple[float, float, float]:
    """
    Generate distinct RGB colors using bit manipulation.
    Works well for many instances.
    """
    idx = idx * 2654435761  # Knuth hash (helps distribution)

    r = (idx & 0xFF) / 255.0
    g = ((idx >> 8) & 0xFF) / 255.0
    b = ((idx >> 16) & 0xFF) / 255.0

    return (r, g, b)


class ViewerApp:
    def __init__(self, width: int = 1280, height: int = 720, headless: bool = False,
                 output_dir: str = "./output_dataset", multiview: bool = False,
                 frame_dt: float = 0.5):
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")

        self.headless = headless
        self.output_dir = output_dir
        self.multiview = bool(multiview)
        # Simulated seconds of traffic between two captured frames (headless).
        self.frame_dt = float(frame_dt)

        if headless:
            self._init_offscreen(width, height)
        else:
            glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
            glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
            glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL.GL_TRUE)
            glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
            glfw.window_hint(glfw.DEPTH_BITS, 24)
            glfw.window_hint(glfw.DOUBLEBUFFER, True)

            self.window = glfw.create_window(width, height, "VisionDrive3D - Synthetic Generator", None, None)
            if not self.window:
                glfw.terminate()
                raise RuntimeError("Failed to create GLFW window")

            glfw.make_context_current(self.window)
            glfw.swap_interval(1)

            # Lock cursor for FPS fly camera
            glfw.set_input_mode(self.window, glfw.CURSOR, glfw.CURSOR_DISABLED)

        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDepthFunc(GL.GL_LESS)
        GL.glClearColor(0.18, 0.20, 0.24, 1.0)

        self.fill_modes = cycle([GL.GL_FILL, GL.GL_LINE, GL.GL_POINT])

        self.scene = Scene()

        # 1. Setup CameraManager and the default free camera
        self.camera_manager = CameraManager()
        self.free_camera = Camera(name="free_cam", resolution=(1280, 720), near=0.1, far=150.0, is_free_cam=True)
        self.free_camera.position[:] = np.array([0.0, 2.2, 12.0], dtype=np.float32)
        self.camera_manager.add_camera(self.free_camera)

        # 2. Create nuScenes cameras WITHOUT a parent for now
        #    They will be re-parented to a traffic car once the pool is ready.
        self._nuscenes_cameras = CameraPresetFactory.create_nuscenes_surround(None)
        self.camera_manager.add_cameras(self._nuscenes_cameras)
        self._camera_host_car = None  # the traffic car cameras are attached to

        # Repo root (src/viewer.py -> parent of src/), used for shaders/ etc.
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.scene_overlay = SceneOverlay()
        self.render_manager = RenderManager(
            scene=self.scene,
            base_dir=base_dir,
            scene_overlay=self.scene_overlay,
            output_dir=os.path.join(self.output_dir, "multiview"),
            near=0.1,
            far=150.0,
        )
        
        self.last_time = glfw.get_time()
        self.last_mouse_pos = None

        # Traffic manager (wired after scene assets load)
        self.traffic_manager = TrafficManager()

        if not self.headless:
            glfw.set_key_callback(self.window, self._on_key)
            glfw.set_cursor_pos_callback(self.window, self._on_mouse_move)

        self._load_scene_assets(base_dir)

        self.scene_id = 0
        self.dataset_manager = DatasetManager(
            root=self.output_dir, validate=True
        )

    # ------------------------------------------------------------------
    # Camera ↔ traffic car attachment
    # ------------------------------------------------------------------
    def _attach_cameras_to_car(self, car):
        """Re-parent all nuScenes cameras to the given traffic car."""
        self._camera_host_car = car
        for cam in self._nuscenes_cameras:
            car.add_child(cam)
        print(f"Cameras attached to: {car.name}")

    def _ensure_camera_host(self):
        """
        If the current host car is gone / finished / inactive,
        pick a random active traffic car and re-attach cameras.
        """
        host = self._camera_host_car
        need_new = (
            host is None
            or not host.is_active
            or host.route_finished
        )
        if not need_new:
            return

        active = [c for c in self.traffic_manager.cars
                  if c.is_active and not c.route_finished]
        if not active:
            return  # no cars available yet

        new_host = random.choice(active)
        self._attach_cameras_to_car(new_host)

    @staticmethod
    def _default_lanes_config() -> list[dict]:
        return [
            {"x_center": -3.5, "z_min": -42.0, "z_max": 10.0, "direction": [0.0, 0.0, -1.0], "ground_y": 0.05, "x_jitter": 0.28, "safe_z": 3.2},
            {"x_center": -1.8, "z_min": -42.0, "z_max": 10.0, "direction": [0.0, 0.0, -1.0], "ground_y": 0.05, "x_jitter": 0.22, "safe_z": 3.0},
            {"x_center": 1.8, "z_min": -42.0, "z_max": 10.0, "direction": [0.0, 0.0, 1.0], "ground_y": 0.05, "x_jitter": 0.22, "safe_z": 3.0},
            {"x_center": 3.5, "z_min": -42.0, "z_max": 10.0, "direction": [0.0, 0.0, 1.0], "ground_y": 0.05, "x_jitter": 0.28, "safe_z": 3.2},
        ]

    def _load_scene_assets(self, base_dir: str):
        # 1) Static environment (do not randomize its transform)
        street_obj = os.path.join("assets", "scene", "Street environment_V01.obj")
        street_mesh = Mesh(street_obj).setup()
        street = Entity(
            name="street_environment",
            mesh=street_mesh,
            class_id=-1,
            instance_color=(-1.0, 0.0, 0.0), # Flag for ground/house dynamic shader
            is_dynamic=False,
        )
        street.position[:] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        street.scale[:] = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        self.scene.add_entity(street)

        # 2) Dynamic cars – Object Pool (5 cars, random car0/car1/car2)
        car_folders = ["assets/car0", "assets/car1", "assets/car2"]
        num_cars = 7
        for i in range(num_cars):
            folder = random.choice(car_folders)
            car = Car(
                car_folder=folder,
                name=f"car_{i:02d}",
                class_id=5,  # dynamic traffic cars
                instance_color=instance_color_from_id(i + 1),
            )
            self.scene.add_entity(car)

        # 3) Initialize traffic manager with the car pool
        traffic_cars = self.scene.get_traffic_cars()
        self.traffic_manager.register_cars(traffic_cars)

        # 4) Attach nuScenes cameras to a random traffic car
        self._ensure_camera_host()

    def _on_key(self, _win, key, _scancode, action, _mods):
        if action not in (glfw.PRESS, glfw.REPEAT):
            return

        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(self.window, True)
            return

        if key == glfw.KEY_F:
            GL.glPolygonMode(GL.GL_FRONT_AND_BACK, next(self.fill_modes))
            return

        if key == glfw.KEY_1:
            self.render_manager.set_mode("RGB")
            return

        if key == glfw.KEY_2:
            self.render_manager.set_mode("MASK")
            return

        if key == glfw.KEY_3:
            self.render_manager.set_mode("DEPTH")
            return

        if key == glfw.KEY_T:
            if self.render_manager.mode == "RGB":
                is_dimmed = self.render_manager.toggle_rgb_dimmed()
                print("RGB lighting: DIMMED" if is_dimmed else "RGB lighting: NORMAL")
            else:
                print("Lighting toggle is available only in RGB mode.")
            return

        if key == glfw.KEY_R:
            # Reset all traffic routes
            traffic_cars = self.scene.get_traffic_cars()
            self.traffic_manager.register_cars(traffic_cars)
            self._ensure_camera_host()
            print("Traffic routes reset.")
            return

        if key == glfw.KEY_TAB:
            self.camera_manager.switch_next()
            print(f"Switched to camera: {self.camera_manager.get_active_camera().name}")
            return

        if key == glfw.KEY_G:
            self._run_auto_generate_hook(num_frames=500)
            return

        if key == glfw.KEY_P:
            views = self.render_manager.export_current_frame(
                self.camera_manager,
                multiview=self.multiview,
                exclude_names=self._label_exclude_names(),
            )
            self._export_frame_to_dataset(self.scene_id, views)
            print(f"Exported scene {self.scene_id:05d}")
            self.scene_id += 1
            return

    def _run_auto_generate_hook(self, num_frames=5):
        print(f"--- Starting automated dataset generation for {num_frames} frames ---")
        for i in range(num_frames):
            # randomizes scene layout
            self.scene.spawn_cars_on_lanes(
                lanes_config=self._default_lanes_config(),
                num_cars=len(self.scene.get_dynamic_entities()),
                scale_range=(0.9, 1.15),
                sat_padding=0.20,
                max_retry_per_car=120,
            )

            # Re-attach cameras to a random active car for this frame
            self._ensure_camera_host()

            views = self.render_manager.export_current_frame(
                self.camera_manager,
                multiview=self.multiview,
                exclude_names=self._label_exclude_names(),
            )
            self._export_frame_to_dataset(self.scene_id, views)
            print(f"Generated scene {self.scene_id:05d}")
            self.scene_id += 1
        print("--- Automated generation complete ---")

    # ------------------------------------------------------------------
    # Unified dataset export (shared by interactive P/G and headless mode)
    # ------------------------------------------------------------------
    def _export_frame_to_dataset(self, scene_id: int, views: dict) -> dict:
        """
        Write the ACTIVE camera view + labels + metadata of one exported
        frame into the dataset tree (images/ masks/ depth/ labels/ metadata/).

        ``views`` is the dict returned by RenderManager.export_current_frame.
        """
        active_cam = self.camera_manager.get_active_camera()
        view = views[active_cam.name]
        w, h = view["width"], view["height"]
        K, E = view["intrinsics"], view["extrinsics"]

        # Camera pose in WORLD space. Free cam keeps its pose in .position /
        # .front; rigged cameras (nuScenes on a car) must be derived from
        # their world matrix (.position is only the car-local offset).
        if hasattr(active_cam, "front"):
            cam_pos = np.asarray(active_cam.position, dtype=np.float32)
            cam_target = cam_pos + np.asarray(active_cam.front, dtype=np.float32)
        elif hasattr(active_cam, "target"):
            cam_pos = np.asarray(active_cam.position, dtype=np.float32)
            cam_target = np.asarray(active_cam.target, dtype=np.float32)
        else:
            wm = active_cam.world_matrix()
            cam_pos = wm[:3, 3]
            cam_target = wm[:3, 3] - wm[:3, 2]  # GL camera looks along local -Z

        camera_params = {
            "position": [float(v) for v in cam_pos],
            "target":   [float(v) for v in cam_target],
            "up":       active_cam.up.tolist() if hasattr(active_cam, 'up') else [0,1,0],
            "fov_deg":  active_cam.fov if hasattr(active_cam, 'fov') else 45.0,
            "near":     float(active_cam.near),
            "far":      float(active_cam.far),
            "intrinsic_matrix":  np.asarray(K).tolist(),
            "extrinsic_matrix":  np.asarray(E).tolist(),
        }

        # Ego-vehicle view: the camera host surrounds the rig and must not
        # be annotated (nuScenes/KITTI convention).
        host_car = self._camera_host_car
        ego_view = active_cam in self._nuscenes_cameras

        objects = []
        for entity in self.scene.entities:
            obj_class = getattr(entity, "class_name", type(entity).__name__)
            is_dynamic = bool(getattr(entity, "is_dynamic", False)) \
                and int(getattr(entity, "class_id", -1)) >= 0
            if is_dynamic and not (ego_view and entity is host_car):
                bbox_2d = project_entity_bbox_2d(entity, K, E, w, h)
                bbox_3d = compute_bbox_3d_kitti(entity)
            else:
                bbox_2d, bbox_3d = [0, 0, 0, 0], None
            objects.append({
                "instance_id":    getattr(entity, "instance_id", id(entity)),
                "class_name":     obj_class,
                "position_world": entity.position.tolist(),
                "rotation_euler": getattr(entity, "rotation", [0,0,0]),
                "scale":          getattr(entity, "scale", [1,1,1]),
                "bbox_2d":        bbox_2d,
                "bbox_3d":        bbox_3d,
                "visible":        getattr(entity, "visible", True),
                "occlusion_ratio":getattr(entity, "occlusion_ratio", 0.0),
                "annotate":       obj_class not in {"Entity", "entity"},
            })

        render_config = {
            "resolution": [w, h],
            "lighting":   "directional",
            "num_lights": len(getattr(self.scene, "lights", [])),
            "background": "road_scene",
        }

        paths = self.dataset_manager.begin_export(scene_id)
        cv2.imwrite(str(paths["image"]), cv2.cvtColor(view["rgb"], cv2.COLOR_RGB2BGR))
        np.save(str(paths["depth"]), view["depth"])
        cv2.imwrite(str(paths["mask"]), view["mask"])

        self.dataset_manager.finish_scene(
            scene_id=scene_id,
            camera_params=camera_params,
            objects=objects,
            render_config=render_config,
            rgb_image=view["rgb"],
            depth_map=view["depth"],
            seg_mask=view["mask"],
        )
        return paths

    def _on_mouse_move(self, _win, xpos, ypos):
        if self.last_mouse_pos is None:
            self.last_mouse_pos = (xpos, ypos)
            return

        dx = xpos - self.last_mouse_pos[0]
        dy = ypos - self.last_mouse_pos[1]
        self.last_mouse_pos = (xpos, ypos)

        # Mouse look for fly navigation
        active_cam = self.camera_manager.get_active_camera()
        active_cam.process_mouse_delta(dx, dy)

    def _init_offscreen(self, width: int, height: int):
        """
        Initialize an OpenGL offscreen context using GLFW with a hidden window.
        Falls back to OSMesa if GLFW headless hint is unavailable.
        """
        # Tell GLFW not to show the window at all
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

        self.window = glfw.create_window(width, height, "offscreen", None, None)
        if not self.window:
            raise RuntimeError(
                "GLFW offscreen context creation failed. "
                "On Linux servers, try: export DISPLAY=:99 or install osmesa."
            )
        glfw.make_context_current(self.window)

        # Create FBO + texture + depth renderbuffer
        self._fbo = GL.glGenFramebuffers(1)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._fbo)

        self._color_tex = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._color_tex)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGB,
                        width, height, 0,
                        GL.GL_RGB, GL.GL_UNSIGNED_BYTE, None)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER,
                                  GL.GL_COLOR_ATTACHMENT0,
                                  GL.GL_TEXTURE_2D, self._color_tex, 0)

        self._depth_rbo = GL.glGenRenderbuffers(1)
        GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, self._depth_rbo)
        GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, GL.GL_DEPTH24_STENCIL8,
                                 width, height)
        GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER,
                                     GL.GL_DEPTH_STENCIL_ATTACHMENT,
                                     GL.GL_RENDERBUFFER, self._depth_rbo)

        status = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
        if status != GL.GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError(f"FBO incomplete, status: {hex(status)}")

        self._offscreen_width  = width
        self._offscreen_height = height

    def _destroy_offscreen(self):
        GL.glDeleteFramebuffers(1, [self._fbo])
        GL.glDeleteTextures(1, [self._color_tex])
        GL.glDeleteRenderbuffers(1, [self._depth_rbo])
        glfw.destroy_window(self.window)
        glfw.terminate()

    def _activate_front_camera(self) -> bool:
        """
        Make the nuScenes CAM_FRONT (currently attached to a traffic car) the
        active camera, so the exported dataset is an ego-vehicle front view.
        """
        for idx, cam in enumerate(self.camera_manager.get_all_cameras()):
            if cam.name == "CAM_FRONT":
                self.camera_manager.set_active_camera(idx)
                host = self._camera_host_car.name if self._camera_host_car else "None"
                print(f"Active camera: {cam.name} (ego host: {host})")
                return True
        return False

    def _label_exclude_names(self) -> set:
        """
        Entity names excluded from labels: the ego vehicle carrying the
        cameras (it surrounds the camera rig, so it must not be annotated —
        same convention as nuScenes/KITTI ego vehicles).
        """
        active_cam = self.camera_manager.get_active_camera()
        if self.multiview or active_cam in self._nuscenes_cameras:
            if self._camera_host_car is not None:
                return {self._camera_host_car.name}
        return set()

    def run_headless(self, num_frames: int):
        """
        Run dataset generation fully offscreen. No window is shown.

        - Cars spawn ONCE (from their traffic routes at startup) and then flow
          continuously; each captured frame advances the simulation by
          ``self.frame_dt`` seconds (--dt, default 0.1 s).
        - The active camera is auto-attached to CAM_FRONT of an available
          traffic car (ego-vehicle front view).
        """
        # Auto-attach the active camera to the front cam of an available car.
        self._ensure_camera_host()
        self._activate_front_camera()

        for i in range(num_frames):
            # Traffic flow only -- NO per-frame re-randomization.
            self._ensure_camera_host()
            # Substep the simulation so large --dt values take effect while
            # keeping per-step dt <= 0.1s (the safety clamp inside
            # TrafficManager.update / Car.update still applies).
            sim_dt = max(0.0, float(self.frame_dt))
            steps = max(1, int(np.ceil(sim_dt / 0.1)))
            sub_dt = sim_dt / steps
            for _ in range(steps):
                self.traffic_manager.update(sub_dt)
                self.scene.update(sub_dt)

            active_cam = self.camera_manager.get_active_camera()
            # To get ground alignment properly if needed
            active_cam.update_ground_alignment(self.scene, self.frame_dt)

            views = self.render_manager.export_current_frame(
                self.camera_manager,
                multiview=self.multiview,
                exclude_names=self._label_exclude_names(),
            )
            paths = self._export_frame_to_dataset(i, views)

            print(f"[headless] scene {i+1}/{num_frames} exported -> {paths['image']}")

        self._destroy_offscreen()
        print(f"[headless] done. Dataset written to: {self.output_dir}")

    def run(self):
        print("Controls: WASD + E/Q(C) move | Mouse look | TAB switch cam | R respawn | P export | G gen dataset | 1/2/3 mode | T toggle RGB light | F fill")

        while not glfw.window_should_close(self.window):
            now = glfw.get_time()
            dt = float(max(1e-6, now - self.last_time))
            self.last_time = now

            glfw.poll_events()

            # Auto-switch camera host if current car is finished/inactive
            self._ensure_camera_host()

            active_cam = self.camera_manager.get_active_camera()
            active_cam.process_keyboard(self.window, dt)
            active_cam.update_ground_alignment(self.scene, dt)

            w, h = glfw.get_framebuffer_size(self.window)
            GL.glViewport(0, 0, w, h)
            # Default clearing for RGB interactive viewing
            if self.render_manager.mode == "RGB" and self.render_manager.is_rgb_dimmed():
                GL.glClearColor(0.07, 0.08, 0.10, 1.0)
            else:
                GL.glClearColor(0.18, 0.20, 0.24, 1.0)
            if self.render_manager.mode != "MASK":
                GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

            projection = active_cam.projection_matrix()
            if active_cam.is_free_cam:
                active_cam.resolution = (w, h)
                projection = active_cam.projection_matrix()

            # Traffic AI: collision avoidance + route recycling
            self.traffic_manager.update(dt)
            self.scene.update(dt)
            view = active_cam.view_matrix()

            self.render_manager.draw(projection, view)

            glfw.swap_buffers(self.window)

        glfw.terminate()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="VisionDrive3D")
    parser.add_argument("--headless",    action="store_true", help="Run offscreen, no window")
    parser.add_argument("--frames",      type=int, default=10, help="Number of scenes to generate (headless only)")
    parser.add_argument("--output",      type=str, default="./output_dataset", help="Dataset output directory (all dataset artifacts go here)")
    parser.add_argument("--multiview",   action="store_true", help="Export ALL surround cameras (default: active camera only)")
    parser.add_argument("--dt",          type=float, default=0.5, help="Simulated seconds of traffic per captured frame (headless; 0.016 = realtime)")
    parser.add_argument("--width",       type=int, default=1280)
    parser.add_argument("--height",      type=int, default=720)
    args = parser.parse_args()

    if not glfw.init():
        raise RuntimeError("GLFW failed to initialize")

    if args.headless:
        app = ViewerApp(width=args.width, height=args.height, headless=True,
                        output_dir=args.output, multiview=args.multiview,
                        frame_dt=args.dt)
        app.run_headless(num_frames=args.frames)
    else:
        app = ViewerApp(width=args.width, height=args.height, headless=False,
                        output_dir=args.output, multiview=args.multiview,
                        frame_dt=args.dt)
        app.run()
