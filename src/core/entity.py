# entity.py
import math
from typing import List, Optional, Tuple

import glfw
import numpy as np

from libs import transform as T


def _rotation_matrix_xyz(deg_xyz: np.ndarray) -> np.ndarray:
    """
    Build Euler rotation matrix with order Rx * Ry * Rz (degrees input).
    """
    rx, ry, rz = np.radians(deg_xyz.astype(np.float32))
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)

    rx_m = np.array([
        [1, 0, 0, 0],
        [0, cx, -sx, 0],
        [0, sx, cx, 0],
        [0, 0, 0, 1],
    ], dtype=np.float32)

    ry_m = np.array([
        [cy, 0, sy, 0],
        [0, 1, 0, 0],
        [-sy, 0, cy, 0],
        [0, 0, 0, 1],
    ], dtype=np.float32)

    rz_m = np.array([
        [cz, -sz, 0, 0],
        [sz, cz, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], dtype=np.float32)

    return rx_m @ ry_m @ rz_m


def _sat_aabb_overlap_xz(a_min: np.ndarray, a_max: np.ndarray, b_min: np.ndarray, b_max: np.ndarray) -> bool:
    """
    SAT for axis-aligned boxes in 2D XZ plane:
    Non-overlap exists if there is a separating axis on X or Z.
    """
    separated_x = a_max[0] <= b_min[0] or a_min[0] >= b_max[0]
    separated_z = a_max[1] <= b_min[1] or a_min[1] >= b_max[1]
    return not (separated_x or separated_z)


class Node:
    """
    Scene graph node with local TRS and hierarchical world matrix.
    """

    def __init__(self, name: str = "node", parent: Optional["Node"] = None):
        self.name = name
        self.parent: Optional[Node] = None
        self.children: List[Node] = []

        self.position = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.rotation = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # euler deg
        self.scale = np.array([1.0, 1.0, 1.0], dtype=np.float32)

        if parent is not None:
            parent.add_child(self)

    def add_child(self, child: "Node"):
        if child.parent is not None:
            child.parent.remove_child(child)
        child.parent = self
        self.children.append(child)

    def remove_child(self, child: "Node"):
        if child in self.children:
            self.children.remove(child)
            child.parent = None

    def local_matrix(self) -> np.ndarray:
        t = T.translate(self.position)
        r = _rotation_matrix_xyz(self.rotation)
        s = T.scale(self.scale)
        return t @ r @ s

    def world_matrix(self) -> np.ndarray:
        local = self.local_matrix()
        if self.parent is None:
            return local
        return self.parent.world_matrix() @ local


class Entity(Node):
    """
    Renderable object wrapping Mesh + annotation metadata.
    """

    def __init__(
        self,
        name: str,
        mesh,
        class_id: int,
        instance_color: Tuple[float, float, float],
        is_dynamic: bool = True,
        parent: Optional[Node] = None,
    ):
        super().__init__(name=name, parent=parent)
        self.mesh = mesh
        self.class_id = int(class_id)
        self.instance_color = np.asarray(instance_color, dtype=np.float32)
        self.is_dynamic = bool(is_dynamic)

        local_min, local_max = self.mesh.compute_local_aabb()
        self.local_aabb_min = local_min.astype(np.float32)
        self.local_aabb_max = local_max.astype(np.float32)
        self.local_size = (self.local_aabb_max - self.local_aabb_min).astype(np.float32)

        # Distance from origin to lowest local Y, used to place object on ground.
        self.bottom_y_offset = float(-self.local_aabb_min[1])

    @property
    def mesh_extents(self) -> List[float]:
        """Return [hx, hy, hz] half-extents for bounding box projection."""
        if self.local_size is None:
            return None
        s = np.array(self.local_size, dtype=np.float32)
        # If local_size is full dimensions, return half-extents:
        return (s / 2.0).tolist()

    def place_on_surface(self, x: float, z: float, ground_y: float):
        """
        Ensure mesh bottom touches road plane:
        y = ground_y + bottom_offset * scale_y
        """
        self.position[0] = float(x)
        self.position[2] = float(z)
        self.position[1] = float(ground_y) + self.bottom_y_offset * float(self.scale[1])

    def world_aabb_xz(self, padding: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Approximate world AABB on XZ (axis-aligned) for SAT overlap checks.
        """
        half_x = 0.5 * float(self.local_size[0] * self.scale[0]) + padding
        half_z = 0.5 * float(self.local_size[2] * self.scale[2]) + padding
        center_x = float(self.position[0])
        center_z = float(self.position[2])

        mn = np.array([center_x - half_x, center_z - half_z], dtype=np.float32)
        mx = np.array([center_x + half_x, center_z + half_z], dtype=np.float32)
        return mn, mx

    def update(self, dt: float):
        _ = dt
        pass


class Scene:
    """
    Holds entities and domain randomization logic.
    """

    def __init__(self):
        self.entities: List[Entity] = []

    def add_entity(self, entity: Entity):
        self.entities.append(entity)

    def get_dynamic_entities(self) -> List[Entity]:
        return [e for e in self.entities if e.is_dynamic and e.class_id >= 0]

    def get_static_entities(self) -> List[Entity]:
        return [e for e in self.entities if not e.is_dynamic]

    def get_traffic_cars(self) -> list:
        """Return dynamic Car entities that participate in traffic (class_id == 5)."""
        from src.scene.car import Car
        return [e for e in self.entities if isinstance(e, Car) and e.class_id == 5]

    def update(self, dt: float):
        for ent in self.get_dynamic_entities():
            ent.update(dt)

    def raycast_ground_height(self, x: float, z: float, default_y: float = 0.0) -> float:
        """
        Lightweight raycast approximation:
        project a vertical ray and find the top Y of static entities whose XZ AABB contains (x, z).
        """
        hit_y = default_y
        found = False

        for ent in self.get_static_entities():
            local_min = ent.local_aabb_min * ent.scale
            local_max = ent.local_aabb_max * ent.scale
            center = ent.position

            x_min = float(center[0] + local_min[0])
            x_max = float(center[0] + local_max[0])
            z_min = float(center[2] + local_min[2])
            z_max = float(center[2] + local_max[2])

            if x_min <= x <= x_max and z_min <= z <= z_max:
                y_top = float(center[1] + local_max[1])
                if not found or y_top > hit_y:
                    hit_y = y_top
                    found = True

        return hit_y

    def spawn_cars_on_lanes(
        self,
        lanes_config: List[dict],
        num_cars: Optional[int] = None,
        scale_range: Tuple[float, float] = (0.9, 1.15),
        lateral_jitter_default: float = 0.25,
        sat_padding: float = 0.12,
        same_lane_safe_distance: float = 2.8,
        max_retry_per_car: int = 60,
    ):
        """
        Lane-based spawning for dynamic cars.

        Each lane dict supports:
            {
                "x_center": float,
                "z_min": float,
                "z_max": float,
                "direction": [dx, dy, dz],
                "ground_y": 0.05,
                "x_jitter": 0.2,
                "safe_z": 3.0
            }

        This method fixes: wrong orientation, building collision risk, car overlap,
        and ground alignment via place_on_surface(bottom_y_offset).
        """
        if not lanes_config:
            return

        cars_all = self.get_dynamic_entities()
        if num_cars is None:
            cars = cars_all
        else:
            cars = cars_all[: max(0, int(num_cars))]
        if not cars:
            return

        # Per-lane placement state for optimized collision tests.
        lane_z_used: dict[int, List[float]] = {idx: [] for idx in range(len(lanes_config))}
        lane_aabbs: dict[int, List[Tuple[np.ndarray, np.ndarray]]] = {idx: [] for idx in range(len(lanes_config))}

        for car in cars:
            spawned = False

            # Slight scale variation while keeping realistic vehicle proportions.
            s = float(np.random.uniform(scale_range[0], scale_range[1]))
            car.scale[:] = np.array([s, s, s], dtype=np.float32)

            for _ in range(max_retry_per_car):
                # Randomly choose a lane for this spawn attempt.
                lane_idx = int(np.random.choice(len(lanes_config)))
                lane = lanes_config[lane_idx]

                x_center = float(lane.get("x_center", 0.0))
                z_min = float(lane.get("z_min", -30.0))
                z_max = float(lane.get("z_max", 5.0))
                ground_y = float(lane.get("ground_y", 0.05))
                x_jitter = float(lane.get("x_jitter", lateral_jitter_default))
                safe_z = float(lane.get("safe_z", same_lane_safe_distance))

                direction = np.asarray(lane.get("dir", lane.get("direction", [0.0, 0.0, -1.0])), dtype=np.float32)
                if direction.shape[0] < 3:
                    direction = np.array([0.0, 0.0, -1.0], dtype=np.float32)

                # Convert lane direction to yaw around global Y.
                # Local +Z of car aligns to lane direction projected on XZ.
                dir_x = float(direction[0])
                dir_z = float(direction[2])
                if abs(dir_x) < 1e-6 and abs(dir_z) < 1e-6:
                    dir_z = -1.0
                yaw_deg = math.degrees(math.atan2(dir_x, dir_z))
                car.rotation[1] = float(yaw_deg)

                tx = x_center + float(np.random.uniform(-x_jitter, x_jitter))
                tz = float(np.random.uniform(min(z_min, z_max), max(z_min, z_max)))

                # Same-lane longitudinal spacing constraint.
                if any(abs(tz - used_z) < safe_z for used_z in lane_z_used[lane_idx]):
                    continue

                # Bottom of local AABB is offset to exactly touch lane ground.
                car.place_on_surface(tx, tz, ground_y)

                # SAT check only with cars already in the same lane.
                mn, mx = car.world_aabb_xz(padding=sat_padding)
                overlap = False
                for pmn, pmx in lane_aabbs[lane_idx]:
                    if _sat_aabb_overlap_xz(mn, mx, pmn, pmx):
                        overlap = True
                        break
                if overlap:
                    continue

                lane_aabbs[lane_idx].append((mn, mx))
                lane_z_used[lane_idx].append(tz)
                spawned = True
                break

            if not spawned:
                # Move far away when no valid lane slot is available.
                car.position[:] = np.array([9999.0, -9999.0, 9999.0], dtype=np.float32)

    def randomize_domain(
        self,
        ground_y: float = 0.05,
        x_range: Tuple[float, float] = (-8.0, 8.0),
        z_range: Tuple[float, float] = (-35.0, 8.0),
        scale_range: Tuple[float, float] = (0.9, 1.15),
        sat_padding: float = 0.12,
        max_retry: int = 60,
    ):
        """
        Compatibility wrapper: internally uses lane-based spawning.
        """
        lanes = [
            {
                "x_center": -3.5,
                "z_min": z_range[0],
                "z_max": z_range[1],
                "direction": [0.0, 0.0, -1.0],
                "ground_y": ground_y,
                "x_jitter": 0.28,
                "safe_z": 3.2,
            },
            {
                "x_center": -1.8,
                "z_min": z_range[0],
                "z_max": z_range[1],
                "direction": [0.0, 0.0, -1.0],
                "ground_y": ground_y,
                "x_jitter": 0.25,
                "safe_z": 3.0,
            },
            {
                "x_center": 1.8,
                "z_min": z_range[0],
                "z_max": z_range[1],
                "direction": [0.0, 0.0, 1.0],
                "ground_y": ground_y,
                "x_jitter": 0.25,
                "safe_z": 3.0,
            },
            {
                "x_center": 3.5,
                "z_min": z_range[0],
                "z_max": z_range[1],
                "direction": [0.0, 0.0, 1.0],
                "ground_y": ground_y,
                "x_jitter": 0.28,
                "safe_z": 3.2,
            },
        ]
        self.spawn_cars_on_lanes(
            lanes_config=lanes,
            num_cars=len(self.get_dynamic_entities()),
            scale_range=scale_range,
            sat_padding=sat_padding,
            max_retry_per_car=max_retry,
        )


