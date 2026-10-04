# traffic.py
"""
Waypoint-based traffic routing system with Bezier turning,
object pooling, lane-following, and intersection control.
"""
import math
import random
from functools import cmp_to_key
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Intersection and road geometry constants
# ---------------------------------------------------------------------------
INTERSECTION = {
    "center_x": -1.85905,
    "center_z": -2.155,
    "x_min": -11.1,
    "x_max": 14.8,
    "z_min": -10.8,
    "z_max": 15.1,
}

ROAD_END_TOP_Z = 48.4
ROAD_END_BOTTOM_Z = -47.8
ROAD_END_LEFT_X = -47.1
ROAD_END_RIGHT_X = 46.9

# Right-hand traffic: drive on the right side of the road.
_VERT_ROAD_CX = INTERSECTION["center_x"]
_HORIZ_ROAD_CZ = INTERSECTION["center_z"]
_LANE_OFFSET_X = 4.6
_LANE_OFFSET_Z = 4.6

LANE_SOUTHBOUND_X = _VERT_ROAD_CX + _LANE_OFFSET_X
LANE_NORTHBOUND_X = _VERT_ROAD_CX - _LANE_OFFSET_X
LANE_EASTBOUND_Z = _HORIZ_ROAD_CZ + _LANE_OFFSET_Z
LANE_WESTBOUND_Z = _HORIZ_ROAD_CZ - _LANE_OFFSET_Z

GROUND_Y = 0.05

SPAWN_POINTS = [
    {
        "label": "top",
        "x": LANE_SOUTHBOUND_X,
        "z": ROAD_END_TOP_Z,
        "yaw": 180.0,
        "dir": (0.0, -1.0),
    },
    {
        "label": "bottom",
        "x": LANE_NORTHBOUND_X,
        "z": ROAD_END_BOTTOM_Z,
        "yaw": 0.0,
        "dir": (0.0, 1.0),
    },
    {
        "label": "left",
        "x": ROAD_END_LEFT_X,
        "z": LANE_EASTBOUND_Z,
        "yaw": 90.0,
        "dir": (1.0, 0.0),
    },
    {
        "label": "right",
        "x": ROAD_END_RIGHT_X,
        "z": LANE_WESTBOUND_Z,
        "yaw": -90.0,
        "dir": (-1.0, 0.0),
    },
]

EXIT_STRAIGHT = {
    "top": {"x": LANE_SOUTHBOUND_X, "z": ROAD_END_BOTTOM_Z},
    "bottom": {"x": LANE_NORTHBOUND_X, "z": ROAD_END_TOP_Z},
    "left": {"x": ROAD_END_RIGHT_X, "z": LANE_EASTBOUND_Z},
    "right": {"x": ROAD_END_LEFT_X, "z": LANE_WESTBOUND_Z},
}

# _TURN_MAP = {
#     "top": {
#         "right": {"x": ROAD_END_LEFT_X, "z": LANE_WESTBOUND_Z, "label": "left"},
#         "left": {"x": ROAD_END_RIGHT_X, "z": LANE_EASTBOUND_Z, "label": "right"},
#     },
#     "bottom": {
#         "right": {"x": ROAD_END_RIGHT_X, "z": LANE_EASTBOUND_Z, "label": "right"},
#         "left": {"x": ROAD_END_LEFT_X, "z": LANE_WESTBOUND_Z, "label": "left"},
#     },
#     "left": {
#         "right": {"x": LANE_SOUTHBOUND_X, "z": ROAD_END_BOTTOM_Z, "label": "bottom"},
#         "left": {"x": LANE_NORTHBOUND_X, "z": ROAD_END_TOP_Z, "label": "top"},
#     },
#     "right": {
#         "right": {"x": LANE_NORTHBOUND_X, "z": ROAD_END_TOP_Z, "label": "top"},
#         "left": {"x": LANE_SOUTHBOUND_X, "z": ROAD_END_BOTTOM_Z, "label": "bottom"},
#     },
# }

_TURN_MAP = {
    "top": {
        "right": {"x": ROAD_END_RIGHT_X, "z": LANE_EASTBOUND_Z, "label": "right"},
        "left": {"x": ROAD_END_LEFT_X, "z": LANE_WESTBOUND_Z, "label": "left"},
    },
    "bottom": {
        "right": {"x": ROAD_END_LEFT_X, "z": LANE_WESTBOUND_Z, "label": "left"},
        "left": {"x": ROAD_END_RIGHT_X, "z": LANE_EASTBOUND_Z, "label": "right"},
    },
    "left": {
        "right": {"x": LANE_NORTHBOUND_X, "z": ROAD_END_TOP_Z, "label": "top"},
        "left": {"x": LANE_SOUTHBOUND_X, "z": ROAD_END_BOTTOM_Z, "label": "bottom"},
    },
    "right": {
        "right": {"x": LANE_SOUTHBOUND_X, "z": ROAD_END_BOTTOM_Z, "label": "bottom"},
        "left": {"x": LANE_NORTHBOUND_X, "z": ROAD_END_TOP_Z, "label": "top"},
    },
}


# ---------------------------------------------------------------------------
# Bezier helpers
# ---------------------------------------------------------------------------
def _bezier_quadratic(
    p0: Tuple[float, float],
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    num_points: int = 12,
) -> List[Tuple[float, float]]:
    pts = []
    for i in range(num_points + 1):
        t = i / num_points
        u = 1.0 - t
        x = u * u * p0[0] + 2.0 * u * t * p1[0] + t * t * p2[0]
        z = u * u * p0[1] + 2.0 * u * t * p1[1] + t * t * p2[1]
        pts.append((x, z))
    return pts


def _bezier_cubic(
    p0: Tuple[float, float],
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    p3: Tuple[float, float],
    num_points: int = 16,
) -> List[Tuple[float, float]]:
    pts = []
    for i in range(num_points + 1):
        t = i / num_points
        u = 1.0 - t
        x = (
            u ** 3 * p0[0]
            + 3 * u ** 2 * t * p1[0]
            + 3 * u * t ** 2 * p2[0]
            + t ** 3 * p3[0]
        )
        z = (
            u ** 3 * p0[1]
            + 3 * u ** 2 * t * p1[1]
            + 3 * u * t ** 2 * p2[1]
            + t ** 3 * p3[1]
        )
        pts.append((x, z))
    return pts


# ---------------------------------------------------------------------------
# Intersection helpers
# ---------------------------------------------------------------------------
def is_inside_intersection(x: float, z: float) -> bool:
    return (
        INTERSECTION["x_min"] <= x <= INTERSECTION["x_max"]
        and INTERSECTION["z_min"] <= z <= INTERSECTION["z_max"]
    )


def _approach_margin() -> float:
    return 12.0


def is_approaching_intersection(x: float, z: float, dx: float, dz: float) -> bool:
    if is_inside_intersection(x, z):
        return False
    margin = _approach_margin()
    fx, fz = x + dx * margin, z + dz * margin
    return is_inside_intersection(fx, fz)


# ---------------------------------------------------------------------------
# Route generation
# ---------------------------------------------------------------------------
def _intersection_entry_point(
    spawn_label: str,
    spawn_x: float,
    spawn_z: float,
) -> Tuple[float, float]:
    if spawn_label == "top":
        return (spawn_x, INTERSECTION["z_max"])
    if spawn_label == "bottom":
        return (spawn_x, INTERSECTION["z_min"])
    if spawn_label == "left":
        return (INTERSECTION["x_min"], spawn_z)
    if spawn_label == "right":
        return (INTERSECTION["x_max"], spawn_z)
    return (spawn_x, spawn_z)


def _intersection_exit_point(
    exit_label: str,
    exit_x: float,
    exit_z: float,
) -> Tuple[float, float]:
    if exit_label == "bottom":
        return (exit_x, INTERSECTION["z_min"])
    if exit_label == "top":
        return (exit_x, INTERSECTION["z_max"])
    if exit_label == "right":
        return (INTERSECTION["x_max"], exit_z)
    if exit_label == "left":
        return (INTERSECTION["x_min"], exit_z)
    return (exit_x, exit_z)


def generate_route(spawn_idx: int, maneuver: str = "straight") -> List[Tuple[float, float]]:
    """
    Generate a list of (x, z) waypoints for a car spawning at `spawn_idx`.
    """
    sp = SPAWN_POINTS[spawn_idx]
    label = sp["label"]
    sx, sz = sp["x"], sp["z"]

    waypoints: List[Tuple[float, float]] = []

    entry = _intersection_entry_point(label, sx, sz)
    num_approach = 3
    for i in range(1, num_approach + 1):
        t = i / (num_approach + 1)
        wx = sx + t * (entry[0] - sx)
        wz = sz + t * (entry[1] - sz)
        waypoints.append((wx, wz))
    waypoints.append(entry)

    if maneuver == "straight":
        exit_cfg = EXIT_STRAIGHT[label]
        ex, ez = exit_cfg["x"], exit_cfg["z"]
        exit_labels = {
            "top": "bottom",
            "bottom": "top",
            "left": "right",
            "right": "left",
        }
        exit_lbl = exit_labels[label]
        exit_pt = _intersection_exit_point(exit_lbl, ex, ez)

        if label in ("top", "bottom"):
            cross_zs = sorted(
                [
                    z
                    for z in (LANE_WESTBOUND_Z, LANE_EASTBOUND_Z)
                    if INTERSECTION["z_min"] < z < INTERSECTION["z_max"]
                ],
                reverse=(label == "top"),
            )
            for cz in cross_zs:
                waypoints.append((entry[0], cz))
        else:
            cross_xs = sorted(
                [
                    x
                    for x in (LANE_NORTHBOUND_X, LANE_SOUTHBOUND_X)
                    if INTERSECTION["x_min"] < x < INTERSECTION["x_max"]
                ],
                reverse=(label == "right"),
            )
            for cx in cross_xs:
                waypoints.append((cx, entry[1]))

        waypoints.append(exit_pt)

        num_exit = 3
        for i in range(1, num_exit + 1):
            t = i / (num_exit + 1)
            wx = exit_pt[0] + t * (ex - exit_pt[0])
            wz = exit_pt[1] + t * (ez - exit_pt[1])
            waypoints.append((wx, wz))
        waypoints.append((ex, ez))

    elif maneuver in ("right", "left"):
        turn_cfg = _TURN_MAP[label][maneuver]
        exit_road_x, exit_road_z = turn_cfg["x"], turn_cfg["z"]
        exit_lbl = turn_cfg["label"]
        exit_pt = _intersection_exit_point(exit_lbl, exit_road_x, exit_road_z)

        if label in ("top", "bottom"):
            corner = (entry[0], exit_pt[1])
        else:
            corner = (exit_pt[0], entry[1])

        if maneuver == "right":
            curve_pts = _bezier_quadratic(entry, corner, exit_pt, num_points=12)
        else:
            cx, cz = INTERSECTION["center_x"], INTERSECTION["center_z"]
            ctrl1 = (
                entry[0] + 0.35 * (corner[0] - entry[0]) + 0.1 * (cx - entry[0]),
                entry[1] + 0.35 * (corner[1] - entry[1]) + 0.1 * (cz - entry[1]),
            )
            ctrl2 = (
                exit_pt[0] + 0.35 * (corner[0] - exit_pt[0]) + 0.1 * (cx - exit_pt[0]),
                exit_pt[1] + 0.35 * (corner[1] - exit_pt[1]) + 0.1 * (cz - exit_pt[1]),
            )
            curve_pts = _bezier_cubic(entry, ctrl1, ctrl2, exit_pt, num_points=14)

        waypoints.extend(curve_pts[1:])

        num_exit = 3
        for i in range(1, num_exit + 1):
            t = i / (num_exit + 1)
            wx = exit_pt[0] + t * (exit_road_x - exit_pt[0])
            wz = exit_pt[1] + t * (exit_road_z - exit_pt[1])
            waypoints.append((wx, wz))
        waypoints.append((exit_road_x, exit_road_z))

    return waypoints


def random_maneuver(is_init: bool = False) -> str:
    r = random.random()
    # if is_init:
    #     # Lần đầu spawn: Ép 100% không có rẽ trái (75% đi thẳng, 25% rẽ phải)
    #     return "straight" if r < 0.6 else "right"
    
    # Các lần spawn sau: 75% đi thẳng, 20% rẽ phải, 5% rẽ trái
    if r < 0.4:
        return "straight"
    elif r < 0.8:
        return "right"
    else:
        return "left"


# ---------------------------------------------------------------------------
# Traffic manager
# ---------------------------------------------------------------------------
_BRAKE_DECEL = 12.0
_ACCEL = 8.0
_MIN_SPEED = 0.0

_FOLLOW_REACTION_TIME = 0.9
_FOLLOW_MIN_GAP = 2.5
_FOLLOW_GAIN = 1.35
_LANE_MATCH_TOL = 2.25
_LANE_EXTRA_WIDTH = 1.0

_STAGGER_DISTANCE = 3.0
_SPAWN_CLEAR_RADIUS = 8.0
_SPAWN_LANE_CLEARANCE = 18.0

_INTERSECTION_APPROACH_MARGIN = 8.0
_STOP_LINE_BUFFER = 0.5
_STOP_LINE_QUEUE_LATERAL_TOL = 3.25

# A tiny unsignalized conflict-zone model.
# TL/TR/BL/BR are the four quadrants of the intersection.
ZONE_TL = "TL"
ZONE_TR = "TR"
ZONE_BL = "BL"
ZONE_BR = "BR"
INTERSECTION_ZONES = (ZONE_TL, ZONE_TR, ZONE_BL, ZONE_BR)
INTERSECTION_ZONE_BOUNDS = {
    ZONE_TL: (
        INTERSECTION["x_min"],
        INTERSECTION["center_x"],
        INTERSECTION["center_z"],
        INTERSECTION["z_max"],
    ),
    ZONE_TR: (
        INTERSECTION["center_x"],
        INTERSECTION["x_max"],
        INTERSECTION["center_z"],
        INTERSECTION["z_max"],
    ),
    ZONE_BL: (
        INTERSECTION["x_min"],
        INTERSECTION["center_x"],
        INTERSECTION["z_min"],
        INTERSECTION["center_z"],
    ),
    ZONE_BR: (
        INTERSECTION["center_x"],
        INTERSECTION["x_max"],
        INTERSECTION["z_min"],
        INTERSECTION["center_z"],
    ),
}

# _ROUTE_ZONE_MAP = {
#     "top": {
#         "right": [ZONE_TL],
#         "straight": [ZONE_TL, ZONE_BL],
#         "left": [ZONE_TL, ZONE_TR, ZONE_BR],
#     },
#     "bottom": {
#         "right": [ZONE_BR],
#         "straight": [ZONE_BR, ZONE_TR],
#         "left": [ZONE_BR, ZONE_BL, ZONE_TL],
#     },
#     "left": {
#         "right": [ZONE_BL],
#         "straight": [ZONE_BL, ZONE_BR],
#         "left": [ZONE_BL, ZONE_TL, ZONE_TR],
#     },
#     "right": {
#         "right": [ZONE_TR],
#         "straight": [ZONE_TR, ZONE_TL],
#         "left": [ZONE_TR, ZONE_BR, ZONE_BL],
#     },
# }

_ROUTE_ZONE_MAP = {
    "top": {
        "right": [ZONE_TR],
        "straight": [ZONE_TR, ZONE_BR],
        "left": [ZONE_TR, ZONE_TL, ZONE_BL],
    },
    "bottom": {
        "right": [ZONE_BL],
        "straight": [ZONE_BL, ZONE_TL],
        "left": [ZONE_BL, ZONE_BR, ZONE_TR],
    },
    "left": {
        "right": [ZONE_TL],
        "straight": [ZONE_TL, ZONE_TR],
        "left": [ZONE_TL, ZONE_BL, ZONE_BR],
    },
    "right": {
        "right": [ZONE_BR],
        "straight": [ZONE_BR, ZONE_BL],
        "left": [ZONE_BR, ZONE_TR, ZONE_TL],
    },
}

_MANEUVER_PRIORITY = {"straight": 1, "right": 0, "left": 2}
_ARRIVAL_EPS = 1e-3
_STOP_LINE_WAIT_DISTANCE = 2.0
_SPAWN_LANE_IDS = {
    "top": "southbound",
    "bottom": "northbound",
    "left": "eastbound",
    "right": "westbound",
}


def required_zones_for_route(spawn_label: str, maneuver: str) -> List[str]:
    """Return the ordered conflict zones used by a route through the intersection."""
    return list(_ROUTE_ZONE_MAP.get(spawn_label, {}).get(maneuver, ()))


class TrafficManager:
    """
    Manages the car pool: assigns routes, handles respawning,
    and runs traffic control each frame.
    """

    def __init__(self):
        self.cars: list = []
        self._car_lookup: dict = {}
        self._spawn_timers: dict = {}
        self._pending_respawns: dict = {}
        self._car_entry_labels: dict = {}
        self._car_maneuvers: dict = {}
        self._zone_owners: dict = {"TL": None, "TR": None, "BL": None, "BR": None}
        self._car_reserved_zones: dict = {}
        self._car_entered_zones: dict = {}
        self._car_entered_intersection: set = set()
        self._arrival_times: dict = {}
        self._intersection_speed_caps: dict = {}
        self._sim_time: float = 0.0

    def register_cars(self, cars: list):
        """Register the car pool and create a clean traffic state."""
        self.cars = list(cars)
        self._car_lookup = {car.name: car for car in self.cars}
        self._spawn_timers.clear()
        self._pending_respawns.clear()
        self._car_entry_labels.clear()
        self._car_maneuvers.clear()
        self._zone_owners = {"TL": None, "TR": None, "BL": None, "BR": None}
        self._car_reserved_zones.clear()
        self._car_entered_zones.clear()
        self._car_entered_intersection.clear()
        self._arrival_times.clear()
        self._intersection_speed_caps.clear()
        self._sim_time = 0.0

        n_spawns = len(SPAWN_POINTS)
        for idx, car in enumerate(self.cars):
            spawn_idx = idx % n_spawns
            self._spawn_car(car, spawn_idx=spawn_idx, stagger_offset=idx, is_init=True)

    def _spawn_car(self, car, spawn_idx: int = -1, stagger_offset: int = 0, is_init: bool = False):
        """
        Teleport a car to a spawn point and assign a new route.
        If no spawn point is clear, leave it in the pending respawn queue.
        """
        self._release_all_zones_for_car(car)

        if spawn_idx < 0:
            spawn_idx = self._pick_clear_spawn()
        if spawn_idx is None:
            self._queue_for_respawn(car)
            return

        sp = SPAWN_POINTS[spawn_idx]
        maneuver = random_maneuver(is_init=is_init)
        waypoints = generate_route(spawn_idx, maneuver)
        target_speed = random.uniform(3.0, 5.0)

        dir_x, dir_z = sp["dir"]
        offset = _STAGGER_DISTANCE * stagger_offset + random.uniform(0.0, 4.0)
        spawn_x = sp["x"] - dir_x * offset
        spawn_z = sp["z"] - dir_z * offset

        car.reset_for_pool(
            x=spawn_x,
            z=spawn_z,
            yaw=sp["yaw"],
            waypoints=waypoints,
            target_speed=target_speed,
        )
        self._pending_respawns.pop(car.name, None)
        self._spawn_timers[car.name] = 0.0
        self._arrival_times.pop(car.name, None)
        self._intersection_speed_caps.pop(car.name, None)
        self._car_entry_labels[car.name] = sp["label"]
        self._car_maneuvers[car.name] = maneuver

    def _queue_for_respawn(self, car, preferred_spawn_idx: Optional[int] = None):
        self._release_all_zones_for_car(car)
        car.is_active = False
        car.route_finished = False
        car.speed = 0.0
        self._pending_respawns[car.name] = preferred_spawn_idx
        self._arrival_times.pop(car.name, None)
        self._intersection_speed_caps.pop(car.name, None)

    def _pick_clear_spawn(self) -> Optional[int]:
        indices = list(range(len(SPAWN_POINTS)))
        random.shuffle(indices)
        for idx in indices:
            if self._is_spawn_point_clear(idx):
                return idx
        return None

    def _is_spawn_point_clear(self, spawn_idx: int) -> bool:
        sp = SPAWN_POINTS[spawn_idx]
        sx, sz = sp["x"], sp["z"]
        dir_x, dir_z = sp["dir"]
        lane_id = _SPAWN_LANE_IDS[sp["label"]]

        for car in self.cars:
            if not car.is_active or car.route_finished:
                continue

            cx = float(car.position[0])
            cz = float(car.position[2])
            if math.hypot(cx - sx, cz - sz) < _SPAWN_CLEAR_RADIUS:
                return False

            if self._lane_id_for_car(car) != lane_id:
                continue

            dx = cx - sx
            dz = cz - sz
            longitudinal = dx * dir_x + dz * dir_z
            lateral = abs(dx * (-dir_z) + dz * dir_x)
            if -2.0 <= longitudinal <= _SPAWN_LANE_CLEARANCE and lateral <= _LANE_MATCH_TOL:
                return False
        return True

    def _process_pending_respawns(self):
        for car_name, preferred_idx in list(self._pending_respawns.items()):
            car = self._car_lookup.get(car_name)
            if car is None:
                self._pending_respawns.pop(car_name, None)
                continue

            spawn_idx = preferred_idx
            if spawn_idx is not None and not self._is_spawn_point_clear(spawn_idx):
                spawn_idx = None
            if spawn_idx is None:
                spawn_idx = self._pick_clear_spawn()
            if spawn_idx is None:
                continue
            self._spawn_car(car, spawn_idx=spawn_idx, stagger_offset=0)

    @staticmethod
    def _apply_speed_toward(car, desired_speed: float, dt: float):
        desired_speed = float(max(_MIN_SPEED, desired_speed))
        if car.speed > desired_speed:
            car.speed = max(desired_speed, car.speed - _BRAKE_DECEL * dt)
        else:
            car.speed = min(desired_speed, car.speed + _ACCEL * dt)

    @staticmethod
    def _car_pos_xz(car) -> Tuple[float, float]:
        return float(car.position[0]), float(car.position[2])

    @staticmethod
    def _distance_to_stop_line_for_label(label: str, x: float, z: float) -> float:
        if label == "top":
            return max(0.0, z - INTERSECTION["z_max"])
        if label == "bottom":
            return max(0.0, INTERSECTION["z_min"] - z)
        if label == "left":
            return max(0.0, INTERSECTION["x_min"] - x)
        if label == "right":
            return max(0.0, x - INTERSECTION["x_max"])
        return float("inf")

    def _distance_to_stop_line(self, car) -> float:
        x, z = self._car_pos_xz(car)
        dist_center = self._distance_to_stop_line_for_label(
            self._car_entry_labels.get(car.name, ""),
            x,
            z,
        )
        
        # --- MỚI: Trừ đi nửa chiều dài xe để tính từ mũi xe thay vì tâm xe ---
        lane_id = self._lane_id_for_car(car)
        if lane_id:
            half_len, _ = self._car_half_extent_for_lane(car, lane_id)
        else:
            half_len = 2.4 # Độ dài mặc định an toàn
            
        return max(0.0, dist_center - half_len)

    def _lane_id_for_car(self, car) -> Optional[str]:
        x, z = self._car_pos_xz(car)
        fwd = car._movement_forward_world()
        fx = float(fwd[0])
        fz = float(fwd[2])

        if abs(fz) >= abs(fx):
            if fz > 0.4 and abs(x - LANE_NORTHBOUND_X) <= _LANE_MATCH_TOL:
                return "northbound"
            if fz < -0.4 and abs(x - LANE_SOUTHBOUND_X) <= _LANE_MATCH_TOL:
                return "southbound"
        else:
            if fx > 0.4 and abs(z - LANE_EASTBOUND_Z) <= _LANE_MATCH_TOL:
                return "eastbound"
            if fx < -0.4 and abs(z - LANE_WESTBOUND_Z) <= _LANE_MATCH_TOL:
                return "westbound"
        return None

    @staticmethod
    def _car_half_extent_for_lane(car, lane_id: str) -> Tuple[float, float]:
        vertical = lane_id in ("northbound", "southbound")
        half_length = 0.5 * float(
            (car.local_size[2] if vertical else car.local_size[0])
            * (car.scale[2] if vertical else car.scale[0])
        )
        half_width = 0.5 * float(
            (car.local_size[0] if vertical else car.local_size[2])
            * (car.scale[0] if vertical else car.scale[2])
        )
        return half_length, half_width

    def _find_lead_car(self, car_a, active_cars: list):
        queue_lead, queue_gap = self._find_stopline_queue_lead(car_a, active_cars)
        if queue_lead is not None:
            return queue_lead, queue_gap

        lane_id = self._lane_id_for_car(car_a)
        if lane_id is None:
            return None, None

        ax, az = self._car_pos_xz(car_a)
        if is_inside_intersection(ax, az):
            return None, None

        fwd_a = car_a._movement_forward_world()
        fx = float(fwd_a[0])
        fz = float(fwd_a[2])
        perp_x = -fz
        perp_z = fx
        half_len_a, half_width_a = self._car_half_extent_for_lane(car_a, lane_id)

        best_car = None
        best_gap = float("inf")
        for car_b in active_cars:
            if car_b is car_a or self._lane_id_for_car(car_b) != lane_id:
                continue

            bx, bz = self._car_pos_xz(car_b)
            dx = bx - ax
            dz = bz - az
            longitudinal = dx * fx + dz * fz
            if longitudinal <= 0.0:
                continue

            _, half_width_b = self._car_half_extent_for_lane(car_b, lane_id)
            lateral = abs(dx * perp_x + dz * perp_z)
            if lateral > half_width_a + half_width_b + _LANE_EXTRA_WIDTH:
                continue

            half_len_b, _ = self._car_half_extent_for_lane(car_b, lane_id)
            gap = longitudinal - half_len_a - half_len_b
            if gap < best_gap:
                best_gap = gap
                best_car = car_b

        if best_car is None:
            return None, None
        return best_car, max(0.0, best_gap)
    
    def _has_passed_stop_line(self, label: str, x: float, z: float) -> bool:
        """Kiểm tra xem xe đã vượt qua vạch dừng của hướng đi tương ứng chưa."""
        if label == "top":
            return z < INTERSECTION["z_max"]
        if label == "bottom":
            return z > INTERSECTION["z_min"]
        if label == "left":
            return x > INTERSECTION["x_min"]
        if label == "right":
            return x < INTERSECTION["x_max"]
        return False

    def _find_stopline_queue_lead(self, car_a, active_cars: list):
        """
        Robustly detect the lead car while approaching the same stop line.
        This avoids queue collisions when lane matching jitters near intersection.
        """
        label = self._car_entry_labels.get(car_a.name)
        if not label:
            return None, None

        ax, az = self._car_pos_xz(car_a)
        if is_inside_intersection(ax, az):
            return None, None

        dist_a = self._distance_to_stop_line_for_label(label, ax, az)
        if dist_a > _INTERSECTION_APPROACH_MARGIN:
            return None, None

        lane_id = _SPAWN_LANE_IDS.get(label)
        if lane_id is None:
            return None, None
        half_len_a, _ = self._car_half_extent_for_lane(car_a, lane_id)

        if label in ("top", "bottom"):
            lateral_a = ax
        else:
            lateral_a = az

        best_car = None
        best_gap = float("inf")

        for car_b in active_cars:
            if car_b is car_a:
                continue
            if self._car_entry_labels.get(car_b.name) != label:
                continue

            bx, bz = self._car_pos_xz(car_b)
            if is_inside_intersection(bx, bz):
                continue

            if self._has_passed_stop_line(label, bx, bz):
                continue

            dist_b = self._distance_to_stop_line_for_label(label, bx, bz)
            if dist_b >= dist_a:
                continue

            lateral_b = bx if label in ("top", "bottom") else bz
            if abs(lateral_b - lateral_a) > _STOP_LINE_QUEUE_LATERAL_TOL:
                continue

            half_len_b, _ = self._car_half_extent_for_lane(car_b, lane_id)
            gap = (dist_a - dist_b) - half_len_a - half_len_b
            if gap < best_gap:
                best_gap = gap
                best_car = car_b

        if best_car is None:
            return None, None
        return best_car, max(0.0, best_gap)

    @staticmethod
    def _following_safe_gap(speed: float, lead_speed: float) -> float:
        closing_speed = max(0.0, speed - lead_speed)
        braking_margin = (closing_speed * closing_speed) / max(1e-3, 2.0 * _BRAKE_DECEL)
        return _FOLLOW_MIN_GAP + _FOLLOW_REACTION_TIME * speed + braking_margin

    def _car_following_speed_cap(self, car, active_cars: list) -> Optional[float]:
        lead_car, gap = self._find_lead_car(car, active_cars)
        if lead_car is None or gap is None:
            return None

        lead_speed = float(max(0.0, lead_car.speed))
        safe_gap = self._following_safe_gap(float(car.speed), lead_speed)
        speed_cap = lead_speed + (gap - safe_gap) * _FOLLOW_GAIN
        return max(_MIN_SPEED, min(car.target_speed, speed_cap))

    def _is_waiting_for_intersection(self, car) -> bool:
        """A car joins the queue only when it is physically at the stop line."""
        x, z = self._car_pos_xz(car)
        if is_inside_intersection(x, z):
            return False

        label = self._car_entry_labels.get(car.name)
        if not label:
            return False

        fwd = car._movement_forward_world()
        approaching = is_approaching_intersection(x, z, float(fwd[0]), float(fwd[2]))
        return approaching and self._distance_to_stop_line(car) < _STOP_LINE_WAIT_DISTANCE

    def _required_zones_for_car(self, car) -> List[str]:
        label = self._car_entry_labels.get(car.name, "")
        maneuver = self._car_maneuvers.get(car.name, "straight")
        return required_zones_for_route(label, maneuver)

    @staticmethod
    def _aabb_corner_inside_zone(mn, mx, zone: str) -> bool:
        """A car occupies a zone if any one of its 4 XZ box corners is inside it."""
        x_min, x_max, z_min, z_max = INTERSECTION_ZONE_BOUNDS[zone]
        corners = (
            (float(mn[0]), float(mn[1])),
            (float(mn[0]), float(mx[1])),
            (float(mx[0]), float(mn[1])),
            (float(mx[0]), float(mx[1])),
        )
        return any(x_min <= x <= x_max and z_min <= z <= z_max for x, z in corners)

    def _car_overlaps_zone(self, car, zone: str) -> bool:
        mn, mx = car.world_aabb_xz()
        return self._aabb_corner_inside_zone(mn, mx, zone)

    def _release_all_zones_for_car(self, car):
        car_name = car.name
        for zone, owner in list(self._zone_owners.items()):
            if owner == car_name:
                self._zone_owners[zone] = None
        self._car_reserved_zones.pop(car_name, None)
        self._car_entered_zones.pop(car_name, None)
        self._car_entered_intersection.discard(car_name)
        self._intersection_speed_caps.pop(car_name, None)

    def _reserve_zones_for_car(self, car, zones: List[str]):
        """Claim every zone on this car's predicted path."""
        car_name = car.name
        self._car_reserved_zones[car_name] = list(zones)
        self._car_entered_zones.setdefault(car_name, set())
        for zone in zones:
            self._zone_owners[zone] = car_name

    def _refresh_zone_reservations(self, active_cars: list):
        """Release zones as soon as their owning car fully leaves them or bypasses them."""
        active_names = {car.name for car in active_cars}

        # Drop reservations for cars that were removed or sent back to the pool.
        for zone, owner in list(self._zone_owners.items()):
            if owner is not None and owner not in active_names:
                self._zone_owners[zone] = None

        for car_name in list(self._car_reserved_zones.keys()):
            if car_name not in active_names:
                car = self._car_lookup.get(car_name)
                if car is not None:
                    self._release_all_zones_for_car(car)
                else:
                    self._car_reserved_zones.pop(car_name, None)
                    self._car_entered_zones.pop(car_name, None)
                    self._car_entered_intersection.discard(car_name)

        for car in active_cars:
            reserved = self._car_reserved_zones.get(car.name)
            x, z = self._car_pos_xz(car)
            overlaps_any_zone = any(self._car_overlaps_zone(car, zone) for zone in INTERSECTION_ZONES)
            
            if is_inside_intersection(x, z) or overlaps_any_zone:
                self._car_entered_intersection.add(car.name)

            if not reserved:
                if is_inside_intersection(x, z):
                    for zone in INTERSECTION_ZONES:
                        if self._zone_owners.get(zone) is None and self._car_overlaps_zone(car, zone):
                            self._zone_owners[zone] = car.name
                            self._car_reserved_zones.setdefault(car.name, []).append(zone)
                            self._car_entered_zones.setdefault(car.name, set()).add(zone)
                continue

            entered = self._car_entered_zones.setdefault(car.name, set())
            
            max_overlapped_idx = -1
            for i, zone in enumerate(reserved):
                if self._car_overlaps_zone(car, zone):
                    max_overlapped_idx = i

            # Lặp qua các zone đã giữ để xét giải phóng
            for i, zone in enumerate(list(reserved)):
                if self._zone_owners.get(zone) != car.name:
                    continue

                is_overlapping = self._car_overlaps_zone(car, zone)

                # --- FIX: Đẩy logic giải phóng sớm lên đầu ---
                if max_overlapped_idx != -1 and i < max_overlapped_idx:
                    # Đã lọt vào zone tiếp theo -> lập tức nhả zone hiện tại (dù đuôi xe có thể vẫn đang đè vạch)
                    self._zone_owners[zone] = None
                    self._car_reserved_zones[car.name].remove(zone)
                elif is_overlapping:
                    # Ghi nhận zone xe đang đè lên
                    entered.add(zone)
                elif zone in entered:
                    # Đã hoàn toàn rời đi -> nhả đuôi xe
                    self._zone_owners[zone] = None
                    self._car_reserved_zones[car.name].remove(zone)

            # Phân giải phóng toàn bộ khi ra khỏi ngã tư
            if (
                car.name in self._car_entered_intersection
                and not is_inside_intersection(x, z)
                and not overlaps_any_zone
            ):
                self._release_all_zones_for_car(car)
                continue

            if not self._car_reserved_zones.get(car.name):
                self._car_reserved_zones.pop(car.name, None)
                self._car_entered_zones.pop(car.name, None)

    def _routes_can_share_intersection(self, car_a, car_b) -> bool:
        """
        Allow known non-conflicting movement pairs to proceed together.
        This reduces deadlock caused by coarse conflict-zone reservations.
        """
        entry_a = self._car_entry_labels.get(car_a.name, "")
        move_a = self._car_maneuvers.get(car_a.name, "straight")
        entry_b = self._car_entry_labels.get(car_b.name, "")
        move_b = self._car_maneuvers.get(car_b.name, "straight")

        if entry_a == entry_b:
            return True

        non_conflicting_pairs = {
            # Các route cũ không đụng nhau
            (("top", "right"), ("left", "left")),
            (("left", "left"), ("top", "right")),
            
            # --- CÁC ROUTE MỚI THÊM VÀO ---
            # Cho phép xe rẽ phải đi cùng lúc với xe đi thẳng từ bên trái tới
            (("bottom", "right"), ("left", "straight")),
            (("left", "straight"), ("bottom", "right")),

            (("right", "right"), ("bottom", "straight")),
            (("bottom", "straight"), ("right", "right")),

            (("top", "right"), ("right", "straight")),
            (("right", "straight"), ("top", "right")),

            (("left", "right"), ("top", "straight")),
            (("top", "straight"), ("left", "right")),
        }
        return ((entry_a, move_a), (entry_b, move_b)) in non_conflicting_pairs

    def _zones_available_for_car(self, car, zones: List[str]) -> bool:
        """
        A zone is available when it is empty, owned by this car, or owned by
        another car with a known non-conflicting maneuver pair.
        """
        for zone in zones:
            owner_name = self._zone_owners.get(zone)
            if owner_name is None or owner_name == car.name:
                continue

            owner_car = self._car_lookup.get(owner_name)
            if owner_car is None or not owner_car.is_active or owner_car.route_finished:
                continue

            if self._routes_can_share_intersection(car, owner_car):
                continue
            return False
        return True

    def _zone_has_physical_car(self, zone: str, exclude_car=None) -> bool:
        for other in self.cars:
            if other is exclude_car or not other.is_active or other.route_finished:
                continue
            if self._car_overlaps_zone(other, zone):
                return True
        return False

    def _release_empty_reserved_zones(self, zones: List[str]):
        """
        Clear reservations whose owner has not physically entered that zone yet.
        This prevents right turns from being blocked by a stale/over-conservative
        zone claim while still preserving owners already inside the intersection.
        """
        for zone in zones:
            owner_name = self._zone_owners.get(zone)
            if owner_name is None:
                continue

            owner = self._car_lookup.get(owner_name)
            if owner is None or not owner.is_active or owner.route_finished:
                self._zone_owners[zone] = None
                continue

            ox, oz = self._car_pos_xz(owner)
            owner_in_intersection = is_inside_intersection(ox, oz)
            owner_in_any_zone = any(self._car_overlaps_zone(owner, z) for z in INTERSECTION_ZONES)
            if not owner_in_intersection and not owner_in_any_zone:
                self._release_all_zones_for_car(owner)

    def _right_turn_can_bypass_empty_reservation(self, car, zones: List[str]) -> bool:
        """Right turns may proceed only when their required zones are physically empty."""
        if self._car_maneuvers.get(car.name) != "right":
            return False
        # Chỉ kiểm tra các zone mà xe này yêu cầu, thay vì kiểm tra toàn bộ ngã tư
        return all(
            not self._zone_has_physical_car(zone, exclude_car=car)
            for zone in zones 
        )

    def _compare_waiting_cars(self, car_a, car_b) -> int:
        """Sort by arrival time, using maneuver priority only for near-ties."""
        arrival_a = self._arrival_times.get(car_a.name, self._sim_time)
        arrival_b = self._arrival_times.get(car_b.name, self._sim_time)

        if abs(arrival_a - arrival_b) > _ARRIVAL_EPS:
            return -1 if arrival_a < arrival_b else 1

        maneuver_a = self._car_maneuvers.get(car_a.name, "straight")
        maneuver_b = self._car_maneuvers.get(car_b.name, "straight")
        priority_a = _MANEUVER_PRIORITY.get(maneuver_a, 99)
        priority_b = _MANEUVER_PRIORITY.get(maneuver_b, 99)
        if priority_a != priority_b:
            return -1 if priority_a < priority_b else 1

        return -1 if car_a.name < car_b.name else (1 if car_a.name > car_b.name else 0)

    def _update_intersection_queue(self, active_cars: list):
        active_names = {car.name for car in active_cars}
        self._intersection_speed_caps.clear()

        for car_name in list(self._arrival_times.keys()):
            if car_name not in active_names:
                self._arrival_times.pop(car_name, None)

        waiting_cars = []
        for car in active_cars:
            if self._car_reserved_zones.get(car.name):
                self._arrival_times.pop(car.name, None)
                continue

            if self._is_waiting_for_intersection(car):
                self._arrival_times.setdefault(car.name, self._sim_time)
                waiting_cars.append(car)
            else:
                self._arrival_times.pop(car.name, None)

        waiting_cars.sort(key=cmp_to_key(self._compare_waiting_cars))
        waiting_names = {car.name for car in waiting_cars}

        for car in waiting_cars:
            zones = self._required_zones_for_car(car)
            
            if zones and self._zones_available_for_car(car, zones):
                self._reserve_zones_for_car(car, zones)
                self._arrival_times.pop(car.name, None)
            elif zones and self._right_turn_can_bypass_empty_reservation(car, zones):
                self._release_empty_reserved_zones(zones)
                if self._zones_available_for_car(car, zones):
                    self._reserve_zones_for_car(car, zones)
                    self._arrival_times.pop(car.name, None)
                else:
                    self._intersection_speed_caps[car.name] = self._stop_line_speed_cap(car)
            else:
                self._intersection_speed_caps[car.name] = self._stop_line_speed_cap(car)

    def _stop_line_speed_cap(self, car) -> float:
        stop_dist = self._distance_to_stop_line(car)
        brake_distance = max(0.0, stop_dist - _STOP_LINE_BUFFER)
        if brake_distance <= 0.0:
            return 0.0
        return math.sqrt(max(0.0, 2.0 * _BRAKE_DECEL * brake_distance))

    def _intersection_speed_cap(self, car) -> Optional[float]:
        """Return this frame's intersection cap, or None when the car may move."""
        return self._intersection_speed_caps.get(car.name)

    def _radar_speed_cap(self, car, active_cars: list) -> Optional[float]:
        lookahead = max(2.5, float(car.speed) * 1.0)
        f_mn, f_mx = car.future_aabb_xz(lookahead)
        
        cx, cz = self._car_pos_xz(car)
        fwd = car._movement_forward_world()
        fx, fz = float(fwd[0]), float(fwd[2])

        min_speed_cap = float("inf")
        
        for other in active_cars:
            if other is car:
                continue
                
            ox, oz = self._car_pos_xz(other)
            dx, dz = ox - cx, oz - cz
            dist_sq = dx*dx + dz*dz
            if dist_sq > 225.0:
                continue
                
            o_mn, o_mx = other.world_aabb_xz()
            
            margin = 0.3 
            overlap = (f_mn[0] <= o_mx[0] + margin and f_mx[0] >= o_mn[0] - margin and
                       f_mn[1] <= o_mx[1] + margin and f_mx[1] >= o_mn[1] - margin)
            
            if overlap:
                dist = math.sqrt(dist_sq)
                if dist < 1e-3: continue
                
                # --- MỚI: Kiểm tra xe kia đang đi hướng nào ---
                o_fwd = other._movement_forward_world()
                o_fx, o_fz = float(o_fwd[0]), float(o_fwd[2])
                
                # fwd_dot > 0.5 nghĩa là 2 xe đang đi cùng hướng (ví dụ: đang merge vào cùng 1 làn)
                fwd_dot = fx * o_fx + fz * o_fz
                
                # dot > 0.4 nghĩa là xe kia đang chắn ở phía trước mặt
                dot = (dx * fx + dz * fz) / dist
                
                if dot > 0.4 and fwd_dot > 0.5:
                    safe_speed = max(0.0, float(other.speed) * 0.8)
                    if dist < 3.8: 
                        safe_speed = 0.0 
                    min_speed_cap = min(min_speed_cap, safe_speed)

        return min_speed_cap if min_speed_cap != float("inf") else None

    def update(self, dt: float):
        """
        Called every frame before scene.update().
        1. Queue finished cars for respawn.
        2. Respawn only when a spawn lane is clear.
        3. Update following and intersection speed limits.
        """
        dt = float(max(0.0, min(dt, 0.1)))
        self._sim_time += dt

        for car in self.cars:
            if car.name in self._spawn_timers:
                self._spawn_timers[car.name] += dt

        for car in self.cars:
            if car.is_active and car.route_finished:
                self._queue_for_respawn(car)

        self._process_pending_respawns()

        active_cars = [c for c in self.cars if c.is_active and not c.route_finished]
        self._refresh_zone_reservations(active_cars)
        self._update_intersection_queue(active_cars)

        for car in active_cars:
            desired_speed = float(car.target_speed)

            if car.name in self._car_entered_intersection:
                desired_speed = min(8.0, desired_speed * 1.5)

            following_cap = self._car_following_speed_cap(car, active_cars)
            if following_cap is not None:
                desired_speed = min(desired_speed, following_cap)

            radar_cap = self._radar_speed_cap(car, active_cars)
            if radar_cap is not None:
                desired_speed = min(desired_speed, radar_cap)

            intersection_cap = self._intersection_speed_cap(car)
            if intersection_cap is not None:
                desired_speed = min(desired_speed, intersection_cap)

            self._apply_speed_toward(car, desired_speed, dt)
