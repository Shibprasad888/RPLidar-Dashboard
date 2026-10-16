from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

from config import (
    DEBUG,
    ENABLE_NAVIGATION,
    ENABLE_OBSTACLE_INFLATION,
    ENABLE_DYNAMIC_REPLANNING,
    ENABLE_NAVIGATION_SIMULATION,
    GOAL_TOLERANCE_MM,
    GRID_INFLATION_RADIUS_MM,
    MAX_SPEED_M_S,
    MAX_TURN_RATE_DEG_S,
    PLANNER_TYPE,
    REPLAN_INTERVAL_S,
    SIMULATION_SPEED,
    UNKNOWN_CELL_COST,
)
from .planner import (
    AStarPlanner,
    Planner,
    PlannerResult,
    PlannerType,
    build_planner,
    inflate_occupancy,
    occupancy_to_cost_map,
    parse_planner_type,
    smooth_path,
)
from .slam import Pose, SLAMSystem
from .fusion import FusedObject
from .geometry import Point2D


class NavigationState(Enum):
    IDLE = "Idle"
    PLANNING = "Planning"
    FOLLOWING_PATH = "Following Path"
    WAITING = "Waiting"
    REPLANNING = "Re-planning"
    GOAL_REACHED = "Goal Reached"
    NAVIGATION_FAILED = "Navigation Failed"
    EMERGENCY_STOP = "Emergency Stop"


@dataclass(slots=True)
class NavigationStatus:
    state: NavigationState = NavigationState.IDLE
    goal: Optional[Point2D] = None
    current_pose: Pose = field(default_factory=Pose)
    remaining_distance_mm: float = 0.0
    current_speed_m_s: float = 0.0
    estimated_time_s: float = 0.0
    path_length_mm: float = 0.0
    waypoints: List[Point2D] = field(default_factory=list)
    current_waypoint_index: int = 0
    planner_result: Optional[PlannerResult] = None
    replan_count: int = 0
    last_planning_time_ms: float = 0.0
    dynamic_obstacle_count: int = 0
    dynamic_obstacles: List[Point2D] = field(default_factory=list)
    event_log: Deque[str] = field(default_factory=lambda: deque(maxlen=40))
    navigation_fps: float = 0.0
    last_update_ts: float = field(default_factory=time.monotonic)

    def log_event(self, message: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        self.event_log.appendleft(f"{timestamp} - {message}")


@dataclass(slots=True)
class NavigationRequest:
    goal_x_mm: float
    goal_y_mm: float
    allow_random: bool = False


class NavigationEngine:
    """Simulated navigation engine for planning and local execution."""

    def __init__(self, slam_system: SLAMSystem, planner_type: PlannerType = PlannerType.A_STAR) -> None:
        self.slam_system = slam_system
        self.planner: Planner = build_planner(planner_type)
        self.status = NavigationStatus()
        self._lock = threading.Lock()
        self._last_replan_ts = 0.0
        self._simulation_speed = SIMULATION_SPEED
        self._max_speed = MAX_SPEED_M_S
        self._max_turn_rate = MAX_TURN_RATE_DEG_S
        self._goal_tolerance_mm = GOAL_TOLERANCE_MM
        self._inflation_cells = max(0, int(round(GRID_INFLATION_RADIUS_MM / self.slam_system.occupancy_map.resolution_mm)))
        self._unknown_cost = UNKNOWN_CELL_COST
        self._nav_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        if self._nav_thread is not None and self._nav_thread.is_alive():
            return
        self._stop_event.clear()
        self._nav_thread = threading.Thread(target=self._run, daemon=True)
        self._nav_thread.start()
        self.status.log_event("Navigation engine started")

    def stop(self) -> None:
        self._stop_event.set()
        if self._nav_thread is not None:
            self._nav_thread.join(timeout=1.0)
        self.status.state = NavigationState.IDLE
        self.status.log_event("Navigation engine stopped")

    def request_goal(self, request: NavigationRequest) -> bool:
        if not ENABLE_NAVIGATION:
            return False

        with self._lock:
            try:
                goal_gx, goal_gy = self.slam_system.occupancy_map.world_to_grid(request.goal_x_mm, request.goal_y_mm)
                if not self._is_reachable(goal_gx, goal_gy):
                    self.status.log_event("Goal unreachable")
                    return False
                self.status.goal = Point2D(request.goal_x_mm, request.goal_y_mm)
                self.status.current_pose = Pose(
                    x_mm=self.slam_system.pose.x_mm,
                    y_mm=self.slam_system.pose.y_mm,
                    heading_deg=self.slam_system.pose.heading_deg,
                )
                self.status.state = NavigationState.PLANNING
                self.status.current_waypoint_index = 0
                self.status.waypoints.clear()
                self.status.replan_count = 0
                self.status.dynamic_obstacles.clear()
                self.status.log_event(f"Goal selected: ({request.goal_x_mm:.0f}, {request.goal_y_mm:.0f})")
                return True
            except Exception:
                self.status.log_event("Failed to validate goal")
                return False

    def _is_reachable(self, gx: int, gy: int) -> bool:
        occupancy = self.slam_system.occupancy_map.probabilities
        if not self.slam_system.occupancy_map._in_bounds(gx, gy):
            return False
        cell_value = float(occupancy[gy, gx])
        return cell_value < 1.0 and cell_value <= 0.5

    def _run(self) -> None:
        while not self._stop_event.is_set():
            if not ENABLE_NAVIGATION or not ENABLE_NAVIGATION_SIMULATION:
                time.sleep(0.1)
                continue
            self._update_navigation()
            time.sleep(0.04)

    def _update_navigation(self) -> None:
        now = time.monotonic()
        if self.status.goal is None:
            return

        if self.status.state in {NavigationState.IDLE, NavigationState.GOAL_REACHED, NavigationState.NAVIGATION_FAILED}:
            self.status.state = NavigationState.PLANNING

        if self.status.state in {NavigationState.PLANNING, NavigationState.REPLANNING}:
            self._plan_path(now)
            return

        if self.status.state == NavigationState.FOLLOWING_PATH:
            self._follow_path(now)
            return

        if self.status.state == NavigationState.WAITING:
            self.status.log_event("Waiting for obstacle to clear")
            if self._should_replan(now):
                self.status.state = NavigationState.REPLANNING
            return

    def _plan_path(self, now: float) -> None:
        self.status.log_event("Path planning started")
        start = self.slam_system.occupancy_map.world_to_grid(self.status.current_pose.x_mm, self.status.current_pose.y_mm)
        goal = self.slam_system.occupancy_map.world_to_grid(self.status.goal.x_mm, self.status.goal.y_mm)
        occupancy = self.slam_system.occupancy_map.probabilities

        if ENABLE_OBSTACLE_INFLATION:
            inflated = inflate_occupancy(occupancy, self._inflation_cells, unknown_cost=self._unknown_cost, avoid_unknown=True)
        else:
            inflated = np.copy(occupancy)
        cost_map = occupancy_to_cost_map(inflated, unknown_cost=self._unknown_cost)

        start_time = time.perf_counter()
        result = self.planner.plan(start, goal, cost_map)
        self.status.last_planning_time_ms = (time.perf_counter() - start_time) * 1000.0
        self.status.planner_result = result
        self.status.open_set_count = result.open_set_count if hasattr(result, "open_set_count") else 0
        self.status.closed_set_count = result.closed_set_count if hasattr(result, "closed_set_count") else 0
        self._last_replan_ts = now

        if not result.path:
            self.status.state = NavigationState.NAVIGATION_FAILED
            self.status.log_event("Path planning failed")
            return

        if PATH_SMOOTHING:
            result.path = smooth_path(result.path, cost_map)

        self.status.waypoints = [Point2D(*self.slam_system.occupancy_map.grid_to_world(gx, gy)) for gx, gy in result.path]
        self.status.current_waypoint_index = 0
        self.status.path_length_mm = self._compute_path_length(self.status.waypoints)
        self.status.remaining_distance_mm = self.status.path_length_mm
        self.status.estimated_time_s = self.status.path_length_mm / max(self._max_speed, 1e-3)
        self.status.state = NavigationState.FOLLOWING_PATH
        self.status.log_event("Path planned")
        self.status.last_update_ts = time.monotonic()

    def _compute_path_length(self, points: List[Point2D]) -> float:
        if len(points) < 2:
            return 0.0
        length = 0.0
        for previous, current in zip(points, points[1:]):
            length += math.hypot(current.x_mm - previous.x_mm, current.y_mm - previous.y_mm)
        return length

    def _follow_path(self, now: float) -> None:
        if self.status.current_waypoint_index >= len(self.status.waypoints):
            self.status.state = NavigationState.GOAL_REACHED
            self.status.remaining_distance_mm = 0.0
            self.status.log_event("Goal reached")
            return

        waypoint = self.status.waypoints[self.status.current_waypoint_index]
        dx = waypoint.x_mm - self.status.current_pose.x_mm
        dy = waypoint.y_mm - self.status.current_pose.y_mm
        distance = math.hypot(dx, dy)

        if distance <= self._goal_tolerance_mm:
            self.status.current_waypoint_index += 1
            self.status.log_event(f"Reached waypoint {self.status.current_waypoint_index}")
            return

        heading_target = math.degrees(math.atan2(dx, dy))
        turn_error = self._normalize_angle(heading_target - self.status.current_pose.heading_deg)
        turn_rate = max(-self._max_turn_rate, min(self._max_turn_rate, turn_error * 0.5))
        move_speed = min(self._max_speed, distance / max(0.1, self._simulation_speed))

        dt = max(now - self.status.last_update_ts, 1e-3)
        self.status.current_speed_m_s = move_speed
        self.status.current_pose.heading_deg += turn_rate * dt
        self.status.current_pose.x_mm += math.sin(math.radians(self.status.current_pose.heading_deg)) * move_speed * dt * 1000.0
        self.status.current_pose.y_mm -= math.cos(math.radians(self.status.current_pose.heading_deg)) * move_speed * dt * 1000.0
        self.status.navigation_fps = 1.0 / max(dt, 1e-6)
        self.status.last_update_ts = now

        self.status.remaining_distance_mm = self._compute_path_length(self.status.waypoints[self.status.current_waypoint_index:])
        self.status.estimated_time_s = self.status.remaining_distance_mm / max(self._max_speed, 1e-3)

        if ENABLE_DYNAMIC_REPLANNING and self._obstacle_on_path(waypoint):
            self.status.state = NavigationState.REPLANNING
            self.status.replan_count += 1
            self.status.log_event("Dynamic obstacle detected, replanning")
            return

        if self.status.remaining_distance_mm <= self._goal_tolerance_mm:
            self.status.state = NavigationState.GOAL_REACHED
            self.status.log_event("Goal reached")

    def _obstacle_on_path(self, waypoint: Point2D) -> bool:
        obstacle_count = 0
        for obstacle in self.status.dynamic_obstacles:
            dx = obstacle.x_mm - waypoint.x_mm
            dy = obstacle.y_mm - waypoint.y_mm
            if math.hypot(dx, dy) <= GRID_INFLATION_RADIUS_MM * 1.2:
                obstacle_count += 1
        self.status.dynamic_obstacle_count = obstacle_count
        return obstacle_count > 0

    def _should_replan(self, now: float) -> bool:
        return ENABLE_DYNAMIC_REPLANNING and (now - self._last_replan_ts) >= REPLAN_INTERVAL_S

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        while angle > 180.0:
            angle -= 360.0
        while angle <= -180.0:
            angle += 360.0
        return angle

    def update_dynamic_obstacles(self, fused_objects: Optional[List[FusedObject]] = None, tracked_objects: Optional[List[Any]] = None) -> None:
        with self._lock:
            obstacles: List[Point2D] = []
            if fused_objects:
                for fused in fused_objects:
                    obstacles.append(Point2D(fused.position_mm[0], fused.position_mm[1]))
            elif tracked_objects:
                for track in tracked_objects:
                    if track.velocity_m_s > 0.05:
                        obstacles.append(Point2D(track.center_x_mm, track.center_y_mm))
            self.status.dynamic_obstacles = obstacles
            self.status.dynamic_obstacle_count = len(obstacles)


class NavigationSimulator:
    def __init__(self, engine: NavigationEngine) -> None:
        self.engine = engine

    def get_navigation_status(self) -> NavigationStatus:
        return self.engine.status

    def select_random_goal(self) -> Optional[NavigationRequest]:
        occupancy = self.engine.slam_system.occupancy_map.probabilities
        free_cells = list(zip(*np.where(occupancy <= 0.5)))
        if not free_cells:
            return None
        y, x = free_cells[np.random.randint(0, len(free_cells))]
        x_mm, y_mm = self.engine.slam_system.occupancy_map.grid_to_world(x, y)
        return NavigationRequest(goal_x_mm=x_mm, goal_y_mm=y_mm, allow_random=True)
