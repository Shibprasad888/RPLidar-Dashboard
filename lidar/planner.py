from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np


class PlannerType(Enum):
    A_STAR = "A*"
    DIJKSTRA = "Dijkstra"
    THETA_STAR = "Theta*"
    RRT = "RRT"
    RRT_STAR = "RRT*"
    HYBRID_A_STAR = "Hybrid A*"


@dataclass(slots=True)
class PlannerResult:
    path: List[Tuple[int, int]]
    cost: float
    expanded_nodes: int
    open_set_count: int
    closed_set_count: int


class Planner:
    """Planner interface for grid-based path planning."""

    def plan(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        cost_map: np.ndarray,
        allow_diagonal: bool = True,
    ) -> PlannerResult:
        raise NotImplementedError


class AStarPlanner(Planner):
    """A* grid-based planner with configurable traversal cost."""

    def __init__(self, heuristic: Optional[Callable[[Tuple[int, int], Tuple[int, int]], float]] = None) -> None:
        self.heuristic = heuristic or self._euclidean_heuristic

    def plan(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        cost_map: np.ndarray,
        allow_diagonal: bool = True,
    ) -> PlannerResult:
        if not self._in_bounds(start, cost_map) or not self._in_bounds(goal, cost_map):
            return PlannerResult([], float("inf"), 0, 0, 0)
        if not self._is_traversable(start, cost_map) or not self._is_traversable(goal, cost_map):
            return PlannerResult([], float("inf"), 0, 0, 0)

        width, height = cost_map.shape[1], cost_map.shape[0]
        open_heap: List[Tuple[float, Tuple[int, int]]] = []
        heapq.heappush(open_heap, (0.0, start))
        came_from: dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score: dict[Tuple[int, int], float] = {start: 0.0}
        closed_set: set[Tuple[int, int]] = set()

        open_set_count = 1
        expanded_nodes = 0

        while open_heap:
            _, current = heapq.heappop(open_heap)
            if current in closed_set:
                continue
            if current == goal:
                return PlannerResult(self._reconstruct_path(came_from, current), g_score[current], expanded_nodes, open_set_count, len(closed_set))

            closed_set.add(current)
            expanded_nodes += 1

            for neighbor, move_cost in self._neighbors(current, allow_diagonal, width, height, cost_map):
                if neighbor in closed_set:
                    continue
                tentative_g = g_score[current] + move_cost
                if tentative_g >= g_score.get(neighbor, float("inf")):
                    continue
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + self.heuristic(neighbor, goal)
                heapq.heappush(open_heap, (f_score, neighbor))
                open_set_count = max(open_set_count, len(open_heap))

        return PlannerResult([], float("inf"), expanded_nodes, open_set_count, len(closed_set))

    def _in_bounds(self, node: Tuple[int, int], cost_map: np.ndarray) -> bool:
        x, y = node
        return 0 <= x < cost_map.shape[1] and 0 <= y < cost_map.shape[0]

    def _is_traversable(self, node: Tuple[int, int], cost_map: np.ndarray) -> bool:
        x, y = node
        return math.isfinite(cost_map[y, x]) and cost_map[y, x] >= 0.0

    def _neighbors(
        self,
        current: Tuple[int, int],
        allow_diagonal: bool,
        width: int,
        height: int,
        cost_map: np.ndarray,
    ) -> Iterable[Tuple[Tuple[int, int], float]]:
        x, y = current
        cardinal = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        diagonal = [(-1, -1), (-1, 1), (1, -1), (1, 1)] if allow_diagonal else []

        for dx, dy in cardinal + diagonal:
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height and self._is_traversable((nx, ny), cost_map):
                base_cost = cost_map[ny, nx]
                step_cost = math.hypot(dx, dy) * max(base_cost, 1.0)
                yield (nx, ny), step_cost

    def _reconstruct_path(self, came_from: dict[Tuple[int, int], Tuple[int, int]], current: Tuple[int, int]) -> List[Tuple[int, int]]:
        path: List[Tuple[int, int]] = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    @staticmethod
    def _euclidean_heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])


class DijkstraPlanner(AStarPlanner):
    def __init__(self) -> None:
        super().__init__(heuristic=lambda a, b: 0.0)


class ThetaStarPlanner(AStarPlanner):
    def __init__(self) -> None:
        super().__init__()


def build_planner(planner_type: PlannerType) -> Planner:
    if planner_type == PlannerType.DIJKSTRA:
        return DijkstraPlanner()
    if planner_type == PlannerType.THETA_STAR:
        return ThetaStarPlanner()
    return AStarPlanner()


def parse_planner_type(planner_name: str) -> PlannerType:
    normalized = planner_name.strip().upper().replace("-", "_").replace(" ", "_").replace("*", "STAR")
    if normalized in PlannerType.__members__:
        return PlannerType[normalized]
    return PlannerType.A_STAR


def smooth_path(path: List[Tuple[int, int]], cost_map: np.ndarray) -> List[Tuple[int, int]]:
    if len(path) < 3:
        return path
    smoothed: List[Tuple[int, int]] = [path[0]]
    index = 0
    while index < len(path) - 1:
        next_index = len(path) - 1
        while next_index > index + 1:
            if is_line_of_sight_free(path[index], path[next_index], cost_map):
                break
            next_index -= 1
        smoothed.append(path[next_index])
        index = next_index
    return smoothed


def inflate_occupancy(
    occupancy: np.ndarray,
    inflation_radius_cells: int,
    unknown_cost: float = 4.0,
    avoid_unknown: bool = False,
) -> np.ndarray:
    inflated = np.copy(occupancy)
    height, width = occupancy.shape
    radius_sq = inflation_radius_cells * inflation_radius_cells
    occupied_mask = occupancy >= 1.0

    for y in range(height):
        for x in range(width):
            if not occupied_mask[y, x]:
                continue
            min_x = max(0, x - inflation_radius_cells)
            max_x = min(width - 1, x + inflation_radius_cells)
            min_y = max(0, y - inflation_radius_cells)
            max_y = min(height - 1, y + inflation_radius_cells)
            for iy in range(min_y, max_y + 1):
                for ix in range(min_x, max_x + 1):
                    if (ix - x) ** 2 + (iy - y) ** 2 <= radius_sq:
                        inflated[iy, ix] = float("inf")

    if avoid_unknown:
        unknown_mask = (occupancy > 0.0) & (occupancy < 1.0)
        inflated[unknown_mask] = unknown_cost
    else:
        inflated[(occupancy > 0.0) & (occupancy < 1.0)] = unknown_cost

    return inflated


def occupancy_to_cost_map(
    occupancy: np.ndarray,
    unknown_cost: float = 4.0,
    free_cost: float = 1.0,
    occupied_cost: float = float("inf"),
) -> np.ndarray:
    cost_map = np.full_like(occupancy, free_cost, dtype=float)
    cost_map[occupancy >= 1.0] = occupied_cost
    cost_map[(occupancy > 0.0) & (occupancy < 1.0)] = unknown_cost
    return cost_map


def is_line_of_sight_free(
    start: Tuple[int, int],
    end: Tuple[int, int],
    cost_map: np.ndarray,
) -> bool:
    x0, y0 = start
    x1, y1 = end
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    while True:
        if not math.isfinite(cost_map[y, x]) or cost_map[y, x] >= float("inf"):
            return False
        if (x, y) == (x1, y1):
            return True
        e2 = err * 2
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
