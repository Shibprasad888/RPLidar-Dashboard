from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pygame

from config import (
    CELL_DECAY_RATE,
    ENABLE_DYNAMIC_CELL_DECAY,
    ENABLE_MAP_EXPORT,
    FREE_THRESHOLD,
    FREE_UPDATE_RATE,
    MAP_HEIGHT_MM,
    MAP_WIDTH_MM,
    MAX_PROBABILITY,
    MIN_PROBABILITY,
    OCCUPIED_THRESHOLD,
    OCCUPIED_UPDATE_RATE,
    GRID_RESOLUTION_MM,
)
from .geometry import polar_to_cartesian
from .models import TrackedObject


class OccupancyGridMap:
    """Probabilistic 2D occupancy grid for local environment mapping."""

    def __init__(
        self,
        width_mm: int = MAP_WIDTH_MM,
        height_mm: int = MAP_HEIGHT_MM,
        resolution_mm: float = GRID_RESOLUTION_MM,
        origin_x_mm: float = 0.0,
        origin_y_mm: float = 0.0,
    ) -> None:
        if resolution_mm <= 0:
            raise ValueError("Grid resolution must be positive")
        self.width_mm = int(width_mm)
        self.height_mm = int(height_mm)
        self.resolution_mm = float(resolution_mm)
        self.origin_x_mm = float(origin_x_mm)
        self.origin_y_mm = float(origin_y_mm)
        self.width_cells = max(1, int(math.ceil(self.width_mm / self.resolution_mm)))
        self.height_cells = max(1, int(math.ceil(self.height_mm / self.resolution_mm)))
        self.probabilities = np.full((self.height_cells, self.width_cells), 0.5, dtype=np.float32)
        self.updated_cells = 0
        self.last_update_ms = 0.0
        self.runtime_s = 0.0
        self.start_time = time.monotonic()
        self._last_decay_ts = time.monotonic()
        self._mapping_fps = 0.0

    def world_to_grid(self, x_mm: float, y_mm: float) -> Tuple[int, int]:
        gx = int(round((x_mm - self.origin_x_mm) / self.resolution_mm)) + self.width_cells // 2
        gy = int(round((y_mm - self.origin_y_mm) / self.resolution_mm)) + self.height_cells // 2
        return gx, gy

    def grid_to_world(self, gx: int, gy: int) -> Tuple[float, float]:
        x_mm = (gx - self.width_cells // 2) * self.resolution_mm + self.origin_x_mm
        y_mm = (gy - self.height_cells // 2) * self.resolution_mm + self.origin_y_mm
        return x_mm, y_mm

    def world_to_screen(self, x_mm: float, y_mm: float, screen_width: int, screen_height: int) -> Tuple[int, int]:
        gx, gy = self.world_to_grid(x_mm, y_mm)
        cell_w = max(1, screen_width // max(1, self.width_cells))
        cell_h = max(1, screen_height // max(1, self.height_cells))
        return int(gx * cell_w), int(gy * cell_h)

    def screen_to_world(self, screen_x: int, screen_y: int, screen_width: int, screen_height: int) -> Tuple[float, float]:
        cell_w = max(1, screen_width // max(1, self.width_cells))
        cell_h = max(1, screen_height // max(1, self.height_cells))
        gx = int(screen_x // cell_w)
        gy = int(screen_y // cell_h)
        return self.grid_to_world(gx, gy)

    def update_from_scan(self, points: Iterable[Dict[str, Any]], tracked_objects: Optional[List[TrackedObject]] = None, timestamp: Optional[float] = None) -> Dict[str, Any]:
        start_time = time.perf_counter()
        self.updated_cells = 0
        self.runtime_s = time.monotonic() - self.start_time

        for point in points:
            try:
                angle_deg = float(point["angle"])
                distance_mm = float(point["distance"])
                quality = int(point.get("quality", 0))
            except (KeyError, TypeError, ValueError):
                continue
            if distance_mm <= 0 or quality < 10 or not math.isfinite(distance_mm):
                continue
            self._update_ray(angle_deg, distance_mm)

        if tracked_objects and ENABLE_DYNAMIC_CELL_DECAY:
            self._decay_dynamic_cells(tracked_objects, timestamp or time.monotonic())

        self.last_update_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
        self._mapping_fps = round(1000.0 / max(self.last_update_ms, 1e-3), 2) if self.last_update_ms > 0 else 0.0
        return self.get_stats()

    def clear(self) -> None:
        self.probabilities.fill(0.5)
        self.updated_cells = 0

    def get_stats(self) -> Dict[str, Any]:
        occupied = int(np.count_nonzero(self.probabilities >= OCCUPIED_THRESHOLD))
        free = int(np.count_nonzero(self.probabilities <= FREE_THRESHOLD))
        unknown = int(self.probabilities.size - occupied - free)
        coverage = round((occupied + free) / max(self.probabilities.size, 1) * 100.0, 2)
        return {
            "resolution_mm": self.resolution_mm,
            "width_mm": self.width_mm,
            "height_mm": self.height_mm,
            "occupied_cells": occupied,
            "free_cells": free,
            "unknown_cells": unknown,
            "coverage_percentage": coverage,
            "updated_cells": self.updated_cells,
            "mapping_fps": self._mapping_fps,
            "memory_usage_mb": round(self.probabilities.nbytes / (1024.0 * 1024.0), 2),
            "map_runtime_s": round(self.runtime_s, 2),
        }

    def state_for_probability(self, probability: float) -> str:
        if probability >= OCCUPIED_THRESHOLD:
            return "Occupied"
        if probability <= FREE_THRESHOLD:
            return "Free"
        return "Unknown"

    def _update_ray(self, angle_deg: float, distance_mm: float) -> None:
        obstacle_x_mm, obstacle_y_mm = polar_to_cartesian(angle_deg, distance_mm)
        start_x_mm, start_y_mm = 0.0, 0.0
        start_gx, start_gy = self.world_to_grid(start_x_mm, start_y_mm)
        end_gx, end_gy = self.world_to_grid(obstacle_x_mm, obstacle_y_mm)

        for gx, gy in self._ray_cells(start_gx, start_gy, end_gx, end_gy):
            if not self._in_bounds(gx, gy):
                break
            if (gx, gy) == (end_gx, end_gy):
                self._update_cell(gx, gy, occupied=True)
                break
            self._update_cell(gx, gy, occupied=False)

    def _update_cell(self, gx: int, gy: int, occupied: bool) -> None:
        if not self._in_bounds(gx, gy):
            return
        prev = float(self.probabilities[gy, gx])
        if occupied:
            new_prob = min(MAX_PROBABILITY, prev + OCCUPIED_UPDATE_RATE)
        else:
            new_prob = max(MIN_PROBABILITY, prev - FREE_UPDATE_RATE)
        if abs(new_prob - prev) > 1e-6:
            self.probabilities[gy, gx] = new_prob
            self.updated_cells += 1

    def _decay_dynamic_cells(self, tracked_objects: Iterable[TrackedObject], timestamp: float) -> None:
        if not ENABLE_DYNAMIC_CELL_DECAY:
            return
        dt_s = max(timestamp - self._last_decay_ts, 1e-3)
        self._last_decay_ts = timestamp
        for obj in tracked_objects:
            if obj.velocity_m_s <= 0.05:
                continue
            gx, gy = self.world_to_grid(obj.center_x_mm, obj.center_y_mm)
            if not self._in_bounds(gx, gy):
                continue
            decay = min(0.15, CELL_DECAY_RATE * dt_s)
            current = float(self.probabilities[gy, gx])
            self.probabilities[gy, gx] = max(MIN_PROBABILITY, current - decay)

    def _ray_cells(self, start_gx: int, start_gy: int, end_gx: int, end_gy: int) -> Iterable[Tuple[int, int]]:
        dx = abs(end_gx - start_gx)
        dy = -abs(end_gy - start_gy)
        sx = 1 if start_gx < end_gx else -1
        sy = 1 if start_gy < end_gy else -1
        err = dx + dy
        x, y = start_gx, start_gy
        while True:
            yield x, y
            if x == end_gx and y == end_gy:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x += sx
            if e2 <= dx:
                err += dx
                y += sy

    def _in_bounds(self, gx: int, gy: int) -> bool:
        return 0 <= gx < self.width_cells and 0 <= gy < self.height_cells

    def to_surface(self, width: int, height: int) -> pygame.Surface:
        surface = pygame.Surface((width, height))
        cell_w = max(1, width // max(1, self.width_cells))
        cell_h = max(1, height // max(1, self.height_cells))
        for gy in range(self.height_cells):
            for gx in range(self.width_cells):
                px = gx * cell_w
                py = gy * cell_h
                probability = float(self.probabilities[gy, gx])
                color = self._cell_color(probability)
                pygame.draw.rect(surface, color, (px, py, cell_w, cell_h))
        return surface

    def _cell_color(self, probability: float) -> Tuple[int, int, int]:
        if probability >= OCCUPIED_THRESHOLD:
            return (255, 255, 255)
        if probability <= FREE_THRESHOLD:
            return (0, 0, 0)
        return (60, 60, 60)

    def save(self, folder: str = "maps") -> Dict[str, Any]:
        if not ENABLE_MAP_EXPORT:
            return {}
        output_dir = Path(folder)
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        npy_path = output_dir / f"occupancy_{timestamp}.npy"
        json_path = output_dir / f"occupancy_{timestamp}.json"
        png_path = output_dir / f"occupancy_{timestamp}.png"
        np.save(npy_path, self.probabilities)
        metadata = {
            "resolution_mm": self.resolution_mm,
            "width_mm": self.width_mm,
            "height_mm": self.height_mm,
            "timestamp": timestamp,
            "origin": {"x_mm": self.origin_x_mm, "y_mm": self.origin_y_mm},
            "configuration": {
                "occupied_threshold": OCCUPIED_THRESHOLD,
                "free_threshold": FREE_THRESHOLD,
                "occupied_update_rate": OCCUPIED_UPDATE_RATE,
                "free_update_rate": FREE_UPDATE_RATE,
            },
        }
        json_path.write_text(json.dumps(metadata, indent=2))
        surface = self.to_surface(800, 800)
        pygame.image.save(surface, str(png_path))
        return {"npy": str(npy_path), "json": str(json_path), "png": str(png_path)}

    def load(self, path: str) -> None:
        loaded = np.load(path)
        if loaded.shape != self.probabilities.shape:
            raise ValueError("Loaded map shape does not match the current grid")
        self.probabilities = loaded.astype(np.float32)
