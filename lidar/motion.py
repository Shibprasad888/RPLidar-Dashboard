from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from config import (
    ACCELERATION_SMOOTHING,
    DEBUG_MOTION,
    PREDICTION_TIME_HORIZON,
    VELOCITY_SMOOTHING,
)
from .models import TrackedObject


class MotionAnalyzer:
    """Analyze motion and predict object trajectories in real time."""

    def __init__(self) -> None:
        self._debug = DEBUG_MOTION

    def analyze(self, track: TrackedObject, timestamp: float) -> Dict[str, Any]:
        dt = max(timestamp - track.last_seen_ts, 1e-3)
        dx = track.center_x_mm - track.prev_center_x_mm
        dy = track.center_y_mm - track.prev_center_y_mm
        distance_mm = math.hypot(dx, dy)
        velocity_m_s = distance_mm / dt / 1000.0
        acceleration_m_s2 = self._estimate_acceleration(track, velocity_m_s)
        heading_deg = self._heading_from_delta(dx, dy)
        angular_velocity_deg_s = self._estimate_angular_velocity(track, heading_deg, dt)
        direction = self._direction_from_heading(heading_deg)
        future_positions = self.predict_future_positions(track, velocity_m_s, heading_deg)

        return {
            "velocity_m_s": velocity_m_s,
            "acceleration_m_s2": acceleration_m_s2,
            "heading_deg": heading_deg,
            "angular_velocity_deg_s": angular_velocity_deg_s,
            "direction": direction,
            "future_positions": future_positions,
            "debug": self._debug,
        }

    def predict_future_positions(self, track: TrackedObject, velocity_m_s: float, heading_deg: float) -> List[Dict[str, float]]:
        positions = []
        for horizon in [0.5, 1.0, 2.0]:
            predicted_distance_mm = velocity_m_s * horizon * 1000.0
            heading_rad = math.radians(heading_deg)
            predicted_x = track.center_x_mm + predicted_distance_mm * math.sin(heading_rad)
            predicted_y = track.center_y_mm - predicted_distance_mm * math.cos(heading_rad)
            positions.append({"time_s": horizon, "x_mm": predicted_x, "y_mm": predicted_y})
        return positions

    def _estimate_acceleration(self, track: TrackedObject, velocity_m_s: float) -> float:
        if track.frames_tracked <= 1:
            return 0.0
        previous_velocity = max(track.velocity_m_s, 0.0)
        alpha = ACCELERATION_SMOOTHING
        return (velocity_m_s - previous_velocity) * alpha

    def _estimate_angular_velocity(self, track: TrackedObject, heading_deg: float, dt: float) -> float:
        if track.frames_tracked <= 1:
            return 0.0
        previous_heading = track.heading_deg
        delta = abs(heading_deg - previous_heading)
        if delta > 180.0:
            delta = 360.0 - delta
        return delta / dt

    def _heading_from_delta(self, dx: float, dy: float) -> float:
        heading = math.degrees(math.atan2(dx, -dy))
        if heading < 0:
            heading += 360.0
        return heading

    def _direction_from_heading(self, heading_deg: float) -> str:
        if heading_deg <= 45.0 or heading_deg >= 315.0:
            return "North"
        if heading_deg <= 135.0:
            return "East"
        if heading_deg <= 225.0:
            return "South"
        return "West"
