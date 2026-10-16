from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, List, Optional, Tuple

from .geometry import Point2D


class ObjectStatus(Enum):
    ACTIVE = "Active"
    TEMPORARILY_LOST = "Temporarily Lost"
    RECOVERED = "Recovered"
    DELETED = "Deleted"


@dataclass(slots=True)
class TrackedObject:
    """A tracked object with persistent ID and motion history."""

    track_id: int
    center_x_mm: float
    center_y_mm: float
    prev_center_x_mm: float
    prev_center_y_mm: float
    distance_mm: float
    angle_deg: float
    velocity_m_s: float
    heading_deg: float
    age_frames: int
    last_seen_ts: float
    frames_tracked: int
    frames_missing: int
    confidence: float
    status: str
    radius_mm: float
    point_count: int
    density: float
    cluster_confidence: float
    trail: Deque[Tuple[float, float]] = field(default_factory=lambda: deque(maxlen=10))
    predicted_x_mm: Optional[float] = None
    predicted_y_mm: Optional[float] = None
    acceleration_m_s2: float = 0.0
    angular_velocity_deg_s: float = 0.0
    direction: str = "Unknown"
    current_zone: str = "Safe"
    zone_entry_ts: Optional[float] = None
    zone_exit_ts: Optional[float] = None
    zone_time_s: float = 0.0
    risk_level: str = "SAFE"
    collision_confidence: float = 0.0
    ttc_s: float = float("inf")
    motion_debug: Dict[str, Any] = field(default_factory=dict)

    def update_from_detection(self, detection: "DetectedObject", timestamp: float, smoothing: float, trail_length: int) -> None:
        self.prev_center_x_mm = self.center_x_mm
        self.prev_center_y_mm = self.center_y_mm
        self.center_x_mm = detection.center_x_mm
        self.center_y_mm = detection.center_y_mm
        self.distance_mm = detection.distance_mm
        self.angle_deg = detection.angle_deg
        self.radius_mm = detection.radius_mm
        self.point_count = detection.point_count
        self.density = detection.density
        self.cluster_confidence = detection.confidence
        self.trail.append((self.center_x_mm, self.center_y_mm))

        dt = max(timestamp - self.last_seen_ts, 1e-3)
        dx = self.center_x_mm - self.prev_center_x_mm
        dy = self.center_y_mm - self.prev_center_y_mm
        speed = math.hypot(dx, dy) / dt / 1000.0
        heading = math.degrees(math.atan2(dx, -dy))
        if heading < 0:
            heading += 360.0

        self.velocity_m_s = (self.velocity_m_s * (1.0 - smoothing)) + (speed * smoothing)
        self.heading_deg = (self.heading_deg * (1.0 - smoothing)) + (heading * smoothing)
        self.frames_tracked += 1
        self.frames_missing = 0
        self.age_frames += 1
        self.last_seen_ts = timestamp
        self.status = ObjectStatus.ACTIVE.value
        self.predicted_x_mm = self.center_x_mm
        self.predicted_y_mm = self.center_y_mm

    def mark_missed(self, timestamp: float) -> None:
        self.prev_center_x_mm = self.center_x_mm
        self.prev_center_y_mm = self.center_y_mm
        self.frames_missing += 1
        self.last_seen_ts = timestamp
        if self.status != ObjectStatus.TEMPORARILY_LOST.value:
            self.status = ObjectStatus.TEMPORARILY_LOST.value

    def predict_position(self, timestamp: float) -> Tuple[float, float]:
        dt = max(timestamp - self.last_seen_ts, 0.0)
        displacement_mm = self.velocity_m_s * dt * 1000.0
        heading_rad = math.radians(self.heading_deg)
        predicted_x = self.center_x_mm + displacement_mm * math.sin(heading_rad)
        predicted_y = self.center_y_mm - displacement_mm * math.cos(heading_rad)
        self.predicted_x_mm = predicted_x
        self.predicted_y_mm = predicted_y
        return predicted_x, predicted_y


@dataclass(slots=True)
class DetectedObject:
    object_id: int
    points: List[Point2D]
    center_x_mm: float
    center_y_mm: float
    distance_mm: float
    angle_deg: float
    width_mm: float
    height_mm: float
    radius_mm: float
    point_count: int
    density: float
    confidence: float
