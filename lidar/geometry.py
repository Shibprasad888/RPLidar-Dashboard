import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass(slots=True)
class Point2D:
    """A 2D point in Cartesian coordinates with millimeter units."""

    x_mm: float
    y_mm: float
    quality: int = 0
    angle_deg: float = 0.0
    distance_mm: float = 0.0


@dataclass(slots=True)
class DetectedObject:
    """A cluster-derived object with geometry and simple metadata."""

    object_id: int
    points: List[Point2D] = field(default_factory=list)
    center_x_mm: float = 0.0
    center_y_mm: float = 0.0
    distance_mm: float = 0.0
    angle_deg: float = 0.0
    width_mm: float = 0.0
    height_mm: float = 0.0
    radius_mm: float = 0.0
    point_count: int = 0
    density: float = 0.0
    confidence: float = 0.0

    def to_dict(self) -> dict[str, float | int | list[dict[str, float | int]]]:
        """Return a serializable dictionary for the renderer and debug output."""
        return {
            "object_id": self.object_id,
            "center_x_mm": self.center_x_mm,
            "center_y_mm": self.center_y_mm,
            "distance_mm": self.distance_mm,
            "angle_deg": self.angle_deg,
            "width_mm": self.width_mm,
            "height_mm": self.height_mm,
            "radius_mm": self.radius_mm,
            "point_count": self.point_count,
            "density": self.density,
            "confidence": self.confidence,
        }


def polar_to_cartesian(angle_deg: float, distance_mm: float) -> Tuple[float, float]:
    """Convert a polar point into Cartesian coordinates in millimeters."""
    angle_rad = math.radians(angle_deg)
    x_mm = distance_mm * math.sin(angle_rad)
    y_mm = -distance_mm * math.cos(angle_rad)
    return x_mm, y_mm


def cartesian_to_polar(x_mm: float, y_mm: float) -> Tuple[float, float]:
    """Convert Cartesian coordinates into polar form."""
    distance_mm = math.hypot(x_mm, y_mm)
    angle_deg = math.degrees(math.atan2(x_mm, -y_mm))
    if angle_deg < 0:
        angle_deg += 360.0
    return angle_deg, distance_mm


def distance_between(a: Point2D, b: Point2D) -> float:
    """Calculate Euclidean distance between two points in millimeters."""
    return math.hypot(a.x_mm - b.x_mm, a.y_mm - b.y_mm)


def clamp(value: float, lower: float, upper: float) -> float:
    """Clamp a value to a safe range."""
    return max(lower, min(upper, value))
