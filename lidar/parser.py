import math
from typing import Any, Dict, Iterable, List, Optional, Tuple


def parse_scan_point(quality: Any, angle: Any, distance: Any) -> Optional[Dict[str, Any]]:
    """Convert a single scan tuple into a structured point dictionary."""
    try:
        quality_value = int(quality)
        angle_value = float(angle)
        distance_value = int(distance)
    except (TypeError, ValueError):
        return None

    if distance_value <= 0:
        return None
    if not math.isfinite(angle_value) or not 0.0 <= angle_value <= 360.0:
        return None
    if not math.isfinite(distance_value):
        return None
    if not 0 <= quality_value <= 255:
        return None

    return {
        "angle": round(angle_value, 2),
        "distance": distance_value,
        "quality": quality_value,
    }


def parse_scan_batch(raw_scan: Iterable[tuple[int, float, float]]) -> List[Dict[str, Any]]:
    """Parse a batch of raw scan measurements into clean points."""
    parsed_points: List[Dict[str, Any]] = []
    for quality, angle, distance in raw_scan:
        point = parse_scan_point(quality, angle, distance)
        if point is not None:
            parsed_points.append(point)
    return parsed_points
