import math
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

from .ai_detector import AIDetection
from .models import TrackedObject
from .geometry import Point2D, distance_between


@dataclass(slots=True)
class FusedObject:
    track_id: Optional[int]
    class_name: Optional[str]
    confidence: float
    bounding_box: Tuple[int, int, int, int]
    distance_m: float
    bearing_deg: float
    velocity_m_s: float
    tracking_status: str
    risk_level: str
    position_mm: Tuple[float, float, float]
    fused_timestamp: float
    ai_timestamp: float
    lidar_timestamp: float


class SensorFusion:
    """Associate camera detections with LiDAR tracked objects."""

    def __init__(
        self,
        max_distance_mm: float = 3000.0,
        distance_threshold_mm: float = 800.0,
        angle_threshold_deg: float = 45.0,
        debug: bool = False,
    ) -> None:
        self.max_distance_mm = max_distance_mm
        self.distance_threshold_mm = distance_threshold_mm
        self.angle_threshold_deg = angle_threshold_deg
        self.debug = debug

    def fuse(
        self,
        detections: List[AIDetection],
        tracks: List[TrackedObject],
        lidar_points: List[dict[str, Any]],
        calibration: Any,
        frame_timestamp: float,
    ) -> List[FusedObject]:
        fused: List[FusedObject] = []
        for detection in detections:
            best_match: Optional[TrackedObject] = None
            best_score = float("inf")
            for track in tracks:
                distance = math.hypot(track.center_x_mm, track.center_y_mm)
                angle_diff = abs(track.angle_deg - self._estimate_angle_from_bbox(detection, calibration))
                if angle_diff > self.angle_threshold_deg or distance > self.max_distance_mm:
                    continue
                spatial_score = math.hypot(track.center_x_mm - detection.center_x, track.center_y_mm - detection.center_y)
                if spatial_score < best_score:
                    best_score = spatial_score
                    best_match = track

            if best_match is not None and best_score <= self.distance_threshold_mm:
                fused.append(
                    FusedObject(
                        track_id=best_match.track_id,
                        class_name=detection.class_name,
                        confidence=detection.confidence,
                        bounding_box=detection.bbox,
                        distance_m=best_match.distance_mm / 1000.0,
                        bearing_deg=best_match.angle_deg,
                        velocity_m_s=best_match.velocity_m_s,
                        tracking_status=best_match.status,
                        risk_level=best_match.risk_level,
                        position_mm=(best_match.center_x_mm, best_match.center_y_mm, 0.0),
                        fused_timestamp=frame_timestamp,
                        ai_timestamp=detection.timestamp,
                        lidar_timestamp=frame_timestamp,
                    )
                )
            else:
                fused.append(
                    FusedObject(
                        track_id=None,
                        class_name=detection.class_name,
                        confidence=detection.confidence,
                        bounding_box=detection.bbox,
                        distance_m=0.0,
                        bearing_deg=0.0,
                        velocity_m_s=0.0,
                        tracking_status="Unmatched",
                        risk_level="UNKNOWN",
                        position_mm=(0.0, 0.0, 0.0),
                        fused_timestamp=frame_timestamp,
                        ai_timestamp=detection.timestamp,
                        lidar_timestamp=frame_timestamp,
                    )
                )
        return fused

    def _estimate_angle_from_bbox(self, detection: AIDetection, calibration: Any) -> float:
        if calibration is None:
            return 0.0
        image_width = calibration.intrinsics.cx * 2.0
        dx = (detection.center_x - calibration.intrinsics.cx) / calibration.intrinsics.cx
        return dx * (calibration.fov_deg / 2.0)
