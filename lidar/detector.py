import math
import time
from typing import Any, List, Optional, Tuple

from sklearn.cluster import DBSCAN

from config import (
    DBSCAN_EPS_MM,
    DBSCAN_MIN_SAMPLES,
    DEBUG,
    ENABLE_BOUNDING_CIRCLES,
    ENABLE_CONNECTION_LINES,
    ENABLE_DEBUG_LABELS,
    ENABLE_LABELS,
    ENABLE_STATISTICS,
    MAX_CLUSTER_SIZE,
    MAX_DETECTION_DISTANCE_MM,
    MIN_CLUSTER_SIZE,
    MIN_DETECTION_DISTANCE_MM,
)
from .geometry import Point2D, clamp, polar_to_cartesian
from .models import DetectedObject


class RealTimeDetector:
    """Cluster LiDAR scan points into simple, real-time detected objects."""

    def __init__(self) -> None:
        self.last_objects: List[DetectedObject] = []
        self._object_counter = 0

    def process_points(self, points: List[dict[str, Any]]) -> Tuple[List[DetectedObject], dict[str, Any]]:
        """Filter raw points, cluster them, and create object descriptions."""
        start_time = time.perf_counter()
        filtered_points = self._filter_points(points)
        clusters = self._cluster_points(filtered_points)
        objects = self._build_objects(clusters)
        objects = self._stabilize_objects(objects)
        detection_time_ms = (time.perf_counter() - start_time) * 1000.0

        stats = {
            "total_scan_points": len(points),
            "filtered_points": len(filtered_points),
            "clusters_found": len(clusters),
            "objects_displayed": len(objects),
            "largest_cluster": max((len(cluster) for cluster in clusters), default=0),
            "smallest_cluster": min((len(cluster) for cluster in clusters), default=0),
            "average_cluster_size": round(sum(len(cluster) for cluster in clusters) / len(clusters), 2) if clusters else 0.0,
            "detection_time_ms": round(detection_time_ms, 2),
            "detection_fps": round(1000.0 / max(detection_time_ms, 1e-3), 2) if detection_time_ms > 0 else 0.0,
            "debug": DEBUG,
            "show_labels": ENABLE_LABELS,
            "show_bounding_circles": ENABLE_BOUNDING_CIRCLES,
            "show_connection_lines": ENABLE_CONNECTION_LINES,
            "show_statistics": ENABLE_STATISTICS,
            "show_debug_labels": ENABLE_DEBUG_LABELS,
        }
        self.last_objects = objects
        return objects, stats

    def _filter_points(self, points: List[dict[str, Any]]) -> List[Point2D]:
        filtered: List[Point2D] = []
        for point in points:
            try:
                angle = float(point["angle"])
                distance_mm = float(point["distance"])
                quality = int(point.get("quality", 0))
            except (KeyError, TypeError, ValueError):
                continue

            if distance_mm <= 0:
                continue
            if quality < 10:
                continue
            if distance_mm < MIN_DETECTION_DISTANCE_MM or distance_mm > MAX_DETECTION_DISTANCE_MM:
                continue
            if not math.isfinite(distance_mm) or not math.isfinite(angle):
                continue

            x_mm, y_mm = polar_to_cartesian(angle, distance_mm)
            filtered.append(Point2D(x_mm=x_mm, y_mm=y_mm, quality=quality, angle_deg=angle, distance_mm=distance_mm))

        return self._deduplicate_points(filtered)

    def _deduplicate_points(self, points: List[Point2D]) -> List[Point2D]:
        deduped: List[Point2D] = []
        seen: set[tuple[float, float]] = set()
        for point in points:
            key = (round(point.x_mm / 20.0), round(point.y_mm / 20.0))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(point)
        return deduped

    def _cluster_points(self, points: List[Point2D]) -> List[List[Point2D]]:
        if not points:
            return []

        coordinates = [[point.x_mm, point.y_mm] for point in points]
        clustering = DBSCAN(eps=DBSCAN_EPS_MM, min_samples=DBSCAN_MIN_SAMPLES).fit(coordinates)
        labels = clustering.labels_
        clusters: List[List[Point2D]] = []
        for label in sorted(set(labels)):
            if label == -1:
                continue
            members = [points[index] for index, cluster_label in enumerate(labels) if cluster_label == label]
            if MIN_CLUSTER_SIZE <= len(members) <= MAX_CLUSTER_SIZE:
                clusters.append(members)
        return clusters

    def _build_objects(self, clusters: List[List[Point2D]]) -> List[DetectedObject]:
        objects: List[DetectedObject] = []
        for cluster in clusters:
            if not cluster:
                continue
            center_x = sum(point.x_mm for point in cluster) / len(cluster)
            center_y = sum(point.y_mm for point in cluster) / len(cluster)
            distances = [math.hypot(point.x_mm, point.y_mm) for point in cluster]
            distance_mm = sum(distances) / len(distances)
            angle_deg = math.degrees(math.atan2(center_x, -center_y))
            if angle_deg < 0:
                angle_deg += 360.0

            width_mm = max(point.x_mm for point in cluster) - min(point.x_mm for point in cluster)
            height_mm = max(point.y_mm for point in cluster) - min(point.y_mm for point in cluster)
            radius_mm = max(math.hypot(point.x_mm - center_x, point.y_mm - center_y) for point in cluster)
            radius_mm = max(radius_mm, 50.0)
            density = len(cluster) / max(radius_mm * radius_mm * math.pi, 1.0)
            confidence = clamp(min(1.0, 0.2 + (len(cluster) / 80.0)), 0.0, 1.0)

            object_id = self._object_counter + 1
            self._object_counter = object_id
            objects.append(
                DetectedObject(
                    object_id=object_id,
                    points=cluster,
                    center_x_mm=center_x,
                    center_y_mm=center_y,
                    distance_mm=distance_mm,
                    angle_deg=angle_deg,
                    width_mm=width_mm,
                    height_mm=height_mm,
                    radius_mm=radius_mm,
                    point_count=len(cluster),
                    density=density,
                    confidence=confidence,
                )
            )

        objects.sort(key=lambda obj: obj.distance_mm)
        return objects

    def _stabilize_objects(self, objects: List[DetectedObject]) -> List[DetectedObject]:
        """Apply lightweight smoothing so nearby detections remain stable between frames."""
        if not objects:
            return []

        stabilized: List[DetectedObject] = []
        for current in objects:
            matched = None
            for previous in self.last_objects:
                distance = math.hypot(current.center_x_mm - previous.center_x_mm, current.center_y_mm - previous.center_y_mm)
                if distance <= max(250.0, previous.radius_mm + 150.0):
                    matched = previous
                    break

            if matched is not None:
                current.center_x_mm = (current.center_x_mm + matched.center_x_mm) / 2.0
                current.center_y_mm = (current.center_y_mm + matched.center_y_mm) / 2.0
                current.distance_mm = (current.distance_mm + matched.distance_mm) / 2.0
                current.angle_deg = (current.angle_deg + matched.angle_deg) / 2.0
                current.radius_mm = max(current.radius_mm, matched.radius_mm * 0.8)
            stabilized.append(current)

        return stabilized
