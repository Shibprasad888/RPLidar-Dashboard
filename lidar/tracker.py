import math
import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

from config import (
    ENABLE_DEBUG_TRACKING,
    ENABLE_MOTION_TRAILS,
    ENABLE_PREDICTION,
    ENABLE_VELOCITY_ARROWS,
    MAX_ASSOCIATION_DISTANCE_MM,
    MAX_LOST_FRAMES,
    MIN_CONFIDENCE,
    TRAIL_LENGTH,
    VELOCITY_SMOOTHING_FACTOR,
)
from .models import DetectedObject, ObjectStatus, TrackedObject
from .motion import MotionAnalyzer
from .safety import SafetyAnalyzer


class ObjectTracker:
    """Persistent centroid tracker for LiDAR-detected objects."""

    def __init__(self) -> None:
        self.tracks: Dict[int, TrackedObject] = {}
        self._next_id = 1
        self._motion_analyzer = MotionAnalyzer()
        self._safety_analyzer = SafetyAnalyzer()
        self._stats: Dict[str, Any] = {
            "tracked_objects": 0,
            "new_objects": 0,
            "recovered_objects": 0,
            "lost_objects": 0,
            "deleted_objects": 0,
            "average_tracking_age": 0.0,
            "association_time_ms": 0.0,
            "tracking_fps": 0.0,
            "average_velocity": 0.0,
            "maximum_velocity": 0.0,
        }

    def update(self, detections: List[DetectedObject], timestamp: float) -> Tuple[List[TrackedObject], Dict[str, Any]]:
        start_time = time.perf_counter()
        self._predict_tracks(timestamp)
        matches, unmatched_tracks, unmatched_detections = self._associate(detections)

        for track_id, detection in matches.items():
            track = self.tracks[track_id]
            track.update_from_detection(detection, timestamp, VELOCITY_SMOOTHING_FACTOR, TRAIL_LENGTH)
            track.status = ObjectStatus.ACTIVE.value

        for track_id in unmatched_tracks:
            track = self.tracks[track_id]
            track.mark_missed(timestamp)
            if track.frames_missing == 1:
                self._stats["lost_objects"] += 1

        for detection in unmatched_detections:
            if detection.confidence < MIN_CONFIDENCE:
                continue
            self.tracks[self._next_id] = TrackedObject(
                track_id=self._next_id,
                center_x_mm=detection.center_x_mm,
                center_y_mm=detection.center_y_mm,
                prev_center_x_mm=detection.center_x_mm,
                prev_center_y_mm=detection.center_y_mm,
                distance_mm=detection.distance_mm,
                angle_deg=detection.angle_deg,
                velocity_m_s=0.0,
                heading_deg=detection.angle_deg,
                age_frames=1,
                last_seen_ts=timestamp,
                frames_tracked=1,
                frames_missing=0,
                confidence=detection.confidence,
                status=ObjectStatus.ACTIVE.value,
                radius_mm=detection.radius_mm,
                point_count=detection.point_count,
                density=detection.density,
                cluster_confidence=detection.confidence,
                trail=self._build_initial_trail(detection, TRAIL_LENGTH),
            )
            self._stats["new_objects"] += 1
            self._next_id += 1

        self._analyze_motion_and_safety(timestamp)
        self._prune_tracks(timestamp)
        self._update_stats(timestamp)
        self._stats["association_time_ms"] = round((time.perf_counter() - start_time) * 1000.0, 2)
        return list(self.tracks.values()), self._stats

    def _predict_tracks(self, timestamp: float) -> None:
        if not ENABLE_PREDICTION:
            return
        for track in self.tracks.values():
            if track.frames_missing == 0:
                continue
            track.predict_position(timestamp)

    def _associate(self, detections: List[DetectedObject]) -> Tuple[Dict[int, DetectedObject], List[int], List[DetectedObject]]:
        matches: Dict[int, DetectedObject] = {}
        unmatched_tracks = list(self.tracks.keys())
        unmatched_detections = detections.copy()

        if not detections or not self.tracks:
            return matches, unmatched_tracks, unmatched_detections

        association_candidates: List[Tuple[int, int, float]] = []
        for track_id, track in self.tracks.items():
            track_x = track.predicted_x_mm if ENABLE_PREDICTION and track.predicted_x_mm is not None else track.center_x_mm
            track_y = track.predicted_y_mm if ENABLE_PREDICTION and track.predicted_y_mm is not None else track.center_y_mm
            for detection in detections:
                dx = detection.center_x_mm - track_x
                dy = detection.center_y_mm - track_y
                distance = math.hypot(dx, dy)
                if distance <= MAX_ASSOCIATION_DISTANCE_MM:
                    association_candidates.append((track_id, detection.object_id, distance))

        association_candidates.sort(key=lambda entry: entry[2])
        used_tracks = set()
        used_detections = set()

        for track_id, detection_id, distance in association_candidates:
            if track_id in used_tracks or detection_id in used_detections:
                continue
            detection = next((d for d in detections if d.object_id == detection_id), None)
            if detection is None:
                continue
            matches[track_id] = detection
            used_tracks.add(track_id)
            used_detections.add(detection_id)

        unmatched_tracks = [track_id for track_id in self.tracks if track_id not in used_tracks]
        unmatched_detections = [d for d in detections if d.object_id not in used_detections]
        return matches, unmatched_tracks, unmatched_detections

    def _analyze_motion_and_safety(self, timestamp: float) -> None:
        for track in self.tracks.values():
            motion = self._motion_analyzer.analyze(track, timestamp)
            detection_distance_mm = max(track.distance_mm, 0.0)
            safety = self._safety_analyzer.analyze_track(track, motion, detection_distance_mm, max(timestamp - track.last_seen_ts, 1e-3))
            track.acceleration_m_s2 = motion.get("acceleration_m_s2", 0.0)
            track.angular_velocity_deg_s = motion.get("angular_velocity_deg_s", 0.0)
            track.direction = motion.get("direction", "Unknown")
            track.current_zone = safety["zone"]["name"]
            track.risk_level = safety["risk_level"]
            track.collision_confidence = safety["collision_confidence"]
            track.ttc_s = safety["ttc_s"]
            track.motion_debug = {**motion, **safety}

    def _prune_tracks(self, timestamp: float) -> None:
        deleted_ids = []
        for track_id, track in list(self.tracks.items()):
            if track.frames_missing > MAX_LOST_FRAMES:
                deleted_ids.append(track_id)
                del self.tracks[track_id]

        self._stats["deleted_objects"] += len(deleted_ids)

    def _update_stats(self, timestamp: float) -> None:
        if not self.tracks:
            self._stats["tracked_objects"] = 0
            self._stats["average_tracking_age"] = 0.0
            self._stats["average_velocity"] = 0.0
            self._stats["maximum_velocity"] = 0.0
            return

        active_tracks = list(self.tracks.values())
        self._stats["tracked_objects"] = len(active_tracks)
        self._stats["average_tracking_age"] = round(sum(track.age_frames for track in active_tracks) / len(active_tracks), 2)
        self._stats["average_velocity"] = round(sum(track.velocity_m_s for track in active_tracks) / len(active_tracks), 3)
        self._stats["maximum_velocity"] = round(max(track.velocity_m_s for track in active_tracks), 3)

    def _build_initial_trail(self, detection: DetectedObject, length: int) -> any:
        trail = []
        for _ in range(length):
            trail.append((detection.center_x_mm, detection.center_y_mm))
        return deque(trail, maxlen=length)
