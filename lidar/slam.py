from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

from collections import deque

from config import (
    ENABLE_LOCALIZATION,
    ENABLE_POSE_HISTORY,
    ENABLE_SCAN_MATCHING,
    ENABLE_SLAM,
    LOCALIZATION_SMOOTHING,
    MAX_KEYFRAMES,
    MAX_POSE_HISTORY,
    MAX_POSE_ERROR,
    MIN_SCAN_OVERLAP,
    SCAN_MATCH_THRESHOLD,
    TRAJECTORY_LENGTH,
    ENABLE_TRAJECTORY,
)
from .geometry import polar_to_cartesian
from .mapping import OccupancyGridMap


@dataclass(slots=True)
class Pose:
    x_mm: float = 0.0
    y_mm: float = 0.0
    heading_deg: float = 0.0


@dataclass(slots=True)
class ScanMatchResult:
    translation_x_mm: float = 0.0
    translation_y_mm: float = 0.0
    rotation_deg: float = 0.0
    confidence: float = 0.0
    error: float = 0.0
    matched_points: int = 0


class SimpleScanMatcher:
    """A lightweight scan matcher that estimates a rigid-body transform between scans."""

    def match(self, prev_scan: List[Dict[str, Any]], curr_scan: List[Dict[str, Any]]) -> ScanMatchResult:
        if not prev_scan or not curr_scan:
            return ScanMatchResult(confidence=0.0, error=float("inf"))

        prev_points = self._extract_points(prev_scan)
        curr_points = self._extract_points(curr_scan)
        if not prev_points or not curr_points:
            return ScanMatchResult(confidence=0.0, error=float("inf"))

        matched = 0
        tx_total = 0.0
        ty_total = 0.0
        rot_total = 0.0
        error_total = 0.0
        for prev_point in prev_points[:12]:
            for curr_point in curr_points[:12]:
                dx = curr_point[0] - prev_point[0]
                dy = curr_point[1] - prev_point[1]
                dist = math.hypot(dx, dy)
                if dist < 250.0:
                    tx_total += dx
                    ty_total += dy
                    rot_total += 0.0
                    error_total += dist
                    matched += 1
                    break
        if matched == 0:
            return ScanMatchResult(confidence=0.0, error=float("inf"))

        tx = tx_total / matched
        ty = ty_total / matched
        error = error_total / matched
        confidence = max(0.0, min(1.0, 1.0 - (error / 1000.0)))
        return ScanMatchResult(
            translation_x_mm=tx,
            translation_y_mm=ty,
            rotation_deg=rot_total / matched,
            confidence=confidence,
            error=error,
            matched_points=matched,
        )

    def _extract_points(self, scan_points: List[Dict[str, Any]]) -> List[Tuple[float, float]]:
        points: List[Tuple[float, float]] = []
        for point in scan_points:
            try:
                angle = float(point["angle"])
                distance = float(point["distance"])
            except (KeyError, TypeError, ValueError):
                continue
            if distance <= 0.0 or not math.isfinite(distance):
                continue
            x_mm, y_mm = polar_to_cartesian(angle, distance)
            points.append((x_mm, y_mm))
        return points


class LocalizationEstimator:
    """Estimate a smooth robot pose from odometry-like scan matching updates."""

    def __init__(self, smoothing: float = LOCALIZATION_SMOOTHING) -> None:
        self.smoothing = max(0.0, min(1.0, smoothing))

    def update(self, previous_pose: Pose, delta_x_mm: float, delta_y_mm: float, delta_heading_deg: float) -> Pose:
        smoothed_x = previous_pose.x_mm + delta_x_mm * self.smoothing
        smoothed_y = previous_pose.y_mm + delta_y_mm * self.smoothing
        smoothed_heading = previous_pose.heading_deg + delta_heading_deg * self.smoothing
        return Pose(x_mm=smoothed_x, y_mm=smoothed_y, heading_deg=smoothed_heading)


class SLAMSystem:
    """A modular SLAM state holder that combines mapping, localization, and scan matching."""

    def __init__(self, occupancy_map: Optional[OccupancyGridMap] = None) -> None:
        self.occupancy_map = occupancy_map or OccupancyGridMap()
        self.pose = Pose()
        self.pose_history: Deque[Pose] = deque(maxlen=MAX_POSE_HISTORY)
        self.keyframes: List[Dict[str, Any]] = []
        self.scan_history: List[List[Dict[str, Any]]] = []
        self.timestamp_history: List[float] = []
        self.pose_graph_nodes: List[Dict[str, Any]] = []
        self.scan_matcher = SimpleScanMatcher()
        self.localization = LocalizationEstimator()
        self.last_scan_match_result: Optional[ScanMatchResult] = None
        self.last_localization_error = 0.0
        self.last_mapping_time_s = 0.0
        self.last_matching_time_s = 0.0
        self.last_localization_time_s = 0.0
        self.last_fps = 0.0
        self.keyframe_counter = 0
        self.trajectory: List[Pose] = []
        self.last_timestamp: Optional[float] = None

    def update_from_scan(self, scan_points: List[Dict[str, Any]], timestamp: Optional[float] = None) -> Dict[str, Any]:
        if not ENABLE_SLAM:
            return self._build_stats()

        start_time = time.perf_counter()
        prev_pose = Pose(x_mm=self.pose.x_mm, y_mm=self.pose.y_mm, heading_deg=self.pose.heading_deg)
        prev_scan = self.scan_history[-1] if self.scan_history else None
        match_result = ScanMatchResult(confidence=0.0, error=float("inf"))
        if prev_scan is not None and ENABLE_SCAN_MATCHING:
            match_start = time.perf_counter()
            match_result = self.scan_matcher.match(prev_scan, scan_points)
            self.last_matching_time_s = time.perf_counter() - match_start
            self.last_scan_match_result = match_result

        if ENABLE_LOCALIZATION and match_result.confidence >= SCAN_MATCH_THRESHOLD:
            localization_start = time.perf_counter()
            delta_x = match_result.translation_x_mm
            delta_y = match_result.translation_y_mm
            delta_heading = match_result.rotation_deg
            self.pose = self.localization.update(prev_pose, delta_x, delta_y, delta_heading)
            self.last_localization_time_s = time.perf_counter() - localization_start
            self.last_localization_error = match_result.error
        else:
            self.pose = prev_pose
            self.last_localization_error = match_result.error if match_result.error != float("inf") else 0.0

        self.occupancy_map.update_from_scan(scan_points, tracked_objects=None, timestamp=timestamp or time.monotonic())
        self.last_mapping_time_s = time.perf_counter() - start_time
        self.last_fps = 1.0 / max(self.last_mapping_time_s, 1e-3)

        self._record_pose_history()
        self._maybe_add_keyframe(scan_points, timestamp)
        self._record_scan(scan_points, timestamp)

        return self._build_stats()

    def clear_trajectory(self) -> None:
        self.trajectory.clear()
        self.pose_history.clear()

    def _record_pose_history(self) -> None:
        if not ENABLE_POSE_HISTORY:
            return
        self.pose_history.append(Pose(x_mm=self.pose.x_mm, y_mm=self.pose.y_mm, heading_deg=self.pose.heading_deg))
        if ENABLE_TRAJECTORY:
            self.trajectory.append(Pose(x_mm=self.pose.x_mm, y_mm=self.pose.y_mm, heading_deg=self.pose.heading_deg))
            if len(self.trajectory) > TRAJECTORY_LENGTH:
                self.trajectory = self.trajectory[-TRAJECTORY_LENGTH:]

    def _maybe_add_keyframe(self, scan_points: List[Dict[str, Any]], timestamp: Optional[float]) -> None:
        if len(self.keyframes) >= MAX_KEYFRAMES:
            return
        if self.last_scan_match_result is None or self.last_scan_match_result.confidence < MIN_SCAN_OVERLAP:
            return
        if self.last_scan_match_result.error > MAX_POSE_ERROR:
            return
        self.keyframes.append({
            "timestamp": timestamp or time.monotonic(),
            "pose": Pose(x_mm=self.pose.x_mm, y_mm=self.pose.y_mm, heading_deg=self.pose.heading_deg),
            "scan": list(scan_points),
        })
        self.keyframe_counter += 1

    def _record_scan(self, scan_points: List[Dict[str, Any]], timestamp: Optional[float]) -> None:
        self.scan_history.append(list(scan_points))
        if len(self.scan_history) > MAX_POSE_HISTORY:
            self.scan_history = self.scan_history[-MAX_POSE_HISTORY:]
        self.timestamp_history.append(timestamp or time.monotonic())
        if len(self.timestamp_history) > MAX_POSE_HISTORY:
            self.timestamp_history = self.timestamp_history[-MAX_POSE_HISTORY:]

    def _build_stats(self) -> Dict[str, Any]:
        return {
            "pose": {
                "x_mm": round(self.pose.x_mm, 3),
                "y_mm": round(self.pose.y_mm, 3),
                "heading_deg": round(self.pose.heading_deg, 3),
            },
            "localization_confidence": round(self.last_scan_match_result.confidence if self.last_scan_match_result else 0.0, 3),
            "scan_match_error": round(self.last_scan_match_result.error if self.last_scan_match_result else float("inf"), 3),
            "trajectory_length": len(self.trajectory),
            "keyframes": len(self.keyframes),
            "pose_graph_nodes": len(self.pose_graph_nodes),
            "slam_fps": round(self.last_fps, 2),
            "mapping_fps": round(self.occupancy_map._mapping_fps, 2),
            "matching_time_ms": round(self.last_matching_time_s * 1000.0, 2),
            "localization_time_ms": round(self.last_localization_time_s * 1000.0, 2),
            "mapping_time_ms": round(self.last_mapping_time_s * 1000.0, 2),
            "loop_closure_candidates": 0,
        }
