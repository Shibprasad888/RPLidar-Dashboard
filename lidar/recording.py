from __future__ import annotations

import csv
import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO

try:
    import cv2
except ImportError as exc:
    cv2 = None
    _opencv_import_error = exc


@dataclass(slots=True)
class RecordingSession:
    base_path: Path
    video_writer: Optional["cv2.VideoWriter"] = None
    lidar_log_file: Optional[TextIO] = None
    fusion_log_file: Optional[TextIO] = None
    event_log_file: Optional[TextIO] = None
    recording_lock: threading.Lock = field(default_factory=threading.Lock)
    paused: bool = False
    session_metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.session_metadata = {
            "started_at": time.time(),
            "recording_path": str(self.base_path.resolve()),
            "events": [],
            "recorded_scans": 0,
            "recorded_fusion": 0,
            "recorded_frames": 0,
        }

    def start_video(self, frame_size: tuple[int, int], fps: int) -> None:
        if cv2 is None:
            raise ImportError(
                "opencv-python is required for video recording. "
                "Install opencv-python or disable recording features."
            ) from _opencv_import_error
        video_path = self.base_path / "camera_recording.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.video_writer = cv2.VideoWriter(str(video_path), fourcc, fps, frame_size)
        self.log_event("video_started", {"path": str(video_path), "fps": fps, "frame_size": frame_size})

    def write_video_frame(self, frame: Any) -> None:
        with self.recording_lock:
            if self.paused or self.video_writer is None:
                return
            self.video_writer.write(frame)
            self.session_metadata["recorded_frames"] += 1

    def start_logging(self) -> None:
        lidar_path = self.base_path / "lidar_log.csv"
        fusion_path = self.base_path / "fusion_log.csv"
        events_path = self.base_path / "session_events.jsonl"
        self.lidar_log_file = open(lidar_path, "w", newline="", encoding="utf-8")
        self.fusion_log_file = open(fusion_path, "w", newline="", encoding="utf-8")
        self.event_log_file = open(events_path, "w", encoding="utf-8")
        lidar_writer = csv.writer(self.lidar_log_file)
        lidar_writer.writerow(["timestamp", "angle_deg", "distance_mm", "quality"])
        fusion_writer = csv.writer(self.fusion_log_file)
        fusion_writer.writerow([
            "timestamp",
            "track_id",
            "class_name",
            "confidence",
            "distance_m",
            "bearing_deg",
            "velocity_m_s",
            "status",
            "risk_level",
            "bounding_box",
        ])
        self.log_event("logging_started", {})

    def pause(self) -> None:
        with self.recording_lock:
            self.paused = True
            self.log_event("recording_paused", {})

    def resume(self) -> None:
        with self.recording_lock:
            self.paused = False
            self.log_event("recording_resumed", {})

    def log_event(self, event_type: str, payload: Dict[str, Any], timestamp: Optional[float] = None) -> None:
        if self.event_log_file is None or self.paused:
            return
        event_record = {
            "timestamp": timestamp if timestamp is not None else time.time(),
            "event_type": event_type,
            "payload": payload,
        }
        self.event_log_file.write(json.dumps(event_record) + "\n")
        self.event_log_file.flush()
        self.session_metadata["events"].append(event_record)

    def log_lidar_scan(self, scan_points: List[Dict[str, Any]], timestamp: float) -> None:
        if self.paused:
            return
        with self.recording_lock:
            if self.lidar_log_file is None:
                return
            writer = csv.writer(self.lidar_log_file)
            for point in scan_points:
                writer.writerow([timestamp, point.get("angle", 0.0), point.get("distance", 0.0), point.get("quality", 0)])
            self.session_metadata["recorded_scans"] += len(scan_points)
            self.log_event("lidar_scan", {"sample_count": len(scan_points), "timestamp": timestamp})

    def log_fusion_results(self, fused_objects: List[Any], timestamp: float) -> None:
        if self.paused:
            return
        with self.recording_lock:
            if self.fusion_log_file is None:
                return
            writer = csv.writer(self.fusion_log_file)
            for fused in fused_objects:
                writer.writerow([
                    timestamp,
                    fused.track_id,
                    fused.class_name,
                    fused.confidence,
                    fused.distance_m,
                    fused.bearing_deg,
                    fused.velocity_m_s,
                    fused.tracking_status,
                    fused.risk_level,
                    fused.bounding_box,
                ])
            self.session_metadata["recorded_fusion"] += len(fused_objects)
            self.log_event("fusion_results", {"object_count": len(fused_objects), "timestamp": timestamp})

    def log_slam_pose(self, pose: Dict[str, Any], timestamp: float) -> None:
        self.log_event("slam_pose", {"pose": pose, "timestamp": timestamp})

    def log_occupancy_map(self, occupancy_summary: Dict[str, Any], timestamp: float) -> None:
        self.log_event("occupancy_map", {"summary": occupancy_summary, "timestamp": timestamp})

    def log_navigation_state(self, state: Dict[str, Any], timestamp: float) -> None:
        self.log_event("navigation_state", {"state": state, "timestamp": timestamp})

    def log_tracking_results(self, tracked_objects: List[Dict[str, Any]], timestamp: float) -> None:
        self.log_event("tracking_results", {"tracked_object_count": len(tracked_objects), "timestamp": timestamp})

    def stop(self) -> None:
        with self.recording_lock:
            self.log_event("recording_stopped", {"duration": time.time() - self.session_metadata.get("started_at", time.time())})
            self.session_metadata["completed_at"] = time.time()
            self.session_metadata["duration_seconds"] = self.session_metadata["completed_at"] - self.session_metadata["started_at"]
            self.session_metadata["paused"] = self.paused

            metadata_path = self.base_path / "session_metadata.json"
            try:
                with metadata_path.open("w", encoding="utf-8") as handle:
                    json.dump(self.session_metadata, handle, indent=2)
            except Exception:
                pass

            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None
            if self.lidar_log_file is not None:
                self.lidar_log_file.close()
                self.lidar_log_file = None
            if self.fusion_log_file is not None:
                self.fusion_log_file.close()
                self.fusion_log_file = None
            if self.event_log_file is not None:
                self.event_log_file.close()
                self.event_log_file = None
