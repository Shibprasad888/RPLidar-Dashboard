import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lidar.motion import MotionAnalyzer
from lidar.safety import SafetyAnalyzer
from lidar.models import TrackedObject


def make_track(track_id: int, x: float, y: float, velocity: float, heading: float) -> TrackedObject:
    return TrackedObject(
        track_id=track_id,
        center_x_mm=x,
        center_y_mm=y,
        prev_center_x_mm=x - 100.0,
        prev_center_y_mm=y,
        distance_mm=abs(y),
        angle_deg=0.0,
        velocity_m_s=velocity,
        heading_deg=heading,
        age_frames=5,
        last_seen_ts=1.0,
        frames_tracked=5,
        frames_missing=0,
        confidence=0.8,
        status="Active",
        radius_mm=200.0,
        point_count=10,
        density=0.1,
        cluster_confidence=0.8,
        trail=[]
    )


def test_motion_and_safety_analysis() -> None:
    analyzer = MotionAnalyzer()
    safety = SafetyAnalyzer()
    track = make_track(1, 1000.0, 1000.0, 1.2, 45.0)

    motion_metrics = analyzer.analyze(track, 1.0)
    assert motion_metrics["velocity_m_s"] > 0
    assert motion_metrics["acceleration_m_s2"] >= 0

    zone = safety.get_zone_for_distance(1200.0)
    assert zone["name"] == "Warning"

    risk = safety.analyze_track(track, motion_metrics, 1000.0, 0.5)
    assert risk["risk_level"] in {"SAFE", "LOW", "MEDIUM", "HIGH", "CRITICAL"}
