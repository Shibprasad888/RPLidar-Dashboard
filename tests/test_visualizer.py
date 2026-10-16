from lidar.models import TrackedObject
from lidar.visualizer import RPLiDARVisualizer


def test_object_summary_is_human_readable() -> None:
    obj = TrackedObject(
        track_id=7,
        center_x_mm=1000.0,
        center_y_mm=0.0,
        prev_center_x_mm=0.0,
        prev_center_y_mm=0.0,
        distance_mm=1500.0,
        angle_deg=0.0,
        velocity_m_s=0.2,
        heading_deg=0.0,
        age_frames=3,
        last_seen_ts=1.0,
        frames_tracked=3,
        frames_missing=0,
        confidence=0.92,
        status="Active",
        radius_mm=250.0,
        point_count=8,
        density=0.6,
        cluster_confidence=0.9,
        current_zone="Warning",
        risk_level="HIGH",
    )

    visualizer = RPLiDARVisualizer()
    summary = visualizer._format_object_summary(obj)

    assert "1.5 m" in summary
    assert "ahead" in summary
    assert "moving" in summary
    assert "high" in summary
