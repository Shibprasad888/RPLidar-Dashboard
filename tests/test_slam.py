from lidar.mapping import OccupancyGridMap
from lidar.slam import LocalizationEstimator, Pose, SimpleScanMatcher, SLAMSystem


def test_scan_matcher_estimates_translation() -> None:
    matcher = SimpleScanMatcher()
    prev_scan = [
        {"angle": 0.0, "distance": 1000.0, "quality": 100},
        {"angle": 30.0, "distance": 1200.0, "quality": 100},
    ]
    curr_scan = [
        {"angle": 0.0, "distance": 1100.0, "quality": 100},
        {"angle": 30.0, "distance": 1250.0, "quality": 100},
    ]

    result = matcher.match(prev_scan, curr_scan)

    assert result.confidence > 0.0
    assert abs(result.translation_x_mm) > 0.0 or abs(result.translation_y_mm) > 0.0


def test_localization_estimator_updates_pose() -> None:
    estimator = LocalizationEstimator(smoothing=0.7)
    pose = Pose(x_mm=0.0, y_mm=0.0, heading_deg=0.0)
    updated = estimator.update(pose, 100.0, 50.0, 5.0)

    assert updated.x_mm > pose.x_mm
    assert updated.y_mm > pose.y_mm
    assert updated.heading_deg > pose.heading_deg


def test_slam_system_tracks_pose_and_keyframes() -> None:
    slam = SLAMSystem(OccupancyGridMap(width_mm=10000, height_mm=10000, resolution_mm=250))
    first_scan = [
        {"angle": 0.0, "distance": 1000.0, "quality": 100},
        {"angle": 45.0, "distance": 800.0, "quality": 100},
    ]
    second_scan = [
        {"angle": 0.0, "distance": 1100.0, "quality": 100},
        {"angle": 45.0, "distance": 850.0, "quality": 100},
    ]

    first_state = slam.update_from_scan(first_scan, timestamp=0.0)
    second_state = slam.update_from_scan(second_scan, timestamp=0.1)

    assert first_state["pose"]["x_mm"] == 0.0
    assert second_state["pose"]["x_mm"] != 0.0
    assert len(slam.pose_history) >= 2
    assert len(slam.keyframes) >= 1
