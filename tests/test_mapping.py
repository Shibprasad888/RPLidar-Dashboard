from lidar.mapping import OccupancyGridMap


def test_grid_updates_and_stats() -> None:
    grid = OccupancyGridMap(width_mm=4000, height_mm=4000, resolution_mm=200)
    points = [
        {"angle": 0.0, "distance": 1000.0, "quality": 100},
        {"angle": 90.0, "distance": 500.0, "quality": 100},
    ]

    stats = grid.update_from_scan(points)

    assert stats["updated_cells"] > 0
    assert stats["occupied_cells"] > 0
    assert stats["free_cells"] > 0
