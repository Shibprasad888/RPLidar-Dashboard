from lidar.navigation import NavigationStatus


def test_navigation_status_initializes_event_log() -> None:
    status = NavigationStatus()

    assert status.event_log.maxlen == 40
    assert status.state.name == "IDLE"
