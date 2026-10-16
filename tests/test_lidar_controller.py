from lidar.lidar import RPLiDARController


class DummySerialPort:
    def __init__(self) -> None:
        self.is_open = True

    def read(self, size: int) -> bytes:
        raise RuntimeError("Descriptor length mismatch")

    def close(self) -> None:
        self.is_open = False

    def reset_input_buffer(self) -> None:
        pass


def test_select_port_prefers_lidar_like_serial_port() -> None:
    controller = RPLiDARController(port=None, baud_rate=115200, timeout=2.0, debug=False)

    ports = [
        "/dev/cu.Bluetooth-Incoming-Port",
        "/dev/cu.OPPOEncoAir3Pro",
        "/dev/cu.debug-console",
        "/dev/cu.usbserial-1140",
    ]

    controller.detect_serial_ports = lambda: ports  # type: ignore[assignment]

    assert controller._select_port() == "/dev/cu.usbserial-1140"


def test_stream_scan_data_stops_cleanly_on_scan_error() -> None:
    controller = RPLiDARController(port="/dev/cu.usbserial-1140", baud_rate=115200, timeout=2.0, debug=False)

    class DummyDriver:
        def grab_scan_data(self, max_nodes: int, timeout_ms: int):
            raise RuntimeError("Descriptor length mismatch")

    controller._driver = DummyDriver()  # type: ignore[assignment]
    controller._connected = True
    controller._scanning = True
    controller._scan_stop_event.clear()

    try:
        controller.stream_scan_data(callback=lambda *args: None, start_time=0.0)
    except RuntimeError as exc:
        assert "Descriptor length mismatch" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError for scan stream failure")
