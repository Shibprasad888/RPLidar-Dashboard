import logging
import serial
import serial.tools.list_ports as list_ports
import sys
import threading
import time
from typing import Any, Callable, Optional

from lidar.parser import parse_scan_batch

logger = logging.getLogger("rplidar_phase1")

DEFAULT_MOTOR_PWM = 660
STOP_SCAN_COMMAND = bytes.fromhex("A525")
START_SCAN_COMMAND = bytes.fromhex("A520")
DESCRIPTOR_LENGTH = 7
SCAN_PACKET_LENGTH = 5
PACKET_LOG_INTERVAL_S = 1.0
SCAN_QUEUE_MAXSIZE = 10000

LIDAR_STATE_DISCONNECTED = "DISCONNECTED"
LIDAR_STATE_SEARCHING = "SEARCHING"
LIDAR_STATE_CONNECTING = "CONNECTING"
LIDAR_STATE_CONNECTED = "CONNECTED"
LIDAR_STATE_INITIALIZING = "INITIALIZING"
LIDAR_STATE_READY = "READY"
LIDAR_STATE_SCANNING = "SCANNING"
LIDAR_STATE_PAUSED = "PAUSED"
LIDAR_STATE_RECONNECTING = "RECONNECTING"
LIDAR_STATE_ERROR = "ERROR"
LIDAR_STATE_STOPPING = "STOPPING"

_HEALTH_STATUSES = {
    0: "Good",
    1: "Warning",
    2: "Error",
}


class LiDARException(Exception):
    pass


class LiDARTimeout(LiDARException):
    pass


class LiDARStopRequested(LiDARException):
    pass


class RPLiDARController:
    """High-level controller for the Slamtec RP LiDAR communication layer."""

    def __init__(self, port: Optional[str], baud_rate: int, timeout: float, debug: bool = False) -> None:
        self.port = port
        self.baud_rate = baud_rate
        self.timeout = timeout
        self.debug = debug
        self._serial_port: Optional[serial.Serial] = None
        self._driver: Optional[Any] = None
        self.device_info: dict[str, Any] = {
            "model": "Unknown",
            "firmware_version": "Unknown",
            "hardware_version": "Unknown",
            "serial_number": "Unknown",
            "health_status": "Unknown",
        }
        self.motor_running = False
        self._connected = False
        self._scanning = False
        self.state = LIDAR_STATE_DISCONNECTED
        self.state_reason: Optional[str] = None
        self._scan_stop_event = threading.Event()
        self._scan_completed = threading.Event()
        self._serial_lock = threading.RLock()
        self._packet_rate_last_log = time.monotonic()
        self._packet_rate_count = 0

    def detect_serial_ports(self) -> list[str]:
        """Return a list of detected serial ports."""
        try:
            ports = [port.device for port in list_ports.comports()]
        except Exception as exc:  # pragma: no cover - platform-specific
            logger.warning("Unable to list serial ports: %s", exc)
            return []
        return sorted(ports)

    def _looks_like_lidar_port(self, port_name: str) -> bool:
        lowered = port_name.lower()
        if any(token in lowered for token in ("bluetooth", "debug-console", "console")):
            return False
        if any(token in lowered for token in ("usbserial", "ttyusb", "tty.usb", "tty.usbserial", "ttyacm", "tty.usbmodem", "slab", "uart")):
            return True
        return False

    def _update_state(self, new_state: str, reason: Optional[str] = None) -> None:
        self.state = new_state
        self.state_reason = reason
        if reason:
            logger.info("LiDAR state=%s reason=%s", new_state, reason)
        else:
            logger.info("LiDAR state=%s", new_state)

    def _select_port(self) -> Optional[str]:
        ports = self.detect_serial_ports()
        if self.port:
            return self.port
        if not ports:
            return None
        lidar_ports = [port for port in ports if self._looks_like_lidar_port(port)]
        if lidar_ports:
            return sorted(lidar_ports)[0]
        if len(ports) == 1:
            return ports[0]
        return None

    def _open_serial_port(self, selected_port: str) -> serial.Serial:
        logger.info("Opening serial port %s", selected_port)
        serial_port = serial.Serial(
            port=selected_port,
            baudrate=self.baud_rate,
            timeout=0.1,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
        )
        return serial_port

    def _initialize_scan(self) -> None:
        assert self._serial_port is not None
        logger.info("Sending stop scan command A5 25")
        self._serial_port.write(STOP_SCAN_COMMAND)
        self._serial_port.flush()
        time.sleep(0.1)
        logger.info("Resetting input buffer")
        self._serial_port.reset_input_buffer()
        self._serial_port.reset_output_buffer()
        logger.info("Sending start scan command A5 20")
        self._serial_port.write(START_SCAN_COMMAND)
        self._serial_port.flush()
        time.sleep(0.1)
        descriptor = self._serial_port.read(DESCRIPTOR_LENGTH)
        logger.info("Descriptor received: %s", descriptor.hex())

    def connect(self) -> None:
        if self.is_connected():
            return

        self._update_state(LIDAR_STATE_SEARCHING, "Locating available serial port")
        selected_port = self._select_port()
        if not selected_port:
            self._update_state(LIDAR_STATE_ERROR, "No serial port detected")
            raise RuntimeError(
                "No RP LiDAR serial port detected. Connect the device or set SERIAL_PORT in config.py "
                "or RP_LIDAR_SERIAL_PORT in the environment and try again."
            )

        self._update_state(LIDAR_STATE_CONNECTING, f"Opening port {selected_port}")
        try:
            serial_port = self._open_serial_port(selected_port)
            with self._serial_lock:
                self._serial_port = serial_port
                self._initialize_scan()
                self._connected = True
                self.port = selected_port
                self._update_state(LIDAR_STATE_READY, "LiDAR connected")
        except Exception as exc:
            logger.warning("LiDAR connect failed: %s", exc)
            self._close_port()
            self._update_state(LIDAR_STATE_ERROR, str(exc))
            raise RuntimeError(f"Unable to open LiDAR serial port: {exc}") from exc

    def get_device_info(self) -> dict[str, Any]:
        if not self.is_connected():
            raise RuntimeError("LiDAR is not connected")
        return self.device_info.copy()

    def start_motor(self) -> None:
        if not self.is_connected():
            raise RuntimeError("LiDAR is not connected")
        if self.motor_running:
            logger.debug("Motor already running; skipping duplicate start")
            return
        self.motor_running = True
        self._update_state(LIDAR_STATE_READY, "Motor assumed started")

    def start_scan(self) -> None:
        if not self.is_connected():
            raise RuntimeError("LiDAR is not connected")
        if self._scanning:
            logger.debug("Scan already in progress; skipping duplicate start")
            return
        self._scan_stop_event.clear()
        self._scan_completed.clear()
        self._scanning = True
        self._update_state(LIDAR_STATE_SCANNING, "Scan stream started")

    def is_connected(self) -> bool:
        return self._connected or self._driver is not None

    def is_scanning(self) -> bool:
        return self._scanning and self._serial_port is not None

    def try_connect(self) -> bool:
        try:
            self.connect()
            return True
        except Exception as exc:
            logger.info("LiDAR reconnect attempt failed: %s", exc)
            return False

    def _close_port(self) -> None:
        with self._serial_lock:
            if self._serial_port is not None:
                try:
                    logger.info("Closing serial port %s", self.port)
                    self._serial_port.close()
                except Exception as exc:
                    logger.warning("Error closing serial port: %s", exc)
                finally:
                    self._serial_port = None
            self._connected = False
            self._scanning = False
            self.motor_running = False

    def _read_packet(self) -> Optional[bytes]:
        if self._driver is not None:
            try:
                packet = self._driver.grab_scan_data(max_nodes=SCAN_PACKET_LENGTH, timeout_ms=int(self.timeout * 1000))
            except Exception as exc:
                raise RuntimeError(str(exc)) from exc
            if packet is None:
                return None
            if isinstance(packet, (bytes, bytearray)):
                return bytes(packet)
            if isinstance(packet, tuple) and packet and isinstance(packet[0], (bytes, bytearray)):
                return bytes(packet[0])
            return None

        assert self._serial_port is not None
        buffer = bytearray()
        while len(buffer) < SCAN_PACKET_LENGTH and not self._scan_stop_event.is_set():
            chunk = self._serial_port.read(SCAN_PACKET_LENGTH - len(buffer))
            if not chunk:
                return None
            buffer.extend(chunk)
        if len(buffer) != SCAN_PACKET_LENGTH:
            return None
        return bytes(buffer)

    def _parse_packet(self, packet: bytes) -> Optional[tuple[int, float, float]]:
        if len(packet) != SCAN_PACKET_LENGTH:
            return None
        quality = packet[0] >> 2
        angle = ((packet[1] | (packet[2] << 8)) >> 1) / 64.0
        distance = (packet[3] | (packet[4] << 8)) / 4.0
        if distance <= 0:
            return None
        return quality, angle, distance

    def stream_scan_data(
        self,
        callback: Callable[[Optional[dict[str, Any]], int, int, int, float], None],
        start_time: float,
        max_samples: Optional[int] = None,
    ) -> None:
        if not self.is_connected():
            raise RuntimeError("LiDAR is not connected")
        if not self._scanning:
            raise RuntimeError("LiDAR is not scanning")
        if self._serial_port is None and self._driver is None:
            raise RuntimeError("LiDAR serial port not initialized")

        total_samples = 0
        valid_samples = 0
        invalid_samples = 0
        self._scan_completed.clear()
        last_rate_log = time.monotonic()
        packet_count = 0

        try:
            while self._scanning and not self._scan_stop_event.is_set():
                if max_samples is not None and total_samples >= max_samples:
                    break
                with self._serial_lock:
                    packet = self._read_packet()
                if packet is None:
                    continue
                packet_count += 1
                total_samples += 1
                parsed = self._parse_packet(packet)
                if parsed is None:
                    invalid_samples += 1
                    callback(None, total_samples, valid_samples, invalid_samples, start_time)
                    continue
                quality, angle, distance = parsed
                point = parse_scan_batch([(quality, angle, distance)])
                if point:
                    valid_samples += 1
                    callback(point[0], total_samples, valid_samples, invalid_samples, start_time)
                else:
                    invalid_samples += 1
                    callback(None, total_samples, valid_samples, invalid_samples, start_time)

                current_time = time.monotonic()
                if current_time - last_rate_log >= PACKET_LOG_INTERVAL_S:
                    logger.info("Packets received per second: %d", packet_count)
                    packet_count = 0
                    last_rate_log = current_time
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            self._scanning = False
            self._update_state(LIDAR_STATE_ERROR, str(exc))
            raise RuntimeError(f"Scan stream interrupted: {exc}") from exc
        finally:
            self._scanning = False
            self._scan_completed.set()

    def stop_scan(self) -> None:
        if not self._scanning:
            return
        self._scan_stop_event.set()
        self._scan_completed.wait(timeout=2.0)
        self._scanning = False
        self._update_state(LIDAR_STATE_PAUSED, "Scan stopped")

    def stop_motor(self) -> None:
        if not self.motor_running:
            return
        self.motor_running = False
        self._update_state(LIDAR_STATE_READY, "Motor stopped")

    def _send_stop_command(self) -> None:
        if self._serial_port is None:
            return
        try:
            logger.info("Sending stop scan command A5 25")
            self._serial_port.write(STOP_SCAN_COMMAND)
            self._serial_port.flush()
        except Exception as exc:
            logger.warning("Unable to send stop scan command: %s", exc)

    def disconnect(self) -> None:
        if not self._connected and self._serial_port is None:
            return
        self._update_state(LIDAR_STATE_STOPPING, "Disconnecting LiDAR")
        self.stop_scan()
        self.stop_motor()
        with self._serial_lock:
            self._send_stop_command()
            self._close_port()
        self._update_state(LIDAR_STATE_DISCONNECTED, "LiDAR disconnected")


if __name__ == "__main__":
    print("This module is part of the lidar package. Run the application from the project root:")
    print("  python main.py")
    sys.exit(1)
