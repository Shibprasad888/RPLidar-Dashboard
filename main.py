import logging
import platform
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

from config import (
    BAUD_RATE,
    CALIBRATION_FILE,
    CONNECTION_TIMEOUT,
    DEBUG,
    ENABLE_AUTO_RECOVERY,
    ENABLE_SAMPLE_PRINT,
    ENABLE_TERMINAL_OBJECT_SUMMARY,
    INITIAL_ZOOM,
    LIDAR_RECONNECT_INTERVAL_S,
    MAX_ASSOCIATION_DISTANCE_MM,
    MAX_POINT_QUEUE_SIZE,
    MAX_SCAN_DISTANCE_MM,
    OBJECT_SUMMARY_PRINT_INTERVAL_S,
    PLANNER_TYPE,
    SCAN_DOWNSAMPLE_RATE,
    SHOW_RAW_POINTS,
    SERIAL_PORT,
    TRAIL_LENGTH,
    WINDOW_HEIGHT,
    WINDOW_WIDTH,
)
from lidar.detector import RealTimeDetector
from lidar.lidar import RPLiDARController
from lidar.mapping import OccupancyGridMap
from lidar.navigation import NavigationEngine, NavigationRequest
from lidar.platform import ProductionPlatform
from lidar.planner import parse_planner_type
from lidar.slam import SLAMSystem
from lidar.models import TrackedObject
from lidar.tracker import ObjectTracker
from lidar.visualizer import RPLiDARVisualizer

logging.basicConfig(level=logging.INFO if DEBUG else logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger("rplidar_phase1")


def print_banner() -> None:
    print("=" * 50)
    print("Slamtec RP LiDAR Production Platform")
    print("Phase 10 - Production Robotics Dashboard")
    print("=" * 50)


def print_version_info() -> None:
    print("Python Version:")
    print(platform.python_version())
    print("Library Version:")
    for package in ("pyserial", "numpy", "sklearn", "pygame"):
        try:
            from importlib.metadata import version

            print(f"- {package}: {version(package)}")
        except Exception:
            print(f"- {package}: not installed")


def print_detected_ports(controller: RPLiDARController) -> None:
    ports = controller.detect_serial_ports()
    print("Detected Serial Ports:")
    if ports:
        for port in ports:
            print(f"- {port}")
    else:
        print("- None detected")


def _describe_heading(heading_deg: float, velocity_m_s: float) -> str:
    if velocity_m_s <= 0.05:
        return "still"
    directions = [
        "N",
        "NE",
        "E",
        "SE",
        "S",
        "SW",
        "W",
        "NW",
    ]
    index = int(((heading_deg + 22.5) % 360) / 45)
    return directions[index]


def _format_terminal_object_summary(obj: "TrackedObject") -> str:
    distance_m = obj.distance_mm / 1000.0
    speed = obj.velocity_m_s
    direction = _describe_heading(obj.heading_deg, speed)
    return f"#{obj.track_id}:{distance_m:.1f}m {speed:.2f}m/s {direction} {obj.risk_level}"


def _print_terminal_objects(
    tracked: list["TrackedObject"],
    current_time: float,
    last_summary_time: list[float],
) -> None:
    if not ENABLE_TERMINAL_OBJECT_SUMMARY:
        return
    if current_time - last_summary_time[0] < OBJECT_SUMMARY_PRINT_INTERVAL_S:
        return
    last_summary_time[0] = current_time
    if not tracked:
        print("[Objects] No objects detected.")
        return
    summary = " | ".join(_format_terminal_object_summary(obj) for obj in tracked[:6])
    overflow = "..." if len(tracked) > 6 else ""
    print(f"[Objects:{len(tracked)}] {summary} {overflow}")


def _build_object_log_line(obj: TrackedObject) -> str:
    distance_m = obj.distance_mm / 1000.0
    direction = _describe_heading(obj.heading_deg, obj.velocity_m_s)
    return (
        f"object#{obj.track_id} distance={distance_m:.2f}m speed={obj.velocity_m_s:.2f}m/s "
        f"heading={direction} risk={obj.risk_level} zone={obj.current_zone}"
    )


def _write_lastlog_entry(log_path: Path, message: str) -> None:
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")


def initialize_lastlog(log_path: Path) -> Path:
    project_root = Path(__file__).resolve().parent
    resolved_path = (project_root / log_path).resolve() if not log_path.is_absolute() else log_path.resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_path.write_text("LiDAR run log\n", encoding="utf-8")
    _write_lastlog_entry(resolved_path, f"started={time.strftime('%Y-%m-%d %H:%M:%S')}")
    _write_lastlog_entry(resolved_path, f"port={SERIAL_PORT or 'auto'}")
    return resolved_path


def _log_detected_objects(log_path: Path, tracked: list[TrackedObject], timestamp: float) -> None:
    if not tracked:
        _write_lastlog_entry(log_path, f"{timestamp:.2f}s detected=0 objects")
        return
    _write_lastlog_entry(log_path, f"{timestamp:.2f}s detected={len(tracked)} objects")
    for obj in tracked:
        _write_lastlog_entry(log_path, f"  {_build_object_log_line(obj)}")


def run_detection_loop(
    point_queue: "queue.Queue[Optional[dict[str, Any]]]",
    detector: RealTimeDetector,
    tracker: "ObjectTracker",
    visualizer: RPLiDARVisualizer,
    mapping: OccupancyGridMap,
    slam: SLAMSystem,
    navigation_engine: NavigationEngine,
    stop_event: threading.Event,
    last_summary_time: list[float],
    lastlog_path: Path,
) -> None:
    """Run detection and tracking asynchronously from a queue of filtered scan points."""
    batch: list[dict[str, Any]] = []
    while not stop_event.is_set():
        try:
            item = point_queue.get(timeout=0.05)
        except queue.Empty:
            if batch:
                detections, detection_stats = detector.process_points(batch)
                tracked, track_stats = tracker.update(detections, time.monotonic())
                _print_terminal_objects(tracked, time.monotonic(), last_summary_time)
                motion_stats = {
                    "tracked_objects": len(tracked),
                    "moving_objects": sum(1 for item in tracked if item.velocity_m_s > 0.05),
                    "stationary_objects": sum(1 for item in tracked if item.velocity_m_s <= 0.05),
                    "critical_zone_objects": sum(1 for item in tracked if item.current_zone == "Critical"),
                    "warning_zone_objects": sum(1 for item in tracked if item.current_zone == "Warning"),
                    "average_velocity": round(sum(item.velocity_m_s for item in tracked) / len(tracked), 3) if tracked else 0.0,
                    "maximum_velocity": round(max((item.velocity_m_s for item in tracked), default=0.0), 3),
                    "predicted_collisions": sum(1 for item in tracked if item.risk_level in {"HIGH", "CRITICAL"}),
                    "average_ttc_s": round(sum(item.ttc_s for item in tracked if item.ttc_s != float("inf")) / max(sum(1 for item in tracked if item.ttc_s != float("inf")), 1), 2) if tracked else float("inf"),
                }
                map_stats = mapping.update_from_scan(batch, tracked, time.monotonic())
                slam_stats = slam.update_from_scan(batch, time.monotonic())
                if navigation_engine is not None:
                    navigation_engine.update_dynamic_obstacles(tracked_objects=tracked)
                _log_detected_objects(lastlog_path, tracked, time.monotonic())
                visualizer.push_tracked_objects(tracked, {**detection_stats, **track_stats, **motion_stats, **map_stats, **slam_stats})
                visualizer.push_map(mapping, {**map_stats, **slam_stats})
                batch = []
            continue

        if item is None:
            break

        batch.append(item)
        if len(batch) >= 30:
            detections, detection_stats = detector.process_points(batch)
            tracked, track_stats = tracker.update(detections, time.monotonic())
            _print_terminal_objects(tracked, time.monotonic(), last_summary_time)
            motion_stats = {
                "tracked_objects": len(tracked),
                "moving_objects": sum(1 for item in tracked if item.velocity_m_s > 0.05),
                "stationary_objects": sum(1 for item in tracked if item.velocity_m_s <= 0.05),
                "critical_zone_objects": sum(1 for item in tracked if item.current_zone == "Critical"),
                "warning_zone_objects": sum(1 for item in tracked if item.current_zone == "Warning"),
                "average_velocity": round(sum(item.velocity_m_s for item in tracked) / len(tracked), 3) if tracked else 0.0,
                "maximum_velocity": round(max((item.velocity_m_s for item in tracked), default=0.0), 3),
                "predicted_collisions": sum(1 for item in tracked if item.risk_level in {"HIGH", "CRITICAL"}),
                "average_ttc_s": round(sum(item.ttc_s for item in tracked if item.ttc_s != float("inf")) / max(sum(1 for item in tracked if item.ttc_s != float("inf")), 1), 2) if tracked else float("inf"),
            }
            map_stats = mapping.update_from_scan(batch, tracked, time.monotonic())
            slam_stats = slam.update_from_scan(batch, time.monotonic())
            _log_detected_objects(lastlog_path, tracked, time.monotonic())
            visualizer.push_tracked_objects(tracked, {**detection_stats, **track_stats, **motion_stats, **map_stats, **slam_stats})
            visualizer.push_map(mapping, {**map_stats, **slam_stats})
            batch = []

    if batch:
        detections, detection_stats = detector.process_points(batch)
        tracked, track_stats = tracker.update(detections, time.monotonic())
        _print_terminal_objects(tracked, time.monotonic(), last_summary_time)
        motion_stats = {
            "tracked_objects": len(tracked),
            "moving_objects": sum(1 for item in tracked if item.velocity_m_s > 0.05),
            "stationary_objects": sum(1 for item in tracked if item.velocity_m_s <= 0.05),
            "critical_zone_objects": sum(1 for item in tracked if item.current_zone == "Critical"),
            "warning_zone_objects": sum(1 for item in tracked if item.current_zone == "Warning"),
            "average_velocity": round(sum(item.velocity_m_s for item in tracked) / len(tracked), 3) if tracked else 0.0,
            "maximum_velocity": round(max((item.velocity_m_s for item in tracked), default=0.0), 3),
            "predicted_collisions": sum(1 for item in tracked if item.risk_level in {"HIGH", "CRITICAL"}),
            "average_ttc_s": round(sum(item.ttc_s for item in tracked if item.ttc_s != float("inf")) / max(sum(1 for item in tracked if item.ttc_s != float("inf")), 1), 2) if tracked else float("inf"),
        }
        visualizer.push_tracked_objects(tracked, {**detection_stats, **track_stats, **motion_stats})


def run_lidar_stream(
    controller: RPLiDARController,
    point_queue: "queue.Queue[Optional[dict[str, Any]]]",
    visualizer: RPLiDARVisualizer,
    start_time: float,
    stop_event: threading.Event,
    last_stats_time: list[float],
    last_print_time: list[float],
    reconnect_interval: float = 2.0,
) -> None:
    while not stop_event.is_set():
        if not controller.is_connected():
            visualizer.set_connection_status("Waiting for LiDAR")
            if not ENABLE_AUTO_RECOVERY:
                time.sleep(reconnect_interval)
                continue

            if controller.try_connect():
                try:
                    controller.start_motor()
                    print("\nMotor Started")
                    controller.start_scan()
                    visualizer.set_connection_status("Connected")
                    print("\nBeginning Scan...")
                    print("-" * 50)
                except Exception as exc:
                    visualizer.set_connection_status("Connection Error")
                    print(f"Unable to initialize scan after reconnect: {exc}")
                    controller.disconnect()
                    time.sleep(reconnect_interval)
                    continue
            else:
                time.sleep(reconnect_interval)
                continue

        if controller.is_connected() and not controller.is_scanning():
            try:
                controller.start_motor()
                controller.start_scan()
                visualizer.set_connection_status("Connected")
                logger.info("Started LiDAR scan after initial connection")
            except Exception as exc:
                visualizer.set_connection_status("Connection Error")
                print(f"Unable to initialize scan after connection: {exc}")
                if visualizer.platform is not None:
                    visualizer.platform.diagnostics.record_error(f"Scan init error: {exc}")
                controller.disconnect()
                time.sleep(reconnect_interval)
                continue

        try:
            controller.stream_scan_data(
                callback=lambda point, total, valid, invalid, current_start_time: handle_scan_sample(
                    point,
                    total,
                    valid,
                    invalid,
                    current_start_time,
                    point_queue,
                    visualizer,
                    last_stats_time,
                    last_print_time,
                ),
                start_time=start_time,
            )
        except KeyboardInterrupt:
            print("\nKeyboard interrupt received. Shutting down gracefully...")
            break
        except RuntimeError as exc:
            visualizer.set_connection_status("Scanning unavailable")
            print(f"Scan error: {exc}")
            if visualizer.platform is not None:
                visualizer.platform.diagnostics.record_error(f"Scan error: {exc}")
            controller.disconnect()
            time.sleep(reconnect_interval)
        except Exception as exc:  # pragma: no cover - safety net
            visualizer.set_connection_status("Scanning unavailable")
            print(f"Unexpected scan failure: {exc}")
            if visualizer.platform is not None:
                visualizer.platform.diagnostics.record_error(f"Unexpected scan failure: {exc}")
            controller.disconnect()
            time.sleep(reconnect_interval)
        else:
            if stop_event.is_set():
                break
            logger.info("Scan loop exited; attempting to restart scan")
            time.sleep(max(0.1, reconnect_interval))


def handle_scan_sample(
    point: Optional[dict[str, Any]],
    total_samples: int,
    valid_samples: int,
    invalid_samples: int,
    start_time: float,
    point_queue: "queue.Queue[Optional[dict[str, Any]]]",
    visualizer: RPLiDARVisualizer,
    last_stats_time: list[float],
    last_print_time: list[float],
) -> None:
    current_time = time.monotonic()
    if point is not None:
        point["timestamp"] = current_time
        should_enqueue = total_samples % SCAN_DOWNSAMPLE_RATE == 0
        if point_queue is not None and should_enqueue:
            try:
                point_queue.put_nowait(point)
            except queue.Full:
                pass

        if SHOW_RAW_POINTS and should_enqueue:
            visualizer.push_scan_point(point, {
                "total_samples": total_samples,
                "valid_samples": valid_samples,
                "invalid_samples": invalid_samples,
                "scan_rate": round(total_samples / max(current_time - start_time, 1e-6), 2),
            })
        else:
            visualizer.push_scan_point(None, {
                "total_samples": total_samples,
                "valid_samples": valid_samples,
                "invalid_samples": invalid_samples,
                "scan_rate": round(total_samples / max(current_time - start_time, 1e-6), 2),
            })

        if ENABLE_SAMPLE_PRINT and (total_samples % 100 == 0 or current_time - last_print_time[0] >= 2.0):
            print(f"Sample {total_samples}")
            print(f"Angle: {point['angle']:.2f}°")
            print(f"Distance: {point['distance']} mm")
            print(f"Quality: {point['quality']}")
            print()
            last_print_time[0] = current_time
    else:
        visualizer.push_scan_point(None, {
            "total_samples": total_samples,
            "valid_samples": valid_samples,
            "invalid_samples": invalid_samples,
            "scan_rate": round(total_samples / max(current_time - start_time, 1e-6), 2),
        })


def main() -> int:
    print_banner()
    print_version_info()

    controller = RPLiDARController(
        port=SERIAL_PORT,
        baud_rate=BAUD_RATE,
        timeout=CONNECTION_TIMEOUT,
        debug=DEBUG,
    )
    visualizer = RPLiDARVisualizer(
        width=WINDOW_WIDTH,
        height=WINDOW_HEIGHT,
        max_distance_mm=MAX_SCAN_DISTANCE_MM,
        zoom=INITIAL_ZOOM,
        title="Slamtec RP LiDAR Production Platform\nPhase 10 - Production Robotics Dashboard",
    )
    visualizer.set_connection_status("Searching")

    production_platform = ProductionPlatform()
    production_platform.initialize()
    production_platform.log_event("Production platform initialized")
    visualizer.attach_platform(production_platform)

    point_queue: "queue.Queue[Optional[dict[str, Any]]]" = queue.Queue(maxsize=MAX_POINT_QUEUE_SIZE)
    stop_event = threading.Event()
    detector = RealTimeDetector()
    tracker = ObjectTracker()
    mapping = OccupancyGridMap()
    slam = SLAMSystem(mapping)
    navigation_engine = NavigationEngine(slam, planner_type=parse_planner_type(PLANNER_TYPE))
    visualizer.set_navigation_engine(navigation_engine)

    start_time = time.monotonic()
    last_stats_time = [start_time]
    last_print_time = [start_time]
    last_summary_time = [start_time]
    lastlog_path = initialize_lastlog(Path("lastlog"))

    detection_thread = threading.Thread(
        target=run_detection_loop,
        args=(point_queue, detector, tracker, visualizer, mapping, slam, navigation_engine, stop_event, last_summary_time, lastlog_path),
        daemon=True,
    )

    detection_thread.start()

    print_detected_ports(controller)
    print("\nSearching for LiDAR...")

    connected = False
    try:
        controller.connect()
        connected = True
    except PermissionError as exc:
        visualizer.set_connection_status("Permission Error")
        production_platform.diagnostics.record_error(f"Permission denied: {exc}")
        print(f"Permission denied while opening serial port: {exc}")
    except OSError as exc:
        visualizer.set_connection_status("Serial Error")
        production_platform.diagnostics.record_error(f"Serial error: {exc}")
        print(f"Serial port error: {exc}")
    except RuntimeError as exc:
        visualizer.set_connection_status("Connection Error")
        production_platform.diagnostics.record_error(str(exc))
        print(str(exc))
    except Exception as exc:  # pragma: no cover - safety net
        visualizer.set_connection_status("Connection Error")
        production_platform.diagnostics.record_error(f"Unexpected LiDAR failure: {exc}")
        print(f"Unable to connect to LiDAR: {exc}")

    if connected:
        visualizer.set_connection_status("Connected")
        navigation_engine.start()
        print("\nConnected Successfully")
        print("\nDevice Info")
        info = controller.get_device_info()
        print(f"Model: {info['model']}")
        print(f"Firmware Version: {info['firmware_version']}")
        print(f"Hardware Version: {info['hardware_version']}")
        print(f"Serial Number: {info['serial_number']}")
        print(f"Health Status: {info['health_status']}")
    else:
        status_text = "Waiting for LiDAR"
        visualizer.set_connection_status(status_text)
        production_platform.log_event("LiDAR unavailable at startup")
        print("LiDAR unavailable at startup. Auto-reconnect enabled.")
        if not ENABLE_AUTO_RECOVERY:
            print("Auto-recovery disabled; the dashboard will continue in read-only mode.")

    start_time = time.monotonic()
    last_stats_time = [start_time]
    last_print_time = [start_time]

    scan_thread = threading.Thread(
        target=run_lidar_stream,
        args=(controller, point_queue, visualizer, start_time, stop_event, last_stats_time, last_print_time, LIDAR_RECONNECT_INTERVAL_S),
        daemon=True,
    )
    scan_thread.start()

    try:
        visualizer.run()
    finally:
        print("\nStopping LiDAR...")
        _write_lastlog_entry(lastlog_path, f"stopped={time.strftime('%Y-%m-%d %H:%M:%S')}")
        stop_event.set()
        point_queue.put(None)
        controller.stop_scan()
        controller.stop_motor()
        controller.disconnect()
        visualizer.set_connection_status("Disconnected")
        visualizer.stop()
        if detection_thread.is_alive():
            detection_thread.join(timeout=2.0)
        if scan_thread.is_alive():
            scan_thread.join(timeout=2.0)
    navigation_engine.stop()
    production_platform.shutdown()
    return 0

if __name__ == "__main__":
    sys.exit(main())
