# RP LiDAR Robotics Dashboard

A Python application for visualizing live Slamtec RPLiDAR scans and experimenting with perception and navigation workflows. It combines a real-time scan display with object detection and tracking, occupancy-grid mapping, SLAM, safety-zone analysis, and simulated path planning.

## Features

- Live LiDAR scan visualization with connection status and diagnostics.
- Clustering and tracking of detected objects, including motion estimates and risk zones.
- Occupancy-grid mapping and scan-based localization with pose history.
- A* path planning, dynamic obstacle updates, and replanning in simulation.
- Optional camera capture, YOLO object detection, sensor fusion, and recording.
- Themes, plugins, performance monitoring, logs, exports, and screenshots.
- Automatic serial-port discovery and reconnection attempts.

## Requirements

- Python 3.10 or newer.
- A compatible Slamtec RP LiDAR connected over USB for live scans.
- macOS, Linux, or Windows with a working serial-port driver.

Camera/AI features are optional. The AI features require an available YOLO model; camera and sensor-fusion features are disabled by default.

## Install

From the repository root, create and activate a virtual environment, then install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate` instead.

## Run

Connect and power on the LiDAR, then start the dashboard:

```bash
python main.py
```

The application tries to select a likely LiDAR serial port automatically. To select a particular port, set `RP_LIDAR_SERIAL_PORT` before launching. For example, on macOS or Linux:

```bash
RP_LIDAR_SERIAL_PORT=/dev/tty.usbserial-0001 python main.py
```

On Windows, use the device name assigned by the system, such as `COM3`:

```powershell
$env:RP_LIDAR_SERIAL_PORT = "COM3"
python main.py
```

If no device is available at startup, the dashboard still opens and attempts to reconnect. Serial permissions and the selected port can vary by operating system and LiDAR model.

## Configuration

Runtime settings are defined in [`config.py`](config.py). They include the serial baud rate, visualization dimensions, detection and tracking thresholds, mapping and SLAM options, navigation planner, and optional camera/AI settings. `RP_LIDAR_SERIAL_PORT` is the environment-variable override for the serial port.

The default settings enable navigation simulation, mapping, and SLAM. Camera capture, AI detection, sensor fusion, and recording are disabled by default. The default calibration file is [`calibration/default_calibration.json`](calibration/default_calibration.json).

## Project Layout

| Path | Purpose |
| --- | --- |
| `main.py` | Application startup, LiDAR stream, and processing orchestration |
| `config.py` | Runtime configuration and feature flags |
| `lidar/` | LiDAR communication, perception, tracking, mapping, SLAM, planning, and visualization |
| `themes/` | Dark, light, and high-contrast theme definitions |
| `plugins/` | Example plugin |
| `calibration/` | Default camera/LiDAR calibration data |
| `tests/` | Automated tests |

## Tests

Install the project dependencies and the test runner, then run the tests with:

```bash
python -m pip install -r requirements.txt pytest
python -m pytest
```

## Safety

Navigation in this project is a planning and simulation feature. The application does not provide a physical robot motor-control interface. Do not use its visualizations or risk estimates as the sole basis for operating a robot or other machinery.