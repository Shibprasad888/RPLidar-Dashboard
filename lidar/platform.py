from __future__ import annotations

import importlib.util
import inspect
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Dict, List, Optional

from config import (
    CONFIG_DIRECTORY,
    DEFAULT_THEME,
    EXPORT_DIRECTORY,
    LOG_DIRECTORY,
    PERFORMANCE_MONITOR_INTERVAL_S,
    PLUGIN_DIRECTORY,
    RECORDING_METADATA_FILENAME,
    SCREENSHOT_DIRECTORY,
    SESSION_PROFILE_FILE,
    THEME_DIRECTORY,
)

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None

logger = logging.getLogger("rplidar.platform")


@dataclass(slots=True)
class Theme:
    name: str
    palette: Dict[str, str]
    accent: str

    @classmethod
    def load(cls, path: Path) -> "Theme":
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return cls(
            name=data.get("name", path.stem),
            palette=data.get("palette", {}),
            accent=data.get("accent", "#5BD6FF"),
        )


@dataclass(slots=True)
class ConfigManager:
    config_dir: Path = Path(CONFIG_DIRECTORY)
    active_profile: str = "default"
    values: Dict[str, Any] = field(default_factory=dict)
    profile_path: Path = field(init=False)
    profile_file: Path = field(init=False)

    def __post_init__(self) -> None:
        self.config_dir = Path(self.config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.profile_path = self.config_dir / self.active_profile
        self.profile_path.mkdir(parents=True, exist_ok=True)
        self.profile_file = self.profile_path / SESSION_PROFILE_FILE
        self.values = self.load_profile()

    def load_profile(self) -> Dict[str, Any]:
        if self.profile_file.exists():
            try:
                with self.profile_file.open("r", encoding="utf-8") as handle:
                    data = json.load(handle)
                return data
            except Exception as exc:
                logger.warning("Failed to load config profile %s: %s", self.profile_file, exc)
        return self.default_values()

    def save_profile(self) -> None:
        try:
            with self.profile_file.open("w", encoding="utf-8") as handle:
                json.dump(self.values, handle, indent=2)
        except Exception as exc:
            logger.error("Unable to save configuration profile: %s", exc)

    def reset_profile(self) -> None:
        self.values = self.default_values()
        self.save_profile()

    def import_profile(self, import_path: Path) -> None:
        with import_path.open("r", encoding="utf-8") as handle:
            self.values = json.load(handle)
        self.save_profile()

    def export_profile(self, export_path: Path) -> None:
        export_path.parent.mkdir(parents=True, exist_ok=True)
        with export_path.open("w", encoding="utf-8") as handle:
            json.dump(self.values, handle, indent=2)

    @staticmethod
    def default_values() -> Dict[str, Any]:
        return {
            "theme": DEFAULT_THEME,
            "window": {"width": 1400, "height": 900, "fullscreen": False},
            "recording_enabled": False,
            "recording_path": RECORDING_METADATA_FILENAME,
            "playback_speed": 1.0,
            "performance_interval": PERFORMANCE_MONITOR_INTERVAL_S,
            "last_export_path": str(Path(EXPORT_DIRECTORY).resolve()),
            "plugin_whitelist": [],
        }

    def validate(self) -> bool:
        if not isinstance(self.values.get("theme"), str):
            self.values["theme"] = DEFAULT_THEME
        if not isinstance(self.values.get("window"), dict):
            self.values["window"] = {"width": 1400, "height": 900, "fullscreen": False}
        return True

    def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.values[key] = value


@dataclass(slots=True)
class DiagnosticsReport:
    timestamp: float
    device_status: Dict[str, str] = field(default_factory=dict)
    thread_status: Dict[str, str] = field(default_factory=dict)
    storage_status: Dict[str, Any] = field(default_factory=dict)
    recent_errors: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)


class DiagnosticsManager:
    def __init__(self, config: ConfigManager) -> None:
        self.config = config
        self.recent_errors: List[str] = []
        self.last_report: Optional[DiagnosticsReport] = None

    def record_error(self, error_message: str) -> None:
        if error_message:
            self.recent_errors.insert(0, f"{time.ctime()}: {error_message}")
            self.recent_errors = self.recent_errors[:20]
            logger.warning("Diagnostics recorded error: %s", error_message)

    def build_report(self, performance_metrics: Dict[str, Any], thread_names: List[str], storage_path: Path) -> DiagnosticsReport:
        device_status = {
            "LiDAR": "Connected",
            "Camera": "Enabled" if self.config.get("recording_enabled", False) else "Disabled",
            "Navigation": "Active",
            "AI Model": "Available",
        }
        thread_status = {name: "running" for name in thread_names}
        try:
            usage = psutil.disk_usage(str(storage_path)) if psutil else None
            storage_status = {
                "available_gb": round(usage.free / 1_073_741_824, 2) if usage else None,
                "total_gb": round(usage.total / 1_073_741_824, 2) if usage else None,
                "percent": round(usage.percent, 1) if usage else None,
            }
        except Exception:
            storage_status = {"error": "Unable to query disk usage"}

        report = DiagnosticsReport(
            timestamp=time.time(),
            device_status=device_status,
            thread_status=thread_status,
            storage_status=storage_status,
            recent_errors=list(self.recent_errors),
            performance_metrics=performance_metrics,
        )
        self.last_report = report
        return report


class PerformanceMonitor:
    def __init__(self, interval_s: float = PERFORMANCE_MONITOR_INTERVAL_S) -> None:
        self.interval_s = interval_s
        self.metrics: Dict[str, Any] = {
            "cpu_percent": 0.0,
            "memory_percent": 0.0,
            "gpu_percent": None,
            "thread_count": 0,
            "disk_percent": 0.0,
            "camera_fps": 0.0,
            "radar_fps": 0.0,
            "ai_fps": 0.0,
            "slam_fps": 0.0,
            "navigation_fps": 0.0,
        }
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        logger.info("Performance monitor started")

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        logger.info("Performance monitor stopped")

    def update_fps(self, radar_fps: float = 0.0, camera_fps: float = 0.0, ai_fps: float = 0.0, slam_fps: float = 0.0, navigation_fps: float = 0.0) -> None:
        self.metrics.update(
            radar_fps=radar_fps,
            camera_fps=camera_fps,
            ai_fps=ai_fps,
            slam_fps=slam_fps,
            navigation_fps=navigation_fps,
        )

    def _monitor_loop(self) -> None:
        while not self._stop_event.is_set():
            self.metrics["thread_count"] = threading.active_count()
            if psutil:
                self.metrics["cpu_percent"] = psutil.cpu_percent(interval=None)
                self.metrics["memory_percent"] = psutil.virtual_memory().percent
                self.metrics["disk_percent"] = psutil.disk_usage(os.getcwd()).percent
                self.metrics["gpu_percent"] = None
            time.sleep(self.interval_s)

    def snapshot(self) -> Dict[str, Any]:
        return dict(self.metrics)


PluginRegister = Callable[["ProductionPlatform"], bool]


class PluginManager:
    def __init__(self, platform: "ProductionPlatform") -> None:
        self.platform = platform
        self.loaded_plugins: List[str] = []
        self.plugin_modules: List[ModuleType] = []

    def load_plugins(self) -> None:
        plugin_dir = Path(PLUGIN_DIRECTORY)
        plugin_dir.mkdir(parents=True, exist_ok=True)
        for plugin_file in plugin_dir.glob("*.py"):
            try:
                spec = importlib.util.spec_from_file_location(plugin_file.stem, plugin_file)
                if spec is None or spec.loader is None:
                    continue
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                if hasattr(module, "register_plugin") and inspect.isfunction(module.register_plugin):
                    if module.register_plugin(self.platform):
                        self.loaded_plugins.append(plugin_file.stem)
                        self.plugin_modules.append(module)
                        logger.info("Loaded plugin: %s", plugin_file.stem)
            except Exception as exc:
                logger.warning("Plugin %s failed to load: %s", plugin_file.name, exc)
                self.platform.diagnostics.record_error(f"Plugin failed: {plugin_file.name} {exc}")


class PlaybackManager:
    def __init__(self) -> None:
        self.session_path: Optional[Path] = None
        self.metadata: Dict[str, Any] = {}
        self.position: int = 0
        self.speed: float = 1.0
        self.is_playing: bool = False
        self.state: List[Dict[str, Any]] = []

    def load_session(self, session_path: Path) -> None:
        self.session_path = session_path
        if not session_path.exists():
            raise FileNotFoundError(f"Playback session path not found: {session_path}")
        metadata_path = session_path / RECORDING_METADATA_FILENAME
        if metadata_path.exists():
            with metadata_path.open("r", encoding="utf-8") as handle:
                self.metadata = json.load(handle)
        self.position = 0
        self.state = []
        self.is_playing = False

    def play(self) -> None:
        self.is_playing = True

    def pause(self) -> None:
        self.is_playing = False

    def stop(self) -> None:
        self.is_playing = False
        self.position = 0

    def seek(self, position: int) -> None:
        self.position = max(0, min(position, len(self.state) - 1))

    def step_forward(self) -> None:
        self.seek(self.position + 1)

    def step_backward(self) -> None:
        self.seek(self.position - 1)


class ProductionPlatform:
    def __init__(self) -> None:
        self.config = ConfigManager()
        self.theme_manager = ThemeManager(Path(THEME_DIRECTORY))
        self.performance_monitor = PerformanceMonitor(self.config.get("performance_interval", PERFORMANCE_MONITOR_INTERVAL_S))
        self.diagnostics = DiagnosticsManager(self.config)
        self.plugin_manager = PluginManager(self)
        self.playback_manager = PlaybackManager()
        self.event_log: List[str] = []
        self.log_dir = Path(LOG_DIRECTORY)
        self.export_dir = Path(EXPORT_DIRECTORY)
        self.screenshot_dir = Path(SCREENSHOT_DIRECTORY)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.export_dir.mkdir(parents=True, exist_ok=True)
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)
        self.platform_start_time = time.time()
        self._register_basic_logger()

    def _register_basic_logger(self) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(self.log_dir / "application.log", encoding="utf-8")
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        file_handler.setFormatter(formatter)
        root_logger = logging.getLogger()
        if not any(isinstance(handler, logging.FileHandler) for handler in root_logger.handlers):
            root_logger.addHandler(file_handler)
        root_logger.setLevel(logging.INFO)

    def initialize(self) -> None:
        self.log_event("Application startup")
        self.plugin_manager.load_plugins()
        self.performance_monitor.start()

    def shutdown(self) -> None:
        self.log_event("Application shutdown")
        self.performance_monitor.stop()

    def log_event(self, message: str) -> None:
        timestamped = f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {message}"
        self.event_log.insert(0, timestamped)
        self.event_log = self.event_log[:50]
        logger.info(message)

    def capture_screenshot(self, surface: Any, filename: str) -> Path:
        path = self.screenshot_dir / filename
        try:
            import pygame

            if hasattr(surface, "convert_alpha"):
                pygame.image.save(surface, str(path))
            else:
                raise TypeError("Surface object is not compatible with screenshot capture")
        except Exception as exc:
            self.diagnostics.record_error(f"Screenshot failed: {exc}")
        return path

    def export_report(self, name: str, data: Dict[str, Any]) -> Path:
        export_path = self.export_dir / f"{name}.json"
        try:
            with export_path.open("w", encoding="utf-8") as handle:
                json.dump(data, handle, indent=2)
        except Exception as exc:
            self.diagnostics.record_error(f"Export failed: {exc}")
        return export_path


class ThemeManager:
    def __init__(self, theme_dir: Path) -> None:
        self.theme_dir = theme_dir
        self.themes: Dict[str, Theme] = {}
        self.current_theme: Theme = self.load_default_theme()
        self.theme_dir.mkdir(parents=True, exist_ok=True)
        self.load_themes()

    def load_default_theme(self) -> Theme:
        return Theme(name="dark", palette={"background": "#0A0C10", "panel": "#141824", "text": "#F6F8FA", "accent": "#44C3FF"}, accent="#44C3FF")

    def load_themes(self) -> None:
        if not self.theme_dir.exists():
            return
        for theme_file in self.theme_dir.glob("*.json"):
            try:
                theme = Theme.load(theme_file)
                self.themes[theme.name] = theme
            except Exception as exc:
                logger.warning("Failed to load theme %s: %s", theme_file.name, exc)
        if DEFAULT_THEME in self.themes:
            self.current_theme = self.themes[DEFAULT_THEME]

    def get_theme(self, name: str) -> Theme:
        return self.themes.get(name, self.current_theme)

    def set_theme(self, name: str) -> None:
        self.current_theme = self.get_theme(name)


class PluginInterface:
    def __init__(self, name: str) -> None:
        self.name = name

    def on_initialize(self, platform: ProductionPlatform) -> None:
        raise NotImplementedError

    def on_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        raise NotImplementedError
