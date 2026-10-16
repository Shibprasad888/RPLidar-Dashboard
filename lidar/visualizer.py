import math
import os
import sys
import threading
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple


def hex_to_rgb(color_hex: str) -> Tuple[int, int, int]:
    cleaned = color_hex.lstrip("#").strip()
    if len(cleaned) == 3:
        cleaned = "".join(ch * 2 for ch in cleaned)
    if len(cleaned) != 6:
        return 0, 0, 0
    return tuple(int(cleaned[i : i + 2], 16) for i in range(0, 6, 2))
if os.environ.get("RPLIDAR_FORCE_HEADLESS", "").lower() in {"1", "true", "yes"}:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
elif sys.platform == "darwin":
    os.environ.setdefault("SDL_VIDEODRIVER", "cocoa")

import pygame

from config import (
    AWARENESS_ZONE_RADIUS_MM,
    CRITICAL_ZONE_RADIUS_MM,
    ENABLE_MOTION_TRAILS,
    ENABLE_PREDICTION_PATHS,
    ENABLE_SAFETY_ZONES,
    ENABLE_VELOCITY_ARROWS,
    PREDICTION_TIME_HORIZON,
    WARNING_ZONE_RADIUS_MM,
    SHOW_RAW_POINTS,
)
from .geometry import Point2D, clamp, polar_to_cartesian
from .mapping import OccupancyGridMap
from .models import TrackedObject


def polar_to_cartesian(angle_deg: float, distance_mm: float) -> Tuple[float, float]:
    """Convert polar scan data into Cartesian coordinates in millimeters."""
    angle_rad = math.radians(angle_deg)
    x_mm = distance_mm * math.sin(angle_rad)
    y_mm = -distance_mm * math.cos(angle_rad)
    return x_mm, y_mm


def cartesian_to_screen(
    x_mm: float,
    y_mm: float,
    center_x: float,
    center_y: float,
    scale_mm_per_px: float,
    pan_x: float = 0.0,
    pan_y: float = 0.0,
) -> Tuple[int, int]:
    """Convert Cartesian millimeter coordinates into screen pixels."""
    screen_x = center_x + pan_x + (x_mm / scale_mm_per_px)
    screen_y = center_y + pan_y + (y_mm / scale_mm_per_px)
    return int(screen_x), int(screen_y)


def distance_to_screen_scale(distance_mm: float, max_distance_mm: float, radius_px: float, zoom: float) -> float:
    """Scale a distance into screen-space using the current zoom level."""
    if max_distance_mm <= 0:
        return 0.0
    return (distance_mm / max_distance_mm) * radius_px * zoom


class RPLiDARVisualizer:
    """Lightweight pygame-based radar visualization for live LiDAR scan data."""

    def __init__(
        self,
        width: int = 1400,
        height: int = 900,
        max_distance_mm: int = 10000,
        zoom: float = 1.0,
        point_limit: int = 8000,
        title: str = "Slamtec RP LiDAR Detection System",
    ) -> None:
        self.width = width
        self.height = height
        self.max_distance_mm = max_distance_mm
        self.zoom = zoom
        self.point_limit = point_limit
        self.title = title
        self.running = True
        self.screen: Optional[pygame.Surface] = None
        self.clock: Optional[pygame.time.Clock] = None
        self.fps = 60.0
        self._initialized = False
        self._headless = False

        self.pan_x = 0.0
        self.pan_y = 0.0
        self._points_lock = threading.Lock()
        self._latest_points: Deque[dict[str, Any]] = deque(maxlen=point_limit)
        self._latest_tracked_objects: List[TrackedObject] = []
        self.occupancy_map: Optional[OccupancyGridMap] = None
        self.map_stats: dict[str, Any] = {}
        self.navigation_engine: Optional[Any] = None
        self._latest_stats: dict[str, Any] = {
            "connection_status": "Initializing",
            "fps": 0.0,
            "scan_rate": 0.0,
            "total_samples": 0,
            "valid_samples": 0,
            "invalid_samples": 0,
            "rotation_speed_rpm": "N/A",
            "max_scan_distance_mm": max_distance_mm,
            "zoom_level": zoom,
            "runtime": 0.0,
        }
        self._start_time = time.monotonic()
        self.platform: Optional[Any] = None
        self._diagnostics_overlay = False
        self.theme_colors: dict[str, Tuple[int, int, int]] = {
            "background": (10, 12, 16),
            "panel": (18, 20, 28),
            "border": (70, 80, 95),
            "text": (235, 240, 248),
            "accent": (68, 195, 255),
            "muted": (150, 160, 170),
            "warn": (255, 80, 80),
            "error": (255, 80, 80),
            "success": (40, 220, 110),
        }

    def set_connection_status(self, status: str) -> None:
        """Update the displayed connection state."""
        with self._points_lock:
            self._latest_stats["connection_status"] = status

    def attach_platform(self, platform: Any) -> None:
        self.platform = platform
        theme = getattr(platform, "theme_manager", None)
        if theme is not None:
            self.apply_theme(theme.current_theme)

    def apply_theme(self, theme: Any) -> None:
        if theme is None:
            return
        palette = getattr(theme, "palette", None) or theme
        accent = getattr(theme, "accent", None)
        if isinstance(palette, dict):
            for key, value in palette.items():
                if isinstance(value, str):
                    self.theme_colors[key] = hex_to_rgb(value)
        if isinstance(accent, str):
            self.theme_colors["accent"] = hex_to_rgb(accent)

    def _theme_color(self, name: str, fallback: Tuple[int, int, int]) -> Tuple[int, int, int]:
        return self.theme_colors.get(name, fallback)

    def push_scan_point(self, point: Optional[dict[str, Any]], stats: Optional[dict[str, Any]] = None) -> None:
        """Store the latest scan point and statistics in a thread-safe buffer."""
        with self._points_lock:
            if point is not None:
                self._latest_points.append(point)
            if stats is not None:
                self._latest_stats.update(stats)

    def get_frame_data(self) -> Tuple[list[dict[str, Any]], List[TrackedObject], dict[str, Any]]:
        """Return a snapshot of the latest scan points, tracked objects, and HUD statistics."""
        with self._points_lock:
            points = list(self._latest_points)
            tracked_objects = list(self._latest_tracked_objects)
            stats = dict(self._latest_stats)
        return points, tracked_objects, stats

    def push_tracked_objects(self, tracked_objects: List[TrackedObject], stats: Optional[dict[str, Any]] = None) -> None:
        """Store the latest tracked object set and summary statistics."""
        with self._points_lock:
            self._latest_tracked_objects = tracked_objects
            if stats is not None:
                self._latest_stats.update(stats)

    def push_map(self, occupancy_map: Optional[OccupancyGridMap], stats: Optional[dict[str, Any]] = None) -> None:
        """Store the latest occupancy map and its summary statistics."""
        with self._points_lock:
            self.occupancy_map = occupancy_map
            if stats is not None:
                self.map_stats = dict(stats)
                self._latest_stats.update(stats)

    def run(self) -> None:
        """Start the pygame render loop."""
        try:
            pygame.init()
            pygame.display.set_caption(self.title)
            self.screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)
            self.clock = pygame.time.Clock()
            self._initialized = True
            self._headless = False
        except Exception as exc:
            self._headless = True
            self._initialized = False
            self._latest_stats["connection_status"] = f"Display unavailable: {exc}"
            return

        while self.running:
            self._handle_events()
            self._render_frame()
            if self.clock is None:
                break
            self.clock.tick(60)
            self.fps = self.clock.get_fps()
            with self._points_lock:
                self._latest_stats["fps"] = round(self.fps, 1)
                self._latest_stats["zoom_level"] = round(self.zoom, 2)
                self._latest_stats["runtime"] = round(time.monotonic() - self._start_time, 2)

        try:
            pygame.quit()
        except Exception:
            pass

    def stop(self) -> None:
        """Ask the render loop to stop."""
        self.running = False

    def _handle_events(self) -> None:
        if self._headless or self.screen is None:
            return
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                mods = event.mod
                is_ctrl = mods & pygame.KMOD_CTRL

                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    self.pan_x = 0.0
                    self.pan_y = 0.0
                    self.zoom = 1.0
                elif event.key == pygame.K_LEFT:
                    self.pan_x -= 30.0
                elif event.key == pygame.K_RIGHT:
                    self.pan_x += 30.0
                elif event.key == pygame.K_UP:
                    self.pan_y -= 30.0
                elif event.key == pygame.K_DOWN:
                    self.pan_y += 30.0
                elif event.key == pygame.K_g:
                    threading.Thread(target=self._prompt_goal_input, daemon=True).start()
                elif event.key == pygame.K_r and self.navigation_engine is not None and not is_ctrl:
                    request = self.navigation_engine.select_random_goal()
                    if request is not None:
                        self.navigation_engine.request_goal(request)
                elif event.key == pygame.K_r and is_ctrl:
                    self._toggle_recording()
                elif event.key == pygame.K_p and is_ctrl:
                    self._toggle_playback()
                elif event.key == pygame.K_s and is_ctrl:
                    self._save_configuration()
                elif event.key == pygame.K_e and is_ctrl:
                    self._export_session()
                elif event.key == pygame.K_l and is_ctrl:
                    self._clear_event_log()
                elif event.key == pygame.K_d and is_ctrl:
                    self._diagnostics_overlay = not self._diagnostics_overlay
                elif event.key == pygame.K_t and is_ctrl:
                    self._cycle_theme()
                elif event.key == pygame.K_F11:
                    self._toggle_fullscreen()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 4:
                    self.zoom = clamp(self.zoom + 0.1, 0.25, 4.0)
                elif event.button == 5:
                    self.zoom = clamp(self.zoom - 0.1, 0.25, 4.0)
                elif event.button == 1 and self.navigation_engine is not None:
                    world_x, world_y = self._screen_to_world(event.pos[0], event.pos[1])
                    from .navigation import NavigationRequest

                    self.navigation_engine.request_goal(NavigationRequest(goal_x_mm=world_x, goal_y_mm=world_y))

    def _render_frame(self) -> None:
        if self._headless or self.screen is None:
            return

        width, height = self.screen.get_size()
        self.width = width
        self.height = height
        self.screen.fill(self._theme_color("background", (10, 12, 16)))

        center_x = width / 2 + self.pan_x
        center_y = height / 2 + self.pan_y
        # Increase base radius so the scanner area occupies most of the canvas
        base_radius = min(width, height) * 0.46
        radius_px = base_radius * self.zoom
        scale_mm_per_px = self.max_distance_mm / radius_px

        self._draw_grid(center_x, center_y, radius_px, scale_mm_per_px)
        self._draw_rings(center_x, center_y, radius_px, scale_mm_per_px)
        self._draw_safety_zones(center_x, center_y, radius_px, scale_mm_per_px)
        self._draw_angle_markers(center_x, center_y, radius_px)
        self._draw_robot(center_x, center_y)
        self._draw_scan_points(center_x, center_y, scale_mm_per_px)
        self._draw_tracked_objects(center_x, center_y, scale_mm_per_px)
        self._draw_navigation_overlay(center_x, center_y, scale_mm_per_px)
        self._draw_map_preview(width - 260, 20, 240, 180)
        self._draw_navigation_dashboard(width - 520, 220, 240, int(height * 0.45))
        self._draw_event_log(20, height - 210, 360, 190)
        self._draw_hud()
        if self._diagnostics_overlay:
            self._draw_diagnostics_overlay(400, height - 210, 520, 190)
        pygame.display.flip()

    def _draw_grid(self, center_x: float, center_y: float, radius_px: float, scale_mm_per_px: float) -> None:
        grid_step_mm = 1000.0
        grid_step_px = grid_step_mm / scale_mm_per_px
        color = (36, 42, 52)

        for x in range(int(-3 * grid_step_px), int(3 * grid_step_px) + 1, int(grid_step_px)):
            pygame.draw.line(self.screen, color, (int(center_x + x), 0), (int(center_x + x), self.height))
        for y in range(int(-3 * grid_step_px), int(3 * grid_step_px) + 1, int(grid_step_px)):
            pygame.draw.line(self.screen, color, (0, int(center_y + y)), (self.width, int(center_y + y)))

        pygame.draw.circle(self.screen, (30, 35, 44), (int(center_x), int(center_y)), int(radius_px), 1)

    def _draw_rings(self, center_x: float, center_y: float, radius_px: float, scale_mm_per_px: float) -> None:
        ring_distances_mm = [500, 1000, 2000, 3000, 5000, 10000]
        ring_color = (60, 72, 88)
        label_color = (205, 220, 235)

        for distance_mm in ring_distances_mm:
            radius = distance_mm / scale_mm_per_px
            pygame.draw.circle(self.screen, ring_color, (int(center_x), int(center_y)), int(radius), 1)
            label_text = f"{distance_mm / 1000:.0f} m"
            font = pygame.font.SysFont("Arial", 14)
            text_surface = font.render(label_text, True, label_color)
            self.screen.blit(text_surface, (int(center_x + radius + 6), int(center_y - 8)))

    def _draw_angle_markers(self, center_x: float, center_y: float, radius_px: float) -> None:
        angles = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
        color = (180, 190, 205)
        font = pygame.font.SysFont("Arial", 14)

        for angle in angles:
            angle_rad = math.radians(90 - angle)
            x = math.sin(angle_rad) * (radius_px - 20)
            y = -math.cos(angle_rad) * (radius_px - 20)
            label = f"{angle}°"
            text_surface = font.render(label, True, color)
            self.screen.blit(text_surface, (int(center_x + x - 10), int(center_y + y - 8)))
            pygame.draw.line(
                self.screen,
                color,
                (int(center_x), int(center_y)),
                (int(center_x + x), int(center_y + y)),
                1,
            )

    def _draw_robot(self, center_x: float, center_y: float) -> None:
        pygame.draw.circle(self.screen, (80, 180, 255), (int(center_x), int(center_y)), 10)
        pygame.draw.circle(self.screen, (255, 255, 255), (int(center_x), int(center_y)), 10, 1)
        pygame.draw.line(self.screen, (255, 255, 255), (int(center_x), int(center_y)), (int(center_x), int(center_y - 24)), 2)

    def _draw_scan_points(self, center_x: float, center_y: float, scale_mm_per_px: float) -> None:
        # Optionally hide raw LiDAR points for a cleaner view
        if not SHOW_RAW_POINTS:
            return

        points, _, _ = self.get_frame_data()
        if not points:
            return

        for point in points:
            try:
                angle = float(point["angle"])
                distance_mm = float(point["distance"])
            except (KeyError, TypeError, ValueError):
                continue

            if distance_mm <= 0 or not math.isfinite(distance_mm):
                continue

            x_mm, y_mm = polar_to_cartesian(angle, distance_mm)
            screen_x, screen_y = cartesian_to_screen(x_mm, y_mm, center_x, center_y, scale_mm_per_px, self.pan_x, self.pan_y)
            depth_ratio = clamp(1.0 - (distance_mm / self.max_distance_mm), 0.15, 1.0)
            color = (int(80 + 120 * depth_ratio), int(210 * depth_ratio), int(80 + 50 * depth_ratio))
            alpha = int(80 + 110 * depth_ratio)

            # reduce visual weight of raw points
            glow_surface = pygame.Surface((6, 6), pygame.SRCALPHA)
            pygame.draw.circle(glow_surface, (color[0], color[1], color[2], max(30, alpha - 40)), (3, 3), 3)
            self.screen.blit(glow_surface, (screen_x - 3, screen_y - 3), special_flags=pygame.BLEND_RGBA_ADD)
            pygame.draw.circle(self.screen, color, (screen_x, screen_y), 1)

    def _draw_safety_zones(self, center_x: float, center_y: float, radius_px: float, scale_mm_per_px: float) -> None:
        if not ENABLE_SAFETY_ZONES:
            return
        zones = [
            (CRITICAL_ZONE_RADIUS_MM, (255, 80, 80), 2),
            (WARNING_ZONE_RADIUS_MM, (255, 200, 80), 1),
            (AWARENESS_ZONE_RADIUS_MM, (90, 180, 255), 1),
        ]
        for radius_mm, color, width in zones:
            radius = radius_mm / scale_mm_per_px
            pygame.draw.circle(self.screen, color, (int(center_x), int(center_y)), int(radius), width)

    def _format_object_summary(self, obj: TrackedObject) -> str:
        distance_m = obj.distance_mm / 1000.0
        direction = self._describe_direction(obj.angle_deg)
        motion = self._describe_motion(obj.velocity_m_s, obj.heading_deg)
        risk = self._describe_risk(obj.risk_level)
        zone = obj.current_zone.lower()
        return f"{distance_m:.1f} m {direction}, {motion}, {zone} zone, {risk}"

    def _describe_direction(self, angle_deg: float) -> str:
        if angle_deg is None:
            return "around you"
        normalized = int(round(angle_deg)) % 360
        if 315 <= normalized or normalized <= 45:
            return "ahead"
        if 45 < normalized < 135:
            return "left"
        if 135 <= normalized < 225:
            return "behind"
        return "right"

    def _describe_motion(self, velocity_m_s: float, heading_deg: float) -> str:
        if velocity_m_s <= 0.05:
            return "stationary"
        direction = self._describe_direction(heading_deg)
        return f"moving {direction} at {velocity_m_s:.2f} m/s"

    def _describe_risk(self, risk_level: str) -> str:
        return {
            "SAFE": "safe",
            "LOW": "low concern",
            "MEDIUM": "medium concern",
            "HIGH": "high concern",
            "CRITICAL": "critical",
        }.get(risk_level, "unknown")

    def _draw_tracked_objects(self, center_x: float, center_y: float, scale_mm_per_px: float) -> None:
        _, tracked_objects, stats = self.get_frame_data()
        if not tracked_objects:
            return

        for obj in tracked_objects:
            try:
                screen_x, screen_y = cartesian_to_screen(
                    obj.center_x_mm,
                    obj.center_y_mm,
                    center_x,
                    center_y,
                    scale_mm_per_px,
                    self.pan_x,
                    self.pan_y,
                )
            except Exception:
                continue

            radius_px = max(10.0, obj.radius_mm / scale_mm_per_px)
            # Emphasize tracked objects: filled circle with outline
            risk_color = {
                "SAFE": (80, 220, 110),
                "LOW": (95, 180, 255),
                "MEDIUM": (255, 220, 90),
                "HIGH": (255, 150, 70),
                "CRITICAL": (255, 80, 80),
            }.get(obj.risk_level, (80, 220, 110))
            pygame.draw.circle(self.screen, risk_color, (screen_x, screen_y), int(max(6, radius_px * 0.4)))
            pygame.draw.circle(self.screen, (255, 255, 255), (screen_x, screen_y), int(radius_px * 0.6), 2)
            if stats.get("show_connection_lines", True):
                pygame.draw.line(self.screen, (70, 120, 255), (int(center_x), int(center_y)), (screen_x, screen_y), 1)

            if ENABLE_VELOCITY_ARROWS and obj.velocity_m_s > 0.01:
                arrow_tip_x = screen_x + math.sin(math.radians(obj.heading_deg)) * 20
                arrow_tip_y = screen_y - math.cos(math.radians(obj.heading_deg)) * 20
                pygame.draw.line(self.screen, (120, 235, 190), (screen_x, screen_y), (int(arrow_tip_x), int(arrow_tip_y)), 2)
                pygame.draw.circle(self.screen, (120, 235, 190), (int(arrow_tip_x), int(arrow_tip_y)), 3)

            if ENABLE_MOTION_TRAILS and obj.trail:
                trail_points = [cartesian_to_screen(x, y, center_x, center_y, scale_mm_per_px, self.pan_x, self.pan_y) for x, y in obj.trail]
                for idx in range(1, len(trail_points)):
                    start = trail_points[idx - 1]
                    end = trail_points[idx]
                    alpha = 40 + int((idx / len(trail_points)) * 100)
                    trail_color = (40, 180, 140, alpha)
                    trail_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
                    pygame.draw.line(trail_surface, trail_color, start, end, 2)
                    self.screen.blit(trail_surface, (0, 0))

            if ENABLE_PREDICTION_PATHS and obj.motion_debug:
                for point in obj.motion_debug.get("future_positions", [])[:3]:
                    future_x, future_y = cartesian_to_screen(point["x_mm"], point["y_mm"], center_x, center_y, scale_mm_per_px, self.pan_x, self.pan_y)
                    pygame.draw.circle(self.screen, (250, 250, 120), (future_x, future_y), 3)

            if obj.risk_level == "CRITICAL":
                pygame.draw.circle(self.screen, (255, 255, 255), (screen_x, screen_y), int(radius_px * 0.8), 2)

            if stats.get("show_labels", True):
                label = f"Object {obj.track_id}"
                font = pygame.font.SysFont("Arial", 13)
                label_surface = font.render(label, True, (255, 255, 255))
                self.screen.blit(label_surface, (screen_x + 8, screen_y - 6))

                summary_text = self._format_object_summary(obj)
                text_lines = [
                    summary_text,
                    f"status: {obj.status}",
                    f"confidence: {obj.confidence:.2f}",
                ]
                for idx, text in enumerate(text_lines):
                    text_surface = font.render(text, True, (230, 240, 250))
                    self.screen.blit(text_surface, (screen_x + 8, screen_y + 10 + idx * 14))

            if stats.get("show_debug_labels", False):
                debug_font = pygame.font.SysFont("Arial", 11)
                debug_text = f"miss={obj.frames_missing} age={obj.age_frames}"
                debug_surface = debug_font.render(debug_text, True, (220, 220, 220))
                self.screen.blit(debug_surface, (screen_x + 8, screen_y + 88))

    def _draw_hud(self) -> None:
        _, _, stats = self.get_frame_data()
        hud_width = 360
        hud_height = 360
        padding = 16
        rect = pygame.Rect(self.width - hud_width - padding, padding, hud_width, hud_height)
        pygame.draw.rect(self.screen, self._theme_color("panel", (18, 20, 28)), rect, border_radius=14)
        pygame.draw.rect(self.screen, self._theme_color("border", (70, 80, 95)), rect, 1, border_radius=14)

        font = pygame.font.SysFont("Arial", 16)
        title_font = pygame.font.SysFont("Arial", 18, bold=True)
        title_surface = title_font.render("Live Status", True, self._theme_color("text", (235, 240, 248)))
        self.screen.blit(title_surface, (rect.x + 14, rect.y + 12))

        entries = [
            ("Connection", stats.get("connection_status", "Unknown")),
            ("FPS", f"{stats.get('fps', 0.0):.1f}"),
            ("Scan Rate", f"{stats.get('scan_rate', 0.0):.1f} sps"),
            ("Samples", str(stats.get("total_samples", 0))),
            ("Valid", str(stats.get("valid_samples", 0))),
            ("Invalid", str(stats.get("invalid_samples", 0))),
            ("RPM", str(stats.get("rotation_speed_rpm", "N/A"))),
            ("Max Range", f"{stats.get('max_scan_distance_mm', self.max_distance_mm) / 1000:.1f} m"),
            ("Zoom", f"{stats.get('zoom_level', self.zoom):.2f}x"),
            ("Runtime", f"{stats.get('runtime', 0.0):.1f}s"),
            ("Objects", str(stats.get("tracked_objects", 0))),
            ("Moving", str(stats.get("moving_objects", 0))),
            ("Stationary", str(stats.get("stationary_objects", 0))),
            ("Critical", str(stats.get("critical_zone_objects", 0))),
            ("Warning", str(stats.get("warning_zone_objects", 0))),
            ("Avg Speed", f"{stats.get('average_velocity', 0.0):.2f} m/s"),
            ("Max Speed", f"{stats.get('maximum_velocity', 0.0):.2f} m/s"),
            ("Pred Coll", str(stats.get("predicted_collisions", 0))),
            ("Avg TTC", f"{stats.get('average_ttc_s', float('inf')):.2f}s"),
        ]

        y = rect.y + 48
        for label, value in entries:
            label_surface = font.render(f"{label}:", True, self._theme_color("muted", (190, 200, 210)))
            value_surface = font.render(str(value), True, self._theme_color("text", (255, 255, 255)))
            self.screen.blit(label_surface, (rect.x + 16, y))
            self.screen.blit(value_surface, (rect.x + 160, y))
            y += 22

        object_panel_y = rect.y + 220
        object_panel_height = max(100, self.height - object_panel_y - 40)
        object_panel = pygame.Rect(rect.x, object_panel_y, hud_width, object_panel_height)
        pygame.draw.rect(self.screen, self._theme_color("background", (22, 24, 34)), object_panel, border_radius=12)
        pygame.draw.rect(self.screen, self._theme_color("border", (70, 80, 95)), object_panel, 1, border_radius=12)
        object_title = title_font.render("Tracked Objects", True, self._theme_color("text", (240, 245, 250)))
        legend = [
            "Green = safe",
            "Blue = low concern",
            "Yellow = medium",
            "Orange = high",
            "Red = critical",
        ]
        for idx, line in enumerate(legend):
            legend_surface = font.render(line, True, self._theme_color("muted", (210, 220, 230)))
            self.screen.blit(legend_surface, (rect.x + 16, object_panel_y + 28 + idx * 14))
        self.screen.blit(object_title, (rect.x + 14, object_panel_y + 10))

        _, tracked_objects, _ = self.get_frame_data()
        object_entries = []
        for obj in sorted(tracked_objects, key=lambda item: (self._risk_rank(item.risk_level), -item.velocity_m_s, item.track_id))[:8]:
            object_entries.extend([
                f"Obj {obj.track_id}: {self._format_object_summary(obj)}",
                f"status={obj.status} conf={obj.confidence:.2f}",
            ])
        text_y = object_panel_y + 40
        for line in object_entries:
            text_surface = font.render(line, True, self._theme_color("muted", (210, 220, 230)))
            self.screen.blit(text_surface, (rect.x + 16, text_y))
            text_y += 18
            if text_y > object_panel_y + object_panel_height - 20:
                break

        status_text = "ESC = quit  •  simple view"
        status_surface = font.render(status_text, True, self._theme_color("muted", (170, 180, 195)))
        self.screen.blit(status_surface, (padding, self.height - 40))

    def _draw_map_preview(self, x: int, y: int, width: int, height: int) -> None:
        if self.occupancy_map is None:
            return

        panel = pygame.Rect(x, y, width, height)
        pygame.draw.rect(self.screen, (18, 20, 28), panel, border_radius=12)
        pygame.draw.rect(self.screen, (70, 80, 95), panel, 1, border_radius=12)

        title_font = pygame.font.SysFont("Arial", 16, bold=True)
        title_surface = title_font.render("Map", True, (240, 245, 250))
        self.screen.blit(title_surface, (x + 12, y + 10))

        map_surface = self.occupancy_map.to_surface(max(1, width - 20), max(1, height - 40))
        self.screen.blit(map_surface, (x + 10, y + 32))

        stats_font = pygame.font.SysFont("Arial", 12)
        stats_text = (
            f"Occ {self.map_stats.get('occupied_cells', 0)}  "
            f"Free {self.map_stats.get('free_cells', 0)}  "
            f"Unknown {self.map_stats.get('unknown_cells', 0)}"
        )
        stats_surface = stats_font.render(stats_text, True, (205, 212, 220))
        self.screen.blit(stats_surface, (x + 12, y + height - 20))


    def set_navigation_engine(self, engine: Any) -> None:
        self.navigation_engine = engine

    def _screen_to_world(self, screen_x: int, screen_y: int) -> tuple[float, float]:
        if self.screen is None:
            return 0.0, 0.0
        width, height = self.screen.get_size()
        center_x = width / 2 + self.pan_x
        center_y = height / 2 + self.pan_y
        base_radius = min(width, height) * 0.34
        scale_mm_per_px = self.max_distance_mm / base_radius
        world_x = (screen_x - center_x - self.pan_x) * scale_mm_per_px
        world_y = (screen_y - center_y - self.pan_y) * scale_mm_per_px
        return world_x, world_y

    def _prompt_goal_input(self) -> None:
        if self.navigation_engine is None:
            return
        try:
            raw = input("Enter goal coordinates as x_mm,y_mm: ")
            x_str, y_str = raw.strip().split(",")
            x_mm = float(x_str)
            y_mm = float(y_str)
            from .navigation import NavigationRequest

            if self.navigation_engine.request_goal(NavigationRequest(goal_x_mm=x_mm, goal_y_mm=y_mm)):
                print(f"Navigation goal set to ({x_mm}, {y_mm})")
            else:
                print("Goal request failed")
        except Exception as exc:
            print(f"Goal input invalid: {exc}")

    def _toggle_recording(self) -> None:
        if self.recording_session is None:
            return
        if self.recording_session.paused:
            self.recording_session.resume()
            self.platform.log_event("Recording resumed") if self.platform else None
        else:
            self.recording_session.pause()
            self.platform.log_event("Recording paused") if self.platform else None

    def _toggle_playback(self) -> None:
        if self.platform is None or self.platform.playback_manager is None:
            return
        playback = self.platform.playback_manager
        if playback.is_playing:
            playback.pause()
            self.platform.log_event("Playback paused")
        else:
            playback.play()
            self.platform.log_event("Playback started")

    def _save_configuration(self) -> None:
        if self.platform is None:
            return
        self.platform.config.save_profile()
        self.platform.log_event("Configuration saved")

    def _export_session(self) -> None:
        if self.platform is None:
            return
        self.platform.log_event("Session exported")

    def _clear_event_log(self) -> None:
        if self.platform is None:
            return
        self.platform.event_log.clear()
        self.platform.log_event("Event log cleared")

    def _cycle_theme(self) -> None:
        if self.platform is None or self.platform.theme_manager is None:
            return
        theme_names = list(self.platform.theme_manager.themes.keys())
        if not theme_names:
            return
        current_name = self.platform.theme_manager.current_theme.name
        try:
            next_index = (theme_names.index(current_name) + 1) % len(theme_names)
        except ValueError:
            next_index = 0
        next_theme = theme_names[next_index]
        self.platform.theme_manager.set_theme(next_theme)
        self.apply_theme(self.platform.theme_manager.current_theme)
        self.platform.log_event(f"Theme switched to {next_theme}")

    def _toggle_fullscreen(self) -> None:
        if self._headless or self.screen is None:
            return
        flags = self.screen.get_flags()
        is_fullscreen = bool(flags & pygame.FULLSCREEN)
        if is_fullscreen:
            pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)
        else:
            pygame.display.set_mode((self.width, self.height), pygame.FULLSCREEN)

    def _draw_diagnostics_overlay(self, x: int, y: int, width: int, height: int) -> None:
        if self.platform is None:
            return
        panel = pygame.Rect(x, y, width, height)
        pygame.draw.rect(self.screen, self._theme_color("panel", (18, 20, 28)), panel, border_radius=12)
        pygame.draw.rect(self.screen, self._theme_color("border", (70, 80, 95)), panel, 1, border_radius=12)

        title_font = pygame.font.SysFont("Arial", 16, bold=True)
        info_font = pygame.font.SysFont("Arial", 13)
        title_surface = title_font.render("Diagnostics", True, self._theme_color("text", (235, 240, 248)))
        self.screen.blit(title_surface, (x + 12, y + 10))

        report = self.platform.diagnostics.last_report
        lines = [
            f"Errors: {len(self.platform.diagnostics.recent_errors)}",
            f"Threads: {threading.active_count()}",
            f"CPU: {self.platform.performance_monitor.metrics.get('cpu_percent', 0.0)}%",
            f"Mem: {self.platform.performance_monitor.metrics.get('memory_percent', 0.0)}%",
            f"Disk: {self.platform.performance_monitor.metrics.get('disk_percent', 0.0)}%",
            f"Radar FPS: {self.platform.performance_monitor.metrics.get('radar_fps', 0.0):.1f}",
            f"AI FPS: {self.platform.performance_monitor.metrics.get('ai_fps', 0.0):.1f}",
            f"SLAM FPS: {self.platform.performance_monitor.metrics.get('slam_fps', 0.0):.1f}",
            f"Nav FPS: {self.platform.performance_monitor.metrics.get('navigation_fps', 0.0):.1f}",
        ]
        if report is not None:
            lines.extend([
                f"LiDAR: {report.device_status.get('LiDAR', 'Unknown')}",
                f"Storage: {report.storage_status.get('available_gb', 'N/A')} GB free",
            ])

        y_offset = y + 38
        for line in lines:
            text_surface = info_font.render(line, True, self._theme_color("muted", (210, 220, 230)))
            self.screen.blit(text_surface, (x + 12, y_offset))
            y_offset += 18

    pass

    def _draw_navigation_overlay(self, center_x: float, center_y: float, scale_mm_per_px: float) -> None:
        if self.navigation_engine is None:
            return
        status = self.navigation_engine.status
        if not status.waypoints:
            return

        path_points = [
            cartesian_to_screen(wp.x_mm, wp.y_mm, center_x, center_y, scale_mm_per_px, self.pan_x, self.pan_y)
            for wp in status.waypoints
        ]
        if len(path_points) >= 2:
            pygame.draw.lines(self.screen, (90, 220, 220), False, path_points, 3)

        if status.current_waypoint_index < len(path_points):
            current_point = path_points[status.current_waypoint_index]
            pygame.draw.circle(self.screen, (255, 235, 59), current_point, 8, 2)

        if status.goal is not None:
            goal_screen = cartesian_to_screen(status.goal.x_mm, status.goal.y_mm, center_x, center_y, scale_mm_per_px, self.pan_x, self.pan_y)
            pygame.draw.circle(self.screen, (255, 80, 80), goal_screen, 10, 2)
            pygame.draw.line(
                self.screen,
                (255, 80, 80),
                (goal_screen[0] - 12, goal_screen[1] - 12),
                (goal_screen[0] + 12, goal_screen[1] + 12),
                2,
            )
            pygame.draw.line(
                self.screen,
                (255, 80, 80),
                (goal_screen[0] + 12, goal_screen[1] - 12),
                (goal_screen[0] - 12, goal_screen[1] + 12),
                2,
            )

    def _draw_navigation_dashboard(self, x: int, y: int, width: int, height: int) -> None:
        panel = pygame.Rect(x, y, width, height)
        pygame.draw.rect(self.screen, (18, 20, 28), panel, border_radius=12)
        pygame.draw.rect(self.screen, (70, 80, 95), panel, 1, border_radius=12)

        title_font = pygame.font.SysFont("Arial", 16, bold=True)
        title_surface = title_font.render("Navigation Dashboard", True, (245, 250, 255))
        self.screen.blit(title_surface, (x + 12, y + 10))

        if self.navigation_engine is None:
            info_font = pygame.font.SysFont("Arial", 13)
            text_surface = info_font.render("No navigation engine", True, (220, 230, 240))
            self.screen.blit(text_surface, (x + 12, y + 40))
            return

        status = self.navigation_engine.status
        info_font = pygame.font.SysFont("Arial", 13)
        entries = [
            ("State", status.state.value),
            ("Goal", f"{status.goal.x_mm:.0f},{status.goal.y_mm:.0f}" if status.goal else "None"),
            ("Position", f"{status.current_pose.x_mm:.0f},{status.current_pose.y_mm:.0f}"),
            ("Remaining", f"{status.remaining_distance_mm/1000:.2f} m"),
            ("ETA", f"{status.estimated_time_s:.1f}s"),
            ("Speed", f"{status.current_speed_m_s:.2f} m/s"),
            ("Path len", f"{status.path_length_mm/1000:.2f} m"),
            ("Waypt", f"{status.current_waypoint_index+1}/{len(status.waypoints)}"),
            ("Replans", str(status.replan_count)),
            ("Dyn obs", str(status.dynamic_obstacle_count)),
            ("Plan ms", f"{status.last_planning_time_ms:.0f}"),
            ("Nav FPS", f"{status.navigation_fps:.1f}"),
        ]
        y_offset = y + 38
        for label, value in entries:
            text_surface = info_font.render(f"{label}: {value}", True, (220, 230, 240))
            self.screen.blit(text_surface, (x + 12, y_offset))
            y_offset += 18

    def _draw_event_log(self, x: int, y: int, width: int, height: int) -> None:
        panel = pygame.Rect(x, y, width, height)
        pygame.draw.rect(self.screen, self._theme_color("panel", (18, 20, 28)), panel, border_radius=12)
        pygame.draw.rect(self.screen, self._theme_color("border", (70, 80, 95)), panel, 1, border_radius=12)

        title_font = pygame.font.SysFont("Arial", 16, bold=True)
        title_surface = title_font.render("Event Log", True, (245, 250, 255))
        self.screen.blit(title_surface, (x + 12, y + 10))

        log_font = pygame.font.SysFont("Arial", 13)
        y_offset = y + 36

        if self.platform is not None:
            with self._points_lock:
                events = list(self.platform.event_log)
        elif self.navigation_engine is not None:
            with self._points_lock:
                events = list(self.navigation_engine.status.event_log)
        else:
            events = []

        if not events:
            message = log_font.render("No events available", True, (220, 230, 240))
            self.screen.blit(message, (x + 12, y_offset))
            return

        for event_text in events[:8]:
            line_surface = log_font.render(event_text, True, (220, 230, 240))
            self.screen.blit(line_surface, (x + 12, y_offset))
            y_offset += 18
            if y_offset > y + height - 20:
                break

    def _risk_rank(self, risk_level: str) -> int:
        return {"SAFE": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}.get(risk_level, 0)
