from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from config import (
    AWARENESS_ZONE_RADIUS_MM,
    CRITICAL_ZONE_RADIUS_MM,
    ENABLE_COLLISION_PREDICTION,
    ENABLE_SAFETY_ZONES,
    RISK_THRESHOLDS,
    WARNING_ZONE_RADIUS_MM,
)
from .models import TrackedObject


class SafetyAnalyzer:
    """Monitor safety zones and predict collision risk for tracked objects."""

    def __init__(self) -> None:
        self._enabled = ENABLE_SAFETY_ZONES
        self._collision_enabled = ENABLE_COLLISION_PREDICTION

    def get_zone_for_distance(self, distance_mm: float) -> Dict[str, Any]:
        if not self._enabled:
            return {"name": "Safe", "level": 0, "min_mm": 0.0, "max_mm": float("inf")}
        if distance_mm <= CRITICAL_ZONE_RADIUS_MM:
            return {"name": "Critical", "level": 4, "min_mm": 0.0, "max_mm": CRITICAL_ZONE_RADIUS_MM}
        if distance_mm <= WARNING_ZONE_RADIUS_MM:
            return {"name": "Warning", "level": 3, "min_mm": CRITICAL_ZONE_RADIUS_MM, "max_mm": WARNING_ZONE_RADIUS_MM}
        if distance_mm <= AWARENESS_ZONE_RADIUS_MM:
            return {"name": "Awareness", "level": 2, "min_mm": WARNING_ZONE_RADIUS_MM, "max_mm": AWARENESS_ZONE_RADIUS_MM}
        return {"name": "Safe", "level": 1, "min_mm": AWARENESS_ZONE_RADIUS_MM, "max_mm": float("inf")}

    def analyze_track(self, track: TrackedObject, motion: Dict[str, Any], current_distance_mm: float, dt_s: float) -> Dict[str, Any]:
        zone = self.get_zone_for_distance(current_distance_mm)
        relative_speed = motion.get("velocity_m_s", 0.0)
        closing_speed = max(relative_speed, 0.0)
        ttc_s = current_distance_mm / 1000.0 / max(closing_speed, 1e-3) if closing_speed > 0 else float("inf")
        collision_confidence = self._estimate_collision_confidence(current_distance_mm, closing_speed, ttc_s)
        risk_level = self._classify_risk(collision_confidence, ttc_s)
        return {
            "zone": zone,
            "relative_speed_m_s": relative_speed,
            "closing_speed_m_s": closing_speed,
            "ttc_s": ttc_s,
            "collision_confidence": collision_confidence,
            "risk_level": risk_level,
            "enabled": self._collision_enabled,
        }

    def _estimate_collision_confidence(self, distance_mm: float, closing_speed: float, ttc_s: float) -> float:
        if not self._collision_enabled:
            return 0.0
        proximity_factor = max(0.0, 1.0 - (distance_mm / max(AWARENESS_ZONE_RADIUS_MM, 1.0)))
        speed_factor = min(1.0, closing_speed / 3.0)
        ttc_factor = 0.0 if math.isinf(ttc_s) else max(0.0, 1.0 - (ttc_s / 4.0))
        return round(min(1.0, 0.2 + 0.45 * proximity_factor + 0.25 * speed_factor + 0.1 * ttc_factor), 3)

    def _classify_risk(self, collision_confidence: float, ttc_s: float) -> str:
        if collision_confidence <= RISK_THRESHOLDS.get("safe", 0.15):
            return "SAFE"
        if collision_confidence <= RISK_THRESHOLDS.get("low", 0.3):
            return "LOW"
        if collision_confidence <= RISK_THRESHOLDS.get("medium", 0.6):
            return "MEDIUM"
        if collision_confidence <= RISK_THRESHOLDS.get("high", 0.85):
            return "HIGH"
        return "CRITICAL"
