import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

try:
    from ultralytics import YOLO
except ImportError as exc:
    YOLO = None
    _ultralytics_import_error = exc
import numpy as np


@dataclass(slots=True)
class AIDetection:
    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]
    center_x: float
    center_y: float
    width: float
    height: float
    timestamp: float


class AIDetector:
    """AI object detection using a YOLO model."""

    def __init__(self, model_path: str, confidence_threshold: float = 0.35, debug: bool = False) -> None:
        if YOLO is None:
            raise ImportError(
                "ultralytics is required for AIDetector. "
                "Install ultralytics or disable AI detection features."
            ) from _ultralytics_import_error
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.debug = debug
        self.model = self._load_model(model_path)

    def _load_model(self, model_path: str) -> YOLO:
        if not Path(model_path).exists():
            raise FileNotFoundError(f"YOLO model not found: {model_path}")
        model = YOLO(model_path)
        return model

    def infer(self, frame: Any) -> tuple[List[AIDetection], float]:
        start_time = time.perf_counter()
        results = self.model(frame, imgsz=max(frame.shape[0], frame.shape[1]), conf=self.confidence_threshold, verbose=False)
        detections: list[AIDetection] = []
        inference_time_ms = (time.perf_counter() - start_time) * 1000.0
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes.data.tolist():
                x1, y1, x2, y2, score, cls_id = box
                class_name = self.model.names.get(int(cls_id), f"class_{int(cls_id)}")
                if score < self.confidence_threshold:
                    continue
                bbox = (int(x1), int(y1), int(x2), int(y2))
                width = x2 - x1
                height = y2 - y1
                center_x = x1 + width / 2.0
                center_y = y1 + height / 2.0
                detections.append(
                    AIDetection(
                        class_name=class_name,
                        confidence=float(score),
                        bbox=bbox,
                        center_x=center_x,
                        center_y=center_y,
                        width=float(width),
                        height=float(height),
                        timestamp=time.time(),
                    )
                )
        return detections, inference_time_ms
