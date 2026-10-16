from __future__ import annotations

import threading
import time
import queue
from dataclasses import dataclass
from typing import Optional, Tuple, Any

try:
    import cv2
except ImportError as exc:
    cv2 = None
    _opencv_import_error = exc


@dataclass(slots=True)
class CameraFrame:
    frame: Any
    timestamp: float


class CameraSource:
    """Asynchronous camera capture supporting USB, built-in, and RTSP sources."""

    def __init__(
        self,
        source: str,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        reconnect_interval: float = 2.0,
        debug: bool = False,
    ) -> None:
        if cv2 is None:
            raise ImportError(
                "opencv-python is required for CameraSource. "
                "Install opencv-python or disable camera features."
            ) from _opencv_import_error
        self.source = self._normalize_source(source)
        self.width = width
        self.height = height
        self.fps = fps
        self.reconnect_interval = reconnect_interval
        self.debug = debug
        self._capture: Optional[cv2.VideoCapture] = None
        self._frame_queue: queue.Queue[CameraFrame] = queue.Queue(maxsize=4)
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    @staticmethod
    def _normalize_source(source: str) -> int | str:
        if source.isdigit():
            return int(source)
        return source

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        with self._lock:
            if self._capture is not None:
                try:
                    self._capture.release()
                except Exception:
                    pass
                self._capture = None

    def _configure_capture(self) -> None:
        if self._capture is None:
            return
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._capture.set(cv2.CAP_PROP_FPS, self.fps)

    def _open_capture(self) -> bool:
        with self._lock:
            try:
                self._capture = cv2.VideoCapture(self.source, cv2.CAP_ANY)
                if not self._capture.isOpened():
                    self._capture.release()
                    self._capture = None
                    return False
                self._configure_capture()
                return True
            except Exception:
                self._capture = None
                return False

    def read(self, timeout: float = 0.2) -> Optional[CameraFrame]:
        try:
            return self._frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def _capture_loop(self) -> None:
        while self._running:
            if self._capture is None or not self._capture.isOpened():
                if self.debug:
                    print("[CameraSource] opening camera source", self.source)
                if not self._open_capture():
                    if self.debug:
                        print("[CameraSource] failed to open camera, retrying in", self.reconnect_interval)
                    time.sleep(self.reconnect_interval)
                    continue

            with self._lock:
                capture = self._capture
            if capture is None:
                continue

            ret, frame = capture.read()
            if not ret or frame is None:
                if self.debug:
                    print("[CameraSource] lost camera frame, reconnecting")
                with self._lock:
                    if self._capture is not None:
                        self._capture.release()
                        self._capture = None
                time.sleep(self.reconnect_interval)
                continue

            timestamp = time.time()
            if self.width > 0 and self.height > 0 and (frame.shape[1] != self.width or frame.shape[0] != self.height):
                frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)

            frame_data = CameraFrame(frame=frame, timestamp=timestamp)
            try:
                self._frame_queue.put(frame_data, timeout=0.05)
            except queue.Full:
                try:
                    _ = self._frame_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._frame_queue.put(frame_data, timeout=0.05)
                except queue.Full:
                    pass

            time.sleep(max(0.0, 1.0 / max(self.fps, 1)))
