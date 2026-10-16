import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass(slots=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    distortion: list[float] = field(default_factory=list)


@dataclass(slots=True)
class Extrinsics:
    tx_mm: float
    ty_mm: float
    tz_mm: float
    roll_deg: float
    pitch_deg: float
    yaw_deg: float


@dataclass(slots=True)
class CalibrationData:
    intrinsics: CameraIntrinsics
    extrinsics: Extrinsics
    lidar_offset_mm: tuple[float, float, float]
    camera_offset_mm: tuple[float, float, float]
    fov_deg: float
    yaw_offset_deg: float
    pitch_offset_deg: float
    roll_offset_deg: float

    @classmethod
    def load(cls, path: str) -> "CalibrationData":
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Calibration file not found: {path}")
        data = json.loads(config_path.read_text())
        intrinsics = CameraIntrinsics(**data["intrinsics"])
        extrinsics = Extrinsics(**data["extrinsics"])
        return cls(
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            lidar_offset_mm=tuple(data.get("lidar_offset_mm", [0.0, 0.0, 0.0])),
            camera_offset_mm=tuple(data.get("camera_offset_mm", [0.0, 0.0, 0.0])),
            fov_deg=float(data.get("fov_deg", 90.0)),
            yaw_offset_deg=float(data.get("yaw_offset_deg", 0.0)),
            pitch_offset_deg=float(data.get("pitch_offset_deg", 0.0)),
            roll_offset_deg=float(data.get("roll_offset_deg", 0.0)),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intrinsics": {
                "fx": self.intrinsics.fx,
                "fy": self.intrinsics.fy,
                "cx": self.intrinsics.cx,
                "cy": self.intrinsics.cy,
                "distortion": self.intrinsics.distortion,
            },
            "extrinsics": {
                "tx_mm": self.extrinsics.tx_mm,
                "ty_mm": self.extrinsics.ty_mm,
                "tz_mm": self.extrinsics.tz_mm,
                "roll_deg": self.extrinsics.roll_deg,
                "pitch_deg": self.extrinsics.pitch_deg,
                "yaw_deg": self.extrinsics.yaw_deg,
            },
            "lidar_offset_mm": list(self.lidar_offset_mm),
            "camera_offset_mm": list(self.camera_offset_mm),
            "fov_deg": self.fov_deg,
            "yaw_offset_deg": self.yaw_offset_deg,
            "pitch_offset_deg": self.pitch_offset_deg,
            "roll_offset_deg": self.roll_offset_deg,
        }
