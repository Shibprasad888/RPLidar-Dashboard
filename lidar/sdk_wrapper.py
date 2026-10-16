import ctypes
from pathlib import Path
from typing import Any, Optional

LIB_NAME = "sdk_wrapper_lib.so"
LEGACY_LIB_NAME = "sdk_wrapper.so"

_wrapper_path = Path(__file__).resolve().parent / LIB_NAME
if not _wrapper_path.exists():
    legacy_path = Path(__file__).resolve().parent / LEGACY_LIB_NAME
    if legacy_path.exists():
        _wrapper_path = legacy_path

if not _wrapper_path.exists():
    raise ImportError(
        "Slamtec SDK wrapper library not found. "
        "Build it from the official SDK and place it in lidar/sdk_wrapper_lib.so"
    )

_sdk = ctypes.CDLL(str(_wrapper_path))

_sdk.create_driver.restype = ctypes.c_void_p
_sdk.destroy_driver.argtypes = [ctypes.c_void_p]
_sdk.disconnect_driver.argtypes = [ctypes.c_void_p]
_sdk.disconnect_driver.restype = ctypes.c_int
_sdk.connect_driver.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
_sdk.connect_driver.restype = ctypes.c_int
_sdk.is_connected.argtypes = [ctypes.c_void_p]
_sdk.is_connected.restype = ctypes.c_int
_sdk.get_device_info.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_char_p,
    ctypes.c_int,
]
_sdk.get_device_info.restype = ctypes.c_int
_sdk.get_health.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
_sdk.get_health.restype = ctypes.c_int
_sdk.set_motor_speed.argtypes = [ctypes.c_void_p, ctypes.c_ushort]
_sdk.set_motor_speed.restype = ctypes.c_int
_sdk.start_scan.argtypes = [ctypes.c_void_p, ctypes.c_bool]
_sdk.start_scan.restype = ctypes.c_int
_sdk.stop_scan.argtypes = [ctypes.c_void_p]
_sdk.stop_scan.restype = ctypes.c_int
_sdk.grab_scan_data.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_ushort),
    ctypes.POINTER(ctypes.c_uint),
    ctypes.POINTER(ctypes.c_ubyte),
    ctypes.POINTER(ctypes.c_ubyte),
    ctypes.POINTER(ctypes.c_size_t),
    ctypes.c_uint,
]
_sdk.grab_scan_data.restype = ctypes.c_int


class SDKError(RuntimeError):
    def __init__(self, message: str, result: int | None = None) -> None:
        super().__init__(message)
        self.result = result


def _check_result(result: int, message: str) -> None:
    if result != 0:
        raise SDKError(f"Slamtec SDK error {result}: {message}", result)


class SLAMTECSDKDriver:
    def __init__(self, device: str, baudrate: int) -> None:
        self.device = device
        self.baudrate = baudrate
        self._handle = _sdk.create_driver()
        if not self._handle:
            raise SDKError("Unable to create Slamtec SDK driver")

    def __del__(self) -> None:
        self.destroy()

    def destroy(self) -> None:
        if self._handle:
            try:
                _sdk.disconnect_driver(self._handle)
            except Exception:
                pass
            _sdk.destroy_driver(self._handle)
            self._handle = None

    def connect(self) -> None:
        result = _sdk.connect_driver(self._handle, self.device.encode("utf-8"), self.baudrate)
        _check_result(result, "connect_driver failed")

    def disconnect(self) -> None:
        if self._handle:
            _sdk.disconnect_driver(self._handle)

    def is_connected(self) -> bool:
        return bool(self._handle and _sdk.is_connected(self._handle))

    def get_device_info(self) -> dict[str, Any]:
        model = ctypes.c_int()
        firmware_version = ctypes.c_int()
        hardware_version = ctypes.c_int()
        serial_number = ctypes.create_string_buffer(32)
        result = _sdk.get_device_info(
            self._handle,
            ctypes.byref(model),
            ctypes.byref(firmware_version),
            ctypes.byref(hardware_version),
            serial_number,
            ctypes.sizeof(serial_number),
        )
        _check_result(result, "get_device_info failed")
        return {
            "model": int(model.value),
            "firmware_version": int(firmware_version.value),
            "hardware_version": int(hardware_version.value),
            "serial_number": serial_number.value.decode("ascii", errors="ignore").strip("\x00"),
        }

    def get_health(self) -> tuple[int, int]:
        status = ctypes.c_int()
        error_code = ctypes.c_int()
        result = _sdk.get_health(self._handle, ctypes.byref(status), ctypes.byref(error_code))
        _check_result(result, "get_health failed")
        return int(status.value), int(error_code.value)

    def set_motor_speed(self, speed: int = 660) -> None:
        result = _sdk.set_motor_speed(self._handle, ctypes.c_ushort(speed))
        _check_result(result, "set_motor_speed failed")

    def start_scan(self, force: bool = True) -> None:
        result = _sdk.start_scan(self._handle, force)
        _check_result(result, "start_scan failed")

    def stop_scan(self) -> None:
        _sdk.stop_scan(self._handle)

    def grab_scan_data(self, max_nodes: int, timeout_ms: int) -> list[tuple[int, int, int, int]]:
        angle_q14 = (ctypes.c_ushort * max_nodes)()
        dist_q2 = (ctypes.c_uint * max_nodes)()
        quality = (ctypes.c_ubyte * max_nodes)()
        flag = (ctypes.c_ubyte * max_nodes)()
        count = ctypes.c_size_t(max_nodes)
        result = _sdk.grab_scan_data(
            self._handle,
            angle_q14,
            dist_q2,
            quality,
            flag,
            ctypes.byref(count),
            timeout_ms,
        )
        if result != 0:
            raise SDKError(f"grab_scan_data failed: {result}")
        actual = int(count.value)
        return [
            (int(angle_q14[i]), int(dist_q2[i]), int(quality[i]), int(flag[i]))
            for i in range(actual)
        ]
