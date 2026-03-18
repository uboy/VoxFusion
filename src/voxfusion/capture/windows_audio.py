"""Shared Windows audio device discovery and device-id parsing."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WindowsAudioDevice:
    """User-selectable Windows audio endpoint."""

    id: str
    label: str
    name: str
    backend: str
    native_index: int
    kind: str
    is_default: bool = False


def parse_windows_device_id(
    device_id: str | int | None,
    *,
    default_backend: str = "sd",
) -> tuple[str | None, int | None]:
    """Parse an opaque Windows device id into backend + native index."""
    if device_id is None:
        return None, None
    if isinstance(device_id, int):
        return default_backend, device_id
    text = str(device_id).strip()
    if not text:
        return None, None
    if ":" in text:
        backend, raw_index = text.split(":", 1)
        return backend, int(raw_index)
    return default_backend, int(text)


def list_windows_microphone_devices() -> list[WindowsAudioDevice]:
    """List Windows microphone/input devices using WASAPI via sounddevice."""
    try:
        import sounddevice as sd
    except Exception:
        return []

    try:
        hostapis = list(sd.query_hostapis())
        hostapi_names = {
            i: str(api.get("name", f"HostAPI {i}"))
            for i, api in enumerate(hostapis)
        }
        devices = list(sd.query_devices())
        default_input = int(sd.default.device[0]) if sd.default.device[0] is not None else -1
    except Exception:
        return []

    result: list[WindowsAudioDevice] = []
    for idx, dev in enumerate(devices):
        hostapi_idx = int(dev.get("hostapi", -1))
        hostapi_name = hostapi_names.get(hostapi_idx, f"HostAPI {hostapi_idx}")
        if "wasapi" not in hostapi_name.lower():
            continue
        channels = int(dev.get("max_input_channels", 0))
        if channels <= 0:
            continue
        name = str(dev.get("name", f"Device {idx}"))
        result.append(
            WindowsAudioDevice(
                id=f"sd:{idx}",
                label=f"{name} [{hostapi_name} #{idx}]",
                name=name,
                backend="sd",
                native_index=idx,
                kind="microphone",
                is_default=(idx == default_input),
            )
        )
    return result


def list_windows_loopback_devices() -> list[WindowsAudioDevice]:
    """List Windows system-audio devices, preferring pyaudiowpatch loopback endpoints."""
    result = _list_pyaudiowpatch_loopback_devices()
    if result:
        return result
    return _list_sounddevice_wasapi_loopback_devices()


def _list_pyaudiowpatch_loopback_devices() -> list[WindowsAudioDevice]:
    try:
        import pyaudiowpatch as pyaudio  # type: ignore[import-not-found]
    except Exception:
        return []

    pa = pyaudio.PyAudio()
    try:
        default_output_name = ""
        try:
            wasapi_info = pa.get_host_api_info_by_type(pyaudio.paWASAPI)
            default_out = pa.get_device_info_by_index(wasapi_info["defaultOutputDevice"])
            default_output_name = str(default_out.get("name", ""))
        except Exception:
            default_output_name = ""
        devices: list[WindowsAudioDevice] = []
        for index in range(pa.get_device_count()):
            dev = pa.get_device_info_by_index(index)
            if not dev.get("isLoopbackDevice"):
                continue
            if int(dev.get("maxInputChannels", 0)) <= 0:
                continue
            name = str(dev.get("name", f"Loopback {index}"))
            devices.append(
                WindowsAudioDevice(
                    id=f"pa:{index}",
                    label=f"{name} [PyAudioWPatch #{index}]",
                    name=name,
                    backend="pa",
                    native_index=index,
                    kind="system",
                    is_default=bool(default_output_name and default_output_name in name),
                )
            )
        return devices
    finally:
        pa.terminate()


def _list_sounddevice_wasapi_loopback_devices() -> list[WindowsAudioDevice]:
    try:
        import sounddevice as sd
    except Exception:
        return []

    try:
        hostapis = list(sd.query_hostapis())
        hostapi_names = {
            i: str(api.get("name", f"HostAPI {i}"))
            for i, api in enumerate(hostapis)
        }
        devices = list(sd.query_devices())
        default_output = int(sd.default.device[1]) if sd.default.device[1] is not None else -1
    except Exception:
        return []

    result: list[WindowsAudioDevice] = []
    for idx, dev in enumerate(devices):
        hostapi_idx = int(dev.get("hostapi", -1))
        hostapi_name = hostapi_names.get(hostapi_idx, f"HostAPI {hostapi_idx}")
        if "wasapi" not in hostapi_name.lower():
            continue
        channels = int(dev.get("max_output_channels", 0))
        if channels <= 0:
            continue
        name = str(dev.get("name", f"Device {idx}"))
        result.append(
            WindowsAudioDevice(
                id=f"sd:{idx}",
                label=f"{name} [{hostapi_name} #{idx}]",
                name=name,
                backend="sd",
                native_index=idx,
                kind="system",
                is_default=(idx == default_output),
            )
        )
    return result


def list_windows_capture_devices() -> list[WindowsAudioDevice]:
    """List user-selectable Windows capture devices for GUI selection."""
    devices = [
        *list_windows_microphone_devices(),
        *list_windows_loopback_devices(),
    ]
    return sorted(
        devices,
        key=lambda device: (
            0 if device.kind == "microphone" else 1,
            0 if device.is_default else 1,
            device.label.lower(),
        ),
    )
