"""Shared Windows capture source construction."""

from __future__ import annotations

from voxfusion.capture.base import AudioCaptureSource
from voxfusion.capture.mixer import AudioMixer
from voxfusion.capture.wasapi import RobustLoopbackCapture, WASAPICapture
from voxfusion.capture.windows_audio import parse_windows_device_id
from voxfusion.config.models import CaptureConfig


def _sounddevice_index(device_id: str | int | None) -> int | None:
    backend, native_index = parse_windows_device_id(device_id, default_backend="sd")
    if backend is None:
        return None
    if backend != "sd":
        raise ValueError(f"Microphone capture requires a sounddevice/WASAPI device, got '{device_id}'.")
    return native_index


def create_windows_capture_source(
    source_type: str,
    config: CaptureConfig,
    *,
    microphone_device_id: str | int | None = None,
    system_device_id: str | int | None = None,
) -> AudioCaptureSource | AudioMixer:
    """Create Windows capture sources with explicit mic/system device ids."""
    if source_type == "both":
        return AudioMixer(
            sources=[
                WASAPICapture(
                    device_index=_sounddevice_index(microphone_device_id),
                    loopback=False,
                    config=config,
                ),
                RobustLoopbackCapture(
                    device_id=system_device_id,
                    config=config,
                ),
            ]
        )
    if source_type == "system":
        return RobustLoopbackCapture(
            device_id=system_device_id,
            config=config,
        )
    return WASAPICapture(
        device_index=_sounddevice_index(microphone_device_id),
        loopback=False,
        config=config,
    )
