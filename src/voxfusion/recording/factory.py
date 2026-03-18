"""Factory helpers for raw audio recording sources."""

from __future__ import annotations

import sys

from voxfusion.capture.base import AudioCaptureSource
from voxfusion.capture.mixer import AudioMixer
from voxfusion.config.models import CaptureConfig
from voxfusion.exceptions import UnsupportedPlatformError


def create_recording_source(
    source_type: str,
    config: CaptureConfig,
    *,
    device_index: str | int | None = None,
    microphone_device_id: str | int | None = None,
    system_device_id: str | int | None = None,
) -> AudioCaptureSource | AudioMixer:
    """Create a live capture source suitable for raw audio recording."""
    if sys.platform == "win32":
        from voxfusion.capture.windows_factory import create_windows_capture_source

        return create_windows_capture_source(
            source_type,
            config,
            microphone_device_id=(
                microphone_device_id
                if microphone_device_id is not None
                else device_index if source_type != "system" else None
            ),
            system_device_id=(
                system_device_id
                if system_device_id is not None
                else device_index if source_type == "system" else None
            ),
        )

    if sys.platform.startswith("linux"):
        from voxfusion.capture.pulseaudio import PulseAudioCapture

        if source_type == "both":
            return AudioMixer(
                sources=[
                    PulseAudioCapture(device_index=device_index, loopback=False, config=config),
                    PulseAudioCapture(device_index=None, loopback=True, config=config),
                ]
            )
        return PulseAudioCapture(
            device_index=device_index,
            loopback=(source_type == "system"),
            config=config,
        )

    raise UnsupportedPlatformError(
        f"Raw audio recording for source '{source_type}' is not supported on platform {sys.platform}."
    )
