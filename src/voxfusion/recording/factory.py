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
    device_index: int | None = None,
) -> AudioCaptureSource | AudioMixer:
    """Create a live capture source suitable for raw audio recording."""
    if sys.platform == "win32":
        from voxfusion.capture.wasapi import WASAPICapture, find_stereo_mix_device

        def _create_system_source() -> WASAPICapture:
            stereo_mix_idx = find_stereo_mix_device()
            if stereo_mix_idx is not None:
                return WASAPICapture(
                    device_index=stereo_mix_idx,
                    loopback=False,
                    source_label="system",
                    config=config,
                )
            return WASAPICapture(
                device_index=None,
                loopback=True,
                config=config,
            )

        if source_type == "both":
            return AudioMixer(
                sources=[
                    WASAPICapture(device_index=device_index, loopback=False, config=config),
                    _create_system_source(),
                ]
            )
        if source_type == "system":
            return _create_system_source()
        return WASAPICapture(
            device_index=device_index,
            loopback=False,
            config=config,
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
