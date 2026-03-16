"""Platform-detection factory for audio capture sources.

Detects the current platform and returns the appropriate audio
capture source implementation. Falls back to ``sounddevice``-based
capture if no platform-specific implementation is available.
"""

import sys
from pathlib import Path

from voxfusion.capture.base import AudioCaptureSource
from voxfusion.capture.file_source import FileAudioSource
from voxfusion.config.models import CaptureConfig
from voxfusion.exceptions import UnsupportedPlatformError
from voxfusion.logging import get_logger

log = get_logger(__name__)


def detect_platform() -> str:
    """Detect the current audio platform.

    Returns:
        One of ``"wasapi"``, ``"coreaudio"``, ``"pulseaudio"``,
        or ``"generic"``.
    """
    match sys.platform:
        case "win32":
            return "wasapi"
        case "darwin":
            return "coreaudio"
        case platform if platform.startswith("linux"):
            return "pulseaudio"
        case _:
            return "generic"


def create_file_source(
    path: Path,
    config: CaptureConfig | None = None,
) -> FileAudioSource:
    """Create a file-based capture source for batch processing.

    Args:
        path: Path to the audio file.
        config: Optional capture configuration.
    """
    chunk_ms = (config or CaptureConfig()).chunk_duration_ms
    return FileAudioSource(path, chunk_duration_ms=chunk_ms)


def create_capture_source(
    source_type: str = "microphone",
    config: CaptureConfig | None = None,
) -> AudioCaptureSource:
    """Create a platform-appropriate live capture source.

    Args:
        source_type: ``"microphone"`` or ``"system"`` (loopback).
        config: Capture configuration.

    Returns:
        An ``AudioCaptureSource`` instance.

    Raises:
        UnsupportedPlatformError: If no capture implementation exists
            for the current platform.

    .. note::
        Live capture implementations (WASAPI, CoreAudio, PulseAudio)
        are not yet implemented. This factory currently raises
        ``UnsupportedPlatformError`` for live capture requests.
    """
    platform = detect_platform()
    log.info("capture.platform_detected", platform=platform, source_type=source_type)

    raise UnsupportedPlatformError(
        f"Live {source_type} capture on {platform} is not yet implemented. "
        f"Use file-based capture with 'voxfusion transcribe <file>' instead."
    )
