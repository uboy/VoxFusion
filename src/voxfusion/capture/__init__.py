"""Audio capture subsystem: platform-specific audio input sources."""

from voxfusion.capture.base import AudioCaptureSource, AudioDeviceEnumerator
from voxfusion.capture.enumerator import SoundDeviceEnumerator
from voxfusion.capture.factory import create_capture_source, create_file_source, detect_platform
from voxfusion.capture.file_source import FileAudioSource

__all__ = [
    "AudioCaptureSource",
    "AudioDeviceEnumerator",
    "FileAudioSource",
    "SoundDeviceEnumerator",
    "create_capture_source",
    "create_file_source",
    "detect_platform",
]
