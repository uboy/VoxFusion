"""Audio data models: AudioChunk and AudioDeviceInfo."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class AudioChunk:
    """A chunk of raw or processed audio data.

    Attributes:
        samples: Audio samples array, shape ``(num_samples,)`` for mono
            or ``(num_samples, channels)`` for multi-channel.
        sample_rate: Sample rate in Hz (e.g. 16000).
        channels: Number of audio channels (1=mono, 2=stereo).
        timestamp_start: Start time in seconds from capture origin.
        timestamp_end: End time in seconds from capture origin.
        source: Origin of audio — ``"system"``, ``"microphone"``,
            ``"file"``, or ``"mixed"``.
        dtype: Sample data type — ``"float32"`` or ``"int16"``.
    """

    samples: NDArray[np.float32 | np.int16]
    sample_rate: int
    channels: int
    timestamp_start: float
    timestamp_end: float
    source: str
    dtype: str = "float32"

    @property
    def duration(self) -> float:
        """Duration in seconds."""
        return self.timestamp_end - self.timestamp_start

    @property
    def num_samples(self) -> int:
        """Total number of audio samples (first axis length)."""
        return int(self.samples.shape[0])


@dataclass(frozen=True)
class AudioDeviceInfo:
    """Metadata about an audio device.

    Attributes:
        id: Unique device identifier within VoxFusion.
        name: Human-readable device name.
        sample_rate: Native sample rate in Hz.
        channels: Number of channels.
        device_type: ``"input"``, ``"loopback"``, or ``"virtual"``.
        is_default: Whether this is the system default device.
        platform_id: OS-specific device identifier.
    """

    id: str
    name: str
    sample_rate: int
    channels: int
    device_type: str
    is_default: bool
    platform_id: str
