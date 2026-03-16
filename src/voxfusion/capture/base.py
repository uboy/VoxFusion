"""AudioCaptureSource and AudioDeviceEnumerator protocol definitions."""

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from voxfusion.models.audio import AudioChunk, AudioDeviceInfo


@runtime_checkable
class AudioCaptureSource(Protocol):
    """Interface for a single audio capture source (mic or system loopback)."""

    @property
    def device_name(self) -> str: ...

    @property
    def sample_rate(self) -> int: ...

    @property
    def channels(self) -> int: ...

    @property
    def is_active(self) -> bool: ...

    async def start(self) -> None:
        """Begin capturing audio."""
        ...

    async def stop(self) -> None:
        """Stop capturing and release resources."""
        ...

    async def read_chunk(self, duration_ms: int = 500) -> AudioChunk:
        """Read the next chunk of audio data."""
        ...

    def stream(self, chunk_duration_ms: int = 500) -> AsyncIterator[AudioChunk]:
        """Yield audio chunks as an async iterator."""
        ...


class AudioDeviceEnumerator(Protocol):
    """Enumerates available audio capture devices on the current platform."""

    def list_input_devices(self) -> list[AudioDeviceInfo]: ...

    def list_loopback_devices(self) -> list[AudioDeviceInfo]: ...

    def get_default_input_device(self) -> AudioDeviceInfo | None: ...

    def get_default_loopback_device(self) -> AudioDeviceInfo | None: ...
