"""AudioPreProcessor and VADFilter protocol definitions."""

from typing import Protocol

from voxfusion.models.audio import AudioChunk


class AudioPreProcessor(Protocol):
    """Transforms raw audio into a format suitable for ASR."""

    def process(self, chunk: AudioChunk) -> AudioChunk:
        """Apply pre-processing to an audio chunk."""
        ...

    def reset(self) -> None:
        """Reset any internal state."""
        ...


class VADFilter(Protocol):
    """Voice Activity Detection filter."""

    def contains_speech(self, chunk: AudioChunk) -> bool:
        """Return ``True`` if the chunk likely contains speech."""
        ...

    def get_speech_segments(self, chunk: AudioChunk) -> list[tuple[float, float]]:
        """Return ``(start, end)`` pairs of speech regions within the chunk."""
        ...
