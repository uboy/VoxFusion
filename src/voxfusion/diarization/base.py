"""DiarizationEngine protocol definition."""

from collections.abc import AsyncIterator
from typing import Protocol

from voxfusion.models.audio import AudioChunk
from voxfusion.models.diarization import DiarizedSegment
from voxfusion.models.transcription import TranscriptionSegment


class DiarizationEngine(Protocol):
    """Speaker diarization engine interface."""

    async def diarize(
        self,
        segments: list[TranscriptionSegment],
        audio: AudioChunk | None = None,
    ) -> list[DiarizedSegment]:
        """Assign speaker identities to transcription segments."""
        ...

    async def diarize_stream(
        self,
        segment_stream: AsyncIterator[tuple[TranscriptionSegment, AudioChunk]],
    ) -> AsyncIterator[DiarizedSegment]:
        """Streaming diarization: yields diarized segments as input arrives."""
        ...
