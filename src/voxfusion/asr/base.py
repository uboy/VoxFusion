"""ASREngine protocol definition."""

from collections.abc import AsyncIterator
from typing import Protocol

from voxfusion.models.audio import AudioChunk
from voxfusion.models.transcription import TranscriptionSegment


class ASREngine(Protocol):
    """Speech-to-text engine interface."""

    @property
    def model_name(self) -> str: ...

    @property
    def supported_languages(self) -> list[str]: ...

    async def transcribe(
        self,
        audio: AudioChunk,
        *,
        language: str | None = None,
        initial_prompt: str | None = None,
        word_timestamps: bool = False,
    ) -> list[TranscriptionSegment]:
        """Transcribe an audio chunk to text segments."""
        ...

    async def transcribe_stream(
        self,
        audio_stream: AsyncIterator[AudioChunk],
        *,
        language: str | None = None,
    ) -> AsyncIterator[TranscriptionSegment]:
        """Streaming transcription: yields segments as audio arrives."""
        ...

    def load_model(self) -> None:
        """Pre-load the ASR model into memory."""
        ...

    def unload_model(self) -> None:
        """Release the ASR model from memory."""
        ...

    def close(self) -> None:
        """Release background resources (executors, handles)."""
        ...
