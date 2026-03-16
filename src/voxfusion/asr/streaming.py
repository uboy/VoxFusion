"""Streaming ASR wrapper with chunked inference and overlap.

Accumulates audio chunks into overlapping windows, runs ASR on each
window, and deduplicates the resulting segments to produce a continuous
transcript without gaps or repeated text.
"""

import asyncio
from collections.abc import AsyncIterator

import numpy as np

from voxfusion.asr.base import ASREngine
from voxfusion.asr.dedup import OverlapDeduplicator
from voxfusion.config.models import ASRConfig
from voxfusion.logging import get_logger
from voxfusion.models.audio import AudioChunk
from voxfusion.models.transcription import TranscriptionSegment

log = get_logger(__name__)


class StreamingASR:
    """Wraps an ``ASREngine`` with overlapping chunk accumulation.

    Audio is accumulated into windows of ``chunk_duration_s`` with
    ``chunk_overlap_s`` overlap.  Each window is transcribed and the
    results are deduplicated to remove repeated text from overlap
    regions.
    """

    def __init__(
        self,
        engine: ASREngine,
        config: ASRConfig | None = None,
    ) -> None:
        self._engine = engine
        self._config = config or ASRConfig()
        self._dedup = OverlapDeduplicator(
            overlap_s=self._config.chunk_overlap_s,
        )
        self._buffer: list[np.ndarray] = []
        self._buffer_duration = 0.0
        self._time_offset = 0.0

    def _target_duration(self) -> float:
        return float(self._config.chunk_duration_s)

    def _overlap_duration(self) -> float:
        return float(self._config.chunk_overlap_s)

    async def process_chunk(self, chunk: AudioChunk) -> list[TranscriptionSegment]:
        """Add a chunk to the buffer; transcribe when enough audio is accumulated.

        Returns:
            Deduplicated segments, or empty list if still accumulating.
        """
        samples = chunk.samples
        if samples.ndim == 2:
            samples = samples.mean(axis=1)
        self._buffer.append(samples.astype(np.float32))
        self._buffer_duration += chunk.duration

        if self._buffer_duration < self._target_duration():
            return []

        # Concatenate buffer into a single array
        window = np.concatenate(self._buffer)
        window_chunk = AudioChunk(
            samples=window,
            sample_rate=chunk.sample_rate,
            channels=1,
            timestamp_start=self._time_offset,
            timestamp_end=self._time_offset + self._buffer_duration,
            source=chunk.source,
            dtype="float32",
        )

        segments = await self._engine.transcribe(window_chunk)

        # Deduplicate against previous window
        deduped = self._dedup.deduplicate(segments)

        # Keep overlap for next window
        overlap_samples = int(chunk.sample_rate * self._overlap_duration())
        if len(window) > overlap_samples:
            self._buffer = [window[-overlap_samples:]]
            self._time_offset += self._buffer_duration - self._overlap_duration()
            self._buffer_duration = self._overlap_duration()
        else:
            self._buffer = []
            self._time_offset += self._buffer_duration
            self._buffer_duration = 0.0

        return deduped

    async def flush(self) -> list[TranscriptionSegment]:
        """Transcribe any remaining audio in the buffer."""
        if not self._buffer or self._buffer_duration < 0.1:
            return []

        window = np.concatenate(self._buffer)
        dummy_chunk = AudioChunk(
            samples=window,
            sample_rate=16000,
            channels=1,
            timestamp_start=self._time_offset,
            timestamp_end=self._time_offset + self._buffer_duration,
            source="file",
            dtype="float32",
        )
        segments = await self._engine.transcribe(dummy_chunk)
        self._buffer.clear()
        self._buffer_duration = 0.0
        return self._dedup.deduplicate(segments)

    async def stream(
        self,
        audio_stream: AsyncIterator[AudioChunk],
    ) -> AsyncIterator[TranscriptionSegment]:
        """Streaming transcription: yields deduplicated segments."""
        async for chunk in audio_stream:
            segments = await self.process_chunk(chunk)
            for seg in segments:
                yield seg

        # Flush remaining
        for seg in await self.flush():
            yield seg

    def reset(self) -> None:
        """Clear internal state."""
        self._buffer.clear()
        self._buffer_duration = 0.0
        self._time_offset = 0.0
        self._dedup.reset()
