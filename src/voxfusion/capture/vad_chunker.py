"""Energy-based VAD chunker that accumulates audio until a natural pause.

Wraps any AudioCaptureSource and yields variable-length chunks at
speech boundaries rather than fixed intervals.
"""

import asyncio
from collections.abc import AsyncIterator

import numpy as np

from voxfusion.logging import get_logger
from voxfusion.models.audio import AudioChunk

log = get_logger(__name__)


class VadChunker:
    """Wraps an AudioCaptureSource to yield pause-bounded audio chunks.

    Accumulates small frames and emits an AudioChunk when:
    - RMS has been below ``silence_threshold`` for at least ``min_silence_ms``,
      AND at least ``min_duration_ms`` of audio has been collected; OR
    - ``max_duration_ms`` of audio has been collected (hard cap).

    The ``chunk_duration_ms`` argument passed to ``stream()`` is ignored;
    chunk boundaries are determined by VAD alone.
    """

    def __init__(
        self,
        source: object,
        max_duration_ms: int = 5000,
        silence_threshold: float = 0.008,
        min_silence_ms: int = 400,
        min_duration_ms: int = 200,
        internal_chunk_ms: int = 50,
    ) -> None:
        self._source = source
        self._max_duration_ms = max_duration_ms
        self._silence_threshold = silence_threshold
        self._min_silence_ms = min_silence_ms
        self._min_duration_ms = min_duration_ms
        self._internal_chunk_ms = internal_chunk_ms

    @property
    def device_name(self) -> str:
        return f"vad:{self._source.device_name}"  # type: ignore[union-attr]

    @property
    def sample_rate(self) -> int:
        return self._source.sample_rate  # type: ignore[union-attr]

    @property
    def channels(self) -> int:
        return self._source.channels  # type: ignore[union-attr]

    @property
    def is_active(self) -> bool:
        return self._source.is_active  # type: ignore[union-attr]

    async def start(self) -> None:
        await self._source.start()  # type: ignore[union-attr]

    async def stop(self) -> None:
        await self._source.stop()  # type: ignore[union-attr]

    async def stream(self, chunk_duration_ms: int = 500) -> AsyncIterator[AudioChunk]:
        """Yield VAD-bounded chunks; ``chunk_duration_ms`` is ignored."""
        sample_rate: int = self._source.sample_rate  # type: ignore[union-attr]
        max_frames = int(sample_rate * self._max_duration_ms / 1000)
        min_silence_frames = int(sample_rate * self._min_silence_ms / 1000)
        min_duration_frames = int(sample_rate * self._min_duration_ms / 1000)

        pieces: list[np.ndarray] = []
        acc_frames: int = 0
        silence_frames: int = 0
        ts_start: float | None = None
        last_source: str = "unknown"
        last_sample_rate: int = sample_rate

        async for small_chunk in self._source.stream(chunk_duration_ms=self._internal_chunk_ms):  # type: ignore[union-attr]
            samples: np.ndarray = small_chunk.samples
            if samples.ndim > 1:
                samples = samples.mean(axis=-1).astype(np.float32)
            else:
                samples = samples.astype(np.float32)

            if ts_start is None:
                ts_start = small_chunk.timestamp_start
            last_source = small_chunk.source
            last_sample_rate = small_chunk.sample_rate

            pieces.append(samples)
            acc_frames += len(samples)

            rms = float(np.sqrt(np.mean(samples ** 2))) if samples.size > 0 else 0.0
            if rms < self._silence_threshold:
                silence_frames += len(samples)
            else:
                silence_frames = 0

            should_emit = (
                silence_frames >= min_silence_frames and acc_frames >= min_duration_frames
            ) or acc_frames >= max_frames

            if should_emit:
                combined = np.concatenate(pieces)
                ts_end = (ts_start or 0.0) + acc_frames / last_sample_rate
                log.debug(
                    "vad_chunker.emit",
                    source=last_source,
                    duration_ms=round(acc_frames * 1000 / last_sample_rate),
                    silence_triggered=silence_frames >= min_silence_frames,
                )
                yield AudioChunk(
                    samples=combined,
                    sample_rate=last_sample_rate,
                    channels=1,
                    timestamp_start=ts_start or 0.0,
                    timestamp_end=ts_end,
                    source=last_source,
                    dtype="float32",
                )
                pieces = []
                acc_frames = 0
                silence_frames = 0
                ts_start = None

        # Emit any remaining audio above minimum duration
        if pieces and acc_frames >= min_duration_frames:
            combined = np.concatenate(pieces)
            ts_end = (ts_start or 0.0) + acc_frames / last_sample_rate
            yield AudioChunk(
                samples=combined,
                sample_rate=last_sample_rate,
                channels=1,
                timestamp_start=ts_start or 0.0,
                timestamp_end=ts_end,
                source=last_source,
                dtype="float32",
            )
