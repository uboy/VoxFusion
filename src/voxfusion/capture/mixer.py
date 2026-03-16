"""Multi-source audio mixer for combining capture streams.

Merges audio from multiple capture sources (e.g. microphone + system
loopback) into a single stream. Each source retains its ``source``
label so downstream diarization can distinguish speakers.
"""

import asyncio
from collections.abc import AsyncIterator

import numpy as np

from voxfusion.capture.base import AudioCaptureSource
from voxfusion.logging import get_logger
from voxfusion.models.audio import AudioChunk

log = get_logger(__name__)


class AudioMixer:
    """Combines audio from multiple capture sources.

    Each source is consumed in a separate task.  Chunks are forwarded
    to a shared output queue with their original ``source`` label
    preserved for channel-based diarization.
    """

    def __init__(
        self,
        sources: list[AudioCaptureSource],
        queue_size: int = 20,
    ) -> None:
        self._sources = sources
        self._queue: asyncio.Queue[AudioChunk | None] = asyncio.Queue(maxsize=queue_size)
        self._tasks: list[asyncio.Task[None]] = []
        self._active = False

    @property
    def device_name(self) -> str:
        names = [s.device_name for s in self._sources]
        return f"mixer:[{', '.join(names)}]"

    @property
    def sample_rate(self) -> int:
        return self._sources[0].sample_rate if self._sources else 16000

    @property
    def channels(self) -> int:
        return 1

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def active_source_count(self) -> int:
        """Return the number of successfully started sources."""
        return len(self._sources)

    async def _consume_source(
        self,
        source: AudioCaptureSource,
        chunk_duration_ms: int,
    ) -> None:
        """Read chunks from one source and push them to the shared queue."""
        try:
            async for chunk in source.stream(chunk_duration_ms=chunk_duration_ms):
                await self._queue.put(chunk)
        except Exception as exc:
            log.warning("mixer.source_error", source=source.device_name, error=str(exc))
        finally:
            await self._queue.put(None)

    async def start(self) -> None:
        """Start all capture sources, skipping any that fail."""
        active: list[AudioCaptureSource] = []
        for source in self._sources:
            try:
                await source.start()
                active.append(source)
            except Exception as exc:
                log.warning(
                    "mixer.source_start_failed",
                    source=source.device_name,
                    error=str(exc),
                )
        if not active:
            from voxfusion.exceptions import AudioCaptureError
            raise AudioCaptureError("All audio sources failed to start.")
        self._sources = active
        self._active = True
        log.info("mixer.started", sources=len(self._sources), active=len(active))

    async def stop(self) -> None:
        """Stop all capture sources and cancel tasks."""
        self._active = False
        for task in self._tasks:
            task.cancel()
        for source in self._sources:
            await source.stop()
        log.info("mixer.stopped")

    async def stream(self, chunk_duration_ms: int = 500) -> AsyncIterator[AudioChunk]:
        """Yield interleaved chunks from all sources."""
        # Launch one consumer task per source
        self._tasks = [
            asyncio.create_task(self._consume_source(src, chunk_duration_ms))
            for src in self._sources
        ]

        finished_count = 0
        total_sources = len(self._sources)

        while finished_count < total_sources:
            chunk = await self._queue.get()
            if chunk is None:
                finished_count += 1
                continue
            yield chunk
