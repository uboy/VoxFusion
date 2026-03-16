"""Streaming pipeline with concurrent asyncio stages and backpressure.

Each stage runs as a concurrent asyncio task, reading from an input
queue and writing to an output queue.  Backpressure is managed via
bounded ``asyncio.Queue`` instances — a slow downstream stage will
cause upstream stages to block on ``put()``.
"""

import asyncio
from collections.abc import Callable
from typing import Any

from voxfusion.asr.base import ASREngine
from voxfusion.config.models import PipelineConfig
from voxfusion.diarization.base import DiarizationEngine
from voxfusion.exceptions import PipelineError
from voxfusion.logging import get_logger
from voxfusion.models.audio import AudioChunk
from voxfusion.models.diarization import DiarizedSegment
from voxfusion.models.transcription import TranscriptionSegment
from voxfusion.models.translation import TranslatedSegment
from voxfusion.pipeline.events import EventType, PipelineEvent, PipelineStage
from voxfusion.preprocessing.pipeline import PreProcessingPipeline
from voxfusion.translation.base import TranslationEngine

log = get_logger(__name__)

_SENTINEL = object()

EventCallback = Callable[[PipelineEvent], None]


class StreamingPipeline:
    """Processes a live audio stream through concurrent pipeline stages.

    Stages communicate via bounded ``asyncio.Queue`` objects::

        capture_q -> preprocess_q -> asr_q -> diarize_q -> output_q
    """

    def __init__(
        self,
        asr_engine: ASREngine,
        diarizer: DiarizationEngine,
        preprocessor: PreProcessingPipeline,
        config: PipelineConfig,
        translator: TranslationEngine | None = None,
        on_event: EventCallback | None = None,
        on_drop: Callable[[AudioChunk], None] | None = None,
        queue_size: int = 2,
    ) -> None:
        self._asr = asr_engine
        self._diarizer = diarizer
        self._preprocessor = preprocessor
        self._translator = translator
        self._config = config
        self._on_event = on_event or (lambda _: None)
        self._on_drop = on_drop or (lambda _: None)
        self._queue_size = queue_size
        self._running = False
        self._tasks: list[asyncio.Task[None]] = []
        self._chunks_dropped: int = 0
        self._chunks_in_asr: int = 0
        self._preprocess_q: asyncio.Queue | None = None
        self._asr_q: asyncio.Queue | None = None

    def _emit(self, event: PipelineEvent) -> None:
        self._on_event(event)

    def get_stats(self) -> dict[str, int]:
        """Return current pipeline stats: queue depths, in-flight ASR, drop count."""
        return {
            "preprocess_q": self._preprocess_q.qsize() if self._preprocess_q else 0,
            "asr_q": self._asr_q.qsize() if self._asr_q else 0,
            "in_asr": self._chunks_in_asr,
            "dropped": self._chunks_dropped,
        }

    async def _preprocess_stage(
        self,
        input_q: asyncio.Queue[AudioChunk | object],
        output_q: asyncio.Queue[AudioChunk | object],
    ) -> None:
        """Read raw chunks, preprocess, and forward."""
        self._emit(PipelineEvent(
            event_type=EventType.STAGE_STARTED,
            stage=PipelineStage.PREPROCESSING,
        ))
        while True:
            item = await input_q.get()
            if item is _SENTINEL:
                await output_q.put(_SENTINEL)
                break
            chunk: AudioChunk = item  # type: ignore[assignment]
            processed = self._preprocessor.process(chunk)
            await output_q.put(processed)

    async def _asr_stage(
        self,
        input_q: asyncio.Queue[AudioChunk | object],
        output_q: asyncio.Queue[tuple[list[TranscriptionSegment], AudioChunk] | object],
    ) -> None:
        """Run ASR on preprocessed chunks."""
        self._emit(PipelineEvent(
            event_type=EventType.STAGE_STARTED,
            stage=PipelineStage.ASR,
        ))
        while True:
            item = await input_q.get()
            if item is _SENTINEL:
                await output_q.put(_SENTINEL)
                break
            chunk: AudioChunk = item  # type: ignore[assignment]
            self._chunks_in_asr += 1
            try:
                segments = await self._asr.transcribe(chunk)
            finally:
                self._chunks_in_asr -= 1
            if segments:
                # Shift segment times to be relative to capture start
                adjusted = [
                    TranscriptionSegment(
                        text=s.text,
                        language=s.language,
                        start_time=s.start_time + chunk.timestamp_start,
                        end_time=s.end_time + chunk.timestamp_start,
                        confidence=s.confidence,
                        words=s.words,
                        no_speech_prob=s.no_speech_prob,
                    )
                    for s in segments
                ]
                await output_q.put((adjusted, chunk))

    async def _diarize_stage(
        self,
        input_q: asyncio.Queue[tuple[list[TranscriptionSegment], AudioChunk] | object],
        output_q: asyncio.Queue[list[DiarizedSegment] | object],
    ) -> None:
        """Assign speakers to transcription segments."""
        self._emit(PipelineEvent(
            event_type=EventType.STAGE_STARTED,
            stage=PipelineStage.DIARIZATION,
        ))
        while True:
            item = await input_q.get()
            if item is _SENTINEL:
                await output_q.put(_SENTINEL)
                break
            segments, chunk = item  # type: ignore[misc]
            diarized = await self._diarizer.diarize(segments, chunk)
            if diarized:
                await output_q.put(diarized)

    async def _translate_stage(
        self,
        input_q: asyncio.Queue[list[DiarizedSegment] | object],
        output_q: asyncio.Queue[list[TranslatedSegment] | object],
    ) -> None:
        """Translate diarized segments if translation is enabled."""
        if not self._translator:
            # No translation - just wrap segments
            while True:
                item = await input_q.get()
                if item is _SENTINEL:
                    await output_q.put(_SENTINEL)
                    break
                diarized_list: list[DiarizedSegment] = item  # type: ignore[assignment]
                translated = [
                    TranslatedSegment(diarized=d, translated_text=None, target_language=None)
                    for d in diarized_list
                ]
                await output_q.put(translated)
            return

        self._emit(PipelineEvent(
            event_type=EventType.STAGE_STARTED,
            stage=PipelineStage.TRANSLATION,
        ))
        while True:
            item = await input_q.get()
            if item is _SENTINEL:
                await output_q.put(_SENTINEL)
                break
            diarized_list: list[DiarizedSegment] = item  # type: ignore[assignment]
            translated = []
            for d in diarized_list:
                translated_text = await self._translator.translate(
                    d.segment.text,
                    source_language=d.segment.language,
                    target_language=self._config.translation.target_language,
                )
                translated.append(
                    TranslatedSegment(
                        diarized=d,
                        translated_text=translated_text,
                        target_language=self._config.translation.target_language,
                    )
                )
            if translated:
                await output_q.put(translated)

    async def _output_stage(
        self,
        input_q: asyncio.Queue[list[TranslatedSegment] | object],
        on_segments: Callable[[list[TranslatedSegment]], Any] | None,
    ) -> None:
        """Deliver translated segments to the callback as they arrive."""
        while True:
            item = await input_q.get()
            if item is _SENTINEL:
                break
            translated_list: list[TranslatedSegment] = item  # type: ignore[assignment]
            if on_segments:
                on_segments(translated_list)

    async def run(
        self,
        audio_source: Any,
        on_segments: Callable[[list[TranslatedSegment]], Any] | None = None,
    ) -> None:
        """Run the streaming pipeline until the source is exhausted or stopped.

        Args:
            audio_source: An ``AudioCaptureSource`` (must be started).
            on_segments: Callback invoked with each batch of translated segments.
        """
        self._running = True
        self._chunks_dropped = 0
        self._emit(PipelineEvent(event_type=EventType.PIPELINE_STARTED))

        self._preprocess_q = asyncio.Queue(self._queue_size)
        preprocess_q = self._preprocess_q
        self._asr_q = asyncio.Queue(self._queue_size)
        asr_q = self._asr_q
        diarize_q: asyncio.Queue[tuple | object] = asyncio.Queue(self._queue_size)
        translate_q: asyncio.Queue[list[DiarizedSegment] | object] = asyncio.Queue(self._queue_size)
        output_q: asyncio.Queue[list[TranslatedSegment] | object] = asyncio.Queue(self._queue_size)

        self._tasks = [
            asyncio.create_task(self._preprocess_stage(preprocess_q, asr_q)),
            asyncio.create_task(self._asr_stage(asr_q, diarize_q)),  # asr_q = self._asr_q
            asyncio.create_task(self._diarize_stage(diarize_q, translate_q)),
            asyncio.create_task(self._translate_stage(translate_q, output_q)),
            asyncio.create_task(self._output_stage(output_q, on_segments)),
        ]

        # Feed audio chunks into the pipeline concurrently with output delivery.
        # In lossy mode (live capture) drop incoming chunks when the pipeline is
        # backed up (slow ASR model) instead of blocking — this prevents the
        # upstream WASAPI capture buffer from overflowing.
        try:
            chunk_ms = self._config.capture.chunk_duration_ms
            async for chunk in audio_source.stream(chunk_duration_ms=chunk_ms):
                if not self._running:
                    break
                if self._config.capture.lossy_mode:
                    try:
                        preprocess_q.put_nowait(chunk)
                    except asyncio.QueueFull:
                        # Evict the oldest queued chunk so ASR always works on
                        # the freshest audio — prevents multi-minute backlog.
                        try:
                            evicted: AudioChunk = preprocess_q.get_nowait()  # type: ignore[assignment]
                            self._chunks_dropped += 1
                            log.warning(
                                "streaming.chunk_dropped",
                                source=evicted.source,
                                total_dropped=self._chunks_dropped,
                            )
                            self._on_drop(evicted)
                        except asyncio.QueueEmpty:
                            pass
                        try:
                            preprocess_q.put_nowait(chunk)
                        except asyncio.QueueFull:
                            self._chunks_dropped += 1
                            log.warning(
                                "streaming.chunk_dropped",
                                source=chunk.source,
                                total_dropped=self._chunks_dropped,
                            )
                            self._on_drop(chunk)
                else:
                    await preprocess_q.put(chunk)
        finally:
            await preprocess_q.put(_SENTINEL)

        await asyncio.gather(*self._tasks)
        self._emit(PipelineEvent(event_type=EventType.PIPELINE_COMPLETED, progress=1.0))
        log.info("streaming.completed")

    async def stop(self) -> None:
        """Signal the pipeline to stop after the current chunk."""
        self._running = False
        for task in self._tasks:
            task.cancel()
