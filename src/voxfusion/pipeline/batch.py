"""Batch pipeline for processing complete audio files.

Reads the entire audio file into chunks, runs preprocessing, ASR,
diarization, and (optionally) translation in sequence, then returns
a complete ``TranscriptionResult``.
"""

import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from voxfusion.asr.base import ASREngine
from voxfusion.capture.file_source import FileAudioSource
from voxfusion.config.models import PipelineConfig
from voxfusion.diarization.base import DiarizationEngine
from voxfusion.exceptions import AudioCaptureError, PipelineError
from voxfusion.logging import get_logger
from voxfusion.media.extractor import extract_audio_async, needs_extraction
from voxfusion.models.audio import AudioChunk
from voxfusion.models.result import TranscriptionResult
from voxfusion.models.translation import TranslatedSegment
from voxfusion.pipeline.events import EventType, PipelineEvent, PipelineStage
from voxfusion.preprocessing.pipeline import PreProcessingPipeline

log = get_logger(__name__)

EventCallback = Callable[[PipelineEvent], None]


class BatchPipeline:
    """Processes a complete audio file through the full pipeline.

    Stages: capture -> preprocess -> ASR -> diarization -> wrap result.
    """

    def __init__(
        self,
        asr_engine: ASREngine,
        diarizer: DiarizationEngine,
        preprocessor: PreProcessingPipeline,
        config: PipelineConfig,
        on_event: EventCallback | None = None,
    ) -> None:
        self._asr = asr_engine
        self._diarizer = diarizer
        self._preprocessor = preprocessor
        self._config = config
        self._on_event = on_event or (lambda _: None)

    def _emit(self, event: PipelineEvent) -> None:
        """Emit a pipeline event."""
        self._on_event(event)

    async def process_file(self, file_path: Path) -> TranscriptionResult:
        """Run the full batch pipeline on an audio file.

        Args:
            file_path: Path to the audio file to transcribe.

        Returns:
            A complete ``TranscriptionResult``.

        Raises:
            PipelineError: If any stage fails fatally.
        """
        t_start = time.monotonic()
        self._emit(PipelineEvent(
            event_type=EventType.PIPELINE_STARTED,
            message=f"Processing {file_path.name}",
        ))

        # -- Stage 1: Capture (extract audio if needed, then read) --
        tmp_audio: Path | None = None
        source_path = file_path

        if needs_extraction(file_path):
            self._emit(PipelineEvent(
                event_type=EventType.STAGE_STARTED,
                stage=PipelineStage.CAPTURE,
                message=f"Extracting audio from {file_path.suffix.lstrip('.').upper()} file...",
            ))
            try:
                tmp_audio = await extract_audio_async(file_path)
                source_path = tmp_audio
            except AudioCaptureError as exc:
                raise PipelineError(str(exc)) from exc
        else:
            self._emit(PipelineEvent(
                event_type=EventType.STAGE_STARTED,
                stage=PipelineStage.CAPTURE,
                message="Reading audio file",
            ))

        try:
            source = FileAudioSource(source_path)
            try:
                await source.start()
            except Exception as exc:
                raise PipelineError(f"Failed to open audio file: {exc}") from exc

            chunks: list[AudioChunk] = []
            try:
                async for chunk in source.stream(
                    chunk_duration_ms=self._config.capture.chunk_duration_ms,
                ):
                    chunks.append(chunk)
            finally:
                await source.stop()
        finally:
            if tmp_audio is not None:
                tmp_audio.unlink(missing_ok=True)

        if not chunks:
            raise PipelineError(f"No audio data read from {file_path}")

        self._emit(PipelineEvent(
            event_type=EventType.STAGE_COMPLETED,
            stage=PipelineStage.CAPTURE,
            message=f"Read {len(chunks)} chunks",
            data={"chunks": len(chunks)},
        ))

        # -- Stage 2: Preprocessing --
        self._emit(PipelineEvent(
            event_type=EventType.STAGE_STARTED,
            stage=PipelineStage.PREPROCESSING,
            message="Preprocessing audio",
        ))

        processed_chunks = [self._preprocessor.process(c) for c in chunks]

        # Concatenate all chunks into one AudioChunk for ASR
        all_samples = np.concatenate([c.samples for c in processed_chunks])
        # Guarantee mono 1D float32 — stereo files produce (N, 2) arrays which
        # confuse every ASR engine and cause "object too deep" errors downstream.
        if all_samples.ndim == 2:
            all_samples = all_samples.mean(axis=1).astype(np.float32)
        elif all_samples.ndim > 2:
            all_samples = all_samples.reshape(all_samples.shape[0], -1).mean(axis=1).astype(np.float32)
        all_samples = np.ascontiguousarray(all_samples, dtype=np.float32)
        sr = processed_chunks[0].sample_rate
        full_audio = AudioChunk(
            samples=all_samples,
            sample_rate=sr,
            channels=1,
            timestamp_start=0.0,
            timestamp_end=len(all_samples) / sr,
            source="file",
            dtype="float32",
        )

        self._emit(PipelineEvent(
            event_type=EventType.STAGE_COMPLETED,
            stage=PipelineStage.PREPROCESSING,
            message="Preprocessing complete",
            data={"duration_s": round(full_audio.duration, 2)},
        ))

        # -- Stage 3: ASR --
        self._emit(PipelineEvent(
            event_type=EventType.STAGE_STARTED,
            stage=PipelineStage.ASR,
            message="Transcribing audio",
        ))

        segments = await self._asr.transcribe(
            full_audio,
            language=self._config.asr.language,
            word_timestamps=self._config.asr.word_timestamps,
        )

        self._emit(PipelineEvent(
            event_type=EventType.STAGE_COMPLETED,
            stage=PipelineStage.ASR,
            message=f"Transcribed {len(segments)} segments",
            data={"segments": len(segments)},
        ))

        # -- Stage 4: Diarization --
        self._emit(PipelineEvent(
            event_type=EventType.STAGE_STARTED,
            stage=PipelineStage.DIARIZATION,
            message="Diarizing segments",
        ))

        diarized = await self._diarizer.diarize(segments, full_audio)

        self._emit(PipelineEvent(
            event_type=EventType.STAGE_COMPLETED,
            stage=PipelineStage.DIARIZATION,
            message=f"Diarized {len(diarized)} segments",
        ))

        # -- Wrap into TranslatedSegment (no translation for MVP) --
        translated = [
            TranslatedSegment(
                diarized=d,
                translated_text=None,
                target_language=None,
            )
            for d in diarized
        ]

        elapsed = time.monotonic() - t_start
        result = TranscriptionResult(
            segments=translated,
            source_info={
                "file": str(file_path),
                "sample_rate": sr,
                "duration_s": round(full_audio.duration, 2),
                "chunks": len(chunks),
            },
            processing_info={
                "asr_model": self._asr.model_name,
                "processing_time_s": round(elapsed, 3),
                "segments": len(segments),
            },
            created_at=datetime.now(timezone.utc).isoformat(),
        )

        self._emit(PipelineEvent(
            event_type=EventType.PIPELINE_COMPLETED,
            message=f"Done in {elapsed:.1f}s — {len(segments)} segments",
            progress=1.0,
            data={"processing_time_s": round(elapsed, 3)},
        ))

        log.info(
            "batch.completed",
            file=str(file_path),
            segments=len(segments),
            elapsed_s=round(elapsed, 3),
        )
        return result
