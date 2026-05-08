"""Batch pipeline for processing complete audio files.

Reads the entire audio file into chunks, runs preprocessing, ASR,
diarization, and (optionally) translation in sequence, then returns
a complete ``TranscriptionResult``.
"""

import asyncio
import itertools
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from voxfusion.asr.base import ASREngine
from voxfusion.capture.file_source import FileAudioSource
from voxfusion.config.models import PipelineConfig
from voxfusion.diarization.alignment import SpeakerTurn
from voxfusion.diarization.base import DiarizationEngine
from voxfusion.diarization.types import DiarizationTurnResult
from voxfusion.exceptions import AudioCaptureError, PipelineError
from voxfusion.logging import get_logger
from voxfusion.media.extractor import extract_audio_async, needs_extraction
from voxfusion.models.audio import AudioChunk
from voxfusion.models.diarization import DiarizedSegment
from voxfusion.models.result import TranscriptionResult
from voxfusion.models.transcription import TranscriptionSegment, WordTiming
from voxfusion.models.translation import TranslatedSegment
from voxfusion.pipeline.events import EventType, PipelineEvent, PipelineStage
from voxfusion.preprocessing.pipeline import PreProcessingPipeline

log = get_logger(__name__)

EventCallback = Callable[[PipelineEvent], None]
_MAX_GIGAAM_WINDOW_S = 24.0
_SAME_SPEAKER_MERGE_GAP_S = 0.2
_DIARIZATION_HEARTBEAT_S = 10.0
_WINDOW_PROGRESS_LOG_INTERVAL = 10
_ASR_PROGRESS_DIARIZATION_START = 0.46
_ASR_PROGRESS_DIARIZATION_CAP = 0.55
_ASR_PROGRESS_WINDOW_START = 0.56
_ASR_PROGRESS_WINDOW_END = 0.78
_DIARIZATION_ETA_FLOOR_S = 20.0
_DIARIZATION_ETA_REALTIME_FACTOR = 0.08
_MIN_GIGAAM_WINDOW_SAMPLES = 320


def _slice_audio_chunk(audio: AudioChunk, start_time: float, end_time: float) -> AudioChunk:
    """Slice *audio* in absolute seconds and keep absolute timeline metadata."""
    sample_rate = audio.sample_rate
    start_idx = max(0, int(round(start_time * sample_rate)))
    end_idx = min(audio.num_samples, int(round(end_time * sample_rate)))
    samples = audio.samples[start_idx:end_idx]
    return AudioChunk(
        samples=np.ascontiguousarray(samples, dtype=np.float32),
        sample_rate=sample_rate,
        channels=audio.channels,
        timestamp_start=start_time,
        timestamp_end=end_time,
        source=audio.source,
        dtype="float32",
    )


def _rebase_segment(segment: TranscriptionSegment, offset_s: float) -> TranscriptionSegment:
    words = None
    if segment.words:
        words = [
            WordTiming(
                word=word.word,
                start_time=word.start_time + offset_s,
                end_time=word.end_time + offset_s,
                probability=word.probability,
            )
            for word in segment.words
        ]
    return TranscriptionSegment(
        text=segment.text,
        language=segment.language,
        start_time=segment.start_time + offset_s,
        end_time=segment.end_time + offset_s,
        confidence=segment.confidence,
        words=words,
        no_speech_prob=segment.no_speech_prob,
    )


def _format_eta(seconds: float | None) -> str:
    if seconds is None:
        return "ETA ~unknown"
    remaining = max(0, int(round(seconds)))
    minutes, secs = divmod(remaining, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"ETA ~{hours:d}:{minutes:02d}:{secs:02d}"
    return f"ETA ~{minutes:02d}:{secs:02d}"


def _estimate_initial_diarization_total(audio_duration_s: float) -> float:
    return max(_DIARIZATION_ETA_FLOOR_S, audio_duration_s * _DIARIZATION_ETA_REALTIME_FACTOR)


def _remaining_diarization_eta(elapsed_s: float, estimated_total_s: float) -> float | None:
    if elapsed_s >= estimated_total_s:
        return None
    return max(0.0, estimated_total_s - elapsed_s)


def _select_alignment_turns(
    turn_result: DiarizationTurnResult | list[SpeakerTurn],
) -> tuple[list[SpeakerTurn], dict[str, object]]:
    if isinstance(turn_result, DiarizationTurnResult):
        selected = list(turn_result.alignment_turns())
        return selected, {
            "turns": len(turn_result.turns),
            "exclusive_turns": (
                len(turn_result.exclusive_turns)
                if turn_result.exclusive_turns is not None
                else None
            ),
            "alignment_turns": len(selected),
            "used_exclusive": bool(turn_result.exclusive_turns),
            "speaker_count_estimate": turn_result.speaker_count_estimate,
            "speaker_count_hint_applied": turn_result.speaker_count_hint_applied,
            "model_id": turn_result.model_id,
        }

    turns = list(turn_result)
    return turns, {"turns": len(turns)}


def _should_fallback_to_asr_first(
    *,
    turn_log: dict[str, object],
    normalized_turns: list[SpeakerTurn],
    audio_duration_s: float,
) -> bool:
    """Decide when diarization-first should fall back to ASR-first.

    In auto speaker-count mode pyannote may emit one full-audio speaker turn.
    Running ASR per turn in that case can collapse transcript granularity into
    one long segment. Prefer ASR-first so native ASR chunking/timestamps are preserved.
    """
    hint = str(turn_log.get("speaker_count_hint_applied") or "").strip().lower()
    if bool(turn_log.get("used_exclusive")):
        return False
    if hint != "auto" or len(normalized_turns) != 1:
        return False

    only_turn = normalized_turns[0]
    covered = max(0.0, only_turn.end_time - only_turn.start_time)
    if audio_duration_s <= 0:
        return False
    coverage_ratio = covered / audio_duration_s
    return coverage_ratio >= 0.95


def _normalize_turns(turns: list[SpeakerTurn]) -> list[SpeakerTurn]:
    """Flatten overlaps, merge adjacent same-speaker turns, then split long windows."""
    ordered = sorted(
        (turn for turn in turns if turn.end_time > turn.start_time),
        key=lambda turn: (turn.start_time, turn.end_time),
    )
    if not ordered:
        return []

    boundaries = sorted({point for turn in ordered for point in (turn.start_time, turn.end_time)})
    flattened: list[SpeakerTurn] = []
    for start, end in itertools.pairwise(boundaries):
        if end <= start:
            continue
        midpoint = start + ((end - start) / 2.0)
        active = [
            turn
            for turn in ordered
            if turn.start_time <= midpoint < turn.end_time
            or (midpoint == turn.end_time and end == turn.end_time)
        ]
        if not active:
            continue
        winner = max(active, key=lambda turn: (turn.start_time, turn.end_time))
        if (
            flattened
            and flattened[-1].speaker_id == winner.speaker_id
            and abs(start - flattened[-1].end_time) <= _SAME_SPEAKER_MERGE_GAP_S
        ):
            flattened[-1] = SpeakerTurn(
                speaker_id=winner.speaker_id,
                start_time=flattened[-1].start_time,
                end_time=end,
            )
        else:
            flattened.append(
                SpeakerTurn(
                    speaker_id=winner.speaker_id,
                    start_time=start,
                    end_time=end,
                )
            )

    merged: list[SpeakerTurn] = []
    for turn in flattened:
        if merged:
            prev = merged[-1]
            if (
                prev.speaker_id == turn.speaker_id
                and turn.start_time - prev.end_time <= _SAME_SPEAKER_MERGE_GAP_S
            ):
                merged[-1] = SpeakerTurn(
                    speaker_id=prev.speaker_id,
                    start_time=prev.start_time,
                    end_time=max(prev.end_time, turn.end_time),
                )
                continue
        merged.append(turn)

    normalized: list[SpeakerTurn] = []
    for turn in merged:
        cursor = turn.start_time
        while cursor < turn.end_time:
            window_end = min(cursor + _MAX_GIGAAM_WINDOW_S, turn.end_time)
            normalized.append(
                SpeakerTurn(
                    speaker_id=turn.speaker_id,
                    start_time=cursor,
                    end_time=window_end,
                )
            )
            cursor = window_end
    return normalized


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
        requested_diarization_strategy: str | None = None,
        resolved_diarization_strategy: str | None = None,
        startup_warnings: tuple[str, ...] = (),
    ) -> None:
        self._asr = asr_engine
        self._diarizer = diarizer
        self._preprocessor = preprocessor
        self._config = config
        self._on_event = on_event or (lambda _: None)
        self._requested_diarization_strategy = (
            requested_diarization_strategy or config.diarization.strategy
        )
        self._resolved_diarization_strategy = (
            resolved_diarization_strategy or self._requested_diarization_strategy
        )
        self._startup_warnings = startup_warnings

    def _emit(self, event: PipelineEvent) -> None:
        """Emit a pipeline event."""
        self._on_event(event)

    def _warn(self, message: str, stage: PipelineStage | None = None) -> None:
        log.warning(
            "pipeline.warning",
            stage=str(stage) if stage is not None else None,
            message=message,
        )
        self._emit(
            PipelineEvent(
                event_type=EventType.WARNING,
                stage=stage,
                message=message,
            )
        )

    def _progress(
        self,
        *,
        stage: PipelineStage,
        message: str,
        progress: float,
        **data: object,
    ) -> None:
        self._emit(
            PipelineEvent(
                event_type=EventType.PROGRESS,
                stage=stage,
                message=message,
                progress=progress,
                data={k: v for k, v in data.items() if v is not None},
            )
        )

    def _diarization_path_decision(self) -> tuple[bool, str]:
        if self._config.asr.engine != "gigaam":
            return False, "ASR backend is not gigaam."
        if self._resolved_diarization_strategy not in {"ml", "hybrid"}:
            return False, (
                f"Resolved diarization strategy is '{self._resolved_diarization_strategy}', "
                "so ASR-first processing will be used."
            )
        if not callable(getattr(self._diarizer, "diarize_turns", None)):
            return False, "Resolved diarizer does not provide speaker-turn diarization."
        return True, "GigaAM with ML-capable diarization selected."

    async def _transcribe_diarized_windows(
        self,
        full_audio: AudioChunk,
    ) -> list[DiarizedSegment] | None:
        diarize_turns_result = getattr(self._diarizer, "diarize_turns_result", None)
        diarize_turns = getattr(self._diarizer, "diarize_turns", None)
        diarize_turn_callable = None
        if callable(diarize_turns_result):
            diarize_turn_callable = diarize_turns_result
        elif callable(diarize_turns):
            diarize_turn_callable = diarize_turns
        if diarize_turn_callable is None:
            return None

        log.info(
            "batch.diarization_turns_started",
            audio_duration_s=round(full_audio.duration, 2),
            sample_rate=full_audio.sample_rate,
        )
        self._progress(
            stage=PipelineStage.ASR,
            message=(
                f"Running speaker diarization on {full_audio.duration / 60.0:.1f} min audio "
                f"({_format_eta(_estimate_initial_diarization_total(full_audio.duration))})..."
            ),
            progress=_ASR_PROGRESS_DIARIZATION_START,
            phase="speaker_turn_diarization",
            audio_duration_s=round(full_audio.duration, 2),
        )
        turns_task = asyncio.create_task(diarize_turn_callable(full_audio))
        started_at = time.monotonic()
        heartbeat_count = 0
        estimated_total = _estimate_initial_diarization_total(full_audio.duration)
        while not turns_task.done():
            await asyncio.sleep(min(_DIARIZATION_HEARTBEAT_S, 0.5))
            if turns_task.done():
                break
            elapsed = time.monotonic() - started_at
            if elapsed < (heartbeat_count + 1) * _DIARIZATION_HEARTBEAT_S:
                continue
            heartbeat_count += 1
            progress = min(
                _ASR_PROGRESS_DIARIZATION_CAP,
                _ASR_PROGRESS_DIARIZATION_START + (heartbeat_count * 0.01),
            )
            eta_remaining = _remaining_diarization_eta(elapsed, estimated_total)
            log.info(
                "batch.diarization_turns_in_progress",
                elapsed_s=round(elapsed, 1),
                audio_duration_s=round(full_audio.duration, 1),
                eta_s=round(eta_remaining, 1) if eta_remaining is not None else None,
            )
            self._progress(
                stage=PipelineStage.ASR,
                message=(
                    f"Running speaker diarization... {elapsed:.0f}s elapsed, "
                    f"{_format_eta(eta_remaining)}"
                ),
                progress=progress,
                phase="speaker_turn_diarization",
                elapsed_s=round(elapsed, 1),
                audio_duration_s=round(full_audio.duration, 2),
                eta_s=round(eta_remaining, 1) if eta_remaining is not None else None,
            )

        turn_result = await turns_task
        turns, turn_log = _select_alignment_turns(turn_result)
        log.info("batch.diarization_turns_received", **turn_log)
        if not turns:
            self._warn(
                "ML diarization produced no speaker turns. Falling back to the standard batch path.",
                stage=PipelineStage.DIARIZATION,
            )
            return None

        normalized_turns = _normalize_turns(turns)
        log.info("batch.diarization_turns_normalized", turns=len(normalized_turns))
        if _should_fallback_to_asr_first(
            turn_log=turn_log,
            normalized_turns=normalized_turns,
            audio_duration_s=full_audio.duration,
        ):
            log.warning(
                "batch.diarization_first_fallback_to_asr_first",
                reason="single_auto_turn_covering_full_audio",
                audio_duration_s=round(full_audio.duration, 2),
            )
            self._warn(
                "Auto speaker diarization returned one full-length turn; "
                "switching to ASR-first path to preserve timestamp granularity.",
                stage=PipelineStage.DIARIZATION,
            )
            return None
        total_windows = len(normalized_turns)

        max_concurrent = max(1, self._config.asr.parallel_windows)
        log.info(
            "batch.window_transcription_start",
            total_windows=total_windows,
            max_concurrent=max_concurrent,
        )
        if total_windows:
            self._progress(
                stage=PipelineStage.ASR,
                message=(f"Transcribing speaker windows 0/{total_windows} ({_format_eta(None)})"),
                progress=_ASR_PROGRESS_WINDOW_START,
                phase="speaker_window_transcription",
                completed_windows=0,
                total_windows=total_windows,
            )

        semaphore = asyncio.Semaphore(max_concurrent)
        completed_count = 0
        window_phase_started_at = time.monotonic()

        async def _process_one(
            turn_index: int,
            turn: SpeakerTurn,
        ) -> list[DiarizedSegment]:
            window = _slice_audio_chunk(full_audio, turn.start_time, turn.end_time)
            if window.num_samples < _MIN_GIGAAM_WINDOW_SAMPLES:
                log.warning(
                    "batch.diarized_window_skipped",
                    window=turn_index,
                    total_windows=total_windows,
                    speaker_id=turn.speaker_id,
                    start_s=round(turn.start_time, 2),
                    end_s=round(turn.end_time, 2),
                    samples=window.num_samples,
                    reason="too_short",
                )
                return []
            async with semaphore:
                segs = await self._asr.transcribe(
                    window,
                    language=self._config.asr.language,
                    word_timestamps=self._config.asr.word_timestamps,
                )
            return [
                DiarizedSegment(
                    segment=_rebase_segment(s, window.timestamp_start),
                    speaker_id=turn.speaker_id,
                    speaker_source="ml",
                )
                for s in segs
            ]

        async def _tracked(turn_index: int, turn: "SpeakerTurn") -> list[DiarizedSegment]:
            nonlocal completed_count
            result = await _process_one(turn_index, turn)
            completed_count += 1
            elapsed_window_phase = max(0.001, time.monotonic() - window_phase_started_at)
            estimated_remaining: float | None = None
            if completed_count < total_windows:
                avg_window_s = elapsed_window_phase / completed_count
                estimated_remaining = avg_window_s * (total_windows - completed_count)
            window_progress = _ASR_PROGRESS_WINDOW_START + (
                (_ASR_PROGRESS_WINDOW_END - _ASR_PROGRESS_WINDOW_START)
                * completed_count
                / max(1, total_windows)
            )
            if (
                completed_count in (1, total_windows)
                or completed_count % _WINDOW_PROGRESS_LOG_INTERVAL == 0
            ):
                log.info(
                    "batch.diarized_window_progress",
                    window=completed_count,
                    total_windows=total_windows,
                    speaker_id=turn.speaker_id,
                    start_s=round(turn.start_time, 2),
                    end_s=round(turn.end_time, 2),
                    eta_s=(
                        round(estimated_remaining, 1) if estimated_remaining is not None else None
                    ),
                )
            self._progress(
                stage=PipelineStage.ASR,
                message=(
                    f"Transcribing speaker windows {completed_count}/{total_windows} "
                    f"({turn.speaker_id} {turn.start_time:.0f}-{turn.end_time:.0f}s, "
                    f"{_format_eta(estimated_remaining)})"
                ),
                progress=min(_ASR_PROGRESS_WINDOW_END, window_progress),
                phase="speaker_window_transcription",
                completed_windows=completed_count,
                total_windows=total_windows,
                speaker_id=turn.speaker_id,
                window_start_s=round(turn.start_time, 2),
                window_end_s=round(turn.end_time, 2),
                eta_s=(round(estimated_remaining, 1) if estimated_remaining is not None else None),
            )
            return result

        nested = await asyncio.gather(
            *[_tracked(i + 1, turn) for i, turn in enumerate(normalized_turns)]
        )
        diarized: list[DiarizedSegment] = [seg for batch in nested for seg in batch]
        diarized.sort(key=lambda item: (item.segment.start_time, item.segment.end_time))
        return diarized

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
        self._emit(
            PipelineEvent(
                event_type=EventType.PIPELINE_STARTED,
                message=f"Processing {file_path.name}",
            )
        )
        for warning in self._startup_warnings:
            self._warn(warning, stage=PipelineStage.DIARIZATION)

        # -- Stage 1: Capture (extract audio if needed, then read) --
        tmp_audio: Path | None = None
        source_path = file_path

        if needs_extraction(file_path):
            self._emit(
                PipelineEvent(
                    event_type=EventType.STAGE_STARTED,
                    stage=PipelineStage.CAPTURE,
                    message=f"Extracting audio from {file_path.suffix.lstrip('.').upper()} file...",
                )
            )
            try:
                tmp_audio = await extract_audio_async(file_path)
                source_path = tmp_audio
            except AudioCaptureError as exc:
                raise PipelineError(str(exc)) from exc
        else:
            self._emit(
                PipelineEvent(
                    event_type=EventType.STAGE_STARTED,
                    stage=PipelineStage.CAPTURE,
                    message="Reading audio file",
                )
            )

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

        self._emit(
            PipelineEvent(
                event_type=EventType.STAGE_COMPLETED,
                stage=PipelineStage.CAPTURE,
                message=f"Read {len(chunks)} chunks",
                data={"chunks": len(chunks)},
            )
        )

        # -- Stage 2: Preprocessing --
        self._emit(
            PipelineEvent(
                event_type=EventType.STAGE_STARTED,
                stage=PipelineStage.PREPROCESSING,
                message="Preprocessing audio",
            )
        )

        processed_chunks = [self._preprocessor.process(c) for c in chunks]

        raw_samples = np.concatenate([c.samples for c in chunks])
        if raw_samples.ndim == 2:
            raw_samples = raw_samples.mean(axis=1).astype(np.float32)
        elif raw_samples.ndim > 2:
            raw_samples = (
                raw_samples.reshape(raw_samples.shape[0], -1).mean(axis=1).astype(np.float32)
            )
        raw_samples = np.ascontiguousarray(raw_samples, dtype=np.float32)
        raw_sample_rate = chunks[0].sample_rate
        raw_full_audio = AudioChunk(
            samples=raw_samples,
            sample_rate=raw_sample_rate,
            channels=1,
            timestamp_start=0.0,
            timestamp_end=len(raw_samples) / raw_sample_rate,
            source="file",
            dtype="float32",
        )

        # Concatenate all chunks into one AudioChunk for ASR
        all_samples = np.concatenate([c.samples for c in processed_chunks])
        # Guarantee mono 1D float32 — stereo files produce (N, 2) arrays which
        # confuse every ASR engine and cause "object too deep" errors downstream.
        if all_samples.ndim == 2:
            all_samples = all_samples.mean(axis=1).astype(np.float32)
        elif all_samples.ndim > 2:
            all_samples = (
                all_samples.reshape(all_samples.shape[0], -1).mean(axis=1).astype(np.float32)
            )
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

        self._emit(
            PipelineEvent(
                event_type=EventType.STAGE_COMPLETED,
                stage=PipelineStage.PREPROCESSING,
                message="Preprocessing complete",
                data={"duration_s": round(full_audio.duration, 2)},
            )
        )

        # -- Stage 3: ASR --
        self._emit(
            PipelineEvent(
                event_type=EventType.STAGE_STARTED,
                stage=PipelineStage.ASR,
                message="Transcribing audio",
            )
        )

        diarized: list[DiarizedSegment] | None = None
        diarization_first, path_reason = self._diarization_path_decision()
        log.info(
            "batch.diarization_path_selected",
            asr_engine=self._config.asr.engine,
            requested_strategy=self._requested_diarization_strategy,
            resolved_strategy=self._resolved_diarization_strategy,
            diarization_first=diarization_first,
            reason=path_reason,
        )
        if diarization_first:
            diarized = await self._transcribe_diarized_windows(raw_full_audio)

        if diarized is None:
            segments = await self._asr.transcribe(
                full_audio,
                language=self._config.asr.language,
                word_timestamps=self._config.asr.word_timestamps,
            )
        else:
            segments = [item.segment for item in diarized]

        self._emit(
            PipelineEvent(
                event_type=EventType.STAGE_COMPLETED,
                stage=PipelineStage.ASR,
                message=f"Transcribed {len(segments)} segments",
                data={"asr_segments": len(segments)},
            )
        )

        # -- Stage 4: Diarization --
        self._emit(
            PipelineEvent(
                event_type=EventType.STAGE_STARTED,
                stage=PipelineStage.DIARIZATION,
                message="Diarizing segments",
            )
        )

        diarization_message = ""
        if diarized is None:
            diarized = await self._diarizer.diarize(segments, raw_full_audio)
            diarization_message = f"Diarized {len(diarized)} segments"
        else:
            diarization_message = f"Diarized {len(diarized)} segments from speaker windows"

        self._emit(
            PipelineEvent(
                event_type=EventType.STAGE_COMPLETED,
                stage=PipelineStage.DIARIZATION,
                message=diarization_message,
                data={"diarized_segments": len(diarized)},
            )
        )

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
                "asr_segments": len(segments),
                "diarized_segments": len(diarized),
                "requested_diarization_strategy": self._requested_diarization_strategy,
                "resolved_diarization_strategy": self._resolved_diarization_strategy,
            },
            created_at=datetime.now(UTC).isoformat(),
        )

        self._emit(
            PipelineEvent(
                event_type=EventType.PIPELINE_COMPLETED,
                message=(
                    f"Done in {elapsed:.1f}s — {len(diarized)} diarized / "
                    f"{len(segments)} ASR segments"
                ),
                progress=1.0,
                data={
                    "processing_time_s": round(elapsed, 3),
                    "asr_segments": len(segments),
                    "diarized_segments": len(diarized),
                },
            )
        )

        log.info(
            "batch.completed",
            file=str(file_path),
            asr_segments=len(segments),
            diarized_segments=len(diarized),
            elapsed_s=round(elapsed, 3),
        )
        return result
