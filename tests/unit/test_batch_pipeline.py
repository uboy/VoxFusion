"""Focused tests for batch pipeline diarization behavior."""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from voxfusion.config.models import ASRConfig, DiarizationConfig, PipelineConfig
from voxfusion.diarization.alignment import SpeakerTurn
from voxfusion.diarization.types import DiarizationTurnResult
from voxfusion.models.audio import AudioChunk
from voxfusion.models.transcription import TranscriptionSegment
from voxfusion.pipeline.batch import BatchPipeline
from voxfusion.pipeline.events import EventType
from voxfusion.preprocessing.pipeline import PreProcessingPipeline

SR = 16_000


class _FakeGigaAMEngine:
    model_name = "gigaam/test"

    def __init__(self) -> None:
        self.transcribe_calls: list[tuple[float, float, int]] = []

    async def transcribe(
        self,
        audio: AudioChunk,
        *,
        language: str | None = None,
        initial_prompt: str | None = None,
        word_timestamps: bool = False,
    ) -> list[TranscriptionSegment]:
        del language, initial_prompt, word_timestamps
        self.transcribe_calls.append(
            (audio.timestamp_start, audio.timestamp_end, audio.num_samples)
        )
        return [
            TranscriptionSegment(
                text=f"{audio.timestamp_start:.1f}-{audio.timestamp_end:.1f}",
                language="ru",
                start_time=0.0,
                end_time=audio.duration,
                confidence=0.0,
                words=None,
                no_speech_prob=0.0,
            )
        ]

    async def transcribe_stream(self, audio_stream, *, language: str | None = None):
        del audio_stream, language
        if False:  # pragma: no cover
            yield

    def load_model(self) -> None:
        pass

    def unload_model(self) -> None:
        pass

    def close(self) -> None:
        pass


class _FakeTurnDiarizer:
    async def diarize(self, segments, audio=None):
        del audio
        return []

    async def diarize_turns(self, audio: AudioChunk) -> list[SpeakerTurn]:
        del audio
        return [
            SpeakerTurn("SPEAKER_A", 0.0, 1.0),
            SpeakerTurn("SPEAKER_B", 1.0, 2.0),
        ]


class _OverlappingTurnDiarizer:
    async def diarize(self, segments, audio=None):
        del segments, audio
        return []

    async def diarize_turns(self, audio: AudioChunk) -> list[SpeakerTurn]:
        del audio
        return [
            SpeakerTurn("SPEAKER_A", 0.0, 2.0),
            SpeakerTurn("SPEAKER_B", 1.0, 3.0),
        ]


class _ZeroProcessor:
    def process(self, chunk: AudioChunk) -> AudioChunk:
        return AudioChunk(
            samples=np.zeros_like(chunk.samples),
            sample_rate=chunk.sample_rate,
            channels=chunk.channels,
            timestamp_start=chunk.timestamp_start,
            timestamp_end=chunk.timestamp_end,
            source=chunk.source,
            dtype=chunk.dtype,
        )

    def reset(self) -> None:
        pass


class _AudioCapturingTurnDiarizer:
    def __init__(self) -> None:
        self.last_max_abs: float | None = None

    async def diarize(self, segments, audio=None):
        del segments, audio
        return []

    async def diarize_turns(self, audio: AudioChunk) -> list[SpeakerTurn]:
        self.last_max_abs = float(np.max(np.abs(audio.samples)))
        return [SpeakerTurn("SPEAKER_A", 0.0, 2.0)]


class _SlowTurnDiarizer:
    async def diarize(self, segments, audio=None):
        del segments, audio
        return []

    async def diarize_turns(self, audio: AudioChunk) -> list[SpeakerTurn]:
        del audio
        await asyncio.sleep(0.03)
        return [
            SpeakerTurn("SPEAKER_A", 0.0, 1.0),
            SpeakerTurn("SPEAKER_B", 1.0, 2.0),
        ]


class _TinyTailTurnDiarizer:
    async def diarize(self, segments, audio=None):
        del segments, audio
        return []

    async def diarize_turns(self, audio: AudioChunk) -> list[SpeakerTurn]:
        del audio
        return [
            SpeakerTurn("SPEAKER_A", 0.0, 1.0),
            SpeakerTurn("SPEAKER_B", 1.0, 1.0 + (100 / SR)),
        ]


class _ExclusiveTurnDiarizer:
    async def diarize(self, segments, audio=None):
        del segments, audio
        return []

    async def diarize_turns(self, audio: AudioChunk) -> list[SpeakerTurn]:
        del audio
        return [SpeakerTurn("SPEAKER_REGULAR", 0.0, 2.0)]

    async def diarize_turns_result(self, audio: AudioChunk) -> DiarizationTurnResult:
        del audio
        return DiarizationTurnResult(
            turns=[SpeakerTurn("SPEAKER_REGULAR", 0.0, 2.0)],
            exclusive_turns=[SpeakerTurn("SPEAKER_EXCLUSIVE", 0.0, 2.0)],
        )


def _write_wav(path: Path, duration_s: float = 2.0) -> Path:
    t = np.linspace(0.0, duration_s, int(SR * duration_s), endpoint=False, dtype=np.float32)
    samples = (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    sf.write(path, samples, SR, subtype="PCM_16")
    return path


@pytest.mark.asyncio
async def test_batch_pipeline_rebases_diarization_first_windows(tmp_path: Path) -> None:
    audio_file = _write_wav(tmp_path / "meeting.wav")
    pipeline = BatchPipeline(
        asr_engine=_FakeGigaAMEngine(),
        diarizer=_FakeTurnDiarizer(),
        preprocessor=PreProcessingPipeline([]),
        config=PipelineConfig(
            asr=ASRConfig(model_size="gigaam-v3-e2e-ctc"),
            diarization=DiarizationConfig(strategy="ml"),
        ),
        resolved_diarization_strategy="ml",
    )

    result = await pipeline.process_file(audio_file)

    assert [seg.diarized.speaker_id for seg in result.segments] == [
        "SPEAKER_A",
        "SPEAKER_B",
    ]
    assert result.segments[0].diarized.segment.start_time == pytest.approx(0.0)
    assert result.segments[1].diarized.segment.start_time == pytest.approx(1.0)
    assert result.processing_info["asr_segments"] == 2
    assert result.processing_info["diarized_segments"] == 2


@pytest.mark.asyncio
async def test_batch_pipeline_prefers_exclusive_turns_for_windows(tmp_path: Path) -> None:
    audio_file = _write_wav(tmp_path / "exclusive.wav")
    pipeline = BatchPipeline(
        asr_engine=_FakeGigaAMEngine(),
        diarizer=_ExclusiveTurnDiarizer(),
        preprocessor=PreProcessingPipeline([]),
        config=PipelineConfig(
            asr=ASRConfig(model_size="gigaam-v3-e2e-ctc"),
            diarization=DiarizationConfig(strategy="ml"),
        ),
        resolved_diarization_strategy="ml",
    )

    result = await pipeline.process_file(audio_file)

    assert [seg.diarized.speaker_id for seg in result.segments] == ["SPEAKER_EXCLUSIVE"]


@pytest.mark.asyncio
async def test_batch_pipeline_flattens_overlapping_turns(tmp_path: Path) -> None:
    audio_file = _write_wav(tmp_path / "overlap.wav", duration_s=3.0)
    pipeline = BatchPipeline(
        asr_engine=_FakeGigaAMEngine(),
        diarizer=_OverlappingTurnDiarizer(),
        preprocessor=PreProcessingPipeline([]),
        config=PipelineConfig(
            asr=ASRConfig(model_size="gigaam-v3-e2e-ctc"),
            diarization=DiarizationConfig(strategy="ml"),
        ),
        resolved_diarization_strategy="ml",
    )

    result = await pipeline.process_file(audio_file)

    assert [seg.diarized.speaker_id for seg in result.segments] == [
        "SPEAKER_A",
        "SPEAKER_B",
    ]
    assert result.segments[0].diarized.segment.start_time == pytest.approx(0.0)
    assert result.segments[0].diarized.segment.end_time == pytest.approx(1.0)
    assert result.segments[1].diarized.segment.start_time == pytest.approx(1.0)
    assert result.segments[1].diarized.segment.end_time == pytest.approx(3.0)


@pytest.mark.asyncio
async def test_batch_pipeline_uses_raw_audio_for_diarization_first(tmp_path: Path) -> None:
    audio_file = _write_wav(tmp_path / "raw_for_diarization.wav")
    diarizer = _AudioCapturingTurnDiarizer()
    pipeline = BatchPipeline(
        asr_engine=_FakeGigaAMEngine(),
        diarizer=diarizer,
        preprocessor=PreProcessingPipeline([_ZeroProcessor()]),
        config=PipelineConfig(
            asr=ASRConfig(model_size="gigaam-v3-e2e-ctc"),
            diarization=DiarizationConfig(strategy="ml"),
        ),
        resolved_diarization_strategy="ml",
    )

    await pipeline.process_file(audio_file)

    assert diarizer.last_max_abs is not None
    assert diarizer.last_max_abs > 0.1


@pytest.mark.asyncio
async def test_batch_pipeline_emits_progress_for_long_diarization_first_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio_file = _write_wav(tmp_path / "progress.wav")
    events = []
    monkeypatch.setattr("voxfusion.pipeline.batch._DIARIZATION_HEARTBEAT_S", 0.01)
    pipeline = BatchPipeline(
        asr_engine=_FakeGigaAMEngine(),
        diarizer=_SlowTurnDiarizer(),
        preprocessor=PreProcessingPipeline([]),
        config=PipelineConfig(
            asr=ASRConfig(model_size="gigaam-v3-e2e-ctc"),
            diarization=DiarizationConfig(strategy="ml"),
        ),
        resolved_diarization_strategy="ml",
        on_event=events.append,
    )

    await pipeline.process_file(audio_file)

    progress_messages = [
        event.message for event in events if event.event_type == EventType.PROGRESS
    ]
    assert any("Running speaker diarization" in message for message in progress_messages)
    assert any("Transcribing speaker windows" in message for message in progress_messages)
    assert any("ETA ~" in message for message in progress_messages)


@pytest.mark.asyncio
async def test_batch_pipeline_reports_unknown_eta_after_initial_estimate_is_exceeded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio_file = _write_wav(tmp_path / "progress_unknown_eta.wav")
    events = []
    monkeypatch.setattr("voxfusion.pipeline.batch._DIARIZATION_HEARTBEAT_S", 0.01)
    monkeypatch.setattr(
        "voxfusion.pipeline.batch._estimate_initial_diarization_total", lambda _duration: 0.01
    )
    pipeline = BatchPipeline(
        asr_engine=_FakeGigaAMEngine(),
        diarizer=_SlowTurnDiarizer(),
        preprocessor=PreProcessingPipeline([]),
        config=PipelineConfig(
            asr=ASRConfig(model_size="gigaam-v3-e2e-ctc"),
            diarization=DiarizationConfig(strategy="ml"),
        ),
        resolved_diarization_strategy="ml",
        on_event=events.append,
    )

    await pipeline.process_file(audio_file)

    progress_messages = [
        event.message for event in events if event.event_type == EventType.PROGRESS
    ]
    assert any(
        "Running speaker diarization" in message and "ETA ~unknown" in message
        for message in progress_messages
    )


@pytest.mark.asyncio
async def test_batch_pipeline_skips_ultra_short_diarized_windows(tmp_path: Path) -> None:
    audio_file = _write_wav(tmp_path / "tiny_tail.wav")
    asr = _FakeGigaAMEngine()
    pipeline = BatchPipeline(
        asr_engine=asr,
        diarizer=_TinyTailTurnDiarizer(),
        preprocessor=PreProcessingPipeline([]),
        config=PipelineConfig(
            asr=ASRConfig(model_size="gigaam-v3-e2e-ctc"),
            diarization=DiarizationConfig(strategy="ml"),
        ),
        resolved_diarization_strategy="ml",
    )

    result = await pipeline.process_file(audio_file)

    assert len(result.segments) == 1
    assert result.segments[0].diarized.speaker_id == "SPEAKER_A"
    assert len(asr.transcribe_calls) == 1
    assert asr.transcribe_calls[0][2] >= 320
