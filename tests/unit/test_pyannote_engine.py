"""Unit tests for pyannote diarization compatibility helpers."""

import sys
import types

import numpy as np
import pytest

from voxfusion.config.models import DiarizationMLConfig
from voxfusion.diarization.alignment import SpeakerTurn
from voxfusion.diarization.pyannote_engine import (
    PyAnnoteDiarizer,
    _extract_annotation,
    _extract_exclusive_annotation,
    _pipeline_auth_kwargs,
    _speaker_count_kwargs,
)
from voxfusion.diarization.types import DiarizationTurnResult
from voxfusion.models.audio import AudioChunk
from voxfusion.models.transcription import TranscriptionSegment


def _make_audio(duration_s: float = 2.0, sample_rate: int = 16_000) -> AudioChunk:
    sample_count = int(duration_s * sample_rate)
    return AudioChunk(
        samples=np.zeros(sample_count, dtype=np.float32),
        sample_rate=sample_rate,
        channels=1,
        timestamp_start=0.0,
        timestamp_end=duration_s,
        source="file",
        dtype="float32",
    )


def test_pipeline_auth_kwargs_prefers_token() -> None:
    def from_pretrained(model: str, token: str) -> object:
        return object()

    assert _pipeline_auth_kwargs(from_pretrained, "hf_test") == {"token": "hf_test"}


def test_pipeline_auth_kwargs_falls_back_to_use_auth_token() -> None:
    def from_pretrained(model: str, use_auth_token: str) -> object:
        return object()

    assert _pipeline_auth_kwargs(from_pretrained, "hf_test") == {"use_auth_token": "hf_test"}


def test_load_pipeline_uses_modern_token_kwarg(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []

    class FakePipeline:
        @staticmethod
        def from_pretrained(model: str, token: str) -> object:
            calls.append((model, token))
            return object()

    pyannote_module = types.ModuleType("pyannote")
    pyannote_audio_module = types.ModuleType("pyannote.audio")
    pyannote_audio_module.Pipeline = FakePipeline
    pyannote_module.audio = pyannote_audio_module

    monkeypatch.setitem(sys.modules, "pyannote", pyannote_module)
    monkeypatch.setitem(sys.modules, "pyannote.audio", pyannote_audio_module)

    diarizer = PyAnnoteDiarizer(
        DiarizationMLConfig(
            hf_auth_token="hf_test",
            device="cpu",
        )
    )

    pipeline = diarizer._load_pipeline()

    assert pipeline is diarizer._pipeline
    assert calls == [("pyannote/speaker-diarization-3.1", "hf_test")]


def test_extract_annotation_supports_new_diarize_output() -> None:
    class Annotation:
        def itertracks(self, yield_label: bool = False) -> list[object]:
            return []

    class DiarizeOutput:
        def __init__(self) -> None:
            self.speaker_diarization = Annotation()

    annotation = _extract_annotation(DiarizeOutput())

    assert hasattr(annotation, "itertracks")


def test_extract_exclusive_annotation_returns_optional_annotation() -> None:
    class Annotation:
        def itertracks(self, yield_label: bool = False) -> list[object]:
            return []

    class DiarizeOutput:
        def __init__(self) -> None:
            self.exclusive_speaker_diarization = Annotation()

    annotation = _extract_exclusive_annotation(DiarizeOutput())

    assert hasattr(annotation, "itertracks")


def test_speaker_count_kwargs_prefers_exact_num_speakers_when_supported() -> None:
    def diarize(input_data, num_speakers: int | None = None) -> object:
        return input_data, num_speakers

    kwargs = _speaker_count_kwargs(
        diarize,
        DiarizationMLConfig(min_speakers=4, max_speakers=4),
    )

    assert kwargs == {"num_speakers": 4}


def test_speaker_count_kwargs_falls_back_to_equal_bounds_when_needed() -> None:
    def diarize(
        input_data,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ) -> object:
        return input_data, min_speakers, max_speakers

    kwargs = _speaker_count_kwargs(
        diarize,
        DiarizationMLConfig(min_speakers=3, max_speakers=3),
    )

    assert kwargs == {"min_speakers": 3, "max_speakers": 3}


@pytest.mark.asyncio
async def test_diarize_prefers_exclusive_turns_for_alignment(monkeypatch) -> None:
    diarizer = PyAnnoteDiarizer(DiarizationMLConfig(device="cpu"))

    async def fake_turns_result(_audio: AudioChunk) -> DiarizationTurnResult:
        return DiarizationTurnResult(
            turns=[SpeakerTurn("REGULAR", 0.0, 2.0)],
            exclusive_turns=[SpeakerTurn("EXCLUSIVE", 0.0, 2.0)],
        )

    monkeypatch.setattr(diarizer, "diarize_turns_result", fake_turns_result)

    segments = [
        TranscriptionSegment(
            text="hello",
            language="ru",
            start_time=0.0,
            end_time=2.0,
            confidence=1.0,
            words=None,
            no_speech_prob=0.0,
        )
    ]

    result = await diarizer.diarize(segments, _make_audio())

    assert [item.speaker_id for item in result] == ["EXCLUSIVE"]
