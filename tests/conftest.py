"""Shared pytest fixtures for VoxFusion tests."""

from datetime import datetime, timezone

import numpy as np
import pytest

from voxfusion.config.models import (
    ASRConfig,
    DiarizationConfig,
    OutputConfig,
    PipelineConfig,
)
from voxfusion.models.audio import AudioChunk, AudioDeviceInfo
from voxfusion.models.diarization import DiarizedSegment
from voxfusion.models.result import TranscriptionResult
from voxfusion.models.transcription import TranscriptionSegment, WordTiming
from voxfusion.models.translation import TranslatedSegment


# -- Audio fixtures -----------------------------------------------------------


@pytest.fixture
def silence_chunk() -> AudioChunk:
    """One second of silence at 16kHz mono."""
    return AudioChunk(
        samples=np.zeros(16000, dtype=np.float32),
        sample_rate=16000,
        channels=1,
        timestamp_start=0.0,
        timestamp_end=1.0,
        source="file",
        dtype="float32",
    )


@pytest.fixture
def sine_chunk() -> AudioChunk:
    """One second of 440 Hz sine wave at 16kHz mono."""
    t = np.linspace(0, 1, 16000, endpoint=False, dtype=np.float32)
    samples = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    return AudioChunk(
        samples=samples,
        sample_rate=16000,
        channels=1,
        timestamp_start=0.0,
        timestamp_end=1.0,
        source="file",
        dtype="float32",
    )


@pytest.fixture
def stereo_chunk() -> AudioChunk:
    """One second of stereo audio at 44100 Hz."""
    t = np.linspace(0, 1, 44100, endpoint=False, dtype=np.float32)
    left = 0.3 * np.sin(2 * np.pi * 440 * t)
    right = 0.3 * np.sin(2 * np.pi * 880 * t)
    samples = np.column_stack([left, right]).astype(np.float32)
    return AudioChunk(
        samples=samples,
        sample_rate=44100,
        channels=2,
        timestamp_start=0.0,
        timestamp_end=1.0,
        source="microphone",
        dtype="float32",
    )


@pytest.fixture
def audio_device_info() -> AudioDeviceInfo:
    """A sample audio device info."""
    return AudioDeviceInfo(
        id="test-device-0",
        name="Test Microphone",
        sample_rate=44100,
        channels=1,
        device_type="input",
        is_default=True,
        platform_id="hw:0,0",
    )


# -- Transcription fixtures ---------------------------------------------------


@pytest.fixture
def word_timings() -> list[WordTiming]:
    """Sample word-level timings."""
    return [
        WordTiming(word="Hello", start_time=0.0, end_time=0.5, probability=0.95),
        WordTiming(word="world", start_time=0.6, end_time=1.0, probability=0.92),
    ]


@pytest.fixture
def transcription_segment(word_timings: list[WordTiming]) -> TranscriptionSegment:
    """A single transcription segment with word timings."""
    return TranscriptionSegment(
        text="Hello world",
        language="en",
        start_time=0.0,
        end_time=1.0,
        confidence=-0.3,
        words=word_timings,
        no_speech_prob=0.01,
    )


@pytest.fixture
def transcription_segment_no_words() -> TranscriptionSegment:
    """A transcription segment without word timings."""
    return TranscriptionSegment(
        text="This is a test",
        language="en",
        start_time=1.5,
        end_time=3.0,
        confidence=-0.25,
        words=None,
        no_speech_prob=0.02,
    )


# -- Diarization fixtures -----------------------------------------------------


@pytest.fixture
def diarized_segment(transcription_segment: TranscriptionSegment) -> DiarizedSegment:
    """A diarized segment."""
    return DiarizedSegment(
        segment=transcription_segment,
        speaker_id="SPEAKER_00",
        speaker_source="channel",
    )


@pytest.fixture
def diarized_segment_no_words(
    transcription_segment_no_words: TranscriptionSegment,
) -> DiarizedSegment:
    """A diarized segment without word timings."""
    return DiarizedSegment(
        segment=transcription_segment_no_words,
        speaker_id="SPEAKER_01",
        speaker_source="channel",
    )


# -- Translation fixtures -----------------------------------------------------


@pytest.fixture
def translated_segment(diarized_segment: DiarizedSegment) -> TranslatedSegment:
    """A translated segment (no actual translation)."""
    return TranslatedSegment(
        diarized=diarized_segment,
        translated_text=None,
        target_language=None,
    )


@pytest.fixture
def translated_segment_with_translation(
    diarized_segment: DiarizedSegment,
) -> TranslatedSegment:
    """A translated segment with translation."""
    return TranslatedSegment(
        diarized=diarized_segment,
        translated_text="Bonjour le monde",
        target_language="fr",
    )


# -- Result fixtures ----------------------------------------------------------


@pytest.fixture
def transcription_result(
    translated_segment: TranslatedSegment,
) -> TranscriptionResult:
    """A complete transcription result with one segment."""
    return TranscriptionResult(
        segments=[translated_segment],
        source_info={"file": "test.wav", "sample_rate": 16000, "duration_s": 1.0},
        processing_info={"asr_model": "faster-whisper/small", "processing_time_s": 0.5},
        created_at=datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc).isoformat(),
    )


@pytest.fixture
def multi_segment_result(
    translated_segment: TranslatedSegment,
    diarized_segment_no_words: DiarizedSegment,
) -> TranscriptionResult:
    """A result with multiple segments, some without words."""
    seg2 = TranslatedSegment(
        diarized=diarized_segment_no_words,
        translated_text=None,
        target_language=None,
    )
    return TranscriptionResult(
        segments=[translated_segment, seg2],
        source_info={"file": "multi.wav", "sample_rate": 16000, "duration_s": 3.0},
        processing_info={"asr_model": "faster-whisper/small", "processing_time_s": 1.2},
        created_at=datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc).isoformat(),
    )


# -- Config fixtures ----------------------------------------------------------


@pytest.fixture
def default_config() -> PipelineConfig:
    """Default PipelineConfig with all defaults."""
    return PipelineConfig()


@pytest.fixture
def asr_config() -> ASRConfig:
    """Default ASR configuration."""
    return ASRConfig()
