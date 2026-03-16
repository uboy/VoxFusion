"""Unit tests for voxfusion.models data classes."""

import numpy as np

from voxfusion.models.audio import AudioChunk, AudioDeviceInfo
from voxfusion.models.diarization import DiarizedSegment
from voxfusion.models.result import TranscriptionResult
from voxfusion.models.transcription import TranscriptionSegment, WordTiming
from voxfusion.models.translation import TranslatedSegment


class TestAudioChunk:
    def test_duration(self, silence_chunk: AudioChunk) -> None:
        assert silence_chunk.duration == 1.0

    def test_num_samples_mono(self, silence_chunk: AudioChunk) -> None:
        assert silence_chunk.num_samples == 16000

    def test_num_samples_stereo(self, stereo_chunk: AudioChunk) -> None:
        assert stereo_chunk.num_samples == 44100

    def test_frozen(self, silence_chunk: AudioChunk) -> None:
        import pytest

        with pytest.raises(AttributeError):
            silence_chunk.sample_rate = 44100  # type: ignore[misc]

    def test_dtype_default(self, silence_chunk: AudioChunk) -> None:
        assert silence_chunk.dtype == "float32"

    def test_source(self, sine_chunk: AudioChunk) -> None:
        assert sine_chunk.source == "file"


class TestAudioDeviceInfo:
    def test_fields(self, audio_device_info: AudioDeviceInfo) -> None:
        assert audio_device_info.name == "Test Microphone"
        assert audio_device_info.is_default is True
        assert audio_device_info.device_type == "input"


class TestWordTiming:
    def test_fields(self, word_timings: list[WordTiming]) -> None:
        w = word_timings[0]
        assert w.word == "Hello"
        assert w.start_time == 0.0
        assert w.end_time == 0.5
        assert w.probability == 0.95


class TestTranscriptionSegment:
    def test_duration(self, transcription_segment: TranscriptionSegment) -> None:
        assert transcription_segment.duration == 1.0

    def test_text(self, transcription_segment: TranscriptionSegment) -> None:
        assert transcription_segment.text == "Hello world"

    def test_language(self, transcription_segment: TranscriptionSegment) -> None:
        assert transcription_segment.language == "en"

    def test_words_present(self, transcription_segment: TranscriptionSegment) -> None:
        assert transcription_segment.words is not None
        assert len(transcription_segment.words) == 2

    def test_words_absent(
        self, transcription_segment_no_words: TranscriptionSegment
    ) -> None:
        assert transcription_segment_no_words.words is None


class TestDiarizedSegment:
    def test_speaker_id(self, diarized_segment: DiarizedSegment) -> None:
        assert diarized_segment.speaker_id == "SPEAKER_00"

    def test_speaker_source(self, diarized_segment: DiarizedSegment) -> None:
        assert diarized_segment.speaker_source == "channel"

    def test_inner_segment(self, diarized_segment: DiarizedSegment) -> None:
        assert diarized_segment.segment.text == "Hello world"


class TestTranslatedSegment:
    def test_no_translation(self, translated_segment: TranslatedSegment) -> None:
        assert translated_segment.translated_text is None
        assert translated_segment.target_language is None

    def test_with_translation(
        self, translated_segment_with_translation: TranslatedSegment
    ) -> None:
        assert translated_segment_with_translation.translated_text == "Bonjour le monde"
        assert translated_segment_with_translation.target_language == "fr"


class TestTranscriptionResult:
    def test_segments(self, transcription_result: TranscriptionResult) -> None:
        assert len(transcription_result.segments) == 1

    def test_source_info(self, transcription_result: TranscriptionResult) -> None:
        assert transcription_result.source_info["file"] == "test.wav"

    def test_created_at(self, transcription_result: TranscriptionResult) -> None:
        assert "2025" in transcription_result.created_at

    def test_multi_segment(self, multi_segment_result: TranscriptionResult) -> None:
        assert len(multi_segment_result.segments) == 2
