"""Unit tests for voxfusion.diarization.channel."""

import pytest

from voxfusion.config.models import DiarizationConfig
from voxfusion.diarization.channel import ChannelDiarizer
from voxfusion.models.audio import AudioChunk
from voxfusion.models.transcription import TranscriptionSegment


@pytest.fixture
def diarizer() -> ChannelDiarizer:
    return ChannelDiarizer()


@pytest.fixture
def custom_diarizer() -> ChannelDiarizer:
    config = DiarizationConfig(
        channel_map={"microphone": "ME", "system": "REMOTE"},
    )
    return ChannelDiarizer(config)


class TestChannelDiarizer:
    @pytest.mark.asyncio
    async def test_diarize_file_source(
        self,
        diarizer: ChannelDiarizer,
        transcription_segment: TranscriptionSegment,
        silence_chunk: AudioChunk,
    ) -> None:
        result = await diarizer.diarize([transcription_segment], silence_chunk)
        assert len(result) == 1
        assert result[0].speaker_id == "SPEAKER_00"
        assert result[0].speaker_source == "channel"

    @pytest.mark.asyncio
    async def test_diarize_microphone_source(
        self,
        diarizer: ChannelDiarizer,
        transcription_segment: TranscriptionSegment,
        stereo_chunk: AudioChunk,
    ) -> None:
        result = await diarizer.diarize([transcription_segment], stereo_chunk)
        assert len(result) == 1
        # stereo_chunk has source="microphone"
        assert result[0].speaker_id == "SPEAKER_LOCAL"

    @pytest.mark.asyncio
    async def test_diarize_no_audio(
        self,
        diarizer: ChannelDiarizer,
        transcription_segment: TranscriptionSegment,
    ) -> None:
        result = await diarizer.diarize([transcription_segment], None)
        assert len(result) == 1
        assert result[0].speaker_id == "SPEAKER_00"

    @pytest.mark.asyncio
    async def test_custom_channel_map(
        self,
        custom_diarizer: ChannelDiarizer,
        transcription_segment: TranscriptionSegment,
        stereo_chunk: AudioChunk,
    ) -> None:
        result = await custom_diarizer.diarize([transcription_segment], stereo_chunk)
        assert result[0].speaker_id == "ME"

    @pytest.mark.asyncio
    async def test_empty_segments(
        self,
        diarizer: ChannelDiarizer,
        silence_chunk: AudioChunk,
    ) -> None:
        result = await diarizer.diarize([], silence_chunk)
        assert result == []
