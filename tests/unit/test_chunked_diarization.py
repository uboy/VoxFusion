"""Focused tests for chunked diarization behavior."""

from __future__ import annotations

import numpy as np
import pytest

from voxfusion.diarization.alignment import SpeakerTurn
from voxfusion.diarization.chunked import ChunkedDiarizer
from voxfusion.models.audio import AudioChunk


class _FakeInnerDiarizer:
    async def diarize_turns(self, audio: AudioChunk) -> list[SpeakerTurn]:
        if audio.timestamp_start < 0.1:
            return [SpeakerTurn("LOCAL_A", 0.0, audio.duration)]
        return [SpeakerTurn("LOCAL_B", 0.0, audio.duration)]

    async def diarize(self, segments, audio=None):
        del segments, audio
        return []

    async def diarize_stream(self, segment_stream):
        del segment_stream
        if False:  # pragma: no cover
            yield


class _FakeThreeChunkInnerDiarizer:
    async def diarize_turns(self, audio: AudioChunk) -> list[SpeakerTurn]:
        if audio.timestamp_start < 0.1:
            speaker_id = "LOCAL_A"
        elif audio.timestamp_start < 5.1:
            speaker_id = "LOCAL_B"
        else:
            speaker_id = "LOCAL_C"
        return [SpeakerTurn(speaker_id, 0.0, audio.duration)]

    async def diarize(self, segments, audio=None):
        del segments, audio
        return []

    async def diarize_stream(self, segment_stream):
        del segment_stream
        if False:  # pragma: no cover
            yield


class _FakeAbsoluteInnerDiarizer:
    async def diarize_turns(self, audio: AudioChunk) -> list[SpeakerTurn]:
        if audio.timestamp_start < 0.1:
            return [SpeakerTurn("LOCAL_A", audio.timestamp_start, audio.timestamp_end)]
        return [SpeakerTurn("LOCAL_B", audio.timestamp_start, audio.timestamp_end)]

    async def diarize(self, segments, audio=None):
        del segments, audio
        return []

    async def diarize_stream(self, segment_stream):
        del segment_stream
        if False:  # pragma: no cover
            yield


def _make_audio(duration_s: float, sample_rate: int = 4) -> AudioChunk:
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


@pytest.mark.asyncio
@pytest.mark.parametrize("max_workers", [1, 2])
async def test_chunked_diarizer_reuses_overlap_to_keep_same_speaker_id(
    max_workers: int,
) -> None:
    diarizer = ChunkedDiarizer(
        _FakeInnerDiarizer,
        chunk_duration_s=5.0,
        chunk_overlap_s=2.0,
        max_workers=max_workers,
        device="cpu",
    )

    turns = await diarizer.diarize_turns(_make_audio(9.0))

    assert [(turn.speaker_id, turn.start_time, turn.end_time) for turn in turns] == [
        ("SPEAKER_00", 0.0, 5.0),
        ("SPEAKER_00", 5.0, 9.0),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("max_workers", [1, 2])
async def test_chunked_diarizer_stitches_same_speaker_across_three_chunks(
    max_workers: int,
) -> None:
    diarizer = ChunkedDiarizer(
        _FakeThreeChunkInnerDiarizer,
        chunk_duration_s=5.0,
        chunk_overlap_s=2.0,
        max_workers=max_workers,
        device="cpu",
    )

    turns = await diarizer.diarize_turns(_make_audio(13.0))

    assert [(turn.speaker_id, turn.start_time, turn.end_time) for turn in turns] == [
        ("SPEAKER_00", 0.0, 5.0),
        ("SPEAKER_00", 5.0, 10.0),
        ("SPEAKER_00", 10.0, 13.0),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("max_workers", [1, 2])
async def test_chunked_diarizer_accepts_absolute_inner_turns(
    max_workers: int,
) -> None:
    diarizer = ChunkedDiarizer(
        _FakeAbsoluteInnerDiarizer,
        chunk_duration_s=5.0,
        chunk_overlap_s=2.0,
        max_workers=max_workers,
        device="cpu",
    )

    turns = await diarizer.diarize_turns(_make_audio(9.0))

    assert [(turn.speaker_id, turn.start_time, turn.end_time) for turn in turns] == [
        ("SPEAKER_00", 0.0, 5.0),
        ("SPEAKER_00", 5.0, 9.0),
    ]
