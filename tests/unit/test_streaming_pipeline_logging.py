"""Structured logging tests for streaming pipeline live stages."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import numpy as np

from voxfusion.config.models import PipelineConfig
from voxfusion.models.audio import AudioChunk
from voxfusion.models.diarization import DiarizedSegment
from voxfusion.models.transcription import TranscriptionSegment
from voxfusion.pipeline import streaming as streaming_module
from voxfusion.pipeline.streaming import StreamingPipeline


class _FakeSource:
    async def stream(self, chunk_duration_ms: int = 500):
        del chunk_duration_ms
        yield AudioChunk(
            samples=np.array([0.1, 0.2, 0.1, 0.0], dtype=np.float32),
            sample_rate=16000,
            channels=1,
            timestamp_start=0.0,
            timestamp_end=0.25,
            source="microphone",
            dtype="float32",
        )


class _FakePreprocessor:
    def process(self, chunk: AudioChunk) -> AudioChunk:
        return chunk


class _FakeASR:
    async def transcribe(self, chunk: AudioChunk):
        del chunk
        return [
            TranscriptionSegment(
                text="hello",
                language="en",
                start_time=0.0,
                end_time=0.25,
                confidence=0.9,
                words=None,
                no_speech_prob=0.0,
            )
        ]


class _FakeDiarizer:
    async def diarize(self, segments, chunk):
        del chunk
        return [DiarizedSegment(segment=segments[0], speaker_id="SPEAKER_00", speaker_source="channel")]


async def _run_pipeline(fake_log: MagicMock) -> list[object]:
    received: list[object] = []
    pipeline = StreamingPipeline(
        asr_engine=_FakeASR(),
        diarizer=_FakeDiarizer(),
        preprocessor=_FakePreprocessor(),
        translator=None,
        config=PipelineConfig(),
        queue_size=2,
    )
    streaming_module.log = fake_log
    await pipeline.run(_FakeSource(), on_segments=received.extend)
    return received


def test_streaming_pipeline_logs_stage_and_output_events() -> None:
    fake_log = MagicMock()

    received = asyncio.run(_run_pipeline(fake_log))

    assert len(received) == 1
    fake_log.info.assert_any_call("streaming.started", queue_size=2, lossy_mode=True)
    fake_log.info.assert_any_call("streaming.stage_started", stage="preprocessing")
    fake_log.info.assert_any_call("streaming.stage_started", stage="asr")
    fake_log.info.assert_any_call("streaming.stage_started", stage="diarization")
    fake_log.info.assert_any_call(
        "streaming.output_batch",
        segments=1,
        first_speaker="SPEAKER_00",
        first_start_s=0.0,
    )
    fake_log.info.assert_any_call("streaming.completed")
