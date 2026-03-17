"""Unit tests for AudioMixer source-fallback behaviour."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import numpy as np
import pytest

from voxfusion.capture.mixer import AudioMixer
from voxfusion.config.models import CaptureConfig
from voxfusion.exceptions import AudioCaptureError
from voxfusion.models.audio import AudioChunk


def _make_chunk(source: str = "test") -> AudioChunk:
    return AudioChunk(
        samples=np.zeros(800, dtype=np.float32),
        sample_rate=16000,
        channels=1,
        timestamp_start=0.0,
        timestamp_end=0.05,
        source=source,
        dtype="float32",
    )


class _WorkingSource:
    """Fake capture source that starts successfully and yields one chunk."""

    def __init__(self, label: str = "working") -> None:
        self._label = label
        self._config = CaptureConfig()

    @property
    def device_name(self) -> str:
        return f"fake:{self._label}"

    @property
    def sample_rate(self) -> int:
        return 16000

    @property
    def channels(self) -> int:
        return 1

    @property
    def is_active(self) -> bool:
        return True

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def stream(self, *, chunk_duration_ms: int = 500) -> AsyncIterator[AudioChunk]:
        yield _make_chunk(self._label)


class _FailingSource:
    """Fake capture source that raises on start()."""

    @property
    def device_name(self) -> str:
        return "fake:failing"

    @property
    def sample_rate(self) -> int:
        return 16000

    @property
    def channels(self) -> int:
        return 1

    @property
    def is_active(self) -> bool:
        return False

    async def start(self) -> None:
        raise OSError("Simulated device unavailable")

    async def stop(self) -> None:
        pass

    async def stream(self, *, chunk_duration_ms: int = 500) -> AsyncIterator[AudioChunk]:
        if False:  # pragma: no cover
            yield


@pytest.mark.asyncio
async def test_mixer_falls_back_to_working_source() -> None:
    """Mixer should stay operational when one source fails to start."""
    mixer = AudioMixer([_WorkingSource("mic"), _FailingSource()])
    await mixer.start()
    assert mixer.active_source_count == 1
    await mixer.stop()


@pytest.mark.asyncio
async def test_mixer_raises_when_all_sources_fail() -> None:
    """Mixer should raise AudioCaptureError when every source fails."""
    mixer = AudioMixer([_FailingSource(), _FailingSource()])
    with pytest.raises(AudioCaptureError):
        await mixer.start()


@pytest.mark.asyncio
async def test_mixer_yields_chunks_from_working_source() -> None:
    """Mixer should forward chunks from the one working source."""
    mixer = AudioMixer([_WorkingSource("mic"), _FailingSource()])
    await mixer.start()
    chunks: list[AudioChunk] = []
    async for chunk in mixer.stream(chunk_duration_ms=500):
        chunks.append(chunk)
    assert len(chunks) == 1
    assert chunks[0].source == "mic"
    await mixer.stop()


@pytest.mark.asyncio
async def test_mixer_forwards_all_sources_when_both_work() -> None:
    """Mixer should yield chunks from both sources when both start."""
    mixer = AudioMixer([_WorkingSource("mic"), _WorkingSource("system")])
    await mixer.start()
    assert mixer.active_source_count == 2
    chunks: list[AudioChunk] = []
    async for chunk in mixer.stream(chunk_duration_ms=500):
        chunks.append(chunk)
    sources = {c.source for c in chunks}
    assert sources == {"mic", "system"}
    await mixer.stop()
