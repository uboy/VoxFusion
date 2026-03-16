"""Unit tests for raw audio recording."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import numpy as np
from click.testing import CliRunner

from voxfusion.cli.record_cmd import record
from voxfusion.models.audio import AudioChunk
from voxfusion.recording.factory import create_recording_source
from voxfusion.recording.recorder import AudioRecorder, _mix_chunks


class FakeCaptureSource:
    """Minimal async capture source for recorder tests."""

    def __init__(self, chunks: list[AudioChunk]) -> None:
        self._chunks = chunks
        self._active = False

    @property
    def device_name(self) -> str:
        return "fake"

    @property
    def sample_rate(self) -> int:
        return self._chunks[0].sample_rate

    @property
    def channels(self) -> int:
        return self._chunks[0].channels

    @property
    def is_active(self) -> bool:
        return self._active

    async def start(self) -> None:
        self._active = True

    async def stop(self) -> None:
        self._active = False

    async def read_chunk(self, duration_ms: int = 500) -> AudioChunk:
        raise NotImplementedError

    async def stream(self, chunk_duration_ms: int = 500) -> AsyncIterator[AudioChunk]:
        for chunk in self._chunks:
            if not self._active:
                break
            yield chunk


def test_mix_chunks_averages_overlapping_sources() -> None:
    sample_rate = 4
    chunks = [
        AudioChunk(
            samples=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            sample_rate=sample_rate,
            channels=1,
            timestamp_start=0.0,
            timestamp_end=1.0,
            source="microphone",
        ),
        AudioChunk(
            samples=np.array([0.0, 0.0, 0.5, 0.5], dtype=np.float32),
            sample_rate=sample_rate,
            channels=1,
            timestamp_start=0.0,
            timestamp_end=1.0,
            source="system",
        ),
    ]

    mixed = _mix_chunks(chunks, sample_rate=sample_rate, channels=1)

    assert mixed.tolist() == [0.5, 0.5, 0.75, 0.75]


async def test_audio_recorder_writes_wav(tmp_path: Path) -> None:
    output = tmp_path / "capture.wav"
    chunk = AudioChunk(
        samples=np.array([0.0, 0.25, -0.25, 0.0], dtype=np.float32),
        sample_rate=4,
        channels=1,
        timestamp_start=0.0,
        timestamp_end=1.0,
        source="microphone",
    )
    source = FakeCaptureSource([chunk])

    recorder = AudioRecorder(chunk_duration_ms=250)
    stats = await recorder.record(source, output)

    assert output.exists()
    assert stats.output_path == output
    assert stats.sample_rate == 4
    assert stats.channels == 1
    assert stats.chunks_captured == 1


async def test_audio_recorder_pause_skips_audio_and_compacts_timeline(tmp_path: Path) -> None:
    output = tmp_path / "paused.wav"
    first = AudioChunk(
        samples=np.array([0.1, 0.1, 0.1, 0.1], dtype=np.float32),
        sample_rate=4,
        channels=1,
        timestamp_start=0.0,
        timestamp_end=1.0,
        source="microphone",
    )
    skipped = AudioChunk(
        samples=np.array([0.9, 0.9, 0.9, 0.9], dtype=np.float32),
        sample_rate=4,
        channels=1,
        timestamp_start=1.0,
        timestamp_end=2.0,
        source="microphone",
    )
    resumed = AudioChunk(
        samples=np.array([0.2, 0.2, 0.2, 0.2], dtype=np.float32),
        sample_rate=4,
        channels=1,
        timestamp_start=2.0,
        timestamp_end=3.0,
        source="microphone",
    )

    class PauseAwareSource(FakeCaptureSource):
        async def stream(self, chunk_duration_ms: int = 500) -> AsyncIterator[AudioChunk]:
            if not self._active:
                return
            yield first
            recorder.request_pause()
            yield skipped
            recorder.request_resume()
            yield resumed

    recorder = AudioRecorder(chunk_duration_ms=250)
    source = PauseAwareSource([first, skipped, resumed])

    stats = await recorder.record(source, output)

    assert output.exists()
    assert stats.duration_s == 2.0


def test_record_command_invokes_recorder(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()
    output = tmp_path / "manual.wav"
    seen: dict[str, object] = {}

    class StubRecorder:
        def __init__(self, *, chunk_duration_ms: int = 500, on_status=None) -> None:
            seen["chunk_duration_ms"] = chunk_duration_ms
            self._on_status = on_status

        def request_stop(self) -> None:
            seen["stopped"] = True

        async def record(self, source, output_path: Path, *, duration_s: float | None = None):
            seen["source"] = source
            seen["output_path"] = output_path
            seen["duration_s"] = duration_s
            return type(
                "Stats",
                (),
                {
                    "output_path": output_path,
                    "duration_s": 3.0,
                    "sample_rate": 44100,
                    "channels": 1,
                },
            )()

    def fake_create_recording_source(source_type: str, config, *, device_index: int | None = None):
        seen["source_type"] = source_type
        seen["device_index"] = device_index
        seen["config_sources"] = config.sources
        return object()

    monkeypatch.setattr("voxfusion.cli.record_cmd.AudioRecorder", StubRecorder)
    monkeypatch.setattr("voxfusion.cli.record_cmd.create_recording_source", fake_create_recording_source)

    result = runner.invoke(
        record,
        ["--source", "system", "--device", "7", "--duration", "3", "--output", str(output)],
        obj={"verbose": False, "quiet": False},
    )

    assert result.exit_code == 0, result.output
    assert seen["source_type"] == "system"
    assert seen["device_index"] == 7
    assert seen["duration_s"] == 3.0
    assert seen["output_path"] == output
    assert seen["config_sources"] == ["system"]
    assert "Saved audio to" in result.output


def test_create_recording_source_prefers_stereo_mix_for_windows_system(monkeypatch) -> None:
    from voxfusion.config.models import CaptureConfig

    created: dict[str, object] = {}

    class StubCapture:
        def __init__(
            self,
            device_index: int | None = None,
            *,
            loopback: bool = False,
            source_label: str | None = None,
            config=None,
        ) -> None:
            created["device_index"] = device_index
            created["loopback"] = loopback
            created["source_label"] = source_label
            created["config"] = config

    monkeypatch.setattr("sys.platform", "win32")
    monkeypatch.setattr("voxfusion.capture.wasapi.find_stereo_mix_device", lambda: 27)
    monkeypatch.setattr("voxfusion.capture.wasapi.WASAPICapture", StubCapture)

    source = create_recording_source("system", CaptureConfig())

    assert isinstance(source, StubCapture)
    assert created["device_index"] == 27
    assert created["loopback"] is False
    assert created["source_label"] == "system"
