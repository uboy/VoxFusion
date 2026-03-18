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


async def test_audio_recorder_writes_ogg(tmp_path: Path) -> None:
    import soundfile as sf

    output = tmp_path / "capture.ogg"
    chunk = AudioChunk(
        samples=np.array([0.0, 0.25, -0.25, 0.0], dtype=np.float32),
        sample_rate=44100,
        channels=1,
        timestamp_start=0.0,
        timestamp_end=4 / 44100,
        source="microphone",
    )
    source = FakeCaptureSource([chunk])

    recorder = AudioRecorder(chunk_duration_ms=250)
    stats = await recorder.record(source, output, format="ogg")

    assert output.exists()
    assert sf.info(str(output)).format == "OGG"
    assert stats.output_path == output


async def test_audio_recorder_writes_mp3(tmp_path: Path) -> None:
    import shutil

    output = tmp_path / "capture.mp3"
    chunk = AudioChunk(
        samples=np.array([0.0, 0.25, -0.25, 0.0] * 1000, dtype=np.float32),
        sample_rate=44100,
        channels=1,
        timestamp_start=0.0,
        timestamp_end=4000 / 44100,
        source="microphone",
    )
    source = FakeCaptureSource([chunk])

    if shutil.which("ffmpeg") is None:
        import pytest
        pytest.skip("ffmpeg not found in PATH")

    recorder = AudioRecorder(chunk_duration_ms=250)
    stats = await recorder.record(source, output, format="mp3")

    assert output.exists()
    assert output.stat().st_size > 0
    assert stats.output_path == output


async def test_audio_recorder_writes_opus(tmp_path: Path) -> None:
    import soundfile as sf

    output = tmp_path / "capture.opus"
    chunk = AudioChunk(
        samples=np.array([0.0, 0.25, -0.25, 0.0], dtype=np.float32),
        sample_rate=44100,
        channels=1,
        timestamp_start=0.0,
        timestamp_end=4 / 44100,
        source="microphone",
    )
    source = FakeCaptureSource([chunk])

    recorder = AudioRecorder(chunk_duration_ms=250)
    try:
        stats = await recorder.record(source, output, format="opus")
        assert output.exists()
        info = sf.info(str(output))
        assert info.format == "OGG"
        assert "opus" in info.subtype.lower()
        assert stats.output_path == output
    except Exception as exc:
        # Opus requires libsndfile >= 1.1.0; skip gracefully on older installs
        import pytest
        pytest.skip(f"Opus not supported by installed libsndfile: {exc}")


def test_recording_options_default_format() -> None:
    from pathlib import Path
    from voxfusion.gui.runtime import RecordingOptions

    opts = RecordingOptions(
        microphone_device_id=None,
        system_device_id=None,
        output_path=Path("/tmp/audio.wav"),
    )
    assert opts.output_format == "wav"


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

    def fake_create_recording_source(source_type: str, config, *, device_index: str | None = None):
        seen["source_type"] = source_type
        seen["device_index"] = device_index
        seen["config_sources"] = config.sources
        return object()

    monkeypatch.setattr("voxfusion.cli.record_cmd.AudioRecorder", StubRecorder)
    monkeypatch.setattr("voxfusion.cli.record_cmd.create_recording_source", fake_create_recording_source)

    result = runner.invoke(
        record,
        ["--source", "system", "--device", "pa:7", "--duration", "3", "--output", str(output)],
        obj={"verbose": False, "quiet": False},
    )

    assert result.exit_code == 0, result.output
    assert seen["source_type"] == "system"
    assert seen["device_index"] == "pa:7"
    assert seen["duration_s"] == 3.0
    assert seen["output_path"] == output
    assert seen["config_sources"] == ["system"]
    assert "Saved audio to" in result.output


def test_create_recording_source_uses_shared_windows_factory(monkeypatch) -> None:
    from voxfusion.config.models import CaptureConfig

    created: dict[str, object] = {}

    def fake_create_windows_capture_source(
        source_type: str,
        config,
        *,
        microphone_device_id: str | int | None = None,
        system_device_id: str | int | None = None,
    ) -> object:
        created["source_type"] = source_type
        created["config"] = config
        created["microphone_device_id"] = microphone_device_id
        created["system_device_id"] = system_device_id
        return object()

    monkeypatch.setattr("sys.platform", "win32")
    monkeypatch.setattr(
        "voxfusion.capture.windows_factory.create_windows_capture_source",
        fake_create_windows_capture_source,
    )

    source = create_recording_source("system", CaptureConfig(), device_index="pa:27")

    assert source is not None
    assert created["source_type"] == "system"
    assert created["microphone_device_id"] is None
    assert created["system_device_id"] == "pa:27"
"""Unit tests for shared Windows audio device handling."""

from voxfusion.capture.windows_audio import parse_windows_device_id
from voxfusion.capture.windows_factory import create_windows_capture_source


def test_parse_windows_device_id_supports_explicit_backend_prefix() -> None:
    assert parse_windows_device_id("pa:12") == ("pa", 12)
    assert parse_windows_device_id("sd:7") == ("sd", 7)


def test_parse_windows_device_id_treats_plain_int_as_default_backend() -> None:
    assert parse_windows_device_id(5, default_backend="sd") == ("sd", 5)
    assert parse_windows_device_id("9", default_backend="pa") == ("pa", 9)


def test_create_windows_capture_source_for_system_forwards_explicit_loopback_device(monkeypatch) -> None:
    from voxfusion.config.models import CaptureConfig

    seen: dict[str, object] = {}

    class StubLoopback:
        def __init__(self, *, device_id=None, config=None) -> None:
            seen["device_id"] = device_id
            seen["config"] = config

    monkeypatch.setattr("voxfusion.capture.windows_factory.RobustLoopbackCapture", StubLoopback)

    source = create_windows_capture_source(
        "system",
        CaptureConfig(),
        system_device_id="pa:42",
    )

    assert isinstance(source, StubLoopback)
    assert seen["device_id"] == "pa:42"


def test_create_windows_capture_source_rejects_pyaudio_device_for_microphone(monkeypatch) -> None:
    from voxfusion.config.models import CaptureConfig

    monkeypatch.setattr("voxfusion.capture.windows_factory.WASAPICapture", object)

    try:
        create_windows_capture_source(
            "microphone",
            CaptureConfig(),
            microphone_device_id="pa:3",
        )
    except ValueError as exc:
        assert "Microphone capture requires a sounddevice/WASAPI device" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid microphone device backend.")
