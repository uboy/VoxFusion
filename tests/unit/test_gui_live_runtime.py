"""Focused runtime tests for live capture diagnostics."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import voxfusion.gui.runtime as gui_runtime
from voxfusion.gui.runtime import CaptureOptions, CaptureWorker


class _FakeProgress:
    def update(self, *_args, **_kwargs) -> None:
        return None


class _FakeASR:
    def load_model(self) -> None:
        return None

    def unload_model(self) -> None:
        return None

    def close(self) -> None:
        return None


class _FakePipeline:
    def __init__(self, **kwargs) -> None:
        del kwargs
        self._stats = {"preprocess_q": 1, "asr_q": 0, "in_asr": 0, "dropped": 0}

    def get_stats(self) -> dict[str, int]:
        return dict(self._stats)

    async def run(self, _audio_source, on_segments=None) -> None:
        del on_segments
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            return None

    async def stop(self) -> None:
        return None


class _FakeAudioSource:
    device_name = "fake:mic"
    is_active = True

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None


def test_capture_worker_logs_waiting_for_segments(monkeypatch) -> None:
    fake_log = MagicMock()
    statuses: list[str] = []

    worker = CaptureWorker(
        options=CaptureOptions(
            model="tiny",
            language="ru",
            translate=None,
            microphone_device_id="sd:17",
            system_device_id="pa:21",
        ),
        on_status=statuses.append,
        on_segment=lambda *_args: None,
        on_error=lambda _message: None,
        on_finished=lambda: None,
    )

    monotonic_values = iter([0.0, 11.0, 11.0, 11.0])

    async def _fake_sleep(_delay: float) -> None:
        worker._stop_event.set()
        return None

    monkeypatch.setattr(gui_runtime, "log", fake_log)
    monkeypatch.setattr(gui_runtime, "_configure_gui_noise_controls", lambda: None)
    monkeypatch.setattr(gui_runtime, "load_config", lambda _overrides: SimpleNamespace(
        asr=SimpleNamespace(model_size="tiny", language="ru", engine="faster-whisper"),
        diarization=SimpleNamespace(),
        capture=SimpleNamespace(lossy_mode=True, chunk_duration_ms=5000),
        translation=SimpleNamespace(target_language="en"),
    ))
    monkeypatch.setattr(gui_runtime, "PreProcessingPipeline", lambda _steps: object())
    monkeypatch.setattr(gui_runtime, "Resampler", lambda _rate: object())
    monkeypatch.setattr(gui_runtime, "Normalizer", lambda: object())
    monkeypatch.setattr(gui_runtime, "ChannelDiarizer", lambda _cfg: object())
    monkeypatch.setattr(gui_runtime, "StreamingPipeline", _FakePipeline)
    monkeypatch.setattr(gui_runtime, "derive_capture_source", lambda _mic, _sys: "microphone")
    monkeypatch.setattr(gui_runtime, "get_stage_progress", lambda _name, total=None: _FakeProgress())
    monkeypatch.setattr(gui_runtime, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(gui_runtime.asyncio, "sleep", _fake_sleep)
    monkeypatch.setattr("voxfusion.asr.factory.create_asr_engine", lambda _cfg: (_FakeASR(), "cpu"))
    monkeypatch.setattr("voxfusion.capture.windows_factory.create_windows_capture_source", lambda *args, **kwargs: _FakeAudioSource())
    monkeypatch.setattr("voxfusion.capture.vad_chunker.VadChunker", lambda source, max_duration_ms=5000: source)

    asyncio.run(worker._run_async())

    assert any("No speech segments yet" in status for status in statuses)
    fake_log.info.assert_any_call(
        "gui.live_waiting_for_segments",
        elapsed_s=11.0,
        since_last_segment_s=11.0,
        pipeline_stats={"preprocess_q": 1, "asr_q": 0, "in_asr": 0, "dropped": 0},
        active_sources=["fake:mic"],
    )
