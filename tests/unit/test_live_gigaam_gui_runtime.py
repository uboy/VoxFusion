"""Unit tests for the GUI live GigaAM runtime path."""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from voxfusion.gui.runtime import CaptureOptions, CaptureWorker, derive_capture_source
from voxfusion.live_gigaam.session import _derive_capture_source
from voxfusion.models.diarization import DiarizedSegment
from voxfusion.models.transcription import TranscriptionSegment
from voxfusion.models.translation import TranslatedSegment


def _segment(text: str, *, start_s: float = 0.0, end_s: float = 1.0) -> TranslatedSegment:
    return TranslatedSegment(
        diarized=DiarizedSegment(
            segment=TranscriptionSegment(
                text=text,
                language="ru",
                start_time=start_s,
                end_time=end_s,
                confidence=0.0,
                words=None,
                no_speech_prob=0.0,
            ),
            speaker_id="SPEAKER_LOCAL",
            speaker_source="channel",
        ),
        translated_text=None,
        target_language=None,
    )


def test_derive_capture_source_treats_zero_indexes_as_selected() -> None:
    assert derive_capture_source(0, None) == "microphone"
    assert derive_capture_source(None, 0) == "system"
    assert derive_capture_source(0, 1) == "both"
    assert _derive_capture_source(0, None) == "microphone"
    assert _derive_capture_source(None, 0) == "system"
    assert _derive_capture_source(0, 1) == "both"


def test_capture_worker_routes_gigaam_segments_and_final_replacement(monkeypatch) -> None:
    statuses: list[str] = []
    rows: list[tuple[str, str, str, str | None]] = []
    replaced: list[list[tuple[str, str, str, str | None]]] = []

    class _FakeController:
        def __init__(self, **kwargs) -> None:
            self._on_status = kwargs["on_status"]
            self._on_segments = kwargs["on_segments"]
            self._on_finalized_segments = kwargs["on_finalized_segments"]

        def get_stats(self) -> dict[str, int]:
            return {"preprocess_q": 0, "asr_q": 0, "in_asr": 0, "dropped": 0}

        async def run(self, _stop_event) -> list[TranslatedSegment]:
            self._on_status("Live GigaAM started. Waiting for speech...")
            self._on_segments([_segment("draft text")])
            finalized = [_segment("final text")]
            self._on_finalized_segments(finalized)
            return finalized

    monkeypatch.setattr("voxfusion.gui.runtime.sys.platform", "win32")
    monkeypatch.setattr("voxfusion.gui.runtime._configure_gui_noise_controls", lambda: None)
    monkeypatch.setattr(
        "voxfusion.gui.runtime.load_config",
        lambda _overrides: SimpleNamespace(
            asr=SimpleNamespace(engine="gigaam", model_size="gigaam-v3-e2e-ctc"),
            capture=SimpleNamespace(),
            diarization=SimpleNamespace(channel_map={"microphone": "SPEAKER_LOCAL"}),
            translation=SimpleNamespace(enabled=False),
            live_gigaam=SimpleNamespace(),
            data_dir=str(Path.cwd()),
        ),
    )
    monkeypatch.setattr("voxfusion.gui.runtime.get_stage_progress", lambda *_args, **_kwargs: SimpleNamespace(update=lambda *_a, **_k: None))
    monkeypatch.setattr("voxfusion.live_gigaam.session.LiveGigaAMSessionController", _FakeController)

    worker = CaptureWorker(
        options=CaptureOptions(
            model="gigaam-v3-e2e-ctc",
            language="ru",
            translate=None,
            microphone_device_id="sd:17",
            system_device_id=None,
        ),
        on_status=statuses.append,
        on_segment=lambda *row: rows.append(row),
        on_replace_segments=lambda new_rows: replaced.append(list(new_rows)),
        on_error=lambda _message: None,
        on_finished=lambda: None,
    )

    asyncio.run(worker._run_async())

    assert any("Live GigaAM started" in status for status in statuses)
    assert rows and rows[0][2] == "draft text"
    assert replaced and replaced[0][0][2] == "final text"


def test_capture_worker_uses_capture_started_timestamp_anchor(monkeypatch) -> None:
    rows: list[tuple[str, str, str, str | None]] = []
    started_at = datetime(2026, 4, 3, 10, 0, 0)

    class _FakeController:
        def __init__(self, **kwargs) -> None:
            self._on_segments = kwargs["on_segments"]
            self._on_capture_started = kwargs["on_capture_started"]

        def get_stats(self) -> dict[str, int]:
            return {"preprocess_q": 0, "asr_q": 0, "in_asr": 0, "dropped": 0}

        async def run(self, _stop_event) -> list[TranslatedSegment]:
            self._on_capture_started(started_at)
            self._on_segments([_segment("anchored text", start_s=5.0, end_s=6.0)])
            return []

    monkeypatch.setattr("voxfusion.gui.runtime.sys.platform", "win32")
    monkeypatch.setattr("voxfusion.gui.runtime._configure_gui_noise_controls", lambda: None)
    monkeypatch.setattr(
        "voxfusion.gui.runtime.load_config",
        lambda _overrides: SimpleNamespace(
            asr=SimpleNamespace(engine="gigaam", model_size="gigaam-v3-e2e-ctc"),
            capture=SimpleNamespace(),
            diarization=SimpleNamespace(channel_map={"microphone": "SPEAKER_LOCAL"}),
            translation=SimpleNamespace(enabled=False),
            live_gigaam=SimpleNamespace(),
            data_dir=str(Path.cwd()),
        ),
    )
    monkeypatch.setattr(
        "voxfusion.gui.runtime.get_stage_progress",
        lambda *_args, **_kwargs: SimpleNamespace(update=lambda *_a, **_k: None),
    )
    monkeypatch.setattr("voxfusion.live_gigaam.session.LiveGigaAMSessionController", _FakeController)

    worker = CaptureWorker(
        options=CaptureOptions(
            model="gigaam-v3-e2e-ctc",
            language="ru",
            translate=None,
            microphone_device_id="sd:17",
            system_device_id=None,
        ),
        on_status=lambda _status: None,
        on_segment=lambda *row: rows.append(row),
        on_replace_segments=lambda _rows: None,
        on_error=lambda _message: None,
        on_finished=lambda: None,
    )

    asyncio.run(worker._run_async())

    assert rows and rows[0][0] == "10:00:05"
