"""Tests for GUI speaker-detection logging."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from voxfusion.gui.main import TranscriptionGUI

gui_main = importlib.import_module("voxfusion.gui.main")


class _FakeVar:
    def __init__(self, value: str = "") -> None:
        self._value = value

    def get(self) -> str:
        return self._value

    def set(self, value: str) -> None:
        self._value = value


class _ImmediateThread:
    def __init__(self, target, daemon: bool = False) -> None:
        del daemon
        self._target = target

    def start(self) -> None:
        self._target()


def test_detect_speakers_logs_completion(tmp_path: Path, monkeypatch) -> None:
    media_path = tmp_path / "meeting.webm"
    media_path.write_bytes(b"container")

    fake_log = MagicMock()

    async def _estimate_speaker_count(_audio, hf_token):
        del hf_token
        return 3

    fake_counter_module = SimpleNamespace(
        estimate_speaker_count=_estimate_speaker_count,
    )

    gui = object.__new__(TranscriptionGUI)
    gui._file_path_var = _FakeVar(str(media_path))
    gui._hf_token_var = _FakeVar("hf_token")
    gui._file_detect_btn = MagicMock()
    gui._file_status_label = MagicMock()
    gui._file_speaker_preset_var = MagicMock()
    gui._on_speaker_preset_changed = MagicMock()
    gui._refresh_file_diarization_controls = MagicMock()
    gui.root = SimpleNamespace(after=lambda _delay, fn, *args: fn(*args))

    monkeypatch.setattr(gui_main, "log", fake_log)
    monkeypatch.setattr(gui_main.threading, "Thread", _ImmediateThread)
    monkeypatch.setattr(
        gui_main,
        "load_detection_audio_chunk",
        lambda _path, max_duration_s=120.0: SimpleNamespace(
            samples=np.ones(8, dtype=np.float32),
            sample_rate=16_000,
        ),
    )
    monkeypatch.setitem(sys.modules, "voxfusion.diarization.speaker_counter", fake_counter_module)

    TranscriptionGUI._detect_speakers(gui)

    fake_log.info.assert_any_call(
        "gui.speaker_detect_started",
        file=str(media_path),
        max_sample_duration_s=120.0,
    )
    fake_log.info.assert_any_call(
        "gui.speaker_detect_completed",
        file=str(media_path),
        detected_speakers=3,
    )
    fake_log.error.assert_not_called()


def test_detect_speakers_logs_failure(tmp_path: Path, monkeypatch) -> None:
    media_path = tmp_path / "broken.webm"
    media_path.write_bytes(b"broken")

    fake_log = MagicMock()

    async def _estimate_speaker_count(_audio, hf_token):
        del hf_token
        return 2

    fake_counter_module = SimpleNamespace(
        estimate_speaker_count=_estimate_speaker_count,
    )

    gui = object.__new__(TranscriptionGUI)
    gui._file_path_var = _FakeVar(str(media_path))
    gui._hf_token_var = _FakeVar("hf_token")
    gui._file_detect_btn = MagicMock()
    gui._file_status_label = MagicMock()
    gui._file_speaker_preset_var = MagicMock()
    gui._on_speaker_preset_changed = MagicMock()
    gui._refresh_file_diarization_controls = MagicMock()
    gui.root = SimpleNamespace(after=lambda _delay, fn, *args: fn(*args))

    monkeypatch.setattr(gui_main, "log", fake_log)
    monkeypatch.setattr(gui_main.threading, "Thread", _ImmediateThread)
    monkeypatch.setattr(
        gui_main,
        "load_detection_audio_chunk",
        lambda _path, max_duration_s=120.0: (_ for _ in ()).throw(RuntimeError("bad media")),
    )
    monkeypatch.setitem(sys.modules, "voxfusion.diarization.speaker_counter", fake_counter_module)

    TranscriptionGUI._detect_speakers(gui)

    fake_log.error.assert_called_once_with(
        "gui.speaker_detect_failed",
        file=str(media_path),
        error="bad media",
    )
