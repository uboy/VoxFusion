"""Regression tests for GUI speaker-range controls."""

from __future__ import annotations

from unittest.mock import MagicMock

from voxfusion.gui.main import TranscriptionGUI


class _FakeVar:
    def __init__(self, value: str = "") -> None:
        self._value = value

    def get(self) -> str:
        return self._value

    def set(self, value: str) -> None:
        self._value = value


def test_refresh_file_diarization_controls_keeps_min_max_editable_for_auto_mode() -> None:
    gui = object.__new__(TranscriptionGUI)
    gui._file_worker = None
    gui._file_path_var = MagicMock(**{"get.return_value": "C:/tmp/test.wav"})
    gui._file_diarization_var = MagicMock(**{"get.return_value": "auto"})
    gui._file_speaker_preset_var = MagicMock(**{"get.return_value": "Auto"})
    gui._file_diarization_combo = MagicMock()
    gui._file_speaker_preset_combo = MagicMock()
    gui._file_min_speakers_entry = MagicMock()
    gui._file_max_speakers_entry = MagicMock()
    gui._file_detect_btn = MagicMock()

    TranscriptionGUI._refresh_file_diarization_controls(gui)

    gui._file_min_speakers_entry.configure.assert_called_with(state="normal")
    gui._file_max_speakers_entry.configure.assert_called_with(state="normal")


def test_on_speaker_preset_changed_custom_preserves_manual_range() -> None:
    gui = object.__new__(TranscriptionGUI)
    gui._file_speaker_preset_var = _FakeVar("custom")
    gui._file_speaker_preset_display_var = _FakeVar("custom")
    gui._file_min_speakers_var = _FakeVar("4")
    gui._file_max_speakers_var = _FakeVar("6")
    gui._refresh_file_diarization_controls = MagicMock()
    gui._tr = lambda key, **kwargs: key

    TranscriptionGUI._on_speaker_preset_changed(gui)

    assert gui._file_speaker_preset_var.get() == "custom"
    assert gui._file_min_speakers_var.get() == "4"
    assert gui._file_max_speakers_var.get() == "6"


def test_on_detect_done_uses_custom_exact_range_for_four_or_more_speakers() -> None:
    gui = object.__new__(TranscriptionGUI)
    gui._file_detect_btn = MagicMock()
    gui._file_status_label = MagicMock()
    gui._refresh_file_diarization_controls = MagicMock()
    gui._file_speaker_preset_var = _FakeVar("auto")
    gui._file_speaker_preset_display_var = _FakeVar("")
    gui._file_min_speakers_var = _FakeVar("")
    gui._file_max_speakers_var = _FakeVar("")
    gui._tr = lambda key, **kwargs: key

    TranscriptionGUI._on_detect_done(gui, 4, None)

    assert gui._file_speaker_preset_var.get() == "custom"
    assert gui._file_min_speakers_var.get() == "4"
    assert gui._file_max_speakers_var.get() == "4"
