"""GUI contract tests for live GigaAM model gating."""

from __future__ import annotations

import importlib
from unittest.mock import MagicMock

from voxfusion.gui.main import TranscriptionGUI


class _FakeVar:
    def __init__(self, value: str = "") -> None:
        self._value = value

    def get(self) -> str:
        return self._value

    def set(self, value: str) -> None:
        self._value = value


def _make_gui_stub() -> TranscriptionGUI:
    gui = object.__new__(TranscriptionGUI)
    gui._worker = None
    gui._record_worker = None
    gui._model_var = _FakeVar("gigaam-v3-e2e-ctc")
    gui._language_var = _FakeVar("Russian")
    gui._translate_var = _FakeVar("")
    gui._selected_microphone_id = "sd:17"
    gui._selected_system_id = None
    gui._set_live_status = MagicMock()
    gui._set_live_controls_enabled = MagicMock()
    gui.stop_button = MagicMock()
    gui.pause_button = MagicMock()
    gui._schedule_live_status = MagicMock()
    gui._schedule_segment = MagicMock()
    gui._schedule_replace_segments = MagicMock()
    gui._schedule_error = MagicMock()
    gui._schedule_finished = MagicMock()
    gui._schedule_drop = MagicMock()
    gui._language_code_for_label = lambda _label, _model: "ru"
    gui._tr = lambda key, **kwargs: key if not kwargs else f"{key}:{kwargs}"
    return gui


def test_start_capture_blocks_file_only_models(monkeypatch) -> None:
    gui = _make_gui_stub()
    gui_main = importlib.import_module("voxfusion.gui.main")

    monkeypatch.setattr(
        gui_main,
        "get_model_info",
        lambda _name: type(
            "ModelInfo",
            (),
            {
                "id": "breeze-asr",
                "name": "Breeze ASR",
                "supports_live_capture": False,
                "supports_translation": False,
            },
        )(),
    )

    TranscriptionGUI._start_capture(gui)

    gui._set_live_status.assert_called_once()
    assert gui._worker is None
    assert "live.status.file_only_model_action" in gui._set_live_status.call_args.args[0]


def test_start_capture_blocks_live_translation_for_gigaam(monkeypatch) -> None:
    gui = _make_gui_stub()
    gui._translate_var = _FakeVar("en")
    gui_main = importlib.import_module("voxfusion.gui.main")

    monkeypatch.setattr(
        gui_main,
        "get_model_info",
        lambda _name: type(
            "ModelInfo",
            (),
            {
                "id": "gigaam-v3-e2e-ctc",
                "name": "GigaAM v3",
                "supports_live_capture": True,
                "supports_translation": False,
            },
        )(),
    )

    TranscriptionGUI._start_capture(gui)

    gui._set_live_status.assert_called_once()
    assert gui._worker is None
    assert "live.status.translate_unsupported" in gui._set_live_status.call_args.args[0]
