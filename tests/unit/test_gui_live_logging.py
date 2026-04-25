"""GUI logging tests for live capture startup."""

from __future__ import annotations

import importlib
from unittest.mock import MagicMock

from voxfusion.gui.main import TranscriptionGUI

gui_main = importlib.import_module("voxfusion.gui.main")


class _FakeVar:
    def __init__(self, value: str = "") -> None:
        self._value = value

    def get(self) -> str:
        return self._value

    def set(self, value: str) -> None:
        self._value = value


def test_start_capture_logs_requested_devices(monkeypatch) -> None:
    fake_log = MagicMock()
    worker_instances: list[object] = []

    class _FakeWorker:
        def __init__(self, **kwargs: object) -> None:
            worker_instances.append(self)
            self.kwargs = kwargs

        def start(self) -> None:
            return None

    gui = object.__new__(TranscriptionGUI)
    gui._worker = None
    gui._record_worker = None
    gui._model_var = _FakeVar("small")
    gui._language_var = _FakeVar("Auto")
    gui._translate_var = _FakeVar("")
    gui._selected_microphone_id = "sd:17"
    gui._selected_system_id = "pa:41"
    gui._set_live_status = MagicMock()
    gui._set_live_controls_enabled = MagicMock()
    gui.stop_button = MagicMock()
    gui.pause_button = MagicMock()
    gui._schedule_live_status = MagicMock()
    gui._schedule_segment = MagicMock()
    gui._schedule_error = MagicMock()
    gui._schedule_finished = MagicMock()
    gui._schedule_drop = MagicMock()
    gui._language_code_for_label = lambda _label, _model: None
    gui._tr = lambda key, **kwargs: key if not kwargs else f"{key}:{kwargs}"

    monkeypatch.setattr(gui_main, "log", fake_log)
    monkeypatch.setattr(gui_main, "CaptureWorker", _FakeWorker)
    monkeypatch.setattr(
        gui_main,
        "get_model_info",
        lambda _name: type(
            "ModelInfo", (), {"id": "small", "name": "Small", "supports_live_capture": True}
        )(),
    )

    TranscriptionGUI._start_capture(gui)

    assert worker_instances
    fake_log.info.assert_any_call(
        "gui.live_capture_requested",
        model="small",
        language=None,
        translate=None,
        source="both",
        microphone_device_id="sd:17",
        system_device_id="pa:41",
    )
