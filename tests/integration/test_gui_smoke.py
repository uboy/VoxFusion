"""Integration smoke tests for the Tkinter GUI."""

from __future__ import annotations

import tkinter as tk
from pathlib import Path

import pytest

from voxfusion.gui.main import CaptureOptions, TranscriptionGUI
from voxfusion.gui.runtime import DeviceOption


def _make_root() -> tk.Tk:
    try:
        root = tk.Tk()
    except tk.TclError as exc:  # pragma: no cover - depends on runner display setup
        pytest.skip(f"Tk is unavailable in this environment: {exc}")
    root.withdraw()
    return root


@pytest.mark.integration
def test_gui_smoke_builds_and_updates_model_summaries() -> None:
    root = _make_root()
    gui = TranscriptionGUI(
        root,
        CaptureOptions(
            model="small",
            language="ru",
            translate=None,
            microphone_device_id=None,
            system_device_id=None,
        ),
    )
    root.update_idletasks()

    assert gui.language_combo.cget("values")
    assert gui._file_lang_combo.cget("values")
    assert "Whisper Small" in gui._live_model_summary._name_label.cget("text")

    gui._model_var.set("large-v3")
    gui._on_model_changed()
    root.update_idletasks()
    assert "Whisper Large v3" in gui._live_model_summary._name_label.cget("text")

    gui._file_model_var.set("medium")
    gui._on_file_model_changed()
    root.update_idletasks()
    assert "Whisper Medium" in gui._file_model_summary._name_label.cget("text")

    gui._restore_redirection()
    root.destroy()


@pytest.mark.integration
def test_gui_smoke_loads_recorded_file_into_workflow(tmp_path: Path) -> None:
    root = _make_root()
    gui = TranscriptionGUI(
        root,
        CaptureOptions(
            model="small",
            language="ru",
            translate=None,
            microphone_device_id=None,
            system_device_id=None,
        ),
    )
    root.update_idletasks()

    recorded = tmp_path / "meeting.wav"
    recorded.write_bytes(b"RIFF")
    gui._load_recorded_file_into_flow(recorded, switch_tab=False)

    assert gui._file_path_var.get() == str(recorded)
    assert gui._last_recorded_file == recorded
    assert "Ready to transcribe recorded audio" in gui._file_status_label.cget("text")

    gui._restore_redirection()
    root.destroy()


@pytest.mark.integration
def test_gui_smoke_disables_live_start_for_file_only_model() -> None:
    root = _make_root()
    gui = TranscriptionGUI(
        root,
        CaptureOptions(
            model="small",
            language="ru",
            translate=None,
            microphone_device_id=None,
            system_device_id=None,
        ),
    )
    root.update_idletasks()

    gui._model_var.set("gigaam-v3-e2e-ctc")
    gui._on_model_changed()
    root.update_idletasks()

    assert str(gui.start_button.cget("state")) == "disabled"
    assert "file transcription only" in gui.status_label.cget("text")

    gui._restore_redirection()
    root.destroy()


@pytest.mark.integration
def test_gui_smoke_falls_back_to_first_devices_when_no_defaults_exist() -> None:
    root = _make_root()
    gui = TranscriptionGUI(
        root,
        CaptureOptions(
            model="small",
            language="ru",
            translate=None,
            microphone_device_id=None,
            system_device_id=None,
        ),
    )
    gui._device_options = [
        DeviceOption("Microphone: Mic A", "sd:1", "microphone", False),
        DeviceOption("System: Speakers A", "pa:2", "system", False),
    ]
    gui._rebuild_device_menu()
    gui._apply_default_device_selection()

    assert gui._selected_microphone_id == "sd:1"
    assert gui._selected_system_id == "pa:2"

    gui._restore_redirection()
    root.destroy()
