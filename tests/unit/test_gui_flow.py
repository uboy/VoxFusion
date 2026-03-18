"""Unit tests for guided GUI flow helpers."""

from pathlib import Path

from voxfusion.gui.main import (
    _build_file_workflow_status,
    _default_transcript_path,
    _load_gui_settings,
    _save_gui_settings,
    DeviceOption,
    TranscriptionGUI,
)
from voxfusion.gui.runtime import derive_capture_source


def test_workflow_status_without_recording_or_transcript() -> None:
    status = _build_file_workflow_status(last_recorded_file=None, transcript_ready=False)
    assert status == "Step 1: Choose a file or record audio, then transcribe it."


def test_workflow_status_with_recording_waits_for_transcription() -> None:
    status = _build_file_workflow_status(
        last_recorded_file=Path("meeting.wav"),
        transcript_ready=False,
    )
    assert status == "Step 2: Transcribe the latest recording (meeting.wav)."


def test_workflow_status_with_transcript_points_to_llm() -> None:
    status = _build_file_workflow_status(
        last_recorded_file=Path("meeting.wav"),
        transcript_ready=True,
    )
    assert status == "Step 3: Review the transcript and send it to Open WebUI."


def test_default_transcript_path_is_next_to_audio() -> None:
    assert _default_transcript_path(Path("C:/tmp/meeting.wav")).name == "meeting.transcript.txt"


def test_gui_settings_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "gui_settings.json"
    _save_gui_settings({"llm_url": "http://localhost:3000", "llm_model": "qwen"}, path)
    loaded = _load_gui_settings(path)
    assert loaded["llm_url"] == "http://localhost:3000"
    assert loaded["llm_model"] == "qwen"


def test_gui_settings_invalid_json_returns_empty(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{broken", encoding="utf-8")
    assert _load_gui_settings(path) == {}


def test_derive_capture_source_supports_microphone_only() -> None:
    assert derive_capture_source("sd:17", None) == "microphone"


def test_derive_capture_source_supports_system_only() -> None:
    assert derive_capture_source(None, "pa:17") == "system"


def test_derive_capture_source_supports_both() -> None:
    assert derive_capture_source("sd:17", "pa:17") == "both"


def test_device_option_fields_support_kind_and_default() -> None:
    option = DeviceOption("Microphone: Headset", "sd:17", "microphone", True)
    assert option.kind == "microphone"
    assert option.is_default is True


def test_language_helpers_use_catalog_labels() -> None:
    assert TranscriptionGUI._language_label_for_code("ru", "small") == "Russian"
    assert TranscriptionGUI._language_code_for_label("Russian", "small") == "ru"


def test_language_helper_returns_auto_for_unsupported_model_language_pair() -> None:
    assert TranscriptionGUI._language_code_for_label("English", "gigaam-v3-e2e-ctc") is None
