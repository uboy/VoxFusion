"""Unit tests for guided GUI flow helpers."""

from pathlib import Path

from voxfusion.gui.main import (
    _build_file_workflow_status,
    _default_transcript_path,
    _load_gui_settings,
    _resolve_preferred_device_index,
    _save_gui_settings,
    DeviceOption,
    TranscriptionGUI,
)


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


def test_resolve_preferred_device_index_prefers_ranked_input_for_auto() -> None:
    options = [
        DeviceOption("Auto (System default)", None),
        DeviceOption("Headset [Windows WASAPI #17]", 17),
        DeviceOption("Microphone [MME #1]", 1),
    ]
    assert _resolve_preferred_device_index(options, "Auto (System default)", "microphone") == 17


def test_resolve_preferred_device_index_keeps_none_for_system_auto() -> None:
    options = [
        DeviceOption("Auto (System default)", None),
        DeviceOption("Headset [Windows WASAPI #17]", 17),
    ]
    assert _resolve_preferred_device_index(options, "Auto (System default)", "system") is None


def test_language_helpers_use_catalog_labels() -> None:
    assert TranscriptionGUI._language_label_for_code("ru", "small") == "Russian"
    assert TranscriptionGUI._language_code_for_label("Russian", "small") == "ru"


def test_language_helper_returns_auto_for_unsupported_model_language_pair() -> None:
    assert TranscriptionGUI._language_code_for_label("English", "gigaam-v3-e2e-ctc") is None
