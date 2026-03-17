"""Reusable GUI helper functions."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from voxfusion.logging import configure_logging


def build_file_workflow_status(
    *,
    last_recorded_file: Path | None,
    transcript_ready: bool,
) -> str:
    """Return a short guided-flow hint for the file/LLM workflow."""
    if transcript_ready:
        return "Step 3: Review the transcript and send it to Open WebUI."
    if last_recorded_file is not None:
        return f"Step 2: Transcribe the latest recording ({last_recorded_file.name})."
    return "Step 1: Choose a file or record audio, then transcribe it."


def default_transcript_path(audio_path: Path) -> Path:
    """Return the default transcript file path next to the audio file."""
    return audio_path.with_suffix(".transcript.txt")


def gui_settings_path() -> Path:
    """Return the persistent GUI settings file path."""
    return Path.home() / ".voxfusion" / "gui_settings.json"


def load_gui_settings(path: Path | None = None) -> dict[str, str]:
    """Load persisted GUI settings."""
    target = path or gui_settings_path()
    if not target.exists():
        return {}
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(key): str(value) for key, value in data.items()}


def save_gui_settings(data: dict[str, str], path: Path | None = None) -> None:
    """Persist GUI settings."""
    target = path or gui_settings_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def configure_gui_logging(level: int = logging.INFO) -> None:
    """Configure project logging for GUI mode."""
    level_name = logging.getLevelName(level)
    if not isinstance(level_name, str):
        level_name = "INFO"
    configure_logging(log_level=level_name, json_format=False, use_colors=False)
