"""Reusable GUI helper functions."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
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


def find_ffmpeg() -> Path | None:
    """Return the path to the ffmpeg executable, or None if not found.

    Checks (in order):
    1. Directory of the current executable (bundled binary)
    2. System PATH
    """
    # 1. Next to the running executable (PyInstaller bundle)
    exe_dir = Path(sys.executable).parent
    for candidate in (exe_dir / "ffmpeg.exe", exe_dir / "ffmpeg"):
        if candidate.exists():
            return candidate
    # 2. System PATH
    found = shutil.which("ffmpeg")
    return Path(found) if found else None


def install_ffmpeg_winget(on_output: "Callable[[str], None] | None" = None) -> bool:  # type: ignore[name-defined]
    """Install FFmpeg via winget (Windows 10/11 built-in package manager).

    Args:
        on_output: Optional callback called with each line of winget output.

    Returns:
        True if installation succeeded, False otherwise.
    """
    winget = shutil.which("winget")
    if not winget:
        return False
    try:
        proc = subprocess.Popen(
            [winget, "install", "--id", "Gyan.FFmpeg.Essentials",
             "--accept-package-agreements", "--accept-source-agreements"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if on_output:
            for line in proc.stdout or []:
                on_output(line.rstrip())
        proc.wait()
        return proc.returncode == 0
    except Exception:
        return False


def configure_gui_logging(level: int = logging.INFO) -> None:
    """Configure project logging for GUI mode."""
    level_name = logging.getLevelName(level)
    if not isinstance(level_name, str):
        level_name = "INFO"
    configure_logging(log_level=level_name, json_format=False, use_colors=False)
