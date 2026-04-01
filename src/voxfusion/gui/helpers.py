"""Reusable GUI helper functions."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf

from voxfusion.logging import configure_logging
from voxfusion.media.extractor import extract_audio, needs_extraction
from voxfusion.media.runtime_ffmpeg import (
    find_ffmpeg as _find_runtime_ffmpeg,
    install_ffmpeg_local as _install_runtime_ffmpeg_local,
)
from voxfusion.models.audio import AudioChunk


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


def app_base_dir() -> Path:
    """Return the base directory for application data.

    - PyInstaller bundle: directory containing the ``.exe``
    - Python script: project root (four levels above ``src/voxfusion/gui/``)
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parents[3]


def models_dir() -> Path:
    """Return (and create) the directory where models are stored.

    Resolves to ``<app_base_dir>/models/``.
    """
    path = app_base_dir() / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def gui_settings_path() -> Path:
    """Return the persistent GUI settings file path.

    Prefers an explicit override via ``VOXFUSION_GUI_SETTINGS_PATH``.
    Otherwise stores settings in a user-scoped directory to avoid
    polluting the repo/worktree during development.
    """
    override = os.environ.get("VOXFUSION_GUI_SETTINGS_PATH", "").strip()
    if override:
        return Path(override).expanduser()
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
    """Return the path to the ffmpeg executable, or None if not found."""
    return _find_runtime_ffmpeg()


def load_detection_audio_chunk(
    file_path: Path,
    *,
    max_duration_s: float = 120.0,
) -> AudioChunk:
    """Load a centered mono sample for GUI speaker-count detection.

    Container formats such as ``.webm`` are first decoded via FFmpeg to a
    temporary WAV file so speaker detection works on the same media types as
    normal file transcription.
    """
    extracted_path: Path | None = None

    def _read_chunk(path: Path) -> AudioChunk:
        with sf.SoundFile(str(path), mode="r") as audio_file:
            sample_rate = audio_file.samplerate
            total_frames = len(audio_file)
            if total_frames <= 0:
                return AudioChunk(
                    samples=np.zeros(0, dtype=np.float32),
                    sample_rate=sample_rate,
                    channels=1,
                    timestamp_start=0.0,
                    timestamp_end=0.0,
                    source="file",
                    dtype="float32",
                )
            max_frames = int(sample_rate * max_duration_s)
            frames_to_read = min(total_frames, max_frames)
            start_frame = max(0, (total_frames - frames_to_read) // 2)
            audio_file.seek(start_frame)
            samples = audio_file.read(
                frames_to_read,
                dtype="float32",
                always_2d=False,
            )

        samples = np.ascontiguousarray(samples, dtype=np.float32)
        if samples.ndim == 2:
            samples = samples.mean(axis=1).astype(np.float32)
        return AudioChunk(
            samples=np.ascontiguousarray(samples, dtype=np.float32),
            sample_rate=sample_rate,
            channels=1,
            timestamp_start=start_frame / sample_rate,
            timestamp_end=(start_frame + len(samples)) / sample_rate,
            source="file",
            dtype="float32",
        )

    try:
        if needs_extraction(file_path):
            extracted_path = extract_audio(file_path)
            return _read_chunk(extracted_path)

        try:
            return _read_chunk(file_path)
        except (RuntimeError, OSError):
            extracted_path = extract_audio(file_path)
            return _read_chunk(extracted_path)
    finally:
        if extracted_path is not None:
            extracted_path.unlink(missing_ok=True)


def _probe_duration_with_soundfile(file_path: Path) -> float | None:
    try:
        with sf.SoundFile(str(file_path), mode="r") as audio_file:
            sample_rate = audio_file.samplerate
            total_frames = len(audio_file)
    except (OSError, RuntimeError, ValueError):
        return None
    if sample_rate <= 0 or total_frames < 0:
        return None
    return total_frames / sample_rate


def _probe_duration_with_ffprobe(file_path: Path) -> float | None:
    ffprobe_path = _find_runtime_ffmpeg("ffprobe")
    if ffprobe_path is None:
        return None
    command = [
        str(ffprobe_path),
        "-v",
        "error",
        "-show_entries",
        "format=duration:stream=duration",
        "-of",
        "json",
        str(file_path),
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    try:
        payload = json.loads(result.stdout or "{}")
    except ValueError:
        return None

    candidates: list[float] = []
    format_block = payload.get("format")
    if isinstance(format_block, dict):
        raw_duration = format_block.get("duration")
        try:
            if raw_duration not in (None, "", "N/A"):
                candidates.append(float(raw_duration))
        except (TypeError, ValueError):
            pass

    streams = payload.get("streams")
    if isinstance(streams, list):
        for stream in streams:
            if not isinstance(stream, dict):
                continue
            raw_duration = stream.get("duration")
            try:
                if raw_duration not in (None, "", "N/A"):
                    candidates.append(float(raw_duration))
            except (TypeError, ValueError):
                continue

    positive = [value for value in candidates if value >= 0]
    if not positive:
        return None
    return max(positive)


def probe_media_size(file_path: Path) -> int | None:
    """Return best-effort file size in bytes."""
    try:
        return file_path.stat().st_size
    except OSError:
        return None


def probe_media_metadata(file_path: Path) -> tuple[float | None, int | None]:
    """Return best-effort `(duration_s, size_bytes)` for a media file."""
    size_bytes = probe_media_size(file_path)

    duration_s = _probe_duration_with_soundfile(file_path)
    if duration_s is None:
        duration_s = _probe_duration_with_ffprobe(file_path)
    if duration_s is None and needs_extraction(file_path):
        extracted_path: Path | None = None
        try:
            extracted_path = extract_audio(file_path)
            duration_s = _probe_duration_with_soundfile(extracted_path)
        except Exception:
            duration_s = None
        finally:
            if extracted_path is not None:
                extracted_path.unlink(missing_ok=True)
    return duration_s, size_bytes


def install_ffmpeg_local(on_output: "Callable[[str], None] | None" = None) -> bool:  # type: ignore[name-defined]
    """Install a local portable FFmpeg copy managed by VoxFusion.

    Args:
        on_output: Optional callback called with progress/status lines.

    Returns:
        True if installation succeeded, False otherwise.
    """
    return _install_runtime_ffmpeg_local(on_output=on_output) is not None


def get_system_proxies() -> dict[str, str]:
    """Return system-configured proxy URLs keyed by 'http' and 'https'.

    On Windows reads IE/WinHTTP proxy settings from the registry via
    ``urllib.request.getproxies()``.
    """
    proxies = urllib.request.getproxies()
    return {
        "http": proxies.get("http", ""),
        "https": proxies.get("https", ""),
        "no": proxies.get("no", "") or proxies.get("bypass", ""),
    }


def apply_proxy_settings(settings: dict[str, str]) -> None:
    """Apply proxy configuration as environment variables.

    HuggingFace Hub, ``requests``, and ``httpx`` all honour
    ``HTTP_PROXY`` / ``HTTPS_PROXY`` / ``NO_PROXY`` / ``REQUESTS_CA_BUNDLE``.

    Args:
        settings: Dict with keys ``proxy_use_system``, ``proxy_http``,
            ``proxy_https``, ``proxy_no``, ``proxy_ca_bundle``.
    """
    use_system = settings.get("proxy_use_system", "true").lower() != "false"

    if use_system:
        sys_proxies = get_system_proxies()
        http_proxy = sys_proxies["http"]
        https_proxy = sys_proxies["https"]
        no_proxy = sys_proxies["no"]
    else:
        http_proxy = settings.get("proxy_http", "").strip()
        https_proxy = settings.get("proxy_https", "").strip()
        no_proxy = settings.get("proxy_no", "").strip()

    ca_bundle = settings.get("proxy_ca_bundle", "").strip()

    for key in ("HTTP_PROXY", "http_proxy"):
        if http_proxy:
            os.environ[key] = http_proxy
        else:
            os.environ.pop(key, None)

    for key in ("HTTPS_PROXY", "https_proxy"):
        if https_proxy:
            os.environ[key] = https_proxy
        else:
            os.environ.pop(key, None)

    for key in ("NO_PROXY", "no_proxy"):
        if no_proxy:
            os.environ[key] = no_proxy
        else:
            os.environ.pop(key, None)

    if ca_bundle and Path(ca_bundle).exists():
        os.environ["REQUESTS_CA_BUNDLE"] = ca_bundle
        os.environ["SSL_CERT_FILE"] = ca_bundle
    else:
        os.environ.pop("REQUESTS_CA_BUNDLE", None)
        os.environ.pop("SSL_CERT_FILE", None)


def configure_gui_logging(level: int = logging.INFO, *, log_mode: str = "normal") -> None:
    """Configure project logging for GUI mode."""
    level_name = logging.getLevelName(level)
    if not isinstance(level_name, str):
        level_name = "INFO"
    configure_logging(
        log_level=level_name,
        json_format=False,
        use_colors=False,
        renderer_style="compact",
        log_mode=log_mode,
    )
