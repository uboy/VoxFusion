"""FFmpeg-based audio extractor for video and non-PCM audio files.

Supports any container/codec that FFmpeg can decode: MP4, MKV, AVI, MOV,
WEBM, MP3, M4A, AAC, OPUS, WMA, etc.  The extracted audio is written to a
temporary WAV file at the requested sample rate and channel count, then
returned to the caller.  The caller is responsible for deleting the temp file.
"""

from __future__ import annotations

import asyncio
import subprocess
import tempfile
from pathlib import Path

from voxfusion.exceptions import AudioCaptureError
from voxfusion.logging import get_logger

log = get_logger(__name__)

# Extensions that soundfile cannot reliably open (video containers or
# compressed-audio formats that require a separate decoder).
_VIDEO_EXTENSIONS: frozenset[str] = frozenset({
    ".mp4", ".m4v", ".mkv", ".avi", ".mov", ".webm",
    ".flv", ".ts", ".m2ts", ".mts", ".wmv", ".3gp",
    ".ogv", ".mpg", ".mpeg", ".vob", ".divx",
})

_COMPRESSED_AUDIO_EXTENSIONS: frozenset[str] = frozenset({
    ".mp3", ".m4a", ".aac", ".opus", ".wma",
    ".ac3", ".dts", ".amr", ".mka",
})

NEEDS_EXTRACTION_EXTENSIONS: frozenset[str] = (
    _VIDEO_EXTENSIONS | _COMPRESSED_AUDIO_EXTENSIONS
)


def needs_extraction(path: Path) -> bool:
    """Return True if *path* requires FFmpeg audio extraction before processing."""
    return path.suffix.lower() in NEEDS_EXTRACTION_EXTENSIONS


def extract_audio(
    source_path: Path,
    *,
    sample_rate: int = 16_000,
    channels: int = 1,
    ffmpeg_binary: str = "ffmpeg",
) -> Path:
    """Extract the audio track from *source_path* to a temporary WAV file.

    The returned path points to a newly created temp file.  The caller must
    delete it when finished (e.g. with ``path.unlink(missing_ok=True)``).

    Args:
        source_path: Path to the input media file.
        sample_rate: Target sample rate in Hz.  Defaults to 16 000 (Whisper).
        channels: Number of output channels.  Defaults to 1 (mono).
        ffmpeg_binary: Name or path of the FFmpeg executable.

    Returns:
        Path to the extracted WAV file.

    Raises:
        AudioCaptureError: If FFmpeg is not found, times out, or exits with a
            non-zero return code.
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False, prefix="voxfusion_")
    tmp_path = Path(tmp.name)
    tmp.close()

    cmd: list[str] = [
        ffmpeg_binary,
        "-y",
        "-hide_banner",
        "-loglevel", "error",
        "-i", str(source_path),
        "-vn",                         # no video
        "-acodec", "pcm_s16le",        # 16-bit PCM
        "-ar", str(sample_rate),
        "-ac", str(channels),
        str(tmp_path),
    ]

    log.info(
        "extractor.start",
        source=str(source_path),
        target=str(tmp_path),
        sample_rate=sample_rate,
    )

    try:
        result = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=600,
        )
    except FileNotFoundError as exc:
        tmp_path.unlink(missing_ok=True)
        raise AudioCaptureError(
            "ffmpeg not found — install FFmpeg and make sure it is on PATH. "
            "Windows: https://www.gyan.dev/ffmpeg/builds/  "
            "Linux: sudo apt install ffmpeg"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        tmp_path.unlink(missing_ok=True)
        raise AudioCaptureError(
            f"FFmpeg timed out while extracting audio from {source_path.name}"
        ) from exc

    if result.returncode != 0:
        tmp_path.unlink(missing_ok=True)
        stderr = result.stderr.strip()
        raise AudioCaptureError(
            f"FFmpeg failed (exit {result.returncode}) processing "
            f"{source_path.name}: {stderr}"
        )

    log.info(
        "extractor.done",
        source=str(source_path),
        size_bytes=tmp_path.stat().st_size,
    )
    return tmp_path


async def extract_audio_async(
    source_path: Path,
    *,
    sample_rate: int = 16_000,
    channels: int = 1,
    ffmpeg_binary: str = "ffmpeg",
) -> Path:
    """Async wrapper around :func:`extract_audio` — offloads to a thread pool."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        lambda: extract_audio(
            source_path,
            sample_rate=sample_rate,
            channels=channels,
            ffmpeg_binary=ffmpeg_binary,
        ),
    )
