"""Runtime helpers for locating or provisioning FFmpeg."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path

_WINDOWS_FFMPEG_SHARED_URL = (
    "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-full-shared.7z"
)
_WINDOWS_FFMPEG_ESSENTIALS_URL = (
    "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
)
_WINDOWS_DLL_DIRECTORY_HANDLES: dict[str, object] = {}


def _binary_name(tool: str) -> str:
    return f"{tool}.exe" if platform.system().lower() == "windows" else tool


def managed_ffmpeg_dir() -> Path:
    """Return the local application-managed FFmpeg directory."""
    override = os.environ.get("VOXFUSION_FFMPEG_DIR", "").strip()
    if override:
        return Path(override).expanduser()
    return Path.home() / ".voxfusion" / "ffmpeg"


def _repo_vendor_ffmpeg_dir() -> Path | None:
    """Return the repo-local FFmpeg vendor dir when running from a checkout."""
    try:
        repo_root = Path(__file__).resolve().parents[3]
    except IndexError:
        return None
    candidate = repo_root / "build" / "vendor" / "ffmpeg-runtime"
    return candidate if candidate.exists() else None


def _candidate_paths(tool: str) -> list[Path]:
    binary_name = _binary_name(tool)
    exe_dir = Path(sys.executable).parent
    managed_dir = managed_ffmpeg_dir()
    candidates = [
        exe_dir / binary_name,
        exe_dir / "bin" / binary_name,
        exe_dir / "_internal" / binary_name,
        exe_dir / "_internal" / "bin" / binary_name,
        managed_dir / binary_name,
        managed_dir / "bin" / binary_name,
    ]
    repo_vendor_dir = _repo_vendor_ffmpeg_dir()
    if repo_vendor_dir is not None:
        candidates.extend(
            [
                repo_vendor_dir / binary_name,
                repo_vendor_dir / "bin" / binary_name,
            ]
        )
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        meipass_dir = Path(meipass)
        candidates.extend(
            [
                meipass_dir / binary_name,
                meipass_dir / "bin" / binary_name,
            ]
        )
    return candidates


def find_ffmpeg(tool: str = "ffmpeg") -> Path | None:
    """Return the resolved path to FFmpeg/FFprobe if available."""
    env_key = f"VOXFUSION_{tool.upper()}_PATH"
    override = os.environ.get(env_key, "").strip()
    if override:
        override_path = Path(override).expanduser()
        if override_path.exists():
            return override_path

    for candidate in _candidate_paths(tool):
        if candidate.exists():
            return candidate

    found = shutil.which(_binary_name(tool)) or shutil.which(tool)
    return Path(found) if found else None


def activate_ffmpeg_runtime() -> Path | None:
    """Expose the FFmpeg bin directory to subprocesses and DLL loaders."""
    ffmpeg_path = find_ffmpeg()
    if ffmpeg_path is None:
        return None

    bin_dir = ffmpeg_path.parent
    current_path = os.environ.get("PATH", "")
    path_entries = [entry for entry in current_path.split(os.pathsep) if entry]
    if str(bin_dir) not in path_entries:
        os.environ["PATH"] = str(bin_dir) + (os.pathsep + current_path if current_path else "")

    add_dll_directory = getattr(os, "add_dll_directory", None)
    if platform.system().lower() == "windows" and callable(add_dll_directory):
        key = str(bin_dir)
        if key not in _WINDOWS_DLL_DIRECTORY_HANDLES:
            try:
                _WINDOWS_DLL_DIRECTORY_HANDLES[key] = add_dll_directory(key)
            except OSError:
                pass

    os.environ["FFMPEG_BINARY"] = str(ffmpeg_path)
    ffprobe_path = find_ffmpeg("ffprobe")
    if ffprobe_path is not None:
        os.environ["FFPROBE_BINARY"] = str(ffprobe_path)

    try:
        from pydub import AudioSegment

        AudioSegment.converter = str(ffmpeg_path)
        if hasattr(AudioSegment, "ffmpeg"):
            AudioSegment.ffmpeg = str(ffmpeg_path)
        if ffprobe_path is not None and hasattr(AudioSegment, "ffprobe"):
            AudioSegment.ffprobe = str(ffprobe_path)
    except Exception:
        pass

    return ffmpeg_path


def _find_7z_executable() -> str | None:
    for candidate in ("7z", "7za"):
        found = shutil.which(candidate)
        if found:
            return found
    return None


def _extract_windows_ffmpeg_zip(zip_path: Path, target_dir: Path) -> Path:
    """Extract FFmpeg binaries and shared DLLs from a Windows ZIP archive."""
    target_dir.mkdir(parents=True, exist_ok=True)
    extracted_ffmpeg: Path | None = None
    with zipfile.ZipFile(zip_path) as archive:
        members = {name.lower(): name for name in archive.namelist()}
        for lowered, member in members.items():
            if not lowered.endswith(".dll") and not lowered.endswith("/bin/ffmpeg.exe") and not lowered.endswith("/bin/ffprobe.exe"):
                continue
            output_path = target_dir / Path(member).name
            with archive.open(member) as src, output_path.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            if output_path.name.lower() == "ffmpeg.exe":
                extracted_ffmpeg = output_path
    if extracted_ffmpeg is None or not extracted_ffmpeg.exists():
        raise RuntimeError("Downloaded FFmpeg archive did not contain bin/ffmpeg.exe")
    return extracted_ffmpeg


def _extract_windows_ffmpeg_7z(archive_path: Path, target_dir: Path) -> Path:
    """Extract FFmpeg binaries and shared DLLs from a Windows 7z archive."""
    target_dir.mkdir(parents=True, exist_ok=True)
    extractor = _find_7z_executable()
    if extractor is None:
        raise RuntimeError("7z/7za is required to unpack the shared FFmpeg archive")

    command = [
        extractor,
        "e",
        "-y",
        f"-o{target_dir}",
        str(archive_path),
        "*/bin/ffmpeg.exe",
        "*/bin/ffprobe.exe",
        "*/bin/*.dll",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "7z extraction failed")

    ffmpeg_path = target_dir / "ffmpeg.exe"
    if not ffmpeg_path.exists():
        raise RuntimeError("Downloaded FFmpeg archive did not contain bin/ffmpeg.exe")
    return ffmpeg_path


def _extract_windows_ffmpeg_archive(archive_path: Path, target_dir: Path) -> Path:
    if archive_path.suffix.lower() == ".zip":
        return _extract_windows_ffmpeg_zip(archive_path, target_dir)
    if archive_path.suffix.lower() == ".7z":
        return _extract_windows_ffmpeg_7z(archive_path, target_dir)
    raise RuntimeError(f"Unsupported FFmpeg archive format: {archive_path.suffix}")


def install_ffmpeg_local(
    on_output: Callable[[str], None] | None = None,
) -> Path | None:
    """Install a local portable FFmpeg copy managed by VoxFusion.

    Returns the resolved `ffmpeg` path on success, or `None` when the current
    platform is unsupported or the download/extraction fails.
    """
    if platform.system().lower() != "windows":
        return None

    emit = on_output or (lambda _line: None)
    target_dir = managed_ffmpeg_dir()
    binary_name = _binary_name("ffmpeg")

    override = os.environ.get("VOXFUSION_FFMPEG_DIR", "").strip()
    if override:
        for candidate in (target_dir / binary_name, target_dir / "bin" / binary_name):
            if candidate.exists():
                return activate_ffmpeg_runtime() or candidate
    else:
        existing = activate_ffmpeg_runtime()
        if existing is not None:
            return existing

    target_dir.mkdir(parents=True, exist_ok=True)
    archives = [
        (
            target_dir / "ffmpeg-release-full-shared.7z",
            _WINDOWS_FFMPEG_SHARED_URL,
            "Downloading portable FFmpeg shared build...",
        ),
        (
            target_dir / "ffmpeg-release-essentials.zip",
            _WINDOWS_FFMPEG_ESSENTIALS_URL,
            "Downloading portable FFmpeg...",
        ),
    ]

    for archive_path, url, message in archives:
        try:
            if not archive_path.exists():
                emit(message)
                with urllib.request.urlopen(url, timeout=120) as response:
                    archive_path.write_bytes(response.read())
            emit("Extracting portable FFmpeg...")
            ffmpeg_path = _extract_windows_ffmpeg_archive(archive_path, target_dir)
            emit("Portable FFmpeg is ready.")
            return activate_ffmpeg_runtime() or ffmpeg_path
        except Exception:
            continue
    return None
