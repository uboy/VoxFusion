"""Regression tests for shared FFmpeg runtime support on Windows."""

from __future__ import annotations

import io
import os
import zipfile
from pathlib import Path

from voxfusion.media.runtime_ffmpeg import (
    _extract_windows_ffmpeg_zip,
    activate_ffmpeg_runtime,
    find_ffmpeg,
)


def _shared_ffmpeg_zip_bytes() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w") as archive:
        archive.writestr("ffmpeg-build/bin/ffmpeg.exe", b"ffmpeg-binary")
        archive.writestr("ffmpeg-build/bin/ffprobe.exe", b"ffprobe-binary")
        archive.writestr("ffmpeg-build/bin/avcodec-61.dll", b"dll-binary")
    return buffer.getvalue()


def test_extract_windows_ffmpeg_zip_keeps_shared_dlls(tmp_path: Path) -> None:
    archive_path = tmp_path / "ffmpeg.zip"
    archive_path.write_bytes(_shared_ffmpeg_zip_bytes())

    ffmpeg_path = _extract_windows_ffmpeg_zip(archive_path, tmp_path / "vendor")

    assert ffmpeg_path.exists()
    assert ffmpeg_path.with_name("ffprobe.exe").exists()
    assert ffmpeg_path.with_name("avcodec-61.dll").exists()


def test_activate_ffmpeg_runtime_prepends_bin_dir_and_adds_dll_directory(
    tmp_path: Path,
    monkeypatch,
) -> None:
    managed = tmp_path / "ffmpeg-home"
    managed.mkdir(parents=True, exist_ok=True)
    ffmpeg_path = managed / "ffmpeg.exe"
    ffmpeg_path.write_bytes(b"ffmpeg")
    (managed / "avcodec-61.dll").write_bytes(b"dll")

    recorded: list[str] = []

    monkeypatch.setenv("VOXFUSION_FFMPEG_DIR", str(managed))
    monkeypatch.setenv("PATH", "")
    monkeypatch.setattr("platform.system", lambda: "Windows")
    monkeypatch.setattr("shutil.which", lambda _name: None)
    monkeypatch.setattr(os, "add_dll_directory", lambda path: recorded.append(path), raising=False)

    resolved = activate_ffmpeg_runtime()

    assert resolved == ffmpeg_path
    assert os.environ["PATH"].split(os.pathsep)[0] == str(managed)
    assert recorded == [str(managed)]


def test_find_ffmpeg_uses_pyinstaller_internal_dir(tmp_path: Path, monkeypatch) -> None:
    bundle_dir = tmp_path / "bundle"
    internal_dir = bundle_dir / "_internal"
    internal_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg_path = internal_dir / "ffmpeg.exe"
    ffmpeg_path.write_bytes(b"ffmpeg")

    monkeypatch.setattr("platform.system", lambda: "Windows")
    monkeypatch.setattr("shutil.which", lambda _name: None)
    monkeypatch.setattr("sys.executable", str(bundle_dir / "voxfusion-gui.exe"))

    assert find_ffmpeg() == ffmpeg_path
