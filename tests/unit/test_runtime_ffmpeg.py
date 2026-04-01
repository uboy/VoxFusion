"""Tests for runtime FFmpeg resolution and local install helpers."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

from voxfusion.media.runtime_ffmpeg import (
    _extract_windows_ffmpeg_zip,
    find_ffmpeg,
    install_ffmpeg_local,
)


def _ffmpeg_zip_bytes() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w") as archive:
        archive.writestr("ffmpeg-build/bin/ffmpeg.exe", b"ffmpeg-binary")
        archive.writestr("ffmpeg-build/bin/ffprobe.exe", b"ffprobe-binary")
    return buffer.getvalue()


def test_extract_windows_ffmpeg_zip_writes_binaries(tmp_path: Path) -> None:
    zip_path = tmp_path / "ffmpeg.zip"
    zip_path.write_bytes(_ffmpeg_zip_bytes())

    ffmpeg_path = _extract_windows_ffmpeg_zip(zip_path, tmp_path / "vendor")

    assert ffmpeg_path.name == "ffmpeg.exe"
    assert ffmpeg_path.exists()
    assert ffmpeg_path.with_name("ffprobe.exe").exists()


def test_find_ffmpeg_prefers_managed_copy(tmp_path: Path, monkeypatch) -> None:
    managed = tmp_path / "ffmpeg-home"
    binary = managed / "ffmpeg.exe"
    managed.mkdir(parents=True, exist_ok=True)
    binary.write_bytes(b"ffmpeg")

    monkeypatch.setenv("VOXFUSION_FFMPEG_DIR", str(managed))
    monkeypatch.setattr("platform.system", lambda: "Windows")
    monkeypatch.setattr("shutil.which", lambda _name: None)

    assert find_ffmpeg() == binary


def test_install_ffmpeg_local_downloads_and_extracts(tmp_path: Path, monkeypatch) -> None:
    managed = tmp_path / "ffmpeg-home"
    messages: list[str] = []

    class _FakeResponse(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setenv("VOXFUSION_FFMPEG_DIR", str(managed))
    monkeypatch.setattr("platform.system", lambda: "Windows")
    monkeypatch.setattr("shutil.which", lambda _name: None)
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda _url, timeout=120: _FakeResponse(_ffmpeg_zip_bytes()),
    )

    ffmpeg_path = install_ffmpeg_local(on_output=messages.append)

    assert ffmpeg_path is not None
    assert ffmpeg_path.exists()
    assert ffmpeg_path.name == "ffmpeg.exe"
    assert any("Downloading portable FFmpeg" in message for message in messages)
