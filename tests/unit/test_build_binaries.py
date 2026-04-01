"""Focused tests for build-time FFmpeg bundling helpers."""

from __future__ import annotations

import importlib.util
import io
import sys
import zipfile
from pathlib import Path


def _load_build_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "build_binaries.py"
    spec = importlib.util.spec_from_file_location("voxfusion_build_binaries_test", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _ffmpeg_zip_bytes() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w") as archive:
        archive.writestr("ffmpeg-build/bin/ffmpeg.exe", b"ffmpeg-binary")
        archive.writestr("ffmpeg-build/bin/ffprobe.exe", b"ffprobe-binary")
    return buffer.getvalue()


def test_extract_ffmpeg_binaries_from_zip(tmp_path: Path) -> None:
    module = _load_build_module()
    zip_path = tmp_path / "ffmpeg.zip"
    zip_path.write_bytes(_ffmpeg_zip_bytes())

    ffmpeg_path = module._extract_ffmpeg_binaries_from_zip(zip_path, tmp_path / "vendor")

    assert ffmpeg_path.name == "ffmpeg.exe"
    assert ffmpeg_path.exists()
    assert ffmpeg_path.with_name("ffprobe.exe").exists()


def test_ffmpeg_data_entries_use_prepared_vendor_copy(tmp_path: Path, monkeypatch) -> None:
    module = _load_build_module()
    prepared = tmp_path / "vendor" / "ffmpeg.exe"
    prepared.parent.mkdir(parents=True, exist_ok=True)
    prepared.write_bytes(b"ffmpeg")
    prepared.with_name("ffprobe.exe").write_bytes(b"ffprobe")

    monkeypatch.setattr(module.platform, "system", lambda: "Windows")
    monkeypatch.setattr(module, "_find_ffmpeg_binary", lambda: None)
    monkeypatch.setattr(module, "_ensure_windows_ffmpeg_binary", lambda: prepared)

    entries = module._ffmpeg_data_entries()

    assert f"{prepared}{module.os.pathsep}." in entries
    assert f"{prepared.with_name('ffprobe.exe')}{module.os.pathsep}." in entries
