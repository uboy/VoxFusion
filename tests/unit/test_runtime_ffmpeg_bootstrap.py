"""Tests for FFmpeg runtime bootstrap details."""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

from voxfusion.media.runtime_ffmpeg import _repo_vendor_ffmpeg_dir
from voxfusion.media.runtime_ffmpeg import activate_ffmpeg_runtime


def test_activate_ffmpeg_runtime_sets_env_and_pydub_paths(
    tmp_path: Path,
    monkeypatch,
) -> None:
    managed = tmp_path / "ffmpeg-home"
    managed.mkdir(parents=True, exist_ok=True)
    ffmpeg_path = managed / "ffmpeg.exe"
    ffprobe_path = managed / "ffprobe.exe"
    ffmpeg_path.write_bytes(b"ffmpeg")
    ffprobe_path.write_bytes(b"ffprobe")

    class _FakeAudioSegment:
        converter = ""
        ffmpeg = ""
        ffprobe = ""

    fake_pydub = types.ModuleType("pydub")
    fake_pydub.AudioSegment = _FakeAudioSegment

    monkeypatch.setenv("VOXFUSION_FFMPEG_DIR", str(managed))
    monkeypatch.setenv("PATH", "")
    monkeypatch.setattr("platform.system", lambda: "Windows")
    monkeypatch.setattr("shutil.which", lambda _name: None)
    monkeypatch.setitem(sys.modules, "pydub", fake_pydub)
    monkeypatch.setattr(os, "add_dll_directory", lambda _path: object(), raising=False)

    resolved = activate_ffmpeg_runtime()

    assert resolved == ffmpeg_path
    assert os.environ["FFMPEG_BINARY"] == str(ffmpeg_path)
    assert os.environ["FFPROBE_BINARY"] == str(ffprobe_path)
    assert _FakeAudioSegment.converter == str(ffmpeg_path)
    assert _FakeAudioSegment.ffmpeg == str(ffmpeg_path)
    assert _FakeAudioSegment.ffprobe == str(ffprobe_path)


def test_repo_vendor_ffmpeg_dir_returns_existing_repo_vendor(
    tmp_path: Path,
    monkeypatch,
) -> None:
    fake_module = tmp_path / "src" / "voxfusion" / "media" / "runtime_ffmpeg.py"
    fake_module.parent.mkdir(parents=True, exist_ok=True)
    fake_module.write_text("# test\n", encoding="utf-8")
    vendor_dir = tmp_path / "build" / "vendor" / "ffmpeg-runtime"
    vendor_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        "voxfusion.media.runtime_ffmpeg.__file__",
        str(fake_module),
    )

    assert _repo_vendor_ffmpeg_dir() == vendor_dir
