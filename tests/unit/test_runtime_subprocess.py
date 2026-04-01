"""Tests for Windows subprocess suppression in GUI runtimes."""

from __future__ import annotations

import importlib
from pathlib import Path


class _DummyStartupInfo:
    def __init__(self) -> None:
        self.dwFlags = 0
        self.wShowWindow = 1


def test_patch_subprocess_popen_no_window_sets_creationflags(monkeypatch) -> None:
    runtime_subprocess = importlib.import_module("voxfusion.runtime_subprocess")
    seen: dict[str, object] = {}

    class DummyPopen:
        def __init__(self, *args, **kwargs) -> None:
            seen["args"] = args
            seen["kwargs"] = kwargs

    monkeypatch.setattr(runtime_subprocess.subprocess, "Popen", DummyPopen)
    monkeypatch.setattr(runtime_subprocess.subprocess, "STARTUPINFO", _DummyStartupInfo, raising=False)
    monkeypatch.setattr(runtime_subprocess.subprocess, "STARTF_USESHOWWINDOW", 1, raising=False)
    monkeypatch.setattr(runtime_subprocess.subprocess, "SW_HIDE", 0, raising=False)
    monkeypatch.setattr(runtime_subprocess.sys, "platform", "win32")

    runtime_subprocess.patch_subprocess_popen_no_window(force=True)
    runtime_subprocess.subprocess.Popen(["ffprobe", "-version"])

    kwargs = seen["kwargs"]
    assert isinstance(kwargs, dict)
    assert int(kwargs["creationflags"]) & runtime_subprocess._WINDOWS_CREATE_NO_WINDOW
    assert "startupinfo" in kwargs


def test_patch_subprocess_popen_no_window_detects_pythonw(monkeypatch) -> None:
    runtime_subprocess = importlib.import_module("voxfusion.runtime_subprocess")
    seen: dict[str, object] = {}

    class DummyPopen:
        def __init__(self, *args, **kwargs) -> None:
            seen["kwargs"] = kwargs

    monkeypatch.setattr(runtime_subprocess.subprocess, "Popen", DummyPopen)
    monkeypatch.setattr(runtime_subprocess.subprocess, "STARTUPINFO", _DummyStartupInfo, raising=False)
    monkeypatch.setattr(runtime_subprocess.subprocess, "STARTF_USESHOWWINDOW", 1, raising=False)
    monkeypatch.setattr(runtime_subprocess.subprocess, "SW_HIDE", 0, raising=False)
    monkeypatch.setattr(runtime_subprocess.sys, "platform", "win32")
    monkeypatch.setattr(runtime_subprocess.sys, "executable", str(Path("C:/Python311/pythonw.exe")))

    runtime_subprocess.patch_subprocess_popen_no_window()
    runtime_subprocess.subprocess.Popen(["ffmpeg", "-version"])

    kwargs = seen["kwargs"]
    assert isinstance(kwargs, dict)
    assert int(kwargs["creationflags"]) & runtime_subprocess._WINDOWS_CREATE_NO_WINDOW
