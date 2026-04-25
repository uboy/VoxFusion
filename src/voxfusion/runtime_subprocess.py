"""Helpers for suppressing transient console windows in GUI contexts."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_WINDOWS_CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)


def _is_windowed_windows_process() -> bool:
    """Return whether the current process is a windowed Windows app."""
    if sys.platform != "win32":
        return False
    exe_name = Path(sys.executable).name.lower()
    return exe_name in {"pythonw.exe", "pythonw", "voxfusion-gui.exe", "voxfusion-gui"}


def patch_subprocess_popen_no_window(*, force: bool = False) -> None:
    """Patch ``subprocess.Popen`` to hide child console windows on Windows."""
    if not force and not _is_windowed_windows_process():
        return

    original_init = subprocess.Popen.__init__
    if getattr(original_init, "_voxfusion_no_window", False):
        return

    def _patched(self: object, *args: object, **kwargs: object) -> None:
        creationflags = int(kwargs.get("creationflags", 0))
        kwargs["creationflags"] = creationflags | _WINDOWS_CREATE_NO_WINDOW

        if "startupinfo" not in kwargs and hasattr(subprocess, "STARTUPINFO"):
            startupinfo = subprocess.STARTUPINFO()
            if hasattr(subprocess, "STARTF_USESHOWWINDOW"):
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            if hasattr(subprocess, "SW_HIDE"):
                startupinfo.wShowWindow = subprocess.SW_HIDE
            kwargs["startupinfo"] = startupinfo

        original_init(self, *args, **kwargs)  # type: ignore[misc]

    _patched._voxfusion_no_window = True
    subprocess.Popen.__init__ = _patched  # type: ignore[method-assign]
