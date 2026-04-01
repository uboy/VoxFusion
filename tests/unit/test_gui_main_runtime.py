"""Tests for GUI startup runtime wiring."""

from __future__ import annotations

import importlib


def test_gui_main_installs_no_window_subprocess_patch_on_windows(monkeypatch) -> None:
    gui_main = importlib.import_module("voxfusion.gui.main")
    seen: dict[str, object] = {}

    class FakeRoot:
        def mainloop(self) -> None:
            seen["mainloop"] = True

    monkeypatch.setattr(gui_main.sys, "platform", "win32")
    monkeypatch.setattr(
        gui_main,
        "patch_subprocess_popen_no_window",
        lambda *, force=False: seen.setdefault("force", force),
    )
    monkeypatch.setattr(gui_main.tk, "Tk", lambda: FakeRoot())
    monkeypatch.setattr(
        gui_main,
        "TranscriptionGUI",
        lambda root, options: seen.setdefault("gui", (root, options)),
    )

    result = gui_main.main([])

    assert result == 0
    assert seen["force"] is True
    assert seen["mainloop"] is True
