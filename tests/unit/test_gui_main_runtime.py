"""Tests for GUI startup runtime wiring."""

from __future__ import annotations

import importlib
import sys
import warnings


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


def test_gui_package_import_is_lazy_for_main_module() -> None:
    sys.modules.pop("voxfusion.gui.main", None)
    sys.modules.pop("voxfusion.gui", None)

    gui_pkg = importlib.import_module("voxfusion.gui")

    assert callable(gui_pkg.main)
    assert "voxfusion.gui.main" not in sys.modules


def test_gui_main_installs_torchcodec_warning_filter() -> None:
    gui_main = importlib.import_module("voxfusion.gui.main")

    assert any(
        action == "ignore"
        and category is UserWarning
        and "torchcodec is not installed correctly" in str(message)
        for action, message, category, _module, _lineno in warnings.filters
    )


def test_translate_label_code_roundtrip_supports_three_letter_codes() -> None:
    gui_main = importlib.import_module("voxfusion.gui.main")
    TranscriptionGUI = gui_main.TranscriptionGUI

    assert TranscriptionGUI._translate_label_to_code("Hawaiian (haw)") == "haw"
    assert TranscriptionGUI._translate_label_to_code("English (en)") == "en"
    assert TranscriptionGUI._translate_label_to_code("Off") == ""
