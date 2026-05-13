"""Tests for GUI startup runtime wiring."""

from __future__ import annotations

import importlib
import sys


def test_gui_main_installs_torchcodec_warning_filter() -> None:
    # Check that the module source contains the filterwarnings call for torchcodec.
    # We verify the source rather than checking warnings.filters at runtime because
    # pytest resets warning filters between tests, making runtime checks unreliable.
    import inspect

    import voxfusion.gui.main as gui_main

    source = inspect.getsource(gui_main)
    assert "warnings.filterwarnings" in source
    assert "torchcodec" in source
    assert '"ignore"' in source


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
    # Snapshot all voxfusion.gui.* modules so we can restore them cleanly.
    saved = {
        k: v
        for k, v in list(sys.modules.items())
        if k == "voxfusion.gui" or k.startswith("voxfusion.gui.")
    }
    # Also snapshot parent package attribute so we can restore it (import sets it).
    saved_gui_attr = getattr(sys.modules["voxfusion"], "gui", None)
    for k in saved:
        sys.modules.pop(k, None)

    gui_pkg = importlib.import_module("voxfusion.gui")

    assert callable(gui_pkg.main)
    assert "voxfusion.gui.main" not in sys.modules

    # Restore original modules so subsequent tests are not poisoned by stale references.
    for k in list(sys.modules):
        if k == "voxfusion.gui" or k.startswith("voxfusion.gui."):
            if k not in saved:
                sys.modules.pop(k, None)
    sys.modules.update(saved)
    # Restore the gui attribute on the parent package so monkeypatch
    # resolution (which goes through parent attributes, not sys.modules) works.
    if saved_gui_attr is not None:
        sys.modules["voxfusion"].gui = saved_gui_attr


def test_translate_label_code_roundtrip_supports_three_letter_codes() -> None:
    gui_main = importlib.import_module("voxfusion.gui.main")
    transcription_gui = gui_main.TranscriptionGUI

    assert transcription_gui._translate_label_to_code("Hawaiian (haw)") == "haw"
    assert transcription_gui._translate_label_to_code("English (en)") == "en"
    assert transcription_gui._translate_label_to_code("Off") == ""
