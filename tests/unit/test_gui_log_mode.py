"""Tests for GUI log-mode helpers."""

from __future__ import annotations

import logging

import voxfusion.gui.helpers as gui_helpers


def test_configure_gui_logging_passes_log_mode(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_configure_logging(**kwargs) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(gui_helpers, "configure_logging", _fake_configure_logging)

    gui_helpers.configure_gui_logging(logging.DEBUG, log_mode="debug")

    assert captured["log_level"] == "DEBUG"
    assert captured["renderer_style"] == "compact"
    assert captured["log_mode"] == "debug"
