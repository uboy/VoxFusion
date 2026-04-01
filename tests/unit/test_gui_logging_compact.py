"""Regression tests for compact GUI log rendering."""

from __future__ import annotations

import os

from voxfusion.gui.runtime import TextRedirector
from voxfusion.gui.runtime import _configure_gui_noise_controls
from voxfusion.logging import _compact_console_renderer
from voxfusion.logging import _should_suppress_log_message


def test_compact_console_renderer_uses_short_gui_format() -> None:
    rendered = _compact_console_renderer(
        None,  # type: ignore[arg-type]
        "info",
        {
            "timestamp": "2026-03-31T07:34:51.361406Z",
            "level": "info",
            "logger": "voxfusion.pipeline.orchestrator",
            "event": "orchestrator.components_ready",
            "asr_backend": "gigaam",
            "startup_warnings": [],
        },
    )

    assert rendered == "07:34:51 | INFO | orchestrator.components_ready | asr_backend=gigaam"


def test_should_suppress_generated_font_manager_noise() -> None:
    assert _should_suppress_log_message("generated new fontManager")


def test_configure_gui_noise_controls_disables_extra_telemetry(monkeypatch) -> None:
    monkeypatch.delenv("HF_HUB_DISABLE_TELEMETRY", raising=False)
    monkeypatch.delenv("PYANNOTE_METRICS_ENABLED", raising=False)

    _configure_gui_noise_controls()

    assert os.environ["HF_HUB_DISABLE_TELEMETRY"] == "1"
    assert os.environ["PYANNOTE_METRICS_ENABLED"] == "false"


def test_text_redirector_suppresses_font_manager_line() -> None:
    class _FakeWidget:
        def after(self, *_args, **_kwargs) -> None:
            pass

    redirector = TextRedirector(_FakeWidget())

    clean = redirector._sanitize(  # noqa: SLF001
        "useful line\n"
        "generated new fontManager\n"
        "still useful\n"
    )

    assert clean == "useful line\nstill useful\n"
