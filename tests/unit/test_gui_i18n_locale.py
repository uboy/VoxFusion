"""Tests for GUI locale loading and localized selector helpers."""

from __future__ import annotations

from voxfusion.gui.i18n import load_gui_locale
from voxfusion.gui.i18n import normalize_gui_language
from voxfusion.gui.main import TranscriptionGUI


def test_normalize_gui_language_falls_back_to_english() -> None:
    assert normalize_gui_language("de") == "en"


def test_load_gui_locale_merges_selected_locale_with_english_fallback() -> None:
    locale = load_gui_locale("ru")

    assert locale["header.logs"] == "Логи"
    assert locale["tooltip.header.settings"].startswith("Open global application settings")


def test_gui_quality_and_speaker_labels_accept_localized_values() -> None:
    gui = object.__new__(TranscriptionGUI)
    gui._locale = load_gui_locale("ru")

    assert gui._normalize_quality_label("Качество") == "Quality"  # noqa: SLF001
    assert gui._normalize_speaker_preset("2 говорящих") == "2"  # noqa: SLF001

