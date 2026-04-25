"""Tests for GUI locale loading and localized selector helpers."""

from __future__ import annotations

from voxfusion.gui.i18n import (
    detect_system_gui_language,
    load_gui_locale,
    normalize_gui_language,
    resolve_initial_gui_language,
)
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

    assert gui._normalize_quality_label("Качество") == "Quality"
    assert gui._normalize_speaker_preset("2 говорящих") == "2"


def test_detect_system_gui_language_uses_supported_locale_prefix(monkeypatch) -> None:
    monkeypatch.setattr("voxfusion.gui.i18n.locale.getlocale", lambda: ("ru_RU", "UTF-8"))
    monkeypatch.delenv("VOXFUSION_GUI_LANGUAGE", raising=False)
    monkeypatch.delenv("LC_ALL", raising=False)
    monkeypatch.delenv("LC_MESSAGES", raising=False)
    monkeypatch.delenv("LANG", raising=False)

    assert detect_system_gui_language() == "ru"


def test_resolve_initial_gui_language_migrates_legacy_english_default(monkeypatch) -> None:
    monkeypatch.setattr("voxfusion.gui.i18n.detect_system_gui_language", lambda: "ru")

    assert resolve_initial_gui_language("en", None) == ("ru", False)


def test_resolve_initial_gui_language_preserves_explicit_english(monkeypatch) -> None:
    monkeypatch.setattr("voxfusion.gui.i18n.detect_system_gui_language", lambda: "ru")

    assert resolve_initial_gui_language("en", "true") == ("en", True)
