"""Lightweight GUI localization helpers."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

DEFAULT_GUI_LANGUAGE = "en"
SUPPORTED_GUI_LANGUAGES: tuple[str, ...] = ("en", "ru", "zh")


def _locale_dir() -> Path:
    return Path(__file__).with_name("locales")


def normalize_gui_language(language_code: str | None) -> str:
    code = str(language_code or "").strip().lower()
    if code in SUPPORTED_GUI_LANGUAGES:
        return code
    return DEFAULT_GUI_LANGUAGE


def _load_locale_file(language_code: str) -> dict[str, str]:
    target = _locale_dir() / f"{language_code}.json"
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    result: dict[str, str] = {}
    for key, value in data.items():
        if isinstance(key, str) and isinstance(value, str):
            result[key] = value
    return result


@lru_cache(maxsize=len(SUPPORTED_GUI_LANGUAGES))
def load_gui_locale(language_code: str | None) -> dict[str, str]:
    normalized = normalize_gui_language(language_code)
    data = _load_locale_file(DEFAULT_GUI_LANGUAGE)
    if normalized != DEFAULT_GUI_LANGUAGE:
        data.update(_load_locale_file(normalized))
    return data
