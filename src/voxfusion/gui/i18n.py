"""Lightweight GUI localization helpers."""

from __future__ import annotations

import json
import locale
import os
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

def detect_system_gui_language() -> str:
    """Best-effort detect the preferred GUI language from the local environment."""
    candidates = [
        os.environ.get("VOXFUSION_GUI_LANGUAGE", ""),
        os.environ.get("LC_ALL", ""),
        os.environ.get("LC_MESSAGES", ""),
        os.environ.get("LANG", ""),
    ]
    try:
        locale_code, _encoding = locale.getlocale()
    except ValueError:
        locale_code = None
    candidates.append(locale_code or "")

    for candidate in candidates:
        raw = str(candidate or "").strip()
        if not raw:
            continue
        normalized = raw.split(".", 1)[0].replace("-", "_").lower()
        prefix = normalized.split("_", 1)[0]
        if prefix in SUPPORTED_GUI_LANGUAGES:
            return prefix
    return DEFAULT_GUI_LANGUAGE


def resolve_initial_gui_language(
    saved_language_code: str | None,
    explicit_flag: str | bool | None = None,
) -> tuple[str, bool]:
    """Resolve the startup GUI language and whether it was chosen explicitly."""
    raw_saved = str(saved_language_code or "").strip()
    normalized_saved = normalize_gui_language(raw_saved) if raw_saved else ""

    parsed_explicit: bool | None = None
    if isinstance(explicit_flag, bool):
        parsed_explicit = explicit_flag
    else:
        raw_explicit = str(explicit_flag or "").strip().lower()
        if raw_explicit in {"1", "true", "yes", "on"}:
            parsed_explicit = True
        elif raw_explicit in {"0", "false", "no", "off"}:
            parsed_explicit = False

    if parsed_explicit is not None:
        if normalized_saved:
            return normalized_saved, parsed_explicit
        return detect_system_gui_language(), False

    if normalized_saved and normalized_saved != DEFAULT_GUI_LANGUAGE:
        return normalized_saved, True
    return detect_system_gui_language(), False
