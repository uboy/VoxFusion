"""Unit tests for ASR catalog metadata and compatibility helpers."""

import importlib.util

import pytest

from voxfusion.asr_catalog import (
    get_language_code,
    get_language_label,
    get_available_model_catalog,
    get_model_info,
    is_model_available,
    list_languages_for_model,
    list_model_ids,
    normalize_language_for_model,
)


def test_model_catalog_lists_builtin_models() -> None:
    ids = list_model_ids()
    assert "tiny" in ids
    assert "small" in ids
    assert "large-v3" in ids
    assert "gigaam-v3-e2e-ctc" in ids
    assert "parakeet-tdt-0.6b-v3" in ids
    assert "breeze-asr" in ids


def test_get_model_info_falls_back_to_default_model() -> None:
    assert get_model_info("missing-model").id == "small"


def test_languages_for_model_include_auto_and_supported_languages() -> None:
    labels = [item.label for item in list_languages_for_model("small")]
    assert labels[0] == "Auto Detect"
    assert "Russian" in labels
    assert "English" in labels


def test_normalize_language_for_model_rejects_unsupported_code() -> None:
    assert normalize_language_for_model("small", "xx") is None


def test_language_label_and_code_roundtrip_for_supported_language() -> None:
    label = get_language_label("ru", "small")
    assert label == "Russian"
    assert get_language_code(label, "small") == "ru"


def test_gigaam_requires_both_transformers_and_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    real_find_spec = importlib.util.find_spec

    def _fake_find_spec(name: str):
        if name == "transformers":
            return object()
        if name == "torch":
            return None
        return real_find_spec(name)

    monkeypatch.setattr(importlib.util, "find_spec", _fake_find_spec)
    assert is_model_available("gigaam-v3-e2e-ctc") is False


def test_available_model_catalog_hides_backends_with_missing_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    real_find_spec = importlib.util.find_spec

    def _fake_find_spec(name: str):
        if name in {"transformers", "torch", "nemo"}:
            return None
        return real_find_spec(name)

    monkeypatch.setattr(importlib.util, "find_spec", _fake_find_spec)
    ids = {model.id for model in get_available_model_catalog()}
    assert "small" in ids
    assert "gigaam-v3-e2e-ctc" not in ids
    assert "breeze-asr" not in ids
    assert "parakeet-tdt-0.6b-v3" not in ids
