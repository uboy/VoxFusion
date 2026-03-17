"""Unit tests for ASR catalog metadata and compatibility helpers."""

from voxfusion.asr_catalog import (
    get_language_code,
    get_language_label,
    get_model_info,
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
