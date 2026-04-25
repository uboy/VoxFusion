"""Unit tests for live GigaAM catalog behavior."""

from __future__ import annotations

from voxfusion import asr_catalog


def test_gigaam_model_info_marks_live_support() -> None:
    model = asr_catalog.get_model_info("gigaam-v3-e2e-ctc")
    assert model.engine == "gigaam"
    assert model.supports_live_capture is True
    assert model.supports_translation is False


def test_live_default_model_stays_on_whisper_even_if_gigaam_is_available(monkeypatch) -> None:
    available = (
        asr_catalog.get_model_info("gigaam-v3-e2e-ctc"),
        asr_catalog.get_model_info("small"),
        asr_catalog.get_model_info("tiny"),
    )
    monkeypatch.setattr(asr_catalog, "get_available_model_catalog", lambda: available)

    assert asr_catalog.get_default_model_id(for_live_capture=True) == "small"
