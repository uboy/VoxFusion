"""Unit tests for GigaAM backend integration."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from voxfusion.asr.factory import create_asr_engine
from voxfusion.asr.gigaam_engine import GigaAMCTCEngine
from voxfusion.config.models import ASRConfig
from voxfusion.models.audio import AudioChunk


class _FakeGigaAMModel:
    """Minimal fake returned by AutoModel.from_pretrained for GigaAM."""

    @classmethod
    def from_pretrained(cls, _model_ref: str, **_kwargs: object) -> "_FakeGigaAMModel":
        return cls()

    def transcribe(self, wav_path: str) -> str:
        del wav_path
        return "privet mir"


def test_asr_config_sets_engine_for_gigaam_model() -> None:
    cfg = ASRConfig(model_size="gigaam-v3-e2e-ctc")
    assert cfg.engine == "gigaam"
    assert cfg.language is None


def test_factory_routes_gigaam_engine() -> None:
    engine, backend = create_asr_engine(ASRConfig(model_size="gigaam-v3-e2e-ctc"))
    assert backend == "gigaam"
    assert isinstance(engine, GigaAMCTCEngine)
    engine.close()


@pytest.mark.asyncio
async def test_gigaam_engine_transcribes_with_fake_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoModel = _FakeGigaAMModel  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    engine = GigaAMCTCEngine(ASRConfig(model_size="gigaam-v3-e2e-ctc", model_path="C:/models/gigaam"))
    chunk = AudioChunk(
        samples=np.ones(16000, dtype=np.float32),
        sample_rate=16000,
        channels=1,
        timestamp_start=0.0,
        timestamp_end=1.0,
        source="file",
        dtype="float32",
    )

    segments = await engine.transcribe(chunk, language="ru")
    assert len(segments) == 1
    assert segments[0].text == "privet mir"
    assert segments[0].language == "ru"
    engine.close()


def test_gigaam_normalize_audio_flattens_deep_arrays() -> None:
    engine = GigaAMCTCEngine()
    samples = np.ones((8, 1, 1), dtype=np.float32)

    normalized = engine._normalize_audio(samples, 16000)

    assert normalized.ndim == 1
    assert normalized.shape[0] == 8
    engine.close()
