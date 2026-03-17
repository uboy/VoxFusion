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


class _FakeProcessor:
    @classmethod
    def from_pretrained(cls, _model_ref: str, *, local_files_only: bool) -> "_FakeProcessor":
        del local_files_only
        return cls()

    def __call__(self, audio, *, sampling_rate: int, return_tensors: str) -> dict[str, np.ndarray]:
        assert sampling_rate == 16000
        assert return_tensors == "np"
        return {"input_values": np.asarray([audio], dtype=np.float32)}

    def batch_decode(self, token_ids, *, skip_special_tokens: bool) -> list[str]:
        del token_ids, skip_special_tokens
        return ["privet mir"]


class _FakeModel:
    @classmethod
    def from_pretrained(
        cls,
        _model_ref: str,
        *,
        local_files_only: bool,
        provider: str,
    ) -> "_FakeModel":
        del local_files_only
        assert provider == "CPUExecutionProvider"
        return cls()

    def __call__(self, **_inputs) -> types.SimpleNamespace:
        logits = np.array([[[0.1, 0.9], [0.8, 0.2]]], dtype=np.float32)
        return types.SimpleNamespace(logits=logits)


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
    fake_optimum = types.ModuleType("optimum")
    fake_ort = types.ModuleType("optimum.onnxruntime")
    fake_ort.ORTModelForCTC = _FakeModel
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoProcessor = _FakeProcessor

    monkeypatch.setitem(sys.modules, "optimum", fake_optimum)
    monkeypatch.setitem(sys.modules, "optimum.onnxruntime", fake_ort)
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
