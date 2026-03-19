"""Unit tests for GigaAM backend integration."""

from __future__ import annotations

import os
import sys
import types

import numpy as np
import pytest

from voxfusion.asr.factory import create_asr_engine
from voxfusion.asr.gigaam_engine import (
    GigaAMCTCEngine,
    _install_megatron_compat_shim,
    _install_torchscript_source_fallback,
    _prepare_huggingface_runtime_env,
)
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


def test_prepare_huggingface_runtime_env_removes_deprecated_transformers_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_HOME", "C:/hf-home")
    monkeypatch.setenv("TRANSFORMERS_CACHE", "C:/old-cache")

    _prepare_huggingface_runtime_env()

    assert os.environ["HUGGINGFACE_HUB_CACHE"].endswith("hf-home\\hub") or os.environ["HUGGINGFACE_HUB_CACHE"].endswith("hf-home/hub")
    assert "TRANSFORMERS_CACHE" not in os.environ


def test_install_megatron_compat_shim_registers_num_microbatches_module() -> None:
    sys.modules.pop("megatron.core.num_microbatches_calculator", None)
    sys.modules.pop("megatron.core", None)
    sys.modules.pop("megatron", None)

    _install_megatron_compat_shim()

    mod = sys.modules["megatron.core.num_microbatches_calculator"]
    assert mod.get_num_microbatches() == 1


def test_install_torchscript_source_fallback_returns_original_object_on_source_error() -> None:
    class _FakeJit:
        def script(self, obj, *args, **kwargs):
            del args, kwargs
            raise RuntimeError(f"Can't get source for {obj}. TorchScript requires source access in order to carry out compilation")

    fake_torch = types.SimpleNamespace(jit=_FakeJit())
    _install_torchscript_source_fallback(fake_torch)
    marker = object()
    assert fake_torch.jit.script(marker) is marker
