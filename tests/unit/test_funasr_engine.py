"""Unit tests for FunASR Paraformer backend wiring."""

from __future__ import annotations

import builtins
import sys
import types

import numpy as np
import pytest

from voxfusion.asr.factory import create_asr_engine
from voxfusion.asr.funasr_engine import FunASREngine
from voxfusion.config.models import ASRConfig
from voxfusion.exceptions import ModelLoadError
from voxfusion.models.audio import AudioChunk


def test_factory_routes_funasr_engine() -> None:
    engine, backend = create_asr_engine(ASRConfig(model_size="funasr-paraformer-zh"))
    assert backend == "funasr"
    assert isinstance(engine, FunASREngine)
    engine.close()


def test_asr_config_sets_engine_for_funasr_model() -> None:
    cfg = ASRConfig(model_size="funasr-paraformer-zh")
    assert cfg.engine == "funasr"


def test_funasr_engine_properties() -> None:
    engine = FunASREngine()
    assert "funasr" in engine.model_name
    assert engine.supported_languages == ["zh"]
    engine.close()


@pytest.mark.asyncio
async def test_funasr_engine_transcribes_with_fake_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_funasr = types.ModuleType("funasr")

    class _FakeAutoModel:
        def __init__(self, model: str, **kwargs: object) -> None:
            del model, kwargs

        def generate(self, **kwargs: object) -> list[dict]:
            del kwargs
            return [{"text": "ni hao shijie"}]

    fake_funasr.AutoModel = _FakeAutoModel
    monkeypatch.setitem(sys.modules, "funasr", fake_funasr)

    engine = FunASREngine(ASRConfig(model_size="funasr-paraformer-zh", model_path="/fake/model"))
    chunk = AudioChunk(
        samples=np.ones(16000, dtype=np.float32),
        sample_rate=16000,
        channels=1,
        timestamp_start=0.0,
        timestamp_end=1.0,
        source="file",
        dtype="float32",
    )

    segments = await engine.transcribe(chunk, language="zh")
    assert len(segments) == 1
    assert segments[0].text == "ni hao shijie"
    assert segments[0].language == "zh"
    engine.close()


@pytest.mark.asyncio
async def test_funasr_engine_returns_empty_for_too_short_audio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_funasr = types.ModuleType("funasr")

    class _FakeAutoModel:
        def __init__(self, model: str, **kwargs: object) -> None:
            del model, kwargs

        def generate(self, **kwargs: object) -> list[dict]:
            del kwargs
            return [{"text": "should not appear"}]

    fake_funasr.AutoModel = _FakeAutoModel
    monkeypatch.setitem(sys.modules, "funasr", fake_funasr)

    engine = FunASREngine(ASRConfig(model_size="funasr-paraformer-zh", model_path="/fake/model"))
    # Only 100 samples — well below the 320 minimum.
    chunk = AudioChunk(
        samples=np.zeros(100, dtype=np.float32),
        sample_rate=16000,
        channels=1,
        timestamp_start=0.0,
        timestamp_end=0.00625,
        source="file",
        dtype="float32",
    )

    segments = await engine.transcribe(chunk)
    assert segments == []
    engine.close()


def test_funasr_engine_reports_missing_dependency_cleanly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):  # noqa: A002
        if name == "funasr":
            raise ImportError("missing funasr")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    engine = FunASREngine(ASRConfig(model_size="funasr-paraformer-zh"))
    with pytest.raises(ModelLoadError, match="funasr"):
        engine.load_model()
    engine.close()
