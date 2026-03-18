"""Unit tests for Breeze and Parakeet backend wiring."""

from __future__ import annotations

import builtins
import sys
import types

import numpy as np
import pytest

from voxfusion.asr.factory import create_asr_engine
from voxfusion.asr.breeze_engine import BreezeASREngine
from voxfusion.asr.parakeet_engine import ParakeetASREngine
from voxfusion.config.models import ASRConfig
from voxfusion.exceptions import ModelLoadError
from voxfusion.models.audio import AudioChunk


def test_factory_routes_breeze_engine() -> None:
    engine, backend = create_asr_engine(ASRConfig(model_size="breeze-asr"))
    assert backend == "breeze"
    assert isinstance(engine, BreezeASREngine)
    engine.close()


def test_factory_routes_parakeet_engine() -> None:
    engine, backend = create_asr_engine(ASRConfig(model_size="parakeet-tdt-0.6b-v3"))
    assert backend == "parakeet"
    assert isinstance(engine, ParakeetASREngine)
    engine.close()


@pytest.mark.asyncio
async def test_breeze_engine_transcribes_with_fake_transformers(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_transformers = types.ModuleType("transformers")
    fake_torch = types.ModuleType("torch")
    fake_torch.float32 = "float32"

    class _FakeProcessor:
        tokenizer = object()
        feature_extractor = object()

        @classmethod
        def from_pretrained(cls, _model_ref: str, *, local_files_only: bool):
            del local_files_only
            return cls()

    class _FakeModel:
        @classmethod
        def from_pretrained(cls, _model_ref: str, *, local_files_only: bool):
            del local_files_only
            return cls()

    def _fake_pipeline(*args, **kwargs):
        del args, kwargs

        def _run(_audio, **_kw):
            return {"text": "ni hao world"}

        return _run

    fake_transformers.AutoProcessor = _FakeProcessor
    fake_transformers.AutoModelForSpeechSeq2Seq = _FakeModel
    fake_transformers.pipeline = _fake_pipeline

    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    engine = BreezeASREngine(ASRConfig(model_size="breeze-asr", model_path="C:/models/breeze"))
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
    assert segments[0].text == "ni hao world"
    engine.close()


@pytest.mark.asyncio
async def test_parakeet_engine_transcribes_with_fake_nemo(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_nemo = types.ModuleType("nemo")
    fake_collections = types.ModuleType("nemo.collections")
    fake_asr = types.ModuleType("nemo.collections.asr")
    fake_models = types.ModuleType("nemo.collections.asr.models")

    class _FakeASRModel:
        @classmethod
        def from_pretrained(cls, *, model_name: str):
            del model_name
            return cls()

        @classmethod
        def restore_from(cls, *, restore_path: str):
            del restore_path
            return cls()

        def transcribe(self, _paths: list[str]) -> list[str]:
            return ["hello from parakeet"]

    fake_models.ASRModel = _FakeASRModel
    monkeypatch.setitem(sys.modules, "nemo", fake_nemo)
    monkeypatch.setitem(sys.modules, "nemo.collections", fake_collections)
    monkeypatch.setitem(sys.modules, "nemo.collections.asr", fake_asr)
    monkeypatch.setitem(sys.modules, "nemo.collections.asr.models", fake_models)

    engine = ParakeetASREngine(ASRConfig(model_size="parakeet-tdt-0.6b-v3", model_path="C:/models/parakeet.nemo"))
    chunk = AudioChunk(
        samples=np.ones(16000, dtype=np.float32),
        sample_rate=16000,
        channels=1,
        timestamp_start=0.0,
        timestamp_end=1.0,
        source="file",
        dtype="float32",
    )

    segments = await engine.transcribe(chunk)
    assert len(segments) == 1
    assert segments[0].text == "hello from parakeet"
    engine.close()


def test_parakeet_engine_reports_missing_dependency_cleanly(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("nemo.collections.asr.models"):
            raise ImportError("missing nemo")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    engine = ParakeetASREngine(ASRConfig(model_size="parakeet-tdt-0.6b-v3"))
    with pytest.raises(ModelLoadError, match="nemo_toolkit\\['asr'\\]"):
        engine.load_model()
    engine.close()


def test_breeze_engine_reports_missing_dependency_cleanly(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "torch" or name.startswith("transformers"):
            raise ImportError("missing transformers or torch")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    engine = BreezeASREngine(ASRConfig(model_size="breeze-asr"))
    with pytest.raises(ModelLoadError, match="transformers and torch"):
        engine.load_model()
    engine.close()
