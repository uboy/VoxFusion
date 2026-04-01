"""Regression tests for Breeze direct decode without torchcodec pipeline."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from voxfusion.asr.breeze_engine import BreezeASREngine
from voxfusion.config.models import ASRConfig
from voxfusion.models.audio import AudioChunk


@pytest.mark.asyncio
async def test_breeze_engine_prefers_direct_decode_when_processor_supports_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_transformers = types.ModuleType("transformers")
    fake_torch = types.ModuleType("torch")
    fake_torch.float32 = "float32"

    class _FakeBatch(dict):
        pass

    class _FakeProcessor:
        tokenizer = object()
        feature_extractor = object()

        @classmethod
        def from_pretrained(cls, _model_ref: str, *, local_files_only: bool):
            del local_files_only
            return cls()

        def __call__(self, _audio, *, sampling_rate: int, return_tensors: str):
            del sampling_rate, return_tensors
            return _FakeBatch(input_features="features")

        def get_decoder_prompt_ids(self, **_kwargs):
            return [(1, 2)]

        def batch_decode(self, generated_ids, *, skip_special_tokens: bool):
            del generated_ids, skip_special_tokens
            return ["ni hao world"]

    class _FakeModel:
        @classmethod
        def from_pretrained(cls, _model_ref: str, *, local_files_only: bool):
            del local_files_only
            return cls()

        def generate(self, _features, **_kwargs):
            return [[1, 2, 3]]

    def _fake_pipeline(*_args, **_kwargs):
        raise AssertionError("pipeline fallback should not be used for direct decode")

    fake_transformers.AutoProcessor = _FakeProcessor
    fake_transformers.AutoModelForSpeechSeq2Seq = _FakeModel
    fake_transformers.pipeline = _fake_pipeline

    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr("voxfusion.asr.breeze_engine.activate_ffmpeg_runtime", lambda: None)

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
