"""Regression tests for Parakeet runtime bootstrap behavior."""

from __future__ import annotations

import sys
import types

from voxfusion.asr.parakeet_engine import ParakeetASREngine
from voxfusion.config.models import ASRConfig


def test_parakeet_load_model_does_not_touch_torchscript(monkeypatch) -> None:
    class _FakeJit:
        def __init__(self) -> None:
            self.calls = 0

        def script(self, *_args, **_kwargs):
            self.calls += 1
            raise AssertionError("torch.jit.script should not be called while loading Parakeet")

    fake_torch = types.ModuleType("torch")
    fake_torch.jit = _FakeJit()  # type: ignore[attr-defined]

    fake_nemo = types.ModuleType("nemo")
    fake_collections = types.ModuleType("nemo.collections")
    fake_asr = types.ModuleType("nemo.collections.asr")
    fake_models = types.ModuleType("nemo.collections.asr.models")

    class _FakeASRModel:
        @classmethod
        def from_pretrained(cls, *, model_name: str):
            del model_name
            return cls()

    fake_models.ASRModel = _FakeASRModel

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "nemo", fake_nemo)
    monkeypatch.setitem(sys.modules, "nemo.collections", fake_collections)
    monkeypatch.setitem(sys.modules, "nemo.collections.asr", fake_asr)
    monkeypatch.setitem(sys.modules, "nemo.collections.asr.models", fake_models)
    monkeypatch.setattr("voxfusion.asr.parakeet_engine.activate_ffmpeg_runtime", lambda: None)

    engine = ParakeetASREngine(
        ASRConfig(model_size="parakeet-tdt-0.6b-v3", model_path="C:/models/parakeet")
    )
    try:
        engine.load_model()
        assert fake_torch.jit.calls == 0
    finally:
        engine.close()
