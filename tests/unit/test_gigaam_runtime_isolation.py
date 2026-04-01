"""Regression tests for GigaAM runtime shims not leaking globally."""

from __future__ import annotations

import sys
import types

from voxfusion.asr.gigaam_engine import GigaAMCTCEngine
from voxfusion.config.models import ASRConfig


class _FakeLoadedModel:
    def transcribe(self, wav_path: str) -> str:
        del wav_path
        return "privet mir"


def test_gigaam_load_model_restores_torchscript_after_model_load(monkeypatch) -> None:
    class _FakeJit:
        def script(self, obj, *args, **kwargs):
            del args, kwargs
            raise RuntimeError(
                f"Can't get source for {obj}. TorchScript requires source access in order to carry out compilation"
            )

    fake_torch = types.ModuleType("torch")
    fake_torch.jit = _FakeJit()  # type: ignore[attr-defined]
    original_script = fake_torch.jit.script

    class _FakeAutoModel:
        @classmethod
        def from_pretrained(cls, _model_ref: str, **_kwargs: object) -> _FakeLoadedModel:
            fake_torch.jit.script(object())
            return _FakeLoadedModel()

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoModel = _FakeAutoModel  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setattr("voxfusion.asr.gigaam_engine.activate_ffmpeg_runtime", lambda: None)

    engine = GigaAMCTCEngine(
        ASRConfig(model_size="gigaam-v3-e2e-ctc", model_path="C:/models/gigaam")
    )
    try:
        engine.load_model()
        assert not getattr(fake_torch.jit.script, "_voxfusion_safe_wrapper", False)
        assert fake_torch.jit.script.__func__ is original_script.__func__
    finally:
        engine.close()
