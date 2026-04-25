"""Regression tests for pyannote runtime TorchScript fallback handling."""

from __future__ import annotations

import sys
import types

import numpy as np

from voxfusion.config.models import DiarizationMLConfig
from voxfusion.diarization.pyannote_engine import PyAnnoteDiarizer
from voxfusion.diarization.speaker_counter import _count_sync


def test_pyannote_load_pipeline_restores_torchscript_after_source_access_fallback(
    monkeypatch,
) -> None:
    class _FakeJit:
        def script(self, obj, *args, **kwargs):
            del args, kwargs
            raise RuntimeError(
                f"Can't get source for {obj}. TorchScript requires source access in order to carry out compilation"
            )

    fake_torch = types.ModuleType("torch")
    fake_torch.jit = _FakeJit()  # type: ignore[attr-defined]
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    original_script = fake_torch.jit.script

    class _FakePipeline:
        @staticmethod
        def from_pretrained(model: str, token: str) -> object:
            del model, token
            fake_torch.jit.script(object())
            return object()

    pyannote_module = types.ModuleType("pyannote")
    pyannote_audio_module = types.ModuleType("pyannote.audio")
    pyannote_audio_module.Pipeline = _FakePipeline
    pyannote_module.audio = pyannote_audio_module

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "pyannote", pyannote_module)
    monkeypatch.setitem(sys.modules, "pyannote.audio", pyannote_audio_module)

    diarizer = PyAnnoteDiarizer(
        DiarizationMLConfig(
            hf_auth_token="hf_test",
            device="cpu",
        )
    )

    pipeline = diarizer._load_pipeline()

    assert pipeline is diarizer._pipeline
    assert not getattr(fake_torch.jit.script, "_voxfusion_safe_wrapper", False)
    assert fake_torch.jit.script.__func__ is original_script.__func__


def test_speaker_counter_uses_torchscript_source_fallback(monkeypatch) -> None:
    class _FakeJit:
        def script(self, obj, *args, **kwargs):
            del args, kwargs
            raise RuntimeError(
                f"Can't get source for {obj}. TorchScript requires source access in order to carry out compilation"
            )

    class _FakeAnnotation:
        def itertracks(self, yield_label: bool = False):
            del yield_label
            return [(object(), None, "SPEAKER_00"), (object(), None, "SPEAKER_01")]

    fake_torch = types.ModuleType("torch")
    fake_torch.jit = _FakeJit()  # type: ignore[attr-defined]
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    original_script = fake_torch.jit.script

    def _from_numpy(array: np.ndarray) -> object:
        class _Tensor:
            def __init__(self, arr: np.ndarray) -> None:
                self._array = arr

            def float(self) -> _Tensor:
                return self

            def unsqueeze(self, _dim: int) -> _Tensor:
                return self

        return _Tensor(array)

    fake_torch.from_numpy = _from_numpy  # type: ignore[attr-defined]

    class _FakePipeline:
        @staticmethod
        def from_pretrained(model: str, token: str) -> object:
            del model, token
            fake_torch.jit.script(object())
            return lambda _input: _FakeAnnotation()

    pyannote_module = types.ModuleType("pyannote")
    pyannote_audio_module = types.ModuleType("pyannote.audio")
    pyannote_audio_module.Pipeline = _FakePipeline
    pyannote_module.audio = pyannote_audio_module

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "pyannote", pyannote_module)
    monkeypatch.setitem(sys.modules, "pyannote.audio", pyannote_audio_module)

    count = _count_sync(np.zeros(1600, dtype=np.float32), 16000, "hf_test", "pyannote/test")

    assert count == 2
    assert not getattr(fake_torch.jit.script, "_voxfusion_safe_wrapper", False)
    assert fake_torch.jit.script.__func__ is original_script.__func__
