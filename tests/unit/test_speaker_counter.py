"""Focused tests for speaker-count estimation helpers."""

from __future__ import annotations

import sys
import types

import numpy as np

from voxfusion.diarization.speaker_counter import _count_sync


def test_count_sync_supports_legacy_use_auth_token(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []

    class _FakeAnnotation:
        def itertracks(self, yield_label: bool = False):
            del yield_label
            return [(object(), None, "SPEAKER_00")]

    class _FakePipeline:
        @staticmethod
        def from_pretrained(model: str, use_auth_token: str) -> object:
            calls.append((model, use_auth_token))
            return lambda _input: _FakeAnnotation()

    pyannote_module = types.ModuleType("pyannote")
    pyannote_audio_module = types.ModuleType("pyannote.audio")
    pyannote_audio_module.Pipeline = _FakePipeline
    pyannote_module.audio = pyannote_audio_module

    torch_module = types.ModuleType("torch")
    torch_module.cuda = types.SimpleNamespace(is_available=lambda: False)

    def _from_numpy(array: np.ndarray) -> object:
        class _Tensor:
            def __init__(self, arr: np.ndarray) -> None:
                self._array = arr

            def float(self) -> "_Tensor":
                return self

            def unsqueeze(self, _dim: int) -> "_Tensor":
                return self

        return _Tensor(array)

    torch_module.from_numpy = _from_numpy

    monkeypatch.setitem(sys.modules, "pyannote", pyannote_module)
    monkeypatch.setitem(sys.modules, "pyannote.audio", pyannote_audio_module)
    monkeypatch.setitem(sys.modules, "torch", torch_module)

    count = _count_sync(np.zeros(1600, dtype=np.float32), 16000, "hf_test", "pyannote/test")

    assert count == 1
    assert calls == [("pyannote/test", "hf_test")]
