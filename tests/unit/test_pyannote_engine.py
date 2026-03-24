"""Unit tests for pyannote diarization compatibility helpers."""

import sys
import types

from voxfusion.config.models import DiarizationMLConfig
from voxfusion.diarization.pyannote_engine import (
    PyAnnoteDiarizer,
    _extract_annotation,
    _pipeline_auth_kwargs,
)


def test_pipeline_auth_kwargs_prefers_token() -> None:
    def from_pretrained(model: str, token: str) -> object:
        return object()

    assert _pipeline_auth_kwargs(from_pretrained, "hf_test") == {"token": "hf_test"}


def test_pipeline_auth_kwargs_falls_back_to_use_auth_token() -> None:
    def from_pretrained(model: str, use_auth_token: str) -> object:
        return object()

    assert _pipeline_auth_kwargs(from_pretrained, "hf_test") == {
        "use_auth_token": "hf_test"
    }


def test_load_pipeline_uses_modern_token_kwarg(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []

    class FakePipeline:
        @staticmethod
        def from_pretrained(model: str, token: str) -> object:
            calls.append((model, token))
            return object()

    pyannote_module = types.ModuleType("pyannote")
    pyannote_audio_module = types.ModuleType("pyannote.audio")
    pyannote_audio_module.Pipeline = FakePipeline
    pyannote_module.audio = pyannote_audio_module

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
    assert calls == [("pyannote/speaker-diarization-3.1", "hf_test")]


def test_extract_annotation_supports_new_diarize_output() -> None:
    class Annotation:
        def itertracks(self, yield_label: bool = False) -> list[object]:
            return []

    class DiarizeOutput:
        def __init__(self) -> None:
            self.speaker_diarization = Annotation()

    annotation = _extract_annotation(DiarizeOutput())

    assert hasattr(annotation, "itertracks")
