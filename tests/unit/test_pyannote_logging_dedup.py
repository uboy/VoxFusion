"""Tests for pyannote pipeline-log suppression in chunked mode."""

from __future__ import annotations

import voxfusion.diarization.factory as diarization_factory
from voxfusion.config.models import DiarizationConfig, DiarizationMLConfig
from voxfusion.diarization.chunked import ChunkedDiarizer
from voxfusion.diarization.factory import create_diarizer


class _FakePyAnnoteDiarizer:
    def __init__(self, config=None, *, emit_pipeline_logs: bool = True) -> None:
        self.config = config
        self.emit_pipeline_logs = emit_pipeline_logs


def test_chunked_factory_suppresses_duplicate_pyannote_pipeline_logs(monkeypatch) -> None:
    monkeypatch.setattr(diarization_factory.importlib.util, "find_spec", lambda _name: object())
    monkeypatch.setattr(diarization_factory, "PyAnnoteDiarizer", _FakePyAnnoteDiarizer)

    config = DiarizationConfig(
        strategy="ml",
        ml=DiarizationMLConfig(
            hf_auth_token="hf_token",
            chunked=True,
            chunk_max_workers=4,
        ),
    )

    selection = create_diarizer(config, mode="live")

    assert isinstance(selection.engine, ChunkedDiarizer)
    inner = selection.engine._factory()
    assert inner.emit_pipeline_logs is False


def test_file_mode_factory_prefers_full_file_pyannote_even_when_chunked_enabled(
    monkeypatch,
) -> None:
    monkeypatch.setattr(diarization_factory.importlib.util, "find_spec", lambda _name: object())
    monkeypatch.setattr(diarization_factory, "PyAnnoteDiarizer", _FakePyAnnoteDiarizer)

    config = DiarizationConfig(
        strategy="ml",
        ml=DiarizationMLConfig(
            hf_auth_token="hf_token",
            chunked=True,
        ),
    )

    selection = create_diarizer(config, mode="file")

    assert isinstance(selection.engine, _FakePyAnnoteDiarizer)
    assert selection.engine.emit_pipeline_logs is True
