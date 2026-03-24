"""Tests for diarization engine selection helpers."""

from __future__ import annotations

import types

import pytest

from voxfusion.config.models import DiarizationConfig
from voxfusion.diarization.channel import ChannelDiarizer
from voxfusion.diarization.factory import create_diarizer
from voxfusion.diarization.pyannote_engine import PyAnnoteDiarizer
from voxfusion.exceptions import DiarizationError


def test_auto_file_mode_falls_back_to_channel_when_ml_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from voxfusion.diarization import factory as factory_module

    monkeypatch.setattr(factory_module.importlib.util, "find_spec", lambda _name: None)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)

    selection = create_diarizer(DiarizationConfig(strategy="auto"), mode="file")

    assert isinstance(selection.engine, ChannelDiarizer)
    assert selection.resolved_strategy == "channel"
    assert selection.warnings


def test_auto_file_mode_prefers_ml_when_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from voxfusion.diarization import factory as factory_module

    monkeypatch.setattr(factory_module.importlib.util, "find_spec", lambda _name: types.SimpleNamespace())
    monkeypatch.setenv("HF_TOKEN", "hf-test-token")

    selection = create_diarizer(DiarizationConfig(strategy="auto"), mode="file")

    assert isinstance(selection.engine, PyAnnoteDiarizer)
    assert selection.resolved_strategy == "ml"
    assert selection.warnings == ()


def test_explicit_ml_requires_ready_prerequisites(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from voxfusion.diarization import factory as factory_module

    monkeypatch.setattr(factory_module.importlib.util, "find_spec", lambda _name: types.SimpleNamespace())
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)

    with pytest.raises(DiarizationError, match="HuggingFace token"):
        create_diarizer(DiarizationConfig(strategy="ml"), mode="file")


def test_auto_file_mode_accepts_documented_voxfusion_token_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from voxfusion.diarization import factory as factory_module

    monkeypatch.setattr(factory_module.importlib.util, "find_spec", lambda _name: types.SimpleNamespace())
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.setenv("VOXFUSION_DIARIZATION__ML__HF_AUTH_TOKEN", "hf-test-token")

    selection = create_diarizer(DiarizationConfig(strategy="auto"), mode="file")

    assert isinstance(selection.engine, PyAnnoteDiarizer)
    assert selection.resolved_strategy == "ml"
