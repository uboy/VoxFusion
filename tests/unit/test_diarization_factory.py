"""Tests for diarization engine selection helpers."""

from __future__ import annotations

import types

import pytest

from voxfusion.config.models import DiarizationConfig
from voxfusion.diarization.channel import ChannelDiarizer
from voxfusion.diarization.chunked import ChunkedDiarizer
from voxfusion.diarization.factory import create_diarizer
from voxfusion.diarization.none import NoneDiarizer
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

    monkeypatch.setattr(
        factory_module.importlib.util,
        "find_spec",
        lambda _name: types.SimpleNamespace(),
    )
    monkeypatch.setenv("HF_TOKEN", "hf-test-token")

    selection = create_diarizer(DiarizationConfig(strategy="auto"), mode="file")

    assert isinstance(selection.engine, PyAnnoteDiarizer)
    assert selection.resolved_strategy == "ml"
    assert selection.warnings == ()


def test_explicit_ml_requires_ready_prerequisites(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from voxfusion.diarization import factory as factory_module

    monkeypatch.setattr(
        factory_module.importlib.util,
        "find_spec",
        lambda _name: types.SimpleNamespace(),
    )
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.setattr(factory_module, "_model_cached", lambda _model_id: False)

    with pytest.raises(DiarizationError, match="HuggingFace token"):
        create_diarizer(DiarizationConfig(strategy="ml"), mode="file")


def test_cached_model_makes_ml_ready_without_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from voxfusion.diarization import factory as factory_module

    monkeypatch.setattr(
        factory_module.importlib.util,
        "find_spec",
        lambda _name: types.SimpleNamespace(),
    )
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.delenv("VOXFUSION_DIARIZATION__ML__HF_AUTH_TOKEN", raising=False)
    monkeypatch.setattr(factory_module, "_interactive_hf_token", None)
    monkeypatch.setattr(factory_module, "_load_saved_token", lambda: None)
    monkeypatch.setattr(factory_module, "_model_cached", lambda _model_id: True)

    config = DiarizationConfig(strategy="ml")
    ready, reason, token_source = factory_module._ml_prerequisites(config)

    assert ready
    assert reason is None
    assert token_source == "local cache (no token)"

    selection = create_diarizer(DiarizationConfig(strategy="auto"), mode="file")

    assert isinstance(selection.engine, PyAnnoteDiarizer)
    assert selection.resolved_strategy == "ml"
    assert selection.warnings == ()


def test_auto_file_mode_no_token_no_cache_falls_back_to_channel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from voxfusion.diarization import factory as factory_module

    monkeypatch.setattr(
        factory_module.importlib.util,
        "find_spec",
        lambda _name: types.SimpleNamespace(),
    )
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.setattr(factory_module, "_interactive_hf_token", None)
    monkeypatch.setattr(factory_module, "_load_saved_token", lambda: None)
    monkeypatch.setattr(factory_module, "_model_cached", lambda _model_id: False)

    selection = create_diarizer(DiarizationConfig(strategy="auto"), mode="file")

    assert isinstance(selection.engine, ChannelDiarizer)
    assert selection.resolved_strategy == "channel"
    assert any("HuggingFace token" in warning for warning in selection.warnings)


def test_auto_file_mode_accepts_documented_voxfusion_token_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from voxfusion.diarization import factory as factory_module

    monkeypatch.setattr(
        factory_module.importlib.util,
        "find_spec",
        lambda _name: types.SimpleNamespace(),
    )
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.setenv("VOXFUSION_DIARIZATION__ML__HF_AUTH_TOKEN", "hf-test-token")

    selection = create_diarizer(DiarizationConfig(strategy="auto"), mode="file")

    assert isinstance(selection.engine, PyAnnoteDiarizer)
    assert selection.resolved_strategy == "ml"


def test_explicit_ml_non_file_mode_can_still_use_chunked_wrapper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from voxfusion.diarization import factory as factory_module

    monkeypatch.setattr(
        factory_module.importlib.util,
        "find_spec",
        lambda _name: types.SimpleNamespace(),
    )
    monkeypatch.setenv("HF_TOKEN", "hf-test-token")

    selection = create_diarizer(DiarizationConfig(strategy="ml"), mode="live")

    assert isinstance(selection.engine, ChunkedDiarizer)
    assert selection.resolved_strategy == "ml"


def test_explicit_none_strategy_returns_none_diarizer() -> None:
    selection = create_diarizer(DiarizationConfig(strategy="none"), mode="file")

    assert isinstance(selection.engine, NoneDiarizer)
    assert selection.resolved_strategy == "none"
