"""Unit tests for the API diarization strategy env override."""

from __future__ import annotations

import pytest
from server.app import DIARIZATION_STRATEGIES, _resolve_diarization_strategy


def test_default_strategy_is_channel() -> None:
    assert _resolve_diarization_strategy(None) == "channel"
    assert _resolve_diarization_strategy("") == "channel"


def test_strategy_env_is_normalized() -> None:
    assert _resolve_diarization_strategy(" ML ") == "ml"
    assert _resolve_diarization_strategy("Auto") == "auto"
    assert set(DIARIZATION_STRATEGIES) == {"auto", "channel", "ml", "hybrid", "none"}


def test_invalid_strategy_fails_at_startup() -> None:
    with pytest.raises(RuntimeError, match="VOXFUSION_API_DIARIZATION_STRATEGY"):
        _resolve_diarization_strategy("yolo")
