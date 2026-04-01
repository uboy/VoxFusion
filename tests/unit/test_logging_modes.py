"""Tests for normal/debug log-mode filtering."""

from __future__ import annotations

import logging

import pytest
import structlog

from voxfusion.logging import _filter_normal_mode_events


def test_normal_mode_keeps_key_stage_info_event() -> None:
    event_dict = {"level": "info", "event": "batch.diarization_turns_started"}

    assert _filter_normal_mode_events(logging.getLogger("test"), "info", event_dict) == event_dict


def test_normal_mode_drops_low_level_info_event() -> None:
    with pytest.raises(structlog.DropEvent):
        _filter_normal_mode_events(
            logging.getLogger("test"),
            "info",
            {"level": "info", "event": "extractor.start"},
        )


def test_normal_mode_keeps_errors() -> None:
    event_dict = {"level": "error", "event": "gui.file_error"}

    assert _filter_normal_mode_events(logging.getLogger("test"), "error", event_dict) == event_dict
