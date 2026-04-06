"""LLM-specific checks for normal/debug log filtering."""

from __future__ import annotations

import logging

from voxfusion.logging import _filter_normal_mode_events


def test_normal_mode_keeps_llm_request_info_event() -> None:
    event_dict = {"level": "info", "event": "llm.request.start"}

    assert _filter_normal_mode_events(logging.getLogger("test"), "info", event_dict) == event_dict


def test_normal_mode_keeps_gui_llm_info_event() -> None:
    event_dict = {"level": "info", "event": "gui.llm_models_loaded"}

    assert _filter_normal_mode_events(logging.getLogger("test"), "info", event_dict) == event_dict
