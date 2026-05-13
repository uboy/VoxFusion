"""Tests for CLI output cleanup: cli log mode and event printer."""

from __future__ import annotations

import logging

import structlog

from voxfusion.logging import normalize_log_mode


def test_normalize_log_mode_recognizes_cli() -> None:
    assert normalize_log_mode("cli") == "cli"
    assert normalize_log_mode("CLI") == "cli"
    assert normalize_log_mode(" Cli ") == "cli"


def test_cli_mode_drops_info_events() -> None:
    from voxfusion.logging import _cli_mode_suppress_events

    with pytest.raises(structlog.DropEvent):
        _cli_mode_suppress_events(
            logging.getLogger("test"),
            "info",
            {"level": "info", "event": "asr.loading_model"},
        )


def test_cli_mode_drops_warning_events() -> None:
    from voxfusion.logging import _cli_mode_suppress_events

    with pytest.raises(structlog.DropEvent):
        _cli_mode_suppress_events(
            logging.getLogger("test"),
            "warning",
            {"level": "warning", "event": "asr.cuda_vram_low"},
        )


def test_cli_mode_drops_key_stage_events() -> None:
    from voxfusion.logging import _cli_mode_suppress_events

    with pytest.raises(structlog.DropEvent):
        _cli_mode_suppress_events(
            logging.getLogger("test"),
            "info",
            {"level": "info", "event": "orchestrator.transcribe_file"},
        )


def test_cli_mode_drops_startup_events() -> None:
    from voxfusion.logging import _cli_mode_suppress_events

    with pytest.raises(structlog.DropEvent):
        _cli_mode_suppress_events(
            logging.getLogger("test"),
            "info",
            {"level": "info", "event": "startup.offline_mode"},
        )


def test_event_printer_shows_stage_started(capsys: object) -> None:
    from voxfusion.cli.transcribe_cmd import _event_printer
    from voxfusion.pipeline.events import EventType, PipelineEvent, PipelineStage

    event = PipelineEvent(
        event_type=EventType.STAGE_STARTED,
        stage=PipelineStage.ASR,
        message="Transcribing audio",
    )
    _event_printer(event)
    captured = capsys.readouterr()
    assert "[asr] Transcribing audio" in captured.err


def test_event_printer_shows_stage_completed(capsys: object) -> None:
    from voxfusion.cli.transcribe_cmd import _event_printer
    from voxfusion.pipeline.events import EventType, PipelineEvent, PipelineStage

    event = PipelineEvent(
        event_type=EventType.STAGE_COMPLETED,
        stage=PipelineStage.ASR,
        message="Transcribed 42 segments",
    )
    _event_printer(event)
    captured = capsys.readouterr()
    assert "[asr] Transcribed 42 segments" in captured.err


def test_event_printer_shows_warning(capsys: object) -> None:
    from voxfusion.cli.transcribe_cmd import _event_printer
    from voxfusion.pipeline.events import EventType, PipelineEvent

    event = PipelineEvent(
        event_type=EventType.WARNING,
        message="ML diarization requires a token",
    )
    _event_printer(event)
    captured = capsys.readouterr()
    assert "WARNING: ML diarization requires a token" in captured.err


def test_event_printer_shows_progress(capsys: object) -> None:
    from voxfusion.cli.transcribe_cmd import _event_printer
    from voxfusion.pipeline.events import EventType, PipelineEvent, PipelineStage

    event = PipelineEvent(
        event_type=EventType.PROGRESS,
        stage=PipelineStage.ASR,
        message="Transcribing",
        progress=0.55,
    )
    _event_printer(event)
    captured = capsys.readouterr()
    assert "55%" in captured.err


def test_event_printer_shows_pipeline_completed(capsys: object) -> None:
    from voxfusion.cli.transcribe_cmd import _event_printer
    from voxfusion.pipeline.events import EventType, PipelineEvent

    event = PipelineEvent(
        event_type=EventType.PIPELINE_COMPLETED,
        message="Done in 12.3s",
    )
    _event_printer(event)
    captured = capsys.readouterr()
    assert "Done in 12.3s" in captured.err


def test_cli_mode_installs_torch_warning_filter() -> None:
    """CLI log mode must install warning filters for torch UserWarning."""
    import re
    import warnings

    from voxfusion.logging import configure_logging

    configure_logging(log_mode="cli")
    found = False
    for action, _msg, cat, module, _lineno in warnings.filters:
        if cat is UserWarning and action == "ignore":
            mod_str = module.pattern if isinstance(module, re.Pattern) else str(module or "")
            if "torch" in mod_str:
                found = True
                break
    assert found, "CLI mode should install a UserWarning ignore filter for torch"


import pytest
