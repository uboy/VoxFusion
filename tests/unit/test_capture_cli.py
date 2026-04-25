"""Unit tests for capture CLI live-path behavior."""

from __future__ import annotations

from dataclasses import dataclass

from click.testing import CliRunner

from voxfusion.cli.capture_cmd import capture
from voxfusion.models.diarization import DiarizedSegment
from voxfusion.models.transcription import TranscriptionSegment
from voxfusion.models.translation import TranslatedSegment


def _segment(text: str, *, speaker: str = "SPEAKER_LOCAL") -> TranslatedSegment:
    return TranslatedSegment(
        diarized=DiarizedSegment(
            segment=TranscriptionSegment(
                text=text,
                language="ru",
                start_time=0.0,
                end_time=1.0,
                confidence=0.0,
                words=None,
                no_speech_prob=0.0,
            ),
            speaker_id=speaker,
            speaker_source="channel",
        ),
        translated_text=None,
        target_language=None,
    )


def test_capture_routes_live_gigaam_to_session_controller(monkeypatch) -> None:
    created: list[object] = []

    @dataclass
    class _FakeController:
        config: object
        microphone_device_id: object
        system_device_id: object
        on_status: object
        on_segments: object
        on_finalized_segments: object | None = None
        requested_source: str | None = None

        def __post_init__(self) -> None:
            created.append(self)

        async def run(self, stop_event) -> list[TranslatedSegment]:
            del stop_event
            self.on_status("Live GigaAM started. Waiting for speech...")
            self.on_segments([_segment("draft text")])
            finalized = [_segment("final text")]
            if self.on_finalized_segments is not None:
                self.on_finalized_segments(finalized)
            return finalized

    monkeypatch.setattr(
        "voxfusion.cli.capture_cmd.detect_platform", lambda: "wasapi", raising=False
    )
    monkeypatch.setattr("voxfusion.capture.factory.detect_platform", lambda: "wasapi")
    monkeypatch.setattr(
        "voxfusion.live_gigaam.session.LiveGigaAMSessionController", _FakeController
    )

    runner = CliRunner()
    result = runner.invoke(
        capture,
        ["--model", "gigaam-v3-e2e-ctc", "--no-save"],
        obj={"verbose": False, "quiet": False},
    )

    assert result.exit_code == 0
    assert created
    assert "draft text" in result.output
    assert created[0].microphone_device_id is None
    assert created[0].system_device_id is None
    assert created[0].requested_source == "microphone"
