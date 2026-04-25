"""Additional CLI tests for the live GigaAM path."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from click.testing import CliRunner

from voxfusion.cli.capture_cmd import capture
from voxfusion.models.diarization import DiarizedSegment
from voxfusion.models.transcription import TranscriptionSegment
from voxfusion.models.translation import TranslatedSegment


def _segment(text: str, *, speaker: str = "SPEAKER_REMOTE") -> TranslatedSegment:
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


@dataclass
class _FakeController:
    config: object
    microphone_device_id: object
    system_device_id: object
    on_status: object
    on_segments: object
    on_finalized_segments: object | None = None
    requested_source: str | None = None

    async def run(self, stop_event) -> list[TranslatedSegment]:
        del stop_event
        self.on_segments([_segment("draft system text")])
        finalized = [_segment("final system text")]
        if self.on_finalized_segments is not None:
            self.on_finalized_segments(finalized)
        return finalized


def test_capture_live_gigaam_rejects_translate(monkeypatch) -> None:
    monkeypatch.setattr("voxfusion.capture.factory.detect_platform", lambda: "wasapi")

    runner = CliRunner()
    result = runner.invoke(
        capture,
        ["--model", "gigaam-v3-e2e-ctc", "--translate", "en", "--no-save"],
        obj={"verbose": False, "quiet": True},
    )

    assert result.exit_code != 0
    assert "does not support translation" in result.output


def test_capture_live_gigaam_saves_finalized_segments(monkeypatch, tmp_path: Path) -> None:
    created: list[_FakeController] = []

    def _factory(**kwargs):
        controller = _FakeController(**kwargs)
        created.append(controller)
        return controller

    monkeypatch.setattr("voxfusion.capture.factory.detect_platform", lambda: "wasapi")
    monkeypatch.setattr("voxfusion.live_gigaam.session.LiveGigaAMSessionController", _factory)

    output_path = tmp_path / "live.txt"
    runner = CliRunner()
    result = runner.invoke(
        capture,
        [
            "--model",
            "gigaam-v3-e2e-ctc",
            "--source",
            "system",
            "--device",
            "pa:21",
            "--save",
            str(output_path),
        ],
        obj={"verbose": False, "quiet": True},
    )

    assert result.exit_code == 0
    assert created
    assert created[0].microphone_device_id is None
    assert created[0].system_device_id == "pa:21"
    assert created[0].requested_source == "system"
    saved = output_path.read_text(encoding="utf-8")
    assert "final system text" in saved
    assert "draft system text" not in saved


def test_capture_live_gigaam_no_save_prints_finalized_segments(monkeypatch) -> None:
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
    assert "[FINALIZED]" in result.output
    assert "final system text" in result.output
