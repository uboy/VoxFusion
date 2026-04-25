"""Tests for file transcription CLI options."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from click.testing import CliRunner

import voxfusion.cli.transcribe_cmd as transcribe_cmd_module
from voxfusion.cli.transcribe_cmd import transcribe


class _FakeOrchestrator:
    def __init__(self, config, on_event=None) -> None:
        del config, on_event
        self._asr = SimpleNamespace(model_name="fake/asr")

    async def transcribe_file(self, file_path: Path):
        del file_path
        return SimpleNamespace(segments=[])

    def format_result(self, result, fmt=None) -> str:
        del result, fmt
        return "ok\n"

    def write_result(self, result, output_path, fmt=None) -> None:
        del result, output_path, fmt

    def close(self) -> None:
        pass


def test_transcribe_rejects_inverted_speaker_hints(tmp_path: Path) -> None:
    audio_file = tmp_path / "input.wav"
    audio_file.write_bytes(b"RIFF")

    runner = CliRunner()
    result = runner.invoke(
        transcribe,
        [str(audio_file), "--min-speakers", "4", "--max-speakers", "2"],
        obj={"verbose": False, "quiet": True},
    )

    assert result.exit_code != 0
    assert "--min-speakers must be <= --max-speakers" in result.output


def test_transcribe_passes_diarization_overrides(tmp_path: Path, monkeypatch) -> None:
    audio_file = tmp_path / "input.wav"
    audio_file.write_bytes(b"RIFF")
    captured: dict[str, object] = {}

    def _fake_load_config(overrides=None):
        captured["overrides"] = overrides
        return SimpleNamespace(output=SimpleNamespace(format="txt"))

    monkeypatch.setattr(transcribe_cmd_module, "load_config", _fake_load_config)
    monkeypatch.setattr(transcribe_cmd_module, "PipelineOrchestrator", _FakeOrchestrator)
    monkeypatch.setattr(transcribe_cmd_module, "find_ffmpeg", lambda: Path("ffmpeg"))

    runner = CliRunner()
    result = runner.invoke(
        transcribe,
        [
            str(audio_file),
            "--diarization-strategy",
            "hybrid",
            "--min-speakers",
            "2",
            "--max-speakers",
            "5",
        ],
        obj={"verbose": False, "quiet": True},
    )

    assert result.exit_code == 0
    overrides = captured["overrides"]
    assert overrides["diarization"]["strategy"] == "hybrid"
    assert overrides["diarization"]["ml"]["min_speakers"] == 2
    assert overrides["diarization"]["ml"]["max_speakers"] == 5


def test_transcribe_accepts_none_diarization_strategy(
    tmp_path: Path,
    monkeypatch,
) -> None:
    audio_file = tmp_path / "input.wav"
    audio_file.write_bytes(b"RIFF")
    captured: dict[str, object] = {}

    def _fake_load_config(overrides=None):
        captured["overrides"] = overrides
        return SimpleNamespace(output=SimpleNamespace(format="txt"))

    monkeypatch.setattr(transcribe_cmd_module, "load_config", _fake_load_config)
    monkeypatch.setattr(transcribe_cmd_module, "PipelineOrchestrator", _FakeOrchestrator)
    monkeypatch.setattr(transcribe_cmd_module, "find_ffmpeg", lambda: Path("ffmpeg"))

    runner = CliRunner()
    result = runner.invoke(
        transcribe,
        [str(audio_file), "--diarization-strategy", "none"],
        obj={"verbose": False, "quiet": True},
    )

    assert result.exit_code == 0
    overrides = captured["overrides"]
    assert overrides["diarization"]["strategy"] == "none"
