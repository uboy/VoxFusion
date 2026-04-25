"""Tests for batch CLI transcription inputs and outputs."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

from click.testing import CliRunner

import voxfusion.cli.transcribe_cmd as transcribe_cmd_module
from voxfusion.cli.transcribe_cmd import transcribe


class _BatchFakeOrchestrator:
    transcribed: ClassVar[list[Path]] = []
    written: ClassVar[list[Path]] = []

    def __init__(self, config, on_event=None) -> None:
        del config, on_event
        self._asr = SimpleNamespace(model_name="fake/asr")

    async def transcribe_file(self, file_path: Path):
        self.transcribed.append(file_path)
        return SimpleNamespace(segments=[])

    def format_result(self, result, fmt=None) -> str:
        del result, fmt
        return "ok\n"

    def write_result(self, result, output_path, fmt=None) -> None:
        del result, fmt
        self.written.append(output_path)

    def close(self) -> None:
        pass


def test_transcribe_batch_reads_input_list_and_writes_default_outputs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    audio_a = tmp_path / "a.wav"
    audio_b = tmp_path / "b.wav"
    audio_a.write_bytes(b"RIFF")
    audio_b.write_bytes(b"RIFF")
    playlist = tmp_path / "batch.txt"
    playlist.write_text("a.wav\nb.wav\n", encoding="utf-8")

    def _fake_load_config(overrides=None):
        del overrides
        return SimpleNamespace(output=SimpleNamespace(format="txt"))

    _BatchFakeOrchestrator.transcribed = []
    _BatchFakeOrchestrator.written = []

    monkeypatch.setattr(transcribe_cmd_module, "load_config", _fake_load_config)
    monkeypatch.setattr(
        transcribe_cmd_module,
        "PipelineOrchestrator",
        _BatchFakeOrchestrator,
    )
    monkeypatch.setattr(transcribe_cmd_module, "find_ffmpeg", lambda: Path("ffmpeg"))

    runner = CliRunner()
    result = runner.invoke(
        transcribe,
        ["--input-list", str(playlist)],
        obj={"verbose": False, "quiet": True},
    )

    assert result.exit_code == 0
    assert _BatchFakeOrchestrator.transcribed == [audio_a, audio_b]
    assert [path.name for path in _BatchFakeOrchestrator.written] == [
        "a.transcript.txt",
        "b.transcript.txt",
    ]


def test_transcribe_batch_rejects_single_output_path_for_multiple_inputs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    audio_a = tmp_path / "a.wav"
    audio_b = tmp_path / "b.wav"
    audio_a.write_bytes(b"RIFF")
    audio_b.write_bytes(b"RIFF")

    def _fake_load_config(overrides=None):
        del overrides
        return SimpleNamespace(output=SimpleNamespace(format="txt"))

    monkeypatch.setattr(transcribe_cmd_module, "load_config", _fake_load_config)
    monkeypatch.setattr(
        transcribe_cmd_module,
        "PipelineOrchestrator",
        _BatchFakeOrchestrator,
    )
    monkeypatch.setattr(transcribe_cmd_module, "find_ffmpeg", lambda: Path("ffmpeg"))

    runner = CliRunner()
    result = runner.invoke(
        transcribe,
        [str(audio_a), str(audio_b), "--output", str(tmp_path / "out.txt")],
        obj={"verbose": False, "quiet": True},
    )

    assert result.exit_code != 0
    assert "--output is only supported for a single input file" in result.output
