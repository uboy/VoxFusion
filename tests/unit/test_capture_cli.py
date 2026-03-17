"""Unit tests for capture CLI edge cases."""

from __future__ import annotations

from click.testing import CliRunner

from voxfusion.cli.capture_cmd import capture


def test_capture_rejects_gigaam_for_live_mode() -> None:
    runner = CliRunner()
    result = runner.invoke(capture, ["--model", "gigaam-v3-e2e-ctc"], obj={"verbose": False, "quiet": True})
    assert result.exit_code != 0
    assert "only supported for file transcription" in result.output
