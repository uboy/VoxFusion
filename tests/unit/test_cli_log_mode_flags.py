"""Tests for CLI log-mode flags."""

from __future__ import annotations

from click.testing import CliRunner

from voxfusion.cli.main import cli


def test_cli_help_exposes_debug_mode_and_hides_quiet() -> None:
    runner = CliRunner()

    result = runner.invoke(cli, ["--help"])

    assert result.exit_code == 0
    assert "--debug" in result.output
    assert "--quiet" not in result.output
