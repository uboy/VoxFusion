"""CLI entry point and main command group."""

import sys

import click

from voxfusion.version import __version__


@click.group()
@click.version_option(__version__, "--version", "-v", prog_name="voxfusion")
@click.option(
    "--debug", "--verbose", "verbose", is_flag=True, help="Enable debug logs from all stages."
)
@click.option("--quiet", "-q", is_flag=True, hidden=True, help="Legacy errors-only mode.")
@click.option("--config", type=click.Path(exists=True), help="Path to config file.")
@click.pass_context
def cli(ctx: click.Context, verbose: bool, quiet: bool, config: str | None) -> None:
    """VoxFusion -- audio capture, transcription, diarization, and translation."""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    ctx.obj["quiet"] = quiet
    ctx.obj["config_path"] = config


# Register subcommands
from voxfusion.cli.capture_cmd import capture
from voxfusion.cli.config_cmd import config_group
from voxfusion.cli.devices_cmd import devices
from voxfusion.cli.models_cmd import models_group
from voxfusion.cli.record_cmd import record
from voxfusion.cli.summarize_cmd import summarize
from voxfusion.cli.transcribe_cmd import transcribe

cli.add_command(capture)
cli.add_command(record)
cli.add_command(transcribe)
cli.add_command(summarize)
cli.add_command(config_group, "config")
cli.add_command(devices)
cli.add_command(models_group, "models")


def _configure_utf8_stdio() -> None:
    """Prefer UTF-8 for CLI stdout/stderr, including redirected output."""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if not callable(reconfigure):
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError, ValueError):
            continue


def main() -> None:
    """Run CLI entry point."""
    _configure_utf8_stdio()
    cli(prog_name="voxfusion")
