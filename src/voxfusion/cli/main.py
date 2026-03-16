"""CLI entry point and main command group."""

import click

from voxfusion.version import __version__


@click.group()
@click.version_option(version=__version__, prog_name="voxfusion")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose (debug) output.")
@click.option("--quiet", "-q", is_flag=True, help="Suppress all output except errors.")
@click.option("--config", type=click.Path(exists=True), help="Path to config file.")
@click.pass_context
def cli(ctx: click.Context, verbose: bool, quiet: bool, config: str | None) -> None:
    """VoxFusion -- audio capture, transcription, diarization, and translation."""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    ctx.obj["quiet"] = quiet
    ctx.obj["config_path"] = config


# Register subcommands
from voxfusion.cli.capture_cmd import capture  # noqa: E402
from voxfusion.cli.config_cmd import config_group  # noqa: E402
from voxfusion.cli.devices_cmd import devices  # noqa: E402
from voxfusion.cli.models_cmd import models_group  # noqa: E402
from voxfusion.cli.record_cmd import record  # noqa: E402
from voxfusion.cli.summarize_cmd import summarize  # noqa: E402
from voxfusion.cli.transcribe_cmd import transcribe  # noqa: E402

cli.add_command(capture)
cli.add_command(record)
cli.add_command(transcribe)
cli.add_command(summarize)
cli.add_command(config_group, "config")
cli.add_command(devices)
cli.add_command(models_group, "models")


def main() -> None:
    """Run CLI entry point."""
    cli(prog_name="voxfusion")
