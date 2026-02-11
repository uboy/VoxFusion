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
