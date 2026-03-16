"""CLI command: voxfusion config -- configuration management."""

import click

from voxfusion.cli.formatting import echo_key_value
from voxfusion.config.loader import get_config_path, load_config, show_config


@click.group("config")
def config_group() -> None:
    """View and manage VoxFusion configuration."""


@config_group.command("show")
@click.option(
    "--format", "-f", "fmt",
    type=click.Choice(["yaml", "json"]),
    default="yaml",
    help="Output format.",
)
@click.pass_context
def config_show(ctx: click.Context, fmt: str) -> None:
    """Display the current resolved configuration."""
    cfg = load_config()
    click.echo(show_config(cfg, fmt=fmt))


@config_group.command("path")
@click.argument("level", type=click.Choice(["system", "user", "project"]), default="user")
def config_path(level: str) -> None:
    """Show the config file path for a given level."""
    path = get_config_path(level)
    exists = path.exists()
    click.echo(str(path))
    if not exists:
        click.echo("  (file does not exist yet)", err=True)


@config_group.command("validate")
@click.pass_context
def config_validate(ctx: click.Context) -> None:
    """Validate the current configuration."""
    try:
        cfg = load_config()
        click.echo("Configuration is valid.")
        echo_key_value("Log level", cfg.log_level)
        echo_key_value("ASR engine", cfg.asr.engine)
        echo_key_value("ASR model", cfg.asr.model_size)
        echo_key_value("Output format", cfg.output.format)
    except Exception as exc:
        raise click.ClickException(f"Invalid configuration: {exc}") from exc
