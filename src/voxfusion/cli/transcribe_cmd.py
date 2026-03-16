"""CLI command: voxfusion transcribe -- batch file transcription."""

import asyncio
import sys
from pathlib import Path

import click

from voxfusion.config.loader import load_config
from voxfusion.logging import configure_logging, get_logger
from voxfusion.output import FORMATTERS
from voxfusion.pipeline.events import EventType, PipelineEvent
from voxfusion.pipeline.orchestrator import PipelineOrchestrator

log = get_logger(__name__)

VALID_FORMATS = sorted(FORMATTERS.keys())


def _event_printer(event: PipelineEvent) -> None:
    """Print pipeline events to stderr for user feedback."""
    try:
        match event.event_type:
            case EventType.PIPELINE_STARTED:
                click.echo(f"  Starting: {event.message}", err=True)
            case EventType.STAGE_STARTED:
                click.echo(f"  [{event.stage}] {event.message} ...", err=True)
            case EventType.STAGE_COMPLETED:
                click.echo(f"  [{event.stage}] {event.message}", err=True)
            case EventType.PIPELINE_COMPLETED:
                click.echo(f"  {event.message}", err=True)
            case EventType.PIPELINE_FAILED:
                click.echo(f"  FAILED: {event.message}", err=True)
    except (OSError, IOError):
        # Ignore console output errors
        pass


@click.command("transcribe")
@click.argument("audio_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--output", "-o",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Output file path. Defaults to stdout.",
)
@click.option(
    "--output-format", "-f",
    type=click.Choice(VALID_FORMATS),
    default=None,
    help="Output format (json, srt, vtt, txt). Defaults to config value.",
)
@click.option(
    "--language", "-l",
    default=None,
    help="Source language code (e.g. 'en'). Auto-detected if omitted.",
)
@click.option(
    "--model", "-m",
    default=None,
    help="ASR model size (tiny, base, small, medium, large-v3).",
)
@click.option(
    "--word-timestamps", "-w",
    is_flag=True,
    default=False,
    help="Include word-level timestamps in output.",
)
@click.pass_context
def transcribe(
    ctx: click.Context,
    audio_file: Path,
    output: Path | None,
    output_format: str | None,
    language: str | None,
    model: str | None,
    word_timestamps: bool,
) -> None:
    """Transcribe an audio file to text.

    Reads AUDIO_FILE, runs speech recognition, and outputs the
    transcription in the chosen format (default: JSON).
    """
    verbose = ctx.obj.get("verbose", False)
    quiet = ctx.obj.get("quiet", False)
    config_path = ctx.obj.get("config_path")

    log_level = "DEBUG" if verbose else ("ERROR" if quiet else "INFO")
    configure_logging(log_level)

    # Build config with CLI overrides
    overrides: dict = {}  # type: ignore[type-arg]
    if language:
        overrides.setdefault("asr", {})["language"] = language
    if model:
        overrides.setdefault("asr", {})["model_size"] = model
    if word_timestamps:
        overrides.setdefault("asr", {})["word_timestamps"] = True
    if output_format:
        overrides.setdefault("output", {})["format"] = output_format

    try:
        config = load_config(overrides if overrides else None)
    except Exception as exc:
        raise click.ClickException(f"Configuration error: {exc}") from exc

    fmt = output_format or config.output.format
    event_cb = _event_printer if not quiet else None

    orchestrator = PipelineOrchestrator(config, on_event=event_cb)

    if not quiet:
        try:
            click.echo(f"Transcribing: {audio_file}", err=True)
            click.echo(f"  Model: {orchestrator._asr.model_name}", err=True)
            click.echo(f"  Format: {fmt}", err=True)
        except (OSError, IOError):
            pass

    try:
        result = asyncio.run(orchestrator.transcribe_file(audio_file))
    except KeyboardInterrupt:
        try:
            click.echo("\nInterrupted.", err=True)
        except (OSError, IOError):
            print("\nInterrupted.")
        sys.exit(130)
    except Exception as exc:
        log.exception("transcribe.failed", file=str(audio_file))
        raise click.ClickException(str(exc)) from exc
    finally:
        orchestrator.close()

    # Output
    if output:
        orchestrator.write_result(result, output, fmt=fmt)
        if not quiet:
            try:
                click.echo(f"Written to: {output}", err=True)
            except (OSError, IOError):
                pass
    else:
        formatted = orchestrator.format_result(result, fmt=fmt)
        try:
            click.echo(formatted)
        except (OSError, IOError):
            print(formatted, flush=True)
