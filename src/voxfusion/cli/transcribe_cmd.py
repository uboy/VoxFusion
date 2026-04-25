"""CLI command: voxfusion transcribe -- batch file transcription."""

import asyncio
import sys
from pathlib import Path

import click

from voxfusion.config.loader import load_config
from voxfusion.gui.helpers import find_ffmpeg
from voxfusion.logging import configure_logging, get_logger
from voxfusion.media.extractor import NEEDS_EXTRACTION_EXTENSIONS
from voxfusion.output import FORMATTERS, get_formatter
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
            case EventType.WARNING:
                click.echo(f"  WARNING: {event.message}", err=True)
    except OSError:
        # Ignore console output errors
        pass


def _read_input_list(list_path: Path) -> list[Path]:
    """Read a newline-delimited batch list, ignoring blanks and comments."""
    items: list[Path] = []
    for raw_line in list_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        path = Path(line.strip("\"'")).expanduser()
        if not path.is_absolute():
            path = (list_path.parent / path).resolve()
        items.append(path)
    return items


def _expand_path(path: Path) -> list[Path]:
    """Expand a single path: return media files inside directories, or the file itself."""
    if path.is_file():
        return [path]
    if path.is_dir():
        from voxfusion.media.extractor import NEEDS_EXTRACTION_EXTENSIONS

        wav_flac = {".wav", ".flac", ".aiff", ".aif"}
        supported = NEEDS_EXTRACTION_EXTENSIONS | wav_flac
        found = sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in supported)
        if not found:
            raise click.ClickException(f"No supported audio/video files found in directory: {path}")
        return found
    return [path]


def _collect_input_files(
    audio_files: tuple[Path, ...],
    input_list: Path | None,
) -> list[Path]:
    """Merge CLI positional inputs with an optional text-file playlist.

    Positional arguments may be files or directories; directories are expanded
    to all supported audio/video files they contain (non-recursive).
    """
    raw: list[Path] = list(audio_files)
    if input_list is not None:
        raw.extend(_read_input_list(input_list))
    if not raw:
        raise click.ClickException(
            "Provide at least one input file or directory, or use --input-list."
        )

    files: list[Path] = []
    for p in raw:
        files.extend(_expand_path(p))

    missing = [str(p) for p in files if not p.exists()]
    if missing:
        raise click.ClickException(f"Input file not found: {missing[0]}")
    return files


def _default_batch_output_path(
    audio_file: Path,
    fmt: str,
    *,
    output_dir: Path | None,
) -> Path:
    """Return the default per-file output path for batch transcription."""
    formatter = get_formatter(fmt)
    filename = f"{audio_file.stem}.transcript{formatter.file_extension}"
    if output_dir is None:
        return audio_file.with_name(filename)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / filename


def _warn_if_ffmpeg_needed(audio_file: Path) -> None:
    """Emit a one-time warning when the input likely needs FFmpeg decoding."""
    if audio_file.suffix.lower() in NEEDS_EXTRACTION_EXTENSIONS and find_ffmpeg() is None:
        click.echo(
            "WARNING: FFmpeg not found. This file may require FFmpeg for audio extraction.\n"
            "  Built VoxFusion bundles can include FFmpeg automatically.\n"
            "  For a local install, use the GUI FFmpeg installer or add FFmpeg to PATH.",
            err=True,
        )


def _transcribe_single_file(
    *,
    orchestrator: PipelineOrchestrator,
    audio_file: Path,
    output: Path | None,
    fmt: str,
    quiet: bool,
) -> None:
    """Run the existing single-file transcription flow."""
    if not quiet:
        try:
            click.echo(f"Transcribing: {audio_file}", err=True)
            click.echo(f"  Model: {orchestrator._asr.model_name}", err=True)
            click.echo(f"  Format: {fmt}", err=True)
        except OSError:
            pass

    try:
        result = asyncio.run(orchestrator.transcribe_file(audio_file))
    except KeyboardInterrupt:
        try:
            click.echo("\nInterrupted.", err=True)
        except OSError:
            print("\nInterrupted.")
        sys.exit(130)
    except Exception as exc:
        log.exception("transcribe.failed", file=str(audio_file))
        raise click.ClickException(str(exc)) from exc

    if output:
        orchestrator.write_result(result, output, fmt=fmt)
        if not quiet:
            try:
                click.echo(f"Written to: {output}", err=True)
            except OSError:
                pass
        return

    formatted = orchestrator.format_result(result, fmt=fmt)
    try:
        click.echo(formatted)
    except OSError:
        print(formatted, flush=True)


def _transcribe_batch(
    *,
    config: object,
    files: list[Path],
    fmt: str,
    output_dir: Path | None,
    quiet: bool,
    event_cb: object,
) -> None:
    """Transcribe multiple files sequentially and write one artifact per input."""
    successes = 0
    failures = 0

    if not quiet:
        try:
            click.echo(f"Batch transcribing {len(files)} files...", err=True)
        except OSError:
            pass

    for index, audio_file in enumerate(files, start=1):
        _warn_if_ffmpeg_needed(audio_file)
        try:
            orchestrator = PipelineOrchestrator(config, on_event=event_cb)
        except Exception as exc:
            raise click.ClickException(str(exc)) from exc
        try:
            if not quiet:
                try:
                    click.echo(f"[{index}/{len(files)}] {audio_file}", err=True)
                    click.echo(f"  Model: {orchestrator._asr.model_name}", err=True)
                    click.echo(f"  Format: {fmt}", err=True)
                except OSError:
                    pass

            result = asyncio.run(orchestrator.transcribe_file(audio_file))
            target = _default_batch_output_path(audio_file, fmt, output_dir=output_dir)
            orchestrator.write_result(result, target, fmt=fmt)
            successes += 1
            if not quiet:
                try:
                    click.echo(f"  Written to: {target}", err=True)
                except OSError:
                    pass
        except KeyboardInterrupt:
            try:
                click.echo("\nInterrupted.", err=True)
            except OSError:
                print("\nInterrupted.")
            sys.exit(130)
        except Exception as exc:
            failures += 1
            log.exception("transcribe.failed", file=str(audio_file))
            if not quiet:
                try:
                    click.echo(f"  FAILED: {audio_file.name}: {exc}", err=True)
                except OSError:
                    pass
        finally:
            orchestrator.close()

    if not quiet:
        try:
            click.echo(
                f"Batch finished: {successes} succeeded, {failures} failed.",
                err=True,
            )
        except OSError:
            pass

    if failures:
        raise click.ClickException(
            f"Batch finished with {failures} failure(s); {successes} succeeded."
        )


@click.command("transcribe")
@click.argument(
    "audio_files",
    nargs=-1,
    type=click.Path(exists=True, dir_okay=True, path_type=Path),
)
@click.option(
    "--input-list",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Text file with one audio/video path per line for batch transcription.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Output file path. Defaults to stdout.",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Directory for per-file outputs in batch mode. Defaults next to each source.",
)
@click.option(
    "--output-format",
    "-f",
    type=click.Choice(VALID_FORMATS),
    default=None,
    help="Output format (json, srt, vtt, txt). Defaults to config value.",
)
@click.option(
    "--language",
    "-l",
    default=None,
    help="Source language code (e.g. 'en'). Auto-detected if omitted.",
)
@click.option(
    "--model",
    "-m",
    default=None,
    help="ASR model size (tiny, base, small, medium, large-v3).",
)
@click.option(
    "--word-timestamps",
    "-w",
    is_flag=True,
    default=False,
    help="Include word-level timestamps in output.",
)
@click.option(
    "--diarization-strategy",
    type=click.Choice(["auto", "none", "channel", "ml", "hybrid"]),
    default="auto",
    show_default=True,
    help="Speaker diarization mode for file transcription.",
)
@click.option(
    "--min-speakers",
    type=click.IntRange(1),
    default=None,
    help="Optional lower hint for ML diarization speaker count.",
)
@click.option(
    "--max-speakers",
    type=click.IntRange(1),
    default=None,
    help="Optional upper hint for ML diarization speaker count.",
)
@click.pass_context
def transcribe(
    ctx: click.Context,
    audio_files: tuple[Path, ...],
    input_list: Path | None,
    output: Path | None,
    output_dir: Path | None,
    output_format: str | None,
    language: str | None,
    model: str | None,
    word_timestamps: bool,
    diarization_strategy: str,
    min_speakers: int | None,
    max_speakers: int | None,
) -> None:
    """Transcribe an audio file to text.

    Reads AUDIO_FILE, runs speech recognition, and outputs the
    transcription in the chosen format (default: JSON).
    """
    verbose = ctx.obj.get("verbose", False)
    quiet = ctx.obj.get("quiet", False)
    ctx.obj.get("config_path")

    log_level = "DEBUG" if verbose else ("ERROR" if quiet else "INFO")
    configure_logging(log_level, log_mode="debug" if verbose else "normal")

    files = _collect_input_files(audio_files, input_list)
    is_batch = len(files) > 1

    if min_speakers is not None and max_speakers is not None and min_speakers > max_speakers:
        raise click.ClickException("--min-speakers must be <= --max-speakers")

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
    overrides.setdefault("diarization", {})["strategy"] = diarization_strategy
    if min_speakers is not None or max_speakers is not None:
        overrides["diarization"].setdefault("ml", {})
        if min_speakers is not None:
            overrides["diarization"]["ml"]["min_speakers"] = min_speakers
        if max_speakers is not None:
            overrides["diarization"]["ml"]["max_speakers"] = max_speakers

    try:
        config = load_config(overrides if overrides else None)
    except Exception as exc:
        raise click.ClickException(f"Configuration error: {exc}") from exc

    if is_batch and output is not None:
        raise click.ClickException(
            "--output is only supported for a single input file. Use --output-dir for batch runs."
        )

    fmt = output_format or config.output.format
    event_cb = _event_printer if not quiet else None
    if is_batch:
        _transcribe_batch(
            config=config,
            files=files,
            fmt=fmt,
            output_dir=output_dir,
            quiet=quiet,
            event_cb=event_cb,
        )
        return

    audio_file = files[0]
    _warn_if_ffmpeg_needed(audio_file)
    try:
        orchestrator = PipelineOrchestrator(config, on_event=event_cb)
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc
    try:
        _transcribe_single_file(
            orchestrator=orchestrator,
            audio_file=audio_file,
            output=output,
            fmt=fmt,
            quiet=quiet,
        )
    finally:
        orchestrator.close()
