"""CLI command: voxfusion capture -- live audio capture and transcription."""

import asyncio
import signal
import sys
from contextlib import suppress

import click

from voxfusion.cli.formatting import echo_error, echo_warning
from voxfusion.asr_catalog import get_model_info
from voxfusion.config.loader import load_config
from voxfusion.logging import configure_logging, get_logger
from voxfusion.output import get_formatter
from voxfusion.pipeline.events import EventType, PipelineEvent

log = get_logger(__name__)


def _event_printer(event: PipelineEvent) -> None:
    """Print pipeline events to stderr."""
    try:
        match event.event_type:
            case EventType.PIPELINE_STARTED:
                click.echo("  Capture started. Press Ctrl+C to stop.", err=True)
            case EventType.STAGE_STARTED:
                click.echo(f"  [{event.stage}] {event.message}", err=True)
            case EventType.PIPELINE_COMPLETED:
                click.echo(f"  {event.message}", err=True)
    except (OSError, IOError):
        # Ignore console output errors (Windows handle issues)
        pass


@click.command("capture")
@click.option(
    "--source", "-s",
    type=click.Choice(["microphone", "system", "both"]),
    default="microphone",
    help="Audio source to capture.",
)
@click.option(
    "--output-format", "-f",
    type=click.Choice(["json", "srt", "vtt", "txt"]),
    default="txt",
    help="Output format for live transcription.",
)
@click.option(
    "--language", "-l",
    default=None,
    help="Source language code (e.g. 'en'). Auto-detected if omitted.",
)
@click.option(
    "--device", "-d",
    type=str,
    default=None,
    help="Audio device id (from 'voxfusion devices').",
)
@click.option(
    "--model", "-m",
    type=str,
    default=None,
    help="ASR model (tiny, base, small). Default: tiny for speed.",
)
@click.option(
    "--duration", "-t",
    type=float,
    default=None,
    help="Maximum capture duration in seconds. Unlimited if omitted.",
)
@click.option(
    "--translate",
    type=str,
    default=None,
    metavar="LANG",
    help="Translate to target language (e.g. 'ru', 'en', 'de'). Requires translation backend.",
)
@click.option(
    "--save",
    type=click.Path(dir_okay=False),
    default=None,
    help="Save transcription to file (auto-generates if not specified).",
)
@click.option(
    "--no-save",
    is_flag=True,
    default=False,
    help="Don't auto-save transcription (only print to console).",
)
@click.pass_context
def capture(
    ctx: click.Context,
    source: str,
    output_format: str,
    language: str | None,
    device: str | None,
    duration: float | None,
    translate: str | None,
    model: str | None,
    save: str | None,
    no_save: bool,
) -> None:
    """Capture live audio and transcribe in real-time.

    Captures audio from the microphone, system audio, or both,
    and outputs transcription segments as they are produced.
    """
    verbose = ctx.obj.get("verbose", False)
    quiet = ctx.obj.get("quiet", False)

    log_level = "DEBUG" if verbose else ("ERROR" if quiet else "INFO")
    configure_logging(log_level)

    overrides: dict = {}  # type: ignore[type-arg]
    if language:
        overrides.setdefault("asr", {})["language"] = language
    overrides.setdefault("capture", {})["sources"] = (
        ["microphone", "system"] if source == "both" else [source]
    )
    # Увеличиваем буфер и включаем lossy mode для streaming
    overrides.setdefault("capture", {})["buffer_size"] = 50  # Было 10
    overrides.setdefault("capture", {})["lossy_mode"] = True  # Сбрасываем при переполнении
    # Используем tiny модель для скорости в streaming режиме (если не указана другая)
    if model:
        overrides.setdefault("asr", {})["model_size"] = model
    else:
        overrides.setdefault("asr", {})["model_size"] = "tiny"
    if translate:
        overrides.setdefault("translation", {})["enabled"] = True
        overrides.setdefault("translation", {})["target_language"] = translate

    config = load_config(overrides if overrides else None)
    if not get_model_info(config.asr.model_size).supports_live_capture:
        raise click.ClickException(
            f"Model '{config.asr.model_size}' is only supported for file transcription."
        )

    # Check platform support
    from voxfusion.capture.factory import detect_platform

    platform = detect_platform()
    if not quiet:
        try:
            click.echo(f"Platform: {platform}", err=True)
            click.echo(f"Source: {source}", err=True)
            click.echo(f"Format: {output_format}", err=True)
        except (OSError, IOError):
            # Fallback to stdout if stderr fails
            print(f"Platform: {platform}")
            print(f"Source: {source}")
            print(f"Format: {output_format}")

    # Build the streaming pipeline
    from voxfusion.asr.faster_whisper import FasterWhisperEngine
    from voxfusion.capture.windows_factory import create_windows_capture_source
    from voxfusion.diarization.channel import ChannelDiarizer
    from voxfusion.models.translation import TranslatedSegment
    from voxfusion.pipeline.streaming import StreamingPipeline
    from voxfusion.preprocessing.normalize import Normalizer
    from voxfusion.preprocessing.pipeline import PreProcessingPipeline
    from voxfusion.preprocessing.resample import Resampler
    from voxfusion.translation.registry import get_translation_engine

    preprocessor = PreProcessingPipeline([Resampler(16_000), Normalizer()])

    # Загружаем модель ASR заранее
    if not quiet:
        try:
            print(f"Загрузка модели {config.asr.model_size}...")
        except (OSError, IOError):
            pass

    asr_engine = FasterWhisperEngine(config.asr)
    asr_engine.load_model()  # Предзагрузка модели

    if not quiet:
        try:
            print("Модель загружена!")
        except (OSError, IOError):
            pass

    diarizer = ChannelDiarizer(config.diarization)

    # Initialize translation if enabled
    translator = None
    if config.translation.enabled:
        try:
            translator = get_translation_engine(config.translation.backend, config.translation)
            if not quiet:
                try:
                    click.echo(f"Translation: {config.translation.backend} -> {config.translation.target_language}", err=True)
                except (OSError, IOError):
                    print(f"Translation: {config.translation.backend} -> {config.translation.target_language}")
        except Exception as exc:
            try:
                echo_warning(f"Translation unavailable: {exc}")
            except (OSError, IOError):
                print(f"WARNING: Translation unavailable: {exc}")

    formatter = get_formatter(output_format)

    # Auto-save setup
    from datetime import datetime
    save_file = None
    if not no_save:
        if save:
            save_file = save
        else:
            # Auto-generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            ext = "txt" if output_format == "txt" else output_format
            save_file = f"transcription_{timestamp}.{ext}"

        try:
            print(f"Сохранение в: {save_file}")
        except (OSError, IOError):
            pass

    all_segments = []  # Store all segments for saving

    def on_segments(segments: list[TranslatedSegment]) -> None:
        all_segments.extend(segments)  # Store for saving
        for seg in segments:
            try:
                line = formatter.format_segment(seg, 0)
                click.echo(line)
            except (OSError, IOError):
                # Fallback to simple print if click.echo fails
                try:
                    print(line, flush=True)
                except Exception:
                    pass

    pipeline = StreamingPipeline(
        asr_engine=asr_engine,
        diarizer=diarizer,
        preprocessor=preprocessor,
        translator=translator,
        config=config,
        on_event=_event_printer if not quiet else None,
    )

    # Create the capture source
    try:
        if platform == "wasapi":
            audio_source = create_windows_capture_source(
                source,
                config.capture,
                microphone_device_id=device if source != "system" else None,
                system_device_id=device if source == "system" else None,
            )
        else:
            echo_error(
                f"Live capture on {platform} is not yet fully supported. "
                "Try 'voxfusion transcribe <file>' for batch mode."
            )
            sys.exit(1)
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc

    import signal
    stop_event = asyncio.Event()

    def signal_handler(sig, frame):
        """Handle Ctrl+C gracefully."""
        stop_event.set()

    signal.signal(signal.SIGINT, signal_handler)

    async def _run() -> None:
        await audio_source.start()
        try:
            # Run with timeout check for stop_event
            pipeline_task = asyncio.create_task(
                pipeline.run(audio_source, on_segments=on_segments)
            )

            while not stop_event.is_set() and not pipeline_task.done():
                await asyncio.sleep(0.1)

            if stop_event.is_set():
                pipeline_task.cancel()
                try:
                    await pipeline_task
                except asyncio.CancelledError:
                    pass
        except Exception:
            pass
        finally:
            await pipeline.stop()
            await audio_source.stop()
            try:
                print("\n[ОСТАНОВЛЕНО]")
            except (OSError, IOError):
                pass

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        pass
    finally:
        with suppress(Exception):
            asr_engine.unload_model()
            asr_engine.close()
        # Save transcription to file
        if save_file and all_segments:
            try:
                from voxfusion.models.result import TranscriptionResult
                result = TranscriptionResult(
                    segments=all_segments,
                    source_info={"source": source, "live": True},
                    processing_info={"model": config.asr.model_size},
                    created_at=datetime.now().isoformat(),
                )
                fmt = get_formatter(output_format)
                with open(save_file, "w", encoding="utf-8") as f:
                    f.write(fmt.format(result))
                print(f"\n[СОХРАНЕНО]: {save_file} ({len(all_segments)} сегментов)")
            except Exception as e:
                print(f"\n[ОШИБКА СОХРАНЕНИЯ]: {e}")
