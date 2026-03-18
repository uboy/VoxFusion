"""CLI command: voxfusion record -- raw audio recording without ASR."""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path

import click

from voxfusion.config.loader import load_config
from voxfusion.logging import configure_logging
from voxfusion.recording import AudioRecorder, create_recording_source


@click.command("record")
@click.option(
    "--source", "-s",
    type=click.Choice(["microphone", "system", "both"]),
    default="microphone",
    help="Audio source to record.",
)
@click.option(
    "--device", "-d",
    type=str,
    default=None,
    help="Audio device id from 'voxfusion devices'.",
)
@click.option(
    "--duration", "-t",
    type=float,
    default=None,
    help="Maximum recording duration in seconds. Unlimited if omitted.",
)
@click.option(
    "--output", "-o",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Output WAV file path.",
)
@click.pass_context
def record(
    ctx: click.Context,
    source: str,
    device: str | None,
    duration: float | None,
    output: Path | None,
) -> None:
    """Record raw audio to WAV without real-time transcription."""
    verbose = ctx.obj.get("verbose", False)
    quiet = ctx.obj.get("quiet", False)

    log_level = "DEBUG" if verbose else ("ERROR" if quiet else "INFO")
    configure_logging(log_level)

    overrides = {
        "capture": {
            "sources": ["microphone", "system"] if source == "both" else [source],
        }
    }
    config = load_config(overrides)

    output_path = output or Path(
        f"recording_{source}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
    )

    recorder = AudioRecorder(
        chunk_duration_ms=config.capture.chunk_duration_ms,
        on_status=(lambda message: click.echo(message, err=True)) if not quiet else None,
    )

    async def _run() -> None:
        audio_source = create_recording_source(
            source,
            config.capture,
            device_index=device,
        )
        stats = await recorder.record(audio_source, output_path, duration_s=duration)
        if not quiet:
            click.echo(
                f"Saved audio to {stats.output_path} "
                f"({stats.duration_s:.1f}s, {stats.sample_rate} Hz, {stats.channels} ch).",
                err=True,
            )

    try:
        asyncio.run(_run())
    except KeyboardInterrupt as exc:
        recorder.request_stop()
        raise click.ClickException("Recording interrupted by user.") from exc
