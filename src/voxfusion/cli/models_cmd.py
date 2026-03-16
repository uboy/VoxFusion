"""CLI command: voxfusion models -- model download and management."""

import click

from voxfusion.cli.formatting import echo_key_value, echo_success, echo_warning
from voxfusion.config.loader import load_config


@click.group("models")
def models_group() -> None:
    """Manage ASR, diarization, and translation models."""


@models_group.command("list")
def models_list() -> None:
    """List currently configured models."""
    cfg = load_config()
    click.echo("Configured models:\n")
    echo_key_value("ASR engine", cfg.asr.engine)
    echo_key_value("ASR model size", cfg.asr.model_size)
    echo_key_value("ASR device", cfg.asr.device)
    echo_key_value("ASR compute type", cfg.asr.compute_type)
    echo_key_value("Diarization", cfg.diarization.strategy)
    if cfg.diarization.strategy == "ml":
        echo_key_value("Diarization model", cfg.diarization.ml.model)
    echo_key_value("Translation enabled", str(cfg.translation.enabled))
    if cfg.translation.enabled:
        echo_key_value("Translation backend", cfg.translation.backend)
        echo_key_value("Target language", cfg.translation.target_language)


@models_group.command("download")
@click.option("--asr", "asr_model", default=None,
              help="ASR model size to download (e.g. small, medium, large-v3).")
@click.option("--diarization", "diar_model", default=None,
              help="Diarization model to download (e.g. pyannote).")
@click.option("--translation", "trans_model", default=None,
              help="Translation model (e.g. 'argos fr-en').")
def models_download(
    asr_model: str | None,
    diar_model: str | None,
    trans_model: str | None,
) -> None:
    """Download models for offline use."""
    if not any([asr_model, diar_model, trans_model]):
        raise click.ClickException(
            "Specify at least one model to download: --asr, --diarization, or --translation"
        )

    if asr_model:
        click.echo(f"Downloading ASR model: {asr_model}")
        try:
            from faster_whisper import WhisperModel

            click.echo("  Loading model (this may download it)...")
            WhisperModel(asr_model, device="cpu", compute_type="int8")
            echo_success(f"  ASR model '{asr_model}' ready.")
        except ImportError:
            raise click.ClickException("faster-whisper is not installed.")
        except Exception as exc:
            raise click.ClickException(f"Failed to download ASR model: {exc}") from exc

    if diar_model:
        echo_warning(f"  Diarization model download not yet implemented: {diar_model}")

    if trans_model:
        echo_warning(f"  Translation model download not yet implemented: {trans_model}")
