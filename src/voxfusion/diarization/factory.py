"""Helpers for choosing the effective diarization engine."""

from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass

from voxfusion.config.models import DiarizationConfig
from voxfusion.diarization.base import DiarizationEngine
from voxfusion.diarization.channel import ChannelDiarizer
from voxfusion.diarization.chunked import ChunkedDiarizer
from voxfusion.diarization.hybrid import HybridDiarizer
from voxfusion.diarization.none import NoneDiarizer
from voxfusion.diarization.pyannote_engine import PyAnnoteDiarizer
from voxfusion.exceptions import DiarizationError
from voxfusion.logging import get_logger

log = get_logger(__name__)

_interactive_hf_token: str | None = None


def _reset_interactive_token() -> None:
    """Clear the session-cached interactive token (for testing)."""
    global _interactive_hf_token  # noqa: PLW0603
    _interactive_hf_token = None


def _prompt_hf_token_interactively() -> str | None:
    """Prompt the user for an HF token via stdin. Returns None on failure."""
    try:
        import sys

        if not sys.stdin.isatty():
            return None
        print(
            "\nML diarization requires a Hugging Face token.\n"
            "Get one at: https://huggingface.co/settings/tokens\n"
            "You also need to accept the model licenses:\n"
            "  https://huggingface.co/pyannote/speaker-diarization-3.1\n"
            "  https://huggingface.co/pyannote/segmentation-3.0\n"
            "\nPaste your token and press Enter (or press Enter to skip): ",
            file=sys.stderr,
            end="",
            flush=True,
        )
        token = sys.stdin.readline().strip()
        return token if token else None
    except (EOFError, KeyboardInterrupt, OSError):
        return None


@dataclass(frozen=True)
class DiarizerSelection:
    """Resolved diarization engine plus user-visible metadata."""

    engine: DiarizationEngine
    requested_strategy: str
    resolved_strategy: str
    warnings: tuple[str, ...] = ()


def _resolve_hf_token(
    config: DiarizationConfig,
    *,
    interactive: bool = False,
) -> tuple[str | None, str | None]:
    global _interactive_hf_token  # noqa: PLW0603

    if config.ml.hf_auth_token:
        return config.ml.hf_auth_token, "config"

    env_candidates = (
        (
            "VOXFUSION_DIARIZATION__ML__HF_AUTH_TOKEN",
            "env:VOXFUSION_DIARIZATION__ML__HF_AUTH_TOKEN",
        ),
        ("HF_TOKEN", "env:HF_TOKEN"),
        ("HUGGING_FACE_HUB_TOKEN", "env:HUGGING_FACE_HUB_TOKEN"),
    )
    for env_name, source in env_candidates:
        token = os.environ.get(env_name)
        if token:
            return token, source

    if _interactive_hf_token:
        return _interactive_hf_token, "interactive (cached)"

    if interactive:
        token = _prompt_hf_token_interactively()
        if token:
            _interactive_hf_token = token
            return token, "interactive"

    return None, None


def _ml_prerequisites(
    config: DiarizationConfig,
    *,
    interactive: bool = False,
) -> tuple[bool, str | None, str | None]:
    try:
        spec = importlib.util.find_spec("pyannote.audio")
    except (ImportError, ModuleNotFoundError):
        spec = None
    if spec is None:
        return False, "ML diarization requires the optional 'pyannote.audio' package.", None
    token, token_source = _resolve_hf_token(config, interactive=interactive)
    if not token:
        return False, "ML diarization requires a HuggingFace token for pyannote models.", None
    return True, None, token_source


def _log_selection(
    *,
    mode: str,
    requested: str,
    selection: DiarizerSelection,
    ml_ready: bool,
    ml_reason: str | None,
    token_source: str | None,
    config: DiarizationConfig,
) -> DiarizerSelection:
    log.info(
        "diarization.selection",
        mode=mode,
        requested_strategy=requested,
        resolved_strategy=selection.resolved_strategy,
        ml_ready=ml_ready,
        fallback_reason=ml_reason,
        token_present=token_source is not None,
        token_source=token_source,
        min_speakers=config.ml.min_speakers,
        max_speakers=config.ml.max_speakers,
        warnings=list(selection.warnings),
    )
    return selection


def _wrap_with_chunked(
    config: DiarizationConfig,
    inner_config: object,
    *,
    mode: str,
) -> object:
    """Return the ML diarizer best suited for the current workflow mode."""
    ml_cfg = config.ml
    # Offline file transcription should prefer full-file diarization so the
    # underlying pipeline can preserve one global speaker space.
    if mode == "file" or not ml_cfg.chunked:
        return PyAnnoteDiarizer(inner_config)  # type: ignore[arg-type]

    def _factory() -> PyAnnoteDiarizer:
        return PyAnnoteDiarizer(  # type: ignore[arg-type]
            inner_config,
            emit_pipeline_logs=False,
        )

    return ChunkedDiarizer(
        _factory,
        chunk_duration_s=ml_cfg.chunk_duration_s,
        chunk_overlap_s=ml_cfg.chunk_overlap_s,
        max_workers=ml_cfg.chunk_max_workers,
        device=ml_cfg.device,
    )


def create_diarizer(
    config: DiarizationConfig,
    *,
    mode: str,
    interactive: bool = False,
) -> DiarizerSelection:
    """Resolve the effective diarizer for the given workflow mode."""
    requested = (config.strategy or "channel").strip().lower()
    if requested not in {"auto", "channel", "ml", "hybrid", "none"}:
        raise DiarizationError(f"Unknown diarization strategy: {config.strategy!r}")

    if requested == "none":
        return _log_selection(
            mode=mode,
            requested=requested,
            selection=DiarizerSelection(NoneDiarizer(), requested, "none"),
            ml_ready=False,
            ml_reason=None,
            token_source=None,
            config=config,
        )

    # "channel" never needs ML — skip prerequisites entirely to avoid
    # spuriously prompting for an HF token the user doesn't need.
    if requested == "channel":
        ml_ready, ml_reason, token_source = _ml_prerequisites(config, interactive=False)
        return _log_selection(
            mode=mode,
            requested=requested,
            selection=DiarizerSelection(ChannelDiarizer(config), requested, "channel"),
            ml_ready=ml_ready,
            ml_reason=ml_reason,
            token_source=token_source,
            config=config,
        )

    # Strategies that actually use ML: prompt interactively when allowed.
    ml_ready, ml_reason, token_source = _ml_prerequisites(config, interactive=interactive)

    if requested == "ml":
        if not ml_ready:
            log.warning(
                "diarization.selection_failed",
                mode=mode,
                requested_strategy=requested,
                reason=ml_reason,
                token_present=token_source is not None,
                token_source=token_source,
            )
            raise DiarizationError(ml_reason or "ML diarization is not available.")
        return _log_selection(
            mode=mode,
            requested=requested,
            selection=DiarizerSelection(
                _wrap_with_chunked(config, config.ml, mode=mode), requested, "ml"
            ),
            ml_ready=ml_ready,
            ml_reason=ml_reason,
            token_source=token_source,
            config=config,
        )

    if requested == "hybrid":
        if mode == "file" and not ml_ready:
            return _log_selection(
                mode=mode,
                requested=requested,
                selection=DiarizerSelection(
                    ChannelDiarizer(config),
                    requested,
                    "channel",
                    warnings=(f"{ml_reason} Falling back to channel diarization.",),
                ),
                ml_ready=ml_ready,
                ml_reason=ml_reason,
                token_source=token_source,
                config=config,
            )
        return _log_selection(
            mode=mode,
            requested=requested,
            selection=DiarizerSelection(HybridDiarizer(config), requested, "hybrid"),
            ml_ready=ml_ready,
            ml_reason=ml_reason,
            token_source=token_source,
            config=config,
        )

    # "auto" — use ML when available, otherwise channel.
    if mode == "file" and ml_ready:
        return _log_selection(
            mode=mode,
            requested=requested,
            selection=DiarizerSelection(
                _wrap_with_chunked(config, config.ml, mode=mode), requested, "ml"
            ),
            ml_ready=ml_ready,
            ml_reason=ml_reason,
            token_source=token_source,
            config=config,
        )
    if mode == "file" and ml_reason:
        return _log_selection(
            mode=mode,
            requested=requested,
            selection=DiarizerSelection(
                ChannelDiarizer(config),
                requested,
                "channel",
                warnings=(f"{ml_reason} Speaker separation is disabled; using channel fallback.",),
            ),
            ml_ready=ml_ready,
            ml_reason=ml_reason,
            token_source=token_source,
            config=config,
        )
    return _log_selection(
        mode=mode,
        requested=requested,
        selection=DiarizerSelection(ChannelDiarizer(config), requested, "channel"),
        ml_ready=ml_ready,
        ml_reason=ml_reason,
        token_source=token_source,
        config=config,
    )
