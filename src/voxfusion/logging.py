"""Logging configuration using structlog.

Provides structured logging with both human-readable console output
and machine-readable JSON output modes.
"""

import json
import logging
import os
import sys
import tempfile
from pathlib import Path

import structlog

_SUPPRESSED_LOG_MESSAGE_FRAGMENTS = (
    "deprecate positional args:",
    "generated new fontManager",
    "NumExpr defaulting to ",
    "NOTE: Redirects are currently not supported in Windows or MacOs.",
    "Megatron num_microbatches_calculator not found, using Apex version.",
    "OneLogger: Setting error_handling_strategy to DISABLE_QUIETLY_AND_REPORT_METRIC_ERROR",
    "No exporters were provided. This means that no telemetry data will be collected.",
    "Final configuration contains 0 exporter(s)",
    "Initializing DefaultRecorder with no exporters, exporting is disabled",
    "Could not save font_manager cache",
    "Couldn't find ffmpeg or avconv - defaulting to ffmpeg, but may not work",
)

_NORMAL_MODE_KEY_EVENTS = frozenset(
    {
        "diarization.selection",
        "batch.diarization_path_selected",
        "batch.diarization_turns_started",
        "batch.window_transcription_start",
        "chunked_diarizer.start",
        "orchestrator.transcribe_file",
        "orchestrator.result_written",
        "streaming.completed",
    }
)

_NORMAL_MODE_KEY_PREFIXES = (
    "asr.",
    "llm.",
    "gui.file_",
    "gui.live_",
    "gui.llm_",
    "gui.speaker_detect_",
    "streaming.",
    "wasapi.",
    "pyaudio_loopback.",
    "gigaam.",
    "live_gigaam.",
    "coreaudio.",
    "pulseaudio.",
    "mixer.",
    "vad_chunker.",
    "startup.",  # Offline/online mode status messages
)


def _should_suppress_log_message(message: str) -> bool:
    """Return True when a third-party log message is known-safe noise."""
    if any(fragment in message for fragment in _SUPPRESSED_LOG_MESSAGE_FRAGMENTS):
        return True
    if "thrown while requesting " in message and "huggingface.co/" in message:
        return True
    if "Retrying in " in message and "[Retry " in message:
        return True
    if "Found only " in message and "min_cluster_size" in message:
        return True
    return False


def normalize_log_mode(mode: str | None) -> str:
    """Return the normalized runtime log mode."""
    return "debug" if str(mode or "").strip().lower() == "debug" else "normal"


def _is_key_stage_event(event_name: str) -> bool:
    if event_name in _NORMAL_MODE_KEY_EVENTS:
        return True
    return any(event_name.startswith(prefix) for prefix in _NORMAL_MODE_KEY_PREFIXES)


def _filter_normal_mode_events(
    _logger: logging.Logger,
    _method_name: str,
    event_dict: structlog.types.EventDict,
) -> structlog.types.EventDict:
    """Keep only key-stage info logs in normal mode; always keep warnings/errors."""
    level = str(event_dict.get("level", "info")).lower()
    if level in {"warning", "error", "critical", "exception"}:
        return event_dict
    event_name = str(event_dict.get("event", "")).strip()
    if _is_key_stage_event(event_name):
        return event_dict
    raise structlog.DropEvent


class _NoisyDependencyFilter(logging.Filter):
    """Drop known-safe third-party noise while keeping real warnings/errors."""

    def filter(self, record: logging.LogRecord) -> bool:
        return not _should_suppress_log_message(record.getMessage())


def _short_timestamp(raw: object) -> str:
    text = str(raw or "")
    if "T" not in text:
        return text
    time_part = text.split("T", 1)[1]
    time_part = time_part.split(".", 1)[0]
    return time_part.rstrip("Z")


def _format_compact_log_value(value: object) -> str:
    if isinstance(value, str):
        if any(ch.isspace() for ch in value) or "\\" in value or "/" in value:
            return json.dumps(value, ensure_ascii=False)
        return value
    if isinstance(value, (list, tuple, dict, set)):
        return json.dumps(value, ensure_ascii=False, default=str)
    return str(value)


def _compact_console_renderer(
    _logger: logging.Logger,
    _method_name: str,
    event_dict: structlog.types.EventDict,
) -> str:
    """Render a shorter single-line log format for the GUI log pane."""
    timestamp = _short_timestamp(event_dict.pop("timestamp", ""))
    level = str(event_dict.pop("level", "info")).upper()
    event = str(event_dict.pop("event", "")).strip()
    event_dict.pop("logger", None)

    parts = [part for part in (timestamp, level, event) if part]
    rendered = " | ".join(parts)

    details = [
        f"{key}={_format_compact_log_value(value)}"
        for key, value in event_dict.items()
        if value not in (None, "", [], (), {})
    ]
    if details:
        rendered = f"{rendered} | {' '.join(details)}"
    return rendered


def _ensure_runtime_environment_defaults() -> None:
    """Set conservative runtime defaults that reduce third-party log noise."""
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("PYANNOTE_METRICS_ENABLED", "false")
    # Force offline mode after initial model download.
    # Set VOXFUSION_ONLINE=1 to temporarily re-enable network access (e.g., for model updates).
    if os.environ.get("VOXFUSION_ONLINE", "0") != "1":
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    thread_count = min(os.cpu_count() or 4, 16)
    os.environ.setdefault("NUMEXPR_MAX_THREADS", str(thread_count))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(thread_count))

    configured = os.environ.get("MPLCONFIGDIR", "").strip()
    if configured:
        target = Path(configured).expanduser()
    else:
        target = Path(tempfile.gettempdir()) / "voxfusion-mplconfig"
    try:
        target.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(target)
    except OSError:
        pass


def configure_logging(
    log_level: str = "INFO",
    json_format: bool = False,
    use_colors: bool | None = None,
    renderer_style: str = "console",
    log_mode: str = "normal",
) -> None:
    """Configure structlog and stdlib logging.

    Args:
        log_level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        json_format: If True, output JSON lines. Otherwise human-readable.
    """
    _ensure_runtime_environment_defaults()
    level = getattr(logging, log_level.upper(), logging.INFO)
    effective_log_mode = normalize_log_mode(log_mode)

    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if json_format:
        renderer: structlog.types.Processor = structlog.processors.JSONRenderer()
    elif renderer_style == "compact":
        renderer = _compact_console_renderer
    else:
        renderer_kwargs: dict[str, bool] = {}
        if use_colors is not None:
            renderer_kwargs["colors"] = use_colors
        renderer = structlog.dev.ConsoleRenderer(**renderer_kwargs)

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.filter_by_level,
            *([_filter_normal_mode_events] if effective_log_mode == "normal" else []),
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)
    handler.addFilter(_NoisyDependencyFilter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

    # Quiet noisy third-party loggers
    for name in (
        "faster_whisper",
        "ctranslate2",
        "urllib3",
        "httpx",
        "graphviz",
        "numexpr.utils",
        "huggingface_hub",
        "matplotlib",
        "matplotlib.font_manager",
        "opentelemetry",
        "pyannote",
    ):
        logging.getLogger(name).setLevel(max(level, logging.WARNING))
    for name in (
        "torch.distributed.elastic.multiprocessing.redirects",
        "nv_one_logger",
        "nv_one_logger.api.config",
        "nv_one_logger.training_telemetry.api.training_telemetry_provider",
        "nemo",
        "nemo_logger",
        "opentelemetry.sdk.metrics._internal.export",
        "pyannote.audio.telemetry",
        "huggingface_hub.utils._http",
    ):
        logging.getLogger(name).setLevel(max(level, logging.ERROR))

    # Log offline mode status for user awareness
    log = get_logger(__name__)
    if os.environ.get("VOXFUSION_ONLINE", "0") == "1":
        log.info(
            "startup.online_mode_enabled",
            note="Network access enabled by VOXFUSION_ONLINE=1. Model downloads and updates may use network.",
        )
    else:
        log.info(
            "startup.offline_mode",
            hf_hub_offline=os.environ.get("HF_HUB_OFFLINE"),
            transformers_offline=os.environ.get("TRANSFORMERS_OFFLINE"),
            note="Models will be loaded from local cache only. Set VOXFUSION_ONLINE=1 to enable network access.",
        )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a bound structlog logger for the given module name.

    Args:
        name: Typically ``__name__`` of the calling module.
    """
    return structlog.get_logger(name)
