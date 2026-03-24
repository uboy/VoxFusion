"""Logging configuration using structlog.

Provides structured logging with both human-readable console output
and machine-readable JSON output modes.
"""

import logging
import sys

import structlog

_SUPPRESSED_LOG_MESSAGE_FRAGMENTS = (
    "deprecate positional args:",
    "NumExpr defaulting to ",
    "NOTE: Redirects are currently not supported in Windows or MacOs.",
    "Megatron num_microbatches_calculator not found, using Apex version.",
    "OneLogger: Setting error_handling_strategy to DISABLE_QUIETLY_AND_REPORT_METRIC_ERROR",
    "No exporters were provided. This means that no telemetry data will be collected.",
    "Final configuration contains 0 exporter(s)",
    "Initializing DefaultRecorder with no exporters, exporting is disabled",
)


def _should_suppress_log_message(message: str) -> bool:
    """Return True when a third-party log message is known-safe noise."""
    return any(fragment in message for fragment in _SUPPRESSED_LOG_MESSAGE_FRAGMENTS)


class _NoisyDependencyFilter(logging.Filter):
    """Drop known-safe third-party noise while keeping real warnings/errors."""

    def filter(self, record: logging.LogRecord) -> bool:
        return not _should_suppress_log_message(record.getMessage())


def configure_logging(
    log_level: str = "INFO",
    json_format: bool = False,
    use_colors: bool | None = None,
) -> None:
    """Configure structlog and stdlib logging.

    Args:
        log_level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        json_format: If True, output JSON lines. Otherwise human-readable.
    """
    level = getattr(logging, log_level.upper(), logging.INFO)

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
    else:
        renderer_kwargs: dict[str, bool] = {}
        if use_colors is not None:
            renderer_kwargs["colors"] = use_colors
        renderer = structlog.dev.ConsoleRenderer(**renderer_kwargs)

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.filter_by_level,
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
    ):
        logging.getLogger(name).setLevel(max(level, logging.WARNING))
    for name in (
        "torch.distributed.elastic.multiprocessing.redirects",
        "nv_one_logger",
        "nv_one_logger.api.config",
        "nv_one_logger.training_telemetry.api.training_telemetry_provider",
        "nemo",
        "nemo_logger",
    ):
        logging.getLogger(name).setLevel(max(level, logging.ERROR))


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a bound structlog logger for the given module name.

    Args:
        name: Typically ``__name__`` of the calling module.
    """
    return structlog.get_logger(name)
