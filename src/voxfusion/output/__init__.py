"""Output formatting: JSON, SRT, VTT, and plain text formatters."""

from voxfusion.exceptions import ConfigurationError
from voxfusion.output.json_formatter import JSONFormatter
from voxfusion.output.srt_formatter import SRTFormatter
from voxfusion.output.txt_formatter import TXTFormatter
from voxfusion.output.vtt_formatter import VTTFormatter

FORMATTERS: dict[str, type] = {
    "json": JSONFormatter,
    "srt": SRTFormatter,
    "vtt": VTTFormatter,
    "txt": TXTFormatter,
}


def get_formatter(format_name: str) -> JSONFormatter | SRTFormatter | VTTFormatter | TXTFormatter:
    """Return an instantiated formatter for *format_name*.

    Raises:
        ConfigurationError: If the format is not recognised.
    """
    cls = FORMATTERS.get(format_name)
    if cls is None:
        valid = ", ".join(sorted(FORMATTERS))
        raise ConfigurationError(
            f"Unknown output format {format_name!r}. Valid formats: {valid}"
        )
    return cls()


__all__ = [
    "FORMATTERS",
    "JSONFormatter",
    "SRTFormatter",
    "TXTFormatter",
    "VTTFormatter",
    "get_formatter",
]
