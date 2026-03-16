"""OutputFormatter protocol definition."""

from pathlib import Path
from typing import Protocol

from voxfusion.models.result import TranscriptionResult
from voxfusion.models.translation import TranslatedSegment


class OutputFormatter(Protocol):
    """Formats transcription results into a specific output format."""

    @property
    def format_name(self) -> str:
        """Short name, e.g. ``"json"``, ``"srt"``."""
        ...

    @property
    def file_extension(self) -> str:
        """File extension including dot, e.g. ``".json"``."""
        ...

    def format(self, result: TranscriptionResult) -> str:
        """Format the complete result as a string."""
        ...

    def format_segment(self, segment: TranslatedSegment, index: int) -> str:
        """Format a single segment (for streaming output)."""
        ...

    def write(self, result: TranscriptionResult, path: Path) -> None:
        """Write formatted result to a file."""
        ...
