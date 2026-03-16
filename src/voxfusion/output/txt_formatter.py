"""Plain text transcript output formatter."""

from pathlib import Path

from voxfusion.models.result import TranscriptionResult
from voxfusion.models.translation import TranslatedSegment


def _format_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


class TXTFormatter:
    """Formats transcription results as human-readable plain text."""

    @property
    def format_name(self) -> str:
        return "txt"

    @property
    def file_extension(self) -> str:
        return ".txt"

    def format(self, result: TranscriptionResult) -> str:
        """Format complete result as plain text."""
        lines: list[str] = []
        for seg in result.segments:
            lines.append(self.format_segment(seg, 0))
        return "\n".join(lines) + "\n"

    def format_segment(self, segment: TranslatedSegment, index: int) -> str:
        """Format a single segment as a text line."""
        ts = segment.diarized.segment
        stamp = _format_timestamp(ts.start_time)
        speaker = segment.diarized.speaker_id
        line = f"[{stamp}] {speaker}: {ts.text}"
        if segment.translated_text:
            line += f"\n  -> {segment.translated_text}"
        return line

    def write(self, result: TranscriptionResult, path: Path) -> None:
        """Write formatted text to *path*."""
        path.write_text(self.format(result), encoding="utf-8")
