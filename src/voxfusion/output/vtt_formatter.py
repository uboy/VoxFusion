"""WebVTT subtitle output formatter."""

from pathlib import Path

from voxfusion.models.result import TranscriptionResult
from voxfusion.models.translation import TranslatedSegment


def _vtt_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


class VTTFormatter:
    """Formats transcription results as WebVTT subtitles."""

    @property
    def format_name(self) -> str:
        return "vtt"

    @property
    def file_extension(self) -> str:
        return ".vtt"

    def format(self, result: TranscriptionResult) -> str:
        """Format complete result as a WebVTT string."""
        blocks = ["WEBVTT", ""]
        for i, seg in enumerate(result.segments):
            blocks.append(self.format_segment(seg, i + 1))
        return "\n".join(blocks) + "\n"

    def format_segment(self, segment: TranslatedSegment, index: int) -> str:
        """Format a single segment as a WebVTT cue."""
        ts = segment.diarized.segment
        start = _vtt_timestamp(ts.start_time)
        end = _vtt_timestamp(ts.end_time)
        text = f"<v {segment.diarized.speaker_id}>{ts.text}"
        if segment.translated_text:
            text += f"\n({segment.translated_text})"
        return f"{start} --> {end}\n{text}\n"

    def write(self, result: TranscriptionResult, path: Path) -> None:
        """Write formatted WebVTT to *path*."""
        path.write_text(self.format(result), encoding="utf-8")
