"""SRT (SubRip) subtitle output formatter."""

from pathlib import Path

from voxfusion.models.result import TranscriptionResult
from voxfusion.models.translation import TranslatedSegment


def _srt_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


class SRTFormatter:
    """Formats transcription results as SRT subtitles."""

    @property
    def format_name(self) -> str:
        return "srt"

    @property
    def file_extension(self) -> str:
        return ".srt"

    def format(self, result: TranscriptionResult) -> str:
        """Format complete result as an SRT string."""
        blocks: list[str] = []
        for i, seg in enumerate(result.segments):
            blocks.append(self.format_segment(seg, i + 1))
        return "\n".join(blocks) + "\n"

    def format_segment(self, segment: TranslatedSegment, index: int) -> str:
        """Format a single segment as an SRT entry."""
        ts = segment.diarized.segment
        start = _srt_timestamp(ts.start_time)
        end = _srt_timestamp(ts.end_time)
        text = f"[{segment.diarized.speaker_id}] {ts.text}"
        if segment.translated_text:
            text += f"\n({segment.translated_text})"
        return f"{index}\n{start} --> {end}\n{text}\n"

    def write(self, result: TranscriptionResult, path: Path) -> None:
        """Write formatted SRT to *path*."""
        path.write_text(self.format(result), encoding="utf-8")
