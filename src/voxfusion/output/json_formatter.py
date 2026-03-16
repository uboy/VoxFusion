"""JSON output formatter with full metadata."""

import json
from pathlib import Path

from voxfusion.models.result import TranscriptionResult
from voxfusion.models.translation import TranslatedSegment
from voxfusion.version import __version__


def _segment_to_dict(seg: TranslatedSegment, index: int) -> dict:  # type: ignore[type-arg]
    ts = seg.diarized.segment
    words = None
    if ts.words:
        words = [
            {
                "word": w.word,
                "start": w.start_time,
                "end": w.end_time,
                "probability": w.probability,
            }
            for w in ts.words
        ]
    return {
        "index": index,
        "start_time": ts.start_time,
        "end_time": ts.end_time,
        "speaker_id": seg.diarized.speaker_id,
        "speaker_source": seg.diarized.speaker_source,
        "original_text": ts.text,
        "original_language": ts.language,
        "translated_text": seg.translated_text,
        "target_language": seg.target_language,
        "confidence": ts.confidence,
        "words": words,
    }


class JSONFormatter:
    """Formats transcription results as JSON."""

    @property
    def format_name(self) -> str:
        return "json"

    @property
    def file_extension(self) -> str:
        return ".json"

    def format(self, result: TranscriptionResult) -> str:
        """Serialize *result* to a pretty-printed JSON string."""
        doc = {
            "voxfusion_version": __version__,
            "created_at": result.created_at,
            "source": result.source_info,
            "processing": result.processing_info,
            "segments": [
                _segment_to_dict(seg, i) for i, seg in enumerate(result.segments)
            ],
        }
        return json.dumps(doc, indent=2, ensure_ascii=False, default=str)

    def format_segment(self, segment: TranslatedSegment, index: int) -> str:
        """Serialize a single segment to a JSON string."""
        return json.dumps(
            _segment_to_dict(segment, index), ensure_ascii=False, default=str
        )

    def write(self, result: TranscriptionResult, path: Path) -> None:
        """Write formatted JSON to *path*."""
        path.write_text(self.format(result), encoding="utf-8")
