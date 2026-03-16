"""Overlap deduplication for streaming transcription segments.

When using overlapping windows for streaming ASR, segments near window
boundaries may appear twice.  This module detects and removes such
duplicates based on temporal overlap and text similarity.
"""

from voxfusion.logging import get_logger
from voxfusion.models.transcription import TranscriptionSegment

log = get_logger(__name__)


class OverlapDeduplicator:
    """Removes duplicate segments caused by overlapping ASR windows.

    A new segment is considered a duplicate of a previous one if:
    1. Their time ranges overlap by more than ``overlap_s``, AND
    2. Their text content is identical or the new segment is a substring.
    """

    def __init__(self, overlap_s: float = 1.0) -> None:
        self._overlap_s = overlap_s
        self._previous_segments: list[TranscriptionSegment] = []

    def deduplicate(
        self,
        segments: list[TranscriptionSegment],
    ) -> list[TranscriptionSegment]:
        """Filter out segments that duplicate the previous window's output.

        Args:
            segments: New segments from the current ASR window.

        Returns:
            Segments with duplicates removed.
        """
        if not self._previous_segments:
            self._previous_segments = list(segments)
            return segments

        result: list[TranscriptionSegment] = []
        for seg in segments:
            if not self._is_duplicate(seg):
                result.append(seg)

        self._previous_segments = list(segments)
        return result

    def _is_duplicate(self, segment: TranscriptionSegment) -> bool:
        """Check if *segment* duplicates any previous segment."""
        for prev in self._previous_segments:
            overlap = self._temporal_overlap(prev, segment)
            if overlap < self._overlap_s * 0.5:
                continue

            # Text match: exact or substring
            prev_text = prev.text.strip().lower()
            seg_text = segment.text.strip().lower()
            if seg_text == prev_text or seg_text in prev_text:
                log.debug(
                    "dedup.removed",
                    text=segment.text[:50],
                    overlap_s=round(overlap, 2),
                )
                return True

        return False

    @staticmethod
    def _temporal_overlap(a: TranscriptionSegment, b: TranscriptionSegment) -> float:
        """Compute seconds of temporal overlap between two segments."""
        start = max(a.start_time, b.start_time)
        end = min(a.end_time, b.end_time)
        return max(0.0, end - start)

    def reset(self) -> None:
        """Clear the previous segment history."""
        self._previous_segments.clear()
