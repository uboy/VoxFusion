"""Tests for streaming ASR wrapper and overlap deduplication."""

import numpy as np
import pytest

from voxfusion.asr.dedup import OverlapDeduplicator
from voxfusion.models.transcription import TranscriptionSegment


def _seg(text: str, start: float, end: float) -> TranscriptionSegment:
    """Helper to create a TranscriptionSegment with defaults."""
    return TranscriptionSegment(
        text=text,
        language="en",
        start_time=start,
        end_time=end,
        confidence=0.9,
        words=None,
        no_speech_prob=0.0,
    )


class TestOverlapDeduplicator:
    """Tests for the OverlapDeduplicator."""

    def test_first_call_returns_all(self):
        dedup = OverlapDeduplicator(overlap_s=1.0)
        segments = [_seg("Hello world", 0.0, 2.0), _seg("How are you", 2.0, 4.0)]
        result = dedup.deduplicate(segments)
        assert len(result) == 2

    def test_exact_duplicate_removed(self):
        dedup = OverlapDeduplicator(overlap_s=1.0)
        dedup.deduplicate([_seg("Hello world", 0.0, 2.0)])

        result = dedup.deduplicate([_seg("Hello world", 0.5, 2.5)])
        assert len(result) == 0

    def test_substring_duplicate_removed(self):
        dedup = OverlapDeduplicator(overlap_s=1.0)
        dedup.deduplicate([_seg("Hello world today", 0.0, 3.0)])

        result = dedup.deduplicate([_seg("Hello world", 0.5, 2.5)])
        assert len(result) == 0

    def test_non_overlapping_kept(self):
        dedup = OverlapDeduplicator(overlap_s=1.0)
        dedup.deduplicate([_seg("Hello", 0.0, 1.0)])

        result = dedup.deduplicate([_seg("World", 5.0, 6.0)])
        assert len(result) == 1
        assert result[0].text == "World"

    def test_different_text_kept(self):
        dedup = OverlapDeduplicator(overlap_s=1.0)
        dedup.deduplicate([_seg("Hello", 0.0, 2.0)])

        result = dedup.deduplicate([_seg("Goodbye", 0.5, 2.5)])
        assert len(result) == 1

    def test_reset_clears_history(self):
        dedup = OverlapDeduplicator(overlap_s=1.0)
        dedup.deduplicate([_seg("Hello", 0.0, 2.0)])
        dedup.reset()

        result = dedup.deduplicate([_seg("Hello", 0.0, 2.0)])
        assert len(result) == 1

    def test_case_insensitive_matching(self):
        dedup = OverlapDeduplicator(overlap_s=1.0)
        dedup.deduplicate([_seg("HELLO WORLD", 0.0, 2.0)])

        result = dedup.deduplicate([_seg("hello world", 0.5, 2.5)])
        assert len(result) == 0
