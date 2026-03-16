"""Tests for ASR-diarization segment alignment."""

from voxfusion.diarization.alignment import SpeakerTurn, align_segments, _overlap
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


class TestOverlap:
    """Tests for the _overlap helper."""

    def test_full_overlap(self):
        turn = SpeakerTurn(speaker_id="A", start_time=0.0, end_time=5.0)
        assert _overlap(1.0, 3.0, turn) == 2.0

    def test_no_overlap(self):
        turn = SpeakerTurn(speaker_id="A", start_time=5.0, end_time=10.0)
        assert _overlap(0.0, 3.0, turn) == 0.0

    def test_partial_overlap(self):
        turn = SpeakerTurn(speaker_id="A", start_time=2.0, end_time=6.0)
        assert _overlap(0.0, 4.0, turn) == 2.0

    def test_turn_inside_segment(self):
        turn = SpeakerTurn(speaker_id="A", start_time=1.0, end_time=3.0)
        assert _overlap(0.0, 5.0, turn) == 2.0


class TestAlignSegments:
    """Tests for align_segments."""

    def test_single_segment_single_turn(self):
        segments = [_seg("Hello", 0.0, 2.0)]
        turns = [SpeakerTurn(speaker_id="SPEAKER_00", start_time=0.0, end_time=3.0)]
        result = align_segments(segments, turns)
        assert len(result) == 1
        assert result[0].speaker_id == "SPEAKER_00"
        assert result[0].segment.text == "Hello"

    def test_assigns_to_best_overlap(self):
        segments = [_seg("Hello", 1.0, 5.0)]
        turns = [
            SpeakerTurn(speaker_id="A", start_time=0.0, end_time=2.0),  # 1s overlap
            SpeakerTurn(speaker_id="B", start_time=2.0, end_time=6.0),  # 3s overlap
        ]
        result = align_segments(segments, turns)
        assert result[0].speaker_id == "B"

    def test_no_overlap_gives_unknown(self):
        segments = [_seg("Hello", 10.0, 12.0)]
        turns = [SpeakerTurn(speaker_id="A", start_time=0.0, end_time=5.0)]
        result = align_segments(segments, turns)
        assert result[0].speaker_id == "SPEAKER_UNKNOWN"

    def test_multiple_segments(self):
        segments = [_seg("Hi", 0.0, 1.0), _seg("Bye", 3.0, 4.0)]
        turns = [
            SpeakerTurn(speaker_id="A", start_time=0.0, end_time=2.0),
            SpeakerTurn(speaker_id="B", start_time=2.5, end_time=5.0),
        ]
        result = align_segments(segments, turns)
        assert result[0].speaker_id == "A"
        assert result[1].speaker_id == "B"

    def test_empty_segments(self):
        result = align_segments([], [])
        assert result == []

    def test_speaker_source_label(self):
        segments = [_seg("Hello", 0.0, 2.0)]
        turns = [SpeakerTurn(speaker_id="X", start_time=0.0, end_time=3.0)]
        result = align_segments(segments, turns, speaker_source="custom")
        assert result[0].speaker_source == "custom"
