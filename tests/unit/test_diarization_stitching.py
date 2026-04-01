"""Unit tests for global speaker stitching across chunk boundaries."""

from __future__ import annotations

from voxfusion.diarization.alignment import SpeakerTurn
from voxfusion.diarization.stitching import (
    _maximize_boundary_matches,
    stitch_chunk_speakers,
)


def test_maximize_boundary_matches_prefers_global_optimum() -> None:
    matches = _maximize_boundary_matches(
        ("LEFT_A", "LEFT_B"),
        ("RIGHT_X", "RIGHT_Y"),
        {
            ("LEFT_A", "RIGHT_X"): 5.0,
            ("LEFT_A", "RIGHT_Y"): 4.0,
            ("LEFT_B", "RIGHT_X"): 4.0,
        },
    )

    assert {(left_id, right_id) for left_id, right_id, _score in matches} == {
        ("LEFT_A", "RIGHT_Y"),
        ("LEFT_B", "RIGHT_X"),
    }


def test_stitch_chunk_speakers_merges_transitive_boundary_matches() -> None:
    mappings = stitch_chunk_speakers(
        [
            [SpeakerTurn("LOCAL_A", 0.0, 7.0)],
            [SpeakerTurn("LOCAL_B", 5.0, 12.0)],
            [SpeakerTurn("LOCAL_C", 10.0, 13.0)],
        ],
        boundaries=[
            (0.0, 7.0, 0.0, 5.0),
            (5.0, 12.0, 5.0, 10.0),
            (10.0, 13.0, 10.0, 13.0),
        ],
        chunk_overlap_s=2.0,
    )

    assert mappings == [
        {"LOCAL_A": "SPEAKER_00"},
        {"LOCAL_B": "SPEAKER_00"},
        {"LOCAL_C": "SPEAKER_00"},
    ]
