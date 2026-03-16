"""Alignment algorithm for matching ASR segments with diarization speaker turns.

When using ML-based diarization (pyannote), speaker turns are produced
independently from ASR segments.  This module aligns them by computing
temporal overlap and assigning each ASR segment to the speaker with
the greatest overlap.
"""

from dataclasses import dataclass

from voxfusion.logging import get_logger
from voxfusion.models.diarization import DiarizedSegment
from voxfusion.models.transcription import TranscriptionSegment

log = get_logger(__name__)


@dataclass(frozen=True)
class SpeakerTurn:
    """A contiguous speaker turn from the diarization engine.

    Attributes:
        speaker_id: Speaker label (e.g. ``"SPEAKER_00"``).
        start_time: Turn start in seconds.
        end_time: Turn end in seconds.
    """

    speaker_id: str
    start_time: float
    end_time: float


def _overlap(seg_start: float, seg_end: float, turn: SpeakerTurn) -> float:
    """Compute seconds of overlap between a segment and a speaker turn."""
    start = max(seg_start, turn.start_time)
    end = min(seg_end, turn.end_time)
    return max(0.0, end - start)


def align_segments(
    segments: list[TranscriptionSegment],
    turns: list[SpeakerTurn],
    speaker_source: str = "ml",
) -> list[DiarizedSegment]:
    """Assign speaker labels to ASR segments based on temporal overlap.

    Each segment is assigned to the speaker whose turn has the
    greatest temporal overlap.  If no turn overlaps, the segment
    gets ``"SPEAKER_UNKNOWN"``.

    Args:
        segments: ASR transcription segments.
        turns: Speaker turns from the diarization engine.
        speaker_source: Label for how the speaker was identified.

    Returns:
        List of ``DiarizedSegment`` with speaker assignments.
    """
    result: list[DiarizedSegment] = []

    for seg in segments:
        best_speaker = "SPEAKER_UNKNOWN"
        best_overlap = 0.0

        for turn in turns:
            ov = _overlap(seg.start_time, seg.end_time, turn)
            if ov > best_overlap:
                best_overlap = ov
                best_speaker = turn.speaker_id

        result.append(DiarizedSegment(
            segment=seg,
            speaker_id=best_speaker,
            speaker_source=speaker_source,
        ))

    log.debug(
        "alignment.completed",
        segments=len(segments),
        turns=len(turns),
        unknown=sum(1 for d in result if d.speaker_id == "SPEAKER_UNKNOWN"),
    )
    return result
