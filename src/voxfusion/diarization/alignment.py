"""Alignment algorithm for matching ASR segments with diarization speaker turns.

When using ML-based diarization (pyannote), speaker turns are produced
independently from ASR segments.  This module aligns them by computing
temporal overlap and assigning each ASR segment to the speaker with
the greatest overlap.
"""

from dataclasses import dataclass

from voxfusion.logging import get_logger
from voxfusion.models.diarization import DiarizedSegment
from voxfusion.models.transcription import TranscriptionSegment, WordTiming

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


def _speaker_for_time(timestamp: float, turns: list[SpeakerTurn]) -> str | None:
    for turn in turns:
        if turn.start_time <= timestamp <= turn.end_time:
            return turn.speaker_id
    return None


def _best_speaker_for_segment(
    segment: TranscriptionSegment,
    turns: list[SpeakerTurn],
) -> str:
    best_speaker = "SPEAKER_UNKNOWN"
    best_overlap = 0.0
    for turn in turns:
        ov = _overlap(segment.start_time, segment.end_time, turn)
        if ov > best_overlap:
            best_overlap = ov
            best_speaker = turn.speaker_id
    return best_speaker


def _split_segment_by_words(
    segment: TranscriptionSegment,
    turns: list[SpeakerTurn],
    *,
    speaker_source: str,
) -> list[DiarizedSegment]:
    if not segment.words:
        return []

    dominant = _best_speaker_for_segment(segment, turns)
    groups: list[tuple[str, list[WordTiming]]] = []

    for word in segment.words:
        midpoint = word.start_time + ((word.end_time - word.start_time) / 2.0)
        speaker = _speaker_for_time(midpoint, turns) or dominant
        if groups and groups[-1][0] == speaker:
            groups[-1][1].append(word)
        else:
            groups.append((speaker, [word]))

    split_segments: list[DiarizedSegment] = []
    for speaker, words in groups:
        text = "".join(word.word for word in words).strip()
        if not text:
            continue
        split_segments.append(
            DiarizedSegment(
                segment=TranscriptionSegment(
                    text=text,
                    language=segment.language,
                    start_time=words[0].start_time,
                    end_time=words[-1].end_time,
                    confidence=segment.confidence,
                    words=words,
                    no_speech_prob=segment.no_speech_prob,
                ),
                speaker_id=speaker,
                speaker_source=speaker_source,
            )
        )
    return split_segments


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
        overlapping_turns = [
            turn for turn in turns
            if _overlap(seg.start_time, seg.end_time, turn) > 0.0
        ]
        if len(overlapping_turns) > 1 and seg.words:
            split = _split_segment_by_words(
                seg,
                overlapping_turns,
                speaker_source=speaker_source,
            )
            if split:
                result.extend(split)
                continue

        best_speaker = _best_speaker_for_segment(seg, turns)

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
