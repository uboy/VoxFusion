"""Diarization data models: DiarizedSegment."""

from dataclasses import dataclass

from voxfusion.models.transcription import TranscriptionSegment


@dataclass(frozen=True)
class DiarizedSegment:
    """A transcription segment annotated with speaker identity.

    Attributes:
        segment: The underlying transcription segment.
        speaker_id: Speaker label (e.g. ``"SPEAKER_00"``).
        speaker_source: How the speaker was identified —
            ``"channel"``, ``"ml"``, or ``"manual"``.
    """

    segment: TranscriptionSegment
    speaker_id: str
    speaker_source: str
