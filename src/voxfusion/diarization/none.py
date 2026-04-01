"""Null diarization engine — assigns all segments to SPEAKER_00.

Use this strategy when speaker separation is not needed (e.g. single-speaker
recordings) to skip neural diarization entirely and maximise pipeline speed.
"""

from collections.abc import AsyncIterator

from voxfusion.logging import get_logger
from voxfusion.models.audio import AudioChunk
from voxfusion.models.diarization import DiarizedSegment
from voxfusion.models.transcription import TranscriptionSegment

log = get_logger(__name__)


class NoneDiarizer:
    """Assigns all segments to SPEAKER_00 without any analysis.

    This diarizer performs no neural inference — it simply labels every
    segment with ``SPEAKER_00`` and returns immediately.  Use it when
    speaker identification is not required to avoid the cost of running
    pyannote.audio.
    """

    async def diarize(
        self,
        segments: list[TranscriptionSegment],
        audio: AudioChunk | None = None,
    ) -> list[DiarizedSegment]:
        """Assign all segments to SPEAKER_00.

        Args:
            segments: ASR transcription segments.
            audio: Ignored — not needed for this strategy.

        Returns:
            List of ``DiarizedSegment`` all labelled ``SPEAKER_00``.
        """
        log.debug("diarize.none", segments=len(segments))
        return [
            DiarizedSegment(
                segment=seg,
                speaker_id="SPEAKER_00",
                speaker_source="none",
            )
            for seg in segments
        ]

    async def diarize_stream(
        self,
        segment_stream: AsyncIterator[tuple[TranscriptionSegment, AudioChunk]],
    ) -> AsyncIterator[DiarizedSegment]:
        """Streaming diarization — passes segments through with SPEAKER_00."""
        async for seg, _audio in segment_stream:
            yield DiarizedSegment(
                segment=seg,
                speaker_id="SPEAKER_00",
                speaker_source="none",
            )
