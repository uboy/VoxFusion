"""Channel-based deterministic speaker diarization.

Assigns speaker labels based on the audio source metadata (e.g.
``"microphone"`` -> ``SPEAKER_LOCAL``, ``"system"`` -> ``SPEAKER_REMOTE``).
For single-source file input, all segments get ``SPEAKER_00``.
"""

from collections.abc import AsyncIterator

from voxfusion.config.models import DiarizationConfig
from voxfusion.logging import get_logger
from voxfusion.models.audio import AudioChunk
from voxfusion.models.diarization import DiarizedSegment
from voxfusion.models.transcription import TranscriptionSegment

log = get_logger(__name__)


class ChannelDiarizer:
    """Assigns speakers based on audio channel/source metadata."""

    def __init__(self, config: DiarizationConfig | None = None) -> None:
        self._config = config or DiarizationConfig()

    def _speaker_for_source(self, source: str) -> str:
        """Look up the speaker label for an audio source."""
        return self._config.channel_map.get(source, "SPEAKER_00")

    async def diarize(
        self,
        segments: list[TranscriptionSegment],
        audio: AudioChunk | None = None,
    ) -> list[DiarizedSegment]:
        """Assign speaker identities based on audio source metadata."""
        source = audio.source if audio else "file"
        speaker = self._speaker_for_source(source)

        log.debug("diarize.channel", source=source, speaker=speaker, segments=len(segments))

        return [
            DiarizedSegment(
                segment=seg,
                speaker_id=speaker,
                speaker_source="channel",
            )
            for seg in segments
        ]

    async def diarize_stream(
        self,
        segment_stream: AsyncIterator[tuple[TranscriptionSegment, AudioChunk]],
    ) -> AsyncIterator[DiarizedSegment]:
        """Streaming diarization: assigns speaker per chunk source."""
        async for seg, audio in segment_stream:
            speaker = self._speaker_for_source(audio.source)
            yield DiarizedSegment(
                segment=seg,
                speaker_id=speaker,
                speaker_source="channel",
            )
