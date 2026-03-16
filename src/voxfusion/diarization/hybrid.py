"""Hybrid diarization combining channel-based and ML-based strategies.

Uses channel-based diarization as the primary strategy for known
source types (microphone vs. system audio), then falls back to
ML-based diarization for further speaker separation within each
channel.
"""

from collections.abc import AsyncIterator

from voxfusion.config.models import DiarizationConfig
from voxfusion.diarization.channel import ChannelDiarizer
from voxfusion.exceptions import DiarizationError
from voxfusion.logging import get_logger
from voxfusion.models.audio import AudioChunk
from voxfusion.models.diarization import DiarizedSegment
from voxfusion.models.transcription import TranscriptionSegment

log = get_logger(__name__)


class HybridDiarizer:
    """Combines channel-based and ML-based speaker diarization.

    Strategy:
    1. First, assign speakers by audio channel/source.
    2. If ML diarization is available, further split segments
       within each channel to identify multiple speakers.
    """

    def __init__(self, config: DiarizationConfig | None = None) -> None:
        self._config = config or DiarizationConfig()
        self._channel = ChannelDiarizer(self._config)
        self._ml_engine: object | None = None

    def _get_ml_engine(self) -> object | None:
        """Lazily load the ML diarization engine."""
        if self._ml_engine is not None:
            return self._ml_engine

        try:
            from voxfusion.diarization.pyannote_engine import PyAnnoteDiarizer

            self._ml_engine = PyAnnoteDiarizer(self._config.ml)
            return self._ml_engine
        except Exception as exc:
            log.warning("hybrid.ml_unavailable", error=str(exc))
            return None

    async def diarize(
        self,
        segments: list[TranscriptionSegment],
        audio: AudioChunk | None = None,
    ) -> list[DiarizedSegment]:
        """Diarize using channel-first, ML-fallback strategy.

        If the audio source is known (microphone/system), channel
        diarization is used.  If the source is ambiguous or ML
        is configured, ML diarization refines the result.
        """
        # Start with channel-based assignment
        channel_result = await self._channel.diarize(segments, audio)

        # If all segments got a known channel speaker, return as-is
        if audio and audio.source in self._config.channel_map:
            return channel_result

        # Attempt ML refinement
        ml_engine = self._get_ml_engine()
        if ml_engine is None or audio is None:
            return channel_result

        try:
            ml_result = await ml_engine.diarize(segments, audio)  # type: ignore[union-attr]
            log.info("hybrid.ml_applied", segments=len(ml_result))
            return ml_result
        except DiarizationError as exc:
            log.warning("hybrid.ml_failed", error=str(exc))
            return channel_result

    async def diarize_stream(
        self,
        segment_stream: AsyncIterator[tuple[TranscriptionSegment, AudioChunk]],
    ) -> AsyncIterator[DiarizedSegment]:
        """Streaming hybrid diarization."""
        async for seg, audio in segment_stream:
            result = await self.diarize([seg], audio)
            for d in result:
                yield d
