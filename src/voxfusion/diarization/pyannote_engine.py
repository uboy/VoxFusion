"""ML-based speaker diarization using pyannote.audio.

Requires the ``pyannote.audio`` package and a Hugging Face auth token
for model download.  The diarization pipeline identifies speaker turns
which are then aligned with ASR segments.
"""

import asyncio
from collections.abc import AsyncIterator

import numpy as np

from voxfusion.config.models import DiarizationMLConfig
from voxfusion.diarization.alignment import SpeakerTurn, align_segments
from voxfusion.exceptions import DiarizationError
from voxfusion.logging import get_logger
from voxfusion.models.audio import AudioChunk
from voxfusion.models.diarization import DiarizedSegment
from voxfusion.models.transcription import TranscriptionSegment

log = get_logger(__name__)


class PyAnnoteDiarizer:
    """Speaker diarization using pyannote.audio.

    The pipeline model is lazily loaded on first ``diarize`` call.
    Inference is offloaded to a thread executor since it is CPU/GPU-bound.
    """

    def __init__(self, config: DiarizationMLConfig | None = None) -> None:
        self._config = config or DiarizationMLConfig()
        self._pipeline: object | None = None

    def _load_pipeline(self) -> object:
        """Load the pyannote diarization pipeline."""
        if self._pipeline is not None:
            return self._pipeline

        try:
            from pyannote.audio import Pipeline
        except ImportError as exc:
            raise DiarizationError(
                "pyannote.audio is not installed. "
                "Install with: pip install pyannote.audio"
            ) from exc

        token = self._config.hf_auth_token
        if not token:
            raise DiarizationError(
                "Hugging Face auth token required for pyannote models. "
                "Set VOXFUSION_DIARIZATION__ML__HF_AUTH_TOKEN"
            )

        log.info("pyannote.loading_pipeline", model=self._config.model)
        try:
            self._pipeline = Pipeline.from_pretrained(
                self._config.model,
                use_auth_token=token,
            )
        except Exception as exc:
            raise DiarizationError(f"Failed to load pyannote pipeline: {exc}") from exc

        # Move to GPU if available
        if self._config.device == "auto" or self._config.device == "cuda":
            try:
                import torch

                if torch.cuda.is_available():
                    self._pipeline.to(torch.device("cuda"))  # type: ignore[union-attr]
                    log.info("pyannote.using_gpu")
            except ImportError:
                pass

        log.info("pyannote.pipeline_loaded")
        return self._pipeline

    def _diarize_sync(self, audio: np.ndarray, sample_rate: int) -> list[SpeakerTurn]:
        """Run diarization synchronously."""
        pipeline = self._load_pipeline()

        # pyannote expects a dict with "waveform" and "sample_rate"
        import torch

        waveform = torch.from_numpy(audio).float().unsqueeze(0)
        input_data = {"waveform": waveform, "sample_rate": sample_rate}

        kwargs = {}
        if self._config.min_speakers is not None:
            kwargs["min_speakers"] = self._config.min_speakers
        if self._config.max_speakers is not None:
            kwargs["max_speakers"] = self._config.max_speakers

        diarization = pipeline(input_data, **kwargs)  # type: ignore[operator]

        turns: list[SpeakerTurn] = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            if turn.duration < self._config.min_segment_duration:
                continue
            turns.append(SpeakerTurn(
                speaker_id=speaker,
                start_time=turn.start,
                end_time=turn.end,
            ))

        log.info("pyannote.diarized", turns=len(turns))
        return turns

    async def diarize(
        self,
        segments: list[TranscriptionSegment],
        audio: AudioChunk | None = None,
    ) -> list[DiarizedSegment]:
        """Diarize segments using pyannote ML pipeline.

        Args:
            segments: ASR transcription segments to assign speakers to.
            audio: The audio chunk (required for ML diarization).

        Returns:
            Diarized segments with ML-assigned speaker labels.
        """
        if audio is None:
            raise DiarizationError("PyAnnoteDiarizer requires audio data")

        samples = audio.samples
        if samples.ndim == 2:
            samples = samples.mean(axis=1)

        loop = asyncio.get_running_loop()
        turns = await loop.run_in_executor(
            None, self._diarize_sync, samples, audio.sample_rate
        )

        return align_segments(segments, turns, speaker_source="ml")

    async def diarize_stream(
        self,
        segment_stream: AsyncIterator[tuple[TranscriptionSegment, AudioChunk]],
    ) -> AsyncIterator[DiarizedSegment]:
        """Streaming diarization — processes each chunk independently."""
        async for seg, audio in segment_stream:
            result = await self.diarize([seg], audio)
            for d in result:
                yield d
