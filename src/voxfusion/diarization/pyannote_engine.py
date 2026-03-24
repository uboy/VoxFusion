"""ML-based speaker diarization using pyannote.audio.

Requires the ``pyannote.audio`` package and a Hugging Face auth token
for model download.  The diarization pipeline identifies speaker turns
which are then aligned with ASR segments.
"""

import asyncio
import inspect
import os
import warnings
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


def _pipeline_auth_kwargs(
    from_pretrained: object,
    token: str,
) -> dict[str, str]:
    """Return auth kwargs compatible with the installed pyannote version."""
    try:
        signature = inspect.signature(from_pretrained)
    except (TypeError, ValueError):
        return {"token": token}

    parameters = signature.parameters
    if "token" in parameters:
        return {"token": token}
    if "use_auth_token" in parameters:
        return {"use_auth_token": token}
    return {"token": token}


def _extract_annotation(diarization_output: object) -> object:
    """Return the pyannote annotation object across API variants."""
    if hasattr(diarization_output, "itertracks"):
        return diarization_output

    speaker_diarization = getattr(diarization_output, "speaker_diarization", None)
    if speaker_diarization is not None and hasattr(speaker_diarization, "itertracks"):
        return speaker_diarization

    raise DiarizationError(
        "Unsupported pyannote diarization output: missing speaker annotation"
    )


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

        warnings.filterwarnings(
            "ignore",
            message=r"\s*torchcodec is not installed correctly so built-in audio decoding will fail\..*",
            category=UserWarning,
            module=r"pyannote\.audio\.core\.io",
        )
        try:
            from pyannote.audio import Pipeline
        except ImportError as exc:
            raise DiarizationError(
                "pyannote.audio is not installed. "
                "Install with: pip install pyannote.audio"
            ) from exc

        token = (
            self._config.hf_auth_token
            or os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        )
        if not token:
            raise DiarizationError(
                "Hugging Face auth token required for pyannote models. "
                "Set VOXFUSION_DIARIZATION__ML__HF_AUTH_TOKEN or HF_TOKEN"
            )

        log.info("pyannote.loading_pipeline", model=self._config.model)
        try:
            auth_kwargs = _pipeline_auth_kwargs(Pipeline.from_pretrained, token)
            self._pipeline = Pipeline.from_pretrained(
                self._config.model,
                **auth_kwargs,
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
        annotation = _extract_annotation(diarization)

        turns: list[SpeakerTurn] = []
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            if turn.duration < self._config.min_segment_duration:
                continue
            turns.append(SpeakerTurn(
                speaker_id=speaker,
                start_time=turn.start,
                end_time=turn.end,
            ))

        log.info("pyannote.diarized", turns=len(turns))
        return turns

    async def diarize_turns(self, audio: AudioChunk) -> list[SpeakerTurn]:
        """Return raw speaker turns in absolute audio coordinates."""
        samples = audio.samples
        if samples.ndim == 2:
            samples = samples.mean(axis=1)

        loop = asyncio.get_running_loop()
        turns = await loop.run_in_executor(
            None, self._diarize_sync, samples, audio.sample_rate
        )
        if not audio.timestamp_start:
            return turns
        return [
            SpeakerTurn(
                speaker_id=turn.speaker_id,
                start_time=turn.start_time + audio.timestamp_start,
                end_time=turn.end_time + audio.timestamp_start,
            )
            for turn in turns
        ]

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

        turns = await self.diarize_turns(audio)
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
