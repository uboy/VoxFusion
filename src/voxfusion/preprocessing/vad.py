"""Voice Activity Detection using Silero VAD.

Silero VAD is a lightweight, pre-trained model that detects speech
segments in audio.  It runs on CPU with minimal latency and is
distributed under the MIT license.
"""

import asyncio

import numpy as np

from voxfusion.config.models import VADParameters
from voxfusion.logging import get_logger
from voxfusion.models.audio import AudioChunk

log = get_logger(__name__)

_SILERO_SAMPLE_RATE = 16000
_WINDOW_SIZE_SAMPLES = 512  # Silero expects 512-sample windows at 16kHz


class SileroVAD:
    """Voice Activity Detection filter using Silero VAD.

    Filters audio chunks to retain only speech regions.  Non-speech
    regions are zeroed out (replaced with silence) so that downstream
    timestamps remain valid.

    The Silero model is lazily loaded on first use.
    """

    def __init__(self, params: VADParameters | None = None) -> None:
        self._params = params or VADParameters()
        self._model: object | None = None
        self._speech_timestamps: list[dict[str, int]] = []

    def _load_model(self) -> object:
        """Load the Silero VAD model via torch.hub."""
        if self._model is not None:
            return self._model

        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "torch is required for Silero VAD. "
                "Install with: pip install torch"
            ) from exc

        log.info("vad.loading_silero_model")
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
        )
        self._model = model
        self._get_speech_timestamps = utils[0]  # get_speech_timestamps
        log.info("vad.model_loaded")
        return self._model

    def _detect_speech_sync(
        self,
        samples: np.ndarray,
        sample_rate: int,
    ) -> list[dict[str, int]]:
        """Run VAD synchronously, returning speech timestamp ranges."""
        import torch

        model = self._load_model()

        tensor = torch.from_numpy(samples).float()

        timestamps = self._get_speech_timestamps(
            tensor,
            model,
            sampling_rate=sample_rate,
            threshold=self._params.threshold,
            min_speech_duration_ms=self._params.min_speech_duration_ms,
            min_silence_duration_ms=self._params.min_silence_duration_ms,
        )
        return timestamps

    def process(self, chunk: AudioChunk) -> AudioChunk:
        """Filter non-speech regions from an audio chunk.

        Speech regions are kept; non-speech regions are zeroed out.
        This preserves the original chunk duration and timestamps.

        Args:
            chunk: Input audio chunk.

        Returns:
            Audio chunk with non-speech regions silenced.
        """
        samples = chunk.samples.copy()
        if samples.ndim == 2:
            samples = samples.mean(axis=1)
        samples = samples.astype(np.float32)

        timestamps = self._detect_speech_sync(samples, chunk.sample_rate)

        if not timestamps:
            log.debug("vad.no_speech_detected", duration=chunk.duration)
            return AudioChunk(
                samples=np.zeros_like(samples),
                sample_rate=chunk.sample_rate,
                channels=1,
                timestamp_start=chunk.timestamp_start,
                timestamp_end=chunk.timestamp_end,
                source=chunk.source,
                dtype="float32",
            )

        # Zero out non-speech regions
        mask = np.zeros(len(samples), dtype=np.float32)
        for ts in timestamps:
            start = ts["start"]
            end = ts["end"]
            mask[start:end] = 1.0

        filtered = samples * mask

        log.debug(
            "vad.filtered",
            speech_regions=len(timestamps),
            speech_ratio=round(float(mask.sum()) / len(mask), 3),
        )

        return AudioChunk(
            samples=filtered,
            sample_rate=chunk.sample_rate,
            channels=1,
            timestamp_start=chunk.timestamp_start,
            timestamp_end=chunk.timestamp_end,
            source=chunk.source,
            dtype="float32",
        )

    async def process_async(self, chunk: AudioChunk) -> AudioChunk:
        """Async wrapper that offloads VAD to an executor."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.process, chunk)

    def get_speech_segments(
        self,
        chunk: AudioChunk,
    ) -> list[tuple[float, float]]:
        """Return speech segment timestamps in seconds.

        Args:
            chunk: Input audio chunk.

        Returns:
            List of (start_seconds, end_seconds) tuples.
        """
        samples = chunk.samples
        if samples.ndim == 2:
            samples = samples.mean(axis=1)
        samples = samples.astype(np.float32)

        timestamps = self._detect_speech_sync(samples, chunk.sample_rate)

        return [
            (
                ts["start"] / chunk.sample_rate + chunk.timestamp_start,
                ts["end"] / chunk.sample_rate + chunk.timestamp_start,
            )
            for ts in timestamps
        ]

    def reset(self) -> None:
        """Reset internal state (re-initialises the model state)."""
        if self._model is not None:
            self._model.reset_states()  # type: ignore[union-attr]
