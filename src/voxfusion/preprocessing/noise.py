"""Optional noise reduction via noisereduce library.

Uses the ``noisereduce`` package (MIT license) which implements
spectral gating for stationary and non-stationary noise reduction.
"""

import asyncio

import numpy as np

from voxfusion.logging import get_logger
from voxfusion.models.audio import AudioChunk

log = get_logger(__name__)


class NoiseReducer:
    """Audio noise reduction using spectral gating.

    Wraps the ``noisereduce`` library to provide stationary and
    non-stationary noise reduction.  The library is lazily imported
    so the system gracefully degrades if it is not installed.
    """

    def __init__(
        self,
        stationary: bool = True,
        prop_decrease: float = 1.0,
        n_fft: int = 1024,
        n_std_thresh_stationary: float = 1.5,
    ) -> None:
        """Initialize noise reducer.

        Args:
            stationary: If True, use stationary noise reduction (faster).
                If False, use non-stationary (adaptive, slower).
            prop_decrease: Proportion to reduce noise by (0.0 to 1.0).
                1.0 = full reduction, 0.0 = no reduction.
            n_fft: FFT size for STFT computation.
            n_std_thresh_stationary: Number of standard deviations above
                the noise floor to consider signal.
        """
        self._stationary = stationary
        self._prop_decrease = prop_decrease
        self._n_fft = n_fft
        self._n_std_thresh = n_std_thresh_stationary

    def _reduce_sync(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        """Run noise reduction synchronously."""
        try:
            import noisereduce as nr
        except ImportError as exc:
            raise RuntimeError(
                "noisereduce is not installed. "
                "Install with: pip install noisereduce"
            ) from exc

        reduced = nr.reduce_noise(
            y=samples,
            sr=sample_rate,
            stationary=self._stationary,
            prop_decrease=self._prop_decrease,
            n_fft=self._n_fft,
            n_std_thresh_stationary=self._n_std_thresh,
        )
        return reduced.astype(np.float32)

    def process(self, chunk: AudioChunk) -> AudioChunk:
        """Apply noise reduction to an audio chunk.

        Args:
            chunk: Input audio chunk.

        Returns:
            Audio chunk with reduced noise.
        """
        samples = chunk.samples
        if samples.ndim == 2:
            samples = samples.mean(axis=1)
        samples = samples.astype(np.float32)

        if len(samples) < self._n_fft:
            log.debug("noise.chunk_too_short", samples=len(samples))
            return chunk

        reduced = self._reduce_sync(samples, chunk.sample_rate)

        log.debug(
            "noise.reduced",
            original_rms=round(float(np.sqrt(np.mean(samples**2))), 6),
            reduced_rms=round(float(np.sqrt(np.mean(reduced**2))), 6),
        )

        return AudioChunk(
            samples=reduced,
            sample_rate=chunk.sample_rate,
            channels=1,
            timestamp_start=chunk.timestamp_start,
            timestamp_end=chunk.timestamp_end,
            source=chunk.source,
            dtype="float32",
        )

    async def process_async(self, chunk: AudioChunk) -> AudioChunk:
        """Async wrapper that offloads noise reduction to an executor."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.process, chunk)

    def reset(self) -> None:
        """Reset state (no-op for stateless noise reduction)."""
