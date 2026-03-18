"""Audio sample rate conversion (resampling to 16kHz for Whisper).

Uses scipy.signal.resample_poly for high-quality polyphase resampling.
Falls back to simple linear interpolation if scipy is not available.
"""

from math import gcd

import numpy as np

from voxfusion.logging import get_logger
from voxfusion.models.audio import AudioChunk

log = get_logger(__name__)

TARGET_SAMPLE_RATE = 16_000


def _resample_array(samples: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample a 1-D or 2-D audio array from *orig_sr* to *target_sr*."""
    if orig_sr == target_sr:
        return samples

    try:
        from scipy.signal import resample_poly
    except ImportError:
        # Fallback: simple linear interpolation (scipy not installed).
        # np.interp only handles 1-D arrays, so process each channel separately.
        n_in = samples.shape[0]
        ratio = target_sr / orig_sr
        n_out = max(1, int(n_in * ratio))
        xs_old = np.arange(n_in, dtype=np.float64)
        xs_new = np.linspace(0, n_in - 1, n_out)
        if samples.ndim == 1:
            return np.interp(xs_new, xs_old, samples).astype(np.float32)
        # 2-D: (frames, channels) — resample each channel independently
        channels = [
            np.interp(xs_new, xs_old, samples[:, ch]) for ch in range(samples.shape[1])
        ]
        return np.stack(channels, axis=1).astype(np.float32)

    divisor = gcd(orig_sr, target_sr)
    up = target_sr // divisor
    down = orig_sr // divisor
    return resample_poly(samples, up, down).astype(np.float32)


class Resampler:
    """Resamples audio chunks to a target sample rate."""

    def __init__(self, target_sample_rate: int = TARGET_SAMPLE_RATE) -> None:
        self._target_sr = target_sample_rate

    def process(self, chunk: AudioChunk) -> AudioChunk:
        """Resample *chunk* to the target sample rate."""
        if chunk.sample_rate == self._target_sr:
            return chunk

        log.debug(
            "resample",
            from_sr=chunk.sample_rate,
            to_sr=self._target_sr,
        )

        resampled = _resample_array(chunk.samples, chunk.sample_rate, self._target_sr)

        return AudioChunk(
            samples=resampled,
            sample_rate=self._target_sr,
            channels=chunk.channels,
            timestamp_start=chunk.timestamp_start,
            timestamp_end=chunk.timestamp_end,
            source=chunk.source,
            dtype="float32",
        )

    def reset(self) -> None:
        """No internal state to reset."""
