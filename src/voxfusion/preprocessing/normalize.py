"""Audio amplitude normalization.

Scales audio samples so the peak absolute value reaches a target
level (default -3 dBFS). This prevents clipping while maximising
the signal range fed to the ASR model.
"""

import numpy as np

from voxfusion.logging import get_logger
from voxfusion.models.audio import AudioChunk

log = get_logger(__name__)

DEFAULT_TARGET_DBFS = -3.0


def _dbfs_to_linear(dbfs: float) -> float:
    """Convert dBFS to linear amplitude."""
    return 10 ** (dbfs / 20.0)


class Normalizer:
    """Peak-normalizes audio chunks to a target dBFS level."""

    def __init__(self, target_dbfs: float = DEFAULT_TARGET_DBFS) -> None:
        self._target_linear = _dbfs_to_linear(target_dbfs)

    def process(self, chunk: AudioChunk) -> AudioChunk:
        """Normalize *chunk* amplitude to the target level."""
        samples = chunk.samples
        if samples.size == 0:
            return chunk

        peak = np.max(np.abs(samples))
        if peak < 1e-8:
            # Silence — nothing to normalize
            return chunk

        gain = self._target_linear / peak
        if abs(gain - 1.0) < 1e-4:
            return chunk

        log.debug("normalize", peak=float(peak), gain=float(gain))
        normalized = (samples * gain).astype(np.float32)

        return AudioChunk(
            samples=normalized,
            sample_rate=chunk.sample_rate,
            channels=chunk.channels,
            timestamp_start=chunk.timestamp_start,
            timestamp_end=chunk.timestamp_end,
            source=chunk.source,
            dtype="float32",
        )

    def reset(self) -> None:
        """No internal state to reset."""
