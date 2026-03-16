"""Composable pre-processing pipeline that chains multiple processors.

Processors are applied in order. Each must conform to the
``AudioPreProcessor`` protocol (``process`` and ``reset`` methods).
"""

from voxfusion.logging import get_logger
from voxfusion.models.audio import AudioChunk
from voxfusion.preprocessing.base import AudioPreProcessor

log = get_logger(__name__)


class PreProcessingPipeline:
    """Chains multiple ``AudioPreProcessor`` implementations in sequence.

    Example::

        pipeline = PreProcessingPipeline([Resampler(16000), Normalizer()])
        processed = pipeline.process(raw_chunk)
    """

    def __init__(self, processors: list[AudioPreProcessor] | None = None) -> None:
        self._processors: list[AudioPreProcessor] = list(processors or [])

    def add(self, processor: AudioPreProcessor) -> None:
        """Append a processor to the end of the chain."""
        self._processors.append(processor)

    def process(self, chunk: AudioChunk) -> AudioChunk:
        """Run *chunk* through all processors in order."""
        for proc in self._processors:
            chunk = proc.process(chunk)
        return chunk

    def reset(self) -> None:
        """Reset all processors in the chain."""
        for proc in self._processors:
            proc.reset()

    def __len__(self) -> int:
        return len(self._processors)
