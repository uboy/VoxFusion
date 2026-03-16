"""Automatic Speech Recognition engine integrations."""

from voxfusion.asr.base import ASREngine
from voxfusion.asr.dedup import OverlapDeduplicator
from voxfusion.asr.faster_whisper import FasterWhisperEngine
from voxfusion.asr.streaming import StreamingASR

__all__ = [
    "ASREngine",
    "FasterWhisperEngine",
    "OverlapDeduplicator",
    "StreamingASR",
]
