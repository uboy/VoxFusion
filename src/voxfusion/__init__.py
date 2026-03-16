"""VoxFusion — cross-platform audio capture, transcription, diarization, and translation."""

from voxfusion.config.loader import load_config
from voxfusion.config.models import PipelineConfig
from voxfusion.models.audio import AudioChunk, AudioDeviceInfo
from voxfusion.models.diarization import DiarizedSegment
from voxfusion.models.result import TranscriptionResult
from voxfusion.models.transcription import TranscriptionSegment, WordTiming
from voxfusion.models.translation import TranslatedSegment
from voxfusion.pipeline.orchestrator import PipelineOrchestrator
from voxfusion.version import __version__, __version_info__

__all__ = [
    "__version__",
    "__version_info__",
    "AudioChunk",
    "AudioDeviceInfo",
    "DiarizedSegment",
    "PipelineConfig",
    "PipelineOrchestrator",
    "TranscriptionResult",
    "TranscriptionSegment",
    "TranslatedSegment",
    "WordTiming",
    "load_config",
]
