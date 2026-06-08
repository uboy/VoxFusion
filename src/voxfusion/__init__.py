"""VoxFusion — cross-platform audio capture, transcription, diarization, and translation."""

import warnings

# Suppress noisy third-party warnings that fire at import time, before any
# voxfusion sub-module is loaded. These come from torch (CUDA CC mismatch),
# pyannote (torchcodec missing), and torchaudio (deprecation).
# MUST be at the top of __init__.py because importing voxfusion triggers
# PipelineOrchestrator -> diarization factory -> pyannote_engine at load time.
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote")
warnings.filterwarnings("ignore", message=".*torchaudio._backend.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*torchcodec.*", category=UserWarning)

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
    "AudioChunk",
    "AudioDeviceInfo",
    "DiarizedSegment",
    "PipelineConfig",
    "PipelineOrchestrator",
    "TranscriptionResult",
    "TranscriptionSegment",
    "TranslatedSegment",
    "WordTiming",
    "__version__",
    "__version_info__",
    "load_config",
]
