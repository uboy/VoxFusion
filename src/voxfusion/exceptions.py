"""Exception hierarchy for VoxFusion.

All domain-specific exceptions inherit from ``VoxFusionError``.
"""


class VoxFusionError(Exception):
    """Base exception for all VoxFusion errors."""


# -- Audio Capture Errors --


class AudioCaptureError(VoxFusionError):
    """Base for audio capture failures."""


class DeviceNotFoundError(AudioCaptureError):
    """Requested audio device does not exist."""


class DeviceAccessDeniedError(AudioCaptureError):
    """OS denied access to the audio device (permissions)."""


class DeviceDisconnectedError(AudioCaptureError):
    """Audio device was disconnected during capture."""


class AudioCaptureTimeout(AudioCaptureError):
    """No audio data received within the timeout period."""


class UnsupportedPlatformError(AudioCaptureError):
    """Current OS platform is not supported."""


# -- ASR Errors --


class ASRError(VoxFusionError):
    """Base for ASR failures."""


class ModelNotFoundError(ASRError):
    """Requested ASR model is not downloaded."""


class ModelLoadError(ASRError):
    """Failed to load ASR model (OOM, corrupted, etc.)."""


class TranscriptionError(ASRError):
    """Transcription failed for a given audio chunk."""


# -- Diarization Errors --


class DiarizationError(VoxFusionError):
    """Base for diarization failures."""


# -- Translation Errors --


class TranslationError(VoxFusionError):
    """Base for translation failures."""


class UnsupportedLanguagePair(TranslationError):
    """The requested language pair is not supported by the backend."""


class TranslationAPIError(TranslationError):
    """External translation API returned an error."""


# -- Configuration Errors --


class ConfigurationError(VoxFusionError):
    """Invalid or missing configuration."""


# -- Pipeline Errors --


class PipelineError(VoxFusionError):
    """Pipeline orchestration failure."""
