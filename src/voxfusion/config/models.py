"""Pydantic v2 configuration models for all VoxFusion settings."""

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from voxfusion.asr_catalog import get_model_info, normalize_language_for_model


class VADParameters(BaseModel):
    """Voice Activity Detection parameters."""

    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    min_speech_duration_ms: int = Field(default=250, ge=0)
    min_silence_duration_ms: int = Field(default=2000, ge=0)


class CaptureConfig(BaseModel):
    """Audio capture configuration."""

    sources: list[str] = Field(default_factory=lambda: ["microphone"])
    sample_rate: int = Field(default=44100, ge=8000, le=48000)  # 44100 более совместимо
    channels: int = Field(default=1, ge=1, le=2)
    chunk_duration_ms: int = Field(default=500, ge=100, le=5000)
    buffer_size: int = Field(default=50, ge=1, le=100)  # Увеличен для streaming
    lossy_mode: bool = True  # Включен для streaming


class ASRConfig(BaseModel):
    """Automatic Speech Recognition configuration."""

    engine: str = "faster-whisper"
    model_size: str = "small"
    model_path: str | None = None
    device: str = "auto"
    compute_type: str = "int8_float32"
    cpu_threads: int = Field(default=0, ge=0)  # 0 = use all available cores
    beam_size: int = Field(default=5, ge=1, le=20)
    best_of: int = Field(default=5, ge=1)
    patience: float = Field(default=1.0, ge=0.0)
    language: str | None = None
    initial_prompt: str | None = None
    word_timestamps: bool = False
    # Max simultaneous ASR calls when transcribing speaker windows in batch mode.
    # 1 = sequential (required for GigaAM which is not thread-safe).
    # faster-whisper supports values up to cpu_count.
    parallel_windows: int = Field(default=1, ge=1, le=16)
    vad_filter: bool = True
    vad_parameters: VADParameters = Field(default_factory=VADParameters)
    no_speech_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    chunk_duration_s: int = Field(default=5, ge=1, le=30)
    chunk_overlap_s: int = Field(default=1, ge=0, le=10)

    @model_validator(mode="after")
    def _normalize_catalog_compatibility(self) -> "ASRConfig":
        model_info = get_model_info(self.model_size)
        self.model_size = model_info.id
        self.engine = model_info.engine
        self.language = normalize_language_for_model(self.model_size, self.language)
        return self


class DiarizationMLConfig(BaseModel):
    """ML-based diarization engine configuration."""

    engine: str = "pyannote"
    model: str = "pyannote/speaker-diarization-3.1"
    hf_auth_token: str | None = None
    device: str = "auto"
    min_speakers: int | None = Field(default=None, ge=1)
    max_speakers: int | None = Field(default=None, ge=1)
    min_segment_duration: float = Field(default=0.5, ge=0.0)
    # Chunked diarization — splits long audio into overlapping chunks so that
    # progress can be reported and processing can be parallelised.
    chunked: bool = True
    chunk_duration_s: float = Field(default=300.0, gt=0.0)
    chunk_overlap_s: float = Field(default=10.0, ge=0.0)
    # Number of parallel chunk workers.  None = auto (cpu_count//2 on CPU, 1 on GPU).
    chunk_max_workers: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _validate_speaker_hints(self) -> "DiarizationMLConfig":
        if (
            self.min_speakers is not None
            and self.max_speakers is not None
            and self.min_speakers > self.max_speakers
        ):
            raise ValueError("diarization.ml.min_speakers must be <= max_speakers")
        return self


class DiarizationConfig(BaseModel):
    """Speaker diarization configuration."""

    strategy: str = "channel"
    channel_map: dict[str, str] = Field(
        default_factory=lambda: {
            "microphone": "SPEAKER_LOCAL",
            "system": "SPEAKER_REMOTE",
        }
    )
    ml: DiarizationMLConfig = Field(default_factory=DiarizationMLConfig)


class TranslationCacheConfig(BaseModel):
    """Translation cache settings."""

    enabled: bool = True
    max_size: int = Field(default=10000, ge=0)
    ttl: int = Field(default=3600, ge=0)


class TranslationConfig(BaseModel):
    """Text translation configuration."""

    enabled: bool = False
    target_language: str = "en"
    backend: str = "argos"
    cache: TranslationCacheConfig = Field(default_factory=TranslationCacheConfig)


class OutputConfig(BaseModel):
    """Output formatting configuration."""

    format: str = "json"
    include_word_timestamps: bool = False
    include_translation: bool = True
    include_confidence: bool = True


class SecurityConfig(BaseModel):
    """Security and privacy configuration."""

    encrypt_output: bool = False
    encryption_passphrase: str | None = None
    log_transcription_content: bool = False
    telemetry: bool = False
    auto_delete_temp_files: bool = True
    temp_file_ttl_hours: int = Field(default=24, ge=1)


class PipelineConfig(BaseSettings):
    """Root configuration model.

    Supports environment variable overrides with the ``VOXFUSION_`` prefix
    and ``__`` as the nested delimiter.  For example::

        VOXFUSION_ASR__MODEL_SIZE=medium
        VOXFUSION_LOG_LEVEL=DEBUG
    """

    model_config = SettingsConfigDict(
        env_prefix="VOXFUSION_",
        env_nested_delimiter="__",
    )

    capture: CaptureConfig = Field(default_factory=CaptureConfig)
    asr: ASRConfig = Field(default_factory=ASRConfig)
    diarization: DiarizationConfig = Field(default_factory=DiarizationConfig)
    translation: TranslationConfig = Field(default_factory=TranslationConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    log_level: str = "INFO"
    data_dir: str = "~/.voxfusion"
