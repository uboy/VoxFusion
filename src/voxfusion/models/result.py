"""Result data models: TranscriptionResult."""

from dataclasses import dataclass, field

from voxfusion.models.translation import TranslatedSegment


@dataclass(frozen=True)
class TranscriptionResult:
    """Complete result of processing an audio source.

    Attributes:
        segments: Ordered list of translated (or untranslated) segments.
        source_info: Metadata about the audio source (device names,
            file path, duration, etc.).
        processing_info: Timing and model information (ASR model,
            processing duration, etc.).
        created_at: ISO 8601 timestamp of when the result was produced.
    """

    segments: list[TranslatedSegment]
    source_info: dict[str, str | int | float] = field(default_factory=dict)
    processing_info: dict[str, str | int | float] = field(default_factory=dict)
    created_at: str = ""
