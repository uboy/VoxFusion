"""Translation data models: TranslatedSegment."""

from dataclasses import dataclass

from voxfusion.models.diarization import DiarizedSegment


@dataclass(frozen=True)
class TranslatedSegment:
    """A diarized segment with optional translation.

    Attributes:
        diarized: The underlying diarized segment.
        translated_text: Translated text, or ``None`` if translation
            was not requested.
        target_language: ISO 639-1 code of the target language,
            or ``None``.
    """

    diarized: DiarizedSegment
    translated_text: str | None
    target_language: str | None
