"""Transcription data models: TranscriptionSegment and WordTiming."""

from dataclasses import dataclass


@dataclass(frozen=True)
class WordTiming:
    """Word-level timing information from ASR.

    Attributes:
        word: The recognized word or punctuation.
        start_time: Start time in seconds.
        end_time: End time in seconds.
        probability: Model confidence for this word (0.0–1.0).
    """

    word: str
    start_time: float
    end_time: float
    probability: float


@dataclass(frozen=True)
class TranscriptionSegment:
    """A single segment of transcribed speech.

    Attributes:
        text: Transcribed text content.
        language: ISO 639-1 language code (e.g. ``"en"``).
        start_time: Segment start time in seconds.
        end_time: Segment end time in seconds.
        confidence: Overall model confidence (0.0–1.0).
        words: Optional word-level timing details.
        no_speech_prob: Probability that the segment contains no speech.
    """

    text: str
    language: str
    start_time: float
    end_time: float
    confidence: float
    words: list[WordTiming] | None
    no_speech_prob: float

    @property
    def duration(self) -> float:
        """Segment duration in seconds."""
        return self.end_time - self.start_time
