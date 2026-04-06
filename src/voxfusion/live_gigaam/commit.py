"""Ordered commit and textual overlap trimming for live GigaAM results."""

from __future__ import annotations

import re
from dataclasses import dataclass

from voxfusion.models.diarization import DiarizedSegment
from voxfusion.models.transcription import TranscriptionSegment
from voxfusion.models.translation import TranslatedSegment
from voxfusion.live_gigaam.types import LiveGigaAMResult

_WORD_RE = re.compile(r"\S+")
_NORMALIZE_RE = re.compile(r"[^\w]+", flags=re.UNICODE)


@dataclass(frozen=True)
class CommitOutcome:
    emitted: list[TranslatedSegment]
    committed: list[TranslatedSegment]


class OrderedTranscriptCommitter:
    """Commit live GigaAM results strictly in seq_id order."""

    def __init__(self, channel_map: dict[str, str]) -> None:
        self._channel_map = dict(channel_map)
        self._next_seq_id = 0
        self._pending: dict[int, LiveGigaAMResult] = {}
        self._committed: list[TranslatedSegment] = []
        self._source_tails: dict[str, str] = {}

    @property
    def committed_segments(self) -> list[TranslatedSegment]:
        return list(self._committed)

    def accept(self, result: LiveGigaAMResult) -> CommitOutcome:
        self._pending[result.seq_id] = result
        emitted: list[TranslatedSegment] = []
        while self._next_seq_id in self._pending:
            current = self._pending.pop(self._next_seq_id)
            emitted_segment = self._convert_result(current)
            if emitted_segment is not None:
                emitted.append(emitted_segment)
                self._committed.append(emitted_segment)
            self._next_seq_id += 1
        return CommitOutcome(emitted=emitted, committed=self.committed_segments)

    def _convert_result(self, result: LiveGigaAMResult) -> TranslatedSegment | None:
        text = self._trim_overlap(result.source, result.text)
        if result.error or not text:
            return None
        segment = TranscriptionSegment(
            text=text,
            language="ru",
            start_time=result.start_s,
            end_time=result.end_s,
            confidence=0.0,
            words=None,
            no_speech_prob=0.0,
        )
        speaker_id = self._channel_map.get(result.source, "SPEAKER_00")
        return TranslatedSegment(
            diarized=DiarizedSegment(
                segment=segment,
                speaker_id=speaker_id,
                speaker_source="channel",
            ),
            translated_text=None,
            target_language=None,
        )

    def _trim_overlap(self, source: str, text: str) -> str:
        current_words = _WORD_RE.findall(text.strip())
        if not current_words:
            return ""
        previous_tail = _WORD_RE.findall(self._source_tails.get(source, ""))
        overlap = self._shared_prefix_suffix(previous_tail, current_words)
        trimmed_words = current_words[overlap:] if overlap else current_words
        trimmed_text = " ".join(trimmed_words).strip()
        self._source_tails[source] = " ".join((previous_tail + trimmed_words)[-24:]).strip()
        return trimmed_text

    def _shared_prefix_suffix(self, previous: list[str], current: list[str]) -> int:
        if not previous or not current:
            return 0
        max_overlap = min(len(previous), len(current), 12)
        normalized_previous = [self._normalize_token(token) for token in previous]
        normalized_current = [self._normalize_token(token) for token in current]
        for length in range(max_overlap, 0, -1):
            if normalized_previous[-length:] == normalized_current[:length]:
                return length
        return 0

    @staticmethod
    def _normalize_token(token: str) -> str:
        return _NORMALIZE_RE.sub("", token.lower())
