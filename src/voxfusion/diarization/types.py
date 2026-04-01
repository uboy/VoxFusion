"""Speaker diarization helper types."""

from __future__ import annotations

from dataclasses import dataclass

from voxfusion.diarization.alignment import SpeakerTurn


@dataclass(frozen=True)
class DiarizationTurnResult:
    """Rich speaker-turn output for offline diarization workflows."""

    turns: list[SpeakerTurn]
    exclusive_turns: list[SpeakerTurn] | None = None
    speaker_count_hint_applied: str = "auto"
    speaker_count_estimate: int | None = None
    model_id: str | None = None
    is_chunk_local: bool = False

    def alignment_turns(self) -> list[SpeakerTurn]:
        """Return the best turn sequence for transcript alignment/windowing."""
        return self.exclusive_turns or self.turns
