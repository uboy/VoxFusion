"""Unit tests for ordered live GigaAM transcript commit."""

from __future__ import annotations

from voxfusion.live_gigaam.commit import OrderedTranscriptCommitter
from voxfusion.live_gigaam.types import LiveGigaAMResult


def test_committer_orders_results_and_trims_same_source_overlap() -> None:
    committer = OrderedTranscriptCommitter({"microphone": "SPEAKER_LOCAL"})

    later = LiveGigaAMResult(
        seq_id=1,
        source="microphone",
        start_s=1.0,
        end_s=2.0,
        text="мир и снова",
        worker_id=1,
    )
    first = LiveGigaAMResult(
        seq_id=0,
        source="microphone",
        start_s=0.0,
        end_s=1.0,
        text="привет мир",
        worker_id=0,
    )

    assert committer.accept(later).emitted == []
    outcome = committer.accept(first)

    assert [segment.diarized.segment.text for segment in outcome.emitted] == [
        "привет мир",
        "и снова",
    ]
    assert [segment.diarized.speaker_id for segment in outcome.emitted] == [
        "SPEAKER_LOCAL",
        "SPEAKER_LOCAL",
    ]


def test_committer_keeps_source_tails_isolated() -> None:
    committer = OrderedTranscriptCommitter(
        {"microphone": "SPEAKER_LOCAL", "system": "SPEAKER_REMOTE"}
    )

    first = LiveGigaAMResult(
        seq_id=0,
        source="microphone",
        start_s=0.0,
        end_s=1.0,
        text="hello there",
        worker_id=0,
    )
    second = LiveGigaAMResult(
        seq_id=1,
        source="system",
        start_s=1.0,
        end_s=2.0,
        text="hello there",
        worker_id=1,
    )

    committer.accept(first)
    outcome = committer.accept(second)

    assert [segment.diarized.segment.text for segment in outcome.committed] == [
        "hello there",
        "hello there",
    ]
    assert outcome.committed[1].diarized.speaker_id == "SPEAKER_REMOTE"


def test_committer_advances_past_empty_error_results() -> None:
    committer = OrderedTranscriptCommitter({"microphone": "SPEAKER_LOCAL"})

    errored = LiveGigaAMResult(
        seq_id=0,
        source="microphone",
        start_s=0.0,
        end_s=1.0,
        text="",
        worker_id=0,
        error="boom",
    )
    valid = LiveGigaAMResult(
        seq_id=1,
        source="microphone",
        start_s=1.0,
        end_s=2.0,
        text="next phrase",
        worker_id=1,
    )

    committer.accept(errored)
    outcome = committer.accept(valid)

    assert [segment.diarized.segment.text for segment in outcome.emitted] == ["next phrase"]
