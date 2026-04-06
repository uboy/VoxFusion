"""Focused tests for live GigaAM soak harness helpers."""

from __future__ import annotations

import numpy as np
import pytest

from voxfusion.live_gigaam.soak import build_utterance_corpus, normalize_sources, summarize_latencies_ms


def test_normalize_sources_expands_both_and_deduplicates() -> None:
    assert normalize_sources("both,microphone") == ("microphone", "system")


def test_normalize_sources_rejects_unknown_value() -> None:
    with pytest.raises(ValueError):
        normalize_sources("line-in")


def test_build_utterance_corpus_splits_audio_and_rotates_sources() -> None:
    samples = np.ones(16000 * 7, dtype=np.float32)

    corpus = build_utterance_corpus(
        samples,
        16000,
        min_duration_s=2.0,
        max_duration_s=3.0,
        sources=("microphone", "system"),
        seed=3,
    )

    assert len(corpus) >= 3
    assert {item.source for item in corpus} == {"microphone", "system"}
    assert all(item.sample_rate == 16000 for item in corpus)
    assert all(item.samples.dtype == np.float32 for item in corpus)
    assert sum(item.samples.size for item in corpus) <= samples.size


def test_build_utterance_corpus_returns_single_chunk_for_short_audio() -> None:
    samples = np.ones(8000, dtype=np.float32)

    corpus = build_utterance_corpus(
        samples,
        16000,
        min_duration_s=1.0,
        max_duration_s=2.0,
        sources=("microphone",),
        seed=5,
    )

    assert len(corpus) == 1
    assert corpus[0].duration_s == 0.5


def test_summarize_latencies_ms_returns_expected_percentiles() -> None:
    summary = summarize_latencies_ms([5.0, 10.0, 15.0, 20.0, 100.0])

    assert summary == {
        "count": 5,
        "avg": 30.0,
        "p50": 15.0,
        "p95": 100.0,
        "max": 100.0,
    }


def test_summarize_latencies_ms_handles_empty_input() -> None:
    assert summarize_latencies_ms([]) == {
        "count": 0,
        "avg": None,
        "p50": None,
        "p95": None,
        "max": None,
    }
