"""Tests for prompt-building helpers used by GUI transcript post-processing."""

from voxfusion.llm.prompts import build_chunk_messages, build_merge_messages


def test_build_chunk_messages_adds_chunk_scope_prefix() -> None:
    messages = build_chunk_messages(
        "summarize",
        "[00:00:01] [SPEAKER_00] Hello",
        chunk_index=2,
        chunk_count=5,
    )

    assert messages[0]["role"] == "system"
    assert "chunk 2 of 5" in messages[1]["content"]
    assert "THIS CHUNK ONLY" in messages[1]["content"]
    assert "[00:00:01] [SPEAKER_00] Hello" in messages[1]["content"]


def test_build_merge_messages_references_partial_outputs() -> None:
    messages = build_merge_messages(
        "summarize",
        "### Partial 1\nTopic A\n\n### Partial 2\nTopic B",
    )

    assert messages[0]["role"] == "system"
    assert "Merge them into one final response" in messages[1]["content"]
    assert "### Partial 1" in messages[1]["content"]
    assert "Original task template:" in messages[1]["content"]
