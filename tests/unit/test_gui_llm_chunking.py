"""Tests for chunked GUI LLM summarization when transcript size exceeds model context."""

from __future__ import annotations

import asyncio
import importlib
from unittest.mock import MagicMock

import voxfusion.gui.runtime as gui_runtime
from voxfusion.gui.runtime import LLMWorker

runtime = importlib.import_module("voxfusion.gui.runtime")


async def _raise_context_error(*args, **kwargs):
    del args, kwargs
    raise runtime.LLMError(
        "HTTP 400: This model's maximum context length is 2048 tokens. Please reduce the length of the input prompt."
    )
    yield ""  # pragma: no cover


def test_llm_worker_uses_chunked_path_when_context_is_estimated_too_large(monkeypatch) -> None:
    tokens: list[str] = []
    errors: list[str] = []
    complete_outputs = iter(["chunk summary 1", "chunk summary 2", "final merged summary"])
    fake_log = MagicMock()

    async def _fake_complete(messages, *, base_url, model, api_key, timeout_read=None):
        del messages, base_url, model, api_key, timeout_read
        return next(complete_outputs)

    worker = LLMWorker(
        text="Very long transcript",
        model="qwen2.5-7b",
        base_url="http://openwebui:3000",
        api_key="secret",
        prompt_name="summarize",
        custom_user_prompt=None,
        context_limit_tokens=None,
        on_token=tokens.append,
        on_error=errors.append,
        on_finished=lambda: None,
    )
    worker._needs_chunked_summary = lambda messages: True  # type: ignore[method-assign]
    worker._split_transcript_into_chunks = lambda: ["chunk one", "chunk two"]  # type: ignore[method-assign]

    monkeypatch.setattr(gui_runtime, "complete", _fake_complete)
    monkeypatch.setattr(gui_runtime, "log", fake_log)

    asyncio.run(worker._run_async())

    assert tokens == ["final merged summary"]
    assert errors == []
    fake_log.info.assert_any_call(
        "llm.chunking.plan",
        model="qwen2.5-7b",
        reason="estimated_context",
        transcript_chars=len("Very long transcript"),
        estimated_input_tokens=worker._estimate_messages_tokens(
            gui_runtime.build_messages("summarize", "Very long transcript")
        ),
        context_tokens=worker._context_limit_tokens(),
        chunk_count=2,
    )


def test_llm_worker_retries_with_chunking_after_context_error(monkeypatch) -> None:
    tokens: list[str] = []
    errors: list[str] = []
    complete_outputs = iter(["chunk summary 1", "chunk summary 2", "final merged summary"])
    fake_log = MagicMock()

    async def _fake_complete(messages, *, base_url, model, api_key, timeout_read=None):
        del messages, base_url, model, api_key, timeout_read
        return next(complete_outputs)

    worker = LLMWorker(
        text="Transcript that will trigger context retry",
        model="qwen2.5-7b",
        base_url="http://openwebui:3000",
        api_key="secret",
        prompt_name="summarize",
        custom_user_prompt=None,
        context_limit_tokens=None,
        on_token=tokens.append,
        on_error=errors.append,
        on_finished=lambda: None,
    )
    worker._needs_chunked_summary = lambda messages: False  # type: ignore[method-assign]
    worker._split_transcript_into_chunks = lambda: ["chunk one", "chunk two"]  # type: ignore[method-assign]

    monkeypatch.setattr(gui_runtime, "stream_completion", _raise_context_error)
    monkeypatch.setattr(gui_runtime, "complete", _fake_complete)
    monkeypatch.setattr(gui_runtime, "log", fake_log)

    asyncio.run(worker._run_async())

    assert tokens == ["final merged summary"]
    assert errors == []
    fake_log.warning.assert_any_call(
        "llm.chunking.context_retry",
        model="qwen2.5-7b",
        reason="context_error",
        error="HTTP 400: This model's maximum context length is 2048 tokens. Please reduce the length of the input prompt.",
    )


def test_llm_worker_prefers_explicit_context_limit() -> None:
    worker = LLMWorker(
        text="short transcript",
        model="qwen2.5-7b",
        base_url="http://openwebui:3000",
        api_key="secret",
        prompt_name="summarize",
        custom_user_prompt=None,
        context_limit_tokens=32768,
        on_token=lambda _token: None,
        on_error=lambda _message: None,
        on_finished=lambda: None,
    )

    assert worker._context_limit_tokens() == 32768


def test_estimate_text_tokens_utf8_based() -> None:
    """Token estimation must use UTF-8 byte length / 4, not raw char count / 2.

    For Russian text (2 bytes per Cyrillic char), the old char/2 formula gave the
    same result as bytes/4, so this test targets a multi-byte emoji (4 bytes each)
    and an ASCII string to confirm both directions are correct.
    """
    # 4 ASCII chars → 4 UTF-8 bytes → ceil(4/4) = 1 token
    assert LLMWorker._estimate_text_tokens("test") == 1
    # 4 Cyrillic chars → 8 UTF-8 bytes → ceil(8/4) = 2 tokens
    # Old formula (len/2): ceil(4/2) = 2 — same result for Russian; caught by emoji test
    assert LLMWorker._estimate_text_tokens("тест") == 2
    # 1 emoji (🎙) → 4 UTF-8 bytes → ceil(4/4) = 1 token
    # Old formula (len/2): ceil(1/2) = 1 — still same; test longer string:
    # 8 emoji → 32 UTF-8 bytes → ceil(32/4) = 8 tokens
    # Old formula (len/2): ceil(8/2) = 4 tokens  ← this is what we fixed
    assert LLMWorker._estimate_text_tokens("🎙" * 8) == 8
    # Empty string → 0
    assert LLMWorker._estimate_text_tokens("") == 0
    assert LLMWorker._estimate_text_tokens("   ") == 1
