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
        estimated_input_tokens=worker._estimate_messages_tokens(gui_runtime.build_messages("summarize", "Very long transcript")),
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
