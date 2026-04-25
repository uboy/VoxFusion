"""Focused logging tests for the Open WebUI client."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock

import pytest

import voxfusion.llm.client as llm_client


class _FakeResponse:
    def __init__(
        self,
        *,
        status_code: int,
        json_payload: object | None = None,
        body_text: str = "",
        lines: list[str] | None = None,
    ) -> None:
        self.status_code = status_code
        self._json_payload = json_payload
        self._body_text = body_text
        self._lines = lines or []

    async def __aenter__(self) -> _FakeResponse:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        del exc_type, exc, tb

    def json(self) -> object:
        if isinstance(self._json_payload, Exception):
            raise self._json_payload
        return self._json_payload

    @property
    def text(self) -> str:
        return self._body_text

    async def aread(self) -> bytes:
        return self._body_text.encode("utf-8")

    async def aiter_lines(self):
        for line in self._lines:
            yield line


class _FakeAsyncClient:
    def __init__(
        self,
        *,
        get_response: _FakeResponse | list[_FakeResponse] | None = None,
        stream_response: _FakeResponse | None = None,
        timeout: object | None = None,
        **_: object,
    ) -> None:
        del timeout
        if isinstance(get_response, list):
            self._get_responses = list(get_response)
        elif get_response is None:
            self._get_responses = []
        else:
            self._get_responses = [get_response]
        self._stream_response = stream_response

    async def __aenter__(self) -> _FakeAsyncClient:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        del exc_type, exc, tb

    async def get(self, url: str, headers: dict[str, str] | None = None) -> _FakeResponse:
        del url, headers
        assert self._get_responses
        return self._get_responses.pop(0)

    def stream(
        self,
        method: str,
        url: str,
        json: dict[str, object] | None = None,
        headers: dict[str, str] | None = None,
    ) -> _FakeResponse:
        del method, url, json, headers
        assert self._stream_response is not None
        return self._stream_response


def test_fetch_models_retries_transient_http_error(monkeypatch) -> None:
    fake_log = MagicMock()
    responses = [
        _FakeResponse(status_code=503, body_text='{"detail":"backend unavailable"}'),
        _FakeResponse(status_code=200, json_payload={"data": [{"id": "qwen2.5-7b"}]}),
    ]

    async def _sleep(_: float) -> None:
        return None

    monkeypatch.setattr(llm_client, "log", fake_log)
    monkeypatch.setattr(llm_client.asyncio, "sleep", _sleep)
    monkeypatch.setattr(
        llm_client.httpx,
        "AsyncClient",
        lambda timeout=None, **kwargs: _FakeAsyncClient(
            get_response=responses, timeout=timeout, **kwargs
        ),
    )

    models = asyncio.run(
        llm_client.fetch_models(base_url="http://openwebui:3000", api_key="secret")
    )

    assert models == ["qwen2.5-7b"]
    fake_log.info.assert_any_call(
        "llm.models_fetch.start",
        base_url="http://openwebui:3000",
        api_key_present=True,
    )
    fake_log.warning.assert_any_call(
        "llm.models_fetch.retry",
        base_url="http://openwebui:3000",
        url="http://openwebui:3000/api/models",
        status_code=503,
        attempt=1,
        max_attempts=llm_client._MODEL_FETCH_MAX_ATTEMPTS,
        retry_in_s=llm_client._MODEL_FETCH_RETRY_DELAY_S,
        api_key_present=True,
    )
    fake_log.info.assert_any_call(
        "llm.models_fetch.done",
        base_url="http://openwebui:3000",
        url="http://openwebui:3000/api/models",
        model_count=1,
        first_model="qwen2.5-7b",
        api_key_present=True,
    )


def test_fetch_models_logs_http_error_after_retries(monkeypatch) -> None:
    fake_log = MagicMock()
    responses = [
        _FakeResponse(status_code=503, body_text='{"detail":"backend unavailable"}'),
        _FakeResponse(status_code=503, body_text='{"detail":"backend unavailable"}'),
        _FakeResponse(status_code=503, body_text='{"detail":"backend unavailable"}'),
        _FakeResponse(status_code=404, body_text='{"detail":"missing"}'),
    ]

    async def _sleep(_: float) -> None:
        return None

    monkeypatch.setattr(llm_client, "log", fake_log)
    monkeypatch.setattr(llm_client.asyncio, "sleep", _sleep)
    monkeypatch.setattr(
        llm_client.httpx,
        "AsyncClient",
        lambda timeout=None, **kwargs: _FakeAsyncClient(
            get_response=responses, timeout=timeout, **kwargs
        ),
    )

    with pytest.raises(llm_client.LLMError, match="HTTP 503"):
        asyncio.run(llm_client.fetch_models(base_url="http://openwebui:3000", api_key="secret"))

    fake_log.error.assert_any_call(
        "llm.models_fetch.http_error",
        base_url="http://openwebui:3000",
        url="http://openwebui:3000/api/models",
        status_code=503,
        body_preview='{"detail":"backend unavailable"}',
        api_key_present=True,
    )


def test_stream_completion_logs_start_and_done(monkeypatch) -> None:
    fake_log = MagicMock()
    body = json.dumps({"choices": [{"delta": {"content": "Hello"}}]})
    response = _FakeResponse(status_code=200, lines=[f"data: {body}", "data: [DONE]"])

    monkeypatch.setattr(llm_client, "log", fake_log)
    monkeypatch.setattr(
        llm_client.httpx,
        "AsyncClient",
        lambda timeout=None, **kwargs: _FakeAsyncClient(
            stream_response=response, timeout=timeout, **kwargs
        ),
    )

    async def _collect() -> list[str]:
        tokens: list[str] = []
        async for token in llm_client.stream_completion(
            [{"role": "user", "content": "Hi"}],
            base_url="http://openwebui:3000",
            model="llama3.2:3b",
        ):
            tokens.append(token)
        return tokens

    assert asyncio.run(_collect()) == ["Hello"]

    fake_log.info.assert_any_call(
        "llm.request.start",
        url="http://openwebui:3000/api/chat/completions",
        model="llama3.2:3b",
        message_count=1,
        input_chars=2,
        timeout_read=llm_client._TIMEOUT_READ,
        api_key_present=False,
    )
    fake_log.info.assert_any_call(
        "llm.request.done",
        url="http://openwebui:3000/api/chat/completions",
        model="llama3.2:3b",
        message_count=1,
        input_chars=2,
        chunk_count=1,
        output_chars=5,
    )


def test_stream_completion_logs_http_error(monkeypatch) -> None:
    fake_log = MagicMock()
    response = _FakeResponse(status_code=503, body_text='{"detail":"upstream overloaded"}')

    monkeypatch.setattr(llm_client, "log", fake_log)
    monkeypatch.setattr(
        llm_client.httpx,
        "AsyncClient",
        lambda timeout=None, **kwargs: _FakeAsyncClient(
            stream_response=response, timeout=timeout, **kwargs
        ),
    )

    async def _consume() -> None:
        async for _ in llm_client.stream_completion(
            [{"role": "user", "content": "hello"}],
            base_url="http://openwebui:3000",
            model="qwen3:32b",
        ):
            pass

    with pytest.raises(llm_client.LLMError, match="HTTP 503"):
        asyncio.run(_consume())

    fake_log.error.assert_any_call(
        "llm.request.http_error",
        url="http://openwebui:3000/api/chat/completions",
        model="qwen3:32b",
        status_code=503,
        body_preview='{"detail":"upstream overloaded"}',
        message_count=1,
        input_chars=5,
    )
