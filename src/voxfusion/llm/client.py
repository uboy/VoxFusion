"""Open WebUI / OpenAI-compatible LLM API client.

Open WebUI exposes an OpenAI-compatible chat completions endpoint:

    POST {base_url}/api/chat/completions
    Authorization: Bearer <api_key>

Both streaming and non-streaming modes are supported.  Streaming uses the
standard Server-Sent Events (SSE) format from the OpenAI specification.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass

import httpx

from voxfusion.logging import get_logger

log = get_logger(__name__)

# Default Open WebUI instance running locally
DEFAULT_BASE_URL = "http://localhost:3000"
DEFAULT_MODEL = "qwen2.5:32b"
_COMPLETIONS_PATH = "/api/chat/completions"
_MODEL_PATHS = ("/api/models", "/api/tags")
_TIMEOUT_CONNECT = 10.0  # seconds to establish connection
_TIMEOUT_READ = 300.0  # seconds to wait for first token / full response
_TIMEOUT_READY_CHECK = 15.0
_MODEL_FETCH_RETRY_STATUSES = frozenset({429, 502, 503, 504})
_MODEL_FETCH_MAX_ATTEMPTS = 3
_MODEL_FETCH_RETRY_DELAY_S = 1.0


class LLMError(Exception):
    """Raised when the LLM API returns an error or is unreachable."""


def _body_preview(raw: str | bytes, limit: int = 300) -> str:
    """Return a compact preview for logging/error messages."""
    if isinstance(raw, bytes):
        text = raw.decode(errors="replace")
    else:
        text = raw
    return " ".join(text.split())[:limit]


def _message_stats(messages: list[dict[str, str]]) -> tuple[int, int]:
    """Return message count and total content length for safe diagnostics."""
    message_count = len(messages)
    input_chars = 0
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content", "")
        if isinstance(content, str):
            input_chars += len(content)
    return message_count, input_chars


@dataclass(frozen=True)
class LLMModelDescriptor:
    """Metadata for one model exposed by Open WebUI-compatible endpoints."""

    id: str
    context_tokens: int | None = None


_MODEL_CONTEXT_KEYS = frozenset(
    {
        "context_length",
        "context_window",
        "context_size",
        "context_tokens",
        "context_length_tokens",
        "max_context_length",
        "max_input_tokens",
        "max_sequence_length",
        "num_ctx",
        "n_ctx",
        "ctx_size",
    }
)
_MODEL_CONTEXT_CONTAINER_KEYS = (
    "details",
    "info",
    "metadata",
    "config",
    "capabilities",
    "parameters",
    "model_info",
    "model",
    "ollama",
)
_MODEL_CONTEXT_RE = re.compile(r"^\s*(?P<count>\d+)\s*(?P<suffix>[kKmM]?)\s*$")


def _extract_model_items(payload: object) -> list[dict[str, object]]:
    if isinstance(payload, dict):
        items = payload.get("data")
        if not isinstance(items, list):
            items = payload.get("models")
        if not isinstance(items, list):
            items = []
    elif isinstance(payload, list):
        items = payload
    else:
        items = []
    return [item for item in items if isinstance(item, dict)]


def _extract_model_id(item: dict[str, object]) -> str | None:
    for key in ("id", "model", "name"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _parse_context_tokens(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, float) and value.is_integer() and value > 0:
        return int(value)
    if not isinstance(value, str):
        return None
    match = _MODEL_CONTEXT_RE.match(value)
    if not match:
        return None
    count = int(match.group("count"))
    suffix = match.group("suffix").lower()
    if suffix == "k":
        count *= 1024
    elif suffix == "m":
        count *= 1024 * 1024
    return count if count > 0 else None


def _extract_context_tokens(value: object, *, depth: int = 0) -> int | None:
    if depth > 4:
        return None
    if isinstance(value, dict):
        direct_match: int | None = None
        for key, candidate in value.items():
            if str(key).strip().lower() in _MODEL_CONTEXT_KEYS:
                parsed = _parse_context_tokens(candidate)
                if parsed is not None:
                    direct_match = parsed
                    break
        if direct_match is not None:
            return direct_match
        for key in _MODEL_CONTEXT_CONTAINER_KEYS:
            if key in value:
                parsed = _extract_context_tokens(value[key], depth=depth + 1)
                if parsed is not None:
                    return parsed
        for nested in value.values():
            parsed = _extract_context_tokens(nested, depth=depth + 1)
            if parsed is not None:
                return parsed
        return None
    if isinstance(value, list):
        for nested in value:
            parsed = _extract_context_tokens(nested, depth=depth + 1)
            if parsed is not None:
                return parsed
    return None


def extract_model_catalog(payload: object) -> list[LLMModelDescriptor]:
    """Extract model ids and optional context limits from common Open WebUI payloads."""
    catalog: dict[str, LLMModelDescriptor] = {}
    for item in _extract_model_items(payload):
        model_id = _extract_model_id(item)
        if not model_id:
            continue
        context_tokens = _extract_context_tokens(item)
        existing = catalog.get(model_id)
        if existing is None or (existing.context_tokens is None and context_tokens is not None):
            catalog[model_id] = LLMModelDescriptor(id=model_id, context_tokens=context_tokens)
    return [catalog[key] for key in sorted(catalog)]


def _extract_model_ids(payload: object) -> list[str]:
    """Extract model identifiers from common Open WebUI response shapes."""
    return [descriptor.id for descriptor in extract_model_catalog(payload)]


async def fetch_model_catalog(
    *,
    base_url: str = DEFAULT_BASE_URL,
    api_key: str = "",
) -> list[LLMModelDescriptor]:
    """Fetch available models and any discoverable context metadata from Open WebUI."""
    api_key_present = bool(api_key)
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    timeout = httpx.Timeout(connect=_TIMEOUT_CONNECT, read=30.0, write=30.0, pool=5.0)
    log.info(
        "llm.models_fetch.start",
        base_url=base_url,
        api_key_present=api_key_present,
    )

    last_error_message: str | None = None

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            for path in _MODEL_PATHS:
                url = base_url.rstrip("/") + path
                for attempt in range(1, _MODEL_FETCH_MAX_ATTEMPTS + 1):
                    response = await client.get(url, headers=headers)
                    if response.status_code == 404:
                        log.debug("llm.models_fetch.path_missing", url=url)
                        break
                    if response.status_code == 401:
                        log.error(
                            "llm.models_fetch.http_error",
                            base_url=base_url,
                            url=url,
                            status_code=response.status_code,
                            body_preview=_body_preview(response.text),
                            api_key_present=api_key_present,
                        )
                        raise LLMError("Authentication failed while loading models (HTTP 401).")
                    if (
                        response.status_code in _MODEL_FETCH_RETRY_STATUSES
                        and attempt < _MODEL_FETCH_MAX_ATTEMPTS
                    ):
                        retry_in_s = _MODEL_FETCH_RETRY_DELAY_S * attempt
                        log.warning(
                            "llm.models_fetch.retry",
                            base_url=base_url,
                            url=url,
                            status_code=response.status_code,
                            attempt=attempt,
                            max_attempts=_MODEL_FETCH_MAX_ATTEMPTS,
                            retry_in_s=retry_in_s,
                            api_key_present=api_key_present,
                        )
                        await asyncio.sleep(retry_in_s)
                        continue
                    if response.status_code != 200:
                        log.error(
                            "llm.models_fetch.http_error",
                            base_url=base_url,
                            url=url,
                            status_code=response.status_code,
                            body_preview=_body_preview(response.text),
                            api_key_present=api_key_present,
                        )
                        last_error_message = (
                            f"Failed to load models (HTTP {response.status_code}) from {url}."
                        )
                        break
                    try:
                        models = extract_model_catalog(response.json())
                    except ValueError as exc:
                        log.warning(
                            "llm.models_fetch.malformed_path",
                            base_url=base_url,
                            url=url,
                            api_key_present=api_key_present,
                            error=str(exc),
                        )
                        last_error_message = "Open WebUI returned malformed model metadata."
                        break
                    if models:
                        log.info(
                            "llm.models_fetch.done",
                            base_url=base_url,
                            url=url,
                            model_count=len(models),
                            first_model=models[0].id,
                            api_key_present=api_key_present,
                        )
                        return models
                    log.warning(
                        "llm.models_fetch.empty_path",
                        base_url=base_url,
                        url=url,
                        api_key_present=api_key_present,
                    )
                    break
        if last_error_message is not None:
            raise LLMError(last_error_message)
        log.error(
            "llm.models_fetch.empty",
            base_url=base_url,
            api_key_present=api_key_present,
        )
        raise LLMError("Open WebUI did not return any models.")
    except httpx.ConnectError as exc:
        log.error(
            "llm.models_fetch.connect_error",
            base_url=base_url,
            api_key_present=api_key_present,
            error=str(exc),
        )
        raise LLMError(
            f"Cannot connect to Open WebUI at {base_url}.\n"
            "Make sure the server is running and the URL is correct."
        ) from exc
    except httpx.TimeoutException as exc:
        log.error(
            "llm.models_fetch.timeout",
            base_url=base_url,
            api_key_present=api_key_present,
            error=str(exc),
        )
        raise LLMError("Timed out while loading model list from Open WebUI.") from exc


async def fetch_models(
    *,
    base_url: str = DEFAULT_BASE_URL,
    api_key: str = "",
) -> list[str]:
    """Fetch available model identifiers from Open WebUI."""
    return [
        descriptor.id
        for descriptor in await fetch_model_catalog(base_url=base_url, api_key=api_key)
    ]


async def stream_completion(
    messages: list[dict[str, str]],
    *,
    base_url: str = DEFAULT_BASE_URL,
    model: str = DEFAULT_MODEL,
    api_key: str = "",
    timeout_read: float = _TIMEOUT_READ,
) -> AsyncIterator[str]:
    """Yield text tokens from an Open WebUI streaming chat completion.

    Args:
        messages: OpenAI-style message list (``[{"role": ..., "content": ...}]``).
        base_url: Base URL of the Open WebUI instance.
        model: Model identifier as shown in Open WebUI (e.g. ``"qwen2.5:32b"``).
        api_key: Optional bearer token / API key.
        timeout_read: Seconds to wait for each SSE chunk before timing out.

    Yields:
        Text delta strings from the model as they arrive.

    Raises:
        LLMError: On connection failure, HTTP error, or malformed response.
    """
    url = base_url.rstrip("/") + _COMPLETIONS_PATH
    api_key_present = bool(api_key)
    message_count, input_chars = _message_stats(messages)
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
    }

    log.info(
        "llm.request.start",
        url=url,
        model=model,
        message_count=message_count,
        input_chars=input_chars,
        timeout_read=timeout_read,
        api_key_present=api_key_present,
    )
    timeout = httpx.Timeout(connect=_TIMEOUT_CONNECT, read=timeout_read, write=30.0, pool=5.0)
    chunk_count = 0
    output_chars = 0

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream("POST", url, json=payload, headers=headers) as response:
                if response.status_code == 401:
                    body = await response.aread()
                    log.error(
                        "llm.request.http_error",
                        url=url,
                        model=model,
                        status_code=response.status_code,
                        body_preview=_body_preview(body),
                        message_count=message_count,
                        input_chars=input_chars,
                    )
                    raise LLMError(
                        "Authentication failed (HTTP 401). "
                        "Check your API key or Open WebUI authentication settings."
                    )
                if response.status_code == 404:
                    body = await response.aread()
                    log.error(
                        "llm.request.http_error",
                        url=url,
                        model=model,
                        status_code=response.status_code,
                        body_preview=_body_preview(body),
                        message_count=message_count,
                        input_chars=input_chars,
                    )
                    raise LLMError(
                        f"Endpoint not found (HTTP 404): {url}\n"
                        "Make sure Open WebUI is running and the URL is correct."
                    )
                if response.status_code != 200:
                    body = await response.aread()
                    log.error(
                        "llm.request.http_error",
                        url=url,
                        model=model,
                        status_code=response.status_code,
                        body_preview=_body_preview(body),
                        message_count=message_count,
                        input_chars=input_chars,
                    )
                    raise LLMError(f"HTTP {response.status_code}: {_body_preview(body)}")

                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[6:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        delta = chunk["choices"][0]["delta"].get("content") or ""
                        if delta:
                            chunk_count += 1
                            output_chars += len(delta)
                            yield delta
                    except (json.JSONDecodeError, KeyError, IndexError):
                        # Malformed chunk — skip silently
                        continue

    except httpx.ConnectError as exc:
        log.error(
            "llm.request.connect_error",
            url=url,
            model=model,
            message_count=message_count,
            input_chars=input_chars,
            error=str(exc),
        )
        raise LLMError(
            f"Cannot connect to Open WebUI at {base_url}.\n"
            "Make sure the server is running and the URL is correct."
        ) from exc
    except httpx.TimeoutException as exc:
        log.error(
            "llm.request.timeout",
            url=url,
            model=model,
            message_count=message_count,
            input_chars=input_chars,
            timeout_read=timeout_read,
            error=str(exc),
        )
        raise LLMError(
            f"Request timed out after {timeout_read}s. "
            "The model may still be loading — try again in a moment."
        ) from exc
    except LLMError:
        raise
    except Exception as exc:
        log.exception(
            "llm.request.unexpected_error",
            url=url,
            model=model,
            message_count=message_count,
            input_chars=input_chars,
        )
        raise LLMError(f"Unexpected error during LLM request: {exc}") from exc

    log.info(
        "llm.request.done",
        url=url,
        model=model,
        message_count=message_count,
        input_chars=input_chars,
        chunk_count=chunk_count,
        output_chars=output_chars,
    )


async def complete(
    messages: list[dict[str, str]],
    *,
    base_url: str = DEFAULT_BASE_URL,
    model: str = DEFAULT_MODEL,
    api_key: str = "",
    timeout_read: float = _TIMEOUT_READ,
) -> str:
    """Return the full LLM response as a single string (non-streaming).

    Internally uses streaming and concatenates all tokens.
    """
    parts: list[str] = []
    async for token in stream_completion(
        messages,
        base_url=base_url,
        model=model,
        api_key=api_key,
        timeout_read=timeout_read,
    ):
        parts.append(token)
    return "".join(parts)


async def verify_model_ready(
    *,
    base_url: str = DEFAULT_BASE_URL,
    model: str = DEFAULT_MODEL,
    api_key: str = "",
    timeout_read: float = _TIMEOUT_READY_CHECK,
) -> None:
    """Run a tiny request to verify the API and selected model are responsive."""
    await complete(
        [{"role": "user", "content": "Reply with OK and nothing else."}],
        base_url=base_url,
        model=model,
        api_key=api_key,
        timeout_read=timeout_read,
    )
