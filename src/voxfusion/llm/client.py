"""Open WebUI / OpenAI-compatible LLM API client.

Open WebUI exposes an OpenAI-compatible chat completions endpoint:

    POST {base_url}/api/chat/completions
    Authorization: Bearer <api_key>

Both streaming and non-streaming modes are supported.  Streaming uses the
standard Server-Sent Events (SSE) format from the OpenAI specification.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx

from voxfusion.logging import get_logger

log = get_logger(__name__)

# Default Open WebUI instance running locally
DEFAULT_BASE_URL = "http://localhost:3000"
DEFAULT_MODEL = "qwen2.5:32b"
_COMPLETIONS_PATH = "/api/chat/completions"
_MODEL_PATHS = ("/api/models", "/api/tags")
_TIMEOUT_CONNECT = 10.0   # seconds to establish connection
_TIMEOUT_READ = 300.0     # seconds to wait for first token / full response


class LLMError(Exception):
    """Raised when the LLM API returns an error or is unreachable."""


def _extract_model_ids(payload: object) -> list[str]:
    """Extract model identifiers from common Open WebUI response shapes."""
    candidates: list[str] = []
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

    for item in items:
        if not isinstance(item, dict):
            continue
        for key in ("id", "model", "name"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                candidates.append(value.strip())
                break
    return sorted(dict.fromkeys(candidates))


async def fetch_models(
    *,
    base_url: str = DEFAULT_BASE_URL,
    api_key: str = "",
) -> list[str]:
    """Fetch available model identifiers from Open WebUI."""
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    timeout = httpx.Timeout(connect=_TIMEOUT_CONNECT, read=30.0, write=30.0, pool=5.0)

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            for path in _MODEL_PATHS:
                url = base_url.rstrip("/") + path
                response = await client.get(url, headers=headers)
                if response.status_code == 404:
                    continue
                if response.status_code == 401:
                    raise LLMError("Authentication failed while loading models (HTTP 401).")
                if response.status_code != 200:
                    raise LLMError(
                        f"Failed to load models (HTTP {response.status_code}) from {url}."
                    )
                models = _extract_model_ids(response.json())
                if models:
                    return models
        raise LLMError("Open WebUI did not return any models.")
    except httpx.ConnectError as exc:
        raise LLMError(
            f"Cannot connect to Open WebUI at {base_url}.\n"
            "Make sure the server is running and the URL is correct."
        ) from exc
    except httpx.TimeoutException as exc:
        raise LLMError("Timed out while loading model list from Open WebUI.") from exc
    except ValueError as exc:
        raise LLMError("Open WebUI returned malformed model metadata.") from exc


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
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
    }

    log.info("llm.stream.start", url=url, model=model)
    timeout = httpx.Timeout(connect=_TIMEOUT_CONNECT, read=timeout_read, write=30.0, pool=5.0)

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream("POST", url, json=payload, headers=headers) as response:
                if response.status_code == 401:
                    raise LLMError(
                        "Authentication failed (HTTP 401). "
                        "Check your API key or Open WebUI authentication settings."
                    )
                if response.status_code == 404:
                    raise LLMError(
                        f"Endpoint not found (HTTP 404): {url}\n"
                        "Make sure Open WebUI is running and the URL is correct."
                    )
                if response.status_code != 200:
                    body = await response.aread()
                    raise LLMError(
                        f"HTTP {response.status_code}: {body.decode(errors='replace')[:300]}"
                    )

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
                            yield delta
                    except (json.JSONDecodeError, KeyError, IndexError):
                        # Malformed chunk — skip silently
                        continue

    except httpx.ConnectError as exc:
        raise LLMError(
            f"Cannot connect to Open WebUI at {base_url}.\n"
            "Make sure the server is running and the URL is correct."
        ) from exc
    except httpx.TimeoutException as exc:
        raise LLMError(
            f"Request timed out after {timeout_read}s. "
            "The model may still be loading — try again in a moment."
        ) from exc
    except LLMError:
        raise
    except Exception as exc:
        raise LLMError(f"Unexpected error during LLM request: {exc}") from exc

    log.info("llm.stream.done", model=model)


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
