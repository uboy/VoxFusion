"""Unit tests for discover_llm_endpoint() auto-detection."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from voxfusion.llm.client import _DISCOVER_CANDIDATES, discover_llm_endpoint

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_httpx_response(status_code: int) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    return resp


def _make_mock_client(responses: dict[str, int]) -> MagicMock:
    """Return a mock httpx.AsyncClient whose get() returns status codes by URL prefix."""

    async def _get(url: str, **_kwargs: object) -> MagicMock:
        for prefix, code in responses.items():
            if url.startswith(prefix):
                return _make_httpx_response(code)
        from httpx import ConnectError

        raise ConnectError("refused")

    client = MagicMock()
    client.get = _get
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    return client


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_discover_returns_none_when_no_server_responds() -> None:
    """All candidates refused → None."""
    import httpx

    async def _refusing_get(url: str, **_kwargs: object) -> MagicMock:
        raise httpx.ConnectError("refused")

    client = MagicMock()
    client.get = _refusing_get
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)

    with patch("voxfusion.llm.client.httpx.AsyncClient", return_value=client):
        result = await discover_llm_endpoint(candidates=_DISCOVER_CANDIDATES)

    assert result is None


@pytest.mark.asyncio
async def test_discover_returns_first_responding_url() -> None:
    """First candidate that gives 200 is returned."""
    candidates = ("http://localhost:9001", "http://localhost:9002")

    call_count = 0

    import httpx

    async def _get(url: str, **_kwargs: object) -> MagicMock:
        nonlocal call_count
        call_count += 1
        if "9001" in url:
            raise httpx.ConnectError("refused")
        return _make_httpx_response(200)

    client = MagicMock()
    client.get = _get
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)

    with patch("voxfusion.llm.client.httpx.AsyncClient", return_value=client):
        result = await discover_llm_endpoint(candidates=candidates)

    assert result == "http://localhost:9002"


@pytest.mark.asyncio
async def test_discover_accepts_auth_gated_server() -> None:
    """A 401 response still means a server is present — should be returned."""
    candidates = ("http://localhost:9003",)

    client = MagicMock()
    client.get = AsyncMock(return_value=_make_httpx_response(401))
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)

    with patch("voxfusion.llm.client.httpx.AsyncClient", return_value=client):
        result = await discover_llm_endpoint(candidates=candidates)

    assert result == "http://localhost:9003"


@pytest.mark.asyncio
async def test_discover_ignores_404_not_found() -> None:
    """404 means path missing, not a valid server — keep probing other paths."""

    candidates = ("http://localhost:9004",)

    # All paths return 404 → treated as not found
    client = MagicMock()
    client.get = AsyncMock(return_value=_make_httpx_response(404))
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)

    with patch("voxfusion.llm.client.httpx.AsyncClient", return_value=client):
        result = await discover_llm_endpoint(candidates=candidates)

    assert result is None


@pytest.mark.asyncio
async def test_discover_probes_all_model_paths_per_candidate() -> None:
    """Both /api/models and /api/tags are tried per candidate."""
    import httpx

    from voxfusion.llm.client import _MODEL_PATHS

    probed_paths: list[str] = []
    candidates = ("http://localhost:9005",)

    async def _get(url: str, **_kwargs: object) -> MagicMock:
        probed_paths.append(url)
        raise httpx.ConnectError("refused")

    client = MagicMock()
    client.get = _get
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)

    with patch("voxfusion.llm.client.httpx.AsyncClient", return_value=client):
        await discover_llm_endpoint(candidates=candidates)

    for path in _MODEL_PATHS:
        assert any(path in p for p in probed_paths), f"Path {path} was not probed"
