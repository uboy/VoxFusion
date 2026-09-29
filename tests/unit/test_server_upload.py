"""Unit tests for server.app._store_upload upload handling.

Covers the cancellation path: a client disconnect / timeout mid-upload
raises asyncio.CancelledError (BaseException, not caught by
``except Exception``) - the partial file must still be removed.
"""

import asyncio
from pathlib import Path

import pytest
from server.app import _store_upload


class _FakeUpload:
    """Minimal stand-in for starlette UploadFile (async .read only)."""

    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    async def read(self, size: int) -> bytes:
        if not self._chunks:
            return b""
        chunk = self._chunks.pop(0)
        if chunk is None:
            raise asyncio.CancelledError()
        return chunk


def test_store_upload_writes_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("server.app.UPLOAD_DIR", tmp_path)
    dest, size = asyncio.run(_store_upload("jobABC", "meeting.wav", _FakeUpload([b"foo", b"bar"])))
    assert dest == tmp_path / "jobABC_meeting.wav"
    assert dest.read_bytes() == b"foobar"
    assert size == 6


def test_store_upload_cancelled_removes_partial_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("server.app.UPLOAD_DIR", tmp_path)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(_store_upload("jobXYZ", "meeting.wav", _FakeUpload([b"partial", None])))
    assert list(tmp_path.iterdir()) == []
