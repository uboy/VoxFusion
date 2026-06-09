"""Unit tests for GigaAM chunk temp-file directory selection and seam deduplication."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from voxfusion.asr.gigaam_engine import GigaAMCTCEngine, _dedup_seam
from voxfusion.config.models import ASRConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_engine() -> GigaAMCTCEngine:
    return GigaAMCTCEngine(ASRConfig(model_size="gigaam-v3-e2e-ctc"))


def _make_fake_model(transcribe_return: str = "hello") -> MagicMock:
    model = MagicMock()
    model.transcribe.return_value = transcribe_return
    return model


def _long_audio(seconds: int = 30) -> np.ndarray:
    """Sine wave long enough to produce at least one 24-second chunk."""
    sr = 16000
    t = np.linspace(0, seconds, seconds * sr, dtype=np.float32)
    return (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_tmpdir_is_dev_shm_on_linux(monkeypatch: pytest.MonkeyPatch) -> None:
    """/dev/shm used as tempfile dir on Linux when available."""
    monkeypatch.setattr(sys, "platform", "linux")

    tmpfiles_created: list[dict[str, object]] = []
    original_ntf = __import__("tempfile").NamedTemporaryFile

    def _capture_ntf(**kwargs: object) -> object:
        tmpfiles_created.append(kwargs)
        return original_ntf(**kwargs)

    engine = _make_engine()
    engine._model = _make_fake_model("привет мир")

    with (
        patch("tempfile.NamedTemporaryFile", side_effect=_capture_ntf),
        patch("os.path.isdir", return_value=True),
    ):
        engine._transcribe_sync(_long_audio(30), language="ru")

    assert tmpfiles_created, "Expected at least one NamedTemporaryFile call"
    assert all(kw.get("dir") == "/dev/shm" for kw in tmpfiles_created), (
        f"Expected all temp files in /dev/shm, got dirs: {[kw.get('dir') for kw in tmpfiles_created]}"
    )


def test_tmpdir_is_none_on_linux_without_dev_shm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Falls back to default tempdir when /dev/shm is absent."""
    monkeypatch.setattr(sys, "platform", "linux")

    tmpfiles_created: list[dict[str, object]] = []
    original_ntf = __import__("tempfile").NamedTemporaryFile

    def _capture_ntf(**kwargs: object) -> object:
        tmpfiles_created.append(kwargs)
        return original_ntf(**kwargs)

    engine = _make_engine()
    engine._model = _make_fake_model("привет мир")

    with (
        patch("tempfile.NamedTemporaryFile", side_effect=_capture_ntf),
        patch("os.path.isdir", return_value=False),
    ):
        engine._transcribe_sync(_long_audio(30), language="ru")

    assert tmpfiles_created, "Expected at least one NamedTemporaryFile call"
    assert all(kw.get("dir") is None for kw in tmpfiles_created), (
        f"Expected dir=None, got: {[kw.get('dir') for kw in tmpfiles_created]}"
    )


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-specific tempdir behavior")
def test_tmpdir_is_none_on_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    """Windows always uses default tempdir regardless of /dev/shm.

    soundfile's sf_wchar_open is unavailable on Linux, so sf.write is also
    mocked to avoid hitting platform-specific libsndfile internals.
    """
    monkeypatch.setattr(sys, "platform", "win32")

    tmpfiles_created: list[dict[str, object]] = []
    original_ntf = __import__("tempfile").NamedTemporaryFile

    def _capture_ntf(**kwargs: object) -> object:
        tmpfiles_created.append(kwargs)
        return original_ntf(**kwargs)

    engine = _make_engine()
    engine._model = _make_fake_model("привет мир")

    with (
        patch("tempfile.NamedTemporaryFile", side_effect=_capture_ntf),
        patch("voxfusion.asr.gigaam_engine.sf.write"),  # avoid sf_wchar_open on Linux
    ):
        engine._transcribe_sync(_long_audio(30), language="ru")

    assert tmpfiles_created
    assert all(kw.get("dir") is None for kw in tmpfiles_created)


# ---------------------------------------------------------------------------
# Seam deduplication
# ---------------------------------------------------------------------------


def test_dedup_seam_removes_overlapping_words() -> None:
    prev = "всем доброе утро и тем кто будет смотреть в записи."
    curr = "смотреть в записи. Делюсь экраном."
    result, removed = _dedup_seam(prev, curr)
    assert result == "Делюсь экраном."
    assert removed == 3


def test_dedup_seam_no_overlap_returns_curr_unchanged() -> None:
    prev = "первая часть текста."
    curr = "вторая часть текста."
    result, removed = _dedup_seam(prev, curr)
    assert result == curr
    assert removed == 0


def test_dedup_seam_exact_duplicate_returns_empty() -> None:
    text = "занятие."
    result, removed = _dedup_seam(text, text)
    assert result == ""
    assert removed == 1


def test_dedup_seam_single_word_overlap() -> None:
    prev = "заканчивается занятие."
    curr = "занятие. И сегодня мы будем."
    result, removed = _dedup_seam(prev, curr)
    assert result == "И сегодня мы будем."
    assert removed == 1


def test_dedup_seam_empty_strings() -> None:
    result, removed = _dedup_seam("", "hello world")
    assert result == "hello world"
    assert removed == 0
    result, removed = _dedup_seam("hello world", "")
    assert result == ""
    assert removed == 0
