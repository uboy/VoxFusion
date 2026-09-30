"""Unit tests for the diarization model env and per-job speaker hints."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from server.app import (
    DEFAULT_DIARIZATION_MODEL,
    _parse_speaker_hint,
    _register_job,
    _resolve_diarization_model,
)
from server.app import (
    app as fastapi_app,
)
from server.worker import TranscribeJob

from voxfusion.config.loader import load_config
from voxfusion.diarization.pyannote_engine import PyAnnoteDiarizer

AUTH = {"Authorization": "Bearer test-token-123"}


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr("server.app.TOKEN", "test-token-123")
    return TestClient(fastapi_app)


# --- Model selection -----------------------------------------------------


def test_model_env_blank_keeps_default() -> None:
    assert _resolve_diarization_model(None) == DEFAULT_DIARIZATION_MODEL
    assert _resolve_diarization_model("  ") == DEFAULT_DIARIZATION_MODEL
    assert _resolve_diarization_model("pyannote/speaker-diarization-community-1") == (
        "pyannote/speaker-diarization-community-1"
    )


def test_model_override_reaches_config() -> None:
    config = load_config(
        overrides={"diarization": {"ml": {"model": "pyannote/speaker-diarization-community-1"}}}
    )
    assert config.diarization.ml.model == "pyannote/speaker-diarization-community-1"
    assert config.diarization.strategy == "channel"  # untouched keys stay default


# --- Speaker hint parsing -------------------------------------------------


def test_parse_speaker_hint_blank_is_none() -> None:
    assert _parse_speaker_hint(None, "min_speakers") is None
    assert _parse_speaker_hint("", "min_speakers") is None
    assert _parse_speaker_hint(" 3 ", "min_speakers") == 3


def test_parse_speaker_hint_rejects_garbage() -> None:
    with pytest.raises(HTTPException) as exc:
        _parse_speaker_hint("two", "min_speakers")
    assert exc.value.status_code == 400
    with pytest.raises(HTTPException) as exc:
        _parse_speaker_hint("0", "max_speakers")
    assert exc.value.status_code == 400


def test_transcribe_rejects_min_gt_max(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("server.app.UPLOAD_DIR", tmp_path)
    response = client.post(
        "/v1/transcribe",
        headers=AUTH,
        files={"file": ("a.wav", b"xx")},
        data={"min_speakers": "3", "max_speakers": "2"},
    )
    assert response.status_code == 400
    assert "min_speakers" in response.json()["detail"]


# --- Per-job plumbing ------------------------------------------------------


def test_engine_shares_the_config_object() -> None:
    """Per-job mutation of config.diarization.ml must reach the engine."""
    config = load_config()
    diarizer = PyAnnoteDiarizer(config.diarization.ml)
    assert diarizer._config is config.diarization.ml
    config.diarization.ml.min_speakers = 2
    config.diarization.ml.max_speakers = 2
    assert diarizer._config.min_speakers == 2
    assert diarizer._config.max_speakers == 2


def test_job_carries_speaker_hints() -> None:
    job = TranscribeJob(
        job_id="j2",
        file_path=Path("x"),
        original_name="x",
        min_speakers=2,
        max_speakers=3,
    )
    assert job.min_speakers == 2 and job.max_speakers == 3
    _register_job(job)
