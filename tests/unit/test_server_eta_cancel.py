"""Unit tests for API ETA estimate and job cancellation."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from server.app import (
    _estimate_eta,
    _probe_duration,
    _register_job,
)
from server.app import (
    app as fastapi_app,
)
from server.worker import (
    STATUS_CANCELLED,
    STATUS_RUNNING,
    JobCancelledError,
    TranscribeJob,
    TranscriptionWorker,
)

AUTH = {"Authorization": "Bearer test-token-123"}


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr("server.app.TOKEN", "test-token-123")
    return TestClient(fastapi_app)


def _make_job(tmp_path: Path, job_id: str = "jobCAN1") -> TranscribeJob:
    path = tmp_path / f"{job_id}_meeting.wav"
    path.write_bytes(b"fake audio")
    job = TranscribeJob(
        job_id=job_id,
        file_path=path,
        original_name="meeting.wav",
        size_bytes=9,
    )
    _register_job(job)
    return job


# --- ETA ---------------------------------------------------------------


def test_estimate_eta_none_without_duration() -> None:
    assert _estimate_eta(None) is None


def test_estimate_eta_applies_multiplier() -> None:
    assert _estimate_eta(600.0) == 1080.0  # 600 * 1.8, default multiplier
    assert _estimate_eta(1.0) == 2.0  # rounded up


def test_probe_duration_parses_ffprobe_json() -> None:
    payload = json.dumps({"format": {"duration": "12.5"}}).encode()

    def fake_run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess([], 0, stdout=payload, stderr=b"")

    with patch("server.app.subprocess.run", fake_run):
        assert _probe_duration(Path("x.m4a")) == 12.5


def test_probe_duration_degrades_to_none_on_failure() -> None:
    def fake_run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess([], 1, stdout=b"", stderr=b"boom")

    with patch("server.app.subprocess.run", fake_run):
        assert _probe_duration(Path("x.m4a")) is None
    with patch("server.app.subprocess.run", side_effect=OSError("no ffprobe")):
        assert _probe_duration(Path("x.m4a")) is None


# --- Cancel endpoint ---------------------------------------------------


def test_cancel_requires_auth(client: TestClient) -> None:
    response = client.post("/v1/jobs/whatever/cancel")
    assert response.status_code == 401


def test_cancel_unknown_job_is_404(client: TestClient) -> None:
    response = client.post("/v1/jobs/nosuchjob/cancel", headers=AUTH)
    assert response.status_code == 404


def test_cancel_queued_job_cancels_and_removes_file(client: TestClient, tmp_path: Path) -> None:
    job = _make_job(tmp_path)
    assert job.file_path.exists()

    response = client.post(f"/v1/jobs/{job.job_id}/cancel", headers=AUTH)

    assert response.status_code == 200
    assert response.json() == {"job_id": job.job_id, "status": STATUS_CANCELLED}
    assert job.status == STATUS_CANCELLED
    assert not job.file_path.exists()


def test_cancel_running_job_sets_flag(client: TestClient, tmp_path: Path) -> None:
    job = _make_job(tmp_path)
    job.status = STATUS_RUNNING

    response = client.post(f"/v1/jobs/{job.job_id}/cancel", headers=AUTH)

    assert response.status_code == 200
    assert response.json() == {
        "job_id": job.job_id,
        "status": STATUS_RUNNING,
        "cancel_requested": True,
    }
    assert job.cancel_requested is True


def test_cancel_finished_job_is_idempotent(client: TestClient, tmp_path: Path) -> None:
    job = _make_job(tmp_path)
    job.status = "done"

    response = client.post(f"/v1/jobs/{job.job_id}/cancel", headers=AUTH)

    assert response.status_code == 200
    assert response.json() == {"job_id": job.job_id, "status": "done"}
    assert job.cancel_requested is False


# --- Worker cancellation mechanics --------------------------------------


def test_event_hook_raises_when_cancel_requested() -> None:
    worker = TranscriptionWorker.__new__(TranscriptionWorker)
    job = TranscribeJob(job_id="j1", file_path=Path("x"), original_name="x")
    worker._running_job = None
    worker._event_hook(object())  # no running job: must not raise
    job.cancel_requested = False
    worker._running_job = job
    worker._event_hook(object())  # flag not set: must not raise
    job.cancel_requested = True
    with pytest.raises(JobCancelledError):
        worker._event_hook(object())
