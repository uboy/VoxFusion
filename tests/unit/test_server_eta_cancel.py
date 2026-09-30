"""Unit tests for API ETA estimate and job cancellation."""

from __future__ import annotations

import json
import queue
import subprocess
import threading
import time
import uuid
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from server.app import (
    _estimate_eta,
    _parse_speaker_hint,
    _probe_duration,
    _register_job,
    _resolve_eta_multiplier,
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


def _make_job(tmp_path: Path, job_id: str | None = None) -> TranscribeJob:
    # Unique ids: the shared registry does not dedupe job ids.
    job_id = job_id or f"jobCAN{uuid.uuid4().hex[:8]}"
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


# --- Worker loop: cancellation branches ----------------------------------


def _make_loop_worker(processed: list[str]) -> TranscriptionWorker:
    """A worker skeleton with the real _run but stubbed heavy parts."""
    worker = TranscriptionWorker.__new__(TranscriptionWorker)
    worker._queue = queue.Queue()
    worker._stop_event = threading.Event()
    worker._lock = threading.Lock()
    worker._running_job_id = None
    worker._running_job = None
    worker._orchestrator = None
    worker._process = lambda job: processed.append(job.job_id)  # type: ignore[method-assign]
    return worker


def _drain(worker: TranscriptionWorker, processed: list[str], expected: str) -> None:
    thread = threading.Thread(target=worker._run, daemon=True)
    thread.start()
    deadline = time.time() + 5
    while expected not in processed and time.time() < deadline:
        time.sleep(0.02)
    worker._stop_event.set()
    thread.join(timeout=5)
    assert not thread.is_alive()


def test_worker_run_drops_cancelled_queued_job(tmp_path: Path) -> None:
    processed: list[str] = []
    worker = _make_loop_worker(processed)
    stale = TranscribeJob(
        job_id="c1",
        file_path=tmp_path / "c1.wav",
        original_name="c1.wav",
        status=STATUS_CANCELLED,
    )
    normal = TranscribeJob(job_id="ok1", file_path=tmp_path / "ok.wav", original_name="ok.wav")
    worker._queue.put(stale)
    worker._queue.put(normal)

    _drain(worker, processed, "ok1")

    assert processed == ["ok1"]
    assert stale.status == STATUS_CANCELLED
    assert normal.status == STATUS_RUNNING  # stubbed _process does not finish it


def test_worker_run_skips_pickup_race_cancel(tmp_path: Path) -> None:
    """Cancel raced the queue pickup: the flag set by the endpoint wins."""
    processed: list[str] = []
    worker = _make_loop_worker(processed)
    raced = TranscribeJob(
        job_id="r1",
        file_path=tmp_path / "r1.wav",
        original_name="r1.wav",
        cancel_requested=True,  # status still queued: pickup-race window
    )
    worker._queue.put(raced)

    _drain(worker, processed, "")  # nothing must be processed

    assert processed == []
    assert raced.status == STATUS_CANCELLED


def test_worker_run_marks_cancelled_on_hook_error(tmp_path: Path) -> None:
    def cancel_itself(job: TranscribeJob) -> None:
        raise JobCancelledError(f"job {job.job_id} cancelled by client request")

    worker = _make_loop_worker([])
    worker._process = cancel_itself  # type: ignore[method-assign]
    job = TranscribeJob(job_id="z1", file_path=tmp_path / "z1.wav", original_name="z1.wav")
    (tmp_path / "z1.wav").write_bytes(b"audio")
    worker._queue.put(job)

    _drain(worker, [], "")

    assert job.status == STATUS_CANCELLED
    assert not job.file_path.exists()


# --- Registry eviction ----------------------------------------------------


def test_register_job_evicts_cancelled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("server.app.MAX_JOBS_HISTORY", 3)
    from server import app as app_module

    # The registry is shared module state; earlier tests may hold entries.
    with app_module._jobs_lock:
        app_module._jobs.clear()
        app_module._jobs_order.clear()

    jobs = []
    for index in range(4):
        job = TranscribeJob(job_id=f"ev{index}", file_path=Path("x"), original_name="x")
        job.status = STATUS_CANCELLED
        _register_job(job)
        jobs.append(job)

    with app_module._jobs_lock:
        registered = set(app_module._jobs)

    assert jobs[0].job_id not in registered
    assert {j.job_id for j in jobs[1:]} <= registered


# --- ffprobe contract ------------------------------------------------------


def test_probe_duration_passes_timeout_and_clean_env() -> None:
    captured: dict = {}

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess:
        captured.update(kwargs)
        return subprocess.CompletedProcess([], 0, stdout=b'{"format": {"duration": "5"}}')

    with patch("server.app.subprocess.run", fake_run):
        assert _probe_duration(Path("x.m4a")) == 5.0
    assert captured.get("timeout") == 30
    env = captured.get("env") or {}
    assert "VOXFUSION_API_TOKEN" not in env


def test_probe_duration_rejects_nonpositive_and_nonfinite() -> None:
    for raw in ("0", "-1", "Infinity", "NaN"):
        payload = json.dumps({"format": {"duration": raw}}).encode()

        def make_fake_run(payload: bytes):
            def fake_run(*_a: object, **_k: object) -> subprocess.CompletedProcess:
                return subprocess.CompletedProcess([], 0, stdout=payload, stderr=b"")

            return fake_run

        with patch("server.app.subprocess.run", make_fake_run(payload)):
            assert _probe_duration(Path("x.m4a")) is None, raw


# --- Speaker hint cap ------------------------------------------------------


def test_speaker_hint_capped_at_32() -> None:
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        _parse_speaker_hint("33", "max_speakers")
    assert exc.value.status_code == 400
    assert "32" in exc.value.detail
    assert _parse_speaker_hint("32", "max_speakers") == 32


def test_eta_multiplier_default_and_value() -> None:
    assert _resolve_eta_multiplier(None) == 1.8
    assert _resolve_eta_multiplier("") == 1.8
    assert _resolve_eta_multiplier("2.2") == 2.2


def test_eta_multiplier_rejects_garbage_and_nonpositive() -> None:
    for raw in ("abc", "0", "-1"):
        with pytest.raises(RuntimeError, match="VOXFUSION_API_ETA_MULTIPLIER"):
            _resolve_eta_multiplier(raw)
