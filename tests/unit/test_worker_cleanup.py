"""Unit tests for the worker upload-cleanup policy.

Policy (owner request 2026-09-29): the uploaded file is deleted as soon as
the job finishes successfully; failed jobs keep the file on disk so the
transcription can be re-run after the failure is fixed.
"""

from pathlib import Path

from server.worker import (
    STATUS_CANCELLED,
    STATUS_DONE,
    STATUS_ERROR,
    STATUS_RUNNING,
    TranscribeJob,
    TranscriptionWorker,
)


def _make_worker() -> TranscriptionWorker:
    # _cleanup_upload does not touch instance state; skip the heavy __init__
    # (config load, model catalog).
    return object.__new__(TranscriptionWorker)


def _make_job(tmp_path: Path) -> TranscribeJob:
    path = tmp_path / "job1_meeting.wav"
    path.write_bytes(b"fake audio")
    return TranscribeJob(job_id="job1", file_path=path, original_name="meeting.wav")


def test_done_job_deletes_upload(tmp_path: Path) -> None:
    worker = _make_worker()
    job = _make_job(tmp_path)
    job.status = STATUS_DONE

    worker._cleanup_upload(job)

    assert not job.file_path.exists()


def test_error_job_keeps_upload(tmp_path: Path) -> None:
    worker = _make_worker()
    job = _make_job(tmp_path)
    job.status = STATUS_ERROR
    job.error = "boom"

    worker._cleanup_upload(job)

    assert job.file_path.exists()


def test_running_job_keeps_upload(tmp_path: Path) -> None:
    worker = _make_worker()
    job = _make_job(tmp_path)
    job.status = STATUS_RUNNING

    worker._cleanup_upload(job)

    assert job.file_path.exists()


def test_cancelled_job_deletes_upload(tmp_path: Path) -> None:
    worker = _make_worker()
    job = _make_job(tmp_path)
    job.status = STATUS_CANCELLED

    worker._cleanup_upload(job)

    assert not job.file_path.exists()


def test_cleanup_is_idempotent(tmp_path: Path) -> None:
    worker = _make_worker()
    job = _make_job(tmp_path)
    job.status = STATUS_DONE

    worker._cleanup_upload(job)
    worker._cleanup_upload(job)  # missing_ok: second call must not raise

    assert not job.file_path.exists()
