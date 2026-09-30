"""VoxFusion HTTP transcription API (FastAPI, CPU-only).

Endpoints:
    GET  /healthz        - liveness and load info (no auth)
    POST /v1/transcribe  - multipart file upload, returns {job_id, status} (auth)
    GET  /v1/jobs/{id}   - job status and result (auth)
    GET  /v1/jobs        - recent jobs (auth)

Uploads are streamed to disk in fixed-size chunks, so multi-GB files do not
consume memory. Transcription runs in one FIFO background worker thread,
reusing the VoxFusion batch pipeline (see server/worker.py).

Auth: ``Authorization: Bearer $VOXFUSION_API_TOKEN`` on every endpoint
except ``/healthz``. The token is read from the environment (systemd
EnvironmentFile); if it is unset, all authenticated endpoints return 401.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import math
import os
import re
import subprocess
import threading
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile

from server.worker import (
    STATUS_CANCELLED,
    STATUS_DONE,
    STATUS_ERROR,
    STATUS_QUEUED,
    STATUS_RUNNING,
    TranscribeJob,
    TranscriptionWorker,
)

DATA_DIR = Path(os.environ.get("VOXFUSION_API_DATA_DIR", "/home/dmazur/voxfusion-api"))
UPLOAD_DIR = DATA_DIR / "uploads"
RETENTION_HOURS = float(os.environ.get("VOXFUSION_API_RETENTION_HOURS", "24"))
MAX_UPLOAD_GB = float(os.environ.get("VOXFUSION_API_MAX_UPLOAD_GB", "32"))
MAX_JOBS_HISTORY = int(os.environ.get("VOXFUSION_API_MAX_JOBS_HISTORY", "500"))
READ_CHUNK = 4 * 1024 * 1024
RETENTION_SWEEP_INTERVAL_S = 600
SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")

TOKEN = os.environ.get("VOXFUSION_API_TOKEN", "")

DIARIZATION_STRATEGIES = ("auto", "channel", "ml", "hybrid", "none")


def _resolve_diarization_strategy(raw: str | None) -> str:
    """Normalize the strategy env value; invalid config must fail at startup."""
    strategy = (raw or "channel").strip().lower()
    if strategy not in DIARIZATION_STRATEGIES:
        raise RuntimeError(
            f"VOXFUSION_API_DIARIZATION_STRATEGY={raw!r} is invalid; "
            f"expected one of: {', '.join(DIARIZATION_STRATEGIES)}"
        )
    return strategy


DIARIZATION_STRATEGY = _resolve_diarization_strategy(
    os.environ.get("VOXFUSION_API_DIARIZATION_STRATEGY")
)

DEFAULT_DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"


def _resolve_diarization_model(raw: str | None) -> str:
    """Diarization model id; blank/unset keeps the bundled default."""
    model = (raw or "").strip()
    return model or DEFAULT_DIARIZATION_MODEL


DIARIZATION_MODEL = _resolve_diarization_model(os.environ.get("VOXFUSION_API_DIARIZATION_MODEL"))


# ETA shown to clients while a job is queued/running. Measured RTF with the
# community-1 diarization model is ~2.0 (CPUQuota=200%); the default adds a
# ~10% buffer on top.
def _resolve_eta_multiplier(raw: str | None) -> float:
    """ETA multiplier from env; invalid or non-positive config fails at startup."""
    try:
        multiplier = float(raw if raw is not None and raw.strip() else "2.2")
    except ValueError:
        raise RuntimeError(f"VOXFUSION_API_ETA_MULTIPLIER={raw!r} is not a number") from None
    if multiplier <= 0:
        raise RuntimeError(f"VOXFUSION_API_ETA_MULTIPLIER={raw!r} must be > 0")
    return multiplier


ETA_MULTIPLIER = _resolve_eta_multiplier(os.environ.get("VOXFUSION_API_ETA_MULTIPLIER"))

_CONFIG_OVERRIDES = {
    "asr": {
        "model_size": os.environ.get("VOXFUSION_API_MODEL", "small"),
        "device": os.environ.get("VOXFUSION_API_DEVICE", "cpu"),
        "cpu_threads": int(os.environ.get("VOXFUSION_API_CPU_THREADS", "6")),
        "language": None,
    },
    "diarization": {
        "strategy": DIARIZATION_STRATEGY,
        "ml": {"model": DIARIZATION_MODEL},
    },
}


# ffprobe parses client-controlled media, so it runs with a minimal
# environment: no service secrets leak into a potentially exploitable child.
_FFPROBE_ENV = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "LANG": "C.UTF-8"}


def _probe_duration(path: Path) -> float | None:
    """Audio duration in seconds via ffprobe; None when unavailable."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            timeout=30,
            check=False,
            env=_FFPROBE_ENV,
        )
        if result.returncode != 0:
            return None
        data = json.loads(result.stdout.decode("utf-8", "replace") or "{}")
        duration = float(data["format"]["duration"])
    except Exception:
        return None
    # math.ceil(Infinity) would raise OverflowError after the upload was
    # already stored, so non-finite durations degrade to "no ETA" as well.
    if not math.isfinite(duration) or duration <= 0:
        return None
    return duration


def _estimate_eta(duration_s: float | None) -> float | None:
    """Expected processing time with buffer; None without a duration."""
    if duration_s is None:
        return None
    return float(math.ceil(duration_s * ETA_MULTIPLIER))


worker = TranscriptionWorker(overrides=_CONFIG_OVERRIDES)

# In-memory job registry (survives only within one service process).
_jobs_lock = threading.Lock()
_jobs: dict[str, TranscribeJob] = {}
_jobs_order: list[str] = []


def _register_job(job: TranscribeJob) -> None:
    with _jobs_lock:
        _jobs[job.job_id] = job
        _jobs_order.append(job.job_id)
        while len(_jobs_order) > MAX_JOBS_HISTORY:
            finished = [
                jid
                for jid in _jobs_order
                if _jobs[jid].status in (STATUS_DONE, STATUS_ERROR, STATUS_CANCELLED)
            ]
            if not finished:
                break
            oldest = finished[0]
            _jobs_order.remove(oldest)
            _jobs.pop(oldest, None)


def _get_job(job_id: str) -> TranscribeJob:
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return job


def _require_auth(request: Request) -> None:
    header = request.headers.get("authorization", "")
    if not TOKEN:
        raise HTTPException(status_code=401, detail="service token is not configured")
    if not header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="missing bearer token")
    supplied = header[len("Bearer ") :].strip()
    if not supplied or not hmac.compare_digest(supplied.encode(), TOKEN.encode()):
        raise HTTPException(status_code=401, detail="invalid token")


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).astimezone().isoformat(timespec="seconds")


def _job_summary(job: TranscribeJob) -> dict[str, Any]:
    return {
        "job_id": job.job_id,
        "status": job.status,
        "filename": job.original_name,
        "size_bytes": job.size_bytes,
        "language": job.language,
        "created": _iso(job.created),
        "processing_time_s": job.processing_time_s,
        "audio_duration_s": job.audio_duration_s,
        "eta_seconds": job.eta_seconds,
        "model": job.model,
    }


def _job_full(job: TranscribeJob) -> dict[str, Any]:
    data = _job_summary(job)
    data.update(
        {
            "text": job.text,
            "segments": job.segments,
            "error": job.error,
        }
    )
    return data


async def _store_upload(job_id: str, filename: str | None, src: UploadFile) -> tuple[Path, int]:
    """Stream the multipart upload to disk without holding it in memory."""
    safe_name = SAFE_NAME_RE.sub("_", Path(filename or "upload").name) or "upload"
    dest = UPLOAD_DIR / f"{job_id}_{safe_name}"
    limit = int(MAX_UPLOAD_GB * (1 << 30))
    size = 0
    try:
        with dest.open("wb") as out:
            while True:
                chunk = await src.read(READ_CHUNK)
                if not chunk:
                    break
                size += len(chunk)
                if size > limit:
                    raise HTTPException(
                        status_code=413,
                        detail=f"upload exceeds VOXFUSION_API_MAX_UPLOAD_GB={MAX_UPLOAD_GB}",
                    )
                out.write(chunk)
    except HTTPException:
        dest.unlink(missing_ok=True)
        raise
    except asyncio.CancelledError:
        # Client disconnect / timeout mid-upload: BaseException, not caught
        # by the branch below - clean up the partial file explicitly.
        dest.unlink(missing_ok=True)
        raise
    except Exception as exc:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"failed to store upload: {exc}") from exc
    if size == 0:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="empty upload")
    return dest, size


async def _retention_loop() -> None:
    """Delete uploaded files (and finished job records) older than RETENTION_HOURS."""
    while True:
        try:
            cutoff = time.time() - RETENTION_HOURS * 3600
            if UPLOAD_DIR.exists():
                for path in UPLOAD_DIR.iterdir():
                    try:
                        job_prefix = path.name.split("_", 1)[0]
                        active = _jobs.get(job_prefix)
                        if active is not None and active.status in (STATUS_QUEUED, STATUS_RUNNING):
                            continue
                        if path.is_file() and path.stat().st_mtime < cutoff:
                            path.unlink()
                    except OSError:
                        pass
            created_cutoff = time.time() - RETENTION_HOURS * 3600
            with _jobs_lock:
                stale = [
                    jid
                    for jid in _jobs_order
                    if _jobs[jid].status in (STATUS_DONE, STATUS_ERROR, STATUS_CANCELLED)
                    and _jobs[jid].created < created_cutoff
                ]
                for jid in stale:
                    _jobs.pop(jid, None)
                    _jobs_order.remove(jid)
        except Exception:
            pass
        await asyncio.sleep(RETENTION_SWEEP_INTERVAL_S)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    worker.start()
    task = asyncio.create_task(_retention_loop())
    try:
        yield
    finally:
        task.cancel()
        worker.stop()


app = FastAPI(
    title="VoxFusion HTTP API",
    description="Asynchronous audio/video transcription service on top of VoxFusion.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    """Liveness probe: no auth."""
    return {
        "status": "ok",
        "model": worker.model_name,
        "device": "cpu",
        "jobs_active": worker.jobs_active(),
    }


MAX_SPEAKER_HINT = 32


def _parse_speaker_hint(raw: str | None, field: str) -> int | None:
    """Parse a min/max speakers form value; blank means 'no hint'."""
    value = (raw or "").strip()
    if not value:
        return None
    try:
        parsed = int(value)
    except ValueError:
        raise HTTPException(
            status_code=400, detail=f"{field} must be an integer, got {raw!r}"
        ) from None
    if parsed < 1:
        raise HTTPException(status_code=400, detail=f"{field} must be >= 1")
    if parsed > MAX_SPEAKER_HINT:
        raise HTTPException(
            status_code=400,
            detail=f"{field} must be <= {MAX_SPEAKER_HINT}",
        )
    return parsed


@app.post("/v1/transcribe")
async def transcribe(
    request: Request,
    file: Annotated[UploadFile, File(...)],
    language: Annotated[str | None, Form()] = None,
    include_segments: Annotated[str | None, Form()] = None,
    min_speakers: Annotated[str | None, Form()] = None,
    max_speakers: Annotated[str | None, Form()] = None,
) -> dict[str, Any]:
    """Queue a transcription job. Returns immediately with a job_id."""
    _require_auth(request)
    lang: str | None = (language or "").strip().lower()
    if lang in ("", "auto"):
        lang = None
    want_segments = (include_segments or "").strip().lower() in ("1", "true", "yes", "on")
    job_min_speakers = _parse_speaker_hint(min_speakers, "min_speakers")
    job_max_speakers = _parse_speaker_hint(max_speakers, "max_speakers")
    if (
        job_min_speakers is not None
        and job_max_speakers is not None
        and job_min_speakers > job_max_speakers
    ):
        raise HTTPException(status_code=400, detail="min_speakers must be <= max_speakers")

    job_id = uuid.uuid4().hex[:12]
    dest, size = await _store_upload(job_id, file.filename, file)

    job = TranscribeJob(
        job_id=job_id,
        file_path=dest,
        original_name=Path(file.filename or safe_upload_name(dest)).name,
        language=lang,
        include_segments=want_segments,
        size_bytes=size,
        min_speakers=job_min_speakers,
        max_speakers=job_max_speakers,
    )
    # Off the event loop: a slow ffprobe would freeze all concurrent requests.
    job.eta_seconds = _estimate_eta(await asyncio.to_thread(_probe_duration, dest))
    _register_job(job)
    worker.submit(job)
    return {"job_id": job_id, "status": "queued", "eta_seconds": job.eta_seconds}


def safe_upload_name(path: Path) -> str:
    return path.name.split("_", 1)[-1]


@app.get("/v1/jobs/{job_id}")
def get_job(job_id: str, request: Request) -> dict[str, Any]:
    _require_auth(request)
    return _job_full(_get_job(job_id))


@app.post("/v1/jobs/{job_id}/cancel")
def cancel_job(job_id: str, request: Request) -> dict[str, Any]:
    """Cancel a queued job immediately or request cancellation of a running one."""
    _require_auth(request)
    job = _get_job(job_id)
    if job.status == STATUS_QUEUED:
        job.status = STATUS_CANCELLED
        # Also raise the flag: if the worker picked the job up between our
        # status read and this write, its post-RUNNING check drops it instead
        # of processing a deleted file.
        job.cancel_requested = True
        try:
            job.file_path.unlink(missing_ok=True)
        except OSError:
            pass
        return {"job_id": job_id, "status": STATUS_CANCELLED}
    if job.status == STATUS_RUNNING:
        # Cooperative cancel: the worker checks the flag at the next
        # pipeline progress event (every few seconds) and flips the status.
        job.cancel_requested = True
        return {"job_id": job_id, "status": STATUS_RUNNING, "cancel_requested": True}
    return {"job_id": job_id, "status": job.status}


@app.get("/v1/jobs")
def list_jobs(request: Request) -> dict[str, Any]:
    _require_auth(request)
    with _jobs_lock:
        ordered = [_jobs[jid] for jid in reversed(_jobs_order)]
    return {
        "count": len(ordered),
        "jobs": [_job_summary(job) for job in ordered],
    }
