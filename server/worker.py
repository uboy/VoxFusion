"""Background transcription worker for the VoxFusion HTTP API.

A single FIFO worker thread processes uploaded files in submission order.
It reuses the canonical VoxFusion batch pipeline
(``PipelineOrchestrator.transcribe_file``) — the same code path the CLI
``voxfusion transcribe`` command uses, including ffmpeg audio extraction
for video/compressed formats (mp4, mkv, webm, mp3, m4a, ...).

The orchestrator (and therefore the ASR model) is created once and reused
across jobs so the model stays loaded in memory. CPU-bound inference runs
inside the engine's own single-worker executor.
"""

from __future__ import annotations

import asyncio
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from voxfusion.config.loader import load_config
from voxfusion.config.models import PipelineConfig
from voxfusion.pipeline.orchestrator import PipelineOrchestrator

STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_ERROR = "error"
STATUS_CANCELLED = "cancelled"


class JobCancelledError(Exception):
    """Raised inside the pipeline event callback to abort a running job."""


@dataclass
class TranscribeJob:
    """One queued transcription request. Mutated in place by the worker."""

    job_id: str
    file_path: Path
    original_name: str
    language: str | None = None  # None = auto-detect
    include_segments: bool = False
    size_bytes: int = 0
    created: float = field(default_factory=time.time)
    status: str = STATUS_QUEUED
    text: str | None = None
    segments: list[dict[str, Any]] | None = None
    error: str | None = None
    processing_time_s: float | None = None
    audio_duration_s: float | None = None
    model: str | None = None
    eta_seconds: float | None = None
    cancel_requested: bool = False
    min_speakers: int | None = None  # None = engine default
    max_speakers: int | None = None  # None = engine default


class TranscriptionWorker:
    """FIFO worker thread around a single shared ``PipelineOrchestrator``."""

    def __init__(self, overrides: dict[str, Any] | None = None) -> None:
        self._queue: queue.Queue[TranscribeJob] = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="voxfusion-api-worker", daemon=True)
        self._stop_event = threading.Event()
        self._orchestrator: PipelineOrchestrator | None = None
        self._lock = threading.Lock()
        self._running_job_id: str | None = None
        self._running_job: TranscribeJob | None = None
        self._overrides = overrides
        self.config: PipelineConfig = load_config(overrides=overrides)
        self.model_name = f"{self.config.asr.engine}/{self.config.asr.model_size}"
        self.device = self.config.asr.device

    # -- public API (called from request handlers) -------------------------

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()

    def submit(self, job: TranscribeJob) -> None:
        self._queue.put(job)

    def jobs_active(self) -> int:
        """Queued + currently running jobs."""
        with self._lock:
            running = 1 if self._running_job_id is not None else 0
        return self._queue.qsize() + running

    # -- worker internals ---------------------------------------------------

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                job = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if job.status == STATUS_CANCELLED:
                # Cancelled while still queued: the cancel endpoint already
                # removed the upload, just drop the entry.
                self._queue.task_done()
                continue
            with self._lock:
                self._running_job_id = job.job_id
            self._running_job = job
            job.status = STATUS_RUNNING
            try:
                self._process(job)
            except JobCancelledError:
                job.status = STATUS_CANCELLED
            except Exception as exc:
                job.status = STATUS_ERROR
                job.error = str(exc) or exc.__class__.__name__
            finally:
                self._running_job = None
                self._cleanup_upload(job)
                with self._lock:
                    self._running_job_id = None
                self._queue.task_done()

    def _event_hook(self, event: object) -> None:
        """Pipeline progress callback: cooperative cancellation point.

        The batch pipeline emits progress events every few seconds; raising
        here aborts the current job within one progress interval.
        """
        job = self._running_job
        if job is not None and job.cancel_requested:
            raise JobCancelledError(f"job {job.job_id} cancelled by client request")

    def _cleanup_upload(self, job: TranscribeJob) -> None:
        """Delete the uploaded file as soon as the job finishes.

        The upload is a transient processing artifact: the client already
        holds the original, and the transcription result lives in the job
        record. Failed jobs keep their file so transcription can be re-run
        (new POST with the same file) after the failure is fixed; the
        retention sweep in app.py is the final safety net for those.
        Cancelled jobs delete their file immediately, same as done ones.
        """
        if job.status not in (STATUS_DONE, STATUS_CANCELLED):
            return
        try:
            job.file_path.unlink(missing_ok=True)
        except Exception:
            # Never let cleanup kill the single worker thread; the retention
            # sweep will pick the file up later.
            pass

    def _get_orchestrator(self) -> PipelineOrchestrator:
        if self._orchestrator is None:
            self._orchestrator = PipelineOrchestrator(
                self.config,
                on_event=self._event_hook,
                interactive=True,
            )
        return self._orchestrator

    def _process(self, job: TranscribeJob) -> None:
        orch = self._get_orchestrator()
        # Per-job language override: the batch pipeline does not pass a
        # per-call language, so the engine falls back to config.asr.language.
        # The worker is single-threaded, hence mutating the shared config
        # between jobs is safe. Same pattern for the diarization speaker
        # hints: the pyannote engine reads them from the shared
        # DiarizationMLConfig at pipeline call time.
        orch._asr._config.language = job.language or None  # type: ignore[attr-defined]
        ml_config = self.config.diarization.ml
        ml_config.min_speakers = job.min_speakers
        ml_config.max_speakers = job.max_speakers

        started = time.monotonic()
        result = asyncio.run(orch.transcribe_file(job.file_path))
        job.processing_time_s = round(time.monotonic() - started, 2)
        raw_audio_s = result.source_info.get("duration_s")
        job.audio_duration_s = float(raw_audio_s) if raw_audio_s is not None else None

        lines: list[str] = []
        segments_out: list[dict[str, Any]] = []
        for translated in result.segments:
            seg = translated.diarized.segment
            lines.append(seg.text)
            if job.include_segments:
                segments_out.append(
                    {
                        "start": seg.start_time,
                        "end": seg.end_time,
                        "text": seg.text,
                        "speaker": translated.diarized.speaker_id,
                    }
                )
        job.text = "\n".join(line for line in lines if line)
        job.segments = segments_out if job.include_segments else None
        job.model = self.model_name
        job.status = STATUS_DONE
