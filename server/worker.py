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
    duration_s: float | None = None
    model: str | None = None


class TranscriptionWorker:
    """FIFO worker thread around a single shared ``PipelineOrchestrator``."""

    def __init__(self, overrides: dict | None = None) -> None:
        self._queue: queue.Queue[TranscribeJob] = queue.Queue()
        self._thread = threading.Thread(
            target=self._run, name="voxfusion-api-worker", daemon=True
        )
        self._stop_event = threading.Event()
        self._orchestrator: PipelineOrchestrator | None = None
        self._lock = threading.Lock()
        self._running_job_id: str | None = None
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
            with self._lock:
                self._running_job_id = job.job_id
            job.status = STATUS_RUNNING
            try:
                self._process(job)
            except Exception as exc:  # noqa: BLE001 - any failure goes to the job
                job.status = STATUS_ERROR
                job.error = str(exc) or exc.__class__.__name__
            finally:
                with self._lock:
                    self._running_job_id = None
                self._queue.task_done()

    def _get_orchestrator(self) -> PipelineOrchestrator:
        if self._orchestrator is None:
            self._orchestrator = PipelineOrchestrator(
                self.config, interactive=True
            )
        return self._orchestrator

    def _process(self, job: TranscribeJob) -> None:
        orch = self._get_orchestrator()
        # Per-job language override: the batch pipeline does not pass a
        # per-call language, so the engine falls back to config.asr.language.
        # The worker is single-threaded, hence mutating the shared config
        # between jobs is safe.
        orch._asr._config.language = job.language or None  # noqa: SLF001

        started = time.monotonic()
        result = asyncio.run(orch.transcribe_file(job.file_path))
        job.duration_s = round(time.monotonic() - started, 2)

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
