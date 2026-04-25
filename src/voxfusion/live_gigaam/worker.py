"""Warm worker-process helpers for live GigaAM transcription."""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from voxfusion.asr.gigaam_engine import GigaAMCTCEngine
from voxfusion.config.models import ASRConfig
from voxfusion.live_gigaam.types import LiveGigaAMJob, LiveGigaAMResult
from voxfusion.logging import get_logger

log = get_logger(__name__)

_WORKER_ENGINE: GigaAMCTCEngine | None = None
_WORKER_ID: int = -1


def _configure_worker_threads(thread_limit: int) -> None:
    limit = max(1, int(thread_limit))
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = str(limit)
    try:
        import torch

        if hasattr(torch, "set_num_threads"):
            torch.set_num_threads(limit)
        if hasattr(torch, "set_num_interop_threads"):
            torch.set_num_interop_threads(1)
    except Exception:
        pass


def init_worker(worker_id: int, asr_payload: dict[str, Any], thread_limit: int) -> None:
    """Initializer for one persistent GigaAM worker process."""
    global _WORKER_ENGINE, _WORKER_ID  # noqa: PLW0603 — required for multiprocessing worker state
    _WORKER_ID = worker_id
    _configure_worker_threads(thread_limit)
    config = ASRConfig(**asr_payload)
    _WORKER_ENGINE = GigaAMCTCEngine(config)
    _WORKER_ENGINE.load_model()
    log.info("live_gigaam.worker_ready", worker_id=worker_id, model=config.model_size)


def ping_worker() -> int:
    """Return the active worker id, forcing process startup when called."""
    return _WORKER_ID


def transcribe_job(job: LiveGigaAMJob) -> LiveGigaAMResult:
    """Run one GigaAM transcription job inside a warm worker process."""
    if _WORKER_ENGINE is None:
        raise RuntimeError("Live GigaAM worker is not initialized.")
    samples = np.ascontiguousarray(np.asarray(job.samples, dtype=np.float32).reshape(-1))
    segments = _WORKER_ENGINE.transcribe_samples_sync(samples, language="ru")
    text = " ".join(segment.text for segment in segments if segment.text.strip()).strip()
    return LiveGigaAMResult(
        seq_id=job.seq_id,
        source=job.source,
        start_s=job.start_s,
        end_s=job.end_s,
        text=text,
        worker_id=_WORKER_ID,
        finalize=job.finalize,
        error=None,
    )
