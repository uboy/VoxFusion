"""Shared dataclasses for the live GigaAM pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class LiveUtterance:
    """One VAD-bounded utterance captured during a live session."""

    seq_id: int
    source: str
    start_s: float
    end_s: float
    sample_rate: int
    samples: NDArray[np.float32]


@dataclass(frozen=True)
class LiveGigaAMJob:
    """One live GigaAM transcription job."""

    seq_id: int
    source: str
    start_s: float
    end_s: float
    sample_rate: int
    samples: NDArray[np.float32]
    finalize: bool = False
    retry_count: int = 0


@dataclass(frozen=True)
class LiveGigaAMResult:
    """Result of one live GigaAM transcription job."""

    seq_id: int
    source: str
    start_s: float
    end_s: float
    text: str
    worker_id: int
    finalize: bool = False
    deferred: bool = False
    error: str | None = None
