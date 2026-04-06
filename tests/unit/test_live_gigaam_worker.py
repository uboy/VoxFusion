"""Unit tests for live GigaAM worker-process helpers."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import voxfusion.live_gigaam.worker as live_worker
from voxfusion.live_gigaam.types import LiveGigaAMJob


def test_init_worker_loads_engine_once_and_transcribe_job_reuses_it(monkeypatch) -> None:
    created: list[object] = []

    class _FakeEngine:
        def __init__(self, config) -> None:
            self.config = config
            self.load_calls = 0
            self.transcribe_calls: list[tuple[np.ndarray, str | None]] = []
            created.append(self)

        def load_model(self) -> None:
            self.load_calls += 1

        def transcribe_samples_sync(self, samples, *, language=None):
            self.transcribe_calls.append((np.asarray(samples), language))
            return [SimpleNamespace(text="hello"), SimpleNamespace(text="world")]

    monkeypatch.setattr(live_worker, "GigaAMCTCEngine", _FakeEngine)
    monkeypatch.setattr(live_worker, "_configure_worker_threads", lambda _limit: None)
    monkeypatch.setattr(live_worker, "_WORKER_ENGINE", None)
    monkeypatch.setattr(live_worker, "_WORKER_ID", -1)

    live_worker.init_worker(
        7,
        {"model_size": "gigaam-v3-e2e-ctc", "language": "ru"},
        3,
    )

    first_job = LiveGigaAMJob(
        seq_id=1,
        source="microphone",
        start_s=0.0,
        end_s=1.0,
        sample_rate=16000,
        samples=[0.1, 0.2, 0.3],
    )
    second_job = LiveGigaAMJob(
        seq_id=2,
        source="microphone",
        start_s=1.0,
        end_s=2.0,
        sample_rate=16000,
        samples=[0.4, 0.5],
        finalize=True,
    )

    first = live_worker.transcribe_job(first_job)
    second = live_worker.transcribe_job(second_job)

    assert len(created) == 1
    assert created[0].load_calls == 1
    assert created[0].transcribe_calls[0][1] == "ru"
    assert created[0].transcribe_calls[1][1] == "ru"
    assert first.worker_id == 7
    assert second.worker_id == 7
    assert first.text == "hello world"
    assert second.text == "hello world"


def test_transcribe_job_requires_initialized_worker(monkeypatch) -> None:
    monkeypatch.setattr(live_worker, "_WORKER_ENGINE", None)

    job = LiveGigaAMJob(
        seq_id=0,
        source="microphone",
        start_s=0.0,
        end_s=1.0,
        sample_rate=16000,
        samples=[0.1],
    )

    try:
        live_worker.transcribe_job(job)
    except RuntimeError as exc:
        assert "not initialized" in str(exc)
    else:
        raise AssertionError("Expected transcribe_job to fail without init_worker().")
