"""Unit tests for live GigaAM dispatch logic."""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import Future
from types import SimpleNamespace

import voxfusion.live_gigaam.dispatcher as live_dispatcher
from voxfusion.config.models import ASRConfig, LiveGigaAMConfig
from voxfusion.live_gigaam.dispatcher import LiveASRDispatcher, _WorkerSlot
from voxfusion.live_gigaam.types import LiveGigaAMJob, LiveGigaAMResult


class _FakeExecutor:
    def __init__(self, outcomes: list[object]) -> None:
        self._outcomes = list(outcomes)

    def submit(self, _fn, _job):
        future: Future = Future()
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            future.set_exception(outcome)
        else:
            future.set_result(outcome)
        return future

    def shutdown(self, **_kwargs) -> None:
        return None


def test_choose_worker_prefers_least_busy_slot() -> None:
    dispatcher = LiveASRDispatcher(
        ASRConfig(model_size="gigaam-v3-e2e-ctc"),
        LiveGigaAMConfig(worker_count=2),
    )
    dispatcher._workers = [
        _WorkerSlot(worker_id=0, executor=_FakeExecutor([]), in_flight=1, completed=0),
        _WorkerSlot(worker_id=1, executor=_FakeExecutor([]), in_flight=0, completed=5),
        _WorkerSlot(worker_id=2, executor=_FakeExecutor([]), in_flight=0, completed=1),
    ]

    chosen = dispatcher._choose_worker()

    assert chosen.worker_id == 2


def test_dispatcher_retries_once_and_returns_success() -> None:
    dispatcher = LiveASRDispatcher(
        ASRConfig(model_size="gigaam-v3-e2e-ctc"),
        LiveGigaAMConfig(worker_count=1, max_retries=1),
    )
    dispatcher._started = True
    dispatcher._workers = [
        _WorkerSlot(
            worker_id=0,
            executor=_FakeExecutor(
                [
                    RuntimeError("transient"),
                    LiveGigaAMResult(
                        seq_id=0,
                        source="microphone",
                        start_s=0.0,
                        end_s=1.0,
                        text="ok",
                        worker_id=0,
                    ),
                ]
            ),
        )
    ]
    job = LiveGigaAMJob(
        seq_id=0,
        source="microphone",
        start_s=0.0,
        end_s=1.0,
        sample_rate=16000,
        samples=[],
    )

    result = asyncio.run(dispatcher.transcribe(job))

    assert result.text == "ok"
    assert dispatcher._workers[0].failed == 1
    assert dispatcher._workers[0].completed == 1


def test_dispatcher_retries_after_sync_submit_failure_and_clears_in_flight() -> None:
    class _SyncFailExecutor(_FakeExecutor):
        def submit(self, _fn, _job):
            outcome = self._outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            future: Future = Future()
            future.set_result(outcome)
            return future

    dispatcher = LiveASRDispatcher(
        ASRConfig(model_size="gigaam-v3-e2e-ctc"),
        LiveGigaAMConfig(worker_count=1, max_retries=1),
    )
    dispatcher._started = True
    dispatcher._workers = [
        _WorkerSlot(
            worker_id=0,
            executor=_SyncFailExecutor(
                [
                    RuntimeError("broken submit"),
                    LiveGigaAMResult(
                        seq_id=0,
                        source="microphone",
                        start_s=0.0,
                        end_s=1.0,
                        text="ok",
                        worker_id=0,
                    ),
                ]
            ),
        )
    ]
    job = LiveGigaAMJob(
        seq_id=0,
        source="microphone",
        start_s=0.0,
        end_s=1.0,
        sample_rate=16000,
        samples=[],
    )

    result = asyncio.run(dispatcher.transcribe(job))

    assert result.text == "ok"
    assert dispatcher._workers[0].failed == 1
    assert dispatcher._workers[0].completed == 1
    assert dispatcher._workers[0].in_flight == 0


def test_shutdown_clears_worker_slots() -> None:
    dispatcher = LiveASRDispatcher(
        ASRConfig(model_size="gigaam-v3-e2e-ctc"),
        LiveGigaAMConfig(worker_count=1),
    )
    dispatcher._started = True
    dispatcher._workers = [_WorkerSlot(worker_id=0, executor=_FakeExecutor([]))]

    asyncio.run(dispatcher.shutdown())

    assert dispatcher._workers == []
    assert dispatcher._started is False


def test_dispatcher_start_warms_all_workers(monkeypatch) -> None:
    created: list[dict[str, object]] = []

    class _FakeProcessPoolExecutor:
        def __init__(self, *, max_workers, mp_context, initializer, initargs) -> None:
            del max_workers, mp_context, initializer
            self.worker_id = int(initargs[0])
            created.append({"worker_id": self.worker_id, "initargs": initargs})

        def submit(self, fn, *args):
            del args
            future: Future = Future()
            if fn is live_dispatcher.ping_worker:
                future.set_result(self.worker_id)
            else:
                future.set_result(None)
            return future

        def shutdown(self, **_kwargs) -> None:
            return None

    monkeypatch.setattr(live_dispatcher, "ProcessPoolExecutor", _FakeProcessPoolExecutor)
    monkeypatch.setattr(
        live_dispatcher.multiprocessing,
        "get_context",
        lambda _name: SimpleNamespace(name="spawn"),
    )

    dispatcher = LiveASRDispatcher(
        ASRConfig(model_size="gigaam-v3-e2e-ctc", cpu_threads=6),
        LiveGigaAMConfig(worker_count=2, threads_per_worker=None),
    )

    asyncio.run(dispatcher.start())

    assert dispatcher._started is True
    assert len(dispatcher._workers) == 2
    assert [item["worker_id"] for item in created] == [0, 1]
    assert created[0]["initargs"][2] == 3


def test_concurrent_dispatch_uses_multiple_workers_under_load() -> None:
    class _DelayedExecutor:
        def __init__(self, worker_id: int, delay_s: float) -> None:
            self.worker_id = worker_id
            self.delay_s = delay_s

        def submit(self, _fn, job):
            future: Future = Future()

            def _resolve() -> None:
                future.set_result(
                    LiveGigaAMResult(
                        seq_id=job.seq_id,
                        source=job.source,
                        start_s=job.start_s,
                        end_s=job.end_s,
                        text=f"worker {self.worker_id}",
                        worker_id=self.worker_id,
                    )
                )

            timer = threading.Timer(self.delay_s, _resolve)
            timer.daemon = True
            timer.start()
            return future

        def shutdown(self, **_kwargs) -> None:
            return None

    dispatcher = LiveASRDispatcher(
        ASRConfig(model_size="gigaam-v3-e2e-ctc"),
        LiveGigaAMConfig(worker_count=2),
    )
    dispatcher._started = True
    dispatcher._workers = [
        _WorkerSlot(worker_id=0, executor=_DelayedExecutor(worker_id=0, delay_s=0.05)),
        _WorkerSlot(worker_id=1, executor=_DelayedExecutor(worker_id=1, delay_s=0.01)),
    ]

    async def _run_jobs() -> list[LiveGigaAMResult]:
        job0 = LiveGigaAMJob(
            seq_id=0,
            source="microphone",
            start_s=0.0,
            end_s=1.0,
            sample_rate=16000,
            samples=[],
        )
        job1 = LiveGigaAMJob(
            seq_id=1,
            source="microphone",
            start_s=1.0,
            end_s=2.0,
            sample_rate=16000,
            samples=[],
        )
        return await asyncio.gather(
            dispatcher.transcribe(job0),
            dispatcher.transcribe(job1),
        )

    results = asyncio.run(_run_jobs())

    assert sorted(result.worker_id for result in results) == [0, 1]
    assert dispatcher._workers[0].completed == 1
    assert dispatcher._workers[1].completed == 1
