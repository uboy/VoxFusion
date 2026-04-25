"""Least-loaded live GigaAM dispatcher over warm worker processes."""

from __future__ import annotations

import asyncio
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

from voxfusion.config.models import ASRConfig, LiveGigaAMConfig
from voxfusion.live_gigaam.types import LiveGigaAMJob, LiveGigaAMResult
from voxfusion.live_gigaam.worker import init_worker, ping_worker, transcribe_job
from voxfusion.logging import get_logger

log = get_logger(__name__)


@dataclass
class _WorkerSlot:
    worker_id: int
    executor: ProcessPoolExecutor
    in_flight: int = 0
    completed: int = 0
    failed: int = 0


class LiveASRDispatcher:
    """Dispatch live GigaAM jobs to the least-busy warm worker process."""

    def __init__(
        self,
        asr_config: ASRConfig,
        live_config: LiveGigaAMConfig,
    ) -> None:
        self._asr_config = asr_config
        self._live_config = live_config
        self._workers: list[_WorkerSlot] = []
        self._started = False

    @property
    def pending_jobs(self) -> int:
        return sum(worker.in_flight for worker in self._workers)

    def get_stats(self) -> dict[str, int]:
        return {
            "workers": len(self._workers),
            "pending": self.pending_jobs,
            "completed": sum(worker.completed for worker in self._workers),
            "failed": sum(worker.failed for worker in self._workers),
        }

    async def start(self) -> None:
        if self._started:
            return
        worker_count = max(1, int(self._live_config.worker_count))
        threads_per_worker = self._resolve_threads_per_worker(worker_count)
        payload = self._asr_config.model_dump()
        ctx = multiprocessing.get_context("spawn")
        self._workers = []
        for worker_id in range(worker_count):
            executor = ProcessPoolExecutor(
                max_workers=1,
                mp_context=ctx,
                initializer=init_worker,
                initargs=(worker_id, payload, threads_per_worker),
            )
            self._workers.append(_WorkerSlot(worker_id=worker_id, executor=executor))
        await self._warm_up()
        self._started = True

    async def shutdown(self) -> None:
        for worker in self._workers:
            worker.executor.shutdown(wait=False, cancel_futures=True)
        self._workers = []
        self._started = False

    async def transcribe(self, job: LiveGigaAMJob) -> LiveGigaAMResult:
        if not self._started:
            await self.start()

        current_job = job
        while True:
            worker = self._choose_worker()
            worker.in_flight += 1
            log.info(
                "live_gigaam.job_dispatched",
                seq_id=current_job.seq_id,
                finalize=current_job.finalize,
                worker_id=worker.worker_id,
                pending=worker.in_flight,
                source=current_job.source,
            )
            try:
                future = worker.executor.submit(transcribe_job, current_job)
                wrapped = asyncio.wrap_future(future)
                result = await wrapped
                worker.completed += 1
                return result
            except Exception as exc:
                worker.failed += 1
                if current_job.retry_count >= self._live_config.max_retries:
                    return LiveGigaAMResult(
                        seq_id=current_job.seq_id,
                        source=current_job.source,
                        start_s=current_job.start_s,
                        end_s=current_job.end_s,
                        text="",
                        worker_id=worker.worker_id,
                        finalize=current_job.finalize,
                        error=str(exc),
                    )
                current_job = LiveGigaAMJob(
                    seq_id=current_job.seq_id,
                    source=current_job.source,
                    start_s=current_job.start_s,
                    end_s=current_job.end_s,
                    sample_rate=current_job.sample_rate,
                    samples=current_job.samples,
                    finalize=current_job.finalize,
                    retry_count=current_job.retry_count + 1,
                )
                log.warning(
                    "live_gigaam.job_retry",
                    seq_id=current_job.seq_id,
                    retry_count=current_job.retry_count,
                    worker_id=worker.worker_id,
                    error=str(exc),
                )
            finally:
                worker.in_flight = max(0, worker.in_flight - 1)

    def _choose_worker(self) -> _WorkerSlot:
        return min(
            self._workers,
            key=lambda worker: (
                worker.in_flight,
                worker.completed + worker.failed,
                worker.worker_id,
            ),
        )

    def _resolve_threads_per_worker(self, worker_count: int) -> int:
        explicit = self._live_config.threads_per_worker
        if explicit is not None:
            return max(1, int(explicit))
        cpu_total = self._asr_config.cpu_threads or os.cpu_count() or 4
        return max(1, int(cpu_total) // max(1, worker_count))

    async def _warm_up(self) -> None:
        loop = asyncio.get_running_loop()
        futures = [
            asyncio.wrap_future(worker.executor.submit(ping_worker), loop=loop)
            for worker in self._workers
        ]
        ready = await asyncio.gather(*futures)
        log.info("live_gigaam.workers_ready", workers=len(ready), worker_ids=ready)
