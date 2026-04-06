"""Live GigaAM session controller with draft and stop-time finalization."""

from __future__ import annotations

import asyncio
import shutil
import threading
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from voxfusion.capture.mixer import AudioMixer
from voxfusion.capture.vad_chunker import VadChunker
from voxfusion.capture.windows_factory import create_windows_capture_source
from voxfusion.config.models import PipelineConfig
from voxfusion.live_gigaam.commit import OrderedTranscriptCommitter
from voxfusion.live_gigaam.dispatcher import LiveASRDispatcher
from voxfusion.live_gigaam.spool import SessionAudioSpool, SpoolingCaptureSource
from voxfusion.live_gigaam.types import LiveGigaAMJob, LiveGigaAMResult, LiveUtterance
from voxfusion.logging import get_logger
from voxfusion.models.translation import TranslatedSegment

log = get_logger(__name__)

_QUEUE_SENTINEL = object()


def _derive_capture_source(
    microphone_device_id: str | int | None,
    system_device_id: str | int | None,
) -> str:
    if microphone_device_id is not None and system_device_id is not None:
        return "both"
    if system_device_id is not None:
        return "system"
    if microphone_device_id is not None:
        return "microphone"
    return "none"


class LiveGigaAMSessionController:
    """Run live GigaAM over VAD-bounded utterances and finalize on stop."""

    def __init__(
        self,
        *,
        config: PipelineConfig,
        microphone_device_id: str | int | None,
        system_device_id: str | int | None,
        on_status: Callable[[str], None],
        on_segments: Callable[[list[TranslatedSegment]], None],
        on_finalized_segments: Callable[[list[TranslatedSegment]], None] | None = None,
        on_capture_started: Callable[[datetime], None] | None = None,
        requested_source: str | None = None,
    ) -> None:
        self._config = config
        self._microphone_device_id = microphone_device_id
        self._system_device_id = system_device_id
        self._on_status = on_status
        self._on_segments = on_segments
        self._on_finalized_segments = on_finalized_segments
        self._on_capture_started = on_capture_started
        self._requested_source = requested_source
        self._dispatcher = LiveASRDispatcher(config.asr, config.live_gigaam)
        self._draft_committer = OrderedTranscriptCommitter(config.diarization.channel_map)
        self._result_queue: asyncio.Queue[LiveGigaAMResult | object] = asyncio.Queue()
        self._pending_tasks: set[asyncio.Task[None]] = set()
        self._utterances: list[LiveUtterance] = []
        self._draft_results: dict[int, LiveGigaAMResult] = {}
        self._spool: SessionAudioSpool | None = None
        self._submitted_jobs = 0
        self._completed_jobs = 0
        self._capture_source_name = "none"
        self._backlog_peak_jobs = 0
        self._deferred_draft_jobs = 0
        self._completed_draft_jobs = 0
        self._stop_reprocessed_utterances = 0
        self._warning_active = False
        self._deferral_active = False

    def get_stats(self) -> dict[str, int]:
        dispatcher_stats = self._dispatcher.get_stats()
        pending = dispatcher_stats["pending"]
        workers = dispatcher_stats["workers"]
        return {
            "preprocess_q": max(0, self._submitted_jobs - self._completed_jobs - pending),
            "asr_q": pending,
            "in_asr": min(pending, workers),
            "dropped": 0,
            "backlog_peak": self._backlog_peak_jobs,
            "deferred_drafts": self._deferred_draft_jobs,
            "completed_drafts": self._completed_draft_jobs,
        }

    async def run(self, stop_event: threading.Event) -> list[TranslatedSegment]:
        self._capture_source_name = self._requested_source or _derive_capture_source(
            self._microphone_device_id,
            self._system_device_id,
        )
        if self._capture_source_name == "none":
            raise RuntimeError("No live capture devices selected for GigaAM.")

        session_dir = self._session_dir()
        audio_source: Any | None = None
        collector_task: asyncio.Task[None] | None = None
        finalized_segments: list[TranslatedSegment] = []
        try:
            self._spool = SessionAudioSpool(session_dir)
            await self._dispatcher.start()
            self._on_status("GigaAM workers ready. Initializing live capture...")
            audio_source = self._build_audio_source()
            await audio_source.start()
            capture_started_at = datetime.now()
            if self._on_capture_started is not None:
                self._on_capture_started(capture_started_at)
            log.info(
                "live_gigaam.capture_started",
                source=self._capture_source_name,
                session_dir=str(session_dir),
            )
            collector_task = asyncio.create_task(self._collect_draft_results())
            self._on_status("Live GigaAM started. Waiting for speech...")
            try:
                async for utterance_chunk in audio_source.stream(
                    chunk_duration_ms=self._config.live_gigaam.utterance_max_duration_ms
                ):
                    if stop_event.is_set():
                        break
                    utterance = LiveUtterance(
                        seq_id=len(self._utterances),
                        source=utterance_chunk.source,
                        start_s=utterance_chunk.timestamp_start,
                        end_s=utterance_chunk.timestamp_end,
                        sample_rate=utterance_chunk.sample_rate,
                        samples=utterance_chunk.samples,
                    )
                    self._utterances.append(utterance)
                    self._submitted_jobs += 1
                    pending = self._dispatcher.pending_jobs
                    draft_backlog_before = max(0, self._submitted_jobs - self._completed_jobs - 1)
                    should_defer_draft = self._should_defer_draft(draft_backlog_before)
                    draft_backlog_jobs = draft_backlog_before + 1
                    self._backlog_peak_jobs = max(self._backlog_peak_jobs, draft_backlog_jobs)
                    log.info(
                        "live_gigaam.utterance_ready",
                        seq_id=utterance.seq_id,
                        source=utterance.source,
                        start_s=round(utterance.start_s, 2),
                        end_s=round(utterance.end_s, 2),
                        duration_ms=round((utterance.end_s - utterance.start_s) * 1000),
                        pending_jobs=pending,
                        draft_backlog_jobs=draft_backlog_jobs,
                        deferred=should_defer_draft,
                    )
                    self._update_backlog_status(
                        draft_backlog_jobs=draft_backlog_jobs,
                        pending_jobs=pending,
                        seq_id=utterance.seq_id,
                        deferred=should_defer_draft,
                    )
                    if should_defer_draft:
                        self._deferred_draft_jobs += 1
                        await self._result_queue.put(self._deferred_result(utterance))
                    else:
                        task = asyncio.create_task(self._submit_draft_job(utterance))
                        self._pending_tasks.add(task)
                        task.add_done_callback(self._pending_tasks.discard)
            finally:
                if audio_source is not None:
                    with suppress(Exception):
                        await audio_source.stop()

            if self._pending_tasks:
                await asyncio.gather(*self._pending_tasks, return_exceptions=True)
            if collector_task is not None:
                await self._result_queue.put(_QUEUE_SENTINEL)
                with suppress(asyncio.CancelledError, Exception):
                    await collector_task
                collector_task = None

            finalized_segments = self._draft_committer.committed_segments
            if self._utterances:
                finalized_segments = await self._finalize_from_spool()
                if self._on_finalized_segments is not None:
                    self._on_finalized_segments(finalized_segments)
                if self._deferred_draft_jobs:
                    self._on_status(
                        f"Live GigaAM finalized: {len(finalized_segments)} segments. "
                        f"Deferred {self._deferred_draft_jobs} draft utterances during overload."
                    )
                else:
                    self._on_status(
                        f"Live GigaAM finalized: {len(finalized_segments)} segments."
                    )
            log.info(
                "live_gigaam.capture_completed",
                utterances=len(self._utterances),
                committed_segments=len(finalized_segments),
                backlog_peak_jobs=self._backlog_peak_jobs,
                deferred_draft_jobs=self._deferred_draft_jobs,
                completed_draft_jobs=self._completed_draft_jobs,
                stop_reprocessed_utterances=self._stop_reprocessed_utterances,
            )
            return finalized_segments
        finally:
            if collector_task is not None:
                await self._result_queue.put(_QUEUE_SENTINEL)
                with suppress(asyncio.CancelledError, Exception):
                    await collector_task
            await self._dispatcher.shutdown()
            if self._spool is not None:
                self._spool.close()
            if self._config.security.auto_delete_temp_files:
                shutil.rmtree(session_dir, ignore_errors=True)

    async def _submit_draft_job(self, utterance: LiveUtterance) -> None:
        job = LiveGigaAMJob(
            seq_id=utterance.seq_id,
            source=utterance.source,
            start_s=utterance.start_s,
            end_s=utterance.end_s,
            sample_rate=utterance.sample_rate,
            samples=utterance.samples,
            finalize=False,
        )
        result = await self._dispatcher.transcribe(job)
        await self._result_queue.put(result)

    async def _collect_draft_results(self) -> None:
        while True:
            item = await self._result_queue.get()
            if item is _QUEUE_SENTINEL:
                break
            result = item  # type: ignore[assignment]
            self._completed_jobs += 1
            if isinstance(result, LiveGigaAMResult):
                if not result.deferred:
                    self._draft_results[result.seq_id] = result
                if result.error:
                    log.error(
                        "live_gigaam.job_failed",
                        seq_id=result.seq_id,
                        worker_id=result.worker_id,
                        error=result.error,
                    )
                else:
                    self._completed_draft_jobs += 1
                    log.info(
                        "live_gigaam.job_done",
                        seq_id=result.seq_id,
                        worker_id=result.worker_id,
                        finalize=result.finalize,
                        text_chars=len(result.text),
                    )
                outcome = self._draft_committer.accept(result)
                if outcome.emitted:
                    self._on_segments(outcome.emitted)

    async def _finalize_from_spool(self) -> list[TranslatedSegment]:
        assert self._spool is not None
        final_committer = OrderedTranscriptCommitter(self._config.diarization.channel_map)
        tasks = {
            utterance.seq_id: asyncio.create_task(self._finalize_utterance(utterance))
            for utterance in self._utterances
            if self._needs_stop_finalize(utterance)
        }
        self._stop_reprocessed_utterances = len(tasks)
        total_utterances = len(self._utterances)
        log.info(
            "live_gigaam.stop_finalize_plan",
            stop_finalize_mode=self._config.live_gigaam.stop_finalize_mode,
            utterances=total_utterances,
            stop_reprocessed_utterances=self._stop_reprocessed_utterances,
            deferred_draft_jobs=self._deferred_draft_jobs,
        )
        if self._stop_reprocessed_utterances == total_utterances:
            self._on_status("Finalizing GigaAM transcript...")
        elif self._stop_reprocessed_utterances > 0:
            self._on_status(
                f"Finalizing GigaAM transcript "
                f"({self._stop_reprocessed_utterances}/{total_utterances} utterances need recovery)..."
            )
        try:
            results_by_seq: dict[int, LiveGigaAMResult] = {}
            if tasks:
                results = await asyncio.gather(*tasks.values())
                results_by_seq = {result.seq_id: result for result in results}
            for utterance in self._utterances:
                result = results_by_seq.get(utterance.seq_id) or self._reuse_draft_result(utterance)
                outcome = final_committer.accept(result)
                if result.error:
                    log.warning(
                        "live_gigaam.finalize_fallback",
                        seq_id=result.seq_id,
                        error=result.error,
                    )
                if outcome.emitted:
                    log.info(
                        "live_gigaam.finalize_commit",
                        emitted=len(outcome.emitted),
                        total=len(outcome.committed),
                    )
        finally:
            for task in tasks.values():
                if not task.done():
                    task.cancel()
                    with suppress(asyncio.CancelledError):
                        await task
        return final_committer.committed_segments

    async def _finalize_utterance(self, utterance: LiveUtterance) -> LiveGigaAMResult:
        assert self._spool is not None
        window_start, window_end = self._context_bounds_for_utterance(utterance)
        samples = self._spool.read_window(utterance.source, window_start, window_end)
        job = LiveGigaAMJob(
            seq_id=utterance.seq_id,
            source=utterance.source,
            start_s=utterance.start_s,
            end_s=utterance.end_s,
            sample_rate=self._spool.sample_rate,
            samples=samples,
            finalize=True,
        )
        result = await self._dispatcher.transcribe(job)
        if result.error:
            draft = self._draft_results.get(utterance.seq_id)
            if draft is not None and draft.text:
                log.warning(
                    "live_gigaam.finalize_fallback",
                    seq_id=utterance.seq_id,
                    error=result.error,
                )
                return LiveGigaAMResult(
                    seq_id=draft.seq_id,
                    source=draft.source,
                    start_s=draft.start_s,
                    end_s=draft.end_s,
                    text=draft.text,
                    worker_id=result.worker_id,
                    finalize=True,
                    error=None,
                )
        return result

    def _needs_stop_finalize(self, utterance: LiveUtterance) -> bool:
        if self._config.live_gigaam.stop_finalize_mode == "always":
            return True
        draft = self._draft_results.get(utterance.seq_id)
        if draft is None:
            return True
        return bool(draft.error or not draft.text.strip())

    def _reuse_draft_result(self, utterance: LiveUtterance) -> LiveGigaAMResult:
        draft = self._draft_results.get(utterance.seq_id)
        if draft is None:
            raise RuntimeError(
                f"Missing draft result for utterance {utterance.seq_id} during selective stop finalization."
            )
        return LiveGigaAMResult(
            seq_id=draft.seq_id,
            source=draft.source,
            start_s=draft.start_s,
            end_s=draft.end_s,
            text=draft.text,
            worker_id=draft.worker_id,
            finalize=True,
            error=None,
        )

    def _should_defer_draft(self, draft_backlog_before_current: int) -> bool:
        hard_limit = self._config.live_gigaam.queue_hard_limit_jobs
        if hard_limit <= 0:
            return False
        return draft_backlog_before_current >= hard_limit

    def _deferred_result(self, utterance: LiveUtterance) -> LiveGigaAMResult:
        return LiveGigaAMResult(
            seq_id=utterance.seq_id,
            source=utterance.source,
            start_s=utterance.start_s,
            end_s=utterance.end_s,
            text="",
            worker_id=-1,
            deferred=True,
        )

    def _update_backlog_status(
        self,
        *,
        draft_backlog_jobs: int,
        pending_jobs: int,
        seq_id: int,
        deferred: bool,
    ) -> None:
        warning_limit = self._config.live_gigaam.queue_warning_jobs
        hard_limit = self._config.live_gigaam.queue_hard_limit_jobs

        if warning_limit > 0 and draft_backlog_jobs >= warning_limit:
            if not self._warning_active:
                self._on_status(
                    f"Live GigaAM backlog: {draft_backlog_jobs} draft utterances queued. "
                    "Draft latency may grow."
                )
                log.warning(
                    "live_gigaam.backlog_warning",
                    draft_backlog_jobs=draft_backlog_jobs,
                    pending_jobs=pending_jobs,
                    queue_warning_jobs=warning_limit,
                )
                self._warning_active = True
        elif self._warning_active:
            log.info(
                "live_gigaam.backlog_recovered",
                draft_backlog_jobs=draft_backlog_jobs,
                pending_jobs=pending_jobs,
                queue_warning_jobs=warning_limit,
            )
            self._warning_active = False

        if deferred:
            if not self._deferral_active:
                self._on_status(
                    "Live GigaAM overloaded. Deferring new draft utterances; "
                    "final transcript will be recovered after stop."
                )
                log.warning(
                    "live_gigaam.draft_deferral_started",
                    draft_backlog_jobs=draft_backlog_jobs,
                    pending_jobs=pending_jobs,
                    queue_hard_limit_jobs=hard_limit,
                )
                self._deferral_active = True
            log.warning(
                "live_gigaam.draft_deferred",
                seq_id=seq_id,
                draft_backlog_jobs=draft_backlog_jobs,
                pending_jobs=pending_jobs,
                queue_hard_limit_jobs=hard_limit,
            )
        elif self._deferral_active and (hard_limit <= 0 or draft_backlog_jobs < hard_limit):
            self._on_status("Live GigaAM backlog recovered. Draft transcription resumed.")
            log.info(
                "live_gigaam.draft_deferral_ended",
                draft_backlog_jobs=draft_backlog_jobs,
                pending_jobs=pending_jobs,
                queue_hard_limit_jobs=hard_limit,
            )
            self._deferral_active = False

    def _context_bounds_for_utterance(self, utterance: LiveUtterance) -> tuple[float, float]:
        left_context_s = self._config.live_gigaam.finalize_left_context_ms / 1000.0
        right_context_s = self._config.live_gigaam.finalize_right_context_ms / 1000.0

        previous_end: float | None = None
        next_start: float | None = None
        for candidate in self._utterances:
            if candidate.source != utterance.source or candidate.seq_id == utterance.seq_id:
                continue
            if candidate.seq_id < utterance.seq_id:
                previous_end = candidate.end_s
                continue
            next_start = candidate.start_s
            break

        window_start = max(0.0, utterance.start_s - left_context_s)
        if previous_end is not None:
            window_start = max(window_start, previous_end)

        window_end = utterance.end_s + right_context_s
        if next_start is not None:
            window_end = min(window_end, next_start)
        if window_end <= window_start:
            window_start = utterance.start_s
            window_end = utterance.end_s
        return window_start, window_end

    def _build_audio_source(self) -> Any:
        assert self._spool is not None
        if self._capture_source_name == "both":
            base_source = create_windows_capture_source(
                "both",
                self._config.capture,
                microphone_device_id=self._microphone_device_id,
                system_device_id=self._system_device_id,
            )
            if not isinstance(base_source, AudioMixer):
                raise RuntimeError("Expected AudioMixer for live GigaAM 'both' capture source.")
            mic_source = VadChunker(
                SpoolingCaptureSource(base_source._sources[0], self._spool),
                max_duration_ms=self._config.live_gigaam.utterance_max_duration_ms,
            )
            sys_source = VadChunker(
                SpoolingCaptureSource(base_source._sources[1], self._spool),
                max_duration_ms=self._config.live_gigaam.utterance_max_duration_ms,
            )
            return AudioMixer([mic_source, sys_source])

        raw_source = create_windows_capture_source(
            self._capture_source_name,
            self._config.capture,
            microphone_device_id=self._microphone_device_id,
            system_device_id=self._system_device_id,
        )
        return VadChunker(
            SpoolingCaptureSource(raw_source, self._spool),
            max_duration_ms=self._config.live_gigaam.utterance_max_duration_ms,
        )

    def _session_dir(self) -> Path:
        root = Path(self._config.data_dir).expanduser() / "live_gigaam"
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return root / f"session_{stamp}"
