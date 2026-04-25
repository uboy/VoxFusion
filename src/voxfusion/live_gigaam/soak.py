"""Manual soak/load harness for long-running live GigaAM validation."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import threading
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf
from numpy.typing import NDArray

from voxfusion.config.models import PipelineConfig
from voxfusion.live_gigaam.dispatcher import LiveASRDispatcher
from voxfusion.live_gigaam.session import LiveGigaAMSessionController
from voxfusion.live_gigaam.types import LiveGigaAMJob, LiveGigaAMResult
from voxfusion.logging import configure_logging, get_logger
from voxfusion.models.audio import AudioChunk

log = get_logger(__name__)

_ALLOWED_SOURCES = frozenset({"microphone", "system"})
_TARGET_SAMPLE_RATE = 16000


@dataclass(frozen=True)
class CorpusUtterance:
    """One replayable utterance-shaped waveform for the harness."""

    samples: NDArray[np.float32]
    sample_rate: int
    duration_s: float
    source: str


@dataclass
class DispatcherMetrics:
    """Aggregated dispatcher metrics captured during one harness run."""

    started_jobs: int = 0
    completed_jobs: int = 0
    failed_jobs: int = 0
    pending_high_water: int = 0
    worker_usage: Counter[int] = field(default_factory=Counter)
    latencies_ms: list[float] = field(default_factory=list)


@dataclass(frozen=True)
class SoakSummary:
    """JSON-friendly result of one soak/load run."""

    mode: str
    requested_duration_s: float
    elapsed_s: float
    replay_pace: float
    corpus_utterances: int
    corpus_audio_s: float
    submitted_utterances: int
    completed_dispatch_jobs: int
    completed_drafts: int
    deferred_drafts: int
    backlog_peak: int
    draft_segments: int
    final_segments: int
    final_text_chars: int
    dispatcher_failures: int
    dispatcher_pending_high_water: int
    latency_ms: dict[str, float | int | None]
    worker_usage: dict[str, int]
    status_tail: list[str]
    data_dir: str
    wav_path: str
    model_path: str | None

    def to_jsonable(self) -> dict[str, object]:
        return asdict(self)


def normalize_sources(value: str | Sequence[str]) -> tuple[str, ...]:
    """Normalize CLI source input into a tuple of replay sources."""
    if isinstance(value, str):
        raw_items = [item.strip().lower() for item in value.split(",") if item.strip()]
    else:
        raw_items = [str(item).strip().lower() for item in value if str(item).strip()]
    if not raw_items:
        raise ValueError("At least one source must be provided.")

    normalized: list[str] = []
    for item in raw_items:
        if item == "both":
            normalized.extend(["microphone", "system"])
            continue
        if item not in _ALLOWED_SOURCES:
            raise ValueError(f"Unsupported soak source '{item}'. Expected microphone/system/both.")
        normalized.append(item)
    return tuple(dict.fromkeys(normalized))


def summarize_latencies_ms(values: Sequence[float]) -> dict[str, float | int | None]:
    """Return compact latency statistics for JSON output."""
    if not values:
        return {
            "count": 0,
            "avg": None,
            "p50": None,
            "p95": None,
            "max": None,
        }
    sorted_values = sorted(float(value) for value in values)
    return {
        "count": len(sorted_values),
        "avg": round(sum(sorted_values) / len(sorted_values), 2),
        "p50": round(_percentile(sorted_values, 0.50), 2),
        "p95": round(_percentile(sorted_values, 0.95), 2),
        "max": round(sorted_values[-1], 2),
    }


def build_utterance_corpus(
    samples: NDArray[np.float32],
    sample_rate: int,
    *,
    min_duration_s: float,
    max_duration_s: float,
    sources: Sequence[str],
    seed: int,
) -> list[CorpusUtterance]:
    """Split one waveform into a repeatable utterance corpus for replay."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be > 0")
    if min_duration_s <= 0 or max_duration_s <= 0:
        raise ValueError("min_duration_s and max_duration_s must be > 0")
    if max_duration_s < min_duration_s:
        raise ValueError("max_duration_s must be >= min_duration_s")
    normalized_sources = normalize_sources(sources)

    mono = _normalize_audio(samples, sample_rate)
    if mono.size == 0:
        raise ValueError("The input waveform is empty after normalization.")

    min_samples = max(1, int(round(min_duration_s * _TARGET_SAMPLE_RATE)))
    max_samples = max(min_samples, int(round(max_duration_s * _TARGET_SAMPLE_RATE)))

    if mono.size <= min_samples:
        return [
            CorpusUtterance(
                samples=mono.copy(),
                sample_rate=_TARGET_SAMPLE_RATE,
                duration_s=round(mono.size / _TARGET_SAMPLE_RATE, 3),
                source=normalized_sources[0],
            )
        ]

    rng = np.random.default_rng(seed)
    corpus: list[CorpusUtterance] = []
    position = 0
    source_index = 0

    while position < mono.size:
        remaining = mono.size - position
        if remaining < min_samples and corpus:
            break
        if remaining <= max_samples:
            end = mono.size
        else:
            span = int(rng.integers(min_samples, max_samples + 1))
            end = min(mono.size, position + span)
            tail = mono.size - end
            if 0 < tail < min_samples:
                end = mono.size

        chunk = mono[position:end].copy()
        corpus.append(
            CorpusUtterance(
                samples=chunk,
                sample_rate=_TARGET_SAMPLE_RATE,
                duration_s=round(chunk.size / _TARGET_SAMPLE_RATE, 3),
                source=normalized_sources[source_index % len(normalized_sources)],
            )
        )
        position = end
        source_index += 1

    if not corpus:
        raise ValueError("Unable to build a replay corpus from the provided waveform.")
    return corpus


class ReplayAudioSource:
    """Fake live source that replays utterance-sized chunks for wall-clock soak runs."""

    def __init__(
        self,
        controller: LiveGigaAMSessionController,
        corpus: Sequence[CorpusUtterance],
        *,
        duration_s: float,
        pace: float,
        idle_gap_s: float,
    ) -> None:
        if duration_s <= 0:
            raise ValueError("duration_s must be > 0")
        if pace <= 0:
            raise ValueError("pace must be > 0")
        if idle_gap_s < 0:
            raise ValueError("idle_gap_s must be >= 0")
        if not corpus:
            raise ValueError("corpus must not be empty")
        self._controller = controller
        self._corpus = list(corpus)
        self._duration_s = duration_s
        self._pace = pace
        self._idle_gap_s = idle_gap_s
        self.device_name = "soak:replay"
        self.sample_rate = _TARGET_SAMPLE_RATE
        self.channels = 1
        self.is_active = False

    async def start(self) -> None:
        self.is_active = True

    async def stop(self) -> None:
        self.is_active = False

    async def stream(self, chunk_duration_ms: int = 5000):
        del chunk_duration_ms
        assert self._controller._spool is not None
        started_at = time.monotonic()
        deadline = started_at + self._duration_s
        next_emit_at = started_at
        timeline_s = 0.0
        utterance_index = 0

        while time.monotonic() < deadline:
            utterance = self._corpus[utterance_index % len(self._corpus)]
            delay_s = next_emit_at - time.monotonic()
            if delay_s > 0:
                await asyncio.sleep(delay_s)
            if time.monotonic() >= deadline:
                break

            start_s = timeline_s
            end_s = start_s + utterance.duration_s
            chunk = AudioChunk(
                samples=utterance.samples,
                sample_rate=utterance.sample_rate,
                channels=1,
                timestamp_start=start_s,
                timestamp_end=end_s,
                source=utterance.source,
                dtype="float32",
            )
            yield self._controller._spool.append(chunk)

            timeline_s = end_s + self._idle_gap_s
            next_emit_at += (utterance.duration_s + self._idle_gap_s) / self._pace
            utterance_index += 1


class InstrumentedDispatcher:
    """Wrapper that records dispatcher load/latency metrics."""

    def __init__(self, inner: LiveASRDispatcher | FakeDispatcher) -> None:
        self._inner = inner
        self.metrics = DispatcherMetrics()

    @property
    def pending_jobs(self) -> int:
        return self._inner.pending_jobs

    async def start(self) -> None:
        await self._inner.start()
        self.metrics.pending_high_water = max(self.metrics.pending_high_water, self.pending_jobs)

    async def shutdown(self) -> None:
        await self._inner.shutdown()

    def get_stats(self) -> dict[str, int]:
        stats = dict(self._inner.get_stats())
        stats["pending_high_water"] = self.metrics.pending_high_water
        return stats

    async def transcribe(self, job: LiveGigaAMJob) -> LiveGigaAMResult:
        self.metrics.started_jobs += 1
        self.metrics.pending_high_water = max(
            self.metrics.pending_high_water, self.pending_jobs + 1
        )
        started_at = time.perf_counter()
        try:
            result = await self._inner.transcribe(job)
        except Exception:
            self.metrics.failed_jobs += 1
            raise
        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        self.metrics.latencies_ms.append(elapsed_ms)
        self.metrics.pending_high_water = max(self.metrics.pending_high_water, self.pending_jobs)
        if result.error:
            self.metrics.failed_jobs += 1
        else:
            self.metrics.completed_jobs += 1
        if result.worker_id >= 0:
            self.metrics.worker_usage[result.worker_id] += 1
        return result


class FakeDispatcher:
    """Cheap fake dispatcher for harness smoke runs without a real model."""

    def __init__(
        self,
        *,
        worker_count: int,
        delay_ms: int,
        jitter_ms: int,
        fail_every: int,
        seed: int,
    ) -> None:
        self._worker_count = max(1, int(worker_count))
        self._delay_ms = max(0, int(delay_ms))
        self._jitter_ms = max(0, int(jitter_ms))
        self._fail_every = max(0, int(fail_every))
        self._rng = np.random.default_rng(seed)
        self._pending_jobs = 0
        self._completed = 0
        self._failed = 0
        self._cursor = 0
        self._started = False

    @property
    def pending_jobs(self) -> int:
        return self._pending_jobs

    async def start(self) -> None:
        self._started = True

    async def shutdown(self) -> None:
        self._started = False

    def get_stats(self) -> dict[str, int]:
        return {
            "workers": self._worker_count,
            "pending": self._pending_jobs,
            "completed": self._completed,
            "failed": self._failed,
        }

    async def transcribe(self, job: LiveGigaAMJob) -> LiveGigaAMResult:
        if not self._started:
            await self.start()
        self._pending_jobs += 1
        self._cursor += 1
        worker_id = (self._cursor - 1) % self._worker_count
        try:
            if self._delay_ms or self._jitter_ms:
                jitter = int(self._rng.integers(0, self._jitter_ms + 1)) if self._jitter_ms else 0
                await asyncio.sleep((self._delay_ms + jitter) / 1000.0)
            if self._fail_every and self._cursor % self._fail_every == 0:
                self._failed += 1
                return LiveGigaAMResult(
                    seq_id=job.seq_id,
                    source=job.source,
                    start_s=job.start_s,
                    end_s=job.end_s,
                    text="",
                    worker_id=worker_id,
                    finalize=job.finalize,
                    error="synthetic soak failure",
                )
            self._completed += 1
            prefix = "final" if job.finalize else "draft"
            return LiveGigaAMResult(
                seq_id=job.seq_id,
                source=job.source,
                start_s=job.start_s,
                end_s=job.end_s,
                text=f"{prefix} {job.seq_id}",
                worker_id=worker_id,
                finalize=job.finalize,
            )
        finally:
            self._pending_jobs = max(0, self._pending_jobs - 1)


async def run_soak_harness(args: argparse.Namespace) -> SoakSummary:
    """Run one manual soak/load session and return its summary."""
    resolved_sources = normalize_sources(args.sources)
    wav_path = resolve_default_wav_path(args.wav)
    model_path = resolve_default_model_path(args.model_path)
    waveform, sample_rate = sf.read(str(wav_path), dtype="float32", always_2d=False)
    corpus = build_utterance_corpus(
        np.asarray(waveform, dtype=np.float32),
        int(sample_rate),
        min_duration_s=float(args.min_utterance_s),
        max_duration_s=float(args.max_utterance_s),
        sources=resolved_sources,
        seed=int(args.seed),
    )
    data_dir = Path(args.data_dir).expanduser()
    requested_source = "both" if len(resolved_sources) > 1 else resolved_sources[0]
    statuses: list[str] = []
    draft_segments: list[object] = []
    finalized_segments: list[object] = []

    config = PipelineConfig(
        asr={
            "model_size": "gigaam-v3-e2e-ctc",
            "model_path": str(model_path) if model_path is not None else None,
            "cpu_threads": int(args.cpu_threads),
        },
        live_gigaam={
            "worker_count": int(args.worker_count),
            "threads_per_worker": int(args.threads_per_worker)
            if args.threads_per_worker is not None
            else None,
            "utterance_max_duration_ms": int(round(float(args.max_utterance_s) * 1000.0)),
            "queue_warning_jobs": int(args.queue_warning_jobs),
            "queue_hard_limit_jobs": int(args.queue_hard_limit_jobs),
        },
        data_dir=str(data_dir),
        security={"auto_delete_temp_files": not bool(args.keep_spool)},
    )
    controller = LiveGigaAMSessionController(
        config=config,
        microphone_device_id="soak:microphone" if "microphone" in resolved_sources else None,
        system_device_id="soak:system" if "system" in resolved_sources else None,
        on_status=statuses.append,
        on_segments=lambda segments: draft_segments.extend(segments),
        on_finalized_segments=lambda segments: finalized_segments.extend(segments),
        requested_source=requested_source,
    )
    controller._build_audio_source = lambda: ReplayAudioSource(  # type: ignore[method-assign]
        controller,
        corpus,
        duration_s=float(args.duration_minutes) * 60.0,
        pace=float(args.pace),
        idle_gap_s=float(args.idle_gap_ms) / 1000.0,
    )

    inner_dispatcher: LiveASRDispatcher | FakeDispatcher
    if args.mode == "fake":
        inner_dispatcher = FakeDispatcher(
            worker_count=int(args.worker_count),
            delay_ms=int(args.fake_delay_ms),
            jitter_ms=int(args.fake_jitter_ms),
            fail_every=int(args.fake_fail_every),
            seed=int(args.seed),
        )
    else:
        inner_dispatcher = LiveASRDispatcher(config.asr, config.live_gigaam)
    instrumented = InstrumentedDispatcher(inner_dispatcher)
    controller._dispatcher = instrumented

    log.info(
        "live_gigaam.soak_start",
        mode=args.mode,
        duration_minutes=float(args.duration_minutes),
        pace=float(args.pace),
        worker_count=int(args.worker_count),
        queue_hard_limit_jobs=int(args.queue_hard_limit_jobs),
        wav_path=str(wav_path),
        model_path=str(model_path) if model_path is not None else None,
    )

    started_at = time.monotonic()
    progress_task = asyncio.create_task(
        _report_progress(
            controller=controller,
            dispatcher=instrumented,
            statuses=statuses,
            interval_s=float(args.report_interval_s),
            started_at=started_at,
        )
    )
    try:
        final_segments = await controller.run(threading.Event())
    finally:
        progress_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await progress_task
    elapsed_s = time.monotonic() - started_at

    if not finalized_segments:
        finalized_segments.extend(final_segments)
    summary = SoakSummary(
        mode=str(args.mode),
        requested_duration_s=round(float(args.duration_minutes) * 60.0, 2),
        elapsed_s=round(elapsed_s, 2),
        replay_pace=float(args.pace),
        corpus_utterances=len(corpus),
        corpus_audio_s=round(sum(item.duration_s for item in corpus), 2),
        submitted_utterances=controller._submitted_jobs,
        completed_dispatch_jobs=instrumented.metrics.completed_jobs,
        completed_drafts=controller._completed_draft_jobs,
        deferred_drafts=controller._deferred_draft_jobs,
        backlog_peak=controller._backlog_peak_jobs,
        draft_segments=len(draft_segments),
        final_segments=len(finalized_segments),
        final_text_chars=sum(
            len(getattr(segment.diarized.segment, "text", "") or "")
            for segment in finalized_segments
        ),
        dispatcher_failures=instrumented.metrics.failed_jobs,
        dispatcher_pending_high_water=instrumented.metrics.pending_high_water,
        latency_ms=summarize_latencies_ms(instrumented.metrics.latencies_ms),
        worker_usage={
            str(key): value for key, value in sorted(instrumented.metrics.worker_usage.items())
        },
        status_tail=statuses[-8:],
        data_dir=str(data_dir),
        wav_path=str(wav_path),
        model_path=str(model_path) if model_path is not None else None,
    )
    if args.json_out:
        json_path = Path(args.json_out).expanduser()
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(summary.to_jsonable(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        log.info("live_gigaam.soak_summary_written", path=str(json_path))
    log.info(
        "live_gigaam.soak_done",
        elapsed_s=summary.elapsed_s,
        submitted_utterances=summary.submitted_utterances,
        completed_dispatch_jobs=summary.completed_dispatch_jobs,
        deferred_drafts=summary.deferred_drafts,
        dispatcher_failures=summary.dispatcher_failures,
        latency_ms=summary.latency_ms,
    )
    return summary


async def _report_progress(
    *,
    controller: LiveGigaAMSessionController,
    dispatcher: InstrumentedDispatcher,
    statuses: list[str],
    interval_s: float,
    started_at: float,
) -> None:
    while True:
        await asyncio.sleep(max(0.1, interval_s))
        stats = controller.get_stats()
        log.info(
            "live_gigaam.soak_progress",
            elapsed_s=round(time.monotonic() - started_at, 1),
            submitted_jobs=controller._submitted_jobs,
            completed_jobs=controller._completed_jobs,
            backlog_peak=stats["backlog_peak"],
            deferred_drafts=stats["deferred_drafts"],
            pending_asr=stats["asr_q"],
            pending_high_water=dispatcher.metrics.pending_high_water,
            last_status=statuses[-1] if statuses else None,
        )


def resolve_default_wav_path(value: str | None) -> Path:
    """Resolve the WAV input, preferring repo-local samples for convenience."""
    if value:
        path = Path(value).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"WAV file not found: {path}")
        return path
    candidates = (
        Path("tmp_sample_10s.wav"),
        Path("tmp_sample_120s.wav"),
        Path("test_audio.wav"),
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError("No default local WAV found; pass --wav explicitly.")


def resolve_default_model_path(value: str | None) -> Path | None:
    """Resolve a local GigaAM snapshot when available."""
    if value:
        path = Path(value).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Model path not found: {path}")
        return path
    snapshots_root = Path("models") / "hub" / "models--ai-sage--GigaAM-v3" / "snapshots"
    if not snapshots_root.exists():
        return None
    snapshots = sorted(path for path in snapshots_root.iterdir() if path.is_dir())
    return snapshots[-1].resolve() if snapshots else None


def create_argument_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the standalone soak harness."""
    parser = argparse.ArgumentParser(
        description="Run a long live GigaAM soak/load harness over replayed local WAV utterances.",
    )
    parser.add_argument("--mode", choices=["real", "fake"], default="real")
    parser.add_argument("--duration-minutes", type=float, default=10.0)
    parser.add_argument("--pace", type=float, default=1.0)
    parser.add_argument("--wav", type=str, default=None)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default="build/live_gigaam_soak")
    parser.add_argument("--json-out", type=str, default=None)
    parser.add_argument("--sources", type=str, default="microphone,system")
    parser.add_argument("--min-utterance-s", type=float, default=2.0)
    parser.add_argument("--max-utterance-s", type=float, default=6.0)
    parser.add_argument("--idle-gap-ms", type=int, default=150)
    parser.add_argument("--worker-count", type=int, default=2)
    parser.add_argument("--threads-per-worker", type=int, default=None)
    parser.add_argument("--cpu-threads", type=int, default=0)
    parser.add_argument("--queue-warning-jobs", type=int, default=8)
    parser.add_argument("--queue-hard-limit-jobs", type=int, default=16)
    parser.add_argument("--report-interval-s", type=float, default=15.0)
    parser.add_argument("--keep-spool", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--fake-delay-ms", type=int, default=120)
    parser.add_argument("--fake-jitter-ms", type=int, default=40)
    parser.add_argument("--fake-fail-every", type=int, default=0)
    parser.add_argument("--debug", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for the standalone soak harness."""
    parser = create_argument_parser()
    args = parser.parse_args(argv)
    configure_logging(
        log_level="DEBUG" if args.debug else "INFO",
        renderer_style="compact",
        log_mode="debug" if args.debug else "normal",
    )
    try:
        summary = asyncio.run(run_soak_harness(args))
    except PermissionError as exc:
        parser.exit(
            2,
            "real soak mode could not start worker processes in this environment "
            f"({exc}). Run outside the restricted sandbox or use --mode fake.\n",
        )
    print(json.dumps(summary.to_jsonable(), ensure_ascii=False, indent=2))
    return 0


def _normalize_audio(samples: NDArray[np.float32], sample_rate: int) -> NDArray[np.float32]:
    """Normalize arbitrary waveform input into mono float32 at 16 kHz."""
    audio = np.asarray(samples, dtype=np.float32)
    if audio.ndim == 0:
        audio = audio.reshape(1)
    elif audio.ndim == 2:
        audio = audio.mean(axis=1, dtype=np.float32)
    elif audio.ndim > 2:
        audio = audio.reshape(audio.shape[0], -1).mean(axis=1, dtype=np.float32)
    audio = np.ascontiguousarray(audio.reshape(-1), dtype=np.float32)
    if sample_rate == _TARGET_SAMPLE_RATE:
        return audio
    duration_s = audio.size / float(sample_rate)
    target_samples = max(1, int(round(duration_s * _TARGET_SAMPLE_RATE)))
    xs_old = np.linspace(0.0, 1.0, num=audio.size, endpoint=False)
    xs_new = np.linspace(0.0, 1.0, num=target_samples, endpoint=False)
    return np.interp(xs_new, xs_old, audio).astype(np.float32)


def _percentile(sorted_values: Sequence[float], fraction: float) -> float:
    """Return a nearest-rank percentile from pre-sorted values."""
    if not sorted_values:
        raise ValueError("sorted_values must not be empty")
    if fraction <= 0:
        return float(sorted_values[0])
    if fraction >= 1:
        return float(sorted_values[-1])
    index = max(0, int(np.ceil(fraction * len(sorted_values))) - 1)
    return float(sorted_values[index])


if __name__ == "__main__":
    raise SystemExit(main())
