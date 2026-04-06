"""Unit tests for live GigaAM session orchestration."""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path

import numpy as np

from voxfusion.config.models import PipelineConfig
from voxfusion.live_gigaam.session import LiveGigaAMSessionController
from voxfusion.live_gigaam.types import LiveGigaAMResult, LiveUtterance
from voxfusion.models.audio import AudioChunk
from voxfusion.models.diarization import DiarizedSegment
from voxfusion.models.transcription import TranscriptionSegment
from voxfusion.models.translation import TranslatedSegment


def _segment(text: str, *, start_s: float, end_s: float, speaker: str = "SPEAKER_LOCAL") -> TranslatedSegment:
    return TranslatedSegment(
        diarized=DiarizedSegment(
            segment=TranscriptionSegment(
                text=text,
                language="ru",
                start_time=start_s,
                end_time=end_s,
                confidence=0.0,
                words=None,
                no_speech_prob=0.0,
            ),
            speaker_id=speaker,
            speaker_source="channel",
        ),
        translated_text=None,
        target_language=None,
    )


class _FakeDispatcher:
    def __init__(self) -> None:
        self.jobs: list[tuple[int, bool]] = []
        self.pending_jobs = 0

    def get_stats(self) -> dict[str, int]:
        return {"workers": 2, "pending": 0, "completed": 0, "failed": 0}

    async def start(self) -> None:
        return None

    async def shutdown(self) -> None:
        return None

    async def transcribe(self, job) -> LiveGigaAMResult:
        self.jobs.append((job.seq_id, job.finalize))
        text = f"final {job.seq_id}" if job.finalize else f"draft {job.seq_id}"
        return LiveGigaAMResult(
            seq_id=job.seq_id,
            source=job.source,
            start_s=job.start_s,
            end_s=job.end_s,
            text=text,
            worker_id=0,
            finalize=job.finalize,
        )


class _FinalizeErrorDispatcher(_FakeDispatcher):
    async def transcribe(self, job) -> LiveGigaAMResult:
        self.jobs.append((job.seq_id, job.finalize))
        if job.finalize:
            return LiveGigaAMResult(
                seq_id=job.seq_id,
                source=job.source,
                start_s=job.start_s,
                end_s=job.end_s,
                text="",
                worker_id=0,
                finalize=True,
                error="finalize failed",
            )
        return await super().transcribe(job)


class _FakeAudioSource:
    def __init__(self, controller: LiveGigaAMSessionController) -> None:
        self._controller = controller
        self.device_name = "fake:mic"
        self.sample_rate = 16000
        self.channels = 1
        self.is_active = True
        self._chunks = [
            AudioChunk(
                samples=np.ones(1600, dtype=np.float32),
                sample_rate=16000,
                channels=1,
                timestamp_start=0.0,
                timestamp_end=0.1,
                source="microphone",
                dtype="float32",
            ),
            AudioChunk(
                samples=np.ones(1600, dtype=np.float32),
                sample_rate=16000,
                channels=1,
                timestamp_start=0.2,
                timestamp_end=0.3,
                source="microphone",
                dtype="float32",
            ),
        ]

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    async def stream(self, chunk_duration_ms: int = 5000):
        del chunk_duration_ms
        assert self._controller._spool is not None
        for chunk in self._chunks:
            yield self._controller._spool.append(chunk)


class _SlowDraftDispatcher(_FakeDispatcher):
    def __init__(self) -> None:
        super().__init__()
        self.pending_jobs = 0

    def get_stats(self) -> dict[str, int]:
        return {"workers": 1, "pending": self.pending_jobs, "completed": 0, "failed": 0}

    async def transcribe(self, job) -> LiveGigaAMResult:
        self.jobs.append((job.seq_id, job.finalize))
        if job.finalize:
            return LiveGigaAMResult(
                seq_id=job.seq_id,
                source=job.source,
                start_s=job.start_s,
                end_s=job.end_s,
                text=f"final {job.seq_id}",
                worker_id=0,
                finalize=True,
            )
        self.pending_jobs += 1
        try:
            await asyncio.sleep(0.05)
            return LiveGigaAMResult(
                seq_id=job.seq_id,
                source=job.source,
                start_s=job.start_s,
                end_s=job.end_s,
                text=f"draft {job.seq_id}",
                worker_id=0,
            )
        finally:
            self.pending_jobs -= 1


class _BurstAudioSource:
    def __init__(self, controller: LiveGigaAMSessionController) -> None:
        self._controller = controller
        self.device_name = "fake:burst"
        self.sample_rate = 16000
        self.channels = 1
        self.is_active = True
        self._chunks = [
            AudioChunk(
                samples=np.ones(1600, dtype=np.float32),
                sample_rate=16000,
                channels=1,
                timestamp_start=index * 0.2,
                timestamp_end=index * 0.2 + 0.1,
                source="microphone",
                dtype="float32",
            )
            for index in range(3)
        ]

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    async def stream(self, chunk_duration_ms: int = 5000):
        del chunk_duration_ms
        assert self._controller._spool is not None
        for chunk in self._chunks:
            yield self._controller._spool.append(chunk)


def test_live_gigaam_session_reuses_successful_drafts_by_default(tmp_path: Path) -> None:
    statuses: list[str] = []
    drafts: list[list[TranslatedSegment]] = []
    finalized: list[list[TranslatedSegment]] = []

    controller = LiveGigaAMSessionController(
        config=PipelineConfig(asr={"model_size": "gigaam-v3-e2e-ctc"}, data_dir=str(tmp_path)),
        microphone_device_id="sd:17",
        system_device_id=None,
        on_status=statuses.append,
        on_segments=lambda segments: drafts.append(list(segments)),
        on_finalized_segments=lambda segments: finalized.append(list(segments)),
    )
    controller._dispatcher = _FakeDispatcher()
    controller._session_dir = lambda: tmp_path / "session"  # type: ignore[method-assign]
    controller._build_audio_source = lambda: _FakeAudioSource(controller)  # type: ignore[method-assign]

    result = asyncio.run(controller.run(threading.Event()))

    assert any("Live GigaAM started" in status for status in statuses)
    assert drafts and [seg.diarized.segment.text for seg in drafts[0]] == ["draft 0"]
    assert drafts[-1][0].diarized.segment.text == "draft 1"
    assert finalized and [seg.diarized.segment.text for seg in finalized[0]] == ["draft 0", "draft 1"]
    assert [seg.diarized.segment.text for seg in result] == ["draft 0", "draft 1"]
    assert controller._dispatcher.jobs == [(0, False), (1, False)]
    assert controller._stop_reprocessed_utterances == 0
    assert not (tmp_path / "session").exists()


def test_live_gigaam_session_can_force_full_stop_finalize(tmp_path: Path) -> None:
    finalized: list[list[TranslatedSegment]] = []

    controller = LiveGigaAMSessionController(
        config=PipelineConfig(
            asr={"model_size": "gigaam-v3-e2e-ctc"},
            live_gigaam={"stop_finalize_mode": "always"},
            data_dir=str(tmp_path),
        ),
        microphone_device_id="sd:17",
        system_device_id=None,
        on_status=lambda _status: None,
        on_segments=lambda _segments: None,
        on_finalized_segments=lambda segments: finalized.append(list(segments)),
    )
    controller._dispatcher = _FakeDispatcher()
    controller._session_dir = lambda: tmp_path / "session"  # type: ignore[method-assign]
    controller._build_audio_source = lambda: _FakeAudioSource(controller)  # type: ignore[method-assign]

    result = asyncio.run(controller.run(threading.Event()))

    assert [seg.diarized.segment.text for seg in result] == ["final 0", "final 1"]
    assert finalized and [seg.diarized.segment.text for seg in finalized[0]] == ["final 0", "final 1"]
    assert controller._dispatcher.jobs == [(0, False), (1, False), (0, True), (1, True)]
    assert controller._stop_reprocessed_utterances == 2


def test_live_gigaam_session_uses_draft_text_when_finalize_fails(tmp_path: Path) -> None:
    finalized: list[list[TranslatedSegment]] = []

    controller = LiveGigaAMSessionController(
        config=PipelineConfig(
            asr={"model_size": "gigaam-v3-e2e-ctc"},
            live_gigaam={"stop_finalize_mode": "always"},
            data_dir=str(tmp_path),
        ),
        microphone_device_id="sd:17",
        system_device_id=None,
        on_status=lambda _status: None,
        on_segments=lambda _segments: None,
        on_finalized_segments=lambda segments: finalized.append(list(segments)),
    )
    controller._dispatcher = _FinalizeErrorDispatcher()
    controller._session_dir = lambda: tmp_path / "session"  # type: ignore[method-assign]
    controller._build_audio_source = lambda: _FakeAudioSource(controller)  # type: ignore[method-assign]

    result = asyncio.run(controller.run(threading.Event()))

    assert [seg.diarized.segment.text for seg in result] == ["draft 0", "draft 1"]
    assert finalized and [seg.diarized.segment.text for seg in finalized[0]] == ["draft 0", "draft 1"]
    assert not (tmp_path / "session").exists()


def test_live_gigaam_session_cleans_up_after_dispatcher_start_failure(tmp_path: Path) -> None:
    class _StartFailDispatcher(_FakeDispatcher):
        async def start(self) -> None:
            raise RuntimeError("warmup failed")

    controller = LiveGigaAMSessionController(
        config=PipelineConfig(asr={"model_size": "gigaam-v3-e2e-ctc"}, data_dir=str(tmp_path)),
        microphone_device_id="sd:17",
        system_device_id=None,
        on_status=lambda _status: None,
        on_segments=lambda _segments: None,
        requested_source="microphone",
    )
    controller._dispatcher = _StartFailDispatcher()
    controller._session_dir = lambda: tmp_path / "session"  # type: ignore[method-assign]

    try:
        asyncio.run(controller.run(threading.Event()))
    except RuntimeError as exc:
        assert str(exc) == "warmup failed"
    else:
        raise AssertionError("Expected warmup failure to propagate.")

    assert not (tmp_path / "session").exists()


def test_finalize_utterance_bounds_context_inside_same_source_neighbors(tmp_path: Path) -> None:
    controller = LiveGigaAMSessionController(
        config=PipelineConfig(
            asr={"model_size": "gigaam-v3-e2e-ctc"},
            live_gigaam={"finalize_left_context_ms": 700, "finalize_right_context_ms": 300},
            data_dir=str(tmp_path),
        ),
        microphone_device_id="sd:17",
        system_device_id=None,
        on_status=lambda _status: None,
        on_segments=lambda _segments: None,
        requested_source="microphone",
    )
    controller._utterances = [
        LiveUtterance(seq_id=0, source="microphone", start_s=0.0, end_s=1.0, sample_rate=16000, samples=np.ones(1600, dtype=np.float32)),
        LiveUtterance(seq_id=1, source="system", start_s=1.1, end_s=1.4, sample_rate=16000, samples=np.ones(1600, dtype=np.float32)),
        LiveUtterance(seq_id=2, source="microphone", start_s=1.5, end_s=2.0, sample_rate=16000, samples=np.ones(1600, dtype=np.float32)),
        LiveUtterance(seq_id=3, source="microphone", start_s=2.2, end_s=3.0, sample_rate=16000, samples=np.ones(1600, dtype=np.float32)),
    ]
    controller._draft_results = {
        2: LiveGigaAMResult(seq_id=2, source="microphone", start_s=1.5, end_s=2.0, text="draft 2", worker_id=0),
    }

    class _ReadWindowRecorder:
        sample_rate = 16000

        def __init__(self) -> None:
            self.calls: list[tuple[str, float, float]] = []

        def read_window(self, source: str, start_s: float, end_s: float):
            self.calls.append((source, start_s, end_s))
            return np.ones(3200, dtype=np.float32)

        def close(self) -> None:
            return None

    class _FinalizeDispatcher(_FakeDispatcher):
        async def transcribe(self, job) -> LiveGigaAMResult:
            return LiveGigaAMResult(
                seq_id=job.seq_id,
                source=job.source,
                start_s=job.start_s,
                end_s=job.end_s,
                text="final 2",
                worker_id=0,
                finalize=True,
            )

    spool = _ReadWindowRecorder()
    controller._spool = spool  # type: ignore[assignment]
    controller._dispatcher = _FinalizeDispatcher()

    result = asyncio.run(controller._finalize_utterance(controller._utterances[2]))

    assert spool.calls == [("microphone", 1.0, 2.2)]
    assert result.text == "final 2"


def test_live_gigaam_session_defers_drafts_under_backlog_and_finalizes_all(tmp_path: Path) -> None:
    statuses: list[str] = []
    drafts: list[list[TranslatedSegment]] = []
    finalized: list[list[TranslatedSegment]] = []

    controller = LiveGigaAMSessionController(
        config=PipelineConfig(
            asr={"model_size": "gigaam-v3-e2e-ctc"},
            live_gigaam={"queue_warning_jobs": 1, "queue_hard_limit_jobs": 1},
            data_dir=str(tmp_path),
        ),
        microphone_device_id="sd:17",
        system_device_id=None,
        on_status=statuses.append,
        on_segments=lambda segments: drafts.append(list(segments)),
        on_finalized_segments=lambda segments: finalized.append(list(segments)),
    )
    controller._dispatcher = _SlowDraftDispatcher()
    controller._session_dir = lambda: tmp_path / "session"  # type: ignore[method-assign]
    controller._build_audio_source = lambda: _BurstAudioSource(controller)  # type: ignore[method-assign]

    result = asyncio.run(controller.run(threading.Event()))

    draft_texts = [segment.diarized.segment.text for batch in drafts for segment in batch]
    final_texts = [segment.diarized.segment.text for segment in result]
    stats = controller.get_stats()

    assert draft_texts == ["draft 0"]
    assert final_texts == ["draft 0", "final 1", "final 2"]
    assert finalized and [seg.diarized.segment.text for seg in finalized[0]] == final_texts
    assert controller._dispatcher.jobs == [
        (0, False),
        (1, True),
        (2, True),
    ]
    assert controller._deferred_draft_jobs == 2
    assert controller._stop_reprocessed_utterances == 2
    assert stats["deferred_drafts"] == 2
    assert stats["backlog_peak"] >= 2
    assert any("Deferring new draft utterances" in status for status in statuses)
    assert any("Deferred 2 draft utterances during overload" in status for status in statuses)
