"""Opt-in live GigaAM hardware-in-the-loop test."""

from __future__ import annotations

import asyncio
import os
import threading
import time
from pathlib import Path

import pytest

from voxfusion.config.loader import load_config
from voxfusion.live_gigaam.session import LiveGigaAMSessionController
from voxfusion.models.translation import TranslatedSegment

pytestmark = [
    pytest.mark.hardware,
    pytest.mark.platform,
    pytest.mark.slow,
]


async def _run_live_gigaam_hardware(
    tmp_path: Path,
    hardware_audio_config,
    speech_wav_path: Path,
    gigaam_model_path: Path,
    play_wav_blocking_fn,
) -> tuple[list[str], list[TranslatedSegment], list[TranslatedSegment], list[TranslatedSegment]]:
    statuses: list[str] = []
    draft_segments: list[TranslatedSegment] = []
    final_segments: list[TranslatedSegment] = []
    capture_started = threading.Event()
    stop_event = threading.Event()

    config = load_config(
        {
            "data_dir": str(tmp_path),
            "capture": {
                "chunk_duration_ms": 250,
                "buffer_size": 24,
                "lossy_mode": True,
            },
            "asr": {
                "model_size": "gigaam-v3-e2e-ctc",
                "model_path": str(gigaam_model_path),
                "cpu_threads": max(1, (os.cpu_count() or 4)),
            },
            "live_gigaam": {
                "worker_count": 1,
                "threads_per_worker": 1,
                "utterance_max_duration_ms": 3000,
            },
            "security": {
                "auto_delete_temp_files": True,
            },
        }
    )

    controller = LiveGigaAMSessionController(
        config=config,
        microphone_device_id=None,
        system_device_id=hardware_audio_config.system_device_id,
        on_status=statuses.append,
        on_segments=lambda segments: draft_segments.extend(segments),
        on_finalized_segments=lambda segments: final_segments.extend(segments),
        on_capture_started=lambda _started_at: capture_started.set(),
        requested_source="system",
    )

    def _playback_worker() -> None:
        if not capture_started.wait(timeout=120):
            raise RuntimeError("Live GigaAM hardware test timed out waiting for capture start.")
        time.sleep(0.35)
        play_wav_blocking_fn(
            speech_wav_path,
            device_id=hardware_audio_config.playback_device_id,
            start_delay_s=0.0,
        )
        time.sleep(0.8)
        stop_event.set()

    playback_task = asyncio.create_task(asyncio.to_thread(_playback_worker))
    controller_error: Exception | None = None
    try:
        result = await controller.run(stop_event)
    except Exception as exc:
        controller_error = exc
        result = []
    finally:
        stop_event.set()
    await playback_task
    if controller_error is not None:
        raise controller_error
    return statuses, draft_segments, final_segments, result


def test_live_gigaam_system_loopback_finalizes_non_empty_transcript(
    tmp_path: Path,
    windows_hardware_only: None,
    hardware_audio_config,
    hardware_speech_wav: Path,
    hardware_gigaam_model_path: Path,
    play_wav_blocking_fn,
) -> None:
    del windows_hardware_only
    statuses, draft_segments, final_segments, result = asyncio.run(
        _run_live_gigaam_hardware(
            tmp_path,
            hardware_audio_config,
            hardware_speech_wav,
            hardware_gigaam_model_path,
            play_wav_blocking_fn,
        )
    )

    capture_started = any(
        "GigaAM workers ready" in status or "Live GigaAM started" in status
        for status in statuses
    )
    assert capture_started is True
    assert result
    final_text = " ".join(segment.diarized.segment.text for segment in result).strip()
    assert final_text
    assert final_segments or draft_segments
