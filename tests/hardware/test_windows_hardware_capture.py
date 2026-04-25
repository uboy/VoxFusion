"""Opt-in Windows hardware capture tests."""

from __future__ import annotations

import asyncio
from contextlib import suppress

import numpy as np
import pytest

from voxfusion.capture.windows_factory import create_windows_capture_source
from voxfusion.config.models import CaptureConfig

pytestmark = [
    pytest.mark.hardware,
    pytest.mark.platform,
    pytest.mark.slow,
]


async def _exercise_microphone_start_stop(device_id: str | int | None) -> None:
    source = create_windows_capture_source(
        "microphone",
        CaptureConfig(chunk_duration_ms=250, buffer_size=8, lossy_mode=True),
        microphone_device_id=device_id,
    )
    await source.start()
    try:
        assert source.is_active is True
        await asyncio.sleep(0.35)
    finally:
        with suppress(Exception):
            await source.stop()
    assert source.is_active is False


def test_microphone_hardware_start_stop_smoke(
    windows_hardware_only: None,
    hardware_audio_config,
) -> None:
    del windows_hardware_only
    asyncio.run(_exercise_microphone_start_stop(hardware_audio_config.microphone_device_id))


async def _exercise_system_loopback_capture(
    hardware_audio_config,
    speech_wav_path,
    play_wav_blocking_fn,
    wait_for_non_silent_chunk_fn,
) -> float:
    source = create_windows_capture_source(
        "system",
        CaptureConfig(chunk_duration_ms=250, buffer_size=12, lossy_mode=True),
        system_device_id=hardware_audio_config.system_device_id,
    )
    await source.start()
    try:
        playback_task = asyncio.create_task(
            asyncio.to_thread(
                play_wav_blocking_fn,
                speech_wav_path,
                device_id=hardware_audio_config.playback_device_id,
                start_delay_s=0.4,
            )
        )
        chunk = await wait_for_non_silent_chunk_fn(source, duration_ms=250, attempts=20)
        await playback_task
        samples = np.asarray(chunk.samples, dtype=np.float32).reshape(-1)
        return float(np.sqrt(np.mean(samples**2)))
    finally:
        with suppress(Exception):
            await source.stop()


def test_system_loopback_captures_controlled_playback(
    windows_hardware_only: None,
    hardware_audio_config,
    hardware_speech_wav,
    play_wav_blocking_fn,
    wait_for_non_silent_chunk_fn,
) -> None:
    del windows_hardware_only
    rms = asyncio.run(
        _exercise_system_loopback_capture(
            hardware_audio_config,
            hardware_speech_wav,
            play_wav_blocking_fn,
            wait_for_non_silent_chunk_fn,
        )
    )
    assert rms >= 0.003
