"""Hardware-in-the-loop test helpers for Windows audio paths."""

from __future__ import annotations

import asyncio
import importlib.util
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from voxfusion.capture.windows_audio import parse_windows_device_id


@dataclass(frozen=True)
class HardwareAudioConfig:
    microphone_device_id: str | int | None
    system_device_id: str | int | None
    playback_device_id: str | int | None
    speech_wav: Path | None
    gigaam_model_path: Path | None


def _env_text(name: str) -> str | None:
    value = os.environ.get(name, "").strip()
    return value or None


def _env_path(name: str) -> Path | None:
    value = _env_text(name)
    return Path(value).expanduser() if value else None


def _pkg(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


@pytest.fixture(scope="session")
def hardware_audio_config() -> HardwareAudioConfig:
    return HardwareAudioConfig(
        microphone_device_id=_env_text("VOXFUSION_HW_MIC_DEVICE"),
        system_device_id=_env_text("VOXFUSION_HW_SYSTEM_DEVICE"),
        playback_device_id=_env_text("VOXFUSION_HW_PLAYBACK_DEVICE"),
        speech_wav=_env_path("VOXFUSION_HW_SPEECH_WAV"),
        gigaam_model_path=_env_path("VOXFUSION_HW_GIGAAM_MODEL_PATH"),
    )


@pytest.fixture(scope="session")
def windows_hardware_only() -> None:
    if os.name != "nt":
        pytest.skip("hardware audio tests currently support Windows only")


@pytest.fixture(scope="session")
def hardware_speech_wav(
    windows_hardware_only: None,
    hardware_audio_config: HardwareAudioConfig,
) -> Path:
    del windows_hardware_only
    path = hardware_audio_config.speech_wav
    if path is None:
        pytest.skip("Set VOXFUSION_HW_SPEECH_WAV to a local speech WAV for hardware playback tests.")
    if not path.exists():
        pytest.skip(f"Configured VOXFUSION_HW_SPEECH_WAV does not exist: {path}")
    return path


@pytest.fixture(scope="session")
def hardware_gigaam_model_path(
    windows_hardware_only: None,
    hardware_audio_config: HardwareAudioConfig,
) -> Path:
    del windows_hardware_only
    if not (_pkg("transformers") and _pkg("torch")):
        pytest.skip("Live GigaAM hardware test requires installed transformers and torch.")
    path = hardware_audio_config.gigaam_model_path
    if path is None:
        pytest.skip("Set VOXFUSION_HW_GIGAAM_MODEL_PATH to a local GigaAM model directory.")
    if not path.exists():
        pytest.skip(f"Configured VOXFUSION_HW_GIGAAM_MODEL_PATH does not exist: {path}")
    return path


def _playback_output_device_index(device_id: str | int | None) -> int | None:
    backend, native_index = parse_windows_device_id(device_id, default_backend="sd")
    if backend is None:
        return None
    if backend != "sd":
        pytest.skip(
            "Hardware playback helper supports only sounddevice output ids (sd:<index>) or the default output."
        )
    return native_index


def play_wav_blocking(
    wav_path: Path,
    *,
    device_id: str | int | None = None,
    start_delay_s: float = 0.35,
) -> None:
    import sounddevice as sd
    import soundfile as sf

    samples, sample_rate = sf.read(str(wav_path), dtype="float32", always_2d=False)
    if start_delay_s > 0:
        time.sleep(start_delay_s)
    sd.play(samples, samplerate=sample_rate, device=_playback_output_device_index(device_id), blocking=True)
    sd.stop()


async def wait_for_non_silent_chunk(
    source: object,
    *,
    duration_ms: int = 250,
    attempts: int = 16,
    rms_threshold: float = 0.003,
) -> object:
    last_rms = 0.0
    last_error: Exception | None = None
    for _ in range(attempts):
        try:
            if hasattr(source, "read_chunk"):
                chunk = await source.read_chunk(duration_ms)  # type: ignore[attr-defined]
            elif hasattr(source, "stream"):
                chunk = None
                async for streamed_chunk in source.stream(chunk_duration_ms=duration_ms):  # type: ignore[attr-defined]
                    chunk = streamed_chunk
                    break
                if chunk is None:
                    await asyncio.sleep(0.1)
                    continue
            else:  # pragma: no cover - defensive harness guard
                raise TypeError(f"Unsupported capture source {type(source)!r}: missing read_chunk/stream")
        except Exception as exc:  # pragma: no cover - hardware-specific timing
            last_error = exc
            await asyncio.sleep(0.1)
            continue
        samples = np.asarray(chunk.samples, dtype=np.float32).reshape(-1)
        if samples.size == 0:
            await asyncio.sleep(0.1)
            continue
        last_rms = float(np.sqrt(np.mean(samples ** 2)))
        if last_rms >= rms_threshold:
            return chunk
    if last_error is not None:
        raise AssertionError(f"Did not capture a non-silent audio chunk: {last_error}") from last_error
    raise AssertionError(
        f"Did not capture a non-silent audio chunk after {attempts} attempts; last rms={last_rms:.6f}"
    )


@pytest.fixture(scope="session")
def play_wav_blocking_fn():
    return play_wav_blocking


@pytest.fixture(scope="session")
def wait_for_non_silent_chunk_fn():
    return wait_for_non_silent_chunk
