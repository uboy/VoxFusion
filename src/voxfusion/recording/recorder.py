"""Raw audio recording without ASR."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from voxfusion.capture.base import AudioCaptureSource
from voxfusion.capture.mixer import AudioMixer
from voxfusion.logging import get_logger
from voxfusion.models.audio import AudioChunk

log = get_logger(__name__)


@dataclass(frozen=True)
class RecordingStats:
    """Summary of a completed raw recording."""

    output_path: Path
    sample_rate: int
    channels: int
    duration_s: float
    chunks_captured: int


_FORMAT_SUBTYPES: dict[str, str] = {
    "wav": "PCM_16",
    "flac": "PCM_16",
    "ogg": "VORBIS",
}


class AudioRecorder:
    """Record streamed audio chunks to an audio file (WAV, FLAC, or OGG)."""

    def __init__(
        self,
        *,
        chunk_duration_ms: int = 500,
        on_status: Callable[[str], None] | None = None,
    ) -> None:
        self._chunk_duration_ms = chunk_duration_ms
        self._on_status = on_status or (lambda _message: None)
        self._stop_requested = False
        self._pause_requested = False
        self._pause_state_emitted = False

    def request_stop(self) -> None:
        """Stop recording after the current chunk."""
        self._stop_requested = True

    def request_pause(self) -> None:
        """Pause recording and drop captured chunks until resumed."""
        self._pause_requested = True

    def request_resume(self) -> None:
        """Resume recording after a pause."""
        self._pause_requested = False
        self._pause_state_emitted = False

    @property
    def is_paused(self) -> bool:
        """Return whether recording is currently paused."""
        return self._pause_requested

    async def record(
        self,
        source: AudioCaptureSource | AudioMixer,
        output_path: Path,
        *,
        duration_s: float | None = None,
        format: str = "wav",
    ) -> RecordingStats:
        """Capture audio from *source* and write it to *output_path*."""
        chunks: list[AudioChunk] = []
        sample_rate: int | None = None
        channels: int | None = None
        deadline = duration_s if duration_s and duration_s > 0 else None
        skipped_duration = 0.0

        self._stop_requested = False
        self._pause_requested = False
        self._pause_state_emitted = False
        self._on_status("Starting audio capture...")
        await source.start()
        if isinstance(source, AudioMixer) and source.active_source_count < 2:
            self._on_status(
                "Warning: only one audio source started. Recording will continue with the available source."
            )
        try:
            async for chunk in source.stream(chunk_duration_ms=self._chunk_duration_ms):
                if self._stop_requested:
                    break
                if self._pause_requested:
                    skipped_duration += chunk.duration
                    if not self._pause_state_emitted:
                        self._on_status("Recording paused.")
                        self._pause_state_emitted = True
                    continue
                if self._pause_state_emitted:
                    self._on_status("Recording resumed.")
                    self._pause_state_emitted = False
                sample_rate = sample_rate or chunk.sample_rate
                channels = channels or chunk.channels
                adjusted = AudioChunk(
                    samples=chunk.samples,
                    sample_rate=chunk.sample_rate,
                    channels=chunk.channels,
                    timestamp_start=max(0.0, chunk.timestamp_start - skipped_duration),
                    timestamp_end=max(0.0, chunk.timestamp_end - skipped_duration),
                    source=chunk.source,
                    dtype=chunk.dtype,
                )
                chunks.append(adjusted)
                if deadline is not None and adjusted.timestamp_end >= deadline:
                    break
        finally:
            await source.stop()

        if not chunks or sample_rate is None or channels is None:
            raise RuntimeError("No audio was captured.")

        mixed = _mix_chunks(chunks, sample_rate=sample_rate, channels=channels, duration_s=deadline)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        subtype = _FORMAT_SUBTYPES.get(format.lower(), "PCM_16")
        sf.write(str(output_path), mixed, sample_rate, subtype=subtype)

        duration_actual = mixed.shape[0] / sample_rate
        stats = RecordingStats(
            output_path=output_path,
            sample_rate=sample_rate,
            channels=(mixed.shape[1] if mixed.ndim == 2 else 1),
            duration_s=duration_actual,
            chunks_captured=len(chunks),
        )
        log.info(
            "recording.completed",
            output=str(output_path),
            duration_s=duration_actual,
            chunks=len(chunks),
        )
        return stats


def _mix_chunks(
    chunks: list[AudioChunk],
    *,
    sample_rate: int,
    channels: int,
    duration_s: float | None = None,
) -> np.ndarray:
    """Mix chunks into a single float32 waveform aligned by timestamps."""
    max_end_s = max(chunk.timestamp_end for chunk in chunks)
    total_duration = min(max_end_s, duration_s) if duration_s is not None else max_end_s
    total_samples = max(1, int(np.ceil(total_duration * sample_rate)))
    output = np.zeros((total_samples, channels), dtype=np.float32)
    overlap = np.zeros((total_samples, channels), dtype=np.float32)

    for chunk in chunks:
        start_idx = max(0, int(round(chunk.timestamp_start * sample_rate)))
        samples = _normalize_samples(chunk.samples, channels)
        if duration_s is not None:
            end_limit = int(np.ceil(duration_s * sample_rate))
            samples = samples[: max(0, end_limit - start_idx)]
        end_idx = min(total_samples, start_idx + samples.shape[0])
        if end_idx <= start_idx:
            continue
        usable = samples[: end_idx - start_idx]
        output[start_idx:end_idx] += usable
        overlap[start_idx:end_idx] += 1.0

    np.divide(output, np.maximum(overlap, 1.0), out=output)
    clipped = np.clip(output, -1.0, 1.0)
    if channels == 1:
        return clipped[:, 0]
    return clipped


def _normalize_samples(samples: np.ndarray, channels: int) -> np.ndarray:
    arr = np.asarray(samples, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, np.newaxis]
    elif arr.ndim != 2:
        raise ValueError(f"Unsupported audio sample shape: {arr.shape}")

    if arr.shape[1] == channels:
        return arr
    if channels == 1:
        return arr.mean(axis=1, keepdims=True)
    if arr.shape[1] == 1:
        return np.repeat(arr, channels, axis=1)
    return arr[:, :channels]
