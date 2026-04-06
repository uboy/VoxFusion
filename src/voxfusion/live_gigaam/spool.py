"""Durable audio spooling for live GigaAM sessions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from voxfusion.capture.base import AudioCaptureSource
from voxfusion.logging import get_logger
from voxfusion.models.audio import AudioChunk

log = get_logger(__name__)

_SPOOL_SAMPLE_RATE = 16000


def normalize_audio_samples(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    """Convert audio to contiguous mono float32 at 16 kHz."""
    audio = np.asarray(samples, dtype=np.float32)
    if audio.ndim == 0:
        audio = audio.reshape(1)
    elif audio.ndim == 2:
        audio = audio.mean(axis=1, dtype=np.float32)
    elif audio.ndim > 2:
        audio = audio.reshape(audio.shape[0], -1).mean(axis=1, dtype=np.float32)
    audio = np.ascontiguousarray(audio.reshape(-1), dtype=np.float32)

    if sample_rate == _SPOOL_SAMPLE_RATE or audio.size <= 1:
        return audio

    duration = len(audio) / sample_rate
    target_samples = max(1, int(round(duration * _SPOOL_SAMPLE_RATE)))
    xs_old = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
    xs_new = np.linspace(0.0, 1.0, num=target_samples, endpoint=False)
    return np.interp(xs_new, xs_old, audio).astype(np.float32)


@dataclass
class _SourceSpoolState:
    path: Path
    writer: sf.SoundFile
    frames_written: int = 0


class SessionAudioSpool:
    """Persist normalized live audio per source while preserving the timeline."""

    def __init__(
        self,
        session_dir: Path,
        *,
        sample_rate: int = _SPOOL_SAMPLE_RATE,
    ) -> None:
        self._session_dir = session_dir
        self._sample_rate = sample_rate
        self._states: dict[str, _SourceSpoolState] = {}
        self._session_dir.mkdir(parents=True, exist_ok=True)

    @property
    def session_dir(self) -> Path:
        return self._session_dir

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def append(self, chunk: AudioChunk) -> AudioChunk:
        """Normalize *chunk*, append it to the source spool, and return it."""
        normalized = normalize_audio_samples(chunk.samples, chunk.sample_rate)
        normalized_chunk = AudioChunk(
            samples=normalized,
            sample_rate=self._sample_rate,
            channels=1,
            timestamp_start=chunk.timestamp_start,
            timestamp_end=chunk.timestamp_start + (len(normalized) / self._sample_rate),
            source=chunk.source,
            dtype="float32",
        )

        state = self._ensure_state(normalized_chunk.source)
        target_start = max(0, int(round(normalized_chunk.timestamp_start * self._sample_rate)))
        samples = normalized_chunk.samples

        if target_start > state.frames_written:
            gap = target_start - state.frames_written
            state.writer.write(np.zeros(gap, dtype=np.float32))
            state.frames_written += gap
        elif target_start < state.frames_written:
            trim = state.frames_written - target_start
            if trim >= len(samples):
                return normalized_chunk
            samples = samples[trim:]
            target_start = state.frames_written

        if len(samples) > 0:
            state.writer.write(samples)
            state.frames_written += len(samples)
            state.writer.flush()

        return normalized_chunk

    def read_window(self, source: str, start_s: float, end_s: float) -> np.ndarray:
        """Read a normalized time window for *source* and zero-pad missing ranges."""
        if end_s <= start_s:
            return np.zeros(1, dtype=np.float32)
        state = self._states.get(source)
        frame_start = max(0, int(round(start_s * self._sample_rate)))
        frame_end = max(frame_start + 1, int(round(end_s * self._sample_rate)))
        frame_count = frame_end - frame_start
        if state is None:
            return np.zeros(frame_count, dtype=np.float32)

        state.writer.flush()
        with sf.SoundFile(state.path, mode="r") as reader:
            if frame_start >= len(reader):
                return np.zeros(frame_count, dtype=np.float32)
            reader.seek(frame_start)
            available = max(0, min(frame_count, len(reader) - frame_start))
            data = reader.read(available, dtype="float32", always_2d=False)

        if not isinstance(data, np.ndarray):
            data = np.asarray(data, dtype=np.float32)
        data = np.ascontiguousarray(data.reshape(-1), dtype=np.float32)
        if len(data) < frame_count:
            data = np.pad(data, (0, frame_count - len(data)))
        return data

    def source_path(self, source: str) -> Path | None:
        state = self._states.get(source)
        return None if state is None else state.path

    def close(self) -> None:
        for state in self._states.values():
            state.writer.close()
        self._states.clear()

    def _ensure_state(self, source: str) -> _SourceSpoolState:
        if source in self._states:
            return self._states[source]
        safe_source = source.replace("/", "_").replace("\\", "_")
        path = self._session_dir / f"{safe_source}.wav"
        writer = sf.SoundFile(
            str(path),
            mode="w",
            samplerate=self._sample_rate,
            channels=1,
            subtype="FLOAT",
            format="WAV",
        )
        log.info("live_gigaam.spool_created", source=source, path=str(path))
        state = _SourceSpoolState(path=path, writer=writer)
        self._states[source] = state
        return state


class SpoolingCaptureSource:
    """Wrap an audio source, normalize its chunks, and spool them durably."""

    def __init__(
        self,
        source: AudioCaptureSource,
        spool: SessionAudioSpool,
    ) -> None:
        self._source = source
        self._spool = spool

    @property
    def device_name(self) -> str:
        return self._source.device_name

    @property
    def sample_rate(self) -> int:
        return self._spool.sample_rate

    @property
    def channels(self) -> int:
        return 1

    @property
    def is_active(self) -> bool:
        return self._source.is_active

    async def start(self) -> None:
        await self._source.start()

    async def stop(self) -> None:
        await self._source.stop()

    async def read_chunk(self, duration_ms: int = 500) -> AudioChunk:
        chunk = await self._source.read_chunk(duration_ms)
        return self._spool.append(chunk)

    async def stream(self, chunk_duration_ms: int = 500):
        async for chunk in self._source.stream(chunk_duration_ms=chunk_duration_ms):
            yield self._spool.append(chunk)
