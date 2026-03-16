"""File-based AudioCaptureSource for batch processing.

Reads audio from a file on disk and yields AudioChunk objects
in the same interface as a live capture source. Supports any
format that ``soundfile`` can read (WAV, FLAC, OGG, etc.).
"""

from collections.abc import AsyncIterator
from pathlib import Path

import numpy as np
import soundfile as sf

from voxfusion.exceptions import AudioCaptureError
from voxfusion.logging import get_logger
from voxfusion.models.audio import AudioChunk

log = get_logger(__name__)


class FileAudioSource:
    """Reads audio from a file and exposes it as an AudioCaptureSource.

    The file is opened on ``start()`` and closed on ``stop()``.
    Chunks are returned in order with correct timestamps.
    """

    def __init__(self, path: Path, *, chunk_duration_ms: int = 500) -> None:
        self._path = path
        self._chunk_duration_ms = chunk_duration_ms
        self._sf: sf.SoundFile | None = None
        self._position: int = 0
        self._active = False

    @property
    def device_name(self) -> str:
        return f"file:{self._path.name}"

    @property
    def sample_rate(self) -> int:
        if self._sf is not None:
            return self._sf.samplerate
        with sf.SoundFile(str(self._path)) as f:
            return f.samplerate

    @property
    def channels(self) -> int:
        if self._sf is not None:
            return self._sf.channels
        with sf.SoundFile(str(self._path)) as f:
            return f.channels

    @property
    def is_active(self) -> bool:
        return self._active

    async def start(self) -> None:
        """Open the audio file for reading."""
        if not self._path.exists():
            raise AudioCaptureError(f"Audio file not found: {self._path}")
        try:
            self._sf = sf.SoundFile(str(self._path), mode="r")
        except sf.LibsndfileError as exc:
            raise AudioCaptureError(f"Cannot open audio file: {exc}") from exc
        self._position = 0
        self._active = True
        log.info(
            "file_source.started",
            path=str(self._path),
            sample_rate=self._sf.samplerate,
            channels=self._sf.channels,
            duration_s=round(len(self._sf) / self._sf.samplerate, 2),
        )

    async def stop(self) -> None:
        """Close the audio file."""
        self._active = False
        if self._sf is not None:
            self._sf.close()
            self._sf = None
        log.info("file_source.stopped", path=str(self._path))

    async def read_chunk(self, duration_ms: int = 500) -> AudioChunk:
        """Read the next chunk of audio data from the file.

        Returns an AudioChunk with ``num_samples == 0`` when the file
        is exhausted.
        """
        if self._sf is None:
            raise AudioCaptureError("FileAudioSource has not been started")

        frames_to_read = int(self._sf.samplerate * duration_ms / 1000)
        data = self._sf.read(frames_to_read, dtype="float32", always_2d=False)

        if isinstance(data, np.ndarray) and data.size == 0:
            data = np.array([], dtype=np.float32)

        ts_start = self._position / self._sf.samplerate
        self._position += len(data) if data.ndim == 1 else data.shape[0]
        ts_end = self._position / self._sf.samplerate

        return AudioChunk(
            samples=data.astype(np.float32) if data.dtype != np.float32 else data,
            sample_rate=self._sf.samplerate,
            channels=self._sf.channels,
            timestamp_start=ts_start,
            timestamp_end=ts_end,
            source="file",
            dtype="float32",
        )

    async def stream(self, chunk_duration_ms: int = 500) -> AsyncIterator[AudioChunk]:
        """Yield audio chunks until the file is exhausted."""
        while self._active:
            chunk = await self.read_chunk(chunk_duration_ms)
            if chunk.num_samples == 0:
                break
            yield chunk
