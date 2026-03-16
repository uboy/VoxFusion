"""Linux PulseAudio/PipeWire capture implementation.

Uses ``sounddevice`` with the PulseAudio or PipeWire backend.
System audio loopback is available via PulseAudio monitor sources.
"""

import asyncio
import sys
from collections.abc import AsyncIterator

import numpy as np

from voxfusion.config.models import CaptureConfig
from voxfusion.exceptions import AudioCaptureError, UnsupportedPlatformError
from voxfusion.logging import get_logger
from voxfusion.models.audio import AudioChunk

log = get_logger(__name__)


class PulseAudioCapture:
    """PulseAudio/PipeWire-based audio capture for Linux.

    Loopback capture uses PulseAudio monitor sources, which mirror
    the audio output of any sink.
    """

    def __init__(
        self,
        device_index: int | None = None,
        *,
        loopback: bool = False,
        config: CaptureConfig | None = None,
    ) -> None:
        if not sys.platform.startswith("linux"):
            raise UnsupportedPlatformError("PulseAudioCapture requires Linux")

        self._device_index = device_index
        self._loopback = loopback
        self._config = config or CaptureConfig()
        self._stream: object | None = None
        self._buffer: asyncio.Queue[np.ndarray] = asyncio.Queue(
            maxsize=self._config.buffer_size,
        )
        self._active = False
        self._position = 0

    @property
    def device_name(self) -> str:
        mode = "loopback" if self._loopback else "microphone"
        return f"pulseaudio:{self._device_index or 'default'}:{mode}"

    @property
    def sample_rate(self) -> int:
        return self._config.sample_rate

    @property
    def channels(self) -> int:
        return self._config.channels

    @property
    def is_active(self) -> bool:
        return self._active

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: object,
        status: object,
    ) -> None:
        """Called by sounddevice from the audio thread."""
        if status:
            log.warning("pulseaudio.callback_status", status=str(status))
        try:
            self._buffer.put_nowait(indata.copy())
        except asyncio.QueueFull:
            if self._config.lossy_mode:
                log.debug("pulseaudio.buffer_full_dropping")
            else:
                log.warning("pulseaudio.buffer_full")

    async def start(self) -> None:
        """Start the PulseAudio input stream."""
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise AudioCaptureError("sounddevice is not installed") from exc

        kwargs: dict = {
            "samplerate": self._config.sample_rate,
            "channels": self._config.channels,
            "dtype": "float32",
            "callback": self._audio_callback,
            "blocksize": int(self._config.sample_rate * self._config.chunk_duration_ms / 1000),
        }
        if self._device_index is not None:
            kwargs["device"] = self._device_index

        try:
            self._stream = sd.InputStream(**kwargs)
            self._stream.start()  # type: ignore[union-attr]
        except Exception as exc:
            raise AudioCaptureError(f"Failed to start PulseAudio stream: {exc}") from exc

        self._active = True
        self._position = 0
        log.info("pulseaudio.started", device=self.device_name)

    async def stop(self) -> None:
        """Stop and close the PulseAudio stream."""
        self._active = False
        if self._stream is not None:
            self._stream.stop()  # type: ignore[union-attr]
            self._stream.close()  # type: ignore[union-attr]
            self._stream = None
        log.info("pulseaudio.stopped")

    async def read_chunk(self, duration_ms: int = 500) -> AudioChunk:
        """Read the next audio chunk from the buffer."""
        if not self._active:
            raise AudioCaptureError("PulseAudioCapture is not active")

        data = await self._buffer.get()
        ts_start = self._position / self._config.sample_rate
        self._position += data.shape[0]
        ts_end = self._position / self._config.sample_rate

        return AudioChunk(
            samples=data.astype(np.float32).squeeze(),
            sample_rate=self._config.sample_rate,
            channels=self._config.channels,
            timestamp_start=ts_start,
            timestamp_end=ts_end,
            source="system" if self._loopback else "microphone",
            dtype="float32",
        )

    async def stream(self, chunk_duration_ms: int = 500) -> AsyncIterator[AudioChunk]:
        """Yield audio chunks as they arrive."""
        while self._active:
            try:
                chunk = await asyncio.wait_for(
                    self.read_chunk(chunk_duration_ms),
                    timeout=chunk_duration_ms / 1000 * 3,
                )
                yield chunk
            except asyncio.TimeoutError:
                if self._active:
                    continue
                break
