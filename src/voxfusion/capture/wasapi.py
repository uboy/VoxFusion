"""Windows WASAPI audio capture implementation.

Uses ``sounddevice`` with the WASAPI host API for both microphone
input and system loopback capture.  Loopback capture requires
Windows 10+ with WASAPI loopback support.
"""

import asyncio
import sys
from collections.abc import AsyncIterator
from contextlib import suppress

import numpy as np

from voxfusion.config.models import CaptureConfig
from voxfusion.exceptions import AudioCaptureError, UnsupportedPlatformError
from voxfusion.logging import get_logger
from voxfusion.models.audio import AudioChunk

log = get_logger(__name__)


class WASAPICapture:
    """WASAPI-based audio capture for Windows.

    Supports both microphone input and system loopback modes.
    """
    _last_working_input_device: int | None = None
    _last_working_loopback_device: int | None = None

    def __init__(
        self,
        device_index: int | None = None,
        *,
        loopback: bool = False,
        source_label: str | None = None,
        config: CaptureConfig | None = None,
    ) -> None:
        if sys.platform != "win32":
            raise UnsupportedPlatformError("WASAPICapture requires Windows")

        self._device_index = device_index
        self._loopback = loopback
        self._source_label = source_label or ("system" if loopback else "microphone")
        self._config = config or CaptureConfig()
        self._stream: object | None = None
        self._buffer: asyncio.Queue[np.ndarray | None] = asyncio.Queue(
            maxsize=self._config.buffer_size,
        )
        self._pending: np.ndarray | None = None
        self._active = False
        self._position = 0
        self._loop: asyncio.AbstractEventLoop | None = None

    @property
    def device_name(self) -> str:
        mode = "loopback" if self._loopback else "microphone"
        return f"wasapi:{self._device_index or 'default'}:{mode}"

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
            log.warning("wasapi.callback_status", status=str(status))
        loop = self._loop
        if loop is None:
            return
        try:
            loop.call_soon_threadsafe(self._enqueue_audio, indata.copy())
        except RuntimeError:
            # Event loop may already be closed during shutdown.
            return

    def _enqueue_audio(self, data: np.ndarray) -> None:
        """Enqueue audio chunk on the event loop thread."""
        if not self._active:
            return
        try:
            self._buffer.put_nowait(data)
        except asyncio.QueueFull:
            if self._config.lossy_mode:
                log.debug("wasapi.buffer_full_dropping")
            else:
                log.warning("wasapi.buffer_full")

    async def start(self) -> None:
        """Start the WASAPI audio stream."""
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise AudioCaptureError("sounddevice is not installed") from exc

        self._loop = asyncio.get_running_loop()
        hostapis = list(sd.query_hostapis())
        wasapi_hostapi = next(
            (idx for idx, api in enumerate(hostapis) if "wasapi" in str(api.get("name", "")).lower()),
            None,
        )
        if wasapi_hostapi is None and self._loopback:
            available = ", ".join(str(api.get("name", "?")) for api in hostapis)
            raise AudioCaptureError(f"WASAPI host API is unavailable. Found: {available}")
        if wasapi_hostapi is not None:
            log.info("wasapi.hostapi_selected", hostapi_index=wasapi_hostapi)

        all_devices = list(sd.query_devices())
        default_input, default_output = sd.default.device
        default_device = default_output if self._loopback else default_input
        wasapi_default_input = -1
        wasapi_default_output = -1
        wasapi_device_ids: list[int] = []
        if wasapi_hostapi is not None:
            wasapi_default_input = int(hostapis[wasapi_hostapi].get("default_input_device", -1))
            wasapi_default_output = int(hostapis[wasapi_hostapi].get("default_output_device", -1))
            wasapi_device_ids = [int(idx) for idx in hostapis[wasapi_hostapi].get("devices", [])]

        log.info(
            "wasapi.defaults",
            default_input=default_input,
            default_output=default_output,
            selected_default=default_device,
            wasapi_default_input=wasapi_default_input,
            wasapi_default_output=wasapi_default_output,
            loopback=self._loopback,
        )

        candidate_indices: list[int] = []
        failure_reasons: list[str] = []
        failure_log_count = 0

        def _record_failure(scope: str, message: str, **details: object) -> None:
            nonlocal failure_log_count
            failure_log_count += 1
            detail_text = ", ".join(f"{key}={value}" for key, value in details.items())
            compact = f"{scope}: {message}" + (f" ({detail_text})" if detail_text else "")
            failure_reasons.append(compact)
            if failure_log_count <= 12:
                log.warning("wasapi.attempt_failed", scope=scope, message=message, **details)
            else:
                log.debug("wasapi.attempt_failed", scope=scope, message=message, **details)

        def _hostapi_name(index: int) -> str:
            try:
                dev = sd.query_devices(index)
                hostapi_idx = int(dev.get("hostapi", -1))
                return str(hostapis[hostapi_idx].get("name", f"id:{hostapi_idx}"))
            except Exception:
                return "unknown"

        def _is_fatal_device_error(message: str) -> bool:
            return (
                "Invalid device [PaErrorCode -9996]" in message
                or "Blocking API not supported yet" in message
            )

        def _hostapi_to_skip(message: str) -> str | None:
            if "MME error 11" in message:
                return "MME"
            if "Windows DirectSound error" in message:
                return "Windows DirectSound"
            return None

        def _add_candidate(index: int | None) -> None:
            if index is None or index < 0 or index in candidate_indices:
                return
            try:
                dev = sd.query_devices(index)
            except Exception:
                return
            hostapi_idx = int(dev.get("hostapi", -1))
            if self._loopback and hostapi_idx != wasapi_hostapi:
                return
            required_channels = (
                int(dev.get("max_output_channels", 0))
                if self._loopback
                else int(dev.get("max_input_channels", 0))
            )
            if required_channels <= 0:
                return
            candidate_indices.append(index)

        if self._device_index is not None:
            _add_candidate(self._device_index)
            if not candidate_indices:
                raise AudioCaptureError(
                    f"Device {self._device_index} is not a valid "
                    f"{'loopback' if self._loopback else 'input'} device."
                )
        else:
            if self._loopback:
                _add_candidate(self.__class__._last_working_loopback_device)
            else:
                _add_candidate(self.__class__._last_working_input_device)
            if self._loopback:
                _add_candidate(default_device)
                _add_candidate(wasapi_default_output)
                for idx in wasapi_device_ids:
                    _add_candidate(idx)
            else:
                _add_candidate(wasapi_default_input)
                _add_candidate(default_device)

                ranked_candidates: list[tuple[int, int]] = []
                for idx, dev in enumerate(all_devices):
                    if int(dev.get("max_input_channels", 0)) <= 0:
                        continue
                    hostapi_name = _hostapi_name(idx).lower()
                    # Prioritize likely-working APIs for modern Windows drivers.
                    # WASAPI > MME/DS > others > WDM-KS (WDM-KS often returns
                    # silence with modern Realtek/Intel drivers even though the
                    # device reports valid input channels).
                    if "wasapi" in hostapi_name:
                        priority = 0
                    elif "mme" in hostapi_name:
                        priority = 1
                    elif "directsound" in hostapi_name:
                        priority = 2
                    elif "wdm-ks" in hostapi_name:
                        priority = 4
                    else:
                        priority = 3
                    ranked_candidates.append((priority, idx))

                ranked_candidates.sort(key=lambda item: (item[0], item[1]))
                for _priority, idx in ranked_candidates:
                    _add_candidate(idx)

        if not candidate_indices:
            mode = "loopback output" if self._loopback else "input"
            raise AudioCaptureError(f"No WASAPI {mode} devices available.")
        log.info(
            "wasapi.candidates",
            count=len(candidate_indices),
            candidates=candidate_indices,
            loopback=self._loopback,
        )
        candidate_names = []
        for idx in candidate_indices:
            try:
                candidate_names.append(
                    f"{idx}:{sd.query_devices(idx).get('name', '?')} [{_hostapi_name(idx)}]"
                )
            except Exception:
                candidate_names.append(f"{idx}:<query_failed>")
        log.info("wasapi.candidate_devices", devices=candidate_names)

        if self._loopback and not hasattr(sd, "WasapiSettings"):
            raise AudioCaptureError("sounddevice build does not support WASAPI stream settings.")

        def _ordered_sample_rates(default_rate: int) -> list[int | None]:
            # Prefer explicit rates first; samplerate=None is tried last.
            rates: list[int | None] = [default_rate, self._config.sample_rate, 48000, 44100, 32000, 16000, None]
            result: list[int | None] = []
            for rate in rates:
                if rate is None and rate not in result:
                    result.append(rate)
                    continue
                if isinstance(rate, int) and rate > 0 and rate not in result:
                    result.append(rate)
            return result

        def _ordered_channels(max_channels: int) -> list[int | None]:
            preferred = min(max(1, self._config.channels), max_channels)
            order: list[int | None] = [preferred, max_channels, 2, 1, None]
            # Some devices only work with a specific channel count (e.g. 4).
            if max_channels > 2:
                mid = max_channels // 2
                if 1 <= mid <= max_channels:
                    order.append(mid)
            deduped: list[int | None] = []
            for channels in order:
                if channels is None and channels not in deduped:
                    deduped.append(channels)
                elif isinstance(channels, int) and 1 <= channels <= max_channels and channels not in deduped:
                    deduped.append(channels)
            return deduped

        last_error: Exception | None = None
        blocked_hostapis: set[str] = set()

        for device_index in candidate_indices:
            dev = sd.query_devices(device_index)
            max_channels = (
                int(dev.get("max_output_channels", 0))
                if self._loopback
                else int(dev.get("max_input_channels", 0))
            )
            if max_channels <= 0:
                continue

            sample_rates = _ordered_sample_rates(int(dev.get("default_samplerate", 44100)))
            channels_to_try = _ordered_channels(max_channels)
            hostapi_idx = int(dev.get("hostapi", -1))
            hostapi_name = str(hostapis[hostapi_idx].get("name", f"id:{hostapi_idx}"))
            if hostapi_name in blocked_hostapis:
                log.debug(
                    "wasapi.skip_blocked_hostapi",
                    device_index=device_index,
                    device=dev.get("name", str(device_index)),
                    hostapi=hostapi_name,
                )
                continue

            for sample_rate in sample_rates:
                for channels in channels_to_try:
                    kwargs = {
                        "device": device_index,
                        "dtype": "float32",
                        # Let backend choose a valid blocksize for WASAPI.
                        "blocksize": 0,
                        "callback": self._audio_callback,
                    }
                    if channels is not None:
                        kwargs["channels"] = channels
                    if sample_rate is not None:
                        kwargs["samplerate"] = sample_rate
                    if hostapi_idx == wasapi_hostapi:
                        try:
                            kwargs["extra_settings"] = sd.WasapiSettings(
                                loopback=self._loopback,
                                auto_convert=True,
                            )
                        except TypeError:
                            try:
                                kwargs["extra_settings"] = sd.WasapiSettings(
                                    loopback=self._loopback,
                                )
                            except TypeError:
                                kwargs["extra_settings"] = sd.WasapiSettings()

                    try:
                        self._stream = sd.InputStream(**kwargs)
                        self._stream.start()  # type: ignore[union-attr]
                        self._device_index = device_index
                        effective_sample_rate = (
                            sample_rate if sample_rate is not None else int(dev.get("default_samplerate", 44100))
                        )
                        effective_channels = channels if channels is not None else max(1, max_channels)
                        self._config.sample_rate = effective_sample_rate
                        self._config.channels = effective_channels
                        self._active = True
                        self._position = 0
                        if self._loopback:
                            self.__class__._last_working_loopback_device = device_index
                        elif self._source_label == "microphone":
                            # Only remember mic devices — not Stereo Mix or other
                            # virtual inputs used as system audio sources.
                            self.__class__._last_working_input_device = device_index
                        log.info(
                            "wasapi.started",
                            device=dev.get("name", str(device_index)),
                            device_index=device_index,
                            hostapi=hostapi_name,
                            loopback=self._loopback,
                            sample_rate=effective_sample_rate,
                            channels=effective_channels,
                        )
                        return
                    except Exception as exc:
                        last_error = exc
                        message = str(exc)
                        _record_failure(
                            scope="wasapi_config",
                            message=message,
                            device=dev.get("name", str(device_index)),
                            device_index=device_index,
                            hostapi=hostapi_name,
                            sample_rate=sample_rate,
                            channels=channels,
                            loopback=self._loopback,
                        )
                        skip_hostapi = _hostapi_to_skip(message)
                        if skip_hostapi is not None:
                            blocked_hostapis.add(skip_hostapi)
                            break
                        if _is_fatal_device_error(message):
                            break
                else:
                    continue
                break
            else:
                continue
            # Broke due fatal condition; continue with next candidate.
            continue

        mode = "loopback" if self._loopback else "input"
        recent_failures = " | ".join(failure_reasons[-5:]) if failure_reasons else "(none)"
        last_error_text = str(last_error)
        troubleshooting_tip = ""
        if "MME error 11" in last_error_text or "Invalid device" in last_error_text:
            troubleshooting_tip = (
                " Troubleshooting: check Windows microphone privacy for desktop apps, "
                "disable Exclusive Mode for the selected recording device, and verify "
                "the device works in Voice Recorder."
            )
        log.error(
            "wasapi.start_failed",
            mode=mode,
            last_error=last_error_text,
            recent_failures=recent_failures,
        )
        raise AudioCaptureError(
            "Failed to start "
            f"{'WASAPI loopback' if self._loopback else 'audio input'} stream with system default and discovered candidate devices. "
            "Select another device from the GUI device list and retry. "
            f"Last error: {last_error_text}. "
            f"Recent attempts: {recent_failures}."
            f"{troubleshooting_tip}"
        ) from last_error

    async def stop(self) -> None:
        """Stop and close the WASAPI stream."""
        self._active = False
        self._pending = None
        self._loop = None
        if self._stream is not None:
            self._stream.stop()  # type: ignore[union-attr]
            self._stream.close()  # type: ignore[union-attr]
            self._stream = None

        dropped = 0
        while True:
            try:
                self._buffer.get_nowait()
                dropped += 1
            except asyncio.QueueEmpty:
                break
        with suppress(asyncio.QueueFull):
            self._buffer.put_nowait(None)
        log.info("wasapi.stopped", dropped_buffer_chunks=dropped)

    async def read_chunk(self, duration_ms: int = 500) -> AudioChunk:
        """Read the next audio chunk from the buffer."""
        if not self._active and self._buffer.empty():
            raise AudioCaptureError("WASAPICapture is not active")

        target_frames = max(1, int(self._config.sample_rate * duration_ms / 1000))
        pieces: list[np.ndarray] = []
        collected_frames = 0

        if self._pending is not None and self._pending.size > 0:
            pieces.append(self._pending)
            collected_frames += int(self._pending.shape[0])
            self._pending = None

        wait_timeout = max(duration_ms / 1000 * 2, 0.25)
        while collected_frames < target_frames:
            if not self._active and self._buffer.empty():
                break
            try:
                data = await asyncio.wait_for(self._buffer.get(), timeout=wait_timeout)
            except asyncio.TimeoutError:
                if pieces:
                    break
                if self._active:
                    raise
                raise AudioCaptureError("WASAPICapture is not active")

            if data is None:
                if not pieces:
                    raise AudioCaptureError("WASAPICapture is not active")
                break

            if data.ndim == 1:
                data = data[:, np.newaxis]
            pieces.append(data)
            collected_frames += int(data.shape[0])

        if not pieces:
            raise AudioCaptureError("WASAPICapture is not active")

        data = np.concatenate(pieces, axis=0)
        if data.shape[0] > target_frames:
            self._pending = data[target_frames:].copy()
            data = data[:target_frames]

        ts_start = self._position / self._config.sample_rate
        self._position += data.shape[0]
        ts_end = self._position / self._config.sample_rate

        final_samples = data.astype(np.float32).squeeze()
        raw_rms = float(np.sqrt(np.mean(final_samples ** 2)))
        raw_peak = float(np.max(np.abs(final_samples)))
        log.debug(
            "wasapi.chunk_raw",
            source=self._source_label,
            rms=round(raw_rms, 6),
            peak=round(raw_peak, 6),
            frames=final_samples.size,
            sample_rate=self._config.sample_rate,
        )

        return AudioChunk(
            samples=final_samples,
            sample_rate=self._config.sample_rate,
            channels=self._config.channels,
            timestamp_start=ts_start,
            timestamp_end=ts_end,
            source=self._source_label,
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
                    log.debug("wasapi.stream_timeout")
                    continue
                break
            except AudioCaptureError as exc:
                if str(exc) == "WASAPICapture is not active" and not self._active:
                    break
                raise
            except AudioCaptureError:
                if self._active:
                    raise
                break


def find_stereo_mix_device() -> int | None:
    """Return the device index of the first Stereo Mix input device, or None."""
    try:
        import sounddevice as sd

        for idx, dev in enumerate(sd.query_devices()):
            name = str(dev.get("name", "")).lower()
            if "stereo mix" in name and int(dev.get("max_input_channels", 0)) > 0:
                return idx
    except Exception:
        pass
    return None
