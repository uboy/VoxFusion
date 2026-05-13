"""PyTorch/CTC ASR backend for GigaAM-v3 via HuggingFace transformers."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import tempfile
import time
import types
import warnings
from collections.abc import AsyncIterator, Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from functools import partial
from pathlib import Path
from typing import Protocol, TypeVar, runtime_checkable

import numpy as np
import soundfile as sf

from voxfusion.asr_catalog import GIGAAM_REVISIONS as _GIGAAM_REVISIONS
from voxfusion.config.models import ASRConfig
from voxfusion.exceptions import ModelLoadError, TranscriptionError
from voxfusion.logging import get_logger
from voxfusion.media.runtime_ffmpeg import activate_ffmpeg_runtime
from voxfusion.models.audio import AudioChunk
from voxfusion.models.transcription import TranscriptionSegment
from voxfusion.runtime_subprocess import patch_subprocess_popen_no_window
from voxfusion.runtime_torchscript import (
    install_torchscript_source_fallback as _install_torchscript_source_fallback,  # noqa: F401 — re-exported for tests
)
from voxfusion.runtime_torchscript import (
    should_use_torchscript_source_fallback as _should_use_torchscript_source_fallback,
)
from voxfusion.runtime_torchscript import (
    temporary_torchscript_source_fallback as _temporary_torchscript_source_fallback,
)

log = get_logger(__name__)

DEFAULT_GIGAAM_MODEL_REF = "ai-sage/GigaAM-v3"

_T = TypeVar("_T")
_TRANSIENT_ERROR_KEYWORDS = ("connection", "timeout", "network", "proxy", "ssl", "reset", "eof")
_DOWNLOAD_MAX_ATTEMPTS = 3
_DOWNLOAD_BACKOFF_BASE_S = 2.0


def _is_transient_error(exc: Exception) -> bool:
    """Return True if *exc* looks like a transient network error worth retrying."""
    msg = str(exc).lower()
    return any(kw in msg for kw in _TRANSIENT_ERROR_KEYWORDS)


def _with_retry(fn: Callable[[], _T], label: str) -> _T:
    """Call *fn()* with exponential-backoff retry for transient network errors.

    Auth errors (401/403) and other permanent failures are re-raised immediately.
    """
    for attempt in range(1, _DOWNLOAD_MAX_ATTEMPTS + 1):
        try:
            return fn()
        except Exception as exc:  # broad: network or HF library may raise any error on download
            if attempt == _DOWNLOAD_MAX_ATTEMPTS or not _is_transient_error(exc):
                raise
            delay = _DOWNLOAD_BACKOFF_BASE_S**attempt
            log.warning(
                "download.retry",
                label=label,
                attempt=attempt,
                max_attempts=_DOWNLOAD_MAX_ATTEMPTS,
                delay_s=delay,
                error=str(exc),
            )
            time.sleep(delay)
    raise RuntimeError("unreachable")  # pragma: no cover


# Minimum free GPU VRAM (MB) required to run GigaAM on CUDA.
# Below this threshold the engine automatically falls back to CPU.
_MIN_CUDA_FREE_MB = 3000

# GigaAM raises ValueError for audio longer than 25 s; chunk at 24 s to be safe.
# transcribe_longform (available in GigaAM-v3 November 2025+) handles long audio
# natively via pyannote VAD and is used when HF_TOKEN is configured.  The manual
# chunking path is the fallback when HF_TOKEN or pyannote/segmentation-3.0 is absent.
_SAMPLE_RATE = 16000
_CHUNK_DURATION_S = 24
_OVERLAP_DURATION_S = 1
_CHUNK_SAMPLES = _CHUNK_DURATION_S * _SAMPLE_RATE
_OVERLAP_SAMPLES = _OVERLAP_DURATION_S * _SAMPLE_RATE
_MIN_TRANSCRIBE_SAMPLES = 320
# Maximum words to inspect at a chunk seam when deduplicating overlap artefacts.
_SEAM_DEDUP_MAX_WORDS = 12


def _dedup_seam(prev: str, curr: str) -> str:
    """Remove from *curr* any word-prefix that duplicates the word-suffix of *prev*.

    The 1-second chunk overlap means the last ~2-3 words of one chunk often
    reappear at the start of the next.  This trims them from *curr* so the
    joined transcript does not contain repeated phrases.
    Used by the manual chunking fallback path.
    """
    prev_words = prev.split()
    curr_words = curr.split()
    limit = min(_SEAM_DEDUP_MAX_WORDS, len(prev_words), len(curr_words))
    for n in range(limit, 0, -1):
        if prev_words[-n:] == curr_words[:n]:
            return " ".join(curr_words[n:])
    return curr


def _resolve_gigaam_device(requested: str) -> str:
    """Return the device to use for GigaAM inference.

    When *requested* is ``'auto'`` or ``'cuda'``, probes free CUDA VRAM.
    Falls back to ``'cpu'`` and logs a warning when VRAM is insufficient.
    """
    if requested == "cpu":
        return "cpu"
    try:
        import torch

        if not torch.cuda.is_available():
            return "cpu"
        free_bytes, _total_bytes = torch.cuda.mem_get_info()
        free_mb = free_bytes // (1024 * 1024)
        if free_mb >= _MIN_CUDA_FREE_MB:
            return "cuda"
        log.warning(
            "gigaam.cuda_memory_low_fallback_cpu",
            free_mb=free_mb,
            required_mb=_MIN_CUDA_FREE_MB,
        )
        return "cpu"
    except Exception:  # broad: torch / CUDA probing can raise any error from native code
        return "cpu"


def _prepare_huggingface_runtime_env() -> None:
    """Normalize Hugging Face cache env vars without deprecated aliases."""
    hf_home = os.environ.get("HF_HOME", "").strip()
    if hf_home:
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(Path(hf_home) / "hub"))
    # Avoid transformers deprecation warning in GUI/binary mode.
    os.environ.pop("TRANSFORMERS_CACHE", None)
    warnings.filterwarnings(
        "ignore",
        message=".*TRANSFORMERS_CACHE.*deprecated.*",
        category=FutureWarning,
    )


def _suppress_gigaam_dependency_noise() -> None:
    """Reduce known-safe third-party noise emitted during GigaAM model import."""
    logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
    logging.getLogger("nv_one_logger").setLevel(logging.ERROR)
    logging.getLogger("nemo").setLevel(logging.ERROR)
    logging.getLogger("nemo_logger").setLevel(logging.ERROR)
    with suppress(Exception):
        from nemo.utils import logging as nemo_logging

        nemo_logging.set_verbosity(nemo_logging.ERROR)

    _disable_pyannote_telemetry()


def _disable_pyannote_telemetry() -> None:
    """Disable pyannote-audio 4.x OpenTelemetry telemetry and stop its background thread.

    pyannote-audio 4.x ships an OpenTelemetry exporter (``pyannote.audio.telemetry``)
    that by default posts usage metrics to ``https://otel.pyannote.ai/v1/metrics``
    every 60 seconds via a ``PeriodicExportingMetricReader`` background thread.
    Setting ``PYANNOTE_METRICS_ENABLED=false`` (done at startup in logging.py) prevents
    data recording, but the background thread and its periodic HTTP POSTs still run.

    This function calls pyannote's own API to:
    1. Disable the flag so no data is recorded.
    2. Shut down the background metric reader thread, eliminating all network traffic.
    """
    with suppress(Exception):
        from pyannote.audio.telemetry.metrics import set_telemetry_metrics

        set_telemetry_metrics(False)

    # Shut down the OpenTelemetry MeterProvider that pyannote installed globally.
    # This stops the PeriodicExportingMetricReader background thread.
    with suppress(Exception):
        from opentelemetry import metrics as _otel_metrics

        prov = _otel_metrics.get_meter_provider()
        if hasattr(prov, "shutdown"):
            prov.shutdown()


def _install_megatron_compat_shim() -> None:
    """Provide a minimal Megatron shim for third-party imports expecting it.

    GigaAM's dependency chain imports ``megatron.core.num_microbatches_calculator``
    even though the actual Megatron-LM training library is not installed.  This
    function injects a stub module that satisfies the import without the real package.

    .. warning::
        This is a fragile compatibility shim.  If upstream libraries (NeMo, pyannote)
        update their Megatron expectations the stubs may need updating.  Check the
        VoxFusion issue tracker if you see ``AttributeError`` on Megatron symbols.
    """
    if "megatron.core.num_microbatches_calculator" in sys.modules:
        return

    log.warning(
        "gigaam.megatron_shim",
        reason=(
            "Injecting Megatron compatibility stub into sys.modules. "
            "This shim is required by GigaAM's dependency chain but is fragile — "
            "if you see unexpected Megatron-related errors, check for upstream updates."
        ),
    )

    megatron_mod = sys.modules.setdefault("megatron", types.ModuleType("megatron"))
    core_mod = sys.modules.setdefault("megatron.core", types.ModuleType("megatron.core"))
    calc_mod = types.ModuleType("megatron.core.num_microbatches_calculator")

    class _ConstantNumMicroBatchesCalculator:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            self.current_global_batch_size = 1
            self.micro_batch_size = 1
            self.num_microbatches = 1

    def _return_one(*_args: object, **_kwargs: object) -> int:
        return 1

    calc_mod.ConstantNumMicroBatchesCalculator = _ConstantNumMicroBatchesCalculator  # type: ignore[attr-defined]
    calc_mod.get_current_global_batch_size = _return_one  # type: ignore[attr-defined]
    calc_mod.get_micro_batch_size = _return_one  # type: ignore[attr-defined]
    calc_mod.get_num_microbatches = _return_one  # type: ignore[attr-defined]
    calc_mod.init_num_microbatches_calculator = lambda *_args, **_kwargs: None  # type: ignore[attr-defined]
    calc_mod.update_num_microbatches = _return_one  # type: ignore[attr-defined]
    calc_mod.reconfigure_num_microbatches_calculator = _return_one  # type: ignore[attr-defined]
    calc_mod.destroy_num_microbatches_calculator = lambda: None  # type: ignore[attr-defined]

    megatron_mod.core = core_mod
    core_mod.num_microbatches_calculator = calc_mod
    sys.modules["megatron.core.num_microbatches_calculator"] = calc_mod


def _suppress_subprocess_windows() -> None:
    """Suppress child console flashes for windowed Windows runtimes."""
    patch_subprocess_popen_no_window()


def _force_torchaudio_soundfile_backend() -> None:
    """Prefer soundfile over SoX for torchaudio audio I/O.

    SoX is an external process; using it in a frozen binary causes a console
    window flash on Windows for every audio file read.  soundfile is a pure
    Python/C extension and works correctly in frozen builds.
    """
    try:
        import torchaudio

        if hasattr(torchaudio, "set_audio_backend"):
            torchaudio.set_audio_backend("soundfile")
    except Exception:  # broad: torchaudio backend selection is best-effort
        pass


def _prepare_gigaam_runtime() -> None:
    _prepare_huggingface_runtime_env()
    _install_megatron_compat_shim()
    _suppress_gigaam_dependency_noise()
    _suppress_subprocess_windows()
    _force_torchaudio_soundfile_backend()
    activate_ffmpeg_runtime()
    try:
        import torch  # noqa: F401 — imported to trigger early ImportError if torch is missing
    except ImportError:
        return


@runtime_checkable
class GigaAMModelProtocol(Protocol):
    """Minimal interface expected from a loaded GigaAM model object."""

    def transcribe(self, wav_path: str) -> str:
        """Transcribe a short WAV file (≤ 25 s) and return the recognised text."""
        ...

    def transcribe_longform(self, wav_path: str) -> list[dict]:
        """Transcribe a long WAV file using pyannote VAD segmentation.

        Returns a list of ``{"transcription": str, "boundaries": (start_s, end_s)}``.
        Available in GigaAM-v3 (November 2025+).  Requires ``HF_TOKEN`` env var and
        a cached ``pyannote/segmentation-3.0`` model.
        """
        ...


class GigaAMCTCEngine:
    """PyTorch/CTC engine for Russian transcription via ai-sage/GigaAM-v3."""

    def __init__(self, config: ASRConfig | None = None) -> None:
        self._config = config or ASRConfig(model_size="gigaam-v3-e2e-ctc")
        self._model: GigaAMModelProtocol | None = None
        self._executor: ThreadPoolExecutor | None = ThreadPoolExecutor(max_workers=1)

    @property
    def model_name(self) -> str:
        return f"gigaam/{self._config.model_size}"

    @property
    def supported_languages(self) -> list[str]:
        return ["ru"]

    def _model_ref(self) -> str:
        if self._config.model_path:
            return self._config.model_path
        return DEFAULT_GIGAAM_MODEL_REF

    def load_model(self) -> None:
        """Load the GigaAM PyTorch model via HuggingFace transformers."""
        if self._model is not None:
            return

        model_ref = self._model_ref()
        local_only = Path(model_ref).exists()
        log.info("asr.loading_model", model=model_ref, engine="gigaam", local_only=local_only)
        _prepare_gigaam_runtime()

        try:
            from transformers import AutoModel
        except ImportError as exc:
            raise ModelLoadError(
                "GigaAM requires these packages:\n"
                "  transformers torch torchaudio sentencepiece omegaconf hydra-core pyannote.audio\n"
                "Install them with:\n"
                "  pip install transformers torch torchaudio sentencepiece omegaconf hydra-core pyannote.audio\n"
            ) from exc

        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None
        log.warning(
            "gigaam.trust_remote_code",
            model_ref=model_ref,
            reason=(
                "GigaAM ships custom Python architecture files that must run locally. "
                "Only load models from sources you trust. "
                "See README.md § Security for details."
            ),
        )
        try:
            kwargs: dict = {"trust_remote_code": True, "token": token}
            if local_only:
                kwargs["local_files_only"] = True
            # Select the correct branch for this model variant.
            revision = _GIGAAM_REVISIONS.get(self._config.model_size)
            if revision and not local_only:
                kwargs["revision"] = revision
                log.info("gigaam.revision_selected", revision=revision)
            try:
                import torch  # type: ignore[import-not-found]
            except ImportError:
                torch = None

            if torch is not None:
                # Use getattr — test mocks may not expose float32.
                float32 = getattr(torch, "float32", None)
                if float32 is not None:
                    kwargs["dtype"] = float32

            if torch is not None and _should_use_torchscript_source_fallback(torch):
                with _temporary_torchscript_source_fallback(torch):
                    self._model = _with_retry(
                        lambda: AutoModel.from_pretrained(model_ref, **kwargs),  # type: ignore[misc]
                        label=model_ref,
                    )
            else:
                self._model = _with_retry(
                    lambda: AutoModel.from_pretrained(model_ref, **kwargs),  # type: ignore[misc]
                    label=model_ref,
                )

            # GigaAM-v3 checkpoint is stored in float16; cast to float32 so that CPU
            # inference doesn't raise "Input type (float) and bias type (c10::Half)".
            # Guard with hasattr so unit-test stubs (which don't have .float/.to) pass.
            if torch is not None and self._model is not None:
                device = _resolve_gigaam_device(self._config.device)
                if callable(getattr(self._model, "float", None)):
                    self._model = self._model.float()
                if callable(getattr(self._model, "to", None)):
                    self._model.to(device)
                log.info("gigaam.device_selected", device=device)
        except (
            Exception
        ) as exc:  # broad: HuggingFace/torch loading surfaces many error types; classified below
            err = str(exc).lower()
            if "401" in err or "unauthorized" in err or "authentication" in err:
                hint = (
                    "The model requires a HuggingFace account token.\n"
                    "  1. Create a free account at https://huggingface.co\n"
                    "  2. Generate a token at https://huggingface.co/settings/tokens\n"
                    "  3. Enter it in VoxFusion Settings → HuggingFace Token"
                )
            elif "403" in err or "gated" in err or "access" in err:
                hint = (
                    "The model is gated — you must accept its license on HuggingFace first.\n"
                    "  1. Visit https://huggingface.co/ai-sage/GigaAM-v3\n"
                    "  2. Accept the model license\n"
                    "  3. Add your HF token in VoxFusion Settings → HuggingFace Token"
                )
            elif "connection" in err or "timeout" in err or "network" in err or "proxy" in err:
                hint = (
                    "Network error while downloading the model.\n"
                    "  - Check your internet connection\n"
                    "  - If behind a proxy, configure it in VoxFusion Settings → Network/Proxy\n"
                    "  - Or pre-download: huggingface-cli download ai-sage/GigaAM-v3"
                )
            else:
                hint = (
                    "  - To download manually: huggingface-cli download ai-sage/GigaAM-v3\n"
                    "  - Or set VOXFUSION_ASR__MODEL_PATH to a local model directory"
                )
            raise ModelLoadError(f"Failed to load GigaAM model: {exc}\n{hint}") from exc

        log.info("asr.model_loaded", model=model_ref, engine="gigaam")

    def unload_model(self) -> None:
        self._model = None
        log.info("asr.model_unloaded", engine="gigaam")

    def close(self) -> None:
        if self._executor is None:
            return
        self._executor.shutdown(wait=False, cancel_futures=True)
        self._executor = None
        log.info("asr.executor_shutdown", engine="gigaam")

    def __del__(self) -> None:
        with suppress(Exception):
            self.close()

    def _ensure_model(self) -> GigaAMModelProtocol:
        if self._model is None:
            self.load_model()
        assert self._model is not None  # load_model raises if load fails
        return self._model

    def _transcribe_sync(
        self,
        audio: np.ndarray,
        *,
        language: str | None = None,
    ) -> list[TranscriptionSegment]:
        if language not in (None, "ru"):
            log.warning("gigaam.language_ignored", requested=language, supported="ru")

        if len(audio) < _MIN_TRANSCRIBE_SAMPLES:
            log.warning(
                "gigaam.audio_too_short",
                samples=len(audio),
                min_samples=_MIN_TRANSCRIBE_SAMPLES,
            )
            return []

        model = self._ensure_model()

        try:
            activate_ffmpeg_runtime()
            total_duration_s = len(audio) / _SAMPLE_RATE
            log.info("gigaam.transcribe_start", duration_s=round(total_duration_s, 1))

            # Write entire audio to a single temp file.  On Linux prefer /dev/shm to avoid disk I/O.
            _tmpdir = "/dev/shm" if sys.platform == "linux" and os.path.isdir("/dev/shm") else None
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=_tmpdir) as f:
                tmp_path = f.name
            try:
                sf.write(tmp_path, audio, _SAMPLE_RATE, subtype="PCM_16")

                # Prefer model's native longform transcription (pyannote VAD, real timestamps).
                # Falls back to manual chunking when HF_TOKEN is absent or the VAD model is not cached.
                if total_duration_s > _CHUNK_DURATION_S and callable(
                    getattr(model, "transcribe_longform", None)
                ):
                    try:
                        return self._try_transcribe_longform(model, tmp_path, total_duration_s)
                    except Exception as exc:
                        log.warning(
                            "gigaam.longform_unavailable",
                            reason=str(exc)[:200],
                            fallback="manual_chunking",
                        )

                # Manual chunking fallback: fixed 24-second windows with 1-second overlap.
                return self._transcribe_chunked(model, audio, total_duration_s)
            finally:
                with suppress(OSError):
                    os.unlink(tmp_path)
        except Exception as exc:  # broad: model inference, numpy, soundfile, or tempfile may fail
            raise TranscriptionError(f"GigaAM transcription failed: {exc}") from exc

    def _try_transcribe_longform(
        self,
        model: GigaAMModelProtocol,
        wav_path: str,
        total_duration_s: float,
    ) -> list[TranscriptionSegment]:
        """Use GigaAM's built-in pyannote-VAD longform transcription.

        Returns one TranscriptionSegment per VAD-detected speech segment with
        accurate start/end timestamps.  Requires HF_TOKEN and a cached
        ``pyannote/segmentation-3.0`` model.
        """
        raw_segments: list[dict] = model.transcribe_longform(wav_path)
        segments: list[TranscriptionSegment] = []
        for seg in raw_segments:
            text = seg.get("transcription", "").strip()
            boundaries = seg.get("boundaries", (0.0, total_duration_s))
            start_s, end_s = float(boundaries[0]), float(boundaries[1])
            if text:
                segments.append(
                    TranscriptionSegment(
                        text=text,
                        language="ru",
                        start_time=start_s,
                        end_time=end_s,
                        confidence=0.0,
                        words=None,
                        no_speech_prob=0.0,
                    )
                )
        log.info(
            "gigaam.transcribe_done",
            mode="longform",
            segments=len(segments),
            result_chars=sum(len(s.text) for s in segments),
        )
        return segments

    def _transcribe_chunked(
        self,
        model: GigaAMModelProtocol,
        audio: np.ndarray,
        total_duration_s: float,
    ) -> list[TranscriptionSegment]:
        """Fallback: fixed 24-second windows with seam dedup and per-window timestamps."""
        total_chunks = max(1, -(-len(audio) // (_CHUNK_SAMPLES - _OVERLAP_SAMPLES)))
        log.info("gigaam.chunked_start", duration_s=round(total_duration_s, 1), chunks=total_chunks)

        _tmpdir = "/dev/shm" if sys.platform == "linux" and os.path.isdir("/dev/shm") else None
        parts: list[tuple[float, float, str]] = []
        pos = 0
        chunk_idx = 0
        while pos < len(audio):
            chunk_idx += 1
            chunk = audio[pos : pos + _CHUNK_SAMPLES]
            chunk_start_s = round(pos / _SAMPLE_RATE, 1)
            chunk_end_s = round(min(pos + _CHUNK_SAMPLES, len(audio)) / _SAMPLE_RATE, 1)
            log.info(
                "gigaam.chunk_start",
                chunk=chunk_idx,
                of=total_chunks,
                start_s=chunk_start_s,
                end_s=chunk_end_s,
            )
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=_tmpdir) as f:
                chunk_path = f.name
            try:
                sf.write(chunk_path, chunk, _SAMPLE_RATE, subtype="PCM_16")
                text = model.transcribe(chunk_path).strip()
                if text:
                    parts.append((chunk_start_s, chunk_end_s, text))
                    log.info("gigaam.chunk_done", chunk=chunk_idx, of=total_chunks, text=text[:80])
                else:
                    log.info(
                        "gigaam.chunk_done", chunk=chunk_idx, of=total_chunks, text="(silence)"
                    )
            finally:
                with suppress(OSError):
                    os.unlink(chunk_path)
            pos += _CHUNK_SAMPLES - _OVERLAP_SAMPLES

        # Deduplicate words duplicated at seam boundaries due to overlap.
        deduped_parts: list[tuple[float, float, str]] = []
        for chunk_start_s, chunk_end_s, part_text in parts:
            prev_text = deduped_parts[-1][2] if deduped_parts else ""
            clean = _dedup_seam(prev_text, part_text) if deduped_parts else part_text
            if clean:
                deduped_parts.append((chunk_start_s, chunk_end_s, clean))

        result_chars = sum(len(text) for _, _, text in deduped_parts)
        log.info(
            "gigaam.transcribe_done", mode="chunked", chunks=chunk_idx, result_chars=result_chars
        )
        if not deduped_parts:
            return []

        segments: list[TranscriptionSegment] = []
        for chunk_start_s, chunk_end_s, text in deduped_parts:
            segments.append(
                TranscriptionSegment(
                    text=text,
                    language="ru",
                    start_time=max(0.0, min(chunk_start_s, total_duration_s)),
                    end_time=max(0.0, min(chunk_end_s, total_duration_s)),
                    confidence=0.0,
                    words=None,
                    no_speech_prob=0.0,
                )
            )
        return segments

    def transcribe_samples_sync(
        self,
        samples: np.ndarray,
        *,
        sample_rate: int = _SAMPLE_RATE,
        language: str | None = None,
    ) -> list[TranscriptionSegment]:
        """Synchronously transcribe a numpy waveform."""
        mono = self._normalize_audio(samples, sample_rate)
        return self._transcribe_sync(mono, language=language)

    @staticmethod
    def _normalize_audio(samples: np.ndarray, sample_rate: int) -> np.ndarray:
        audio = np.asarray(samples, dtype=np.float32)
        if audio.ndim == 0:
            audio = audio.reshape(1)
        elif audio.ndim == 2:
            audio = audio.mean(axis=1, dtype=np.float32)
        elif audio.ndim > 2:
            audio = audio.reshape(audio.shape[0], -1).mean(axis=1, dtype=np.float32)
        audio = np.ascontiguousarray(audio.reshape(-1), dtype=np.float32)

        if sample_rate != _SAMPLE_RATE:
            duration = len(audio) / sample_rate
            target_samples = max(1, int(duration * _SAMPLE_RATE))
            xs_old = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
            xs_new = np.linspace(0.0, 1.0, num=target_samples, endpoint=False)
            audio = np.interp(xs_new, xs_old, audio).astype(np.float32)
        return audio

    async def transcribe(
        self,
        audio: AudioChunk,
        *,
        language: str | None = None,
        initial_prompt: str | None = None,
        word_timestamps: bool = False,
    ) -> list[TranscriptionSegment]:
        """Transcribe an audio chunk."""
        del initial_prompt, word_timestamps
        loop = asyncio.get_running_loop()
        mono = self._normalize_audio(audio.samples, audio.sample_rate)
        executor = self._executor
        if executor is None:
            raise TranscriptionError("GigaAM executor is not available.")
        return await loop.run_in_executor(
            executor,
            partial(self._transcribe_sync, mono, language=language),
        )

    async def transcribe_stream(
        self,
        audio_stream: AsyncIterator[AudioChunk],
        *,
        language: str | None = None,
    ) -> AsyncIterator[TranscriptionSegment]:
        """GigaAM does not currently support streaming transcription."""
        del audio_stream, language
        if False:  # pragma: no cover
            yield
        raise TranscriptionError("GigaAM v3 is only supported for file/batch transcription.")
