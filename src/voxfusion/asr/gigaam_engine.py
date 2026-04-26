"""PyTorch/CTC ASR backend for GigaAM-v3 via HuggingFace transformers."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import tempfile
import types
import warnings
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from functools import partial
from pathlib import Path
from typing import Protocol, runtime_checkable

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

# Minimum free GPU VRAM (MB) required to run GigaAM on CUDA.
# Below this threshold the engine automatically falls back to CPU.
_MIN_CUDA_FREE_MB = 3000

# GigaAM raises ValueError for audio longer than 25 s; chunk at 24 s to be safe.
_SAMPLE_RATE = 16000
_CHUNK_DURATION_S = 24
_OVERLAP_DURATION_S = 1
_CHUNK_SAMPLES = _CHUNK_DURATION_S * _SAMPLE_RATE
_OVERLAP_SAMPLES = _OVERLAP_DURATION_S * _SAMPLE_RATE
_MIN_TRANSCRIBE_SAMPLES = 320


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
    except Exception:
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


def _install_megatron_compat_shim() -> None:
    """Provide a minimal Megatron shim for third-party imports expecting it."""
    if "megatron.core.num_microbatches_calculator" in sys.modules:
        return

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
    except Exception:
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
        """Transcribe a WAV file and return the recognised text."""
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
                    kwargs["torch_dtype"] = float32

            if torch is not None and _should_use_torchscript_source_fallback(torch):
                with _temporary_torchscript_source_fallback(torch):
                    self._model = AutoModel.from_pretrained(model_ref, **kwargs)
            else:
                self._model = AutoModel.from_pretrained(model_ref, **kwargs)

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
        except Exception as exc:
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
            total_chunks = max(1, -(-len(audio) // (_CHUNK_SAMPLES - _OVERLAP_SAMPLES)))
            log.info(
                "gigaam.transcribe_start",
                duration_s=round(total_duration_s, 1),
                chunks=total_chunks,
            )

            parts: list[str] = []
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
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    tmp_path = f.name
                try:
                    sf.write(tmp_path, chunk, _SAMPLE_RATE, subtype="PCM_16")
                    text = model.transcribe(tmp_path).strip()
                    if text:
                        parts.append(text)
                        log.info(
                            "gigaam.chunk_done", chunk=chunk_idx, of=total_chunks, text=text[:80]
                        )
                    else:
                        log.info(
                            "gigaam.chunk_done", chunk=chunk_idx, of=total_chunks, text="(silence)"
                        )
                finally:
                    with suppress(OSError):
                        os.unlink(tmp_path)
                pos += _CHUNK_SAMPLES - _OVERLAP_SAMPLES

            text = " ".join(parts).strip()
            log.info("gigaam.transcribe_done", chunks=chunk_idx, result_chars=len(text))
        except Exception as exc:
            raise TranscriptionError(f"GigaAM transcription failed: {exc}") from exc

        if not text:
            return []

        return [
            TranscriptionSegment(
                text=text,
                language="ru",
                start_time=0.0,
                end_time=len(audio) / float(_SAMPLE_RATE),
                confidence=0.0,
                words=None,
                no_speech_prob=0.0,
            )
        ]

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
