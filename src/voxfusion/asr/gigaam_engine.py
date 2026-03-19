"""PyTorch/CTC ASR backend for GigaAM-v3 via HuggingFace transformers."""

from __future__ import annotations

import asyncio
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

import numpy as np
import soundfile as sf

from voxfusion.config.models import ASRConfig
from voxfusion.exceptions import ModelLoadError, TranscriptionError
from voxfusion.logging import get_logger
from voxfusion.models.audio import AudioChunk
from voxfusion.models.transcription import TranscriptionSegment

log = get_logger(__name__)

DEFAULT_GIGAAM_MODEL_REF = "ai-sage/GigaAM-v3"
# The HF repo has only one branch (main) — no revision parameter needed.

# GigaAM raises ValueError for audio longer than 25 s; chunk at 24 s to be safe.
_SAMPLE_RATE = 16000
_CHUNK_DURATION_S = 24
_OVERLAP_DURATION_S = 1
_CHUNK_SAMPLES = _CHUNK_DURATION_S * _SAMPLE_RATE
_OVERLAP_SAMPLES = _OVERLAP_DURATION_S * _SAMPLE_RATE


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


def _install_megatron_compat_shim() -> None:
    """Provide a minimal Megatron shim for third-party imports expecting it."""
    if "megatron.core.num_microbatches_calculator" in sys.modules:
        return

    megatron_mod = sys.modules.setdefault("megatron", types.ModuleType("megatron"))
    core_mod = sys.modules.setdefault("megatron.core", types.ModuleType("megatron.core"))
    calc_mod = types.ModuleType("megatron.core.num_microbatches_calculator")

    def _return_one(*_args: object, **_kwargs: object) -> int:
        return 1

    calc_mod.get_num_microbatches = _return_one  # type: ignore[attr-defined]
    calc_mod.update_num_microbatches = _return_one  # type: ignore[attr-defined]
    calc_mod.reconfigure_num_microbatches_calculator = _return_one  # type: ignore[attr-defined]
    calc_mod.destroy_num_microbatches_calculator = lambda: None  # type: ignore[attr-defined]

    setattr(megatron_mod, "core", core_mod)
    setattr(core_mod, "num_microbatches_calculator", calc_mod)
    sys.modules["megatron.core.num_microbatches_calculator"] = calc_mod


def _install_torchscript_source_fallback(torch_module: object) -> None:
    """Fallback to eager objects when TorchScript cannot access source code."""
    jit = getattr(torch_module, "jit", None)
    if jit is None:
        return
    original_script = getattr(jit, "script", None)
    if original_script is None or getattr(original_script, "_voxfusion_safe_wrapper", False):
        return

    def _safe_script(obj: object, *args: object, **kwargs: object) -> object:
        try:
            return original_script(obj, *args, **kwargs)
        except (OSError, RuntimeError) as exc:
            if "requires source access" not in str(exc).lower():
                raise
            log.warning("gigaam.torchscript_source_fallback", error=str(exc))
            return obj

    setattr(_safe_script, "_voxfusion_safe_wrapper", True)
    jit.script = _safe_script  # type: ignore[assignment]


def _prepare_gigaam_runtime() -> None:
    _prepare_huggingface_runtime_env()
    _install_megatron_compat_shim()
    try:
        import torch
    except ImportError:
        return
    _install_torchscript_source_fallback(torch)


class GigaAMCTCEngine:
    """PyTorch/CTC engine for Russian transcription via ai-sage/GigaAM-v3."""

    def __init__(self, config: ASRConfig | None = None) -> None:
        self._config = config or ASRConfig(model_size="gigaam-v3-e2e-ctc")
        self._model: object | None = None
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
            self._model = AutoModel.from_pretrained(model_ref, **kwargs)
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

    def _ensure_model(self) -> object:
        if self._model is None:
            self.load_model()
        return self._model  # type: ignore[return-value]

    def _transcribe_sync(
        self,
        audio: np.ndarray,
        *,
        language: str | None = None,
    ) -> list[TranscriptionSegment]:
        if language not in (None, "ru"):
            log.warning("gigaam.language_ignored", requested=language, supported="ru")

        model = self._ensure_model()

        try:
            parts: list[str] = []
            pos = 0
            while pos < len(audio):
                chunk = audio[pos : pos + _CHUNK_SAMPLES]
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    tmp_path = f.name
                try:
                    sf.write(tmp_path, chunk, _SAMPLE_RATE, subtype="PCM_16")
                    text = model.transcribe(tmp_path).strip()  # type: ignore[attr-defined]
                    if text:
                        parts.append(text)
                finally:
                    with suppress(OSError):
                        os.unlink(tmp_path)
                pos += _CHUNK_SAMPLES - _OVERLAP_SAMPLES

            text = " ".join(parts).strip()
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
