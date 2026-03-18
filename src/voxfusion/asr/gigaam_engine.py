"""ONNX/CTC ASR backend for GigaAM-style models."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from functools import partial
from pathlib import Path

import numpy as np

from voxfusion.config.models import ASRConfig
from voxfusion.exceptions import ModelLoadError, TranscriptionError
from voxfusion.logging import get_logger
from voxfusion.models.audio import AudioChunk
from voxfusion.models.transcription import TranscriptionSegment

log = get_logger(__name__)

DEFAULT_GIGAAM_MODEL_REF = "salute-developers/GigaAM-CTC-v3"


class GigaAMCTCEngine:
    """Batch-friendly ONNX/CTC engine for Russian transcription."""

    def __init__(self, config: ASRConfig | None = None) -> None:
        self._config = config or ASRConfig(model_size="gigaam-v3-e2e-ctc")
        self._processor: object | None = None
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
        """Load processor and ONNX model."""
        if self._model is not None and self._processor is not None:
            return

        model_ref = self._model_ref()
        local_only = Path(model_ref).exists()
        log.info("asr.loading_model", model=model_ref, engine="gigaam", local_only=local_only)
        try:
            from optimum.onnxruntime import ORTModelForCTC
            from transformers import AutoProcessor
        except ImportError as exc:
            raise ModelLoadError(
                "GigaAM backend requires optimum[onnxruntime] and transformers."
            ) from exc

        try:
            self._processor = AutoProcessor.from_pretrained(model_ref, local_files_only=local_only)
            self._model = ORTModelForCTC.from_pretrained(
                model_ref,
                local_files_only=local_only,
                provider="CPUExecutionProvider",
            )
        except Exception as exc:
            raise ModelLoadError(
                "Failed to load GigaAM model. Set VOXFUSION_ASR__MODEL_PATH to a local model "
                f"directory or ensure the model is available in Hugging Face cache. Details: {exc}"
            ) from exc

        log.info("asr.model_loaded", model=model_ref, engine="gigaam")

    def unload_model(self) -> None:
        self._model = None
        self._processor = None
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

    def _ensure_model(self) -> tuple[object, object]:
        if self._model is None or self._processor is None:
            self.load_model()
        return self._model, self._processor  # type: ignore[return-value]

    def _transcribe_sync(
        self,
        audio: np.ndarray,
        *,
        language: str | None = None,
    ) -> list[TranscriptionSegment]:
        model, processor = self._ensure_model()
        if language not in (None, "ru"):
            log.warning("gigaam.language_ignored", requested=language, supported="ru")

        try:
            inputs = processor(
                audio,
                sampling_rate=16000,
                return_tensors="np",
            )
            outputs = model(**inputs)
            logits = outputs.logits
            token_ids = np.asarray(logits).argmax(axis=-1)
            text = processor.batch_decode(token_ids, skip_special_tokens=True)[0].strip()
        except Exception as exc:
            raise TranscriptionError(f"GigaAM transcription failed: {exc}") from exc

        if not text:
            return []

        return [
            TranscriptionSegment(
                text=text,
                language="ru",
                start_time=0.0,
                end_time=len(audio) / 16000.0,
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

        if sample_rate != 16000:
            duration = len(audio) / sample_rate
            target_samples = max(1, int(duration * 16000))
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
