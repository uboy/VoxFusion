"""Transformers-based Whisper backend for Breeze ASR models."""

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

DEFAULT_BREEZE_MODEL_REF = "MediaTek-Research/Breeze-ASR-25"


class BreezeASREngine:
    """Batch-oriented Whisper-style backend for Breeze ASR."""

    def __init__(self, config: ASRConfig | None = None) -> None:
        self._config = config or ASRConfig(model_size="breeze-asr")
        self._pipeline: object | None = None
        self._executor: ThreadPoolExecutor | None = ThreadPoolExecutor(max_workers=1)

    @property
    def model_name(self) -> str:
        return f"breeze/{self._config.model_size}"

    @property
    def supported_languages(self) -> list[str]:
        return ["zh", "en"]

    def _model_ref(self) -> str:
        if self._config.model_path:
            return self._config.model_path
        return DEFAULT_BREEZE_MODEL_REF

    def load_model(self) -> None:
        if self._pipeline is not None:
            return

        model_ref = self._model_ref()
        local_only = Path(model_ref).exists()
        log.info("asr.loading_model", model=model_ref, engine="breeze", local_only=local_only)
        try:
            import torch
            from transformers import (
                AutoModelForSpeechSeq2Seq,
                AutoProcessor,
                pipeline,
            )
        except ImportError as exc:
            raise ModelLoadError(
                "Breeze backend requires transformers and torch. "
                "Install them in the active environment and retry file transcription."
            ) from exc

        try:
            processor = AutoProcessor.from_pretrained(model_ref, local_files_only=local_only)
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_ref,
                local_files_only=local_only,
            )
            self._pipeline = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                torch_dtype=torch.float32,
                device=-1,
            )
        except Exception as exc:
            raise ModelLoadError(
                "Failed to load Breeze model. Set VOXFUSION_ASR__MODEL_PATH to a local model "
                f"directory or ensure the model is available in Hugging Face cache. Details: {exc}"
            ) from exc

        log.info("asr.model_loaded", model=model_ref, engine="breeze")

    def unload_model(self) -> None:
        self._pipeline = None
        log.info("asr.model_unloaded", engine="breeze")

    def close(self) -> None:
        if self._executor is None:
            return
        self._executor.shutdown(wait=False, cancel_futures=True)
        self._executor = None
        log.info("asr.executor_shutdown", engine="breeze")

    def __del__(self) -> None:
        with suppress(Exception):
            self.close()

    def _ensure_pipeline(self):
        if self._pipeline is None:
            self.load_model()
        return self._pipeline

    def _transcribe_sync(self, audio: np.ndarray, *, language: str | None = None) -> list[TranscriptionSegment]:
        pipe = self._ensure_pipeline()
        generate_kwargs: dict[str, object] = {"task": "transcribe"}
        if language:
            generate_kwargs["language"] = language
        try:
            result = pipe(  # type: ignore[operator]
                {"raw": audio, "sampling_rate": 16000},
                generate_kwargs=generate_kwargs,
                return_timestamps=False,
            )
            text = str(result.get("text", "")).strip()
        except Exception as exc:
            raise TranscriptionError(f"Breeze transcription failed: {exc}") from exc

        if not text:
            return []

        return [
            TranscriptionSegment(
                text=text,
                language=language or "zh",
                start_time=0.0,
                end_time=len(audio) / 16000.0,
                confidence=0.0,
                words=None,
                no_speech_prob=0.0,
            )
        ]

    async def transcribe(
        self,
        audio: AudioChunk,
        *,
        language: str | None = None,
        initial_prompt: str | None = None,
        word_timestamps: bool = False,
    ) -> list[TranscriptionSegment]:
        del initial_prompt, word_timestamps
        loop = asyncio.get_running_loop()
        if audio.samples.ndim > 1:
            mono = audio.samples.mean(axis=1, dtype=np.float32)
        else:
            mono = np.asarray(audio.samples, dtype=np.float32)
        if audio.sample_rate != 16000:
            duration = len(mono) / audio.sample_rate
            target_samples = max(1, int(duration * 16000))
            xs_old = np.linspace(0.0, 1.0, num=len(mono), endpoint=False)
            xs_new = np.linspace(0.0, 1.0, num=target_samples, endpoint=False)
            mono = np.interp(xs_new, xs_old, mono).astype(np.float32)
        executor = self._executor
        if executor is None:
            raise TranscriptionError("Breeze executor is not available.")
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
        del audio_stream, language
        if False:  # pragma: no cover
            yield
        raise TranscriptionError("Breeze ASR is only supported for file/batch transcription.")
