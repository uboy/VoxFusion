"""FunASR Paraformer backend for Chinese (zh) ASR."""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from functools import partial

import numpy as np
import soundfile as sf

from voxfusion.config.models import ASRConfig
from voxfusion.exceptions import ModelLoadError, TranscriptionError
from voxfusion.logging import get_logger
from voxfusion.media.runtime_ffmpeg import activate_ffmpeg_runtime
from voxfusion.models.audio import AudioChunk
from voxfusion.models.transcription import TranscriptionSegment

log = get_logger(__name__)

DEFAULT_FUNASR_MODEL = "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
_SAMPLE_RATE = 16000
_MIN_TRANSCRIBE_SAMPLES = 320


class FunASREngine:
    """Batch-oriented FunASR Paraformer engine for Chinese transcription."""

    def __init__(self, config: ASRConfig | None = None) -> None:
        self._config = config or ASRConfig(model_size="funasr-paraformer-zh")
        self._model: object | None = None
        self._executor: ThreadPoolExecutor | None = ThreadPoolExecutor(max_workers=1)

    @property
    def model_name(self) -> str:
        return f"funasr/{self._config.model_size}"

    @property
    def supported_languages(self) -> list[str]:
        return ["zh"]

    def _model_ref(self) -> str:
        if self._config.model_path:
            return self._config.model_path
        return DEFAULT_FUNASR_MODEL

    def load_model(self) -> None:
        if self._model is not None:
            return

        model_ref = self._model_ref()
        local_only = os.path.isdir(model_ref)
        log.info("asr.loading_model", model=model_ref, engine="funasr", local_only=local_only)
        activate_ffmpeg_runtime()

        try:
            from funasr import AutoModel
        except ImportError as exc:
            raise ModelLoadError(
                "FunASR backend requires the 'funasr' package.\n"
                "Install it with:\n"
                "  pip install funasr torch torchaudio\n"
                "Or add the 'chinese' extra:\n"
                "  poetry install --extras chinese"
            ) from exc

        try:
            kwargs: dict = {}
            if local_only:
                kwargs["hub"] = "local"
            self._model = AutoModel(model=model_ref, **kwargs)
        except Exception as exc:
            raise ModelLoadError(
                f"Failed to load FunASR model '{model_ref}'.\n"
                f"  Details: {exc}\n"
                f"  Set VOXFUSION_ASR__MODEL_PATH to a local model directory "
                f"or ensure the model is cached."
            ) from exc

        log.info("asr.model_loaded", model=model_ref, engine="funasr")

    def unload_model(self) -> None:
        self._model = None
        log.info("asr.model_unloaded", engine="funasr")

    def close(self) -> None:
        if self._executor is None:
            return
        self._executor.shutdown(wait=False, cancel_futures=True)
        self._executor = None
        log.info("asr.executor_shutdown", engine="funasr")

    def __del__(self) -> None:
        with suppress(Exception):
            self.close()

    def _ensure_model(self) -> object:
        if self._model is None:
            self.load_model()
        assert self._model is not None
        return self._model

    def _transcribe_sync(
        self,
        audio: np.ndarray,
        *,
        language: str | None = None,
    ) -> list[TranscriptionSegment]:
        if language not in (None, "zh"):
            log.warning("funasr.language_ignored", requested=language, supported="zh")

        if len(audio) < _MIN_TRANSCRIBE_SAMPLES:
            log.warning(
                "funasr.audio_too_short",
                samples=len(audio),
                min_samples=_MIN_TRANSCRIBE_SAMPLES,
            )
            return []

        model = self._ensure_model()

        try:
            activate_ffmpeg_runtime()
            total_duration_s = len(audio) / _SAMPLE_RATE
            log.info("funasr.transcribe_start", duration_s=round(total_duration_s, 1))

            _tmpdir = "/dev/shm" if sys.platform == "linux" and os.path.isdir("/dev/shm") else None
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=_tmpdir) as f:
                tmp_path = f.name
            try:
                sf.write(tmp_path, audio, _SAMPLE_RATE, subtype="PCM_16")
                result = model.generate(input=tmp_path)
            finally:
                with suppress(OSError):
                    os.unlink(tmp_path)

            # FunASR returns a list of dicts with "text" key.
            text = ""
            if isinstance(result, list):
                for item in result:
                    if isinstance(item, dict):
                        text += item.get("text", "")
                    elif isinstance(item, str):
                        text += item
            elif isinstance(result, str):
                text = result

            text = text.strip()
            log.info("funasr.transcribe_done", result_chars=len(text))

            if not text:
                return []

            return [
                TranscriptionSegment(
                    text=text,
                    language="zh",
                    start_time=0.0,
                    end_time=total_duration_s,
                    confidence=0.0,
                    words=None,
                    no_speech_prob=0.0,
                )
            ]
        except Exception as exc:
            raise TranscriptionError(f"FunASR transcription failed: {exc}") from exc

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
        del initial_prompt, word_timestamps
        loop = asyncio.get_running_loop()
        mono = self._normalize_audio(audio.samples, audio.sample_rate)
        executor = self._executor
        if executor is None:
            raise TranscriptionError("FunASR executor is not available.")
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
        raise TranscriptionError("FunASR is only supported for file/batch transcription.")
