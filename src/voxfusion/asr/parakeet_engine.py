"""NeMo-based backend for NVIDIA Parakeet models."""

from __future__ import annotations

import asyncio
import tempfile
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

DEFAULT_PARAKEET_MODEL_REF = "nvidia/parakeet-tdt-0.6b-v3"


class ParakeetASREngine:
    """Batch-oriented NeMo backend for Parakeet ASR."""

    def __init__(self, config: ASRConfig | None = None) -> None:
        self._config = config or ASRConfig(model_size="parakeet-tdt-0.6b-v3")
        self._model: object | None = None
        self._executor: ThreadPoolExecutor | None = ThreadPoolExecutor(max_workers=1)

    @property
    def model_name(self) -> str:
        return f"parakeet/{self._config.model_size}"

    @property
    def supported_languages(self) -> list[str]:
        return ["en", "bg", "hr", "cs", "da", "nl", "et", "fi", "fr", "de", "el", "hu", "it", "lv", "lt", "mt", "pl", "pt", "ro", "sk", "sl", "es", "sv", "ru", "uk"]

    def _model_ref(self) -> str:
        if self._config.model_path:
            return self._config.model_path
        return DEFAULT_PARAKEET_MODEL_REF

    def load_model(self) -> None:
        if self._model is not None:
            return

        model_ref = self._model_ref()
        local_path = Path(model_ref)
        log.info("asr.loading_model", model=model_ref, engine="parakeet", local_only=local_path.exists())
        try:
            from nemo.collections.asr.models import ASRModel
        except ImportError as exc:
            raise ModelLoadError(
                "Parakeet backend requires nemo_toolkit['asr']. "
                "Install it in the active environment and retry file transcription."
            ) from exc

        try:
            if local_path.exists():
                if local_path.is_file():
                    self._model = ASRModel.restore_from(restore_path=str(local_path))
                else:
                    self._model = ASRModel.from_pretrained(model_name=str(local_path))
            else:
                self._model = ASRModel.from_pretrained(model_name=model_ref)
        except Exception as exc:
            raise ModelLoadError(
                "Failed to load Parakeet model. Set VOXFUSION_ASR__MODEL_PATH to a local .nemo file "
                f"or use an available NeMo/Hugging Face model reference. Details: {exc}"
            ) from exc

        log.info("asr.model_loaded", model=model_ref, engine="parakeet")

    def unload_model(self) -> None:
        self._model = None
        log.info("asr.model_unloaded", engine="parakeet")

    def close(self) -> None:
        if self._executor is None:
            return
        self._executor.shutdown(wait=False, cancel_futures=True)
        self._executor = None
        log.info("asr.executor_shutdown", engine="parakeet")

    def __del__(self) -> None:
        with suppress(Exception):
            self.close()

    def _ensure_model(self):
        if self._model is None:
            self.load_model()
        return self._model

    @staticmethod
    def _extract_text(result: object) -> str:
        if isinstance(result, str):
            return result.strip()
        if isinstance(result, list) and result:
            return ParakeetASREngine._extract_text(result[0])
        if hasattr(result, "text"):
            return str(result.text).strip()
        if hasattr(result, "hypotheses"):
            hypotheses = getattr(result, "hypotheses")
            if hypotheses:
                return ParakeetASREngine._extract_text(hypotheses[0])
        return str(result).strip()

    def _transcribe_sync(self, audio: np.ndarray, *, language: str | None = None) -> list[TranscriptionSegment]:
        del language
        model = self._ensure_model()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            wav_path = Path(handle.name)
        try:
            sf.write(wav_path, audio, 16000)
            result = model.transcribe([str(wav_path)])  # type: ignore[union-attr]
            text = self._extract_text(result)
        except Exception as exc:
            raise TranscriptionError(f"Parakeet transcription failed: {exc}") from exc
        finally:
            wav_path.unlink(missing_ok=True)

        if not text:
            return []

        return [
            TranscriptionSegment(
                text=text,
                language="en",
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
            raise TranscriptionError("Parakeet executor is not available.")
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
        raise TranscriptionError("Parakeet ASR is only supported for file/batch transcription.")
