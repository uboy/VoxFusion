"""Primary ASR engine implementation using faster-whisper (CTranslate2).

Wraps the ``faster_whisper.WhisperModel`` to conform to the
``ASREngine`` protocol. CPU-bound inference is offloaded to a
single-worker ``ThreadPoolExecutor`` via ``asyncio.loop.run_in_executor()``.
"""

import asyncio
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from functools import partial
from time import monotonic

import numpy as np

from voxfusion.config.models import ASRConfig
from voxfusion.exceptions import ModelLoadError, ModelNotFoundError, TranscriptionError
from voxfusion.logging import get_logger
from voxfusion.models.audio import AudioChunk
from voxfusion.models.transcription import TranscriptionSegment, WordTiming

log = get_logger(__name__)

_HALLUCINATION_PATTERNS: frozenset[str] = frozenset({
    "продолжение следует",
    "субтитр",
    "редактор субтитров",
    "переведено",
    "над субтитрами работал",
    "перевод субтитров",
    "thank you for watching",
    "subscribe",
    "like and subscribe",
    "www.",
    "http",
})


def _is_hallucination(text: str) -> bool:
    """Return True if *text* looks like a Whisper hallucination artifact."""
    t = text.strip().lower()
    if len(t) < 2:
        return True
    for pattern in _HALLUCINATION_PATTERNS:
        if pattern in t:
            return True
    return False


def _resolve_device(device: str) -> tuple[str, str]:
    """Resolve ``'auto'`` device to ``('cuda', 'float16')`` or ``('cpu', 'int8')``."""
    if device == "auto":
        try:
            import ctranslate2

            if "cuda" in ctranslate2.get_supported_compute_types("cuda"):
                return "cuda", "float16"
        except Exception:
            pass
        return "cpu", "int8"
    return device, "float16" if device == "cuda" else "int8"


class FasterWhisperEngine:
    """ASR engine backed by faster-whisper.

    The model is lazily loaded on first ``transcribe`` call or
    explicitly via ``load_model()``.
    """

    def __init__(self, config: ASRConfig | None = None) -> None:
        self._config = config or ASRConfig()
        self._model: object | None = None  # faster_whisper.WhisperModel
        self._executor: ThreadPoolExecutor | None = ThreadPoolExecutor(max_workers=1)
        self._last_empty_log_ts = 0.0
        self._empty_batch_count = 0

    @property
    def model_name(self) -> str:
        return f"faster-whisper/{self._config.model_size}"

    @property
    def supported_languages(self) -> list[str]:
        return [
            "af", "am", "ar", "as", "az", "ba", "be", "bg", "bn", "bo",
            "br", "bs", "ca", "cs", "cy", "da", "de", "el", "en", "es",
            "et", "eu", "fa", "fi", "fo", "fr", "gl", "gu", "ha", "haw",
            "he", "hi", "hr", "ht", "hu", "hy", "id", "is", "it", "ja",
            "jw", "ka", "kk", "km", "kn", "ko", "la", "lb", "ln", "lo",
            "lt", "lv", "mg", "mi", "mk", "ml", "mn", "mr", "ms", "mt",
            "my", "ne", "nl", "nn", "no", "oc", "pa", "pl", "ps", "pt",
            "ro", "ru", "sa", "sd", "si", "sk", "sl", "sn", "so", "sq",
            "sr", "su", "sv", "sw", "ta", "te", "tg", "th", "tk", "tl",
            "tr", "tt", "uk", "ur", "uz", "vi", "yi", "yo", "zh",
        ]

    def load_model(self) -> None:
        """Pre-load the faster-whisper model into memory."""
        if self._model is not None:
            return

        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise ModelLoadError(
                "faster-whisper is not installed. "
                "Install with: pip install faster-whisper"
            ) from exc

        device, compute_type = _resolve_device(self._config.device)
        if self._config.compute_type != "auto":
            compute_type = self._config.compute_type

        model_size = self._config.model_size
        cpu_threads = self._config.cpu_threads  # 0 = auto (all cores)
        log.info(
            "asr.loading_model",
            model=model_size,
            device=device,
            compute_type=compute_type,
            cpu_threads=cpu_threads or "auto",
        )

        try:
            self._model = WhisperModel(
                model_size,
                device=device,
                compute_type=compute_type,
                cpu_threads=cpu_threads,
            )
        except ValueError as exc:
            raise ModelNotFoundError(
                f"Model '{model_size}' not found or failed to download: {exc}"
            ) from exc
        except Exception as exc:
            raise ModelLoadError(f"Failed to load model '{model_size}': {exc}") from exc

        log.info("asr.model_loaded", model=model_size)

    def unload_model(self) -> None:
        """Release the model from memory."""
        self._model = None
        log.info("asr.model_unloaded")

    def close(self) -> None:
        """Release background resources."""
        if self._executor is None:
            return
        self._executor.shutdown(wait=False, cancel_futures=True)
        self._executor = None
        log.info("asr.executor_shutdown")

    def __del__(self) -> None:
        with suppress(Exception):
            self.close()

    def _ensure_model(self) -> object:
        """Lazily load the model if not already loaded."""
        if self._model is None:
            self.load_model()
        return self._model  # type: ignore[return-value]

    def _transcribe_sync(
        self,
        audio: np.ndarray,
        *,
        language: str | None = None,
        initial_prompt: str | None = None,
        word_timestamps: bool = False,
    ) -> list[TranscriptionSegment]:
        """Synchronous transcription — called inside the executor."""
        model = self._ensure_model()

        kwargs: dict = {
            "beam_size": self._config.beam_size,
            "best_of": self._config.best_of,
            "patience": self._config.patience,
            "word_timestamps": word_timestamps or self._config.word_timestamps,
            "vad_filter": self._config.vad_filter,
        }
        if language:
            kwargs["language"] = language
        elif self._config.language:
            kwargs["language"] = self._config.language

        prompt = initial_prompt or self._config.initial_prompt
        if prompt:
            kwargs["initial_prompt"] = prompt

        if self._config.vad_filter:
            kwargs["vad_parameters"] = {
                "threshold": self._config.vad_parameters.threshold,
                "min_speech_duration_ms": self._config.vad_parameters.min_speech_duration_ms,
                "min_silence_duration_ms": self._config.vad_parameters.min_silence_duration_ms,
            }

        try:
            segments_iter, info = model.transcribe(audio, **kwargs)  # type: ignore[union-attr]
        except Exception as exc:
            raise TranscriptionError(f"Transcription failed: {exc}") from exc

        detected_language = info.language
        results: list[TranscriptionSegment] = []

        for seg in segments_iter:
            words: list[WordTiming] | None = None
            if seg.words:
                words = [
                    WordTiming(
                        word=w.word,
                        start_time=w.start,
                        end_time=w.end,
                        probability=w.probability,
                    )
                    for w in seg.words
                ]

            results.append(
                TranscriptionSegment(
                    text=seg.text.strip(),
                    language=detected_language,
                    start_time=seg.start,
                    end_time=seg.end,
                    confidence=seg.avg_logprob,
                    words=words,
                    no_speech_prob=seg.no_speech_prob,
                )
            )

        results = [
            seg for seg in results
            if seg.no_speech_prob < self._config.no_speech_threshold
        ]
        results = [seg for seg in results if not _is_hallucination(seg.text)]

        if results:
            self._empty_batch_count = 0
            self._last_empty_log_ts = monotonic()
            log.info(
                "asr.transcribed",
                language=detected_language,
                segments=len(results),
                language_probability=round(info.language_probability, 3),
            )
        else:
            self._empty_batch_count += 1
            now = monotonic()
            if self._last_empty_log_ts <= 0 or now - self._last_empty_log_ts >= 5:
                log.debug(
                    "asr.transcribed_empty",
                    language=detected_language,
                    language_probability=round(info.language_probability, 3),
                    attempts=self._empty_batch_count,
                )
                self._empty_batch_count = 0
                self._last_empty_log_ts = now
            else:
                log.debug(
                    "asr.transcribed_empty",
                    language=detected_language,
                    language_probability=round(info.language_probability, 3),
                )
        return results

    async def transcribe(
        self,
        audio: AudioChunk,
        *,
        language: str | None = None,
        initial_prompt: str | None = None,
        word_timestamps: bool = False,
    ) -> list[TranscriptionSegment]:
        """Transcribe an audio chunk to text segments.

        The CPU-bound inference runs in a dedicated executor worker.
        """
        samples = audio.samples
        # Ensure mono float32 for faster-whisper
        if samples.ndim == 2:
            samples = samples.mean(axis=1)
        samples = samples.astype(np.float32)

        if samples.size == 0:
            return []

        rms = float(np.sqrt(np.mean(samples ** 2)))
        peak = float(np.max(np.abs(samples)))
        log.debug(
            "asr.audio_level",
            source=audio.source,
            rms=round(rms, 5),
            peak=round(peak, 5),
            samples=samples.size,
            sample_rate=audio.sample_rate,
        )
        if rms < 1e-5:
            log.debug("asr.skip_silence", source=audio.source, rms=rms)
            return []

        loop = asyncio.get_running_loop()
        fn = partial(
            self._transcribe_sync,
            samples,
            language=language,
            initial_prompt=initial_prompt,
            word_timestamps=word_timestamps,
        )

        executor = self._executor
        if executor is None:
            executor = ThreadPoolExecutor(max_workers=1)
            self._executor = executor
        return await loop.run_in_executor(executor, fn)

    async def transcribe_stream(
        self,
        audio_stream: AsyncIterator[AudioChunk],
        *,
        language: str | None = None,
    ) -> AsyncIterator[TranscriptionSegment]:
        """Streaming transcription: yields segments as audio arrives.

        Accumulates chunks and transcribes periodically.
        """
        async for chunk in audio_stream:
            segments = await self.transcribe(chunk, language=language)
            for seg in segments:
                yield seg
