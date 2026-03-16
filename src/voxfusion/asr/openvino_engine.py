"""OpenVINO-accelerated Whisper ASR engine via optimum-intel.

On first use the model is exported to OpenVINO IR format and cached
under ``~/.voxfusion/openvino/<model_size>/``.  Subsequent loads are
fast (load from disk).  Requires ``optimum-intel[openvino]`` to be
installed::

    pip install optimum-intel[openvino]

If Intel Iris Xe / Arc GPU is available it is used automatically;
otherwise falls back to CPU (which is still faster than plain
CTranslate2 due to OpenVINO-optimised kernels).
"""

import asyncio
import os
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

# Map VoxFusion model-size names to HuggingFace repo IDs
_HF_MODEL_MAP: dict[str, str] = {
    "tiny":     "openai/whisper-tiny",
    "base":     "openai/whisper-base",
    "small":    "openai/whisper-small",
    "medium":   "openai/whisper-medium",
    "large-v2": "openai/whisper-large-v2",
    "large-v3": "openai/whisper-large-v3",
}

_HALLUCINATION_PATTERNS: frozenset[str] = frozenset({
    "продолжение следует",
    "субтитр",
    "thank you for watching",
    "subscribe",
    "like and subscribe",
    "www.",
    "http",
})


def _is_hallucination(text: str) -> bool:
    t = text.strip().lower()
    if len(t) < 2:
        return True
    return any(p in t for p in _HALLUCINATION_PATTERNS)


def is_openvino_available() -> bool:
    """Return True if optimum-intel with OpenVINO backend is importable."""
    try:
        from optimum.intel.openvino import OVModelForSpeechSeq2Seq  # noqa: F401
        return True
    except ImportError:
        return False


def _detect_ov_device() -> str:
    """Return 'GPU' if Intel GPU is available for OpenVINO, else 'CPU'."""
    try:
        from openvino import Core
        devices = Core().available_devices
        log.info("openvino.available_devices", devices=devices)
        if "GPU" in devices:
            return "GPU"
    except Exception as exc:
        log.debug("openvino.device_probe_failed", error=str(exc))
    return "CPU"


class OpenVINOWhisperEngine:
    """Whisper ASR engine backed by Intel OpenVINO via optimum-intel.

    Implements the same async interface as ``FasterWhisperEngine`` so it
    can be used as a drop-in replacement.
    """

    def __init__(self, config: ASRConfig | None = None) -> None:
        self._config = config or ASRConfig()
        self._pipeline: object | None = None
        self._executor: ThreadPoolExecutor | None = ThreadPoolExecutor(max_workers=1)
        self._ov_device: str = "CPU"

    @property
    def model_name(self) -> str:
        return f"openvino/{self._config.model_size}"

    def _cache_dir(self) -> Path:
        return Path(os.path.expanduser("~/.voxfusion/openvino")) / self._config.model_size

    def load_model(self) -> None:
        """Load (or export+cache) the OpenVINO Whisper model."""
        if self._pipeline is not None:
            return

        try:
            from optimum.intel.openvino import OVModelForSpeechSeq2Seq
            from transformers import AutoProcessor
            from transformers import pipeline as hf_pipeline
        except ImportError as exc:
            raise ModelLoadError(
                "optimum-intel is not installed. "
                "Run: pip install optimum-intel[openvino]"
            ) from exc

        hf_name = _HF_MODEL_MAP.get(self._config.model_size)
        if hf_name is None:
            raise ModelLoadError(
                f"Unknown model size {self._config.model_size!r}. "
                f"Valid sizes: {list(_HF_MODEL_MAP)}"
            )

        self._ov_device = _detect_ov_device()
        cache_dir = self._cache_dir()

        if cache_dir.exists() and any(cache_dir.iterdir()):
            log.info(
                "openvino.loading_cached",
                model=self._config.model_size,
                path=str(cache_dir),
                device=self._ov_device,
            )
            try:
                model = OVModelForSpeechSeq2Seq.from_pretrained(
                    str(cache_dir),
                    compile=True,
                    device=self._ov_device,
                )
            except Exception as exc:
                log.warning(
                    "openvino.cache_load_failed",
                    error=str(exc),
                    hint="Will re-export model",
                )
                # Remove broken cache and retry via export path
                import shutil
                shutil.rmtree(cache_dir, ignore_errors=True)
                return self.load_model()
        else:
            log.info(
                "openvino.exporting",
                model=hf_name,
                cache=str(cache_dir),
                device=self._ov_device,
                hint="First run: downloading + converting model (may take several minutes)",
            )
            try:
                model = OVModelForSpeechSeq2Seq.from_pretrained(
                    hf_name,
                    export=True,
                    compile=True,
                    device=self._ov_device,
                )
            except Exception as exc:
                raise ModelLoadError(
                    f"Failed to export {hf_name!r} to OpenVINO format: {exc}. "
                    "Make sure 'torch' and 'optimum-intel[openvino]' are installed."
                ) from exc
            cache_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(cache_dir))
            log.info("openvino.export_done", cache=str(cache_dir))

        # Load processor (tokenizer + feature extractor)
        try:
            processor = AutoProcessor.from_pretrained(str(cache_dir))
        except Exception:
            processor = AutoProcessor.from_pretrained(hf_name)

        generate_kwargs: dict = {}
        if self._config.language:
            generate_kwargs["language"] = self._config.language
            generate_kwargs["task"] = "transcribe"

        self._pipeline = hf_pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            return_timestamps=True,
            generate_kwargs=generate_kwargs,
        )
        log.info(
            "openvino.model_loaded",
            model=self._config.model_size,
            device=self._ov_device,
        )

    def unload_model(self) -> None:
        """Release the model from memory."""
        self._pipeline = None
        log.info("openvino.model_unloaded")

    def close(self) -> None:
        """Release background executor."""
        if self._executor is None:
            return
        self._executor.shutdown(wait=False, cancel_futures=True)
        self._executor = None

    def __del__(self) -> None:
        with suppress(Exception):
            self.close()

    def _transcribe_sync(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> list[TranscriptionSegment]:
        """Synchronous transcription — runs inside the executor thread."""
        if self._pipeline is None:
            raise TranscriptionError("Model is not loaded")

        try:
            result = self._pipeline(  # type: ignore[operator]
                {"raw": audio, "sampling_rate": sample_rate},
                chunk_length_s=30,
                stride_length_s=5,
            )
        except Exception as exc:
            raise TranscriptionError(f"OpenVINO transcription failed: {exc}") from exc

        raw_chunks: list[dict] = result.get("chunks") or []  # type: ignore[union-attr]
        if not raw_chunks and result.get("text"):  # type: ignore[union-attr]
            raw_chunks = [{"timestamp": (0.0, None), "text": result["text"]}]  # type: ignore[index]

        segments: list[TranscriptionSegment] = []
        for chunk in raw_chunks:
            text = chunk.get("text", "").strip()
            if not text or _is_hallucination(text):
                continue
            ts = chunk.get("timestamp") or (0.0, None)
            t_start = float(ts[0] or 0.0)
            t_end = float(ts[1]) if ts[1] is not None else t_start + 1.0
            segments.append(
                TranscriptionSegment(
                    text=text,
                    language=self._config.language or "auto",
                    start_time=t_start,
                    end_time=t_end,
                    confidence=-0.5,  # OpenVINO pipeline doesn't expose log-prob
                    words=None,
                    no_speech_prob=0.0,
                )
            )

        if segments:
            log.info("openvino.transcribed", segments=len(segments))
        else:
            log.debug("openvino.transcribed_empty")

        return segments

    async def transcribe(
        self,
        audio: AudioChunk,
        *,
        language: str | None = None,
        initial_prompt: str | None = None,
        word_timestamps: bool = False,
    ) -> list[TranscriptionSegment]:
        """Transcribe an audio chunk asynchronously."""
        samples = audio.samples
        if samples.ndim == 2:
            samples = samples.mean(axis=1)
        samples = samples.astype(np.float32)

        if samples.size == 0:
            return []

        rms = float(np.sqrt(np.mean(samples ** 2)))
        if rms < 1e-5:
            log.debug("openvino.skip_silence", source=audio.source, rms=rms)
            return []

        loop = asyncio.get_running_loop()
        fn = partial(self._transcribe_sync, samples, audio.sample_rate)
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
        """Streaming transcription: yield segments as audio arrives."""
        async for chunk in audio_stream:
            segments = await self.transcribe(chunk, language=language)
            for seg in segments:
                yield seg
