"""ML-based speaker diarization using pyannote.audio.

Requires the ``pyannote.audio`` package and a Hugging Face auth token
for model download.  The diarization pipeline identifies speaker turns
which are then aligned with ASR segments.
"""

import asyncio
import inspect
import os
import warnings
from collections.abc import AsyncIterator
from contextlib import suppress
from dataclasses import replace

import numpy as np

from voxfusion.config.models import DiarizationMLConfig
from voxfusion.diarization.alignment import SpeakerTurn, align_segments
from voxfusion.diarization.types import DiarizationTurnResult
from voxfusion.exceptions import DiarizationError
from voxfusion.logging import get_logger
from voxfusion.models.audio import AudioChunk
from voxfusion.models.diarization import DiarizedSegment
from voxfusion.models.transcription import TranscriptionSegment
from voxfusion.runtime_torchscript import (
    should_use_torchscript_source_fallback,
    temporary_torchscript_source_fallback,
)

log = get_logger(__name__)


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


# Disable pyannote telemetry at module import time (before Pipeline.from_pretrained).
_disable_pyannote_telemetry()


def _pipeline_auth_kwargs(
    from_pretrained: object,
    token: str,
) -> dict[str, str]:
    """Return auth kwargs compatible with the installed pyannote version."""
    try:
        signature = inspect.signature(from_pretrained)
    except (TypeError, ValueError):
        return {"token": token}

    parameters = signature.parameters
    if "token" in parameters:
        return {"token": token}
    if "use_auth_token" in parameters:
        return {"use_auth_token": token}
    return {"token": token}


def _extract_annotation(diarization_output: object) -> object:
    """Return the pyannote annotation object across API variants."""
    if hasattr(diarization_output, "itertracks"):
        return diarization_output

    speaker_diarization = getattr(diarization_output, "speaker_diarization", None)
    if speaker_diarization is not None and hasattr(speaker_diarization, "itertracks"):
        return speaker_diarization

    raise DiarizationError("Unsupported pyannote diarization output: missing speaker annotation")


def _extract_exclusive_annotation(diarization_output: object) -> object | None:
    """Return the optional exclusive speaker diarization annotation."""
    annotation = getattr(diarization_output, "exclusive_speaker_diarization", None)
    if annotation is not None and hasattr(annotation, "itertracks"):
        return annotation
    return None


def _speaker_count_kwargs(
    diarize_callable: object,
    config: DiarizationMLConfig,
) -> dict[str, int]:
    """Return speaker-count kwargs compatible with the installed pyannote version."""
    exact_count = config.min_speakers
    if (
        exact_count is not None
        and config.max_speakers is not None
        and exact_count == config.max_speakers
    ):
        try:
            signature = inspect.signature(diarize_callable)
        except (TypeError, ValueError):
            return {"num_speakers": exact_count}

        parameters = signature.parameters
        if "num_speakers" in parameters:
            return {"num_speakers": exact_count}
        if "min_speakers" in parameters or "max_speakers" in parameters:
            return {
                "min_speakers": exact_count,
                "max_speakers": exact_count,
            }
        return {"num_speakers": exact_count}

    kwargs: dict[str, int] = {}
    if config.min_speakers is not None:
        kwargs["min_speakers"] = config.min_speakers
    if config.max_speakers is not None:
        kwargs["max_speakers"] = config.max_speakers
    return kwargs


def _speaker_count_hint_applied(config: DiarizationMLConfig) -> str:
    if (
        config.min_speakers is not None
        and config.max_speakers is not None
        and config.min_speakers == config.max_speakers
    ):
        return "exact"
    if config.min_speakers is not None or config.max_speakers is not None:
        return "bounded"
    return "auto"


def _annotation_to_turns(
    annotation: object,
    *,
    min_segment_duration: float,
) -> list[SpeakerTurn]:
    turns: list[SpeakerTurn] = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        duration = getattr(turn, "duration", max(0.0, turn.end - turn.start))
        if duration < min_segment_duration:
            continue
        turns.append(
            SpeakerTurn(
                speaker_id=speaker,
                start_time=turn.start,
                end_time=turn.end,
            )
        )
    return turns


def _shift_turns(turns: list[SpeakerTurn], offset_s: float) -> list[SpeakerTurn]:
    if not offset_s:
        return turns
    return [
        SpeakerTurn(
            speaker_id=turn.speaker_id,
            start_time=turn.start_time + offset_s,
            end_time=turn.end_time + offset_s,
        )
        for turn in turns
    ]


class PyAnnoteDiarizer:
    """Speaker diarization using pyannote.audio.

    The pipeline model is lazily loaded on first ``diarize`` call.
    Inference is offloaded to a thread executor since it is CPU/GPU-bound.
    """

    def __init__(
        self,
        config: DiarizationMLConfig | None = None,
        *,
        emit_pipeline_logs: bool = True,
    ) -> None:
        self._config = config or DiarizationMLConfig()
        self._pipeline: object | None = None
        self._emit_pipeline_logs = emit_pipeline_logs

    def _load_pipeline(self) -> object:
        """Load the pyannote diarization pipeline."""
        if self._pipeline is not None:
            return self._pipeline

        warnings.filterwarnings(
            "ignore",
            message=r"\s*torchcodec is not installed correctly so built-in audio decoding will fail\..*",
            category=UserWarning,
            module=r"pyannote\.audio\.core\.io",
        )
        try:
            from pyannote.audio import Pipeline
        except ImportError as exc:
            raise DiarizationError(
                "pyannote.audio is not installed. Install with: pip install pyannote.audio"
            ) from exc

        token = (
            self._config.hf_auth_token
            or os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        )
        if not token:
            raise DiarizationError(
                "Hugging Face auth token required for pyannote models. "
                "Set VOXFUSION_DIARIZATION__ML__HF_AUTH_TOKEN or HF_TOKEN"
            )

        if self._emit_pipeline_logs:
            log.info("pyannote.loading_pipeline", model=self._config.model)
        try:
            auth_kwargs = _pipeline_auth_kwargs(Pipeline.from_pretrained, token)
            try:
                import torch
            except ImportError:
                torch = None

            if torch is None:
                self._pipeline = Pipeline.from_pretrained(
                    self._config.model,
                    **auth_kwargs,
                )
            elif not should_use_torchscript_source_fallback(torch):
                self._pipeline = Pipeline.from_pretrained(
                    self._config.model,
                    **auth_kwargs,
                )
            else:
                with temporary_torchscript_source_fallback(torch):
                    self._pipeline = Pipeline.from_pretrained(
                        self._config.model,
                        **auth_kwargs,
                    )
        except Exception as exc:
            raise DiarizationError(f"Failed to load pyannote pipeline: {exc}") from exc

        if self._config.device in {"auto", "cuda"}:
            try:
                import torch

                if torch.cuda.is_available():
                    self._pipeline.to(torch.device("cuda"))  # type: ignore[union-attr]
                    if self._emit_pipeline_logs:
                        log.info("pyannote.using_gpu")
            except ImportError:
                pass

        if self._emit_pipeline_logs:
            log.info("pyannote.pipeline_loaded")
        return self._pipeline

    def _diarize_result_sync(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> DiarizationTurnResult:
        """Run diarization synchronously and return the richer turn result."""
        pipeline = self._load_pipeline()

        import torch

        waveform = torch.from_numpy(audio).float().unsqueeze(0)
        input_data = {"waveform": waveform, "sample_rate": sample_rate}
        kwargs = _speaker_count_kwargs(pipeline, self._config)

        diarization_output = pipeline(input_data, **kwargs)  # type: ignore[operator]
        annotation = _extract_annotation(diarization_output)
        turns = _annotation_to_turns(
            annotation,
            min_segment_duration=self._config.min_segment_duration,
        )

        exclusive_annotation = _extract_exclusive_annotation(diarization_output)
        exclusive_turns = None
        if exclusive_annotation is not None:
            exclusive_turns = _annotation_to_turns(
                exclusive_annotation,
                min_segment_duration=self._config.min_segment_duration,
            )

        alignment_turns = exclusive_turns or turns
        speaker_count_estimate = len({turn.speaker_id for turn in alignment_turns}) or None
        result = DiarizationTurnResult(
            turns=turns,
            exclusive_turns=exclusive_turns,
            speaker_count_hint_applied=_speaker_count_hint_applied(self._config),
            speaker_count_estimate=speaker_count_estimate,
            model_id=self._config.model,
            is_chunk_local=False,
        )

        log.info(
            "pyannote.diarized",
            turns=len(turns),
            exclusive_turns=(len(exclusive_turns) if exclusive_turns is not None else None),
            speaker_count_estimate=speaker_count_estimate,
        )
        return result

    async def diarize_turns_result(self, audio: AudioChunk) -> DiarizationTurnResult:
        """Return raw speaker turns plus optional exclusive turns in absolute time."""
        samples = audio.samples
        if samples.ndim == 2:
            samples = samples.mean(axis=1)

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, self._diarize_result_sync, samples, audio.sample_rate
        )
        if not audio.timestamp_start:
            return result
        return replace(
            result,
            turns=_shift_turns(result.turns, audio.timestamp_start),
            exclusive_turns=(
                _shift_turns(result.exclusive_turns, audio.timestamp_start)
                if result.exclusive_turns is not None
                else None
            ),
        )

    async def diarize_turns(self, audio: AudioChunk) -> list[SpeakerTurn]:
        """Return raw speaker turns in absolute audio coordinates."""
        result = await self.diarize_turns_result(audio)
        return result.turns

    async def diarize(
        self,
        segments: list[TranscriptionSegment],
        audio: AudioChunk | None = None,
    ) -> list[DiarizedSegment]:
        """Diarize segments using pyannote ML pipeline.

        Args:
            segments: ASR transcription segments to assign speakers to.
            audio: The audio chunk (required for ML diarization).

        Returns:
            Diarized segments with ML-assigned speaker labels.
        """
        if audio is None:
            raise DiarizationError("PyAnnoteDiarizer requires audio data")

        turn_result = await self.diarize_turns_result(audio)
        return align_segments(
            segments,
            turn_result.alignment_turns(),
            speaker_source="ml",
        )

    async def diarize_stream(
        self,
        segment_stream: AsyncIterator[tuple[TranscriptionSegment, AudioChunk]],
    ) -> AsyncIterator[DiarizedSegment]:
        """Streaming diarization — processes each chunk independently."""
        async for seg, audio in segment_stream:
            result = await self.diarize([seg], audio)
            for d in result:
                yield d
