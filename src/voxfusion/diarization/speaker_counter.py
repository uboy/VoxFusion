"""Fast speaker count estimation from an audio sample.

Runs pyannote.audio on a short centre-sampled slice of the audio to quickly
estimate how many speakers are present without performing full transcription.
This is intended for the GUI "Detect Speakers" button where the user wants a
fast count before committing to a full ML-diarization run.
"""

from __future__ import annotations

import asyncio
import warnings

import numpy as np

from voxfusion.diarization.pyannote_engine import _pipeline_auth_kwargs
from voxfusion.logging import get_logger
from voxfusion.models.audio import AudioChunk
from voxfusion.runtime_torchscript import should_use_torchscript_source_fallback
from voxfusion.runtime_torchscript import temporary_torchscript_source_fallback

log = get_logger(__name__)

# Default maximum sample length fed to pyannote for counting.
# Shorter = faster estimation; 120 s is usually sufficient to find all speakers
# in a typical meeting/interview recording.
_DEFAULT_SAMPLE_DURATION_S = 120.0


def _is_pyannote_available() -> bool:
    """Return True if pyannote.audio can be imported."""
    import importlib.util

    return importlib.util.find_spec("pyannote.audio") is not None


def _count_sync(
    samples: np.ndarray,
    sample_rate: int,
    hf_token: str,
    model: str,
) -> int:
    """Synchronous speaker counting — intended to run in an executor."""
    warnings.filterwarnings("ignore", category=UserWarning)
    try:
        import torch
        from pyannote.audio import Pipeline
    except ImportError:
        log.warning("speaker_counter.pyannote_unavailable")
        return 0

    try:
        auth_kwargs = _pipeline_auth_kwargs(Pipeline.from_pretrained, hf_token)
        if not should_use_torchscript_source_fallback(torch):
            pipeline = Pipeline.from_pretrained(model, **auth_kwargs)
        else:
            with temporary_torchscript_source_fallback(torch):
                pipeline = Pipeline.from_pretrained(model, **auth_kwargs)
    except Exception as exc:
        log.warning("speaker_counter.pipeline_load_failed", error=str(exc))
        return 0

    # Move to GPU if available
    try:
        if torch.cuda.is_available():
            pipeline.to(torch.device("cuda"))
    except Exception:
        pass

    waveform = torch.from_numpy(samples).float().unsqueeze(0)
    input_data = {"waveform": waveform, "sample_rate": sample_rate}

    try:
        diarization = pipeline(input_data)
    except Exception as exc:
        log.warning("speaker_counter.inference_failed", error=str(exc))
        return 0

    annotation = diarization
    if hasattr(diarization, "speaker_diarization"):
        annotation = diarization.speaker_diarization

    speakers: set[str] = set()
    try:
        for _turn, _, speaker in annotation.itertracks(yield_label=True):
            speakers.add(speaker)
    except Exception as exc:
        log.warning("speaker_counter.parse_failed", error=str(exc))
        return 0

    count = len(speakers)
    log.info("speaker_counter.done", count=count)
    return count


async def estimate_speaker_count(
    audio: AudioChunk,
    *,
    hf_token: str,
    model: str = "pyannote/speaker-diarization-3.1",
    sample_duration_s: float = _DEFAULT_SAMPLE_DURATION_S,
) -> int:
    """Quickly estimate the number of speakers in *audio*.

    Samples at most *sample_duration_s* seconds from the centre of the audio
    and runs pyannote.audio on it.  Returns 0 if detection fails or pyannote
    is unavailable.

    Args:
        audio: Full audio to analyse.
        hf_token: HuggingFace auth token required by pyannote gated models.
        model: pyannote pipeline model identifier.
        sample_duration_s: Maximum sample length in seconds.

    Returns:
        Estimated speaker count (>= 1 on success, 0 on failure).
    """
    if not _is_pyannote_available():
        log.warning("speaker_counter.pyannote_not_installed")
        return 0

    samples = audio.samples
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    samples = np.ascontiguousarray(samples, dtype=np.float32)

    # Sample from the centre of the audio for more representative coverage
    if audio.duration > sample_duration_s:
        sr = audio.sample_rate
        start_s = (audio.duration - sample_duration_s) / 2.0
        start_idx = int(start_s * sr)
        end_idx = start_idx + int(sample_duration_s * sr)
        samples = samples[start_idx:end_idx]

    log.info(
        "speaker_counter.start",
        sample_duration_s=round(len(samples) / audio.sample_rate, 2),
        original_duration_s=round(audio.duration, 2),
    )

    loop = asyncio.get_running_loop()
    try:
        count = await loop.run_in_executor(
            None, _count_sync, samples, audio.sample_rate, hf_token, model
        )
    except Exception as exc:
        log.warning("speaker_counter.executor_error", error=str(exc))
        return 0

    return count
