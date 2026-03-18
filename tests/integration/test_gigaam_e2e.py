"""Integration test: real GigaAM transcription end-to-end.

This test requires a locally available GigaAM model.  It is skipped
automatically when the model is not present.

To enable the test, either:
  1. Set VOXFUSION_ASR__MODEL_PATH to a local directory that contains
     the downloaded model.
  2. Pre-download the model via:
       huggingface-cli download ai-sage/GigaAM-v3 --revision ctc
     so that it lands in the Hugging Face cache.


The test verifies that:
  - the engine loads without error,
  - transcribing a short synthetic sine-wave clip returns a list (may be
    empty for pure noise — that is acceptable),
  - the returned segments have the expected schema.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Availability guard
# ---------------------------------------------------------------------------

_MODEL_PATH_ENV = "VOXFUSION_ASR__MODEL_PATH"
# HuggingFace hub stores the model under "models--<org>--<repo>" with "--" separating path segments.
_HF_GIGAAM_DIR = "models--ai-sage--GigaAM-v3"


def _hf_cache_root() -> Path:
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _gigaam_available() -> bool:
    explicit = os.environ.get(_MODEL_PATH_ENV, "")
    if explicit and Path(explicit).exists():
        return True
    # Check whether the HuggingFace cache already has the model
    return (_hf_cache_root() / _HF_GIGAAM_DIR).exists()


_SKIP_REASON = (
    "GigaAM model not available locally. "
    f"Set {_MODEL_PATH_ENV} to a local model directory, "
    "or run: voxfusion models download --asr gigaam-v3-e2e-ctc"
)


@pytest.mark.integration
@pytest.mark.skipif(not _gigaam_available(), reason=_SKIP_REASON)
@pytest.mark.asyncio
async def test_gigaam_transcribes_real_audio() -> None:
    """Load the real GigaAM model and transcribe a short synthetic clip."""
    from voxfusion.asr.gigaam_engine import GigaAMCTCEngine
    from voxfusion.config.models import ASRConfig
    from voxfusion.models.audio import AudioChunk
    from voxfusion.models.transcription import TranscriptionSegment

    explicit_path = os.environ.get(_MODEL_PATH_ENV, "")
    cfg = ASRConfig(
        model_size="gigaam-v3-e2e-ctc",
        model_path=explicit_path if explicit_path else None,
    )
    engine = GigaAMCTCEngine(cfg)
    try:
        engine.load_model()

        # 1-second 440 Hz sine wave at 16 kHz
        t = np.linspace(0, 1, 16000, endpoint=False, dtype=np.float32)
        samples = (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

        chunk = AudioChunk(
            samples=samples,
            sample_rate=16000,
            channels=1,
            timestamp_start=0.0,
            timestamp_end=1.0,
            source="file",
            dtype="float32",
        )

        segments = await engine.transcribe(chunk, language="ru")

        assert isinstance(segments, list)
        for seg in segments:
            assert isinstance(seg, TranscriptionSegment)
            assert isinstance(seg.text, str)
            assert seg.language == "ru"
    finally:
        engine.close()


@pytest.mark.integration
@pytest.mark.skipif(not _gigaam_available(), reason=_SKIP_REASON)
def test_gigaam_stream_raises() -> None:
    """Streaming transcription must raise for GigaAM (batch-only backend)."""
    import asyncio

    from voxfusion.asr.gigaam_engine import GigaAMCTCEngine
    from voxfusion.config.models import ASRConfig
    from voxfusion.exceptions import TranscriptionError

    cfg = ASRConfig(model_size="gigaam-v3-e2e-ctc")
    engine = GigaAMCTCEngine(cfg)
    try:

        async def _drain() -> None:
            async for _ in engine.transcribe_stream(
                aiter([]),  # type: ignore[name-defined]
            ):
                pass

        with pytest.raises(TranscriptionError):
            asyncio.run(_drain())
    finally:
        engine.close()
