"""Unit tests for the faster-whisper ASR engine.

All tests mock ``faster_whisper.WhisperModel`` so they run without a downloaded
model or GPU.
"""

from __future__ import annotations

import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from voxfusion.asr.faster_whisper import FasterWhisperEngine, _is_hallucination, _resolve_device
from voxfusion.config.models import ASRConfig
from voxfusion.exceptions import ModelLoadError, TranscriptionError
from voxfusion.models.audio import AudioChunk

# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------


def _make_engine(model_size: str = "tiny", device: str = "cpu") -> FasterWhisperEngine:
    cfg = ASRConfig(model_size=model_size, device=device, compute_type="int8")
    return FasterWhisperEngine(cfg)


def _audio_chunk(samples: int = 16000, sr: int = 16000) -> AudioChunk:
    # Use a sine wave so RMS > 1e-5 and the engine doesn't skip as silence.
    t = np.linspace(0, samples / sr, samples, dtype=np.float32)
    data = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
    return AudioChunk(
        samples=data,
        sample_rate=sr,
        channels=1,
        timestamp_start=0.0,
        timestamp_end=samples / sr,
        source="test",
    )


def _fake_segment(text: str, start: float = 0.0, end: float = 1.0) -> MagicMock:
    seg = MagicMock()
    seg.text = text
    seg.start = start
    seg.end = end
    seg.no_speech_prob = 0.0
    seg.words = None
    return seg


def _make_fake_whisper_model(segments: list[MagicMock]) -> MagicMock:
    model = MagicMock()
    info = MagicMock()
    info.language = "en"
    info.language_probability = 0.99
    model.transcribe.return_value = (iter(segments), info)
    return model


# ---------------------------------------------------------------------------
# Hallucination filter
# ---------------------------------------------------------------------------


def test_is_hallucination_short_text() -> None:
    assert _is_hallucination("") is True
    assert _is_hallucination(" ") is True


def test_is_hallucination_common_patterns() -> None:
    assert _is_hallucination("thank you for watching") is True
    assert _is_hallucination("Subscribe to our channel") is True
    assert _is_hallucination("www.example.com") is True


def test_is_hallucination_clean_text() -> None:
    assert _is_hallucination("Hello, how are you?") is False
    assert _is_hallucination("The meeting starts at noon.") is False


# ---------------------------------------------------------------------------
# Device resolution
# ---------------------------------------------------------------------------


def test_resolve_device_cpu_passthrough() -> None:
    device, compute = _resolve_device("cpu")
    assert device == "cpu"
    assert compute == "int8"


def test_resolve_device_auto_without_cuda() -> None:
    """When ctranslate2 doesn't report CUDA support, auto → cpu."""
    fake_ct2 = types.ModuleType("ctranslate2")
    fake_ct2.get_supported_compute_types = lambda _dev: []  # type: ignore[attr-defined]
    with patch.dict("sys.modules", {"ctranslate2": fake_ct2}):
        device, _compute = _resolve_device("auto")
    assert device == "cpu"


# ---------------------------------------------------------------------------
# Model lifecycle
# ---------------------------------------------------------------------------


def test_load_model_raises_on_missing_package() -> None:
    engine = _make_engine()
    with patch.dict("sys.modules", {"faster_whisper": None}):
        with pytest.raises(ModelLoadError):
            engine.load_model()


def test_load_model_success() -> None:
    engine = _make_engine()
    fake_model = _make_fake_whisper_model([])
    fake_fw = types.ModuleType("faster_whisper")
    fake_fw.WhisperModel = MagicMock(return_value=fake_model)  # type: ignore[attr-defined]
    with patch.dict("sys.modules", {"faster_whisper": fake_fw}):
        engine.load_model()
    assert engine._model is fake_model


def test_unload_model_clears_reference() -> None:
    engine = _make_engine()
    engine._model = MagicMock()
    engine.unload_model()
    assert engine._model is None


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------


def _run_transcribe_sync(engine: FasterWhisperEngine) -> list:
    """Run engine.transcribe() synchronously by executing the _transcribe_sync directly."""
    return engine._transcribe_sync(
        _audio_chunk().samples,
        language=None,
        initial_prompt=None,
        word_timestamps=False,
    )


def test_transcribe_sync_returns_segments() -> None:
    engine = _make_engine()
    engine._model = _make_fake_whisper_model(
        [_fake_segment("Hello world", 0.0, 1.5), _fake_segment("How are you?", 1.5, 3.0)]
    )
    segments = _run_transcribe_sync(engine)
    assert len(segments) == 2
    assert segments[0].text == "Hello world"
    assert segments[1].text == "How are you?"


def test_transcribe_sync_filters_hallucinations() -> None:
    engine = _make_engine()
    engine._model = _make_fake_whisper_model(
        [_fake_segment("thank you for watching"), _fake_segment("Real speech here.")]
    )
    segments = _run_transcribe_sync(engine)
    texts = [s.text for s in segments]
    assert "thank you for watching" not in texts
    assert "Real speech here." in texts


def test_transcribe_sync_raises_transcription_error_on_model_failure() -> None:
    engine = _make_engine()
    bad_model = MagicMock()
    bad_model.transcribe.side_effect = RuntimeError("model crashed")
    engine._model = bad_model

    with pytest.raises(TranscriptionError):
        _run_transcribe_sync(engine)


@pytest.mark.asyncio
async def test_transcribe_skips_silent_audio() -> None:
    """Engine must return [] without calling model when audio is silent."""
    engine = _make_engine()
    engine._model = MagicMock()
    silent = AudioChunk(
        samples=np.zeros(16000, dtype=np.float32),
        sample_rate=16000,
        channels=1,
        timestamp_start=0.0,
        timestamp_end=1.0,
        source="test",
    )
    result = await engine.transcribe(silent)
    assert result == []
    engine._model.transcribe.assert_not_called()
