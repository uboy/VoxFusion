"""Integration tests — verifies that each ASR engine loads and transcribes real audio.

Run with:
    pytest tests/integration/test_all_models_e2e.py -v -m integration

Each test is skipped automatically when its required packages or downloaded
model weights are not present.  To enable a test, install the relevant
package and/or pre-download the model:

    # Whisper large-v3 (faster-whisper downloads on first use):
    #   no extra setup needed — model downloads automatically on first run

    # GigaAM v3:
    #   pip install transformers torch
    #   huggingface-cli download ai-sage/GigaAM-v3

    # Breeze ASR:
    #   pip install transformers torch
    #   huggingface-cli download MediaTek-Research/Breeze-ASR-25

    # Parakeet v3:
    #   pip install "nemo_toolkit[asr]" torchaudio
    #   (model downloads automatically via NeMo on first run)
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_HF_CACHE_ENV = "HF_HOME"
_HF_GIGAAM_DIR = "models--ai-sage--GigaAM-v3"
_HF_BREEZE_DIR = "models--MediaTek-Research--Breeze-ASR-25"


def _hf_hub_root() -> Path:
    hf_home = os.environ.get(_HF_CACHE_ENV)
    if hf_home:
        return Path(hf_home) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _pkg(name: str) -> bool:
    """Return True if *name* is importable."""
    return importlib.util.find_spec(name) is not None


def _hf_model_cached(subdir: str) -> bool:
    return (_hf_hub_root() / subdir).exists()


def _make_sine(duration_s: float = 1.0, sr: int = 16000) -> np.ndarray:
    """Return a short 440 Hz sine-wave clip for smoke-testing transcription."""
    t = np.linspace(0, duration_s, int(duration_s * sr), endpoint=False, dtype=np.float32)
    return (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def _make_chunk(duration_s: float = 1.0, sr: int = 16000):
    from voxfusion.models.audio import AudioChunk

    samples = _make_sine(duration_s, sr)
    return AudioChunk(
        samples=samples,
        sample_rate=sr,
        channels=1,
        timestamp_start=0.0,
        timestamp_end=duration_s,
        source="file",
        dtype="float32",
    )


# ---------------------------------------------------------------------------
# Whisper large-v3  (faster-whisper, always available)
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.asyncio
async def test_whisper_large_v3_engine_transcribes() -> None:
    """Whisper large-v3 via faster-whisper — no extra packages needed."""
    from voxfusion.asr.faster_whisper import FasterWhisperEngine
    from voxfusion.config.models import ASRConfig
    from voxfusion.models.transcription import TranscriptionSegment

    cfg = ASRConfig(model_size="large-v3", device="cpu", compute_type="int8")
    engine = FasterWhisperEngine(cfg)
    try:
        # Loading the model downloads it on first run (~1.5 GB) — skip via env var
        skip_download = os.environ.get("VOXFUSION_SKIP_LARGE_DOWNLOAD", "").lower() in ("1", "true", "yes")
        if skip_download:
            pytest.skip("VOXFUSION_SKIP_LARGE_DOWNLOAD set — skipping large-v3 download")

        engine.load_model()
        chunk = _make_chunk()
        segments = await engine.transcribe(chunk, language="en")

        assert isinstance(segments, list)
        for seg in segments:
            assert isinstance(seg, TranscriptionSegment)
            assert isinstance(seg.text, str)
    finally:
        engine.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_whisper_small_engine_transcribes() -> None:
    """Whisper small via faster-whisper — quick smoke-test, no large download."""
    from voxfusion.asr.faster_whisper import FasterWhisperEngine
    from voxfusion.config.models import ASRConfig
    from voxfusion.models.transcription import TranscriptionSegment

    cfg = ASRConfig(model_size="small", device="cpu", compute_type="int8")
    engine = FasterWhisperEngine(cfg)
    try:
        engine.load_model()
        chunk = _make_chunk()
        segments = await engine.transcribe(chunk, language="en")

        assert isinstance(segments, list)
        for seg in segments:
            assert isinstance(seg, TranscriptionSegment)
            assert isinstance(seg.text, str)
    finally:
        engine.close()


# ---------------------------------------------------------------------------
# GigaAM v3  (requires transformers + torch + downloaded model)
# ---------------------------------------------------------------------------

_GIGAAM_AVAILABLE = (
    _pkg("transformers") and _pkg("torch")
    and _pkg("torchaudio") and _pkg("sentencepiece")
    and _pkg("omegaconf") and _pkg("hydra")
    and _hf_model_cached(_HF_GIGAAM_DIR)
)
_GIGAAM_SKIP = (
    "GigaAM v3 not ready: install transformers torch torchaudio sentencepiece "
    "omegaconf hydra-core and run: huggingface-cli download ai-sage/GigaAM-v3"
)


@pytest.mark.integration
@pytest.mark.skipif(not _GIGAAM_AVAILABLE, reason=_GIGAAM_SKIP)
@pytest.mark.asyncio
async def test_gigaam_v3_engine_transcribes() -> None:
    """GigaAM v3 — loads model and transcribes a short audio clip."""
    from voxfusion.asr.gigaam_engine import GigaAMCTCEngine
    from voxfusion.config.models import ASRConfig
    from voxfusion.models.transcription import TranscriptionSegment

    cfg = ASRConfig(model_size="gigaam-v3-e2e-ctc")
    engine = GigaAMCTCEngine(cfg)
    try:
        engine.load_model()
        chunk = _make_chunk(duration_s=2.0)
        segments = await engine.transcribe(chunk, language="ru")

        assert isinstance(segments, list)
        for seg in segments:
            assert isinstance(seg, TranscriptionSegment)
            assert isinstance(seg.text, str)
            assert seg.language == "ru"
    finally:
        engine.close()


@pytest.mark.integration
@pytest.mark.skipif(not _GIGAAM_AVAILABLE, reason=_GIGAAM_SKIP)
def test_gigaam_v3_factory_roundtrip() -> None:
    """ASR factory selects the GigaAM engine and can create it without error."""
    from voxfusion.asr.factory import create_asr_engine
    from voxfusion.asr.gigaam_engine import GigaAMCTCEngine
    from voxfusion.config.models import ASRConfig

    engine, backend = create_asr_engine(ASRConfig(model_size="gigaam-v3-e2e-ctc"))
    assert backend == "gigaam"
    assert isinstance(engine, GigaAMCTCEngine)
    engine.close()


# ---------------------------------------------------------------------------
# Breeze ASR  (requires transformers + torch + downloaded model)
# ---------------------------------------------------------------------------

_BREEZE_AVAILABLE = _pkg("transformers") and _pkg("torch") and _hf_model_cached(_HF_BREEZE_DIR)
_BREEZE_SKIP = "Breeze ASR not ready: install transformers+torch and run: huggingface-cli download MediaTek-Research/Breeze-ASR-25"


@pytest.mark.integration
@pytest.mark.skipif(not _BREEZE_AVAILABLE, reason=_BREEZE_SKIP)
@pytest.mark.asyncio
async def test_breeze_asr_engine_transcribes() -> None:
    """Breeze ASR — loads model and transcribes a short audio clip."""
    from voxfusion.asr.breeze_engine import BreezeASREngine
    from voxfusion.config.models import ASRConfig
    from voxfusion.models.transcription import TranscriptionSegment

    cfg = ASRConfig(model_size="breeze-asr")
    engine = BreezeASREngine(cfg)
    try:
        engine.load_model()
        chunk = _make_chunk(duration_s=2.0)
        segments = await engine.transcribe(chunk, language="en")

        assert isinstance(segments, list)
        for seg in segments:
            assert isinstance(seg, TranscriptionSegment)
            assert isinstance(seg.text, str)
    finally:
        engine.close()


# ---------------------------------------------------------------------------
# Parakeet v3  (requires nemo_toolkit + downloaded model)
# ---------------------------------------------------------------------------

_PARAKEET_AVAILABLE = _pkg("nemo")
_PARAKEET_SKIP = 'Parakeet not ready: pip install "nemo_toolkit[asr]" torchaudio'


@pytest.mark.integration
@pytest.mark.skipif(not _PARAKEET_AVAILABLE, reason=_PARAKEET_SKIP)
@pytest.mark.asyncio
async def test_parakeet_v3_engine_transcribes() -> None:
    """Parakeet v3 via NeMo — loads model and transcribes a short audio clip."""
    from voxfusion.asr.parakeet_engine import ParakeetASREngine
    from voxfusion.config.models import ASRConfig
    from voxfusion.models.transcription import TranscriptionSegment

    cfg = ASRConfig(model_size="parakeet-tdt-0.6b-v3")
    engine = ParakeetASREngine(cfg)
    try:
        engine.load_model()
        chunk = _make_chunk(duration_s=2.0)
        segments = await engine.transcribe(chunk)

        assert isinstance(segments, list)
        for seg in segments:
            assert isinstance(seg, TranscriptionSegment)
            assert isinstance(seg.text, str)
    finally:
        engine.close()


# ---------------------------------------------------------------------------
# GUI smoke — each model selectable in the file transcription tab
# ---------------------------------------------------------------------------

def _skip_if_no_display() -> None:
    import tkinter as tk
    try:
        root = tk.Tk()
        root.destroy()
    except Exception as exc:
        pytest.skip(f"No display available: {exc}")


@pytest.mark.integration
@pytest.mark.parametrize("model_id", [
    "small",
    "large-v3",
    pytest.param(
        "gigaam-v3-e2e-ctc",
        marks=pytest.mark.skipif(not _pkg("transformers"), reason="transformers not installed"),
    ),
    pytest.param(
        "breeze-asr",
        marks=pytest.mark.skipif(not _pkg("transformers"), reason="transformers not installed"),
    ),
    pytest.param(
        "parakeet-tdt-0.6b-v3",
        marks=pytest.mark.skipif(not _pkg("nemo"), reason="nemo not installed"),
    ),
])
def test_gui_can_select_model_in_file_tab(model_id: str) -> None:
    """The GUI builds without error and the file-tab model combo accepts each model."""
    _skip_if_no_display()
    import tkinter as tk
    from voxfusion.gui.main import CaptureOptions, TranscriptionGUI

    root = tk.Tk()
    root.withdraw()
    try:
        gui = TranscriptionGUI(
            root,
            CaptureOptions(
                model="small",
                language="ru",
                translate=None,
                microphone_device_id=None,
                system_device_id=None,
            ),
        )
        root.update_idletasks()
        gui._file_model_var.set(model_id)
        gui._on_file_model_changed()
        root.update_idletasks()
        assert gui._file_model_var.get() == model_id
        gui._restore_redirection()
    finally:
        root.destroy()


@pytest.mark.integration
def test_gui_file_tab_defaults_to_gigaam_when_available() -> None:
    """On first launch the file-transcription tab should default to GigaAM v3."""
    _skip_if_no_display()
    import tkinter as tk
    from voxfusion.gui.main import CaptureOptions, TranscriptionGUI
    from voxfusion.asr_catalog import get_available_model_catalog

    available = {m.id for m in get_available_model_catalog()}
    if "gigaam-v3-e2e-ctc" not in available:
        pytest.skip("GigaAM v3 not available in this build")

    root = tk.Tk()
    root.withdraw()
    try:
        gui = TranscriptionGUI(
            root,
            CaptureOptions(
                model="small",
                language="ru",
                translate=None,
                microphone_device_id=None,
                system_device_id=None,
            ),
        )
        root.update_idletasks()
        assert gui._file_model_var.get() == "gigaam-v3-e2e-ctc", (
            f"Expected file-tab default to be 'gigaam-v3-e2e-ctc', got '{gui._file_model_var.get()}'"
        )
        gui._restore_redirection()
    finally:
        root.destroy()
