"""Real end-to-end tests — recording, model download, and transcription.

All tests use real inference (no mocks).  Tests are automatically skipped
when the required package or cached model is not present.

Run groups:
    # Fast — cached models only, no downloads
    pytest tests/integration/test_realworld_e2e.py -v

    # Including downloads (slow, requires internet)
    pytest tests/integration/test_realworld_e2e.py -v -m download

    # Everything available
    pytest tests/integration/ -v
"""

from __future__ import annotations

import asyncio
import importlib.util
import os
import time
from collections.abc import AsyncIterator
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

# ---------------------------------------------------------------------------
# Helpers & shared fixtures
# ---------------------------------------------------------------------------

SR = 16_000  # all engines expect 16 kHz


def _pkg(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _hf_cache(subdir: str) -> bool:
    """Return True when a HuggingFace model directory exists in the cache."""
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        root = Path(hf_home) / "hub"
    else:
        root = Path.home() / ".cache" / "huggingface" / "hub"
    return (root / subdir).exists()


def _sine_wav(path: Path, duration_s: float = 3.0, freq: float = 440.0) -> Path:
    """Write a mono 16-kHz sine-wave WAV to *path* and return it."""
    t = np.linspace(0, duration_s, int(SR * duration_s), endpoint=False, dtype=np.float32)
    samples = (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    sf.write(str(path), samples, SR, subtype="PCM_16")
    return path


def _audio_chunk(samples: np.ndarray, sr: int = SR):
    from voxfusion.models.audio import AudioChunk

    duration = len(samples) / sr
    return AudioChunk(
        samples=samples,
        sample_rate=sr,
        channels=1,
        timestamp_start=0.0,
        timestamp_end=duration,
        source="file",
        dtype="float32",
    )


@pytest.fixture(scope="session")
def test_wav(tmp_path_factory) -> Path:
    """Session-scoped WAV file used by all transcription tests."""
    path = tmp_path_factory.mktemp("audio") / "test_speech.wav"
    _sine_wav(path, duration_s=3.0)
    assert path.exists() and path.stat().st_size > 0
    return path


# ---------------------------------------------------------------------------
# 1. RECORDING  — end-to-end pipeline without real hardware
# ---------------------------------------------------------------------------

class _FakeSource:
    """Emits a finite list of AudioChunks with sequential timestamps, then stops."""

    def __init__(self, chunks: list) -> None:
        self._active = False
        # Re-stamp chunks sequentially so the mixer places them end-to-end.
        restamped = []
        t = 0.0
        for c in chunks:
            dur = len(c.samples) / c.sample_rate
            from voxfusion.models.audio import AudioChunk as _AC
            restamped.append(
                _AC(
                    samples=c.samples,
                    sample_rate=c.sample_rate,
                    channels=c.channels,
                    timestamp_start=t,
                    timestamp_end=t + dur,
                    source=c.source,
                    dtype=c.dtype,
                )
            )
            t += dur
        self._chunks = restamped

    @property
    def device_name(self) -> str:
        return "fake-mic"

    @property
    def sample_rate(self) -> int:
        return self._chunks[0].sample_rate if self._chunks else SR

    @property
    def channels(self) -> int:
        return 1

    @property
    def is_active(self) -> bool:
        return self._active

    async def start(self) -> None:
        self._active = True

    async def stop(self) -> None:
        self._active = False

    async def read_chunk(self, duration_ms: int = 500):
        raise NotImplementedError

    async def stream(self, chunk_duration_ms: int = 500) -> AsyncIterator:
        for c in self._chunks:
            if not self._active:
                break
            yield c


@pytest.mark.integration
@pytest.mark.asyncio
async def test_recording_pipeline_writes_valid_wav(tmp_path: Path) -> None:
    """AudioRecorder produces a valid, non-empty WAV file from a fake source."""
    from voxfusion.recording.recorder import AudioRecorder

    # Build 2 seconds of audio (4 chunks × 0.5 s each)
    chunk_samples = int(SR * 0.5)
    t = np.linspace(0, 0.5, chunk_samples, endpoint=False, dtype=np.float32)
    base = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    chunks = [_audio_chunk(base, SR) for _ in range(4)]

    output = tmp_path / "recording.wav"
    source = _FakeSource(chunks)
    recorder = AudioRecorder(chunk_duration_ms=500)
    stats = await recorder.record(source, output)

    assert output.exists(), "WAV file was not created"
    assert output.stat().st_size > 0, "WAV file is empty"

    info = sf.info(str(output))
    assert info.samplerate == SR
    assert info.channels == 1
    assert info.duration > 1.5, f"Expected ~2 s, got {info.duration:.2f} s"

    assert stats.output_path == output
    assert stats.duration_s > 1.5
    assert stats.chunks_captured == 4


@pytest.mark.integration
@pytest.mark.asyncio
async def test_recording_pipeline_pause_resume(tmp_path: Path) -> None:
    """Pause discards audio; resume continues — final duration reflects only active time."""
    from voxfusion.recording.recorder import AudioRecorder

    chunk_samples = int(SR * 0.5)
    t = np.linspace(0, 0.5, chunk_samples, endpoint=False, dtype=np.float32)
    active_samples = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    paused_samples = (0.9 * np.ones(chunk_samples, dtype=np.float32))

    recorder = AudioRecorder(chunk_duration_ms=500)

    class _PauseSource(_FakeSource):
        async def stream(self, chunk_duration_ms: int = 500) -> AsyncIterator:
            if not self._active:
                return
            # Yield pre-stamped chunks: timestamps are 0→0.5, 0.5→1.0, 1.0→1.5
            yield self._chunks[0]               # active  (0.0–0.5)
            recorder.request_pause()
            yield self._chunks[1]               # paused  (0.5–1.0) — recorder drops it
            recorder.request_resume()
            yield self._chunks[2]               # resumed (1.0–1.5) → adjusted to 0.5–1.0

    output = tmp_path / "pause_test.wav"
    source = _PauseSource([
        _audio_chunk(active_samples),
        _audio_chunk(paused_samples),
        _audio_chunk(active_samples),
    ])
    stats = await recorder.record(source, output)

    assert output.exists()
    # 2 active chunks × 0.5 s = 1.0 s; the paused chunk must NOT appear
    assert abs(stats.duration_s - 1.0) < 0.1, f"Expected 1.0 s, got {stats.duration_s:.2f} s"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_recording_pipeline_writes_ogg(tmp_path: Path) -> None:
    """OGG format is written and readable by soundfile."""
    from voxfusion.recording.recorder import AudioRecorder

    chunk_samples = int(SR * 0.5)
    t = np.linspace(0, 0.5, chunk_samples, endpoint=False, dtype=np.float32)
    samples = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    source = _FakeSource([_audio_chunk(samples, SR)])
    output = tmp_path / "out.ogg"
    recorder = AudioRecorder(chunk_duration_ms=500)
    stats = await recorder.record(source, output, format="ogg")

    assert output.exists()
    assert sf.info(str(output)).format == "OGG"
    assert stats.duration_s > 0


# ---------------------------------------------------------------------------
# 2. WHISPER — always available (models in HF cache)
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.asyncio
async def test_whisper_tiny_transcribes_wav(test_wav: Path) -> None:
    """Whisper tiny loads and transcribes a WAV file without exceptions."""
    from voxfusion.asr.faster_whisper import FasterWhisperEngine
    from voxfusion.config.models import ASRConfig

    cfg = ASRConfig(model_size="tiny", device="cpu", compute_type="int8")
    engine = FasterWhisperEngine(cfg)
    try:
        engine.load_model()
        samples, _ = sf.read(str(test_wav), dtype="float32")
        segments = await engine.transcribe(_audio_chunk(samples), language="en")
        assert isinstance(segments, list)
        for seg in segments:
            assert isinstance(seg.text, str)
            assert seg.start_time >= 0
            assert seg.end_time >= seg.start_time
    finally:
        engine.close()


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.asyncio
@pytest.mark.skipif(
    not _hf_cache("models--Systran--faster-whisper-large-v3"),
    reason="large-v3 not cached; download first or run without this skip",
)
async def test_whisper_large_v3_transcribes_wav(test_wav: Path) -> None:
    """Whisper large-v3 processes a WAV file end-to-end without crashing."""
    from voxfusion.asr.faster_whisper import FasterWhisperEngine
    from voxfusion.config.models import ASRConfig

    cfg = ASRConfig(model_size="large-v3", device="cpu", compute_type="int8")
    engine = FasterWhisperEngine(cfg)
    try:
        engine.load_model()
        samples, _ = sf.read(str(test_wav), dtype="float32")
        segments = await engine.transcribe(_audio_chunk(samples), language="ru")
        assert isinstance(segments, list)
        for seg in segments:
            assert isinstance(seg.text, str)
    finally:
        engine.close()


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.asyncio
@pytest.mark.skipif(
    not _hf_cache("models--Systran--faster-whisper-large-v3"),
    reason="large-v3 not cached",
)
async def test_whisper_large_v3_full_batch_pipeline(test_wav: Path) -> None:
    """Full BatchPipeline with large-v3 — includes preprocessing and diarization stub."""
    from voxfusion.asr.factory import create_asr_engine
    from voxfusion.config.models import ASRConfig, PipelineConfig
    from voxfusion.diarization.channel import ChannelDiarizer
    from voxfusion.pipeline.batch import BatchPipeline
    from voxfusion.preprocessing.pipeline import PreProcessingPipeline

    asr_cfg = ASRConfig(model_size="large-v3", device="cpu", compute_type="int8")
    asr_engine, _ = create_asr_engine(asr_cfg)
    pipeline = BatchPipeline(
        asr_engine=asr_engine,
        diarizer=ChannelDiarizer(),
        preprocessor=PreProcessingPipeline([]),
        config=PipelineConfig(),
    )
    result = await pipeline.process_file(test_wav)

    assert result is not None
    assert isinstance(result.segments, list)


# ---------------------------------------------------------------------------
# 3. GIGAAM v3  — download if missing, then transcribe
# ---------------------------------------------------------------------------

_GIGAAM_CACHED = _hf_cache("models--ai-sage--GigaAM-v3")
_GIGAAM_PKGS = (
    _pkg("transformers") and _pkg("torch")
    and _pkg("torchaudio") and _pkg("sentencepiece")
    and _pkg("omegaconf") and _pkg("hydra")
)


@pytest.mark.integration
@pytest.mark.download
@pytest.mark.slow
@pytest.mark.skipif(not _GIGAAM_PKGS, reason="transformers or torch not installed")
def test_gigaam_v3_download() -> None:
    """Download ai-sage/GigaAM-v3 from HuggingFace (idempotent — uses cache)."""
    import os
    from transformers import AutoModel

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    # AutoModel.from_pretrained caches the model; subsequent calls are instant
    model = AutoModel.from_pretrained(
        "ai-sage/GigaAM-v3", trust_remote_code=True, token=token
    )
    assert model is not None, "Model object should not be None after download"


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.skipif(
    not (_GIGAAM_PKGS and _GIGAAM_CACHED),
    reason="GigaAM v3 not cached — run test_gigaam_v3_download first, or: "
           "huggingface-cli download ai-sage/GigaAM-v3",
)
@pytest.mark.asyncio
async def test_gigaam_v3_transcribes_wav(test_wav: Path) -> None:
    """GigaAM v3 transcribes a WAV file without crashing."""
    from voxfusion.asr.gigaam_engine import GigaAMCTCEngine
    from voxfusion.config.models import ASRConfig

    engine = GigaAMCTCEngine(ASRConfig(model_size="gigaam-v3-e2e-ctc"))
    try:
        engine.load_model()
        samples, _ = sf.read(str(test_wav), dtype="float32")
        segments = await engine.transcribe(_audio_chunk(samples), language="ru")
        assert isinstance(segments, list)
        for seg in segments:
            assert isinstance(seg.text, str)
            assert seg.language == "ru"
    finally:
        engine.close()


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.skipif(
    not (_GIGAAM_PKGS and _GIGAAM_CACHED),
    reason="GigaAM v3 not cached",
)
@pytest.mark.asyncio
async def test_gigaam_v3_full_batch_pipeline(test_wav: Path) -> None:
    """Full BatchPipeline with GigaAM v3 — end-to-end without crashes."""
    from voxfusion.asr.factory import create_asr_engine
    from voxfusion.config.models import ASRConfig, PipelineConfig
    from voxfusion.diarization.channel import ChannelDiarizer
    from voxfusion.pipeline.batch import BatchPipeline
    from voxfusion.preprocessing.pipeline import PreProcessingPipeline

    asr_cfg = ASRConfig(model_size="gigaam-v3-e2e-ctc")
    asr_engine, backend = create_asr_engine(asr_cfg)
    assert backend == "gigaam"

    pipeline = BatchPipeline(
        asr_engine=asr_engine,
        diarizer=ChannelDiarizer(),
        preprocessor=PreProcessingPipeline([]),
        config=PipelineConfig(),
    )
    result = await pipeline.process_file(test_wav)
    assert result is not None
    assert isinstance(result.segments, list)


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.skipif(
    not (_GIGAAM_PKGS and _GIGAAM_CACHED),
    reason="GigaAM v3 not cached",
)
def test_gigaam_v3_streaming_raises_transcription_error(test_wav: Path) -> None:
    """GigaAM streaming mode must raise TranscriptionError (batch-only engine)."""
    from voxfusion.asr.gigaam_engine import GigaAMCTCEngine
    from voxfusion.config.models import ASRConfig
    from voxfusion.exceptions import TranscriptionError

    engine = GigaAMCTCEngine(ASRConfig(model_size="gigaam-v3-e2e-ctc"))
    try:
        async def _drain():
            async for _ in engine.transcribe_stream(
                aiter([]),  # type: ignore[name-defined]
            ):
                pass

        with pytest.raises(TranscriptionError):
            asyncio.run(_drain())
    finally:
        engine.close()


# ---------------------------------------------------------------------------
# 4. BREEZE ASR  — download if missing, then transcribe
# ---------------------------------------------------------------------------

_BREEZE_CACHED = _hf_cache("models--MediaTek-Research--Breeze-ASR-25")
_BREEZE_PKGS = _pkg("transformers") and _pkg("torch")


@pytest.mark.integration
@pytest.mark.download
@pytest.mark.slow
@pytest.mark.skipif(not _BREEZE_PKGS, reason="transformers or torch not installed")
def test_breeze_asr_download() -> None:
    """Download MediaTek-Research/Breeze-ASR-25 (idempotent — uses HF cache)."""
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    ref = "MediaTek-Research/Breeze-ASR-25"
    AutoProcessor.from_pretrained(ref)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(ref)
    assert model is not None


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.skipif(
    not (_BREEZE_PKGS and _BREEZE_CACHED),
    reason="Breeze ASR not cached — run test_breeze_asr_download first, or: "
           "huggingface-cli download MediaTek-Research/Breeze-ASR-25",
)
@pytest.mark.asyncio
async def test_breeze_asr_transcribes_wav(test_wav: Path) -> None:
    """Breeze ASR transcribes a WAV file without crashing."""
    from voxfusion.asr.breeze_engine import BreezeASREngine
    from voxfusion.config.models import ASRConfig

    engine = BreezeASREngine(ASRConfig(model_size="breeze-asr"))
    try:
        engine.load_model()
        samples, _ = sf.read(str(test_wav), dtype="float32")
        segments = await engine.transcribe(_audio_chunk(samples), language="en")
        assert isinstance(segments, list)
        for seg in segments:
            assert isinstance(seg.text, str)
    finally:
        engine.close()


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.skipif(
    not (_BREEZE_PKGS and _BREEZE_CACHED),
    reason="Breeze ASR not cached",
)
@pytest.mark.asyncio
async def test_breeze_asr_full_batch_pipeline(test_wav: Path) -> None:
    """Full BatchPipeline with Breeze ASR — end-to-end without crashes."""
    from voxfusion.asr.factory import create_asr_engine
    from voxfusion.config.models import ASRConfig, PipelineConfig
    from voxfusion.diarization.channel import ChannelDiarizer
    from voxfusion.pipeline.batch import BatchPipeline
    from voxfusion.preprocessing.pipeline import PreProcessingPipeline

    asr_cfg = ASRConfig(model_size="breeze-asr")
    asr_engine, backend = create_asr_engine(asr_cfg)
    assert backend == "breeze"

    pipeline = BatchPipeline(
        asr_engine=asr_engine,
        diarizer=ChannelDiarizer(),
        preprocessor=PreProcessingPipeline([]),
        config=PipelineConfig(),
    )
    result = await pipeline.process_file(test_wav)
    assert result is not None
    assert isinstance(result.segments, list)


# ---------------------------------------------------------------------------
# 5. PARAKEET v3  — download via NeMo if missing, then transcribe
# ---------------------------------------------------------------------------

_PARAKEET_PKGS = _pkg("nemo")


@pytest.mark.integration
@pytest.mark.download
@pytest.mark.slow
@pytest.mark.skipif(not _PARAKEET_PKGS, reason='nemo_toolkit not installed — pip install "nemo_toolkit[asr]" torchaudio')
def test_parakeet_v3_download() -> None:
    """Download nvidia/parakeet-tdt-0.6b-v3 via NeMo (idempotent — uses .nemo cache)."""
    from nemo.collections.asr.models import ASRModel

    model = ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v3")
    assert model is not None


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.skipif(not _PARAKEET_PKGS, reason='nemo_toolkit not installed — pip install "nemo_toolkit[asr]" torchaudio')
@pytest.mark.asyncio
async def test_parakeet_v3_transcribes_wav(test_wav: Path) -> None:
    """Parakeet v3 transcribes a WAV file without crashing."""
    from voxfusion.asr.parakeet_engine import ParakeetASREngine
    from voxfusion.config.models import ASRConfig

    engine = ParakeetASREngine(ASRConfig(model_size="parakeet-tdt-0.6b-v3"))
    try:
        engine.load_model()
        samples, _ = sf.read(str(test_wav), dtype="float32")
        segments = await engine.transcribe(_audio_chunk(samples))
        assert isinstance(segments, list)
        for seg in segments:
            assert isinstance(seg.text, str)
    finally:
        engine.close()


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.skipif(not _PARAKEET_PKGS, reason='nemo_toolkit not installed')
@pytest.mark.asyncio
async def test_parakeet_v3_full_batch_pipeline(test_wav: Path) -> None:
    """Full BatchPipeline with Parakeet v3 — end-to-end without crashes."""
    from voxfusion.asr.factory import create_asr_engine
    from voxfusion.config.models import ASRConfig, PipelineConfig
    from voxfusion.diarization.channel import ChannelDiarizer
    from voxfusion.pipeline.batch import BatchPipeline
    from voxfusion.preprocessing.pipeline import PreProcessingPipeline

    asr_cfg = ASRConfig(model_size="parakeet-tdt-0.6b-v3")
    asr_engine, backend = create_asr_engine(asr_cfg)
    assert backend == "parakeet"

    pipeline = BatchPipeline(
        asr_engine=asr_engine,
        diarizer=ChannelDiarizer(),
        preprocessor=PreProcessingPipeline([]),
        config=PipelineConfig(),
    )
    result = await pipeline.process_file(test_wav)
    assert result is not None
    assert isinstance(result.segments, list)


# ---------------------------------------------------------------------------
# 6. GUI download button — verify the _download_file_model method works
# ---------------------------------------------------------------------------

def _skip_if_no_display() -> None:
    import tkinter as tk
    try:
        r = tk.Tk()
        r.destroy()
    except Exception as exc:
        pytest.skip(f"No display: {exc}")


@pytest.mark.integration
@pytest.mark.slow
def test_gui_download_button_triggers_whisper_download_without_crash() -> None:
    """The GUI ↓ Download button completes for a cached Whisper model without crashing."""
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

        # Select Whisper tiny (definitely cached) and trigger download
        gui._file_model_var.set("tiny")
        gui._on_file_model_changed()
        gui._download_file_model()

        # Give the background thread up to 30 seconds to complete
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            root.update()
            # Download finished when button re-enables
            if str(gui._file_download_btn.cget("state")) == "normal":
                break
            time.sleep(0.1)

        assert str(gui._file_download_btn.cget("state")) == "normal", (
            "Download button was never re-enabled — background thread may have crashed"
        )
        gui._restore_redirection()
    finally:
        root.destroy()


@pytest.mark.integration
@pytest.mark.download
@pytest.mark.slow
@pytest.mark.skipif(not _GIGAAM_PKGS, reason="transformers or torch not installed")
def test_gui_download_button_triggers_gigaam_download() -> None:
    """The GUI ↓ Download button downloads GigaAM v3 without crashing the UI thread."""
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

        gui._file_model_var.set("gigaam-v3-e2e-ctc")
        gui._on_file_model_changed()
        gui._download_file_model()

        # Wait up to 10 minutes for the download to complete (~1.5 GB)
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            root.update()
            if str(gui._file_download_btn.cget("state")) == "normal":
                break
            time.sleep(0.5)

        assert str(gui._file_download_btn.cget("state")) == "normal", (
            "GigaAM download did not complete within 10 minutes"
        )
        # Verify model is now in the HF cache
        assert _hf_cache("models--ai-sage--GigaAM-v3"), (
            "Model was reported as downloaded but is not in the HF cache"
        )
        gui._restore_redirection()
    finally:
        root.destroy()


# ---------------------------------------------------------------------------
# 7. FileTranscribeWorker — real file transcription via GUI worker
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.slow
def test_file_transcribe_worker_whisper_small(test_wav: Path) -> None:
    """FileTranscribeWorker processes a WAV file with Whisper small end-to-end."""
    from voxfusion.gui.runtime import FileTranscribeWorker

    errors: list[str] = []
    segments_received: list = []
    finished = threading.Event() if False else __import__("threading").Event()

    worker = FileTranscribeWorker(
        file_path=test_wav,
        model="small",
        language="ru",
        quality="fast",
        on_status=lambda msg, pct: None,
        on_segments=lambda segs: segments_received.extend(segs),
        on_error=lambda e: errors.append(e),
        on_finished=lambda: finished.set(),
    )
    worker.start()
    finished.wait(timeout=120)

    assert finished.is_set(), "FileTranscribeWorker did not finish within 120 s"
    assert not errors, f"FileTranscribeWorker reported errors: {errors}"


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.skipif(
    not (_GIGAAM_PKGS and _GIGAAM_CACHED),
    reason="GigaAM v3 not cached",
)
def test_file_transcribe_worker_gigaam(test_wav: Path) -> None:
    """FileTranscribeWorker processes a WAV file with GigaAM v3 end-to-end."""
    from voxfusion.gui.runtime import FileTranscribeWorker
    import threading

    errors: list[str] = []
    finished = threading.Event()

    worker = FileTranscribeWorker(
        file_path=test_wav,
        model="gigaam-v3-e2e-ctc",
        language="ru",
        quality="balanced",
        on_status=lambda msg, pct: None,
        on_segments=lambda segs: None,
        on_error=lambda e: errors.append(e),
        on_finished=lambda: finished.set(),
    )
    worker.start()
    finished.wait(timeout=300)

    assert finished.is_set(), "GigaAM FileTranscribeWorker did not finish within 5 min"
    assert not errors, f"GigaAM transcription errors: {errors}"


# ---------------------------------------------------------------------------
# 8. Record-then-transcribe — full user workflow simulation
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.asyncio
async def test_record_then_transcribe_whisper_small(tmp_path: Path) -> None:
    """Simulate the full workflow: record → save WAV → transcribe with Whisper small."""
    from voxfusion.recording.recorder import AudioRecorder
    from voxfusion.asr.faster_whisper import FasterWhisperEngine
    from voxfusion.config.models import ASRConfig

    # Step 1: Record
    chunk_samples = int(SR * 0.5)
    t = np.linspace(0, 0.5, chunk_samples, endpoint=False, dtype=np.float32)
    audio = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    chunks = [_audio_chunk(audio) for _ in range(6)]  # 3 seconds

    output = tmp_path / "recorded.wav"
    recorder = AudioRecorder(chunk_duration_ms=500)
    stats = await recorder.record(_FakeSource(chunks), output)

    assert output.exists(), "Recording did not produce a WAV file"
    assert stats.duration_s > 2.0

    # Step 2: Transcribe
    cfg = ASRConfig(model_size="small", device="cpu", compute_type="int8")
    engine = FasterWhisperEngine(cfg)
    try:
        engine.load_model()
        samples, sr = sf.read(str(output), dtype="float32")
        segments = await engine.transcribe(_audio_chunk(samples, sr), language="ru")
        assert isinstance(segments, list)
        # Result may be empty for a sine wave — that's acceptable; crash is not
    finally:
        engine.close()


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.skipif(
    not (_GIGAAM_PKGS and _GIGAAM_CACHED),
    reason="GigaAM v3 not cached",
)
@pytest.mark.asyncio
async def test_record_then_transcribe_gigaam(tmp_path: Path) -> None:
    """Simulate the full workflow: record → save WAV → transcribe with GigaAM v3."""
    from voxfusion.recording.recorder import AudioRecorder
    from voxfusion.asr.gigaam_engine import GigaAMCTCEngine
    from voxfusion.config.models import ASRConfig

    # Step 1: Record
    chunk_samples = int(SR * 0.5)
    t = np.linspace(0, 0.5, chunk_samples, endpoint=False, dtype=np.float32)
    audio = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    chunks = [_audio_chunk(audio) for _ in range(6)]  # 3 seconds

    output = tmp_path / "recorded_ru.wav"
    recorder = AudioRecorder(chunk_duration_ms=500)
    stats = await recorder.record(_FakeSource(chunks), output)

    assert output.exists()
    assert stats.duration_s > 2.0

    # Step 2: Transcribe
    engine = GigaAMCTCEngine(ASRConfig(model_size="gigaam-v3-e2e-ctc"))
    try:
        engine.load_model()
        samples, sr = sf.read(str(output), dtype="float32")
        segments = await engine.transcribe(_audio_chunk(samples, sr), language="ru")
        assert isinstance(segments, list)
    finally:
        engine.close()
