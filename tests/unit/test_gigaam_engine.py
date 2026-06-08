"""Unit tests for GigaAM backend integration."""

from __future__ import annotations

import os
import sys
import types

import numpy as np
import pytest

from voxfusion.asr.factory import create_asr_engine
from voxfusion.asr.gigaam_engine import (
    GigaAMCTCEngine,
    _install_megatron_compat_shim,
    _install_torchscript_source_fallback,
    _prepare_huggingface_runtime_env,
)
from voxfusion.config.models import ASRConfig
from voxfusion.models.audio import AudioChunk


class _FakeGigaAMModel:
    """Minimal fake returned by AutoModel.from_pretrained for GigaAM."""

    @classmethod
    def from_pretrained(cls, _model_ref: str, **_kwargs: object) -> _FakeGigaAMModel:
        return cls()

    def transcribe(self, wav_path: str) -> str:
        del wav_path
        return "privet mir"


class _FakeChunkedModel:
    def __init__(self) -> None:
        self._idx = 0

    def transcribe(self, wav_path: str) -> str:
        del wav_path
        self._idx += 1
        return f"chunk {self._idx}"


def test_asr_config_sets_engine_for_gigaam_model() -> None:
    cfg = ASRConfig(model_size="gigaam-v3-e2e-ctc")
    assert cfg.engine == "gigaam"
    assert cfg.language is None


def test_factory_routes_gigaam_engine() -> None:
    engine, backend = create_asr_engine(ASRConfig(model_size="gigaam-v3-e2e-ctc"))
    assert backend == "gigaam"
    assert isinstance(engine, GigaAMCTCEngine)
    engine.close()


@pytest.mark.asyncio
async def test_gigaam_engine_transcribes_with_fake_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoModel = _FakeGigaAMModel  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    engine = GigaAMCTCEngine(
        ASRConfig(model_size="gigaam-v3-e2e-ctc", model_path="C:/models/gigaam")
    )
    chunk = AudioChunk(
        samples=np.ones(16000, dtype=np.float32),
        sample_rate=16000,
        channels=1,
        timestamp_start=0.0,
        timestamp_end=1.0,
        source="file",
        dtype="float32",
    )

    segments = await engine.transcribe(chunk, language="ru")
    assert len(segments) == 1
    assert segments[0].text == "privet mir"
    assert segments[0].language == "ru"
    engine.close()


def test_gigaam_engine_returns_empty_for_too_short_audio() -> None:
    engine = GigaAMCTCEngine()

    segments = engine._transcribe_sync(np.ones(100, dtype=np.float32), language="ru")

    assert segments == []
    engine.close()


def test_gigaam_normalize_audio_flattens_deep_arrays() -> None:
    engine = GigaAMCTCEngine()
    samples = np.ones((8, 1, 1), dtype=np.float32)

    normalized = engine._normalize_audio(samples, 16000)

    assert normalized.ndim == 1
    assert normalized.shape[0] == 8
    engine.close()


def test_prepare_huggingface_runtime_env_removes_deprecated_transformers_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_HOME", "C:/hf-home")
    monkeypatch.setenv("TRANSFORMERS_CACHE", "C:/old-cache")

    _prepare_huggingface_runtime_env()

    assert os.environ["HUGGINGFACE_HUB_CACHE"].endswith("hf-home\\hub") or os.environ[
        "HUGGINGFACE_HUB_CACHE"
    ].endswith("hf-home/hub")
    assert "TRANSFORMERS_CACHE" not in os.environ


def test_install_megatron_compat_shim_registers_num_microbatches_module() -> None:
    sys.modules.pop("megatron.core.num_microbatches_calculator", None)
    sys.modules.pop("megatron.core", None)
    sys.modules.pop("megatron", None)

    _install_megatron_compat_shim()

    mod = sys.modules["megatron.core.num_microbatches_calculator"]
    assert mod.get_num_microbatches() == 1
    assert mod.get_current_global_batch_size() == 1
    assert mod.get_micro_batch_size() == 1
    assert mod.init_num_microbatches_calculator() is None
    calculator = mod.ConstantNumMicroBatchesCalculator()
    assert calculator.num_microbatches == 1


def test_install_torchscript_source_fallback_returns_original_object_on_source_error() -> None:
    class _FakeJit:
        def script(self, obj, *args, **kwargs):
            del args, kwargs
            raise RuntimeError(
                f"Can't get source for {obj}. TorchScript requires source access in order to carry out compilation"
            )

    fake_torch = types.SimpleNamespace(jit=_FakeJit())
    _install_torchscript_source_fallback(fake_torch)
    marker = object()
    assert fake_torch.jit.script(marker) is marker


def test_gigaam_chunked_fallback_returns_timestamped_segments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import voxfusion.asr.gigaam_engine as gigaam_module

    monkeypatch.setattr(gigaam_module, "_CHUNK_SAMPLES", 16_000)
    monkeypatch.setattr(gigaam_module, "_OVERLAP_SAMPLES", 0)

    engine = GigaAMCTCEngine()
    model = _FakeChunkedModel()
    audio = np.ones(32_000, dtype=np.float32)  # 2 seconds

    segments = engine._transcribe_chunked(model, audio, total_duration_s=2.0)

    assert len(segments) == 2
    assert segments[0].text == "chunk 1"
    assert segments[0].start_time == pytest.approx(0.0)
    assert segments[0].end_time == pytest.approx(1.0)
    assert segments[1].text == "chunk 2"
    assert segments[1].start_time == pytest.approx(1.0)
    assert segments[1].end_time == pytest.approx(2.0)
    engine.close()


class TestDedupSeam:
    def test_no_overlap(self) -> None:
        from voxfusion.asr.gigaam_engine import _dedup_seam

        text, removed = _dedup_seam("alpha beta", "gamma delta")
        assert text == "gamma delta"
        assert removed == 0

    def test_removes_duplicate_prefix(self) -> None:
        from voxfusion.asr.gigaam_engine import _dedup_seam

        text, removed = _dedup_seam("alpha beta gamma", "gamma delta epsilon")
        assert text == "delta epsilon"
        assert removed == 1

    def test_removes_multi_word_overlap(self) -> None:
        from voxfusion.asr.gigaam_engine import _dedup_seam

        text, removed = _dedup_seam("a b c d e", "c d e f g")
        assert text == "f g"
        assert removed == 3

    def test_full_overlap_returns_empty(self) -> None:
        from voxfusion.asr.gigaam_engine import _dedup_seam

        text, removed = _dedup_seam("hello world", "hello world")
        assert text == ""
        assert removed == 2

    def test_empty_prev(self) -> None:
        from voxfusion.asr.gigaam_engine import _dedup_seam

        text, removed = _dedup_seam("", "hello world")
        assert text == "hello world"
        assert removed == 0

    def test_empty_curr(self) -> None:
        from voxfusion.asr.gigaam_engine import _dedup_seam

        text, removed = _dedup_seam("hello world", "")
        assert text == ""
        assert removed == 0

    def test_prefers_longest_match(self) -> None:
        from voxfusion.asr.gigaam_engine import _dedup_seam

        text, removed = _dedup_seam("a b c d", "b c d e f")
        assert text == "e f"
        assert removed == 3


class TestChunkedTimestampAdjustment:
    def test_start_time_shifted_after_dedup(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import voxfusion.asr.gigaam_engine as gigaam_module

        monkeypatch.setattr(gigaam_module, "_CHUNK_SAMPLES", 16_000)
        monkeypatch.setattr(gigaam_module, "_OVERLAP_SAMPLES", 4_000)

        class _OverlapModel:
            def __init__(self) -> None:
                self._idx = 0

            def transcribe(self, wav_path: str) -> str:
                del wav_path
                self._idx += 1
                if self._idx == 1:
                    return "alpha beta gamma"
                return "gamma delta epsilon"

        engine = GigaAMCTCEngine()
        model = _OverlapModel()
        audio = np.ones(28_000, dtype=np.float32)

        segments = engine._transcribe_chunked(model, audio, total_duration_s=1.75)

        assert len(segments) == 2
        assert segments[0].start_time == pytest.approx(0.0)
        assert segments[1].start_time > 0.75
        assert segments[1].text == "delta epsilon"
        engine.close()


class TestChunkedShortChunkSkip:
    def test_skips_trailing_chunk_shorter_than_1s(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import voxfusion.asr.gigaam_engine as gigaam_module

        monkeypatch.setattr(gigaam_module, "_CHUNK_SAMPLES", 16_000)
        monkeypatch.setattr(gigaam_module, "_OVERLAP_SAMPLES", 8_000)

        class _CountingModel:
            def __init__(self) -> None:
                self.calls = 0

            def transcribe(self, wav_path: str) -> str:
                del wav_path
                self.calls += 1
                return f"text {self.calls}"

        engine = GigaAMCTCEngine()
        model = _CountingModel()
        audio = np.ones(26_000, dtype=np.float32)

        segments = engine._transcribe_chunked(model, audio, total_duration_s=1.625)

        assert model.calls == 2
        assert len(segments) == 2
        engine.close()

    def test_returns_empty_for_audio_shorter_than_min(self) -> None:
        engine = GigaAMCTCEngine()
        model = _FakeChunkedModel()
        audio = np.ones(100, dtype=np.float32)

        segments = engine._transcribe_chunked(model, audio, total_duration_s=0.00625)

        assert segments == []
        engine.close()


class TestChunkedPerChunkErrorHandling:
    def test_failed_chunk_does_not_crash_others(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import voxfusion.asr.gigaam_engine as gigaam_module

        monkeypatch.setattr(gigaam_module, "_CHUNK_SAMPLES", 16_000)
        monkeypatch.setattr(gigaam_module, "_OVERLAP_SAMPLES", 0)

        class _FailingOnSecondModel:
            def __init__(self) -> None:
                self._idx = 0

            def transcribe(self, wav_path: str) -> str:
                del wav_path
                self._idx += 1
                if self._idx == 2:
                    raise RuntimeError("STFT crash")
                return f"text {self._idx}"

        engine = GigaAMCTCEngine()
        model = _FailingOnSecondModel()
        audio = np.ones(48_000, dtype=np.float32)

        segments = engine._transcribe_chunked(model, audio, total_duration_s=3.0)

        assert len(segments) == 2
        assert segments[0].text == "text 1"
        assert segments[1].text == "text 3"
        engine.close()
