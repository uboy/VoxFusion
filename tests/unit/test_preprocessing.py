"""Unit tests for voxfusion.preprocessing modules."""

import numpy as np
import pytest

from voxfusion.models.audio import AudioChunk
from voxfusion.preprocessing.normalize import Normalizer
from voxfusion.preprocessing.pipeline import PreProcessingPipeline
from voxfusion.preprocessing.resample import Resampler


class TestResampler:
    def test_passthrough_same_rate(self, silence_chunk: AudioChunk) -> None:
        r = Resampler(target_sample_rate=16000)
        result = r.process(silence_chunk)
        assert result is silence_chunk  # no-op returns same object

    def test_resample_44100_to_16000(self, stereo_chunk: AudioChunk) -> None:
        # Build a mono 44100 Hz chunk
        mono_samples = stereo_chunk.samples[:, 0]
        chunk = AudioChunk(
            samples=mono_samples,
            sample_rate=44100,
            channels=1,
            timestamp_start=0.0,
            timestamp_end=1.0,
            source="file",
            dtype="float32",
        )
        r = Resampler(target_sample_rate=16000)
        result = r.process(chunk)
        assert result.sample_rate == 16000
        # Expect roughly 16000 samples for 1 second
        assert abs(result.num_samples - 16000) < 100

    def test_resample_preserves_timestamps(self) -> None:
        chunk = AudioChunk(
            samples=np.zeros(44100, dtype=np.float32),
            sample_rate=44100,
            channels=1,
            timestamp_start=5.0,
            timestamp_end=6.0,
            source="file",
        )
        r = Resampler(target_sample_rate=16000)
        result = r.process(chunk)
        assert result.timestamp_start == 5.0
        assert result.timestamp_end == 6.0

    def test_reset_is_noop(self) -> None:
        r = Resampler()
        r.reset()  # should not raise


class TestNormalizer:
    def test_silence_passthrough(self, silence_chunk: AudioChunk) -> None:
        n = Normalizer()
        result = n.process(silence_chunk)
        assert result is silence_chunk  # silence returns same object

    def test_normalizes_peak(self, sine_chunk: AudioChunk) -> None:
        n = Normalizer(target_dbfs=-3.0)
        result = n.process(sine_chunk)
        peak = float(np.max(np.abs(result.samples)))
        expected = 10 ** (-3.0 / 20.0)
        assert abs(peak - expected) < 0.01

    def test_empty_chunk(self) -> None:
        chunk = AudioChunk(
            samples=np.array([], dtype=np.float32),
            sample_rate=16000,
            channels=1,
            timestamp_start=0.0,
            timestamp_end=0.0,
            source="file",
        )
        n = Normalizer()
        result = n.process(chunk)
        assert result is chunk

    def test_reset_is_noop(self) -> None:
        n = Normalizer()
        n.reset()  # should not raise


class TestPreProcessingPipeline:
    def test_empty_pipeline(self, sine_chunk: AudioChunk) -> None:
        p = PreProcessingPipeline()
        assert len(p) == 0
        result = p.process(sine_chunk)
        assert result is sine_chunk

    def test_single_processor(self, sine_chunk: AudioChunk) -> None:
        p = PreProcessingPipeline([Normalizer()])
        assert len(p) == 1
        result = p.process(sine_chunk)
        assert result is not sine_chunk

    def test_chain_order(self) -> None:
        """Resampler then Normalizer should produce 16kHz normalized output."""
        chunk = AudioChunk(
            samples=0.1 * np.ones(44100, dtype=np.float32),
            sample_rate=44100,
            channels=1,
            timestamp_start=0.0,
            timestamp_end=1.0,
            source="file",
        )
        p = PreProcessingPipeline([Resampler(16000), Normalizer()])
        result = p.process(chunk)
        assert result.sample_rate == 16000
        # Peak should be near the normalize target
        peak = float(np.max(np.abs(result.samples)))
        assert peak > 0.5  # normalized up from 0.1

    def test_add_processor(self) -> None:
        p = PreProcessingPipeline()
        p.add(Resampler())
        p.add(Normalizer())
        assert len(p) == 2

    def test_reset_all(self) -> None:
        p = PreProcessingPipeline([Resampler(), Normalizer()])
        p.reset()  # should not raise
