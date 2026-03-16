"""Tests for the NoiseReducer preprocessing module."""

import numpy as np

from voxfusion.models.audio import AudioChunk
from voxfusion.preprocessing.noise import NoiseReducer


class TestNoiseReducer:
    """Tests for NoiseReducer (without noisereduce dependency)."""

    def test_init_defaults(self):
        nr = NoiseReducer()
        assert nr._stationary is True
        assert nr._prop_decrease == 1.0
        assert nr._n_fft == 1024

    def test_init_custom_params(self):
        nr = NoiseReducer(
            stationary=False,
            prop_decrease=0.8,
            n_fft=2048,
            n_std_thresh_stationary=2.0,
        )
        assert nr._stationary is False
        assert nr._prop_decrease == 0.8
        assert nr._n_fft == 2048
        assert nr._n_std_thresh == 2.0

    def test_short_chunk_passthrough(self):
        """Chunks shorter than n_fft should pass through unchanged."""
        nr = NoiseReducer(n_fft=1024)
        samples = np.random.randn(512).astype(np.float32)
        chunk = AudioChunk(
            samples=samples,
            sample_rate=16000,
            channels=1,
            timestamp_start=0.0,
            timestamp_end=0.032,
            source="file",
            dtype="float32",
        )
        result = nr.process(chunk)
        assert result is chunk  # Exact same object returned

    def test_reset_is_noop(self):
        nr = NoiseReducer()
        nr.reset()  # Should not raise
