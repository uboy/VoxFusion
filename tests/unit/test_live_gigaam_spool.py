"""Unit tests for live GigaAM audio spooling."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from voxfusion.live_gigaam.spool import SessionAudioSpool, normalize_audio_samples
from voxfusion.models.audio import AudioChunk


def test_normalize_audio_samples_flattens_stereo_and_resamples() -> None:
    stereo = np.column_stack(
        [
            np.ones(800, dtype=np.float32),
            np.ones(800, dtype=np.float32) * 3,
        ]
    )

    normalized = normalize_audio_samples(stereo, 8000)

    assert normalized.dtype == np.float32
    assert normalized.ndim == 1
    assert len(normalized) == 1600
    assert np.allclose(normalized[:20], 2.0, atol=0.25)


def test_session_audio_spool_zero_pads_gaps_and_reads_windows(tmp_path: Path) -> None:
    spool = SessionAudioSpool(tmp_path / "spool")
    try:
        first = AudioChunk(
            samples=np.ones(8000, dtype=np.float32),
            sample_rate=16000,
            channels=1,
            timestamp_start=0.0,
            timestamp_end=0.5,
            source="microphone",
            dtype="float32",
        )
        second = AudioChunk(
            samples=np.ones(8000, dtype=np.float32) * 2,
            sample_rate=16000,
            channels=1,
            timestamp_start=1.0,
            timestamp_end=1.5,
            source="microphone",
            dtype="float32",
        )

        spool.append(first)
        spool.append(second)
        window = spool.read_window("microphone", 0.0, 1.5)

        assert spool.source_path("microphone") == tmp_path / "spool" / "microphone.wav"
        assert window.shape[0] == 24000
        assert np.allclose(window[:8000], 1.0)
        assert np.allclose(window[8000:16000], 0.0)
        assert np.allclose(window[16000:], 2.0)
    finally:
        spool.close()


def test_session_audio_spool_tracks_sources_separately(tmp_path: Path) -> None:
    spool = SessionAudioSpool(tmp_path / "spool")
    try:
        spool.append(
            AudioChunk(
                samples=np.ones(1600, dtype=np.float32),
                sample_rate=16000,
                channels=1,
                timestamp_start=0.0,
                timestamp_end=0.1,
                source="microphone",
                dtype="float32",
            )
        )
        spool.append(
            AudioChunk(
                samples=np.ones(1600, dtype=np.float32) * 2,
                sample_rate=16000,
                channels=1,
                timestamp_start=0.0,
                timestamp_end=0.1,
                source="system",
                dtype="float32",
            )
        )

        assert spool.source_path("microphone") != spool.source_path("system")
        assert np.allclose(spool.read_window("microphone", 0.0, 0.1), 1.0)
        assert np.allclose(spool.read_window("system", 0.0, 0.1), 2.0)
    finally:
        spool.close()
