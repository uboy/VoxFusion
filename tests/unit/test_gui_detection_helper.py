"""Regression tests for GUI speaker-detection audio loading."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from voxfusion.gui import helpers


def test_load_detection_audio_chunk_extracts_container_media(tmp_path: Path, monkeypatch) -> None:
    media_path = tmp_path / "meeting.webm"
    media_path.write_bytes(b"container")
    extracted_path = tmp_path / "meeting.wav"
    extracted_path.write_bytes(b"wav")

    seek_calls: list[int] = []
    read_calls: list[tuple[int, str, bool]] = []

    def fake_extract_audio(path: Path) -> Path:
        assert path == media_path
        return extracted_path

    class FakeSoundFile:
        samplerate = 16_000

        def __init__(self, path: str, mode: str = "r") -> None:
            assert mode == "r"
            assert Path(path) == extracted_path

        def __enter__(self) -> FakeSoundFile:
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def __len__(self) -> int:
            return 16_000 * 300

        def seek(self, offset: int) -> None:
            seek_calls.append(offset)

        def read(
            self,
            frames: int,
            *,
            dtype: str,
            always_2d: bool,
        ) -> np.ndarray:
            read_calls.append((frames, dtype, always_2d))
            left = np.ones(frames, dtype=np.float32)
            right = np.full(frames, 3.0, dtype=np.float32)
            return np.column_stack((left, right))

    monkeypatch.setattr(helpers, "needs_extraction", lambda path: True)
    monkeypatch.setattr(helpers, "extract_audio", fake_extract_audio)
    monkeypatch.setattr(helpers.sf, "SoundFile", FakeSoundFile)

    audio = helpers.load_detection_audio_chunk(media_path, max_duration_s=120.0)

    assert seek_calls == [16_000 * 90]
    assert read_calls == [(16_000 * 120, "float32", False)]
    assert audio.sample_rate == 16_000
    assert audio.channels == 1
    assert audio.timestamp_start == 90.0
    assert audio.timestamp_end == 210.0
    assert np.allclose(audio.samples[:8], np.full(8, 2.0, dtype=np.float32))
    assert not extracted_path.exists()


def test_load_detection_audio_chunk_falls_back_to_ffmpeg_when_direct_open_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    media_path = tmp_path / "recording.bin"
    media_path.write_bytes(b"opaque")
    extracted_path = tmp_path / "recording.wav"
    extracted_path.write_bytes(b"wav")

    opened_paths: list[Path] = []

    def fake_extract_audio(path: Path) -> Path:
        assert path == media_path
        return extracted_path

    class FakeSoundFile:
        samplerate = 8_000

        def __init__(self, path: str, mode: str = "r") -> None:
            current = Path(path)
            opened_paths.append(current)
            if current == media_path:
                raise RuntimeError("unsupported format")

        def __enter__(self) -> FakeSoundFile:
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def __len__(self) -> int:
            return 8_000 * 10

        def seek(self, offset: int) -> None:
            assert offset == 0

        def read(
            self,
            frames: int,
            *,
            dtype: str,
            always_2d: bool,
        ) -> np.ndarray:
            assert dtype == "float32"
            assert always_2d is False
            return np.zeros(frames, dtype=np.float32)

    monkeypatch.setattr(helpers, "needs_extraction", lambda path: False)
    monkeypatch.setattr(helpers, "extract_audio", fake_extract_audio)
    monkeypatch.setattr(helpers.sf, "SoundFile", FakeSoundFile)

    audio = helpers.load_detection_audio_chunk(media_path, max_duration_s=120.0)

    assert opened_paths == [media_path, extracted_path]
    assert audio.sample_rate == 8_000
    assert audio.timestamp_start == 0.0
    assert audio.timestamp_end == 10.0
    assert not extracted_path.exists()
